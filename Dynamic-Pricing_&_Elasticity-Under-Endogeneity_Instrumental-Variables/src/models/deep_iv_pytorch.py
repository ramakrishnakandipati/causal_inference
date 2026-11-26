"""PyTorch (CUDA when available) DeepIV and DeepGMM structural estimators.

DeepIV   Stage 1: Mixture Density Network for P(T | X, Z) (Gaussian mixture, NLL).
         Stage 2: outcome network h(X, T) trained on  (Y - mean_m h(X, T_m))^2  with T_m ~ stage-1 MDN.
DeepGMM  Min-max optimally-weighted GMM for E[(Y - g(X, T)) f(X, Z)] = 0 (Bennett et al., 2019).
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.generate_hospitality_data import FEATURES, INSTRUMENTS, OUTCOME, TREATMENT, ln_reference_price
from src.models.base import BaseElasticityModel
from src.utils.gpu_utils import get_device, set_seed
from src.utils.logger import get_logger

log = get_logger("train_benchmark", "train_benchmark.log")


def mlp(d_in: int, d_out: int, hidden: int, depth: int = 3, act=nn.SiLU) -> nn.Sequential:
    layers: List[nn.Module] = []
    d = d_in
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), act()]
        d = hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class MDN(nn.Module):
    """Gaussian mixture density network over a scalar treatment."""

    def __init__(self, d_in: int, hidden: int, k: int):
        super().__init__()
        self.k = k
        self.body = mlp(d_in, 3 * k, hidden, depth=2)

    def params(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        o = self.body(x)
        logit, mu, raw = o[:, : self.k], o[:, self.k: 2 * self.k], o[:, 2 * self.k:]
        sigma = F.softplus(raw) + 1e-3
        return F.log_softmax(logit, dim=-1), mu, sigma

    def nll(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        logpi, mu, sigma = self.params(x)
        lp = -0.5 * ((t[:, None] - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
        return -torch.logsumexp(logpi + lp, dim=-1).mean()

    @torch.no_grad()
    def sample(self, x: torch.Tensor, m: int) -> torch.Tensor:
        logpi, mu, sigma = self.params(x)
        comp = torch.multinomial(logpi.exp(), m, replacement=True)            # (n, m)
        mu_s, sg_s = torch.gather(mu, 1, comp), torch.gather(sigma, 1, comp)
        return mu_s + sg_s * torch.randn_like(mu_s)


class _Standardizer:
    def fit(self, a: np.ndarray) -> "_Standardizer":
        self.m = a.mean(axis=0)
        self.s = a.std(axis=0) + 1e-8
        return self

    def __call__(self, a: np.ndarray) -> np.ndarray:
        return (a - self.m) / self.s


class _DeepStructural(BaseElasticityModel):
    """Shared plumbing: scaling, device handling, autograd elasticity, structural demand curve."""

    family = "deep_iv"
    is_iv = True

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.device = get_device(self.cfg)
        self.seed = self.cfg["project"]["seed"]

    # -- scaling -------------------------------------------------------------------------------
    def _fit_scalers(self, train: pd.DataFrame) -> None:
        self.sx = _Standardizer().fit(train[FEATURES].to_numpy(float))
        self.sz = _Standardizer().fit(train[INSTRUMENTS].to_numpy(float))
        self.st = _Standardizer().fit(train[[TREATMENT]].to_numpy(float))
        self.sy = _Standardizer().fit(train[[OUTCOME]].to_numpy(float))

    def _tensors(self, df: pd.DataFrame, with_y: bool = True):
        dev = self.device
        x = torch.tensor(self.sx(df[FEATURES].to_numpy(float)), dtype=torch.float32, device=dev)
        z = torch.tensor(self.sz(df[INSTRUMENTS].to_numpy(float)), dtype=torch.float32, device=dev)
        t = torch.tensor(self.st(df[[TREATMENT]].to_numpy(float))[:, 0], dtype=torch.float32, device=dev)
        if not with_y:
            return x, z, t
        y = torch.tensor(self.sy(df[[OUTCOME]].to_numpy(float))[:, 0], dtype=torch.float32, device=dev)
        return x, z, t, y

    def _val_split(self, train: pd.DataFrame, frac: float = 0.15):
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(train))
        nv = int(len(train) * frac)
        return train.iloc[idx[nv:]], train.iloc[idx[:nv]]

    # -- structural function h(X, ln p) --------------------------------------------------------
    def _h(self, x: torch.Tensor, t_std: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def predict_log_demand(self, df: pd.DataFrame, ln_price: np.ndarray) -> np.ndarray:
        x = torch.tensor(self.sx(df[FEATURES].to_numpy(float)), dtype=torch.float32, device=self.device)
        t = torch.tensor(self.st(np.asarray(ln_price, float)[:, None])[:, 0], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self._h(x, t).cpu().numpy()
        return out * self.sy.s[0] + self.sy.m[0]

    def elasticity(self, df: pd.DataFrame) -> np.ndarray:
        """d h / d ln p evaluated at each row's reference price (autograd, central in original units)."""
        ln_p = ln_reference_price(df, self.cfg)
        x = torch.tensor(self.sx(df[FEATURES].to_numpy(float)), dtype=torch.float32, device=self.device)
        t = torch.tensor(self.st(ln_p[:, None])[:, 0], dtype=torch.float32, device=self.device).requires_grad_(True)
        h = self._h(x, t)
        (g,) = torch.autograd.grad(h.sum(), t)
        # d h_orig / d ln p = sy.s * dh/dt_std / st.s
        return (g.detach().cpu().numpy() * self.sy.s[0] / self.st.s[0]).astype(float)

    def save(self, path) -> None:
        torch.save({k: v.state_dict() for k, v in self._modules_to_save().items()}, path)

    def _modules_to_save(self) -> Dict[str, nn.Module]:  # pragma: no cover - abstract
        raise NotImplementedError


class DeepIVTorch(_DeepStructural):
    name = "DeepIV (PyTorch)"

    def fit(self, train: pd.DataFrame) -> "DeepIVTorch":
        t0 = time.time()
        set_seed(self.seed)
        cfg = self.cfg["models"]["deep"]
        self._fit_scalers(train)
        tr, va = self._val_split(train)
        xt, zt, tt, yt = self._tensors(tr)
        xv, zv, tv, yv = self._tensors(va)
        hid, bs = cfg["hidden"], cfg["batch_size"]
        n = len(tr)

        # ---------------- Stage 1: MDN  P(T | X, Z) ----------------
        self.mdn = MDN(xt.shape[1] + zt.shape[1], hid, cfg["mdn_components"]).to(self.device)
        opt = torch.optim.Adam(self.mdn.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        best, best_state, bad = float("inf"), None, 0
        hist1: Dict[str, List[float]] = {"train": [], "val": []}
        for ep in range(cfg["epochs"]):
            self.mdn.train()
            perm = torch.randperm(n, device=self.device)
            tot = 0.0
            for i in range(0, n, bs):
                b = perm[i: i + bs]
                loss = self.mdn.nll(torch.cat([xt[b], zt[b]], 1), tt[b])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.mdn.parameters(), 5.0)
                opt.step()
                tot += loss.item() * len(b)
            self.mdn.eval()
            with torch.no_grad():
                vl = self.mdn.nll(torch.cat([xv, zv], 1), tv).item()
            hist1["train"].append(tot / n)
            hist1["val"].append(vl)
            log.info("[DeepIV stage1 MDN] epoch %02d/%d train_nll=%.4f val_nll=%.4f", ep + 1, cfg["epochs"], tot / n, vl)
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.mdn.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    log.info("[DeepIV stage1] early stop at epoch %d", ep + 1)
                    break
        if best_state is not None:
            self.mdn.load_state_dict(best_state)
        self.mdn.eval()
        for p in self.mdn.parameters():
            p.requires_grad_(False)

        # ---------------- Stage 2: structural outcome net h(X, T) ----------------
        self.h_net = mlp(xt.shape[1] + 1, 1, hid, depth=3).to(self.device)
        opt2 = torch.optim.Adam(self.h_net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        m_s = cfg["mc_samples"]
        best, best_state, bad = float("inf"), None, 0
        hist2: Dict[str, List[float]] = {"train": [], "val": []}

        def mc_loss(x, z, y):
            ts = self.mdn.sample(torch.cat([x, z], 1), m_s)                      # (b, M)
            xr = x[:, None, :].expand(-1, m_s, -1)
            inp = torch.cat([xr, ts[..., None]], -1)
            hbar = self.h_net(inp).squeeze(-1).mean(dim=1)                         # integrate over F(t|x,z)
            return F.mse_loss(hbar, y)

        for ep in range(cfg["epochs"]):
            self.h_net.train()
            perm = torch.randperm(n, device=self.device)
            tot = 0.0
            for i in range(0, n, bs):
                b = perm[i: i + bs]
                loss = mc_loss(xt[b], zt[b], yt[b])
                opt2.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.h_net.parameters(), 5.0)
                opt2.step()
                tot += loss.item() * len(b)
            self.h_net.eval()
            with torch.no_grad():
                vl = mc_loss(xv, zv, yv).item()
            hist2["train"].append(tot / n)
            hist2["val"].append(vl)
            log.info("[DeepIV stage2 structural] epoch %02d/%d train_mse=%.4f val_mse=%.4f", ep + 1, cfg["epochs"], tot / n, vl)
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.h_net.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    log.info("[DeepIV stage2] early stop at epoch %d", ep + 1)
                    break
        if best_state is not None:
            self.h_net.load_state_dict(best_state)
        self.h_net.eval()

        self.loss_history = {"stage1_train_nll": hist1["train"], "stage1_val_nll": hist1["val"],
                             "stage2_train_mse": hist2["train"], "stage2_val_mse": hist2["val"]}
        self.fit_seconds = time.time() - t0
        return self

    def _h(self, x: torch.Tensor, t_std: torch.Tensor) -> torch.Tensor:
        return self.h_net(torch.cat([x, t_std[:, None]], 1)).squeeze(-1)

    def _modules_to_save(self):
        return {"stage1_mdn": self.mdn, "stage2_outcome": self.h_net}

    def save_stages(self, stage1_path, stage2_path) -> None:
        torch.save(self.mdn.state_dict(), stage1_path)
        torch.save(self.h_net.state_dict(), stage2_path)


class DeepGMMTorch(_DeepStructural):
    """Optimally-weighted min-max GMM:  min_g max_f  E[f(X,Z) psi] - 1/4 E[f(X,Z)^2 psi^2],  psi = Y - g(X,T)."""

    name = "DeepGMM (PyTorch)"

    def fit(self, train: pd.DataFrame) -> "DeepGMMTorch":
        t0 = time.time()
        set_seed(self.seed)
        cfg, gcfg = self.cfg["models"]["deep"], self.cfg["models"]["deepgmm"]
        self._fit_scalers(train)
        tr, va = self._val_split(train)
        xt, zt, tt, yt = self._tensors(tr)
        xv, zv, tv, yv = self._tensors(va)
        hid, bs, n = cfg["hidden"], gcfg["batch_size"], len(tr)
        k_test = 8
        self.g_net = mlp(xt.shape[1] + 1, 1, hid, depth=3).to(self.device)
        self.f_net = mlp(xt.shape[1] + zt.shape[1], k_test, hid, depth=2, act=nn.Tanh).to(self.device)
        opt_g = torch.optim.Adam(self.g_net.parameters(), lr=gcfg["lr_learner"], betas=(0.5, 0.9), weight_decay=cfg["weight_decay"])
        opt_f = torch.optim.Adam(self.f_net.parameters(), lr=gcfg["lr_adversary"], betas=(0.5, 0.9))

        def objective(x, z, t, y):
            psi = y - self.g_net(torch.cat([x, t[:, None]], 1)).squeeze(-1)           # (b,)
            f = self.f_net(torch.cat([x, z], 1))                                         # (b, K)
            return ((f * psi[:, None]).mean(0) - 0.25 * ((f * psi[:, None]) ** 2).mean(0)).sum(), psi

        hist = {"train": [], "val": [], "val_mse": []}
        best, best_state, bad = float("inf"), None, 0
        for ep in range(gcfg["epochs"]):
            perm = torch.randperm(n, device=self.device)
            tot = 0.0
            for i in range(0, n, bs):
                b = perm[i: i + bs]
                # adversary ascent
                obj, _ = objective(xt[b], zt[b], tt[b], yt[b])
                opt_f.zero_grad()
                (-obj).backward()
                nn.utils.clip_grad_norm_(self.f_net.parameters(), 5.0)
                opt_f.step()
                # learner descent
                obj, psi = objective(xt[b], zt[b], tt[b], yt[b])
                opt_g.zero_grad()
                obj.backward()
                nn.utils.clip_grad_norm_(self.g_net.parameters(), 5.0)
                opt_g.step()
                tot += obj.item() * len(b)
            with torch.no_grad():
                vobj, vpsi = objective(xv, zv, tv, yv)
                vmse = float((vpsi ** 2).mean())
            hist["train"].append(tot / n)
            hist["val"].append(float(vobj))
            hist["val_mse"].append(vmse)
            log.info("[DeepGMM] epoch %02d/%d train_moment_obj=%.5f val_moment_obj=%.5f val_psi_mse=%.4f",
                     ep + 1, gcfg["epochs"], tot / n, float(vobj), vmse)
            # model selection by the *validation moment objective* (an IV-valid criterion), not by MSE
            if float(vobj) < best:
                best, bad = float(vobj), 0
                best_state = {k: v.clone() for k, v in self.g_net.state_dict().items()}
            else:
                bad += 1
        if best_state is not None:
            self.g_net.load_state_dict(best_state)
        self.g_net.eval()
        self.loss_history = {"train_moment_objective": hist["train"], "val_moment_objective": hist["val"],
                             "val_residual_mse": hist["val_mse"]}
        self.fit_seconds = time.time() - t0
        return self

    def _h(self, x: torch.Tensor, t_std: torch.Tensor) -> torch.Tensor:
        return self.g_net(torch.cat([x, t_std[:, None]], 1)).squeeze(-1)

    def _modules_to_save(self):
        return {"learner_g": self.g_net, "adversary_f": self.f_net}
