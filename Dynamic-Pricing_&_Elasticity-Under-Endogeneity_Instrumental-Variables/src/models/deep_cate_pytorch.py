"""PyTorch (CUDA when available) deep CATE networks: TARNet, DragonNet, CEVAE (promo treatment)."""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.generate_hospitality_data import OUTCOME
from src.models.base import META_FEATURES, BaseElasticityModel
from src.models.deep_iv_pytorch import _Standardizer, mlp
from src.utils.gpu_utils import get_device, set_seed
from src.utils.logger import get_logger

log = get_logger("train_benchmark", "train_benchmark.log")


class _DeepCate(BaseElasticityModel):
    family = "cate_deep"
    native_target = "promo_cate"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.device = get_device(self.cfg)
        self.seed = self.cfg["project"]["seed"]
        self.model: nn.Module | None = None

    # -- data ----------------------------------------------------------------------------------
    def _prep(self, train: pd.DataFrame):
        self.sx = _Standardizer().fit(train[META_FEATURES].to_numpy(float))
        self.sy = _Standardizer().fit(train[[OUTCOME]].to_numpy(float))
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(train))
        nv = int(0.15 * len(train))
        parts = []
        for sl in (idx[nv:], idx[:nv]):
            d = train.iloc[sl]
            parts.append((
                torch.tensor(self.sx(d[META_FEATURES].to_numpy(float)), dtype=torch.float32, device=self.device),
                torch.tensor(d["promo"].to_numpy(float), dtype=torch.float32, device=self.device),
                torch.tensor(self.sy(d[[OUTCOME]].to_numpy(float))[:, 0], dtype=torch.float32, device=self.device),
            ))
        return parts

    def _tx(self, df: pd.DataFrame) -> torch.Tensor:
        return torch.tensor(self.sx(df[META_FEATURES].to_numpy(float)), dtype=torch.float32, device=self.device)

    # -- generic training loop -----------------------------------------------------------------
    def _train(self, loss_fn, params, train_part, val_part, tag: str, epochs: int | None = None):
        cfg = self.cfg["models"]["deep"]
        epochs = epochs or cfg["epochs"]
        opt = torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        x, w, y = train_part
        n, bs = len(x), cfg["batch_size"]
        hist = {"train": [], "val": []}
        best, best_state, bad = float("inf"), None, 0
        for ep in range(epochs):
            self.model.train()
            perm = torch.randperm(n, device=self.device)
            tot = 0.0
            for i in range(0, n, bs):
                b = perm[i: i + bs]
                loss = loss_fn(x[b], w[b], y[b])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
                tot += loss.item() * len(b)
            self.model.eval()
            with torch.no_grad():
                vl = loss_fn(*val_part).item()
            hist["train"].append(tot / n)
            hist["val"].append(vl)
            log.info("[%s] epoch %02d/%d train_loss=%.4f val_loss=%.4f", tag, ep + 1, epochs, tot / n, vl)
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    log.info("[%s] early stop at epoch %d", tag, ep + 1)
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self.loss_history = {"train_loss": hist["train"], "val_loss": hist["val"]}

    def _tau_std(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def promo_cate(self, df: pd.DataFrame) -> np.ndarray:
        with torch.no_grad():
            tau = self._tau_std(self._tx(df)).cpu().numpy()
        return tau.astype(float) * self.sy.s[0]

    def elasticity(self, df: pd.DataFrame) -> np.ndarray:
        return self.promo_cate(df) / self.ln_one_minus_d

    def propensity(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), np.nan)


class _TwoHeadNet(nn.Module):
    def __init__(self, d_in: int, hid: int, dragon: bool):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_in, hid), nn.ELU(), nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, hid), nn.ELU())
        self.h0 = mlp(hid, 1, hid // 2, depth=2, act=nn.ELU)
        self.h1 = mlp(hid, 1, hid // 2, depth=2, act=nn.ELU)
        self.dragon = dragon
        if dragon:
            self.e_head = nn.Linear(hid, 1)
            self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        r = self.phi(x)
        y0, y1 = self.h0(r).squeeze(-1), self.h1(r).squeeze(-1)
        e = torch.sigmoid(self.e_head(r)).squeeze(-1).clamp(0.02, 0.98) if self.dragon else None
        return y0, y1, e


class TARNetTorch(_DeepCate):
    """Treatment-Agnostic Representation Network: shared Phi(x) + two outcome heads."""

    name = "TARNet"

    def fit(self, train: pd.DataFrame) -> "TARNetTorch":
        t0 = time.time()
        set_seed(self.seed)
        tr, va = self._prep(train)
        self.model = _TwoHeadNet(tr[0].shape[1], self.cfg["models"]["deep"]["hidden"], dragon=False).to(self.device)

        def loss_fn(x, w, y):
            y0, y1, _ = self.model(x)
            return F.mse_loss(torch.where(w > 0.5, y1, y0), y)

        self._train(loss_fn, self.model.parameters(), tr, va, "TARNet")
        self.fit_seconds = time.time() - t0
        self.fit_structural_intercept(train)
        return self

    def _tau_std(self, x):
        y0, y1, _ = self.model(x)
        return y1 - y0


class DragonNetTorch(_DeepCate):
    """DragonNet: shared representation, propensity head and targeted regularisation (Shi et al., 2019)."""

    name = "DragonNet"

    def __init__(self, cfg=None, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(cfg)
        self.alpha, self.beta = alpha, beta

    def fit(self, train: pd.DataFrame) -> "DragonNetTorch":
        t0 = time.time()
        set_seed(self.seed)
        tr, va = self._prep(train)
        self.model = _TwoHeadNet(tr[0].shape[1], self.cfg["models"]["deep"]["hidden"], dragon=True).to(self.device)

        def loss_fn(x, w, y):
            y0, y1, e = self.model(x)
            yw = torch.where(w > 0.5, y1, y0)
            outcome = F.mse_loss(yw, y)
            prop = F.binary_cross_entropy(e, w)
            h = w / e - (1 - w) / (1 - e)                                   # clever covariate
            y_tilde = yw.detach() + self.model.eps * h
            targeted = F.mse_loss(y_tilde, y)
            return outcome + self.alpha * prop + self.beta * targeted

        self._train(loss_fn, self.model.parameters(), tr, va, "DragonNet")
        self.fit_seconds = time.time() - t0
        self.fit_structural_intercept(train)
        return self

    def _tau_std(self, x):
        y0, y1, _ = self.model(x)
        return y1 - y0

    def propensity(self, df: pd.DataFrame) -> np.ndarray:
        with torch.no_grad():
            return self.model(self._tx(df))[2].cpu().numpy().astype(float)


class _CEVAENet(nn.Module):
    def __init__(self, d_x: int, hid: int, zdim: int):
        super().__init__()
        self.zdim = zdim
        # inference network q(z | x, w, y) and auxiliary heads q(w | x), q(y | x, w)
        self.enc = mlp(d_x + 2, 2 * zdim, hid, depth=2, act=nn.ELU)
        self.q_w = mlp(d_x, 1, hid // 2, depth=2, act=nn.ELU)
        self.q_y = nn.ModuleList([mlp(d_x, 1, hid // 2, depth=2, act=nn.ELU) for _ in range(2)])
        # generative network p(x | z), p(w | z), p(y | z, w)
        self.p_x = mlp(zdim, 2 * d_x, hid, depth=2, act=nn.ELU)
        self.p_w = mlp(zdim, 1, hid // 2, depth=1, act=nn.ELU)
        self.p_y = nn.ModuleList([mlp(zdim, 1, hid // 2, depth=2, act=nn.ELU) for _ in range(2)])

    def encode(self, x, w, y):
        o = self.enc(torch.cat([x, w[:, None], y[:, None]], 1))
        return o[:, : self.zdim], o[:, self.zdim:].clamp(-6, 3)

    def y_given_z(self, z, w):
        y0, y1 = self.p_y[0](z).squeeze(-1), self.p_y[1](z).squeeze(-1)
        return torch.where(w > 0.5, y1, y0)


class CEVAETorch(_DeepCate):
    """Causal Effect VAE (Louizos et al., 2017): latent confounder recovered from proxies X (bonus model)."""

    name = "CEVAE"

    def fit(self, train: pd.DataFrame) -> "CEVAETorch":
        t0 = time.time()
        set_seed(self.seed)
        tr, va = self._prep(train)
        d_x = tr[0].shape[1]
        self.model = _CEVAENet(d_x, self.cfg["models"]["deep"]["hidden"], self.cfg["models"]["cevae"]["latent_dim"]).to(self.device)
        m = self.model

        def loss_fn(x, w, y):
            mu, logvar = m.encode(x, w, y)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            px = m.p_x(z)
            x_rec = F.gaussian_nll_loss(px[:, :d_x], x, F.softplus(px[:, d_x:]) + 1e-3, reduction="mean")
            w_rec = F.binary_cross_entropy_with_logits(m.p_w(z).squeeze(-1), w)
            y_rec = F.mse_loss(m.y_given_z(z, w), y)
            kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1))
            # auxiliary predictive heads used at test time
            aux_w = F.binary_cross_entropy_with_logits(m.q_w(x).squeeze(-1), w)
            aux_y = F.mse_loss(torch.where(w > 0.5, m.q_y[1](x).squeeze(-1), m.q_y[0](x).squeeze(-1)), y)
            return x_rec + w_rec + y_rec + 0.1 * kl + aux_w + aux_y

        self._train(loss_fn, m.parameters(), tr, va, "CEVAE", epochs=self.cfg["models"]["cevae"]["epochs"])
        self.fit_seconds = time.time() - t0
        self.fit_structural_intercept(train)
        return self

    def _tau_std(self, x, n_samples: int = 16):
        m = self.model
        tau = torch.zeros(len(x), device=self.device)
        pw = torch.sigmoid(m.q_w(x).squeeze(-1))
        for _ in range(n_samples):
            w = torch.bernoulli(pw)
            y = torch.where(w > 0.5, m.q_y[1](x).squeeze(-1), m.q_y[0](x).squeeze(-1))
            mu, logvar = m.encode(x, w, y)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            tau += m.p_y[1](z).squeeze(-1) - m.p_y[0](z).squeeze(-1)
        return tau / n_samples
