"""PyTorch (CUDA) TARNet and DragonNet with targeted regularization."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.config import load_config
from src.common.hardware import torch_device
from src.models.cate_learners import BaseCATEEstimator


class _Net(nn.Module):
    def __init__(self, d_in, hidden, rep_layers, dragon: bool):
        super().__init__()
        layers, d = [], d_in
        for _ in range(rep_layers):
            layers += [nn.Linear(d, hidden), nn.ELU()]
            d = hidden
        self.phi = nn.Sequential(*layers)
        head = lambda: nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ELU(), nn.Linear(hidden // 2, hidden // 2),
                                     nn.ELU(), nn.Linear(hidden // 2, 1))
        self.h1, self.h0 = head(), head()
        self.dragon = dragon
        if dragon:
            self.prop = nn.Linear(hidden, 1)
            self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        z = self.phi(x)
        out = dict(o1=self.h1(z).squeeze(-1), o0=self.h0(z).squeeze(-1))
        if self.dragon:
            out["e"] = torch.sigmoid(self.prop(z).squeeze(-1)).clamp(0.01, 0.99)
        return out


class _DeepCATE(BaseCATEEstimator):
    family = "deep"
    dragon = False

    def _fit(self, X, T, y):
        cfg = load_config()["deep"]
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        dev = torch_device()
        self.dev = dev
        self.mu_x = X.values.mean(0); self.sd_x = X.values.std(0) + 1e-6
        self.mu_y, self.sd_y = (0.0, 1.0) if self.binary else (y.mean(), y.std() + 1e-6)
        x = torch.tensor((X.values - self.mu_x) / self.sd_x, dtype=torch.float32, device=dev)
        t = torch.tensor(T, dtype=torch.float32, device=dev)
        yy = torch.tensor((y - self.mu_y) / self.sd_y, dtype=torch.float32, device=dev)
        n = len(x)
        perm = torch.randperm(n, device=dev)
        nv = int(0.15 * n)
        va, tr = perm[:nv], perm[nv:]
        self.net = _Net(x.shape[1], cfg["hidden"], cfg["rep_layers"], self.dragon).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        epochs = cfg["epochs_fast"] if self.fast else cfg["epochs"]
        best, best_state, bad = np.inf, None, 0
        for ep in range(epochs):
            self.net.train()
            idx = tr[torch.randperm(len(tr), device=dev)]
            for i in range(0, len(idx), cfg["batch_size"]):
                b = idx[i:i + cfg["batch_size"]]
                loss = self._loss(x[b], t[b], yy[b], cfg)
                opt.zero_grad(); loss.backward(); opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = self._loss(x[va], t[va], yy[va], cfg).item()
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.epochs_run = ep + 1
        self.net.eval()

    def _outcome_loss(self, o, y):
        if self.binary:
            return F.binary_cross_entropy_with_logits(o, y)
        return F.mse_loss(o, y)

    def _loss(self, x, t, y, cfg):
        out = self.net(x)
        of = torch.where(t == 1, out["o1"], out["o0"])
        loss = self._outcome_loss(of, y)
        if self.dragon:
            e = out["e"]
            loss = loss + cfg["dragon_alpha"] * F.binary_cross_entropy(e, t)
            # targeted regularization: perturb the factual prediction along the clever covariate
            pf = torch.sigmoid(of) if self.binary else of
            h = (t - e) / (e * (1 - e))
            y_tilde = pf + self.net.eps * h
            loss = loss + cfg["dragon_beta"] * F.mse_loss(y_tilde, y)
        return loss

    @torch.no_grad()
    def _predict(self, X):
        x = torch.tensor((X.values - self.mu_x) / self.sd_x, dtype=torch.float32, device=self.dev)
        o = self.net(x)
        if self.binary:
            return (torch.sigmoid(o["o1"]) - torch.sigmoid(o["o0"])).cpu().numpy()
        return ((o["o1"] - o["o0"]) * self.sd_y).cpu().numpy()

    @torch.no_grad()
    def predict_propensity(self, X):
        if not self.dragon:
            return None
        x = torch.tensor((X.values - self.mu_x) / self.sd_x, dtype=torch.float32, device=self.dev)
        return self.net(x)["e"].cpu().numpy()

    def save(self, path):
        torch.save(dict(state=self.net.state_dict(), mu_x=self.mu_x, sd_x=self.sd_x, mu_y=self.mu_y, sd_y=self.sd_y,
                        binary=self.binary, dragon=self.dragon, d_in=len(self.mu_x),
                        cfg=load_config()["deep"]), path)

    def __getstate__(self):  # allow joblib too
        s = dict(self.__dict__)
        s.pop("net", None); s.pop("dev", None)
        return s


class TARNet(_DeepCATE):
    name = "TARNet (PyTorch)"
    dragon = False


class DragonNet(_DeepCATE):
    name = "DragonNet (PyTorch)"
    dragon = True


def load_deep(path, device=None):
    """Rebuild a saved TARNet/DragonNet for inference."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    m = (DragonNet if d["dragon"] else TARNet)(binary=d["binary"])
    m.dev = device or torch_device()
    m.net = _Net(d["d_in"], d["cfg"]["hidden"], d["cfg"]["rep_layers"], d["dragon"]).to(m.dev)
    m.net.load_state_dict(d["state"]); m.net.eval()
    m.mu_x, m.sd_x, m.mu_y, m.sd_y = d["mu_x"], d["sd_x"], d["mu_y"], d["sd_y"]
    return m
