"""TARNet and DragonNet (with Targeted Regularisation) in PyTorch, on CUDA when available.

Early stopping uses a validation split *by DMA* (rows of one DMA share a treatment status).
Outcome and inputs are standardised internally; CATE is returned on the original outcome scale.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.models.meta_learners import BaseCATE, E_KNOWN
from src.utils.config import load_config
from src.utils.device import get_hardware


class _Net(nn.Module):
    def __init__(self, d_in: int, hidden: int, rep: int, dragon: bool):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, hidden), nn.ELU(), nn.Linear(hidden, rep), nn.ELU())
        head = lambda: nn.Sequential(nn.Linear(rep, hidden), nn.ELU(), nn.Linear(hidden, 1))
        self.h0, self.h1 = head(), head()
        self.dragon = dragon
        if dragon:
            self.prop = nn.Linear(rep, 1)
            self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        z = self.enc(x)
        out = {"y0": self.h0(z).squeeze(-1), "y1": self.h1(z).squeeze(-1)}
        if self.dragon:
            out["e"] = torch.sigmoid(self.prop(z)).squeeze(-1).clamp(0.02, 0.98)
        return out


class _DeepCATE(BaseCATE):
    dragon = False

    def __init__(self, seed: int = 0, **over):
        super().__init__("torch", "estimated", seed)
        self.cfg = {**load_config()["models"]["deep"], **over}
        self.device = get_hardware().torch_device
        self.history = []

    def _loss(self, out, y, t):
        pred = torch.where(t > 0.5, out["y1"], out["y0"])
        loss = ((pred - y) ** 2).mean()
        if not self.dragon:
            return loss
        c = self.cfg
        bce = nn.functional.binary_cross_entropy(out["e"], t)
        e = out["e"]
        h = t / e - (1 - t) / (1 - e)
        q_tilde = pred + self.net.eps * h                                 # targeted regularisation
        treg = ((y - q_tilde) ** 2).mean()
        return loss + c["dragonnet_alpha"] * bce + c["dragonnet_beta"] * treg

    def fit(self, X, T, Y, groups):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        c = self.cfg
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-8
        self.ym, self.ys = Y.mean(), Y.std() + 1e-8
        Xn = torch.tensor((X - self.xm) / self.xs, dtype=torch.float32, device=self.device)
        yn = torch.tensor((Y - self.ym) / self.ys, dtype=torch.float32, device=self.device)
        tn = torch.tensor(T, dtype=torch.float32, device=self.device)
        ug = np.unique(groups)
        rng = np.random.default_rng(self.seed)
        val_g = rng.choice(ug, max(2, int(0.2 * len(ug))), replace=False)
        va = np.isin(groups, val_g)
        if T[~va].sum() == 0 or T[va].sum() == 0:       # ensure both arms in train and val
            va = np.zeros(len(T), bool)
        tr_idx = torch.tensor(np.where(~va)[0], device=self.device)
        va_idx = torch.tensor(np.where(va)[0], device=self.device)
        self.net = _Net(X.shape[1], c["hidden"], c["rep_dim"], self.dragon).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        best, best_state, bad = float("inf"), None, 0
        for ep in range(c["epochs"]):
            self.net.train()
            perm = tr_idx[torch.randperm(len(tr_idx), device=self.device)]
            for i in range(0, len(perm), c["batch_size"]):
                b = perm[i:i + c["batch_size"]]
                opt.zero_grad()
                loss = self._loss(self.net(Xn[b]), yn[b], tn[b])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vi = va_idx if len(va_idx) else tr_idx
                vl = self._loss(self.net(Xn[vi]), yn[vi], tn[vi]).item()
            self.history.append(vl)
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= c["patience"]:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    @torch.no_grad()
    def predict(self, X):
        Xn = torch.tensor((X - self.xm) / self.xs, dtype=torch.float32, device=self.device)
        out = self.net(Xn)
        tau = out["y1"] - out["y0"]
        if self.dragon:  # targeted-regularisation shift of the effect estimate
            tau = tau + self.net.eps * (1 / out["e"] + 1 / (1 - out["e"]))
        return (tau * self.ys).cpu().numpy()

    @torch.no_grad()
    def propensity(self, X):
        Xn = torch.tensor((X - self.xm) / self.xs, dtype=torch.float32, device=self.device)
        return self.net(Xn)["e"].cpu().numpy() if self.dragon else np.full(len(X), E_KNOWN)


class TARNet(_DeepCATE):
    name = "TARNet"
    dragon = False


class DragonNet(_DeepCATE):
    name = "DragonNet"
    dragon = True
