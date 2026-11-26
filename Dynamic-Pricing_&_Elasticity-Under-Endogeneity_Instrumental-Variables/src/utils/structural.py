"""Structural demand / revenue helpers shared by the DGP, the optimizer and the models.

Structural model (per stay-night):   log D = c + theta * ln(net price) + gamma_u * U + eps
Occupancy                       :   occ = 1 - exp(-D)         (saturating, bounded in [0, 1))
Net RevPAR                      :   occ * (net_price - unit_cost)
"""
from __future__ import annotations

import numpy as np

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(15)  # probabilists' Hermite: weight exp(-x^2/2)
_GH_W = _GH_W / _GH_W.sum()


def expected_occupancy(log_d: np.ndarray, noise_sd: float) -> np.ndarray:
    """E_eps[1 - exp(-exp(log_d + eps))], eps ~ N(0, noise_sd^2), by Gauss-Hermite quadrature."""
    log_d = np.asarray(log_d, float)
    if noise_sd <= 0:
        return 1.0 - np.exp(-np.exp(log_d))
    shifted = log_d[..., None] + noise_sd * _GH_X
    return (1.0 - np.exp(-np.exp(shifted))) @ _GH_W


def net_revpar(occupancy: np.ndarray, price: np.ndarray, discount: np.ndarray | float, unit_cost: float) -> np.ndarray:
    net_price = np.asarray(price, float) * (1.0 - np.asarray(discount, float))
    return np.asarray(occupancy, float) * (net_price - unit_cost)
