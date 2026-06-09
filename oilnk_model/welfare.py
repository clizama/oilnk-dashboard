"""Welfare loss  L = Var(pi_core) + theta_x * Var(x)  under each policy.

Unconditional variances come from the discrete Lyapunov equation of the solved
law of motion, for one selected shock at a time. With the oil shock this
reproduces deck7's loss ratios (optimal=1, core x1.84, headline x3.21).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from .model import IDX, SHOCKS, solve
from .optimal import RIDX, solve_optimal
from .params import Params

# standard-deviation of each structural shock (matches the Dynare .mod)
SIGMA = {"oil": 0.10, "costpush": 0.01}


def _loss_from(G1, impact, ipi, ix, theta_x, sigma, col):
    imp = impact[:, col]
    Q = sigma ** 2 * np.outer(imp, imp)
    Sigma = solve_discrete_lyapunov(G1, Q)
    return float(Sigma[ipi, ipi] + theta_x * Sigma[ix, ix])


def loss_rule(p: Params, rule_cpi: float, which: str = "oil") -> float:
    G1, impact, _ = solve(p.update(rule_cpi=rule_cpi))
    return _loss_from(G1, impact, IDX["pi"], IDX["x"], p.theta_x, SIGMA[which], SHOCKS[which])


def loss_optimal(p: Params, which: str = "oil") -> float:
    G1, impact, _ = solve_optimal(p)
    return _loss_from(G1, impact, RIDX["pi"], RIDX["x"], p.theta_x, SIGMA[which], SHOCKS[which])


def welfare_table(p: Params, which: str = "oil") -> dict:
    """Absolute and relative-to-optimal losses for the three policies."""
    opt = loss_optimal(p, which)
    core = loss_rule(p, 0.0, which)
    head = loss_rule(p, 1.0, which)
    out = {"optimo": opt, "core": core, "headline": head}
    rel = {k: (v / opt if opt > 0 else float("nan")) for k, v in out.items()}
    return {"loss": out, "rel": rel}
