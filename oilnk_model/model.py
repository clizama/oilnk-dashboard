"""Master Blanchard-Gali model in gensys canonical form.

The equations are exactly those of deck6/deck7's Dynare `.mod` (oil in
production and consumption, real-wage rigidity, Calvo prices, Taylor rule on
core or headline). Forward terms in deterministic-state variables are
substituted analytically:
    E_t s_{t+1}  = rho_s * s_t           (AR(1) oil price)
    E_t ye_{t+1} = -psi_e * rho_s * s_t  (ye = -psi_e s)
so the efficient rate collapses to  r_e = sigma*(psi_e+chi)*(1-rho_s)*s.
The only genuine jumps are pi, picpi, x (three expectational errors).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .gensys import gensys
from .params import Params

# endogenous variables (order fixed); the last three are the expectations
# E_t[pi_{t+1}], E_t[picpi_{t+1}], E_t[x_{t+1}]. u_exo is the exogenous cost-push.
VARS = [
    "pi", "picpi", "x", "i", "r_e", "y", "ye", "n", "c", "w",
    "mrs", "wedge", "u", "s", "u_exo", "Epi", "Epicpi", "Ex",
]
IDX = {v: k for k, v in enumerate(VARS)}
N = len(VARS)
# observable variables (exclude the auxiliary expectation variables)
OBS = VARS[:15]

# shock columns: 0 = oil price (eps_s), 1 = exogenous cost-push (eps_u)
SHOCKS = {"oil": 0, "costpush": 1}


def build_matrices(p: Params):
    """Return (g0, g1, psi, pi_mat) for the gensys canonical form."""
    g0 = np.zeros((N, N))
    g1 = np.zeros((N, N))
    psi = np.zeros((N, 2))        # shocks: col 0 = eps_s (oil), col 1 = eps_u (cost-push)
    pim = np.zeros((N, 3))        # expectational errors: pi, picpi, x

    i_ = IDX
    sig, phi, gam, chi = p.sigma, p.phi, p.gamma, p.chi
    kappa, lam, zeta, psi_e, beta = p.kappa, p.lam, p.zeta, p.psi_e, p.beta
    rho, rc, ppi, px, rho_u = p.rho_s, p.rule_cpi, p.phi_pi, p.phi_x, p.rho_u

    r = 0
    # E1 NKPC: pi = beta*Epi + kappa*x + u + u_exo
    g0[r, i_["pi"]] = 1; g0[r, i_["x"]] = -kappa; g0[r, i_["u"]] = -1
    g0[r, i_["u_exo"]] = -1; g0[r, i_["Epi"]] = -beta; r += 1
    # E2 cost-push: u = lam*wedge
    g0[r, i_["u"]] = 1; g0[r, i_["wedge"]] = -lam; r += 1
    # E3 wedge = w - mrs
    g0[r, i_["wedge"]] = 1; g0[r, i_["w"]] = -1; g0[r, i_["mrs"]] = 1; r += 1
    # E4 mrs = sigma*c + phi*n
    g0[r, i_["mrs"]] = 1; g0[r, i_["c"]] = -sig; g0[r, i_["n"]] = -phi; r += 1
    # E5 rigidity: w = gamma*w(-1) + (1-gamma)*mrs
    g0[r, i_["w"]] = 1; g0[r, i_["mrs"]] = -(1 - gam); g1[r, i_["w"]] = gam; r += 1
    # E6 production: n = y + zeta*s
    g0[r, i_["n"]] = 1; g0[r, i_["y"]] = -1; g0[r, i_["s"]] = -zeta; r += 1
    # E7 consumption: c = y - chi*s
    g0[r, i_["c"]] = 1; g0[r, i_["y"]] = -1; g0[r, i_["s"]] = chi; r += 1
    # E8 efficient output: ye = -psi_e*s
    g0[r, i_["ye"]] = 1; g0[r, i_["s"]] = psi_e; r += 1
    # E9 welfare gap: x = y - ye
    g0[r, i_["x"]] = 1; g0[r, i_["y"]] = -1; g0[r, i_["ye"]] = 1; r += 1
    # E10 efficient rate: r_e = sigma*(psi_e+chi)*(1-rho)*s
    g0[r, i_["r_e"]] = 1; g0[r, i_["s"]] = -sig * (psi_e + chi) * (1 - rho); r += 1
    # E11 IS: x = Ex - (1/sigma)*(i - Epicpi - r_e)
    g0[r, i_["x"]] = 1; g0[r, i_["Ex"]] = -1
    g0[r, i_["i"]] = 1 / sig; g0[r, i_["Epicpi"]] = -1 / sig; g0[r, i_["r_e"]] = -1 / sig
    r += 1
    # E12 headline: picpi = pi + chi*(s - s(-1))
    g0[r, i_["picpi"]] = 1; g0[r, i_["pi"]] = -1; g0[r, i_["s"]] = -chi
    g1[r, i_["s"]] = -chi; r += 1
    # E13 Taylor rule: i = r_e + Epicpi + phi_pi*((1-rc)*pi + rc*picpi) + phi_x*x
    g0[r, i_["i"]] = 1; g0[r, i_["r_e"]] = -1; g0[r, i_["Epicpi"]] = -1
    g0[r, i_["pi"]] = -ppi * (1 - rc); g0[r, i_["picpi"]] = -ppi * rc
    g0[r, i_["x"]] = -px; r += 1
    # E14 oil: s = rho*s(-1) + eps_s
    g0[r, i_["s"]] = 1; g1[r, i_["s"]] = rho; psi[r, 0] = 1; r += 1
    # E15 exogenous cost-push: u_exo = rho_u*u_exo(-1) + eps_u
    g0[r, i_["u_exo"]] = 1; g1[r, i_["u_exo"]] = rho_u; psi[r, 1] = 1; r += 1
    # I1..I3 expectational identities: v_t = Ev_{t-1} + eta_v
    for k, (v, ev) in enumerate([("pi", "Epi"), ("picpi", "Epicpi"), ("x", "Ex")]):
        g0[r, i_[v]] = 1; g1[r, i_[ev]] = 1; pim[r, k] = 1; r += 1

    return g0, g1, psi, pim


def solve(p: Params):
    """Solve the model; return (G1, impact, eu)."""
    g0, g1, psi, pim = build_matrices(p)
    return gensys(g0, g1, psi, pim)


def irf(p: Params, horizon: int = 16, shock: float = 0.10, which: str = "oil"):
    """Impulse responses to a one-time innovation of size `shock`.

    `which` selects the shock: "oil" (eps_s) or "costpush" (eps_u). Returns a
    tidy DataFrame with columns period, variable, value (long format, matching
    the Dynare CSV layout). Only observable variables are returned.
    """
    G1, impact, eu = solve(p)
    col = SHOCKS[which]
    Y = np.zeros((horizon, N))
    Y[0] = (impact[:, col] * shock)
    for t in range(1, horizon):
        Y[t] = G1 @ Y[t - 1]

    rows = []
    for j, v in enumerate(OBS):
        for t in range(horizon):
            rows.append((t + 1, v, Y[t, j]))
    df = pd.DataFrame(rows, columns=["period", "variable", "value"])
    df.attrs["eu"] = eu
    return df
