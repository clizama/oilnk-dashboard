"""Optimal monetary policy (Ramsey, commitment) for the master model.

The planner minimises  L = 1/2 E_0 sum beta^t (pi_t^2 + theta_x x_t^2)  subject to
the economy. Only core inflation pi and the welfare gap x carry weight (Natal):
headline/CPI inflation reflects an efficient relative-price move and is not a
distortion. The binding constraints for the (pi, x) choice are the NKPC and the
real-wage law of motion — the latter because the wage is an endogenous state that
feeds tomorrow's cost-push. Everything else (IS, definitions) is recursive and
only backs out the instrument i.

Reduced cost-push form (substituting the static identities):
    mrs_t  = (sigma+phi) x_t + Ms s_t,    Ms = -(sigma+phi) psi_e + phi*zeta - sigma*chi
    NKPC:  pi_t = beta E_t pi_{t+1} + ktil x_t + lam*gamma w_{t-1} - lam*gamma Ms s_t
    wage:  w_t  = gamma w_{t-1} + (1-gamma)[(sigma+phi) x_t + Ms s_t]
with ktil = kappa(1-gamma).

First-order conditions (phi1 on NKPC, phi2 on the wage law):
    (F1)  phi1_t - phi1_{t-1} + pi_t = 0
    (F2)  theta_x x_t - ktil phi1_t - (1-gamma)(sigma+phi) phi2_t = 0
    (F3)  phi2_t - beta*gamma (lam E_t phi1_{t+1} + E_t phi2_{t+1}) = 0
At gamma=0 these collapse to the textbook rule pi_t = -(theta_x/kappa)(x_t - x_{t-1}).

Validated against deck7's Dynare Ramsey IRFs (regime 'optimo').
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .gensys import gensys
from .params import Params

# reduced state for the Ramsey system (u_exo = exogenous cost-push forcing)
RVARS = ["pi", "x", "w", "s", "u_exo", "phi1", "phi2", "Epi", "Ephi1", "Ephi2"]
RIDX = {v: k for k, v in enumerate(RVARS)}
RN = len(RVARS)
OBS = ["pi", "picpi", "x", "i", "r_e", "y", "ye", "n", "c", "w", "mrs", "wedge", "u", "s"]
SHOCKS = {"oil": 0, "costpush": 1}


def _coeffs(p: Params):
    sig, phi, gam, chi = p.sigma, p.phi, p.gamma, p.chi
    Ms = -(sig + phi) * p.psi_e + phi * p.zeta - sig * chi
    ktil = p.kappa * (1 - gam)
    return sig, phi, gam, chi, Ms, ktil


def build_optimal_system(p: Params):
    g0 = np.zeros((RN, RN)); g1 = np.zeros((RN, RN))
    psi = np.zeros((RN, 2)); pim = np.zeros((RN, 3))   # shocks: 0 = oil, 1 = cost-push
    i_ = RIDX
    sig, phi, gam, chi, Ms, ktil = _coeffs(p)
    lam, beta, rho, psi_e, rho_u = p.lam, p.beta, p.rho_s, p.psi_e, p.rho_u

    r = 0
    # NKPC: pi - beta*Epi - ktil*x - lam*gam*w(-1) + lam*gam*Ms*s - u_exo = 0
    g0[r, i_["pi"]] = 1; g0[r, i_["Epi"]] = -beta; g0[r, i_["x"]] = -ktil
    g0[r, i_["s"]] = lam * gam * Ms; g1[r, i_["w"]] = lam * gam
    g0[r, i_["u_exo"]] = -1; r += 1
    # wage law: w - gam*w(-1) - (1-gam)(sigma+phi)x - (1-gam)Ms*s = 0
    g0[r, i_["w"]] = 1; g0[r, i_["x"]] = -(1 - gam) * (sig + phi)
    g0[r, i_["s"]] = -(1 - gam) * Ms; g1[r, i_["w"]] = gam; r += 1
    # oil: s - rho*s(-1) = eps_s
    g0[r, i_["s"]] = 1; g1[r, i_["s"]] = rho; psi[r, 0] = 1; r += 1
    # exogenous cost-push: u_exo - rho_u*u_exo(-1) = eps_u
    g0[r, i_["u_exo"]] = 1; g1[r, i_["u_exo"]] = rho_u; psi[r, 1] = 1; r += 1
    # F1: phi1 + pi = phi1(-1)
    g0[r, i_["phi1"]] = 1; g0[r, i_["pi"]] = 1; g1[r, i_["phi1"]] = 1; r += 1
    # F2: theta_x*x - ktil*phi1 - (1-gam)(sigma+phi)*phi2 = 0
    g0[r, i_["x"]] = p.theta_x; g0[r, i_["phi1"]] = -ktil
    g0[r, i_["phi2"]] = -(1 - gam) * (sig + phi); r += 1
    # F3: phi2 - beta*gam*lam*Ephi1 - beta*gam*Ephi2 = 0
    g0[r, i_["phi2"]] = 1; g0[r, i_["Ephi1"]] = -beta * gam * lam
    g0[r, i_["Ephi2"]] = -beta * gam; r += 1
    # expectation identities  v_t = Ev_{t-1} + eta_v
    for k, (v, ev) in enumerate([("pi", "Epi"), ("phi1", "Ephi1"), ("phi2", "Ephi2")]):
        g0[r, i_[v]] = 1; g1[r, i_[ev]] = 1; pim[r, k] = 1; r += 1
    return g0, g1, psi, pim


def solve_optimal(p: Params):
    g0, g1, psi, pim = build_optimal_system(p)
    return gensys(g0, g1, psi, pim)


def optimal_irf(p: Params, horizon: int = 16, shock: float = 0.10, which: str = "oil"):
    """Ramsey (commitment) impulse responses; tidy DataFrame like model.irf.

    `which` selects the shock: "oil" (eps_s) or "costpush" (eps_u).
    """
    G1, impact, eu = solve_optimal(p)
    col = SHOCKS[which]
    X = np.zeros((horizon, RN))
    X[0] = impact[:, col] * shock
    for t in range(1, horizon):
        X[t] = G1 @ X[t - 1]

    sig, phi, gam, chi, Ms, ktil = _coeffs(p)
    psi_e, lam, rho = p.psi_e, p.lam, p.rho_s
    ipi, ix, iw, is_ = RIDX["pi"], RIDX["x"], RIDX["w"], RIDX["s"]

    rows = []
    s_prev = 0.0
    for t in range(horizon):
        st = X[t]
        pi, x, w, s = st[ipi], st[ix], st[iw], st[is_]
        ye = -psi_e * s
        y = x + ye
        n = y + p.zeta * s
        c = y - chi * s
        mrs = (sig + phi) * x + Ms * s
        wedge = w - mrs
        u = lam * wedge
        r_e = sig * (psi_e + chi) * (1 - rho) * s
        picpi = pi + chi * (s - s_prev)
        # instrument from the IS curve, using one-step-ahead expectations
        Est = G1 @ st
        Epi1, Ex1, Es1 = Est[ipi], Est[ix], Est[is_]
        Epicpi1 = Epi1 + chi * (Es1 - s)
        i = r_e + Epicpi1 + sig * (Ex1 - x)
        vals = dict(pi=pi, picpi=picpi, x=x, i=i, r_e=r_e, y=y, ye=ye, n=n,
                    c=c, w=w, mrs=mrs, wedge=wedge, u=u, s=s)
        for v in OBS:
            rows.append((t + 1, v, vals[v]))
        s_prev = s

    df = pd.DataFrame(rows, columns=["period", "variable", "value"])
    df.attrs["eu"] = eu
    return df
