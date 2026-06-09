"""Validation: the Python solver must reproduce the Dynare IRFs of the decks.

The Dynare `.mod` files are the source of truth. These tests certify that the
master model + gensys solver give the same impulse responses (to machine
precision) at the decks' calibrations, and that the nested special cases behave
as the theory predicts.
"""
import os

import numpy as np
import pandas as pd
import pytest

from oilnk_model import Params, irf, solve
from oilnk_model.optimal import optimal_irf, solve_optimal
from oilnk_model.welfare import welfare_table

HERE = os.path.dirname(__file__)
CSV_DIR = os.path.join(HERE, "dynare_csv")

BASE = dict(beta=0.99, sigma=1, phi=1, theta=0.75, alpha=0.05, chi=0.06,
            phi_pi=1.5, phi_x=0.125, rho_s=0.9, epsilon=6)
TOL = 1e-9


def _csv(path):
    # The Dynare CSVs (source of truth, exported from the decks' .mod files)
    # are vendored under tests/dynare_csv/; look them up by basename.
    return pd.read_csv(os.path.join(CSV_DIR, os.path.basename(path)))


@pytest.mark.parametrize("gamma", [0.0, 0.8])
def test_matches_deck6_gamma(gamma):
    dyn = _csv("deck6-bg-structural/model/output/irfs_gamma.csv")
    df = irf(Params(**BASE, gamma=gamma, rule_cpi=0), horizon=16, shock=0.10)
    for v in ["pi", "picpi", "x", "y", "i", "r_e"]:
        py = df[df.variable == v].sort_values("period").value.values
        dd = dyn[(dyn.gamma == gamma) & (dyn.variable == v)].sort_values("period").value.values
        assert np.max(np.abs(py[: len(dd)] - dd)) < TOL, v


@pytest.mark.parametrize("rule_cpi,regime", [(0, "core"), (1, "headline")])
def test_matches_deck7_rule(rule_cpi, regime):
    dyn = _csv("deck7-natal2012/model/output/irfs_policy.csv")
    df = irf(Params(**BASE, gamma=0.8, rule_cpi=rule_cpi), horizon=16, shock=0.10)
    for v in ["pi", "picpi", "x", "i"]:
        py = df[df.variable == v].sort_values("period").value.values
        dd = dyn[(dyn.regime == regime) & (dyn.variable == v)].sort_values("period").value.values
        assert np.max(np.abs(py[: len(dd)] - dd)) < TOL, v


def test_existence_uniqueness():
    _, _, eu = solve(Params(**BASE, gamma=0.8))
    assert eu == [1, 1]


def test_divine_coincidence_gamma0():
    # gamma = 0: no cost-push -> core inflation and the welfare gap stay at 0.
    df = irf(Params(**BASE, gamma=0.0), horizon=16, shock=0.10)
    for v in ["pi", "x"]:
        assert np.max(np.abs(df[df.variable == v].value.values)) < 1e-10, v


def test_optimal_matches_deck7():
    dyn = _csv("deck7-natal2012/model/output/irfs_policy.csv")
    df = optimal_irf(Params(**BASE, gamma=0.8), horizon=16, shock=0.10)
    for v in ["pi", "x", "i", "picpi"]:
        py = df[df.variable == v].sort_values("period").value.values
        dd = dyn[(dyn.regime == "optimo") & (dyn.variable == v)].sort_values("period").value.values
        assert np.max(np.abs(py[: len(dd)] - dd)) < TOL, v


def test_optimal_exists_unique():
    _, _, eu = solve_optimal(Params(**BASE, gamma=0.8))
    assert eu == [1, 1]


def test_welfare_ratios_deck7():
    # deck7 Dynare: optimal=1, core~1.84, headline~3.21
    rel = welfare_table(Params(**BASE, gamma=0.8))["rel"]
    assert abs(rel["optimo"] - 1.0) < 1e-9
    assert abs(rel["core"] - 1.84) < 0.02
    assert abs(rel["headline"] - 3.21) < 0.02


def test_optimal_overshoot_commitment():
    # under commitment core inflation overshoots and reverts below zero
    df = optimal_irf(Params(**BASE, gamma=0.8), horizon=16, shock=0.10)
    pi = df[df.variable == "pi"].sort_values("period").value.values
    assert pi[0] > 0 and np.min(pi) < 0


def test_no_oil_flat_potential():
    # alpha = chi = 0: classic NK, oil price has no effect on efficient output.
    df = irf(Params(**{**BASE, "alpha": 0.0, "chi": 0.0, "gamma": 0.0}), horizon=16, shock=0.10)
    assert np.max(np.abs(df[df.variable == "ye"].value.values)) < 1e-10


@pytest.mark.parametrize("gamma", [0.0, 0.8])
def test_costpush_matches_deck6(gamma):
    dyn = _csv("deck6-bg-structural/model/output/irfs_costpush.csv")
    df = irf(Params(**BASE, gamma=gamma, rule_cpi=0, rho_u=0.5),
             horizon=16, shock=0.01, which="costpush")
    for v in ["pi", "x", "ye", "y", "picpi", "i", "u_exo"]:
        py = df[df.variable == v].sort_values("period").value.values
        dd = dyn[(dyn.gamma == gamma) & (dyn.variable == v)].sort_values("period").value.values
        assert np.max(np.abs(py[: len(dd)] - dd)) < TOL, v


def test_costpush_flat_potential_and_tradeoff():
    # a pure cost-push: potential is flat, but a trade-off appears even without
    # real-wage rigidity (gamma = 0) — unlike the oil shock.
    df = irf(Params(**BASE, gamma=0.0), horizon=16, shock=0.01, which="costpush")
    assert np.max(np.abs(df[df.variable == "ye"].value.values)) < 1e-12
    pi = df[df.variable == "pi"].sort_values("period").value.values
    x = df[df.variable == "x"].sort_values("period").value.values
    assert pi[0] > 0 and x[0] < 0


def test_optimal_costpush_flat_potential():
    df = optimal_irf(Params(**BASE, gamma=0.8), horizon=16, shock=0.01, which="costpush")
    assert np.max(np.abs(df[df.variable == "ye"].value.values)) < 1e-12


def test_optimal_costpush_textbook_rule():
    # with no oil (alpha = chi = 0) the optimal commitment rule is
    # pi_t = -(theta_x/kappa)(x_t - x_{t-1}) = -(1/epsilon)(x_t - x_{t-1}).
    p = Params(**{**BASE, "alpha": 0.0, "chi": 0.0, "gamma": 0.0})
    df = optimal_irf(p, horizon=16, shock=0.01, which="costpush")
    pi = df[df.variable == "pi"].sort_values("period").value.values
    x = df[df.variable == "x"].sort_values("period").value.values
    x_lag = np.concatenate([[0.0], x[:-1]])
    assert np.max(np.abs(pi + (1.0 / p.epsilon) * (x - x_lag))) < 1e-10
