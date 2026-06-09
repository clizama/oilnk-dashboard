"""Sims (2002) gensys solver for linear rational-expectations models.

Solves the canonical system

    g0 @ y_t = g1 @ y_{t-1} + psi @ z_t + pi @ eta_t

(constant dropped: the project works in log-deviations, steady state 0), where
z_t are exogenous innovations and eta_t are one-step-ahead expectational errors.

Returns G1, impact such that the equilibrium law of motion is

    y_t = G1 @ y_{t-1} + impact @ z_t .

Implementation mirrors Sims' gensys.m, using scipy's reordering QZ (ordqz) to
push explosive generalized eigenvalues to the lower-right block.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import ordqz

_TOL = 1e-9


def gensys(g0, g1, psi, pi, div: float = 1.0 + 1e-8):
    g0 = np.asarray(g0, dtype=complex)
    g1 = np.asarray(g1, dtype=complex)
    psi = np.asarray(psi, dtype=complex)
    pi = np.asarray(pi, dtype=complex)
    n = g0.shape[0]

    def stable(alpha, beta):
        # AR root mu solves g1 x = mu g0 x  =>  mu = beta/alpha. Stable: |mu|<div.
        return np.abs(beta) < div * np.abs(alpha)

    a, b, alpha, beta, Q, Z = ordqz(g0, g1, sort=stable, output="complex")
    # scipy: g0 = Q a Z^H , g1 = Q b Z^H. Premultiplying the system by q := Q^H
    # diagonalises it in the coordinates w_t = Z^H y_t:   a w_t = b w_{t-1} + ...
    q = Q.conj().T

    nunstab = int(np.sum(~stable(alpha, beta)))
    nstab = n - nunstab

    q1, q2 = q[:nstab, :], q[nstab:, :]

    # SVD of the explosive-block loadings on the expectational errors.
    ueta, deta, vetah = np.linalg.svd(q2 @ pi, full_matrices=False)
    md = int(np.sum(deta > _TOL))
    ueta, veta, deta = ueta[:, :md], vetah.conj().T[:, :md], deta[:md]

    ueta1, deta1, veta1h = np.linalg.svd(q1 @ pi, full_matrices=False)
    md1 = int(np.sum(deta1 > _TOL))
    ueta1, veta1, deta1 = ueta1[:, :md1], veta1h.conj().T[:, :md1], deta1[:md1]

    # existence (eu0) and uniqueness (eu1)
    eu = [0, 0]
    if md1 == 0:
        eu[0] = 1
    else:
        proj = veta1 - veta @ (veta.conj().T @ veta1)
        eu[0] = int(np.linalg.norm(proj) < _TOL * n)
    eu[1] = int(md == nunstab)

    # tmat encodes the stable-manifold restriction (Sims).
    if md > 0 and md1 > 0:
        inner = (
            ueta
            @ np.diag(1.0 / deta)
            @ veta.conj().T
            @ veta1
            @ np.diag(deta1)
            @ ueta1.conj().T
        )  # (nunstab x nstab)
    else:
        inner = np.zeros((nunstab, nstab), dtype=complex)
    tmat = np.hstack([np.eye(nstab), -inner.conj().T])  # (nstab x n)

    G0 = np.vstack([tmat @ a, np.hstack([np.zeros((nunstab, nstab)), np.eye(nunstab)])])
    G1m = np.vstack([tmat @ b, np.zeros((nunstab, n))])
    Psi = np.vstack([tmat @ q @ psi, np.zeros((nunstab, psi.shape[1]))])

    G0I = np.linalg.inv(G0)
    G1m = G0I @ G1m
    impact = G0I @ Psi

    # back to original coordinates  (y_t = Z w_t)
    G1_out = (Z @ G1m @ Z.conj().T).real
    impact_out = (Z @ impact).real
    return G1_out, impact_out, eu
