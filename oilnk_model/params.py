"""Structural parameters of the master Blanchard-Gali model.

A single parameter set nests every model in the project. Special cases:
    alpha = 0, chi = 0   -> classic 3-equation NK, no oil          (Deck 3)
    chi = 0,   alpha > 0 -> oil only in production, core = headline (Deck 5)
    gamma = 0            -> no real-wage rigidity => divine coincidence (Decks 4, 6)
    theta -> 0           -> flexible prices, money neutral

Derived coefficients (zeta, lambda, kappa, psi_e, theta_x) come from the
structural primitives exactly as in the Dynare .mod files of decks 6-7.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Params:
    # --- structural primitives -------------------------------------------
    beta: float = 0.99      # discount factor
    sigma: float = 1.0      # CRRA / inverse intertemporal elasticity
    phi: float = 1.0        # inverse Frisch elasticity
    theta: float = 0.75     # Calvo price stickiness
    alpha: float = 0.05     # oil share in gross output (production)
    chi: float = 0.06       # energy share in the CPI (consumption)
    gamma: float = 0.8      # real-wage rigidity (0 = flexible wage)
    rho_s: float = 0.9      # persistence of the (real) oil price
    rho_u: float = 0.5      # persistence of the exogenous cost-push (markup) shock
    epsilon: float = 6.0    # elasticity of substitution across varieties

    # --- policy ----------------------------------------------------------
    phi_pi: float = 1.5     # Taylor response to inflation
    phi_x: float = 0.125    # Taylor response to the welfare gap
    rule_cpi: float = 0.0   # 0 = Taylor on core, 1 = Taylor on headline (CPI)

    # ---- derived coefficients ------------------------------------------
    @property
    def zeta(self) -> float:
        return self.alpha / (1.0 - self.alpha)

    @property
    def lam(self) -> float:
        """Calvo slope lambda = (1-theta)(1-beta*theta)/theta."""
        return (1.0 - self.theta) * (1.0 - self.beta * self.theta) / self.theta

    @property
    def kappa(self) -> float:
        return self.lam * (self.sigma + self.phi)

    @property
    def psi_e(self) -> float:
        """Fall of efficient output per unit of s."""
        return (self.zeta * (1.0 + self.phi) + self.chi * (1.0 - self.sigma)) / (
            self.sigma + self.phi
        )

    @property
    def theta_x(self) -> float:
        """Welfare weight on the output gap: kappa / epsilon."""
        return self.kappa / self.epsilon

    def update(self, **kwargs) -> "Params":
        return replace(self, **kwargs)


# Presets: each one turns on one more feature of the model, for the guided tour.
PRESETS = {
    "NK estándar (sin petróleo)": dict(alpha=0.0, chi=0.0, gamma=0.0),
    "Petróleo solo en producción": dict(alpha=0.05, chi=0.0, gamma=0.0),
    "Petróleo en producción y consumo": dict(alpha=0.05, chi=0.06, gamma=0.0),
    "Con rigidez de salario real (trade-off)": dict(alpha=0.05, chi=0.06, gamma=0.8),
    "Shock casi permanente": dict(alpha=0.05, chi=0.06, gamma=0.8, rho_s=0.98),
    "Precios casi flexibles": dict(alpha=0.05, chi=0.06, gamma=0.8, theta=0.05),
}
