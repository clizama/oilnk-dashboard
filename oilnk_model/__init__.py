"""oilnk_model — master Blanchard-Gali model + RE solver for the dashboard."""
from .params import Params, PRESETS
from .model import irf, solve, build_matrices, VARS, OBS
from .optimal import optimal_irf, solve_optimal
from .welfare import welfare_table, loss_rule, loss_optimal

__all__ = [
    "Params", "PRESETS",
    "irf", "solve", "build_matrices", "VARS", "OBS",
    "optimal_irf", "solve_optimal",
    "welfare_table", "loss_rule", "loss_optimal",
]
