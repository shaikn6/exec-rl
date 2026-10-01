"""Schedule baselines: TWAP, sell-immediately, and the Almgren-Chriss risk-averse trajectory."""
import numpy as np
from .env import T, SIGMA, ETA


def twap_fracs(t):
    return 1.0 / (T - t)


def immediate_fracs(t):
    return 1.0


def ac_fracs(t, risk_aversion):
    """Fraction of *remaining* inventory to sell at step t so holdings follow the AC sinh trajectory."""
    kappa = np.sqrt(risk_aversion * SIGMA ** 2 / ETA)
    hold = lambda k: np.sinh(kappa * (T - k)) / np.sinh(kappa * T)
    return 1.0 - hold(t + 1) / hold(t)
