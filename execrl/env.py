"""Vectorised optimal-execution environment (Almgren-Chriss dynamics + an AR(1) alpha signal)."""

import numpy as np

X0 = 100_000.0  # shares to liquidate
S0 = 100.0  # arrival price
T = 20  # decision steps
SIGMA = 0.15  # per-step price volatility ($)
GAMMA = 2e-6  # permanent impact ($ per share sold)
ETA = 2e-5  # temporary impact ($ per share sold in a step)
PHI = 0.8  # signal persistence
DRIFT = 0.12  # $ price drift per unit of signal per step
IMPACT_P = 1.0  # temporary-impact exponent: step cost = temp_eta()*sold**(1+IMPACT_P). 1.0 = quadratic
#                     (Almgren-Chriss, closed-form Riccati optimum); 0.5 = square-root law (Almgren 2005, no closed form).


def temp_eta(p=None):
    """Temporary-impact coefficient, matched at the TWAP schedule so the naive total cost is identical across exponents;
    only the curvature (marginal cost vs trade size) changes. temp_eta(1.0) == ETA exactly.
    """
    p = IMPACT_P if p is None else p
    return ETA * X0**2 / (T**2 * (X0 / T) ** (1 + p))


def temp_cost(sold, p=None):
    """Per-step temporary-impact cost in price*share units."""
    p = IMPACT_P if p is None else p
    return temp_eta(p) * np.abs(sold) ** (1 + p)


class ExecutionEnv:
    """Sell X0 shares over T steps. Action = fraction of remaining inventory to sell now.

    Observation: [inventory left, time left, alpha signal]. Reward is the negative per-step shortfall in basis points of
    arrival notional, so the return is -(implementation shortfall). The alpha signal predicts the next price drift; a
    signal-blind policy cannot exploit it.
    """

    obs_dim = 3

    def __init__(self, n_envs, seed=0, use_signal=True):
        self.n = n_envs
        self.rng = np.random.default_rng(seed)
        self.use_signal = use_signal
        self.reset()

    def reset(self):
        self.t = 0
        self.inv = np.full(self.n, X0)
        self.price = np.full(self.n, S0)
        self.alpha = self.rng.standard_normal(self.n)
        return self._obs()

    def _obs(self):
        a = self.alpha if self.use_signal else np.zeros(self.n)
        return np.stack([self.inv / X0, np.full(self.n, 1 - self.t / T), a], axis=1).astype(np.float32)

    def step(self, frac):
        frac = np.clip(frac, 0.0, 1.0)
        if self.t == T - 1:
            frac = np.ones(self.n)
        sold = frac * self.inv
        inv_after = self.inv - sold
        old_price = self.price
        z = self.rng.standard_normal(self.n)
        self.price = old_price - GAMMA * sold + DRIFT * self.alpha + SIGMA * z
        # Summation-by-parts of the shortfall: sum_t r_t == -(X0*S0 - proceeds)/(X0*S0) in bps, but
        # each r_t depends only on step-t noise, which keeps policy-gradient variance low.
        reward = (inv_after * (self.price - old_price) - temp_cost(sold)) / (X0 * S0) * 1e4
        self.inv = inv_after
        self.alpha = PHI * self.alpha + np.sqrt(1 - PHI**2) * self.rng.standard_normal(self.n)
        self.t += 1
        done = self.t >= T
        return self._obs(), reward, done
