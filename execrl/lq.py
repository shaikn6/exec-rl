"""Exact risk-neutral optimum of the execution environment by backward dynamic programming (a scalar Riccati recursion).

With y = q - v the inventory kept after selling v shares, the expected one-step cost in dollars is
    E[c_t] = GAMMA * v * y - DRIFT * a * y + ETA * v**2,
and the value function is quadratic, V_t(q, a) = A_t q^2 + B_t q a + C_t a^2 + K_t, with V_{T-1} = ETA * q^2
(everything left is sold at the last step). Minimising over y gives the policy y = alpha_t q + beta_t a.
The optimum ignores the environment's constraint 0 <= v <= q; `simulate(..., clip=True)` applies it."""
import numpy as np

from . import env as E


def riccati(drift=None, eta=None, gamma=None, phi=None, T=None):
    """Return (alpha, beta, expected_cost_bps). alpha[t], beta[t] for t < T-1; at T-1 everything is sold."""
    drift = E.DRIFT if drift is None else drift
    eta = E.ETA if eta is None else eta
    gamma = E.GAMMA if gamma is None else gamma
    phi = E.PHI if phi is None else phi
    T = E.T if T is None else T
    A, B, C, K = eta, 0.0, 0.0, 0.0
    alpha, beta = np.zeros(T - 1), np.zeros(T - 1)
    for t in range(T - 2, -1, -1):
        k = A + eta - gamma                      # curvature in y; must be positive
        assert k > 0, "non-convex stage problem"
        lq, la = 2 * eta - gamma, drift - B * phi  # linear coefficient of y is -(lq q + la a)
        alpha[t], beta[t] = lq / (2 * k), la / (2 * k)
        K = K + C * (1 - phi ** 2)
        A, B, C = eta - lq ** 2 / (4 * k), -lq * la / (2 * k), C * phi ** 2 - la ** 2 / (4 * k)
    expected = (A * E.X0 ** 2 + C + K) / (E.X0 * E.S0) * 1e4  # a_0 ~ N(0, 1), so E[a_0] = 0 and E[a_0^2] = 1
    return alpha, beta, expected


def simulate(keep, n_paths=20000, seed=12345, clip=True):
    """Replay ExecutionEnv's exact random draws with a policy in shares: keep(t, q, a) -> inventory kept.

    With clip=True sales are limited to [0, q] as in the environment; with clip=False the policy may buy back
    or sell more than it holds. Returns per-path shortfall in bps."""
    rng = np.random.default_rng(seed)
    q, s = np.full(n_paths, E.X0), np.full(n_paths, E.S0)
    rng.standard_normal(n_paths)  # ExecutionEnv draws a signal in __init__ and again in reset()
    a = rng.standard_normal(n_paths)
    cost = np.zeros(n_paths)
    for t in range(E.T):
        y = np.zeros(n_paths) if t == E.T - 1 else keep(t, q, a)
        if clip:
            y = np.clip(y, 0.0, q)
        v = q - y
        z = rng.standard_normal(n_paths)
        s_new = s - E.GAMMA * v + E.DRIFT * a + E.SIGMA * z
        cost -= (y * (s_new - s) - E.ETA * v ** 2) / (E.X0 * E.S0) * 1e4
        q, s = y, s_new
        a = E.PHI * a + np.sqrt(1 - E.PHI ** 2) * rng.standard_normal(n_paths)
    return cost


def lq_policy(alpha, beta):
    return lambda t, q, a: alpha[t] * q + beta[t] * a


def as_frac(alpha, beta):
    """The clipped optimum as an environment policy (fraction of remaining inventory)."""
    def pol(t, obs):
        if t >= len(alpha):
            return np.ones(len(obs))
        q = obs[:, 0].astype(float) * E.X0
        y = np.clip(alpha[t] * q + beta[t] * obs[:, 2], 0.0, q)
        return np.where(q > 0, 1 - y / np.maximum(q, 1e-12), 1.0)
    return pol
