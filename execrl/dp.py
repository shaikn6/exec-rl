"""Constrained risk-neutral optimum (sales limited to 0 <= v <= q, as in the environment) by grid dynamic programming.

Same stage cost as execrl.lq, but the value function is tabulated on an inventory x signal grid, the next-step
signal expectation uses Gauss-Hermite quadrature, and the kept inventory y is searched over [0, q]."""
import numpy as np

from . import env as E


def solve(nq=401, na=81, a_max=4.0, n_gh=21):
    q = np.linspace(0.0, E.X0, nq)
    a = np.linspace(-a_max, a_max, na)
    x, w = np.polynomial.hermite_e.hermegauss(n_gh)
    w = w / w.sum()
    s = np.sqrt(1 - E.PHI ** 2)
    V = np.repeat((E.ETA * q ** 2)[:, None], na, 1)  # V_{T-1}: everything left is sold
    Ws = [None] * (E.T - 1)
    for t in range(E.T - 2, -1, -1):
        W = np.zeros_like(V)                    # W[j, k] = E[V_{t+1}(q_j, phi a_k + s eps)]
        for xi, wi in zip(x, w):
            W += wi * np.stack([np.interp(E.PHI * a + s * xi, a, V[j]) for j in range(nq)])
        Ws[t] = W
        qi, yj = q[:, None], q[None, :]
        v = qi - yj
        base = E.GAMMA * v * yj + E.ETA * v ** 2  # (nq, nq), the a-free part
        total = base[:, :, None] - E.DRIFT * a[None, None, :] * yj[:, :, None] + W[None, :, :]
        total = np.where((yj <= qi)[:, :, None], total, np.inf)
        V = total.min(axis=1)
    value = float(sum(wi * np.interp(xi, a, V[-1]) for xi, wi in zip(x, w))) / (E.X0 * E.S0) * 1e4
    return {"q": q, "a": a, "W": Ws, "value_bps": value}


def _interp2(grid_q, grid_a, W, yq, ya):
    """Bilinear interpolation of W on (grid_q, grid_a) at points (yq, ya); signal is clamped to the grid."""
    ya = np.clip(ya, grid_a[0], grid_a[-1])
    iq = np.clip(np.searchsorted(grid_q, yq) - 1, 0, len(grid_q) - 2)
    ia = np.clip(np.searchsorted(grid_a, ya) - 1, 0, len(grid_a) - 2)
    fq = (yq - grid_q[iq]) / (grid_q[iq + 1] - grid_q[iq])
    fa = (ya - grid_a[ia]) / (grid_a[ia + 1] - grid_a[ia])
    return ((1 - fq) * (1 - fa) * W[iq, ia] + fq * (1 - fa) * W[iq + 1, ia]
            + (1 - fq) * fa * W[iq, ia + 1] + fq * fa * W[iq + 1, ia + 1])


def policy(sol, n_frac=201):
    """keep(t, q, a) for execrl.lq.simulate: the kept inventory minimising stage cost plus continuation value."""
    fr = np.linspace(0.0, 1.0, n_frac)
    def keep(t, q, a):
        y = q[:, None] * fr[None, :]
        v = q[:, None] - y
        c = E.GAMMA * v * y + E.ETA * v ** 2 - E.DRIFT * a[:, None] * y
        c = c + _interp2(sol["q"], sol["a"], sol["W"][t], y, np.repeat(a[:, None], n_frac, 1))
        return y[np.arange(len(q)), c.argmin(1)]
    return keep
