"""Score policies on held-out paths (fixed seeds never used in training)."""
import numpy as np
import torch
from .env import ExecutionEnv, T
from .ppo import act_frac


def rollout(policy, n_paths=20000, seed=12345, use_signal=True):
    env = ExecutionEnv(n_paths, seed=seed, use_signal=use_signal)
    obs, cost = env.reset(), np.zeros(n_paths)
    for t in range(T):
        obs, r, _ = env.step(policy(t, obs))
        cost -= r
    return cost


def summarize(cost):
    return dict(mean=float(cost.mean()), std=float(cost.std()),
                se=float(cost.std() / np.sqrt(len(cost))), p95=float(np.percentile(cost, 95)))


def fixed(fn):
    return lambda t, obs: np.full(len(obs), fn(t))


def learned(net):
    def pol(t, obs):
        with torch.no_grad():
            mu, _ = net(torch.as_tensor(obs))
        return act_frac(mu, torch.as_tensor(obs)).numpy()
    return pol


def paired_diff(a, b):
    d = a - b
    return float(d.mean()), float(d.std() / np.sqrt(len(d)))


if __name__ == "__main__":
    # Fast, deterministic reproduce of the baseline rows (no training, no file writes): a few seconds on a CPU.
    from .baselines import twap_fracs, immediate_fracs, ac_fracs

    for name, fn in [("Sell immediately", immediate_fracs), ("TWAP", twap_fracs)]:
        print(f"{name:34s} {summarize(rollout(fixed(fn)))['mean']:8.3f} bps")
    lam, cost = min(((l, rollout(fixed(lambda t, l=l: ac_fracs(t, l)))) for l in [1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2]),
                    key=lambda x: x[1].mean())
    print(f"{'Almgren-Chriss (lambda=' + format(lam, 'g') + ')':34s} {cost.mean():8.3f} bps")
