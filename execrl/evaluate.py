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
