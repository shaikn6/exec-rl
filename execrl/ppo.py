"""Minimal PPO (clipped objective, GAE) for the execution environment, written from scratch in PyTorch."""
import torch
import torch.nn as nn
from .env import ExecutionEnv, T

REWARD_SCALE = 10.0  # rewards are trained in units of 10 bps; mean_cost multiplies it back


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=ExecutionEnv.obs_dim, hidden=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.mu = nn.Linear(hidden, 1)
        nn.init.zeros_(self.mu.weight); nn.init.zeros_(self.mu.bias)
        self.v = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((1,), -1.0))

    def forward(self, obs):
        h = self.body(obs)
        return self.mu(h).squeeze(-1), self.v(h).squeeze(-1)

    def dist(self, obs):
        mu, v = self(obs)
        return torch.distributions.Normal(mu, self.log_std.exp()), v


def act_frac(raw, obs):
    """Policy output is a logit offset from the TWAP slice, so an untrained policy already equals TWAP."""
    steps_left = (obs[:, 1] * T).clamp(min=1.0)
    twap = (1.0 / steps_left).clamp(1e-4, 1 - 1e-4)
    return torch.sigmoid(torch.logit(twap) + raw)


def train(use_signal=True, iters=300, n_envs=2048, seed=0, lr=1e-3, epochs=4, clip=0.2, gae_lambda=0.95, log=None):
    torch.manual_seed(seed)
    env = ExecutionEnv(n_envs, seed=seed, use_signal=use_signal)
    net = ActorCritic()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    history = []
    for it in range(iters):
        obs = torch.as_tensor(env.reset())
        O, A, LP, R, V = [], [], [], [], []
        for _ in range(T):
            with torch.no_grad():
                d, v = net.dist(obs)
                raw = d.sample()
            nxt, r, _ = env.step(act_frac(raw, obs).numpy())
            O.append(obs); A.append(raw); LP.append(d.log_prob(raw)); V.append(v)
            R.append(torch.as_tensor(r, dtype=torch.float32) / REWARD_SCALE)
            obs = torch.as_tensor(nxt)
        V.append(torch.zeros(n_envs))
        adv, last = [None] * T, torch.zeros(n_envs)
        for t in reversed(range(T)):
            delta = R[t] + V[t + 1] - V[t]
            last = delta + gae_lambda * last
            adv[t] = last
        adv = torch.stack(adv)
        ret = adv + torch.stack(V[:T])
        O, A, LP = torch.stack(O).flatten(0, 1), torch.stack(A).flatten(), torch.stack(LP).flatten()
        adv_f, ret_f = adv.flatten(), ret.flatten()
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
        for _ in range(epochs):
            d, v = net.dist(O)
            ratio = (d.log_prob(A) - LP).exp()
            pg = -torch.min(ratio * adv_f, ratio.clamp(1 - clip, 1 + clip) * adv_f).mean()
            loss = pg + 0.5 * ((v - ret_f) ** 2).mean() - 1e-3 * d.entropy().mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
        mean_cost = -torch.stack(R).sum(0).mean().item() * REWARD_SCALE
        history.append(mean_cost)
        if log and it % 25 == 0:
            log(f"iter {it:3d}  mean shortfall {mean_cost:7.2f} bps")
    return net, history
