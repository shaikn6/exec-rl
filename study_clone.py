"""Is PPO's gap to the optimum a limit of the network or of training? Two experiments per signal strength.

1. Cloning: the PPO network (same architecture and action parametrisation) is fitted by regression to the constrained
   optimum's decisions, first on states visited by the optimum and then, for three rounds, on states visited by the
   network itself and relabelled by the optimum. It is scored like every other policy. Its gap to the optimum is what
   the network cannot represent.
2. Warm start: PPO is continued from the cloned network (critic fitted first, actor frozen) with the usual exploration
   noise and with a small one. Its gap shows whether PPO's training keeps a near-optimal policy.

Same held-out paths as study.py and study_extra.py. Writes study_clone.json."""
import json
import time

import numpy as np
import torch

import execrl.env as E
from execrl import dp, lq
from execrl.evaluate import learned, rollout
from execrl.ppo import REWARD_SCALE, ActorCritic, act_frac, train
from study_extra import CKPT_EVERY, EVAL_SEED, SEEDS, clip_rate, configure, paired, stats

torch.set_num_threads(4)
DRIFTS = (0.12, 0.24)
N_FIT, ROUNDS, EPOCHS, BATCH, LR = 20000, 3, 30, 4096, 1e-3
CRITIC_STEPS, WARM_ITERS = 300, 300
SMALL_LOG_STD = -3.0


def optimum_policy(sol):
    keep = dp.policy(sol)
    def pol(t, obs):
        if t >= E.T - 1:
            return np.ones(len(obs))
        q = np.maximum(obs[:, 0].astype(float) * E.X0, 1e-9)
        return 1 - keep(t, q, obs[:, 2].astype(float)) / q
    return pol


def visited(policy, expert, seed):
    """States visited by `policy` on fresh paths, each labelled with the optimum's sell fraction (last step excluded:
    the environment sells everything there whatever the policy says)."""
    env = E.ExecutionEnv(N_FIT, seed=seed)
    obs, O, Y = env.reset(), [], []
    for t in range(E.T - 1):
        O.append(obs); Y.append(expert(t, obs).astype(np.float32))
        obs, _, _ = env.step(policy(t, obs))
    return torch.as_tensor(np.concatenate(O)), torch.as_tensor(np.concatenate(Y))


def fit_actor(net, O, Y, seed):
    """Least squares on shares sold as a fraction of the order: (fraction error) x (inventory left)."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(list(net.body.parameters()) + list(net.mu.parameters()), lr=LR)
    for _ in range(EPOCHS):
        for idx in torch.randperm(len(O), generator=g).split(BATCH):
            mu, _ = net(O[idx])
            loss = (((act_frac(mu, O[idx]) - Y[idx]) * O[idx, 0]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


def fit_critic(net, seed):
    """Fit the value head to the cloned policy's reward-to-go, actor frozen, so that PPO does not start with a random critic."""
    opt = torch.optim.Adam(net.v.parameters(), lr=LR)
    pol = learned(net)
    for i in range(CRITIC_STEPS):
        env = E.ExecutionEnv(2048, seed=10_000 * (seed + 1) + i)
        obs, O, R = env.reset(), [], []
        for t in range(E.T):
            O.append(torch.as_tensor(obs))
            obs, r, _ = env.step(pol(t, obs))
            R.append(torch.as_tensor(r, dtype=torch.float32) / REWARD_SCALE)
        togo = torch.flip(torch.cumsum(torch.flip(torch.stack(R), [0]), 0), [0])
        _, v = net(torch.stack(O).flatten(0, 1))
        loss = ((v - togo.flatten()) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()


def clone(expert, seed):
    torch.manual_seed(seed)
    net = ActorCritic()
    O, Y = visited(expert, expert, 500 + seed)
    losses = [fit_actor(net, O, Y, seed)]
    for r in range(ROUNDS):
        O2, Y2 = visited(learned(net), expert, 600 + 10 * seed + r)
        O, Y = torch.cat([O, O2]), torch.cat([Y, Y2])
        losses.append(fit_actor(net, O, Y, seed + 100 * (r + 1)))
    return net, losses


def warm(net0, seed, log_std):
    curve = []
    def hook(it, net):
        if (it + 1) % CKPT_EVERY == 0:
            curve.append([it + 1, float(rollout(learned(net), n_paths=5000, seed=EVAL_SEED + 1).mean())])
    net, _ = train(use_signal=True, seed=seed, iters=WARM_ITERS, init=net0.state_dict(), init_log_std=log_std, on_iter=hook)
    return net, curve


def score(net, opt):
    c = rollout(learned(net))
    return {**stats(c), "vs_dp": paired(c, opt), "zero_sell_rate": clip_rate(learned(net))}, c


if __name__ == "__main__":
    t0 = time.time()
    out = {"drifts": DRIFTS, "seeds": SEEDS, "n_fit_paths": N_FIT, "rounds": ROUNDS, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
           "critic_steps": CRITIC_STEPS, "warm_iters": WARM_ITERS, "small_log_std": SMALL_LOG_STD, "ckpt_every": CKPT_EVERY,
           "settings": {}}
    for d in DRIFTS:
        configure(d)
        sol = dp.solve()
        expert = optimum_policy(sol)
        opt = lq.simulate(dp.policy(sol))
        check = rollout(expert)
        res = {"dp": {**stats(opt), "zero_sell_rate": clip_rate(expert)}, "dp_as_env_policy_vs_dp": paired(check, opt), "seeds": []}
        for s in SEEDS:
            net, losses = clone(expert, s)
            cl, c_cl = score(net, opt)
            before = float(rollout(learned(net), n_paths=5000, seed=EVAL_SEED + 1).mean())
            fit_critic(net, s)
            row = {"seed": s, "clone": {**cl, "fit_loss": losses}, "clone_heldout_5000": before}
            for name, ls in (("warm", -1.0), ("warm_small_noise", SMALL_LOG_STD)):
                wn, curve = warm(net, s, ls)
                w, c_w = score(wn, opt)
                row[name] = {**w, "vs_clone": paired(c_w, c_cl), "heldout_curve": curve, "final_log_std": float(wn.log_std.item())}
            res["seeds"].append(row)
            print(f"drift {d} seed {s}: clone {cl['vs_dp']['mean']:+.2f}  warm {row['warm']['vs_dp']['mean']:+.2f}  "
                  f"warm small noise {row['warm_small_noise']['vs_dp']['mean']:+.2f}  ({time.time() - t0:.0f}s)", flush=True)
        out["settings"][str(d)] = res
        json.dump(out, open("study_clone.json", "w"), indent=1)
    configure(0.12)
    out["wall_s"] = round(time.time() - t0, 1)
    json.dump(out, open("study_clone.json", "w"), indent=1)
