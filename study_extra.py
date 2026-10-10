"""Follow-up experiments for the paper: risk, what PPO learned, signal persistence, a richer rule, training curves,
and wider exploration. Same protocol as study.py: rules tuned on training paths (seed 999), every PPO seed reported,
all policies scored on the same held-out paths (seed 12345). Writes study_extra.json."""
import json
import os
import time

import numpy as np
import torch

import execrl.baselines as B
import execrl.env as E
from execrl import dp, lq
from execrl.baselines import ac_fracs, twap_fracs
from execrl.evaluate import fixed, learned, rollout
from execrl.ppo import train
from study import AC_RA, KS, linear_rule

torch.set_num_threads(4)
SEEDS = (0, 1, 2, 3, 4)
SEEDS_SMALL = (0, 1, 2)
N_EVAL, EVAL_SEED, TUNE_SEED = 20000, 12345, 999
BASE_ETA, BASE_PHI = E.ETA, E.PHI
FRONTIER_RA = (1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
SIG_GRID = np.linspace(-2.5, 2.5, 41)
GRID_T = (0, 5, 10, 15)
CKPT_EVERY = 25


def configure(drift, eta=None, phi=None):
    E.DRIFT, E.PHI = drift, BASE_PHI if phi is None else phi
    E.ETA = B.ETA = BASE_ETA if eta is None else eta


def stats(c):
    return {"mean": float(c.mean()), "std": float(c.std()), "p95": float(np.percentile(c, 95)),
            "p99": float(np.percentile(c, 99)), "se": float(c.std() / np.sqrt(len(c)))}


def paired(a, b):
    d = a - b
    return {"mean": float(d.mean()), "se": float(d.std() / np.sqrt(len(d)))}


def tune(make, grid):
    return min(grid, key=lambda p: rollout(make(p), n_paths=5000, seed=TUNE_SEED).mean())


def rule_time(params):
    """Linear rule whose signal coefficient changes linearly over the horizon: k(t) = k0 + k1 * t / T."""
    k0, k1 = params
    def pol(t, obs):
        return np.clip(twap_fracs(t) * (1 - (k0 + k1 * t / E.T) * obs[:, 2]), 0, 1)
    return pol


def rule_exp(params):
    """Log-linear rule on the PPO action scale: logit(frac) = logit(TWAP slice) - k * signal + c * (inventory lead).

    Inventory lead is held inventory minus the TWAP schedule's, so c > 0 catches up after selling less."""
    k, c = params
    def pol(t, obs):
        tw = np.clip(twap_fracs(t), 1e-4, 1 - 1e-4)
        lead = obs[:, 0] - (1 - t / E.T)
        return 1 / (1 + np.exp(-(np.log(tw / (1 - tw)) - k * obs[:, 2] + c * lead)))
    return pol


TIME_GRID = [(k0, k1) for k0 in np.linspace(0, 4, 21) for k1 in np.linspace(-6, 3, 19)]
EXP_GRID = [(k, c) for k in np.linspace(0, 6, 25) for c in (0.0, 2.0, 5.0, 10.0, 20.0)]


def policy_surface(pol):
    """Sell fraction against signal at times GRID_T, with inventory on the TWAP schedule."""
    out = {}
    for t in GRID_T:
        obs = np.stack([np.full(len(SIG_GRID), 1 - t / E.T), np.full(len(SIG_GRID), 1 - t / E.T), SIG_GRID], 1).astype(np.float32)
        out[str(t)] = [float(x) for x in pol(t, obs)]
    return out


def first_signal():
    return E.ExecutionEnv(N_EVAL, seed=EVAL_SEED).alpha.copy()


def by_signal_quintile(diff, a0):
    edges = np.percentile(a0, [20, 40, 60, 80])
    q = np.digitize(a0, edges)
    return [float(diff[q == i].mean()) for i in range(5)]


def clip_rate(pol):
    """Share of decisions (steps 0..T-2) where the policy sells nothing or under 0.1% of inventory."""
    env = E.ExecutionEnv(N_EVAL, seed=EVAL_SEED)
    obs, zero = env.reset(), []
    for t in range(E.T):
        f = pol(t, obs)
        if t < E.T - 1:
            zero.append(np.mean(f < 1e-3))
        obs, _, _ = env.step(f)
    return float(np.mean(zero))


def train_eval(seed, iters=300, ckpt=False, **kw):
    curve = []
    def hook(it, net):
        if ckpt and (it + 1) % CKPT_EVERY == 0:
            curve.append([it + 1, float(rollout(learned(net), n_paths=5000, seed=EVAL_SEED + 1).mean())])
    t0 = time.time()
    net, hist = train(use_signal=True, seed=seed, iters=iters, on_iter=hook, **kw)
    return net, hist, curve, time.time() - t0


def core(drift, eta=None):
    """Risk, policy surface, conditional gaps, training curves and optimality gaps at one setting."""
    configure(drift, eta)
    tw = rollout(fixed(twap_fracs))
    sol = dp.solve()
    opt = lq.simulate(dp.policy(sol))
    frontier = []
    for ra in FRONTIER_RA:
        c = rollout(fixed(lambda t: ac_fracs(t, ra)))
        frontier.append({"risk_aversion": ra, **stats(c)})
    k = tune(linear_rule, KS)
    lin = rollout(linear_rule(k))
    a0 = first_signal()
    seeds = []
    for s in SEEDS:
        net, hist, curve, secs = train_eval(s, ckpt=True)
        pol = learned(net)
        c = rollout(pol)
        seeds.append({"seed": s, **stats(c), "vs_twap": paired(c, tw), "vs_linear": paired(c, lin), "vs_dp": paired(c, opt),
                      "quintile_gap_vs_dp": by_signal_quintile(c - opt, a0),
                      "quintile_gap_vs_linear": by_signal_quintile(c - lin, a0), "surface": policy_surface(pol),
                      "train_curve": [float(x) for x in hist], "heldout_curve": curve, "seconds": secs,
                      "zero_sell_rate": clip_rate(pol), "final_log_std": float(net.log_std.item())})
        print(f"core {drift} seed {s} mean {c.mean():.2f} ({secs:.0f}s)", flush=True)
    keep = dp.policy(sol)
    def dp_frac(t, obs):
        if t >= E.T - 1:
            return np.ones(len(obs))
        q = np.maximum(obs[:, 0].astype(float) * E.X0, 1e-9)
        return 1 - keep(t, q, obs[:, 2].astype(float)) / q
    return {"twap": stats(tw), "linear": {"k": float(k), **stats(lin), "surface": policy_surface(linear_rule(k)),
                                          "zero_sell_rate": clip_rate(linear_rule(k)), "vs_dp": paired(lin, opt),
                                          "quintile_gap_vs_dp": by_signal_quintile(lin - opt, a0)},
            "dp": {**stats(opt), "value_bps": sol["value_bps"], "surface": policy_surface(dp_frac),
                   "zero_sell_rate": clip_rate(dp_frac)},
            "ac_frontier": frontier, "ppo": seeds}


def richer(drift, eta=None):
    """Two richer rules tuned on training paths, compared with the one-parameter rule on held-out paths."""
    configure(drift, eta)
    k = tune(linear_rule, KS)
    lin = rollout(linear_rule(k))
    pt = tune(rule_time, TIME_GRID)
    pe = tune(rule_exp, EXP_GRID)
    ct, ce = rollout(rule_time(pt)), rollout(rule_exp(pe))
    return {"linear": {"k": float(k), **stats(lin)},
            "time_rule": {"params": [float(x) for x in pt], **stats(ct), "vs_linear": paired(ct, lin)},
            "logit_rule": {"params": [float(x) for x in pe], **stats(ce), "vs_linear": paired(ce, lin),
                           "surface": policy_surface(rule_exp(pe)), "zero_sell_rate": clip_rate(rule_exp(pe))}}


def persistence(phi):
    configure(0.12, phi=phi)
    tw = rollout(fixed(twap_fracs))
    k = tune(linear_rule, KS)
    lin = rollout(linear_rule(k))
    ppo = []
    for s in SEEDS_SMALL:
        c = rollout(learned(train_eval(s)[0]))
        ppo.append({"seed": s, "mean": float(c.mean()), "vs_twap": paired(c, tw), "vs_linear": paired(c, lin)})
        print(f"phi {phi} seed {s} mean {c.mean():.2f}", flush=True)
    return {"twap": stats(tw), "linear": {"k": float(k), **stats(lin)}, "ppo": ppo}


def exploration(drift, log_std, iters):
    configure(drift)
    lin = rollout(linear_rule(tune(linear_rule, KS)))
    out = []
    for s in SEEDS_SMALL:
        net, _, curve, _ = train_eval(s, iters=iters, ckpt=True, init_log_std=log_std)
        c = rollout(learned(net))
        out.append({"seed": s, "mean": float(c.mean()), "vs_linear": paired(c, lin), "heldout_curve": curve,
                    "final_log_std": float(net.log_std.item())})
        print(f"explore {drift} log_std {log_std} iters {iters} seed {s} mean {c.mean():.2f}", flush=True)
    return out


def optimum(drift, eta=None, phi=None):
    """Exact unconstrained LQ optimum, clipped LQ, constrained DP, and every non-learned policy's gap to the DP."""
    configure(drift, eta, phi)
    al, be, expected = lq.riccati()
    unc = lq.simulate(lq.lq_policy(al, be), clip=False)
    clp = lq.simulate(lq.lq_policy(al, be))
    sol = dp.solve()
    opt = lq.simulate(dp.policy(sol))
    tw = rollout(fixed(twap_fracs))
    ra = tune(lambda r: fixed(lambda t: ac_fracs(t, r)), AC_RA)
    ac = rollout(fixed(lambda t: ac_fracs(t, ra)))
    lin = rollout(linear_rule(tune(linear_rule, KS)))
    pe = tune(rule_exp, EXP_GRID)
    lg = rollout(rule_exp(pe))
    g = lambda c: {**paired(c, opt), "mean_cost": float(c.mean())}
    print(f"optimum drift {drift} eta {eta} phi {phi}: dp {opt.mean():.2f} clipped {clp.mean():.2f} unc {unc.mean():.2f}", flush=True)
    return {"riccati_expected": expected, "alpha": [float(x) for x in al], "beta": [float(x) for x in be],
            "unconstrained": g(unc), "clipped_lq": g(clp), "dp": {**stats(opt), "value_bps": sol["value_bps"]},
            "twap": g(tw), "almgren_chriss": g(ac), "linear": g(lin), "logit_rule": {**g(lg), "params": [float(x) for x in pe]}}


def brute_force_linear(drift, n=20000, max_iter=300):
    """Fit the 2(T-1) coefficients of a time-varying linear policy y = alpha_t q + beta_t a by L-BFGS on training
    paths (seed 999, no clipping) and compare with the Riccati solution."""
    configure(drift)
    rng = np.random.default_rng(TUNE_SEED)
    T = E.T
    a0, Z, EPS = rng.standard_normal(n), rng.standard_normal((T, n)), rng.standard_normal((T, n))
    tw = torch.tensor([1 - 1 / (T - t) for t in range(T - 1)], dtype=torch.float64)
    p = torch.zeros(2, T - 1, dtype=torch.float64, requires_grad=True)
    def cost():
        q, a, c = torch.full((n,), E.X0, dtype=torch.float64), torch.tensor(a0), 0.0
        for t in range(T):
            y = (tw[t] + p[0, t]) * q + 1e4 * p[1, t] * a if t < T - 1 else torch.zeros(n, dtype=torch.float64)
            v = q - y
            c = c - (y * (-E.GAMMA * v + E.DRIFT * a + E.SIGMA * torch.tensor(Z[t])) - E.ETA * v ** 2) / (E.X0 * E.S0) * 1e4
            q, a = y, E.PHI * a + np.sqrt(1 - E.PHI ** 2) * torch.tensor(EPS[t])
        return c.mean()
    opt = torch.optim.LBFGS([p], max_iter=max_iter, line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-14)
    def closure():
        opt.zero_grad()
        loss = cost()
        loss.backward()
        return loss
    opt.step(closure)
    ba, bb = (tw + p[0]).detach().numpy(), 1e4 * p[1].detach().numpy()
    al, be, _ = lq.riccati()
    ref, fit = lq.simulate(lq.lq_policy(al, be), clip=False), lq.simulate(lq.lq_policy(ba, bb), clip=False)
    return {"max_abs_alpha_diff": float(np.abs(ba - al).max()), "max_rel_beta_diff": float(np.abs(bb - be).max() / np.abs(be).max()),
            "heldout_fit_minus_riccati": paired(fit, ref), "beta_fit": [float(x) for x in bb]}


def dp_convergence(drift):
    configure(drift)
    out = []
    for nq, na in ((201, 41), (401, 81), (601, 121)):
        sol = dp.solve(nq=nq, na=na)
        out.append({"nq": nq, "na": na, "value_bps": sol["value_bps"], "sim_mean": float(lq.simulate(dp.policy(sol)).mean())})
    return out


if __name__ == "__main__":
    out = {"seeds": SEEDS, "seeds_small": SEEDS_SMALL, "eval_paths": N_EVAL, "ckpt_every": CKPT_EVERY,
           "ckpt_paths": 5000, "sig_grid": [float(x) for x in SIG_GRID], "grid_t": GRID_T,
           "time_grid_size": len(TIME_GRID), "exp_grid_size": len(EXP_GRID),
           "env": {"X0": E.X0, "S0": E.S0, "T": E.T, "SIGMA": E.SIGMA, "GAMMA": E.GAMMA, "ETA": BASE_ETA, "PHI": BASE_PHI},
           "core": {}, "richer": {}, "persistence": {}, "exploration": {}, "optimum": {}, "brute_force": {}}
    for d in (0.0, 0.03, 0.06, 0.12, 0.24):
        if os.path.exists("study_extra_partial.json"):  # resume: the optimum section is deterministic
            prev = json.load(open("study_extra_partial.json"))
            out.update({k: prev[k] for k in ("optimum", "brute_force", "dp_convergence")})
            break
        out["optimum"]["drift " + str(d)] = optimum(d)
    if "dp_convergence" not in out:
        for sc in (0.5, 2.0):
            out["optimum"]["eta x" + str(sc)] = optimum(0.12, BASE_ETA * sc)
        for phi in (0.0, 0.5, 0.95):
            out["optimum"]["phi " + str(phi)] = optimum(0.12, phi=phi)
        for d in (0.12, 0.24):
            out["brute_force"][str(d)] = brute_force_linear(d)
        out["dp_convergence"] = dp_convergence(0.24)
        json.dump(out, open("study_extra_partial.json", "w"), indent=1)
    for d in (0.0, 0.12, 0.24):
        out["core"][str(d)] = core(d)
    out["core"]["eta2"] = core(0.12, BASE_ETA * 2)
    for d in (0.06, 0.12, 0.24):
        out["richer"][str(d)] = richer(d)
    out["richer"]["eta2"] = richer(0.12, BASE_ETA * 2)
    for phi in (0.0, 0.5, 0.95):
        out["persistence"][str(phi)] = persistence(phi)
    out["exploration"]["logstd0_300"] = exploration(0.24, 0.0, 300)
    out["exploration"]["logstd0_600"] = exploration(0.24, 0.0, 600)
    configure(0.12)
    json.dump(out, open("study_extra.json", "w"), indent=1)
