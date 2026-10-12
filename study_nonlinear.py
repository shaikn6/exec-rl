"""Does RL add value when the market-impact cost is non-linear, so no closed-form optimum exists?

The main study uses quadratic (linear-price) impact, where the Riccati/LQ rule is exactly optimal and RL
cannot beat it. Here we add the square-root impact law (Almgren 2005): per-step cost scales as
sold**1.5 instead of sold**2. There is no closed form, so the benchmark is the grid dynamic-programming
optimum (execrl.dp). We compare PPO, the tuned linear signal rule, TWAP and Almgren-Chriss against it, in
the square-root regime (p=0.5) and, as a control that must reproduce the main paper, the quadratic regime (p=1.0).

The impact coefficient is matched at the TWAP schedule across exponents (execrl.env.temp_eta), so only the
cost curvature changes, not the overall scale. Writes study_nonlinear.json.
Run: python3 study_nonlinear.py
"""

import json

import numpy as np

import execrl.baselines as B
import execrl.env as E
from execrl import dp, lq
from execrl.baselines import ac_fracs, twap_fracs
from execrl.evaluate import fixed, learned, rollout
from execrl.ppo import train

SEEDS = (0, 1, 2, 3, 4)
DRIFTS = (0.0, 0.12, 0.24)
KS = np.linspace(0, 3, 31)
AC_RA = (1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2)
BASE_ETA = E.ETA


def linear_rule(k):
    def pol(t, obs):
        return np.clip(twap_fracs(t) * (1 - k * obs[:, 2]), 0, 1)

    return pol


def regime(p, drift, iters=300):
    E.IMPACT_P, E.DRIFT, E.ETA, B.ETA = p, drift, BASE_ETA, BASE_ETA
    dp_sh = lq.simulate(dp.policy(dp.solve()))  # DP optimum (no closed form when p != 1)
    tw = rollout(fixed(twap_fracs))
    ra_best = min(AC_RA, key=lambda ra: rollout(fixed(lambda t: ac_fracs(t, ra)), n_paths=5000, seed=999).mean())
    ac = rollout(fixed(lambda t: ac_fracs(t, ra_best)))
    k_best = min(KS, key=lambda k: rollout(linear_rule(k), n_paths=5000, seed=999).mean())
    lin = rollout(linear_rule(k_best))
    ppo = [rollout(learned(train(use_signal=True, seed=s, iters=iters)[0])) for s in SEEDS]
    m = lambda c: round(float(c.mean()), 3)
    dpm = float(dp_sh.mean())
    return {
        "impact_p": p,
        "drift": drift,
        "dp_optimum": m(dp_sh),
        "twap": m(tw),
        "almgren_chriss": m(ac),
        "ac_risk_aversion": ra_best,
        "linear_rule": m(lin),
        "linear_rule_k": round(float(k_best), 2),
        "ppo_seed_means": [m(c) for c in ppo],
        "ppo_mean": round(float(np.mean([c.mean() for c in ppo])), 3),
        "ppo_gap_vs_dp": round(float(np.mean([c.mean() for c in ppo]) - dpm), 3),
        "linear_gap_vs_dp": round(float(lin.mean() - dpm), 3),
        "twap_gap_vs_dp": round(float(tw.mean() - dpm), 3),
        # fraction of TWAP's excess cost over the optimum that each policy removes (1 = optimal, 0 = no better than TWAP)
        "ppo_capture": round(float((tw.mean() - np.mean([c.mean() for c in ppo])) / (tw.mean() - dpm)), 3)
        if tw.mean() != dpm
        else None,
        "linear_capture": round(float((tw.mean() - lin.mean()) / (tw.mean() - dpm)), 3) if tw.mean() != dpm else None,
    }


def main():
    out = {
        "note": "Shortfall in bps of arrival notional (lower is better). dp_optimum is the grid-DP benchmark; "
        "in the square-root regime (p=0.5) it is the only optimum available. 'capture' = share of TWAP's "
        "excess-over-optimum that the policy removes. Impact matched at TWAP across exponents.",
        "seeds": list(SEEDS),
        "eval_paths_default": 20000,
        "regimes": {},
    }
    for p in (1.0, 0.5):
        for d in DRIFTS:
            key = f"p{p}_drift{d}"
            out["regimes"][key] = regime(p, d)
            print(key, out["regimes"][key], flush=True)
    json.dump(out, open("study_nonlinear.json", "w"), indent=1)
    print("wrote study_nonlinear.json")


if __name__ == "__main__":
    main()
