"""Paper study: how much of an alpha signal's value does PPO capture, across signal strength and impact?

Every PPO seed is reported (no best-of-seed selection). The linear signal rule is tuned on separate training paths
(seed 999), never on the evaluation paths (seed 12345). Writes study.json."""
import json

import numpy as np

import execrl.baselines as B
import execrl.env as E
from execrl.baselines import ac_fracs, twap_fracs
from execrl.evaluate import fixed, learned, rollout
from execrl.ppo import train

SEEDS = (0, 1, 2, 3, 4)
LONG_ITERS = 600  # twice the default training budget, to test whether PPO's shortfall at strong signals is under-training
DRIFTS = (0.0, 0.03, 0.06, 0.12, 0.24)
ETA_SCALES = (0.5, 2.0)  # temporary impact relative to the default, at the default drift
KS = np.linspace(0, 3, 31)
AC_RA = (1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2)  # run.py's grid


def linear_rule(k):
    """Sell the TWAP slice scaled by (1 - k * signal): sell less when the signal says the price will rise."""
    def pol(t, obs):
        return np.clip(twap_fracs(t) * (1 - k * obs[:, 2]), 0, 1)
    return pol


def setting(drift, eta, iters=300):
    E.DRIFT, E.ETA, B.ETA = drift, eta, eta  # baselines copied ETA at import, so set both
    tw = rollout(fixed(twap_fracs))
    ra_best = min(AC_RA, key=lambda ra: rollout(fixed(lambda t: ac_fracs(t, ra)), n_paths=5000, seed=999).mean())  # tuned on training paths
    ac = rollout(fixed(lambda t: ac_fracs(t, ra_best)))
    k_best = min(KS, key=lambda k: rollout(linear_rule(k), n_paths=5000, seed=999).mean())  # tuned on training paths
    lin = rollout(linear_rule(k_best))
    ppo = [rollout(learned(train(use_signal=True, seed=s, iters=iters)[0])) for s in SEEDS]
    m = lambda c: round(float(c.mean()), 2)
    return {"twap": m(tw), "almgren_chriss": m(ac), "ac_risk_aversion": ra_best, "linear_rule": m(lin), "linear_rule_k": round(float(k_best), 2),
            "ppo_seed_means": [m(c) for c in ppo], "ppo_mean_over_seeds": round(float(np.mean([c.mean() for c in ppo])), 2),
            "ppo_minus_twap_paired": [round(float((c - tw).mean()), 2) for c in ppo],
            "ppo_minus_linear_paired": [round(float((c - lin).mean()), 2) for c in ppo],
            "se_paired_vs_twap": [round(float((c - tw).std() / np.sqrt(len(c))), 3) for c in ppo]}


if __name__ == "__main__":
    base_eta = E.ETA
    out = {"drift_sweep": {}, "eta_sweep": {}, "long_training": {}, "long_iters": LONG_ITERS, "eval_paths": 20000, "seeds": SEEDS}
    for d in DRIFTS:
        out["drift_sweep"][str(d)] = setting(d, base_eta)
        print("drift", d, out["drift_sweep"][str(d)], flush=True)
    for s in ETA_SCALES:
        out["eta_sweep"][str(s)] = setting(0.12, base_eta * s)
        print("eta x", s, out["eta_sweep"][str(s)], flush=True)
    for d in (0.12, 0.24):
        out["long_training"][str(d)] = setting(d, base_eta, iters=LONG_ITERS)
        print("long", d, out["long_training"][str(d)], flush=True)
    json.dump(out, open("study.json", "w"), indent=2)
