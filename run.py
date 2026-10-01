"""Train PPO (with and without the alpha signal), benchmark against baselines, save results + plot."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from execrl.ppo import train
from execrl.evaluate import rollout, summarize, fixed, learned, paired_diff
from execrl.baselines import twap_fracs, immediate_fracs, ac_fracs

log = lambda m: print(m, flush=True)
res, costs = {}, {}

costs["TWAP"] = rollout(fixed(twap_fracs))
costs["Sell immediately"] = rollout(fixed(immediate_fracs))
best_ac = min(((lam, rollout(fixed(lambda t, l=lam: ac_fracs(t, l)))) for lam in [1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2]),
              key=lambda x: x[1].mean())
costs[f"Almgren-Chriss (lambda={best_ac[0]:g})"] = best_ac[1]

seeds = [0, 1, 2]
for use_signal, name in [(False, "PPO (no signal)"), (True, "PPO (with alpha signal)")]:
    per_seed = []
    for s in seeds:
        log(f"--- {name} seed {s}")
        net, hist = train(use_signal=use_signal, seed=s, log=log)
        per_seed.append((rollout(learned(net), use_signal=use_signal), hist))
    best = min(per_seed, key=lambda x: x[0].mean())
    costs[name] = best[0]
    res[name + " seed means"] = [round(float(c.mean()), 2) for c, _ in per_seed]
    if use_signal:
        curve = np.mean([h for _, h in per_seed], axis=0)

for k, c in costs.items():
    res[k] = summarize(c)
ac_key = [k for k in costs if k.startswith("Almgren")][0]
res["PPO(signal) minus TWAP (bps, paired)"] = paired_diff(costs["PPO (with alpha signal)"], costs["TWAP"])
res["PPO(signal) minus AC (bps, paired)"] = paired_diff(costs["PPO (with alpha signal)"], costs[ac_key])
json.dump(res, open("results.json", "w"), indent=2)
for k, v in res.items():
    log(f"{k}: {v}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(curve); ax[0].set_title("PPO training curve (mean of 3 seeds)")
ax[0].set_xlabel("iteration"); ax[0].set_ylabel("mean implementation shortfall (bps)")
names = list(costs); ax[1].bar(range(len(names)), [costs[n].mean() for n in names],
                               yerr=[costs[n].std() / np.sqrt(len(costs[n])) * 1.96 for n in names], capsize=3)
ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=7)
ax[1].set_title("Held-out shortfall, 20k paths (lower is better)")
plt.tight_layout(); plt.savefig("results.png", dpi=140)
