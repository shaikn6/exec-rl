import torch

import execrl.env as E
import study_clone as C
from execrl import dp
from execrl.baselines import twap_fracs
from execrl.evaluate import fixed, learned, rollout
from execrl.ppo import ActorCritic, train
from study_extra import configure


def test_warm_start_begins_from_the_given_weights():
    torch.manual_seed(3)
    src = ActorCritic()
    with torch.no_grad():
        src.mu.weight.normal_(0, 0.1)
    net, hist = train(iters=0, init=src.state_dict(), init_log_std=-2.0)
    assert hist == [] and net.log_std.item() == -2.0
    for a, b in zip(src.body.parameters(), net.body.parameters()):
        assert torch.equal(a, b)
    assert torch.equal(src.mu.weight, net.mu.weight)


def test_cloning_moves_the_network_from_twap_towards_the_optimum(monkeypatch):
    monkeypatch.setattr(C, "N_FIT", 1500)
    monkeypatch.setattr(C, "EPOCHS", 8)
    monkeypatch.setattr(C, "ROUNDS", 1)
    try:
        configure(0.24)
        expert = C.optimum_policy(dp.solve(nq=101, na=41, n_gh=9))
        O, Y = C.visited(expert, expert, seed=1)
        assert O.shape == (1500 * (E.T - 1), 3) and float(Y.min()) >= 0 and float(Y.max()) <= 1
        net, losses = C.clone(expert, seed=0)
        twap = rollout(fixed(twap_fracs), n_paths=3000).mean()
        best = rollout(expert, n_paths=3000).mean()
        got = rollout(learned(net), n_paths=3000).mean()
        assert best < twap - 30                      # the signal is worth tens of basis points at this strength
        assert got < twap - 0.8 * (twap - best)      # the fitted network recovers most of it without any reinforcement
    finally:
        configure(0.12)
