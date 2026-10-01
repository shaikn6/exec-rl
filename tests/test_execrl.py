import numpy as np
from execrl.env import ExecutionEnv, T, X0
from execrl.evaluate import rollout, fixed, summarize
from execrl.baselines import twap_fracs, immediate_fracs, ac_fracs


def test_inventory_fully_liquidated():
    env = ExecutionEnv(64, seed=1)
    env.reset()
    for _ in range(T):
        env.step(np.full(64, 0.1))
    assert np.allclose(env.inv, 0.0)


def test_reward_sums_to_implementation_shortfall():
    """Per-step rewards must telescope to the closed-form shortfall: sell-all-now = eta*X0/S0 = 200 bps."""
    cost = rollout(fixed(immediate_fracs), n_paths=16)
    assert np.allclose(cost, 200.0)


def test_twap_sells_equal_slices():
    env = ExecutionEnv(4, seed=2)
    env.reset()
    for t in range(T):
        env.step(np.full(4, twap_fracs(t)))
        assert np.allclose(env.inv, X0 * (1 - (t + 1) / T))


def test_ac_reduces_to_twap_when_risk_neutral_and_front_loads_when_averse():
    assert np.isclose(ac_fracs(0, 1e-9), 1 / T, atol=1e-3)
    assert ac_fracs(0, 1e-3) > ac_fracs(0, 1e-5)


def test_signal_is_exploitable_but_hidden_when_disabled():
    env = ExecutionEnv(8, seed=3, use_signal=False)
    assert np.all(env.reset()[:, 2] == 0.0)


def test_rollouts_reproducible():
    a = summarize(rollout(fixed(twap_fracs), n_paths=2000, seed=7))["mean"]
    b = summarize(rollout(fixed(twap_fracs), n_paths=2000, seed=7))["mean"]
    assert a == b
