import numpy as np
import pytest

import execrl.env as E
from execrl import dp, lq
from execrl.baselines import twap_fracs
from execrl.evaluate import fixed, rollout


@pytest.fixture
def drift():
    old = E.DRIFT
    yield lambda d: setattr(E, "DRIFT", d)
    E.DRIFT = old


def test_replay_matches_environment_paths():
    env_cost = rollout(fixed(twap_fracs), n_paths=500, seed=5)
    replay = lq.simulate(lambda t, q, a: q * (1 - twap_fracs(t)), n_paths=500, seed=5)
    assert np.allclose(env_cost, replay)


def test_riccati_without_signal_is_twap(drift):
    drift(0.0)
    alpha, beta, _ = lq.riccati()
    assert np.allclose(alpha, [1 - 1 / (E.T - t) for t in range(E.T - 1)])
    assert np.allclose(beta, 0.0)


def test_riccati_expected_cost_matches_simulation(drift):
    drift(0.12)
    alpha, beta, expected = lq.riccati()
    c = lq.simulate(lq.lq_policy(alpha, beta), n_paths=40000, seed=3, clip=False)
    assert abs(c.mean() - expected) < 4 * c.std() / np.sqrt(len(c))


def test_riccati_policy_is_a_local_minimum(drift):
    drift(0.12)
    alpha, beta, _ = lq.riccati()
    base = lq.simulate(lq.lq_policy(alpha, beta), seed=4, clip=False)
    for t in (0, 9, 17):
        for da, db in ((0.02, 0), (-0.02, 0), (0, 0.1), (0, -0.1)):
            a2, b2 = alpha.copy(), beta.copy()
            a2[t] += da
            b2[t] += db * abs(beta[t])
            assert lq.simulate(lq.lq_policy(a2, b2), seed=4, clip=False).mean() > base.mean()


def test_constrained_dp_reduces_to_twap_and_beats_clipped_lq(drift):
    drift(0.0)
    sol = dp.solve(nq=101, na=21)
    twap = lq.simulate(lambda t, q, a: q * (1 - twap_fracs(t)), n_paths=2000, seed=6)
    assert abs(lq.simulate(dp.policy(sol), n_paths=2000, seed=6).mean() - twap.mean()) < 0.05
    drift(0.24)
    alpha, beta, _ = lq.riccati()
    sol = dp.solve(nq=201, na=41)
    clipped = lq.simulate(lq.lq_policy(alpha, beta), n_paths=4000, seed=6)
    assert lq.simulate(dp.policy(sol), n_paths=4000, seed=6).mean() < clipped.mean()
