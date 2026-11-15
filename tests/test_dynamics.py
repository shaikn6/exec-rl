"""Simulator dynamics, the shortfall identity, the AC/TWAP baselines, and a tiny CPU training smoke test."""
import numpy as np

import execrl.env as E
from execrl.env import ExecutionEnv, T, X0, S0, GAMMA, ETA, temp_cost, temp_eta
from execrl.baselines import twap_fracs, ac_fracs
from execrl.evaluate import rollout, fixed, learned, summarize
from execrl.ppo import train


def test_permanent_impact_moves_price_by_gamma_times_sold(monkeypatch):
    """With noise and drift switched off, the price falls by exactly GAMMA * shares sold."""
    monkeypatch.setattr(E, "SIGMA", 0.0)
    monkeypatch.setattr(E, "DRIFT", 0.0)
    env = ExecutionEnv(5, seed=1)
    env.reset()
    p0, sold = env.price.copy(), 0.1 * env.inv
    env.step(np.full(5, 0.1))
    assert np.allclose(p0 - env.price, GAMMA * sold)


def test_last_step_forces_full_liquidation():
    """Even if the policy asks to sell nothing, the final step liquidates the remaining inventory."""
    env = ExecutionEnv(3, seed=2)
    env.reset()
    for _ in range(T):
        env.step(np.zeros(3))
    assert np.allclose(env.inv, 0.0)


def test_action_is_clipped_to_the_unit_interval():
    env = ExecutionEnv(3, seed=2)
    env.reset()
    env.step(np.full(3, 5.0))  # frac > 1 clips to "sell everything", never a short position
    assert np.all(env.inv >= 0.0) and np.allclose(env.inv, 0.0)


def test_signal_is_ar1_with_persistence_phi():
    """The alpha signal is stationary (unit variance) with lag-1 autocorrelation PHI."""
    n = 200_000
    env = ExecutionEnv(n, seed=4)
    env.reset()
    a_prev = env.alpha.copy()
    env.step(np.zeros(n))
    a_next = env.alpha
    assert abs(a_next.var() - 1.0) < 0.02
    assert abs(np.corrcoef(a_prev, a_next)[0, 1] - E.PHI) < 0.02


def test_shortfall_identity_matches_reward_telescoping():
    """The summation-by-parts reward must telescope to (arrival notional - proceeds), proceeds recomputed
    independently as sum_t (shares sold * price before the step - temporary-impact cost)."""
    n, rng = 2000, np.random.default_rng(0)
    env = ExecutionEnv(n, seed=9)
    env.reset()
    cost_from_reward, proceeds = np.zeros(n), np.zeros(n)
    for t in range(T):
        frac = np.ones(n) if t == T - 1 else rng.uniform(0.0, 0.3, n)
        inv_before, price_before = env.inv.copy(), env.price.copy()
        _, r, _ = env.step(frac)
        cost_from_reward -= r
        sold = np.clip(frac, 0.0, 1.0) * inv_before
        proceeds += sold * price_before - temp_cost(sold)
    shortfall = (X0 * S0 - proceeds) / (X0 * S0) * 1e4
    assert np.allclose(cost_from_reward, shortfall, atol=1e-6)


def test_temp_eta_is_calibrated_to_eta_at_the_quadratic_exponent():
    assert temp_eta(1.0) == ETA
    assert np.isclose(temp_cost(5000.0, p=1.0), ETA * 5000.0 ** 2)


def test_twap_sample_mean_matches_closed_form():
    """TWAP sells X0/T each step; noise and signal are mean-zero, so the expected shortfall has a closed form.

    E[cost] = (GAMMA * (X0/T) * X0 * (T-1)/2 + ETA * X0**2 / T) / (X0 * S0) * 1e4.
    The held-out sample mean must sit within a few standard errors of it.
    """
    closed = (GAMMA * (X0 / T) * X0 * (T - 1) / 2 + ETA * X0 ** 2 / T) / (X0 * S0) * 1e4
    s = summarize(rollout(fixed(twap_fracs)))
    assert abs(s["mean"] - closed) < 4 * s["se"]


def test_ac_inventory_follows_the_sinh_trajectory():
    """Replaying ac_fracs must reproduce the Almgren-Chriss sinh holdings, and the last slice sells everything."""
    lam = 1e-3
    kappa = np.sqrt(lam * E.SIGMA ** 2 / E.ETA)
    hold = lambda k: np.sinh(kappa * (T - k)) / np.sinh(kappa * T)
    inv = 1.0
    for t in range(T - 1):
        inv *= 1.0 - ac_fracs(t, lam)
        assert np.isclose(inv, hold(t + 1))
    assert np.isclose(ac_fracs(T - 1, lam), 1.0)


def test_tiny_training_is_deterministic_and_finite():
    """A 2-iteration CPU run must be reproducible and produce finite shortfalls (fast smoke test)."""
    _, h1 = train(iters=2, n_envs=64, seed=0)
    _, h2 = train(iters=2, n_envs=64, seed=0)
    assert h1 == h2 and len(h1) == 2 and all(np.isfinite(h1))


def test_untrained_policy_equals_twap():
    """The actor head is zero-initialised and anchored on the TWAP slice, so an untrained policy is TWAP."""
    net, _ = train(iters=0, n_envs=64, seed=0, use_signal=False)
    c_net = rollout(learned(net), use_signal=False, n_paths=2000)
    c_twap = rollout(fixed(twap_fracs), n_paths=2000)
    assert np.allclose(c_net, c_twap, atol=1e-3)
