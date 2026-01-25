import numpy as np

from src.online_learning import OCOPolicy


def test_ons_convergence():
    rng = np.random.default_rng(42)
    policy = OCOPolicy(lambda_0=5.0, lambda_max=20.0, B_max=20)
    samples = rng.poisson(lam=10.0, size=100).tolist()
    policy.update(samples)
    assert abs(policy.lambda_hat - 10.0) < 1.0


def test_ons_tracks_step_change():
    rng = np.random.default_rng(42)
    policy = OCOPolicy(lambda_0=5.0, lambda_max=30.0, B_max=20)
    policy.update(rng.poisson(lam=5.0, size=50).tolist())
    policy.update(rng.poisson(lam=15.0, size=50).tolist())
    assert policy.lambda_hat > 12.0


def test_oco_sum_available():
    policy = OCOPolicy(lambda_0=10.0, lambda_max=20.0, B_max=20)
    state = {
        "B_occ": 5,
        "q_crit": 10,
        "q_urg": 10,
        "q_rout": 10,
    }
    action = policy.get_action(state)
    assert sum(action.values()) <= 15  # available = 20 - 5 = 15


def test_oco_acuity_order():
    policy = OCOPolicy(lambda_0=10.0, lambda_max=20.0, B_max=20)
    state = {
        "B_occ": 0,
        "q_crit": 20,
        "q_urg": 20,
        "q_rout": 20,
    }
    action = policy.get_action(state)
    assert action["admit_critical"] >= action["admit_urgent"] >= action["admit_routine"]


def test_oco_zero_arrivals():
    policy = OCOPolicy(lambda_0=10.0, lambda_max=20.0, B_max=20)
    policy.update([])
    assert policy.lambda_hat == 10.0
    state = {
        "B_occ": 0,
        "q_crit": 0,
        "q_urg": 0,
        "q_rout": 0,
    }
    action = policy.get_action(state)
    assert sum(action.values()) == 0
