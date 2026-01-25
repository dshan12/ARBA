import pytest

from src.cusum import CUSUMModule
from src.policies import AdaptivePolicy, RobustPolicy


def test_cusum_rejects_invalid_lambda_0():
    with pytest.raises(ValueError, match="must be > 0"):
        CUSUMModule(lambda_0=0.0, lambda_1=10.0)


def test_cusum_rejects_invalid_lambda_1():
    with pytest.raises(ValueError, match="must be > 0"):
        CUSUMModule(lambda_0=10.0, lambda_1=-1.0)


def test_cusum_rejects_equal_lambdas():
    with pytest.raises(ValueError, match="must differ"):
        CUSUMModule(lambda_0=10.0, lambda_1=10.0)


def test_cusum_rejects_negative_threshold():
    with pytest.raises(ValueError, match="threshold must be >= 0"):
        CUSUMModule(lambda_0=10.0, lambda_1=15.0, threshold=-1.0)


def test_robust_policy_rejects_invalid_gamma():
    with pytest.raises(ValueError, match="gamma must be in"):
        RobustPolicy(lambda_adv=10.0, B_max=20, gamma=-0.1)
    with pytest.raises(ValueError, match="gamma must be in"):
        RobustPolicy(lambda_adv=10.0, B_max=20, gamma=1.5)


def test_robust_policy_rejects_invalid_beds():
    with pytest.raises(ValueError, match="B_max must be > 0"):
        RobustPolicy(lambda_adv=10.0, B_max=0)


def test_robust_policy_rejects_invalid_lambda():
    with pytest.raises(ValueError, match="lambda_adv must be >= 0"):
        RobustPolicy(lambda_adv=-1.0, B_max=20)


def test_robust_policy_admits_non_negative():
    policy = RobustPolicy(lambda_adv=10.0, B_max=20)
    state = {"B_occ": 25, "q_crit": 5, "q_urg": 3, "q_rout": 2}
    action = policy.get_action(state)
    assert action["admit_critical"] >= 0
    assert action["admit_urgent"] >= 0
    assert action["admit_routine"] >= 0
    assert sum(action.values()) == 0


def test_adaptive_policy_adapts_in_adaptive_mode():
    policy = AdaptivePolicy(lambda_adv=10.0, B_max=20, delta=0.5, threshold=3.0)
    policy.adaptive_mode = True
    policy.lambda_current = 20.0
    policy.lambda_base = 10.0
    state = {"B_occ": 5, "q_crit": 2, "q_urg": 3, "q_rout": 5}
    action = policy.get_action(state)
    assert sum(action.values()) >= 0
    assert all(v >= 0 for v in action.values())


def test_adaptive_policy_update_with_empty_history_no_crash():
    policy = AdaptivePolicy(lambda_adv=10.0, B_max=20)
    policy.adaptive_mode = True
    policy.expansion_remaining = 0
    policy.arrival_history = []
    policy.update([])
    assert not policy.adaptive_mode or policy.lambda_current > 0
