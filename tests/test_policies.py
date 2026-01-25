from src.policies import AdaptivePolicy, RobustPolicy


def test_robust_policy_reserves_critical():
    policy = RobustPolicy(lambda_adv=10.0, B_max=20, gamma=0.2)
    state = {
        "B_occ": 5,
        "q_crit": 2,
        "q_urg": 3,
        "q_rout": 5,
    }
    action = policy.get_action(state)
    assert action["admit_critical"] == 2
    assert action["admit_urgent"] == 3
    assert action["admit_routine"] == 5
    assert sum(action.values()) == 10


def test_robust_policy_holds_reserve():
    policy = RobustPolicy(lambda_adv=10.0, B_max=20, gamma=0.5)
    state = {
        "B_occ": 0,
        "q_crit": 0,
        "q_urg": 15,
        "q_rout": 5,
    }
    action = policy.get_action(state)
    assert action["admit_urgent"] == 10
    assert action["admit_routine"] == 0
    assert sum(action.values()) == 10


def test_robust_policy_capacity():
    policy = RobustPolicy(lambda_adv=10.0, B_max=20)
    state = {
        "B_occ": 18,
        "q_crit": 5,
        "q_urg": 3,
        "q_rout": 2,
    }
    action = policy.get_action(state)
    assert sum(action.values()) <= 2


def test_adaptive_policy_initial_mode():
    policy = AdaptivePolicy(lambda_adv=10.0, B_max=20)
    assert not policy.adaptive_mode


def test_adaptive_policy_triggers():
    policy = AdaptivePolicy(lambda_adv=10.0, B_max=20, delta=0.5, threshold=3.0)
    policy.update([20.0] * 10)
    assert policy.adaptive_mode
