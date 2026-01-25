from unittest.mock import MagicMock

from src.environment import EDEnvironment


def test_full_simulation_runs():
    env = EDEnvironment(B_max=10, lambda_base=5.0, seed=42)
    patients = env.run(duration=10.0)
    assert len(patients) > 0
    for p in patients:
        if p.service_start is not None:
            assert p.wait_time >= 0
        assert p.service_start is not None or p.service_end is None


def test_policy_invoked():
    mock_policy = MagicMock()
    mock_policy.get_action.return_value = {
        "admit_critical": 0,
        "admit_urgent": 0,
        "admit_routine": 0,
    }
    env = EDEnvironment(B_max=5, lambda_base=10.0, seed=42)
    patients = env.run(duration=5.0, allocation_policy=mock_policy)
    assert len(patients) > 0
    mock_policy.get_action.assert_called()
