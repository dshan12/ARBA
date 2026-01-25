from src.environment import EDEnvironment


def test_environment_runs():
    env = EDEnvironment(B_max=10, lambda_base=5.0, seed=42)
    patients = env.run(duration=10.0)
    assert len(patients) > 0


def test_lambda_t_varies():
    env = EDEnvironment(B_max=10, lambda_base=10.0, seed=42)
    lam_0 = env.lambda_t(0)
    lam_12 = env.lambda_t(12)
    assert lam_0 != lam_12


def test_surge_increases_arrivals():
    env = EDEnvironment(
        B_max=10, lambda_base=5.0, surge_start=5, surge_mult=3.0, seed=42
    )
    lam_before = env.lambda_t(4)
    lam_during = env.lambda_t(6)
    assert lam_during > lam_before
