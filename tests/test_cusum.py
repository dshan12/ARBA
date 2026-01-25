import numpy as np

from src.cusum import CUSUMModule

rng = np.random.default_rng(42)


def test_cusum_no_drift():
    cusum = CUSUMModule(lambda_0=10.0, lambda_1=11.0, threshold=20.0)
    for _ in range(50):
        x = rng.poisson(10.0)
        cusum.update(x)
    assert not cusum.is_triggered()


def test_cusum_drift_detected():
    cusum = CUSUMModule(lambda_0=10.0, lambda_1=15.0, threshold=5.0)
    for _ in range(30):
        x = rng.poisson(15.0)
        cusum.update(x)
    assert cusum.is_triggered()


def test_cusum_reset():
    cusum = CUSUMModule(lambda_0=10.0, lambda_1=11.0, threshold=5.0)
    for _ in range(30):
        cusum.update(rng.poisson(15.0))
    assert cusum.S_t > 0
    cusum.reset(lambda_0=15.0)
    assert cusum.S_t == 0.0
    assert cusum.lambda_0 == 15.0


def test_cusum_lambda_1():
    cusum = CUSUMModule(lambda_0=10.0, lambda_1=12.0, threshold=5.0)
    expected_k = (12.0 - 10.0) / np.log(12.0 / 10.0)
    assert abs(cusum.k - expected_k) < 1e-10
    assert cusum.lambda_1 == 12.0


def test_cusum_calibrate_threshold():
    cusum = CUSUMModule(lambda_0=10.0, lambda_1=12.0, threshold=5.0)
    h = cusum.calibrate_threshold(arl_target=100.0, n_sim=2000, seed=42)
    assert 1.0 <= h <= 50.0
    assert cusum.threshold == h
