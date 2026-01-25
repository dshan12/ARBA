import numpy as np

from src.environment import EDEnvironment, Scenario
from src.patient import Acuity
from src.simulation import SimulationRunner, run_stress_test
from src.theory import bound_adaptive_regret, compute_sublinear_gap


def test_flash_flood_scenario():
    env = EDEnvironment(B_max=20, lambda_base=10.0, seed=42)
    env.configure_scenario(Scenario.FLASH_FLOOD.value, {
        "surge_start": 50,
        "surge_mult": 3.0,
        "surge_duration": 20,
    })
    lam_normal = env.lambda_t(40)
    lam_surge = env.lambda_t(55)
    assert lam_surge > lam_normal * 2, f"Surge {lam_surge} vs normal {lam_normal}"


def test_creeping_crisis_scenario():
    env = EDEnvironment(B_max=20, lambda_base=10.0, seed=42)
    env.configure_scenario(Scenario.CREEPING_CRISIS.value, {"drift_rate": 0.01})
    # At t=100, drift hasn't started (drift_start=100), only sinusoidal
    lam_100 = env.lambda_t(100)
    no_drift_100 = env.lambda_base * (1 + 0.3 * np.sin(2 * np.pi * 100 / 24 + np.pi / 2))
    assert abs(lam_100 / no_drift_100 - 1.0) < 0.01, f"No drift expected at t=100, got {lam_100}"
    # At t=150, drift has been active for 50 hours: 1.01**50
    lam_150 = env.lambda_t(150)
    drift_factor = 1.01 ** 50
    no_drift_150 = env.lambda_base * (1 + 0.3 * np.sin(2 * np.pi * 150 / 24 + np.pi / 2))
    expected_150 = no_drift_150 * drift_factor
    assert abs(lam_150 / expected_150 - 1.0) < 0.01, f"Expected {expected_150}, got {lam_150}"


def test_acuity_flip_scenario():
    env = EDEnvironment(B_max=20, lambda_base=10.0, seed=42)
    env.configure_scenario(Scenario.ACUITY_FLIP.value, {
        "t0": 100,
        "composition": {Acuity.CRITICAL: 0.4, Acuity.URGENT: 0.3, Acuity.ROUTINE: 0.3},
    })
    counts_pre: dict[Acuity, int] = {Acuity.CRITICAL: 0, Acuity.URGENT: 0, Acuity.ROUTINE: 0}
    counts_post: dict[Acuity, int] = {Acuity.CRITICAL: 0, Acuity.URGENT: 0, Acuity.ROUTINE: 0}
    for _ in range(1000):
        counts_pre[env.sample_acuity(t=50)] += 1
        counts_post[env.sample_acuity(t=150)] += 1
    total_pre = sum(counts_pre.values())
    total_post = sum(counts_post.values())
    crit_ratio_pre = counts_pre[Acuity.CRITICAL] / total_pre
    crit_ratio_post = counts_post[Acuity.CRITICAL] / total_post
    assert crit_ratio_post > crit_ratio_pre, (
        f"Critical ratio: pre={crit_ratio_pre:.3f}, post={crit_ratio_post:.3f}"
    )


def test_stress_test_runner_runs():
    result = run_stress_test(
        scenario_name=Scenario.FLASH_FLOOD.value,
        lambda_base=3.0,
        n_reps=2,
        duration=10.0,
        B_max=5,
    )
    assert result["scenario"] == Scenario.FLASH_FLOOD.value
    assert result["n_reps"] == 2
    for pt in ("robust", "adaptive"):
        assert "mean_regret" in result[pt]
        assert "ci_95" in result[pt]
        assert len(result[pt]["values"]) == 2


def test_flash_flood_simulation_runs():
    runner = SimulationRunner(B_max=5, duration=10.0, seed=42)
    metrics = runner.run_scenario(
        lambda_base=3.0,
        policy_type="robust",
        scenario=Scenario.FLASH_FLOOD.value,
        scenario_params={"surge_start": 3, "surge_mult": 3.0, "surge_duration": 3},
    )
    assert "V_star" in metrics
    assert "regret" in metrics
    assert metrics["n_patients"] > 0


def test_creeping_crisis_simulation_runs():
    runner = SimulationRunner(B_max=5, duration=10.0, seed=42)
    metrics = runner.run_scenario(
        lambda_base=3.0,
        policy_type="adaptive",
        scenario=Scenario.CREEPING_CRISIS.value,
        scenario_params={"drift_rate": 0.01},
    )
    assert "V_star" in metrics
    assert isinstance(metrics["regret"], (int, float))


def test_acuity_flip_simulation_runs():
    runner = SimulationRunner(B_max=5, duration=10.0, seed=42)
    metrics = runner.run_scenario(
        lambda_base=3.0,
        policy_type="robust",
        scenario=Scenario.ACUITY_FLIP.value,
        scenario_params={
            "t0": 5,
            "composition": {Acuity.CRITICAL: 0.4, Acuity.URGENT: 0.3, Acuity.ROUTINE: 0.3},
        },
    )
    assert "V_star" in metrics
    assert isinstance(metrics["regret"], (int, float))


def test_compute_sublinear_gap_log():
    T = 100
    regret = 5.0 * np.log(np.arange(1, T + 1))
    alpha, ci_lo, ci_hi = compute_sublinear_gap(regret)
    assert alpha < 0.6, f"Log growth should give sublinear exponent, got {alpha}"
    assert ci_lo <= alpha <= ci_hi


def test_compute_sublinear_gap_sqrt():
    T = 100
    regret = 2.0 * np.sqrt(np.arange(1, T + 1))
    alpha, ci_lo, ci_hi = compute_sublinear_gap(regret)
    assert 0.3 <= alpha <= 0.7, f"Sqrt growth should give ~0.5 exponent, got {alpha}"
    assert ci_lo <= alpha <= ci_hi


def test_compute_sublinear_gap_linear():
    T = 100
    regret = 3.0 * np.arange(1, T + 1)
    alpha, ci_lo, ci_hi = compute_sublinear_gap(regret)
    assert alpha > 0.7, f"Linear growth should give ~1.0 exponent, got {alpha}"
    assert ci_lo <= alpha <= ci_hi


def test_bound_adaptive_regret():
    bound = bound_adaptive_regret(T=168, delta_min=0.5, B_max=20)
    assert bound > 0
    assert np.isfinite(bound)


def test_bound_adaptive_regret_zero_delta():
    bound = bound_adaptive_regret(T=168, delta_min=0.0, B_max=20)
    assert bound == float("inf")


def test_bound_adaptive_regret_monotonic():
    b1 = bound_adaptive_regret(T=100, delta_min=0.5, B_max=20)
    b2 = bound_adaptive_regret(T=200, delta_min=0.5, B_max=20)
    assert b2 > b1


def test_bound_adaptive_regret_lorden_detection_delay():
    """Verify the detection delay component uses KL divergence (Lorden)."""
    bound = bound_adaptive_regret(T=168, delta_min=2.0, B_max=20)
    assert np.isfinite(bound)
    assert bound > 0
    kl = 12.0 * np.log(12.0 / 10.0) - 2.0
    h = np.log(169)
    expected_delay = (h + kl) / kl
    expected_post = (12.0 / 2.0) * np.log(169) / kl
    expected = 20.0 * (expected_delay + expected_post)
    assert abs(bound / expected - 1.0) < 1e-6


def test_compute_sublinear_gap_significance():
    """Statistically significant sublinear exponent should have CI below 1."""
    rng = np.random.default_rng(42)
    T = 200
    regret = 3.0 * np.sqrt(np.arange(1, T + 1)) + rng.normal(0, 0.5, T).cumsum()
    regret = np.abs(regret)
    alpha, ci_lo, ci_hi = compute_sublinear_gap(regret)
    assert alpha < 0.8
    assert ci_lo < ci_hi
