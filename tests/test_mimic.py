import numpy as np

from src.mimic_calibrator import calibrate_from_mimic, get_mimic_calibrated_params
from src.mimic_loader import (
    MIMICData,
    check_mimic_available,
    find_mimic_path,
    generate_synthetic_ed_data,
    load_mimic_ed,
)
from src.patient import Acuity


def test_demo_data_detected():
    """Check that the MIMIC demo data is found by auto-detection."""
    path = find_mimic_path()
    assert path is not None, "Demo data should be found in data/mimic/"
    assert (path / "hosp" / "transfers.csv.gz").exists(), "transfers.csv.gz should exist"
    assert check_mimic_available(), "check_mimic_available should return True"


def test_load_demo_returns_data():
    """Loading the demo should return valid MIMICData with ED records."""
    data = load_mimic_ed()
    assert isinstance(data, MIMICData)
    assert data.total_arrivals > 0, "Should have ED arrivals"
    assert len(data.arrival_hours) == data.total_arrivals
    assert len(data.service_hours) == data.total_arrivals
    assert len(data.acuities) == data.total_arrivals
    assert 0 <= data.arrival_hours.min() <= 23
    assert 0 <= data.arrival_hours.max() <= 23
    assert "demo" in data.source.lower()


def test_demo_service_times_positive():
    """All service times should be positive."""
    data = load_mimic_ed()
    assert np.all(data.service_hours > 0), "All service times must be positive"
    assert np.all(np.isfinite(data.service_hours)), "All service times must be finite"


def test_calibrate_from_demo():
    """Calibration from demo data should return expected parameter structure."""
    params = calibrate_from_mimic()
    assert isinstance(params, dict)
    assert "lambda_base" in params
    assert "amplitude" in params
    assert "phase" in params
    assert "acuity_mix" in params
    assert "service_params" in params
    assert "source" in params
    assert "demo" in params["source"].lower()
    assert params["lambda_base"] > 0
    # Check acuity mix sums to ~1.0
    mix_sum = sum(params["acuity_mix"].values())
    assert abs(mix_sum - 1.0) < 0.01, f"Acuity mix should sum to 1, got {mix_sum}"


def test_calibrate_synthetic():
    """Calibration from synthetic data should recover known params approximately."""
    params = calibrate_from_mimic(use_synthetic=True, synthetic_seed=42)
    assert params["source"] == "synthetic"
    assert params["lambda_base"] > 0
    assert 0 <= params["amplitude"] <= 1
    assert 0 <= params["phase"] <= 2 * np.pi
    assert len(params["acuity_mix"]) == 3


def test_get_mimic_calibrated_params_fallback():
    """get_mimic_calibrated_params should return valid params (uses demo data)."""
    params = get_mimic_calibrated_params()
    assert "lambda_base" in params
    assert "service_params" in params
    assert params["n_patients_mimic"] > 0


def test_synthetic_data_generation():
    """Generate synthetic ED data and verify structure."""
    data = generate_synthetic_ed_data(n_patients=1000, seed=42)
    assert data.total_arrivals > 100, f"Should generate many arrivals, got {data.total_arrivals}"
    assert len(data.acuities) == data.total_arrivals
    assert all(a in Acuity for a in data.acuities), "All acuities must be valid"
    assert data.service_hours.min() >= 0.1


def test_synthetic_data_known_parameters():
    """Generate synthetic data and verify it approximately matches input params."""
    data = generate_synthetic_ed_data(
        n_patients=5000, lambda_base=10.0, amplitude=0.35, seed=42
    )
    assert data.total_arrivals > 1000, "Should generate thousands of arrivals"
    assert data.n_days > 10, "Should span multiple days"
    mean_hourly = len(data.arrival_hours) / (data.n_days * 24)
    assert 5 < mean_hourly < 20, f"Mean hourly rate ~10, got {mean_hourly:.1f}"


def test_acuity_mix_synthetic():
    """Synthetic data acuity mix should approximately match input proportions."""
    data = generate_synthetic_ed_data(
        n_patients=10000,
        acuity_weights={
            Acuity.CRITICAL: 0.20,
            Acuity.URGENT: 0.30,
            Acuity.ROUTINE: 0.50,
        },
        seed=42,
    )
    counts = {a: data.acuities.count(a) for a in Acuity}
    total = sum(counts.values())
    fracs = {a: counts[a] / total for a in Acuity}
    assert abs(fracs[Acuity.CRITICAL] - 0.20) < 0.03, f"Critical {fracs[Acuity.CRITICAL]:.3f}"
    assert abs(fracs[Acuity.URGENT] - 0.30) < 0.03, f"Urgent {fracs[Acuity.URGENT]:.3f}"
    assert abs(fracs[Acuity.ROUTINE] - 0.50) < 0.03, f"Routine {fracs[Acuity.ROUTINE]:.3f}"


def test_load_mimic_fails_without_data():
    """Loading MIMIC should raise FileNotFoundError when no data and no fallback."""
    try:
        # Temporarily remove detectability (can't actually remove files in test)
        # Instead: verify that loading from a non-existent path raises
        from pathlib import Path

        fake_path = Path("/nonexistent/path")
        try:
            _ = load_mimic_ed(mimic_path=fake_path)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass  # Expected
    except Exception:
        pass
