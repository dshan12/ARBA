"""Calibrate ARBA simulation parameters from MIMIC data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import ks_2samp, lognorm

from src.mimic_loader import (
    MIMICData,
    check_mimic_available,
    generate_synthetic_ed_data,
    load_mimic_ed,
)
from src.patient import ACUITY_WEIGHTS, Acuity


def sinusoidal_nhpp(hour: float, base: float, amplitude: float, phase: float) -> float:
    """Sinusoidal NHPP rate function: λ(t) = base * (1 + amplitude * sin(2πt/24 + phase))."""
    return base * (1.0 + amplitude * np.sin(2.0 * np.pi * hour / 24.0 + phase))


def fit_nhpp_sinusoidal(
    arrival_hours: np.ndarray,
    n_days: float,
) -> dict[str, float]:
    """Fit sinusoidal NHPP parameters from hourly arrival data.

    Args:
        arrival_hours: Array of hour-of-day (0-23) for each arrival.
        n_days: Number of unique days in the data.

    Returns:
        Dict with 'lambda_base', 'amplitude', 'phase'.
    """
    if len(arrival_hours) == 0 or n_days <= 0:
        return {"lambda_base": 10.0, "amplitude": 0.3, "phase": np.pi / 2}

    # Bin arrivals by hour and average over days
    hourly_counts = np.zeros(24)
    for h in arrival_hours:
        bin_idx = min(int(h), 23)
        hourly_counts[bin_idx] += 1
    hourly_rate = hourly_counts / n_days

    # Fit sinusoidal model
    hours = np.arange(24, dtype=float)
    try:
        amp_guess = (hourly_rate.max() - hourly_rate.mean()) / hourly_rate.mean()
        initial_guess = [hourly_rate.mean(), max(0.05, amp_guess), np.pi / 2]

        popt, _ = curve_fit(
            sinusoidal_nhpp,
            hours,
            hourly_rate,
            p0=initial_guess,
            bounds=([0.01, 0.0, 0.0], [200.0, 1.0, 2 * np.pi]),
            maxfev=5000,
        )
        base, amplitude, phase = popt
        return {"lambda_base": float(base), "amplitude": float(amplitude), "phase": float(phase)}
    except (RuntimeError, ValueError):
        # Fallback to simple average + default amplitude
        return {
            "lambda_base": float(hourly_rate.mean()),
            "amplitude": 0.3,
            "phase": np.pi / 2,
        }


def fit_service_time_lognormal(
    service_hours: np.ndarray,
    acuities: list[Acuity],
) -> dict[Acuity, dict[str, float]]:
    """Fit log-normal service time parameters per acuity level.

    Args:
        service_hours: LOS in hours for each patient.
        acuities: Acuity for each patient.

    Returns:
        Dict mapping Acuity -> {'mu': float, 'sigma': float}.
    """
    params: dict[Acuity, dict[str, float]] = {}
    all_acuities = list(ACUITY_WEIGHTS.keys())

    for a in all_acuities:
        mask = np.array([ac == a for ac in acuities])
        subset = service_hours[mask]
        if len(subset) < 5:
            # Use defaults
            defaults = {
                Acuity.CRITICAL: {"mu": 3.0, "sigma": 1.0},
                Acuity.URGENT: {"mu": 2.0, "sigma": 0.75},
                Acuity.ROUTINE: {"mu": 1.0, "sigma": 0.5},
            }
            params[a] = defaults.get(a, {"mu": 1.0, "sigma": 0.5})
            continue

        log_data = np.log(np.clip(subset, 0.01, None))
        params[a] = {
            "mu": float(np.mean(log_data)),
            "sigma": float(np.std(log_data, ddof=1)),
        }

    return params


def compute_acuity_mix(acuities: list[Acuity]) -> dict[Acuity, float]:
    """Compute proportion of each acuity level.

    Args:
        acuities: List of Acuity values.

    Returns:
        Dict mapping Acuity -> proportion (sums to 1.0).
    """
    if not acuities:
        return {Acuity.CRITICAL: 0.13, Acuity.URGENT: 0.37, Acuity.ROUTINE: 0.50}
    counts = {a: acuities.count(a) for a in ACUITY_WEIGHTS}
    total = sum(counts.values())
    return {a: counts[a] / total for a in ACUITY_WEIGHTS}


def validate_mimic_params(
    mimic_data: MIMICData,
    calibrated_params: dict,
    ks_threshold: float = 0.05,
) -> dict[str, bool]:
    """Validate calibrated params against MIMIC data using KS tests.

    Args:
        mimic_data: Original MIMIC data.
        calibrated_params: Calibrated parameter dict.
        ks_threshold: p-value threshold for KS test.

    Returns:
        Dict with validation results.
    """
    results: dict[str, bool] = {}

    # Test service time distributions
    for a in ACUITY_WEIGHTS:
        mask = np.array([ac == a for ac in mimic_data.acuities])
        observed = mimic_data.service_hours[mask]
        if len(observed) < 5:
            continue
        sp = calibrated_params["service_params"].get(a, {})
        mu = sp.get("mu", 1.0)
        sigma = sp.get("sigma", 0.5)
        s = sigma
        scale = np.exp(mu)
        _, p_value = ks_2samp(
            observed,
            lognorm.rvs(s=s, scale=scale, size=len(observed), random_state=42),
        )
        results[f"ks_service_{a.name}"] = p_value > ks_threshold

    return results


def calibrate_from_mimic(
    mimic_path: Optional[Path] = None,
    use_synthetic: bool = False,
    synthetic_seed: int = 42,
) -> dict:
    """Calibrate simulation parameters from MIMIC data.

    Args:
        mimic_path: Path to MIMIC data. If None, auto-detect.
        use_synthetic: Generate synthetic data instead of loading real MIMIC.
        synthetic_seed: Seed for synthetic data generation.

    Returns:
        Dict with 'lambda_base', 'amplitude', 'phase', 'acuity_mix',
        'service_params', 'source', and validation info.
    """
    if use_synthetic:
        mimic_data = generate_synthetic_ed_data(seed=synthetic_seed)
    else:
        if not check_mimic_available():
            raise FileNotFoundError(
                "No MIMIC data available. Use use_synthetic=True for testing, "
                "or place MIMIC CSVs in data/mimic/"
            )
        mimic_data = load_mimic_ed(mimic_path)

    # Fit NHPP parameters
    nhpp = fit_nhpp_sinusoidal(mimic_data.arrival_hours, mimic_data.n_days)

    # Scale lambda_base to a realistic hourly rate for simulation
    # The demo has ~1 arrival/day; scale up to match realistic ED
    # Use the actual data's shape (amplitude, phase) but realistic rate
    raw_base = nhpp["lambda_base"]
    if raw_base < 2.0:
        # Demo/limited data: use NHAMCS rate but MIMIC shape
        lambda_base = 10.0
    else:
        lambda_base = raw_base

    # Fit service times
    service_params = fit_service_time_lognormal(mimic_data.service_hours, mimic_data.acuities)

    # Compute acuity mix
    acuity_mix = compute_acuity_mix(mimic_data.acuities)

    # Validate
    params = {
        "lambda_base": lambda_base,
        "amplitude": nhpp["amplitude"],
        "phase": nhpp["phase"],
        "acuity_mix": {a.name: v for a, v in acuity_mix.items()},
        "service_params": {a.name: v for a, v in service_params.items()},
        "source": mimic_data.source,
        "n_patients_mimic": mimic_data.total_arrivals,
        "n_days_mimic": mimic_data.n_days,
    }

    # Run validation (best-effort)
    try:
        validation = validate_mimic_params(mimic_data, params)
        params["validation"] = validation
    except Exception:
        params["validation"] = {}

    return params


def get_mimic_calibrated_params(
    mimic_path: Optional[Path] = None,
    fallback_to_synthetic: bool = True,
) -> dict:
    """Get calibrated parameters from MIMIC or synthetic fallback.

    Args:
        mimic_path: Path to MIMIC data.
        fallback_to_synthetic: If True, generate synthetic data when MIMIC unavailable.

    Returns:
        Calibrated parameter dict.
    """
    if check_mimic_available():
        return calibrate_from_mimic(mimic_path)
    elif fallback_to_synthetic:
        return calibrate_from_mimic(use_synthetic=True)
    else:
        raise FileNotFoundError("MIMIC data not available and fallback disabled")
