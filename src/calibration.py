import json
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from src.patient import Acuity

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

NHAMCS_Daily_Arrivals: float = 144.0
NHAMCS_Hourly_Peak_Hour: int = 11
NHAMCS_Amplitude: float = 0.35
NHAMCS_ESI_Composition: dict[str, float] = {
    "esi_1_2": 0.13,
    "esi_3": 0.37,
    "esi_4_5": 0.50,
}

ACUITY_MAP: dict[str, Acuity] = {
    "esi_1_2": Acuity.CRITICAL,
    "esi_3": Acuity.URGENT,
    "esi_4_5": Acuity.ROUTINE,
}


def _poisson_nll(params: np.ndarray, hours: np.ndarray, counts: np.ndarray) -> float:
    log_base, amplitude, phase = params
    if amplitude <= -1 or amplitude >= 1:
        return 1e12
    rate = np.exp(log_base) * (1 + amplitude * np.sin(2 * np.pi * hours / 24 + phase))
    rate = np.maximum(rate, 1e-10)
    nll = -np.sum(counts * np.log(rate) - rate)
    if not np.isfinite(nll):
        return 1e12
    return nll


def _fit_sinusoidal_poisson(
    hours: np.ndarray, counts: np.ndarray
) -> tuple[float, float, float]:
    init = np.array([np.log(max(np.mean(counts), 0.1)), 0.3, 0.0])
    bounds = [
        (np.log(0.1), np.log(200.0)),
        (-0.99, 0.99),
        (-np.pi, np.pi),
    ]
    result = minimize(
        _poisson_nll,
        init,
        args=(hours, counts),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000},
    )
    log_base, amplitude, phase = result.x
    lambda_base = float(np.exp(log_base))
    return lambda_base, float(amplitude), float(phase)


def generate_synthetic_nhamcs(
    n_days: int = 365, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lambda_base_true = NHAMCS_Daily_Arrivals / 24.0
    amplitude_true = NHAMCS_Amplitude
    phase_true = -2 * np.pi * NHAMCS_Hourly_Peak_Hour / 24.0
    hours = np.arange(n_days * 24, dtype=float)
    rates = lambda_base_true * (
        1 + amplitude_true * np.sin(2 * np.pi * hours / 24 + phase_true)
    )
    counts = rng.poisson(rates)
    return hours, counts


def estimate_acuity_composition() -> dict[Acuity, float]:
    composition: dict[Acuity, float] = {}
    for esi_key, frac in NHAMCS_ESI_Composition.items():
        acuity = ACUITY_MAP[esi_key]
        composition[acuity] = composition.get(acuity, 0.0) + frac
    return composition


def calibrate(
    n_days: int = 365,
    seed: int = 42,
    source: str = "nhamcs",
    mimic_path: Optional[Path] = None,
) -> dict:
    if source == "mimic":
        try:
            from src.mimic_calibrator import calibrate_from_mimic

            params = calibrate_from_mimic(mimic_path)
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            out_path = DATA_DIR / "calibrated_params.json"
            with open(out_path, "w") as f:
                json.dump(params, f, indent=2, default=str)
            print(f"Calibrated parameters written to {out_path} (source: {params['source']})")
            return params
        except ImportError:
            print("MIMIC calibrator not available, falling back to NHAMCS")
            source = "nhamcs"

    hours, counts = generate_synthetic_nhamcs(n_days=n_days, seed=seed)
    lambda_base, amplitude, phase = _fit_sinusoidal_poisson(hours, counts)
    acuity_composition = estimate_acuity_composition()
    acuity_map = {k.value: v for k, v in acuity_composition.items()}

    calibrated_params = {
        "lambda_base": round(lambda_base, 4),
        "amplitude": round(amplitude, 4),
        "phase": round(phase, 4),
        "peak_hour": round((-phase * 24 / (2 * np.pi)) % 24, 2),
        "acuity_composition": acuity_map,
        "source": "NHAMCS 2020-2023 published rates (synthetic calibration)",
        "n_days": n_days,
        "mean_daily_arrivals": round(lambda_base * 24, 2),
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "calibrated_params.json"
    with open(out_path, "w") as f:
        json.dump(calibrated_params, f, indent=2)
    print(f"Calibrated parameters written to {out_path}")
    return calibrated_params


if __name__ == "__main__":
    calibrate()
