"""Load and preprocess MIMIC-IV data for ARBA calibration.

Handles two data formats:
  1. Full MIMIC-IV-ED: edstays.csv + triage.csv (ESI acuity)
  2. Core MIMIC-IV demo: transfers.csv + admissions.csv + patients.csv (admission_type proxy)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.patient import Acuity

MIMIC_DIR = Path(__file__).resolve().parent.parent / "data" / "mimic"

# Admission type → acuity mapping (for core demo format)
ADMIT_TYPE_ACUITY: dict[str, Acuity] = {
    "EW EMER.": Acuity.CRITICAL,
    "DIRECT EMER.": Acuity.URGENT,
    "URGENT": Acuity.URGENT,
    "EU OBSERVATION": Acuity.URGENT,
    "OBSERVATION ADMIT": Acuity.ROUTINE,
    "SURGICAL SAME DAY ADMISSION": Acuity.ROUTINE,
    "ELECTIVE": Acuity.ROUTINE,
    "DIRECT OBSERVATION": Acuity.ROUTINE,
    "AMBULATORY OBSERVATION": Acuity.ROUTINE,
}

# ESI → acuity mapping (for full MIMIC-IV-ED format)
ESI_ACUITY: dict[int, Acuity] = {
    1: Acuity.CRITICAL,
    2: Acuity.CRITICAL,
    3: Acuity.URGENT,
    4: Acuity.ROUTINE,
    5: Acuity.ROUTINE,
}


@dataclass
class MIMICData:
    """Container for loaded and processed MIMIC data."""

    arrival_hours: np.ndarray  # hour-of-day for each arrival (0-23)
    service_hours: np.ndarray  # LOS in hours for each arrival
    acuities: list[Acuity]  # acuity for each arrival
    total_arrivals: int
    n_days: float  # number of unique days in the data
    source: str = "unknown"


def find_mimic_path() -> Optional[Path]:
    """Find the MIMIC data directory, checking common locations."""
    candidates = [
        MIMIC_DIR / "edstays.csv",  # full MIMIC-IV-ED
        MIMIC_DIR / "edstays.csv.gz",  # compressed
        MIMIC_DIR / "triage.csv",
        MIMIC_DIR / "triage.csv.gz",
    ]
    for c in candidates:
        if c.exists():
            return MIMIC_DIR

    # Check for demo subdirectory
    for sub in MIMIC_DIR.iterdir():
        if sub.is_dir() and (sub / "hosp" / "transfers.csv.gz").exists():
            return sub

    return None


def check_mimic_available() -> bool:
    """Check if MIMIC data is available (any format)."""
    return find_mimic_path() is not None


def _load_triage(mimic_path: Path) -> pd.DataFrame:
    """Load triage.csv or triage.csv.gz."""
    for name in ("triage.csv", "triage.csv.gz"):
        p = mimic_path / name
        if p.exists():
            return pd.read_csv(p, low_memory=False)
    raise FileNotFoundError(f"No triage file found in {mimic_path}")


def _load_edstays(mimic_path: Path) -> pd.DataFrame:
    """Load edstays.csv or edstays.csv.gz."""
    for name in ("edstays.csv", "edstays.csv.gz"):
        p = mimic_path / name
        if p.exists():
            return pd.read_csv(p, parse_dates=["intime", "outtime"], low_memory=False)
    raise FileNotFoundError(f"No edstays file found in {mimic_path}")


def _load_transfers(mimic_path: Path) -> pd.DataFrame:
    """Load transfers.csv.gz from demo subdirectory."""
    p = mimic_path / "hosp" / "transfers.csv.gz"
    if not p.exists():
        raise FileNotFoundError(f"No transfers file found in {mimic_path}/hosp/")
    return pd.read_csv(p, parse_dates=["intime", "outtime"], low_memory=False)


def _load_admissions(mimic_path: Path) -> pd.DataFrame:
    """Load admissions.csv.gz from demo subdirectory."""
    p = mimic_path / "hosp" / "admissions.csv.gz"
    if not p.exists():
        raise FileNotFoundError(f"No admissions file found in {mimic_path}/hosp/")
    return pd.read_csv(p, parse_dates=["admittime", "edregtime", "edouttime"], low_memory=False)


def load_mimic_ed(mimic_path: Optional[Path] = None) -> MIMICData:
    """Load MIMIC data, auto-detecting format (full ED or demo).

    Args:
        mimic_path: Path to MIMIC data directory. If None, auto-detect.

    Returns:
        MIMICData with arrival hours, service times, and acuities.
    """
    if mimic_path is None:
        found = find_mimic_path()
        if found is None:
            raise FileNotFoundError(
                "No MIMIC data found. Place edstays.csv + triage.csv in "
                "data/mimic/, or use the demo with transfers.csv.gz."
            )
        mimic_path = found

    # Try full MIMIC-IV-ED format first
    ed_file = mimic_path / "edstays.csv"
    ed_file_gz = mimic_path / "edstays.csv.gz"
    triage_file = mimic_path / "triage.csv"
    triage_file_gz = mimic_path / "triage.csv.gz"

    has_full_ed = (ed_file.exists() or ed_file_gz.exists()) and (
        triage_file.exists() or triage_file_gz.exists()
    )

    if has_full_ed:
        return _load_full_ed(mimic_path)

    # Fall back to demo/core format
    return _load_demo(mimic_path)


def _load_full_ed(mimic_path: Path) -> MIMICData:
    """Load full MIMIC-IV-ED format (edstays + triage with ESI)."""
    edstays = _load_edstays(mimic_path)
    triage = _load_triage(mimic_path)

    # Merge triage acuity onto stays
    merged = edstays.merge(triage[["stay_id", "acuity"]], on="stay_id", how="left")
    merged = merged.dropna(subset=["acuity"])
    merged["acuity"] = merged["acuity"].astype(int)
    merged["acuity_label"] = merged["acuity"].map(ESI_ACUITY)
    merged = merged.dropna(subset=["acuity_label"])

    merged["arrival_hour"] = merged["intime"].dt.hour
    merged["los_hours"] = (merged["outtime"] - merged["intime"]).dt.total_seconds() / 3600
    merged = merged[merged["los_hours"] > 0]

    n_days = merged["intime"].dt.date.nunique()
    return MIMICData(
        arrival_hours=merged["arrival_hour"].to_numpy(dtype=float),
        service_hours=merged["los_hours"].to_numpy(dtype=float),
        acuities=merged["acuity_label"].tolist(),
        total_arrivals=len(merged),
        n_days=float(n_days),
        source="MIMIC-IV-ED (full)",
    )


def _load_demo(mimic_path: Path) -> MIMICData:
    """Load core MIMIC demo format (transfers + admissions)."""
    transfers = _load_transfers(mimic_path)
    admissions = _load_admissions(mimic_path)

    # Get ED stays from transfers
    ed = transfers[transfers["eventtype"] == "ED"].copy()
    if ed.empty:
        raise ValueError("No ED records found in transfers table")

    ed["arrival_hour"] = ed["intime"].dt.hour
    ed["los_hours"] = (ed["outtime"] - ed["intime"]).dt.total_seconds() / 3600
    ed = ed[ed["los_hours"] > 0]

    # Map acuity from admission_type using admissions table
    adm_map = admissions.set_index("hadm_id")["admission_type"].to_dict()
    ed["admission_type"] = ed["hadm_id"].map(adm_map)
    ed["acuity"] = ed["admission_type"].map(ADMIT_TYPE_ACUITY)
    # Default to ROUTINE if no mapping found
    ed["acuity"] = ed["acuity"].fillna(Acuity.ROUTINE)

    n_days = ed["intime"].dt.date.nunique()
    return MIMICData(
        arrival_hours=ed["arrival_hour"].to_numpy(dtype=float),
        service_hours=ed["los_hours"].to_numpy(dtype=float),
        acuities=ed["acuity"].tolist(),
        total_arrivals=len(ed),
        n_days=float(n_days),
        source="MIMIC-IV (demo)",
    )


def generate_synthetic_ed_data(
    n_patients: int = 5000,
    lambda_base: float = 10.0,
    amplitude: float = 0.35,
    phase: float = np.pi / 2,
    seed: int = 42,
    acuity_weights: Optional[dict[Acuity, float]] = None,
) -> MIMICData:
    """Generate synthetic ED data with known parameters for testing.

    Args:
        n_patients: Number of synthetic patients.
        lambda_base: Base hourly arrival rate.
        amplitude: Sinusoidal amplitude.
        phase: Sinusoidal phase.
        seed: Random seed.
        acuity_weights: Dict mapping Acuity -> proportion (must sum to 1).

    Returns:
        MIMICData with generated data.
    """
    rng = np.random.default_rng(seed)

    if acuity_weights is None:
        acuity_weights = {
            Acuity.CRITICAL: 0.13,
            Acuity.URGENT: 0.37,
            Acuity.ROUTINE: 0.50,
        }

    # Generate arrival times across N days
    n_days = max(1, int(n_patients / lambda_base / 24))
    total_hours = n_days * 24
    arrival_times = rng.exponential(scale=1.0 / lambda_base, size=n_patients).cumsum()
    arrival_times = arrival_times[arrival_times < total_hours]
    n_actual = len(arrival_times)

    arrival_hours = arrival_times % 24

    # Apply sinusoidal modulation via rejection sampling
    acceptance_prob = 1.0 / (1 + amplitude * np.sin(2 * np.pi * arrival_hours / 24 + phase))
    accepted = rng.uniform(size=n_actual) < acceptance_prob / acceptance_prob.max()
    arrival_hours = arrival_hours[accepted]
    arrival_times = arrival_times[accepted]

    # Generate service times per acuity
    acuity_list = list(acuity_weights.keys())
    acuity_probs = [acuity_weights[a] for a in acuity_list]
    acuities = rng.choice(
        np.array(acuity_list, dtype=object),
        size=len(arrival_hours),
        p=acuity_probs,
    )

    service_params = {
        Acuity.CRITICAL: {"mu": 3.0, "sigma": 1.0},
        Acuity.URGENT: {"mu": 2.0, "sigma": 0.75},
        Acuity.ROUTINE: {"mu": 1.0, "sigma": 0.5},
    }
    service_hours = np.array([
        np.exp(rng.normal(service_params[a]["mu"], service_params[a]["sigma"]))
        for a in acuities
    ])
    service_hours = np.clip(service_hours, 0.1, 168)

    return MIMICData(
        arrival_hours=np.array(arrival_hours, dtype=float),
        service_hours=np.array(service_hours, dtype=float),
        acuities=list(acuities),
        total_arrivals=len(arrival_hours),
        n_days=float(n_days),
        source="synthetic",
    )
