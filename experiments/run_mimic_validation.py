"""Multi-replication experiment using MIMIC-calibrated parameters.

Usage:
    python -m experiments.mimic_validation [--synthetic] [--reps 10] [--duration 48]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.environment import EDEnvironment
from src.mimic_calibrator import get_mimic_calibrated_params
from src.oracle import HindsightOracle
from src.patient import ACUITY_WEIGHTS
from src.policies import AdaptivePolicy, RobustPolicy

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DURATION = 48.0
B_MAX = 20
N_REPS = 10


def run_single_rep(
    seed: int,
    lambda_base: float,
    amplitude: float,
    phase: float,
    duration: float = DURATION,
    B_max: int = B_MAX,
) -> dict:
    """Run one replication comparing robust vs adaptive on MIMIC params."""
    env_r = EDEnvironment(
        B_max=B_max,
        lambda_base=lambda_base,
        seed=seed,
    )
    env_a = EDEnvironment(
        B_max=B_max,
        lambda_base=lambda_base,
        seed=seed,
    )

    # Override NHPP params with MIMIC-calibrated values
    env_r.amplitude = amplitude
    env_r.phase = phase
    env_a.amplitude = amplitude
    env_a.phase = phase

    robust = RobustPolicy(lambda_adv=lambda_base, B_max=B_max)
    adaptive = AdaptivePolicy(lambda_adv=lambda_base, B_max=B_max)

    pat_r = env_r.run(duration=duration, allocation_policy=robust)
    pat_a = env_a.run(duration=duration, allocation_policy=adaptive)

    oracle = HindsightOracle(B_max=B_max)
    V_star = oracle.compute_optimal(pat_r, total_time=duration)

    def compute_cost(patients) -> float:
        total = 0.0
        for p in patients:
            w = p.wait_time if p.service_start is not None else duration
            total += ACUITY_WEIGHTS[p.acuity] * w
        return total

    cost_r = compute_cost(pat_r)
    cost_a = compute_cost(pat_a)

    return {
        "robust_cost": cost_r,
        "adaptive_cost": cost_a,
        "V_star": V_star,
        "robust_regret": cost_r - V_star,
        "adaptive_regret": cost_a - V_star,
        "n_patients": len(pat_r),
    }


def main():
    parser = argparse.ArgumentParser(description="MIMIC validation experiment")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data instead of MIMIC")
    parser.add_argument("--reps", type=int, default=N_REPS,
                        help="Number of replications")
    parser.add_argument("--duration", type=float, default=DURATION,
                        help="Simulation duration (hours)")
    args = parser.parse_args()

    # Get MIMIC-calibrated parameters
    if args.synthetic:
        from src.mimic_calibrator import calibrate_from_mimic
        params = calibrate_from_mimic(use_synthetic=True, synthetic_seed=42)
    else:
        params = get_mimic_calibrated_params()

    lambda_base = params["lambda_base"]
    amplitude = params["amplitude"]
    phase = params["phase"]

    print("=== MIMIC Validation Experiment ===")
    print(f"Source: {params['source']}")
    print(f"λ_base = {lambda_base:.2f}, amplitude = {amplitude:.2f}, phase = {phase:.2f}")
    print(f"Acuity mix: {params['acuity_mix']}")
    print(f"Replications: {args.reps}, Duration: {args.duration}h")
    print()

    robust_regrets = []
    adaptive_regrets = []

    for rep in range(args.reps):
        seed = 42 + rep
        result = run_single_rep(
            seed=seed,
            lambda_base=lambda_base,
            amplitude=amplitude,
            phase=phase,
            duration=args.duration,
        )
        robust_regrets.append(result["robust_regret"])
        adaptive_regrets.append(result["adaptive_regret"])
        print(f"  Rep {rep+1:2d}/{args.reps}: robust={result['robust_regret']:.0f} "
              f"adaptive={result['adaptive_regret']:.0f} V*={result['V_star']:.0f}")

    rr = np.array(robust_regrets)
    ar = np.array(adaptive_regrets)

    def mean_ci(arr):
        m = float(np.mean(arr))
        se = float(np.std(arr, ddof=1)) / np.sqrt(len(arr))
        return m, 1.96 * se

    rm, rci = mean_ci(rr)
    am, aci = mean_ci(ar)

    print()
    print("=== Results ===")
    print(f"{'Metric':<25} {'Robust':<15} {'Adaptive':<15}")
    print("-" * 55)
    print(f"{'Mean Regret':<25} {rm:<15.0f} {am:<15.0f}")
    print(f"{'95% CI':<25} ±{rci:<13.0f} ±{aci:<13.0f}")
    improvement = rm - am
    pct = (improvement / rm * 100) if rm != 0 else 0
    print(f"{'Improvement':<25} {improvement:<15.0f} ({pct:.1f}%)")

    # Save CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"mimic_validation_{args.reps}rep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rep", "robust_regret", "adaptive_regret", "source"])
        for rep in range(args.reps):
            writer.writerow([rep, robust_regrets[rep], adaptive_regrets[rep], params["source"]])
        writer.writerow([])
        writer.writerow(["robust_mean", rm])
        writer.writerow(["robust_ci95", rci])
        writer.writerow(["adaptive_mean", am])
        writer.writerow(["adaptive_ci95", aci])
        writer.writerow(["improvement", improvement])
        writer.writerow(["improvement_pct", pct])

    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
