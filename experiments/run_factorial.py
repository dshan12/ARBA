import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from src.environment import EDEnvironment
from src.oracle import HindsightOracle
from src.patient import ACUITY_WEIGHTS, Acuity
from src.policies import AdaptivePolicy, RobustPolicy

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

DURATION = 48.0
N_REPS = 10

LAMBDA_BASE_VALS = [5, 10, 20]
SURGE_MULT_VALS = [2, 3, 5]
SCENARIO_KEYS = ["flash_flood", "creeping_crisis", "acuity_flip"]
H_VALS = [3, 5, 8]
B_MAX_VALS = [10, 20, 30]

SCENARIO_PARAMS = {
    "flash_flood": {
        "surge_start": 24,
        "surge_duration": 12,
    },
    "creeping_crisis": {
        "surge_start": 12,
        "surge_duration": 36,
    },
    "acuity_flip": {
        "surge_start": 24,
        "surge_duration": 24,
        "acuity_composition": {
            Acuity.CRITICAL: 0.25,
            Acuity.URGENT: 0.45,
            Acuity.ROUTINE: 0.30,
        },
    },
}


def run_config(lambda_base, surge_mult, scenario_key, H, B_max, rep, seed):
    scenario = SCENARIO_PARAMS[scenario_key]

    env_r = EDEnvironment(
        B_max=B_max,
        lambda_base=lambda_base,
        surge_mult=surge_mult,
        seed=seed,
        **scenario,
    )
    env_a = EDEnvironment(
        B_max=B_max,
        lambda_base=lambda_base,
        surge_mult=surge_mult,
        seed=seed,
        **scenario,
    )

    robust_policy = RobustPolicy(lambda_adv=lambda_base, B_max=B_max)
    adaptive_policy = AdaptivePolicy(
        lambda_adv=lambda_base, B_max=B_max, threshold=H
    )

    pat_r = env_r.run(duration=DURATION, allocation_policy=robust_policy)
    pat_a = env_a.run(duration=DURATION, allocation_policy=adaptive_policy)

    cost_r = _total_weighted_wait(pat_r, DURATION)
    cost_a = _total_weighted_wait(pat_a, DURATION)

    oracle = HindsightOracle(B_max=B_max)
    V_star = oracle.compute_optimal(pat_r, total_time=DURATION)

    regret_r = cost_r - V_star
    regret_a = cost_a - V_star

    return {
        "lambda_base": lambda_base,
        "surge_mult": surge_mult,
        "scenario": scenario_key,
        "H": H,
        "B_max": B_max,
        "rep": rep,
        "V_star": round(V_star, 2),
        "cost_robust": round(cost_r, 2),
        "cost_adaptive": round(cost_a, 2),
        "regret_robust": round(regret_r, 2),
        "regret_adaptive": round(regret_a, 2),
        "n_patients_robust": len(pat_r),
        "n_patients_adaptive": len(pat_a),
        "avg_wait_robust": round(
            float(np.mean([p.wait_time for p in pat_r if p.service_start is not None])), 4
        ),
        "avg_wait_adaptive": round(
            float(np.mean([p.wait_time for p in pat_a if p.service_start is not None])), 4
        ),
        "p95_wait_robust": round(
            float(np.percentile([p.wait_time for p in pat_r if p.service_start is not None], 95)), 4
        ),
        "p95_wait_adaptive": round(
            float(np.percentile([p.wait_time for p in pat_a if p.service_start is not None], 95)), 4
        ),
    }


def _total_weighted_wait(patients, total_time: float):
    total = 0.0
    for p in patients:
        w = total_time if p.service_start is None else p.wait_time
        total += ACUITY_WEIGHTS[p.acuity] * w
    return total


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    factors = list(itertools.product(
        LAMBDA_BASE_VALS, SURGE_MULT_VALS, SCENARIO_KEYS, H_VALS, B_MAX_VALS
    ))
    total = len(factors)
    print(f"Running {total} configurations × {N_REPS} reps = {total * N_REPS} simulations")

    rows = []
    idx = 0
    for lambda_base, surge_mult, scenario_key, H, B_max in factors:
        for rep in range(N_REPS):
            seed = 2000 + idx * 100 + rep
            row = run_config(lambda_base, surge_mult, scenario_key, H, B_max, rep, seed)
            rows.append(row)
        idx += 1
        if idx % 10 == 0:
            print(f"  {idx}/{total} configurations done")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "factorial_results.csv", index=False)
    print(f"Wrote {len(df)} rows to factorial_results.csv")
    print(df.groupby(["scenario", "H"]).agg({
        "cost_robust": "mean",
        "cost_adaptive": "mean",
        "avg_wait_robust": "mean",
        "avg_wait_adaptive": "mean",
    }).round(2))


if __name__ == "__main__":
    main()
