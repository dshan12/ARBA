from pathlib import Path

import numpy as np
import pandas as pd

from src.cusum import CUSUMModule

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

LAMBDA_0 = 10.0
N_H = 20
N_L1 = 15
H_MIN, H_MAX = 1.0, 20.0
L1_MIN = 1.1 * LAMBDA_0
L1_MAX = 3.0 * LAMBDA_0
N_SIM = 500
MAX_STEPS = 500


def simulate_arl(cusum, lam, n_sim, max_steps, rng):
    run_lengths = np.zeros(n_sim)
    for i in range(n_sim):
        S = 0.0
        steps = 0
        while steps < max_steps:
            x = rng.poisson(lam)
            S = max(
                0.0,
                S
                + x * np.log(cusum.lambda_1 / cusum.lambda_0)
                - (cusum.lambda_1 - cusum.lambda_0),
            )
            steps += 1
            if S >= cusum.threshold:
                break
        run_lengths[i] = steps if steps < max_steps else max_steps
    return float(np.mean(run_lengths))


def compute_pareto_frontier(points):
    sorted_points = points.sort_values("arl_0", ascending=False)
    frontier = []
    best_arl1 = float("inf")
    for _, row in sorted_points.iterrows():
        if row["arl_1"] < best_arl1:
            frontier.append(row)
            best_arl1 = row["arl_1"]
    return pd.DataFrame(frontier)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    h_vals = np.linspace(H_MIN, H_MAX, N_H)
    l1_vals = np.linspace(L1_MIN, L1_MAX, N_L1)

    rows = []
    for i, H in enumerate(h_vals):
        for l1 in l1_vals:
            cusum = CUSUMModule(lambda_0=LAMBDA_0, lambda_1=l1, threshold=H)

            arl_0 = simulate_arl(cusum, LAMBDA_0, N_SIM, MAX_STEPS, rng)
            arl_1 = simulate_arl(cusum, l1, N_SIM, MAX_STEPS, rng)

            rows.append({
                "H": round(H, 2),
                "lambda_1": round(l1, 2),
                "arl_0": round(arl_0, 2),
                "arl_1": round(arl_1, 2),
            })

        print(f"  H={H:.1f} done ({i+1}/{N_H})")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "calibration_full.csv", index=False)

    frontier = compute_pareto_frontier(df)
    frontier.to_csv(RESULTS_DIR / "calibration_frontier.csv", index=False)

    print(f"Wrote {len(df)} grid points to calibration_full.csv")
    print(f"Wrote {len(frontier)} Pareto-optimal points to calibration_frontier.csv")
    print("\nPareto frontier:")
    print(frontier.to_string(index=False))


if __name__ == "__main__":
    main()
