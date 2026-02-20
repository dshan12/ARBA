from pathlib import Path

import numpy as np
import pandas as pd

from src.environment import EDEnvironment
from src.oracle import HindsightOracle
from src.patient import ACUITY_WEIGHTS, Acuity
from src.policies import AdaptivePolicy, RobustPolicy

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

EPOCH_HOURS = 8
DURATION = 48.0
B_MAX = 20
N_REPS = 30
LAMBDA_BASE = 10.0

SCENARIOS = {
    "baseline": {
        "surge_start": None,
        "surge_mult": 1.0,
        "surge_duration": 0,
    },
    "flash_flood": {
        "surge_start": 24,
        "surge_mult": 3.0,
        "surge_duration": 12,
    },
    "creeping_crisis": {
        "surge_start": 12,
        "surge_mult": 2.5,
        "surge_duration": 36,
    },
    "acuity_flip": {
        "surge_start": 24,
        "surge_mult": 2.0,
        "surge_duration": 24,
        "acuity_composition": {
            Acuity.CRITICAL: 0.25,
            Acuity.URGENT: 0.45,
            Acuity.ROUTINE: 0.30,
        },
    },
}


def get_hourly_counts(patients, duration):
    n_hours = int(np.ceil(duration))
    counts = np.zeros(n_hours, dtype=int)
    for p in patients:
        h = int(p.arrival_time)
        if h < n_hours:
            counts[h] += 1
    return counts


def compute_epoch_costs(patients, duration, epoch_hours):
    n_epochs = int(np.ceil(duration / epoch_hours))
    costs = np.zeros(n_epochs)
    for p in patients:
        epoch = int(p.arrival_time // epoch_hours)
        if epoch < n_epochs:
            w = duration if p.service_start is None else p.wait_time
            costs[epoch] += ACUITY_WEIGHTS[p.acuity] * w
    return costs


def run_single(scenario_params, policy_type, seed):
    kwargs = dict(scenario_params)
    env = EDEnvironment(
        B_max=B_MAX,
        lambda_base=LAMBDA_BASE,
        seed=seed,
        **kwargs,
    )

    if policy_type == "robust":
        policy = RobustPolicy(lambda_adv=LAMBDA_BASE, B_max=B_MAX)
    else:
        policy = AdaptivePolicy(lambda_adv=LAMBDA_BASE, B_max=B_MAX, threshold=5.0)

    patients = env.run(duration=DURATION, allocation_policy=policy)
    return patients, policy, env


def compute_cusum_metrics(policy, patients, duration, surge_start):
    hourly = get_hourly_counts(patients, duration)
    trigger_hour = None
    for h, count in enumerate(hourly):
        if count > 0:
            policy.update([float(count)], current_time=float(h))
            if policy.trigger_times and trigger_hour is None:
                trigger_hour = float(h)

    detect_delay = np.nan
    false_alarm = np.nan
    if trigger_hour is not None:
        if surge_start is not None and surge_start > 0:
            if trigger_hour >= surge_start:
                detect_delay = trigger_hour - surge_start
            else:
                false_alarm = 1.0
        else:
            false_alarm = 1.0
    else:
        detect_delay = np.nan
        false_alarm = 0.0

    return detect_delay, false_alarm, trigger_hour


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    epoch_rows = []

    n_epochs = int(np.ceil(DURATION / EPOCH_HOURS))

    base_seed = 1000
    for scenario_name, scenario_params in SCENARIOS.items():
        surge_start = scenario_params.get("surge_start")
        print(f"  Scenario: {scenario_name}")

        for rep in range(N_REPS):
            seed = base_seed + rep

            pat_r, pol_r, env_r = run_single(scenario_params, "robust", seed)
            pat_a, pol_a, env_a = run_single(scenario_params, "adaptive", seed)

            oracle = HindsightOracle(B_max=B_MAX)
            V_star = oracle.compute_optimal(pat_r, total_time=DURATION)

            epoch_r = compute_epoch_costs(pat_r, DURATION, EPOCH_HOURS)
            epoch_a = compute_epoch_costs(pat_a, DURATION, EPOCH_HOURS)

            cost_r = float(np.sum(epoch_r))
            cost_a = float(np.sum(epoch_a))

            regret_r = cost_r - V_star if V_star != 0 else cost_r
            regret_a = cost_a - V_star if V_star != 0 else cost_a

            detect_delay_a, false_alarm_a, _ = compute_cusum_metrics(
                pol_a, pat_a, DURATION, surge_start
            )

            summary_rows.append({
                "scenario": scenario_name,
                "policy": "robust",
                "rep": rep,
                "total_cost": round(cost_r, 2),
                "V_star": round(V_star, 2),
                "regret": round(regret_r, 2),
                "detect_delay": np.nan,
                "false_alarm": np.nan,
                "n_patients": len(pat_r),
            })
            summary_rows.append({
                "scenario": scenario_name,
                "policy": "adaptive",
                "rep": rep,
                "total_cost": round(cost_a, 2),
                "V_star": round(V_star, 2),
                "regret": round(regret_a, 2),
                "detect_delay": (
                    round(detect_delay_a, 2) if not np.isnan(detect_delay_a) else np.nan
                ),
                "false_alarm": (
                    round(false_alarm_a, 2) if not np.isnan(false_alarm_a) else np.nan
                ),
                "n_patients": len(pat_a),
            })

            for e in range(n_epochs):
                cum_regret_r = float(np.sum(epoch_r[: e + 1])) - V_star
                cum_regret_a = float(np.sum(epoch_a[: e + 1])) - V_star
                epoch_rows.append({
                    "scenario": scenario_name,
                    "policy": "robust",
                    "rep": rep,
                    "epoch": e,
                    "epoch_cost": round(float(epoch_r[e]), 2),
                    "cumulative_regret": (
                        round(cum_regret_r, 2) if V_star != 0
                        else round(float(np.sum(epoch_r[: e + 1])), 2)
                    ),
                })
                epoch_rows.append({
                    "scenario": scenario_name,
                    "policy": "adaptive",
                    "rep": rep,
                    "epoch": e,
                    "epoch_cost": round(float(epoch_a[e]), 2),
                    "cumulative_regret": (
                        round(cum_regret_a, 2) if V_star != 0
                        else round(float(np.sum(epoch_a[: e + 1])), 2)
                    ),
                })

    df_summary = pd.DataFrame(summary_rows)
    df_epoch = pd.DataFrame(epoch_rows)

    df_summary.to_csv(RESULTS_DIR / "multi_rep_summary.csv", index=False)
    df_epoch.to_csv(RESULTS_DIR / "multi_rep_epoch_detail.csv", index=False)

    print(f"Wrote {len(df_summary)} summary rows, {len(epoch_rows)} epoch rows")
    print(df_summary.groupby(["scenario", "policy"]).agg({
        "total_cost": ["mean", "std"],
        "regret": ["mean", "std"],
        "detect_delay": "mean",
        "false_alarm": "mean",
    }).round(2))

    # --- LaTeX-ready results table ---
    latex_rows = []
    for scenario_name in SCENARIOS:
        sub = df_summary[df_summary["scenario"] == scenario_name]
        for policy in ["robust", "adaptive"]:
            pol = sub[sub["policy"] == policy]
            cost_mean = pol["total_cost"].mean()
            cost_std = pol["total_cost"].std()
            regret_mean = pol["regret"].mean()
            regret_std = pol["regret"].std()
            detect = pol["detect_delay"].mean()
            latex_rows.append(
                f"  {scenario_name} & {policy} "
                f"& ${cost_mean:.1f} \\pm {cost_std:.1f}$ "
                f"& ${regret_mean:.1f} \\pm {regret_std:.1f}$ "
                f"& {detect:.1f} \\\\"
            )
    latex_table = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Multi-replication results across scenarios.}\n"
        "\\label{tab:multi_rep}\n"
        "\\begin{tabular}{lcccr}\n"
        "\\toprule\n"
        "Scenario & Policy & Total Cost & Regret & Detect Delay (h) \\\\\n"
        "\\midrule\n"
        + "\n".join(latex_rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )
    print("\nLaTeX-ready table:\n")
    print(latex_table)

    tex_path = RESULTS_DIR / "multi_rep_table.tex"
    with open(tex_path, "w") as f:
        f.write(latex_table)
    print(f"Wrote LaTeX table to {tex_path}")


if __name__ == "__main__":
    main()
