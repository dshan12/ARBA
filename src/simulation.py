import csv
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.environment import EDEnvironment
from src.oracle import HindsightOracle
from src.patient import ACUITY_WEIGHTS
from src.policies import AdaptivePolicy, AllocationPolicy, RobustPolicy


class SimulationRunner:
    def __init__(self, B_max: int = 20, duration: float = 168.0, seed: int = 42):
        self.B_max = B_max
        self.duration = duration
        self.seed = seed

    def run_scenario(
        self,
        lambda_base: float,
        surge_start: int | None = None,
        surge_mult: float = 3.0,
        surge_duration: int = 20,
        policy_type: str = "robust",
        scenario: str | None = None,
        scenario_params: dict | None = None,
    ):
        env = EDEnvironment(
            B_max=self.B_max,
            lambda_base=lambda_base,
            surge_start=surge_start,
            surge_mult=surge_mult,
            surge_duration=surge_duration,
            seed=self.seed,
        )

        if scenario is not None:
            env.configure_scenario(scenario, scenario_params or {})

        policy: AllocationPolicy
        if policy_type == "robust":
            policy = RobustPolicy(lambda_adv=lambda_base, B_max=self.B_max)
        else:
            policy = AdaptivePolicy(lambda_adv=lambda_base, B_max=self.B_max)

        patients = env.run(self.duration, allocation_policy=policy)
        metrics = self._compute_metrics(patients, total_time=self.duration)

        oracle = HindsightOracle(B_max=self.B_max)
        V_star = oracle.compute_optimal(patients, total_time=self.duration)
        realized_cost = metrics["total_cost"]
        regret = realized_cost - V_star

        metrics["V_star"] = V_star
        metrics["regret"] = regret
        return metrics

    def _compute_metrics(self, patients, total_time: float = 0.0):
        if not patients:
            return {"avg_wait": 0, "max_wait": 0, "total_cost": 0}

        total_cost = 0.0
        waits_list = []
        for p in patients:
            if p.service_start is not None:
                w = p.wait_time
            else:
                w = total_time
            waits_list.append(w)
            total_cost += ACUITY_WEIGHTS[p.acuity] * w

        waits = np.array(waits_list)
        return {
            "avg_wait": float(np.mean(waits)),
            "max_wait": float(np.max(waits)),
            "p95_wait": float(np.percentile(waits, 95)),
            "total_cost": total_cost,
            "n_patients": len(patients),
        }

    def compare_policies(
        self,
        lambda_base: float,
        surge_start: int = 100,
        surge_mult: float = 3.0,
    ):
        robust_metrics = self.run_scenario(
            lambda_base=lambda_base,
            surge_start=surge_start,
            surge_mult=surge_mult,
            policy_type="robust",
        )
        adaptive_metrics = self.run_scenario(
            lambda_base=lambda_base,
            surge_start=surge_start,
            surge_mult=surge_mult,
            policy_type="adaptive",
        )
        return {
            "robust": {
                "avg_wait": robust_metrics["avg_wait"],
                "max_wait": robust_metrics["max_wait"],
                "p95_wait": robust_metrics["p95_wait"],
                "total_cost": robust_metrics["total_cost"],
                "n_patients": robust_metrics["n_patients"],
                "V_star": robust_metrics["V_star"],
                "regret": robust_metrics["regret"],
            },
            "adaptive": {
                "avg_wait": adaptive_metrics["avg_wait"],
                "max_wait": adaptive_metrics["max_wait"],
                "p95_wait": adaptive_metrics["p95_wait"],
                "total_cost": adaptive_metrics["total_cost"],
                "n_patients": adaptive_metrics["n_patients"],
                "V_star": adaptive_metrics["V_star"],
                "regret": adaptive_metrics["regret"],
            },
            "regret_comparison": {
                "robust_regret": robust_metrics["regret"],
                "adaptive_regret": adaptive_metrics["regret"],
                "improvement": robust_metrics["regret"] - adaptive_metrics["regret"],
            },
        }


def run_stress_test(
    scenario_name: str,
    lambda_base: float = 10.0,
    n_reps: int = 10,
    seeds: list[int] | None = None,
    scenario_params: dict | None = None,
    duration: float = 168.0,
    B_max: int = 20,
) -> dict:
    if seeds is None:
        seeds = [42 + i for i in range(n_reps)]

    results: dict[str, list[float]] = {"robust": [], "adaptive": []}

    for rep in range(n_reps):
        for policy_type in ("robust", "adaptive"):
            runner = SimulationRunner(
                B_max=B_max, duration=duration, seed=seeds[rep]
            )
            metrics = runner.run_scenario(
                lambda_base=lambda_base,
                policy_type=policy_type,
                scenario=scenario_name,
                scenario_params=scenario_params,
            )
            results[policy_type].append(metrics["regret"])

    out: dict[str, Any] = {}
    for pt in ("robust", "adaptive"):
        arr = np.array(results[pt])
        mean = float(np.mean(arr))
        se = float(np.std(arr, ddof=1)) / np.sqrt(len(arr))
        ci = 1.96 * se
        out[pt] = {
            "mean_regret": mean,
            "ci_95": ci,
            "lower": mean - ci,
            "upper": mean + ci,
            "values": results[pt],
        }

    results_dir = Path(__file__).resolve().parent.parent / "results"
    os.makedirs(str(results_dir), exist_ok=True)
    csv_path = results_dir / f"stress_{scenario_name}.csv"
    with open(str(csv_path), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rep", "robust_regret", "adaptive_regret"])
        for rep in range(n_reps):
            writer.writerow([rep, results["robust"][rep], results["adaptive"][rep]])
        writer.writerow([])
        for pt in ("robust", "adaptive"):
            writer.writerow([f"{pt}_mean", out[pt]["mean_regret"]])
            writer.writerow([f"{pt}_ci95", out[pt]["ci_95"]])

    out["scenario"] = scenario_name
    out["n_reps"] = n_reps
    return out
