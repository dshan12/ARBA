import argparse
import json

from src.simulation import SimulationRunner


def main():
    parser = argparse.ArgumentParser(description="ARBA Simulation")
    parser.add_argument("--B_max", type=int, default=20, help="Number of beds")
    parser.add_argument(
        "--lambda_base", type=float, default=10.0, help="Base arrival rate"
    )
    parser.add_argument("--duration", type=float, default=168.0, help="Simulation hours")
    parser.add_argument(
        "--surge_start", type=int, default=100, help="Hour to start surge"
    )
    parser.add_argument(
        "--surge_mult", type=float, default=3.0, help="Surge multiplier"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--policy",
        type=str,
        choices=["robust", "adaptive", "compare"],
        default="compare",
    )
    args = parser.parse_args()

    runner = SimulationRunner(
        B_max=args.B_max, duration=args.duration, seed=args.seed
    )

    if args.policy == "compare":
        results = runner.compare_policies(
            lambda_base=args.lambda_base,
            surge_start=args.surge_start,
            surge_mult=args.surge_mult,
        )
        robust = results["robust"]
        adaptive = results["adaptive"]
        rc = results["regret_comparison"]
        print("=== ARBA Simulation Results ===")
        print(f"{'Metric':<20} {'Robust':<15} {'Adaptive':<15}")
        print("-" * 50)
        print(f"{'Avg Wait':<20} {robust['avg_wait']:<15.3f} {adaptive['avg_wait']:<15.3f}")
        print(f"{'Max Wait':<20} {robust['max_wait']:<15.3f} {adaptive['max_wait']:<15.3f}")
        print(f"{'P95 Wait':<20} {robust['p95_wait']:<15.3f} {adaptive['p95_wait']:<15.3f}")
        print(f"{'Total Cost':<20} {robust['total_cost']:<15.3f} {adaptive['total_cost']:<15.3f}")
        print(f"{'V* (Oracle)':<20} {robust['V_star']:<15.3f} {adaptive['V_star']:<15.3f}")
        print(f"{'Regret':<20} {robust['regret']:<15.3f} {adaptive['regret']:<15.3f}")
        print(f"{'n Patients':<20} {robust['n_patients']:<15} {adaptive['n_patients']:<15}")
        print()
        print(f"Regret Improvement (robust - adaptive): {rc['improvement']:.3f}")
    else:
        metrics = runner.run_scenario(
            lambda_base=args.lambda_base,
            surge_start=args.surge_start,
            surge_mult=args.surge_mult,
            policy_type=args.policy,
        )
        print("=== ARBA Simulation Results ===")
        print(f"{'Metric':<20} {'Value':<15}")
        print("-" * 35)
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k:<20} {v:<15.3f}")
            else:
                print(f"{k:<20} {v:<15}")


if __name__ == "__main__":
    main()
