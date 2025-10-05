def compute_regret(V_star: float, realized_cost: float) -> float:
    return realized_cost - V_star


def compute_cumulative_regret(
    episode_costs: list[float], V_star: float
) -> list[float]:
    return [compute_regret(V_star, cost) for cost in episode_costs]
