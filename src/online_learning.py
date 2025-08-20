from typing import Any

import numpy as np

from src.patient import ACUITY_WEIGHTS, Acuity
from src.policies import AllocationPolicy


class OCOPolicy(AllocationPolicy):
    """EXPERIMENTAL: Online Convex Optimization policy for post-detection rate estimation.

    Uses online gradient descent to estimate the arrival rate with O(log T) regret
    (Hazan et al. 2007). Currently NOT integrated into the simulation pipeline.
    The AdaptivePolicy uses sliding-window MLE + exponential smoothing instead.
    This implementation is kept as a reference for future work on Theorem 2.
    """

    def __init__(self, lambda_0: float, lambda_max: float, B_max: int):
        self.lambda_hat = lambda_0
        self.lambda_max = lambda_max
        self.B_max = B_max
        self.t = 0

    def update(self, actual_arrivals: list[float]):
        for x in actual_arrivals:
            self.t += 1
            gradient = 2.0 * (self.lambda_hat - x)
            step_size = 0.15 / max(0.1, self.t**0.5)
            self.lambda_hat = self.lambda_hat - step_size * gradient
            self.lambda_hat = float(np.clip(self.lambda_hat, 0.0, self.lambda_max * 2))

    def get_action(self, state: dict[str, Any]) -> dict[str, Any]:
        B_occ = state["B_occ"]
        q_crit = state.get("q_crit", 0)
        q_urg = state.get("q_urg", 0)
        q_rout = state.get("q_rout", 0)
        available = max(0, self.B_max - B_occ)

        if available <= 0:
            return {"admit_critical": 0, "admit_urgent": 0, "admit_routine": 0}

        weights = np.array([
            ACUITY_WEIGHTS[Acuity.CRITICAL],
            ACUITY_WEIGHTS[Acuity.URGENT],
            ACUITY_WEIGHTS[Acuity.ROUTINE],
        ])
        queues = np.array([q_crit, q_urg, q_rout], dtype=float)

        logits = weights * self.lambda_hat
        exp_logits = np.exp(logits - logits.max())
        proportions = exp_logits / exp_logits.sum()
        admit = np.floor(proportions * available).astype(int)
        admit = np.minimum(admit, queues.astype(int))

        remaining = available - int(admit.sum())
        if remaining > 0:
            order = np.argsort(-weights)
            for i in order:
                deficit = int(queues[i]) - int(admit[i])
                extra = min(remaining, max(0, deficit))
                admit[i] += extra
                remaining -= extra
                if remaining <= 0:
                    break

        return {
            "admit_critical": int(admit[0]),
            "admit_urgent": int(admit[1]),
            "admit_routine": int(admit[2]),
        }
