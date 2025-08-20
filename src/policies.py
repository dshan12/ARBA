from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.cusum import CUSUMModule


class AllocationPolicy(ABC):
    @abstractmethod
    def get_action(self, state: dict[str, Any]) -> dict[str, Any]: ...

    def update(self, actual_arrivals: list[float]):
        pass


class RobustPolicy(AllocationPolicy):
    def __init__(self, lambda_adv: float, B_max: int, gamma: float = 0.2):
        if gamma < 0.0 or gamma >= 1.0:
            raise ValueError(f"gamma must be in [0, 1), got {gamma}")
        if B_max <= 0:
            raise ValueError(f"B_max must be > 0, got {B_max}")
        if lambda_adv < 0:
            raise ValueError(f"lambda_adv must be >= 0, got {lambda_adv}")
        self.lambda_adv = lambda_adv
        self.B_max = B_max
        self.gamma = gamma

    def get_action(self, state: dict[str, Any]) -> dict[str, Any]:
        B_occ = state["B_occ"]
        q_crit = state.get("q_crit", 0)
        q_urg = state.get("q_urg", 0)
        q_rout = state.get("q_rout", 0)
        available = max(0, self.B_max - B_occ)

        reserved = int(np.ceil(self.gamma * available))
        non_reserved = available - reserved

        admit_crit = min(q_crit, non_reserved)
        remaining = non_reserved - admit_crit
        admit_urg = min(q_urg, remaining)
        remaining -= admit_urg
        admit_rout = min(q_rout, remaining)

        return {
            "admit_critical": int(admit_crit),
            "admit_urgent": int(admit_urg),
            "admit_routine": int(admit_rout),
        }


class AdaptivePolicy(AllocationPolicy):
    def __init__(
        self,
        lambda_adv: float,
        B_max: int,
        delta: float = 0.5,
        threshold: float = 5.0,
        fast_window: int = 6,
        expansion_factor: float = 1.5,
        expansion_duration: int = 4,
        gamma: float = 0.2,
    ):
        self.lambda_base = lambda_adv
        self.lambda_current = lambda_adv
        self.B_max = B_max
        self.gamma = gamma
        self.fast_window = fast_window
        self.expansion_factor = expansion_factor
        self.expansion_duration = expansion_duration
        self.expansion_remaining = 0

        self.cusum = CUSUMModule(
            lambda_0=lambda_adv,
            lambda_1=lambda_adv + 2.0 * delta,
            threshold=threshold,
        )
        self.arrival_history: list[float] = []
        self.adaptive_mode = False
        self.trigger_times: list[float] = []
        self.surge_start_time: float | None = None

    @property
    def detection_delay(self) -> float | None:
        if not self.trigger_times or self.surge_start_time is None:
            return None
        delay = self.trigger_times[0] - self.surge_start_time
        return delay if delay >= 0 else None

    @property
    def false_alarm(self) -> bool:
        if not self.trigger_times or self.surge_start_time is None:
            return False
        return self.trigger_times[0] < self.surge_start_time

    def get_action(self, state: dict[str, Any]) -> dict[str, Any]:
        B_occ = state["B_occ"]
        q_crit = state.get("q_crit", 0)
        q_urg = state.get("q_urg", 0)
        q_rout = state.get("q_rout", 0)
        available = max(0, self.B_max - B_occ)

        if self.adaptive_mode and self.lambda_current > self.lambda_base:
            surge_ratio = self.lambda_current / max(self.lambda_base, 1e-6)
            gamma_eff = min(0.5, self.gamma * surge_ratio)
        else:
            gamma_eff = 0.0

        reserved = int(np.ceil(gamma_eff * available))
        non_reserved = available - reserved

        admit_crit = min(q_crit, non_reserved)
        remaining = non_reserved - admit_crit
        admit_urg = min(q_urg, remaining)
        remaining -= admit_urg
        admit_rout = min(q_rout, remaining)

        return {
            "admit_critical": int(admit_crit),
            "admit_urgent": int(admit_urg),
            "admit_routine": int(admit_rout),
        }

    def update(self, actual_arrivals: list[float], current_time: float = 0.0):
        self.arrival_history.extend(actual_arrivals)
        self.arrival_history = self.arrival_history[-200:]

        for x in actual_arrivals:
            self.cusum.update(x)

        if self.cusum.is_triggered() and not self.adaptive_mode:
            self.adaptive_mode = True
            self.trigger_times.append(current_time)
            self.expansion_remaining = self.expansion_duration
            window = self.arrival_history[-self.fast_window:]
            self.lambda_current = max(0.1, float(np.mean(window)))
            self.cusum.reset(lambda_0=self.lambda_current)

        if self.adaptive_mode:
            if self.expansion_remaining > 0:
                self.expansion_remaining -= 1
            else:
                recent = self.arrival_history[-self.fast_window:]
                if len(recent) > 0:
                    new_est = float(np.mean(recent))
                    if abs(new_est - self.lambda_base) < 0.5:
                        self.adaptive_mode = False
                        self.lambda_current = self.lambda_base
                        self.cusum.reset(lambda_0=self.lambda_base)
                    else:
                        self.lambda_current = 0.7 * self.lambda_current + 0.3 * new_est
