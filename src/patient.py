from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Acuity(Enum):
    CRITICAL = "critical"
    URGENT = "urgent"
    ROUTINE = "routine"


ACUITY_WEIGHTS = {Acuity.CRITICAL: 100, Acuity.URGENT: 10, Acuity.ROUTINE: 1}

SERVICE_TIME_PARAMS = {
    Acuity.CRITICAL: {"mu": 3.0, "sigma": 1.0},
    Acuity.URGENT: {"mu": 2.0, "sigma": 0.75},
    Acuity.ROUTINE: {"mu": 1.0, "sigma": 0.5},
}


@dataclass
class Patient:
    id: int
    acuity: Acuity
    arrival_time: float
    service_time: float
    service_start: Optional[float] = None
    service_end: Optional[float] = None

    @property
    def wait_time(self) -> float:
        if self.service_start is None:
            raise ValueError("wait_time is not defined for patients that never received a bed")
        return max(0.0, self.service_start - self.arrival_time)

    @property
    def is_complete(self) -> bool:
        return self.service_end is not None
