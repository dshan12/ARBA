from enum import Enum

import numpy as np
import simpy

from src.patient import SERVICE_TIME_PARAMS, Acuity, Patient


class Scenario(str, Enum):
    FLASH_FLOOD = "flash_flood"
    CREEPING_CRISIS = "creeping_crisis"
    ACUITY_FLIP = "acuity_flip"


class EDEnvironment:
    def __init__(
        self,
        B_max: int,
        lambda_base: float = 10.0,
        surge_start: int | None = None,
        surge_mult: float = 3.0,
        surge_duration: int = 20,
        acuity_composition: dict[Acuity, float] | None = None,
        seed: int = 42,
    ):
        self.env = simpy.Environment()
        self.B_max = B_max
        self.lambda_base = lambda_base
        self.surge_start = surge_start
        self.surge_mult = surge_mult
        self.surge_duration = surge_duration
        self.surge_remaining = 0
        self.rng = np.random.default_rng(seed)

        self.acuity_composition = acuity_composition or {
            Acuity.CRITICAL: 0.1,
            Acuity.URGENT: 0.3,
            Acuity.ROUTINE: 0.6,
        }
        self.acuity_list = list(self.acuity_composition.keys())
        self.acuity_probs = list(self.acuity_composition.values())

        self.patients: list[Patient] = []
        self.patient_counter = 0
        self.arrival_log: list[tuple[float, float]] = []

        self.B_occ = 0
        self.queues: dict[Acuity, list[Patient]] = {
            Acuity.CRITICAL: [],
            Acuity.URGENT: [],
            Acuity.ROUTINE: [],
        }

        self.acuity_composition_post: dict[Acuity, float] | None = None
        self.acuity_list_post: list[Acuity] = []
        self.acuity_probs_post: list[float] = []
        self.t0_acuity_flip: float | None = None
        self.drift_start: float = 100

    def configure_scenario(self, scenario_name: str, params: dict | None = None):
        if params is None:
            params = {}
        if scenario_name == Scenario.FLASH_FLOOD.value:
            self.surge_start = params.get("surge_start", 50)
            self.surge_mult = params.get("surge_mult", 3.0)
            self.surge_duration = params.get("surge_duration", 20)
        elif scenario_name == Scenario.CREEPING_CRISIS.value:
            self.drift_rate = params.get("drift_rate", 0.01)
            self.drift_start = params.get("drift_start", 100)
        elif scenario_name == Scenario.ACUITY_FLIP.value:
            self.t0_acuity_flip = params.get("t0", 100)
            self.acuity_composition_post = params.get(
                "composition",
                {Acuity.CRITICAL: 0.4, Acuity.URGENT: 0.3, Acuity.ROUTINE: 0.3},
            )
            self.acuity_list_post = list(self.acuity_composition_post.keys())
            self.acuity_probs_post = list(self.acuity_composition_post.values())

    def lambda_t(self, t: float) -> float:
        base = self.lambda_base
        if hasattr(self, "drift_rate"):
            drift_hours = max(0, t - self.drift_start)
            base *= (1 + self.drift_rate) ** drift_hours
        rate = base * (1 + 0.3 * np.sin(2 * np.pi * t / 24 + np.pi / 2))
        if (
            self.surge_start is not None
            and self.surge_start <= t < self.surge_start + self.surge_duration
        ):
            return rate * self.surge_mult
        return rate

    def sample_acuity(self, t: float | None = None) -> Acuity:
        if t is None:
            t = self.env.now if hasattr(self, "env") else 0.0
        if (
            self.t0_acuity_flip is not None
            and t >= self.t0_acuity_flip
            and self.acuity_list_post
        ):
            return self.rng.choice(
                np.array(self.acuity_list_post, dtype=object), p=self.acuity_probs_post
            )  # type: ignore[arg-type]
        return self.rng.choice(
            np.array(self.acuity_list, dtype=object), p=self.acuity_probs
        )  # type: ignore[arg-type]

    def _start_treatment(self, patient: Patient):
        patient.service_start = self.env.now
        self.B_occ += 1
        self.env.process(self._treatment_process(patient))

    def _dequeue_next(self) -> Patient | None:
        for acuity in (Acuity.CRITICAL, Acuity.URGENT, Acuity.ROUTINE):
            if self.queues[acuity]:
                return self.queues[acuity].pop(0)
        return None

    def _try_admit_from_queues(self):
        while self.B_occ < self.B_max:
            if self.allocation_policy is not None:
                state = {
                    "B_occ": self.B_occ,
                    "q_crit": len(self.queues[Acuity.CRITICAL]),
                    "q_urg": len(self.queues[Acuity.URGENT]),
                    "q_rout": len(self.queues[Acuity.ROUTINE]),
                }
                action = self.allocation_policy.get_action(state)
                admitted_any = False
                for acuity, key in [
                    (Acuity.CRITICAL, "admit_critical"),
                    (Acuity.URGENT, "admit_urgent"),
                    (Acuity.ROUTINE, "admit_routine"),
                ]:
                    for _ in range(action.get(key, 0)):
                        if self.B_occ < self.B_max and self.queues[acuity]:
                            p = self.queues[acuity].pop(0)
                            self._start_treatment(p)
                            admitted_any = True
                if not admitted_any:
                    break
            else:
                patient = self._dequeue_next()
                if patient is None:
                    break
                self._start_treatment(patient)

    def _treatment_process(self, patient: Patient):
        service = patient.service_time
        yield self.env.timeout(service)
        patient.service_end = self.env.now
        self.B_occ -= 1
        self._try_admit_from_queues()

    def run(self, duration: float, allocation_policy=None):
        self.allocation_policy = allocation_policy

        def admission_loop():
            while self.env.now < duration:
                lam = self.lambda_t(self.env.now)
                interarrival = self.rng.exponential(1.0 / max(lam, 0.1))
                yield self.env.timeout(interarrival)

                if self.env.now > duration:
                    break

                acuity = self.sample_acuity(t=self.env.now)
                params = SERVICE_TIME_PARAMS[acuity]
                service_time = self.rng.lognormal(
                    mean=params["mu"], sigma=params["sigma"]
                )
                patient = Patient(
                    id=self.patient_counter,
                    acuity=acuity,
                    arrival_time=self.env.now,
                    service_time=service_time,
                )
                self.patient_counter += 1
                self.patients.append(patient)
                self.arrival_log.append((self.env.now, lam))

                if allocation_policy is not None:
                    self.queues[patient.acuity].append(patient)
                    state = {
                        "B_occ": self.B_occ,
                        "q_crit": len(self.queues[Acuity.CRITICAL]),
                        "q_urg": len(self.queues[Acuity.URGENT]),
                        "q_rout": len(self.queues[Acuity.ROUTINE]),
                    }
                    action = allocation_policy.get_action(state)
                    for acuity, key in [
                        (Acuity.CRITICAL, "admit_critical"),
                        (Acuity.URGENT, "admit_urgent"),
                        (Acuity.ROUTINE, "admit_routine"),
                    ]:
                        for _ in range(action.get(key, 0)):
                            if self.B_occ < self.B_max and self.queues[acuity]:
                                p = self.queues[acuity].pop(0)
                                self._start_treatment(p)
                else:
                    if self.B_occ < self.B_max:
                        self._start_treatment(patient)
                    else:
                        self.queues[patient.acuity].append(patient)

        self.env.process(admission_loop())
        self.env.run(until=duration)
        return self.patients
