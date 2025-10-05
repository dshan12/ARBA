import highspy
import numpy as np

from src.patient import ACUITY_WEIGHTS, Patient

_MAX_MILP_VARS = 20000


class HindsightOracle:
    def __init__(self, B_max: int, time_resolution: float = 1.0):
        self.B_max = B_max
        self.time_resolution = time_resolution

    def compute_optimal(
        self,
        patients: list[Patient],
        total_time: float,
    ) -> float:
        if not patients:
            return 0.0
        if total_time <= 0:
            return 0.0

        try:
            return self._solve_milp(patients, total_time)
        except Exception:
            return self._greedy_heuristic(patients, total_time)

    def _greedy_heuristic(
        self,
        patients: list[Patient],
        total_time: float,
    ) -> float:
        wait_weights = np.array(
            [ACUITY_WEIGHTS[p.acuity] for p in patients], dtype=float
        )
        order = np.lexsort(
            (
                [p.arrival_time for p in patients],
                [-ACUITY_WEIGHTS[p.acuity] for p in patients],
            )
        )
        beds_free_until = np.zeros(self.B_max, dtype=float)
        total_cost = 0.0
        for idx in order:
            p = patients[idx]
            w = wait_weights[idx]
            bed_idx = int(np.argmin(beds_free_until))
            start = max(p.arrival_time, beds_free_until[bed_idx])
            wait = start - p.arrival_time
            if wait >= total_time:
                total_cost += w * total_time
            else:
                total_cost += w * wait
                beds_free_until[bed_idx] = start + p.service_time
        return float(total_cost)

    def _solve_milp(
        self,
        patients: list[Patient],
        total_time: float,
    ) -> float:
        R = self.time_resolution
        n = len(patients)

        arrivals = np.array([p.arrival_time for p in patients])
        services = np.array([p.service_time for p in patients])
        weights = np.array([ACUITY_WEIGHTS[p.acuity] for p in patients])

        earliest_slot = np.ceil(arrivals / R).astype(int)
        service_slots = np.maximum(1, np.ceil(services / R).astype(int))
        max_service = float(np.max(services))

        T = int(np.ceil((total_time + max_service) / R)) + 1

        estimated_vars = sum(max(0, T - earliest_slot[i]) for i in range(n))
        if estimated_vars > _MAX_MILP_VARS:
            raise RuntimeError(
                f"MILP too large (estimated {estimated_vars} >= {_MAX_MILP_VARS})"
            )
        if estimated_vars == 0:
            raise RuntimeError("No MILP variables created")

        h = highspy.Highs()
        h.silent()

        var_map: dict[tuple[int, int], highspy.highs_var] = {}
        for i in range(n):
            for t in range(earliest_slot[i], T):
                var = h.addVariable(
                    lb=0,
                    ub=1,
                    obj=weights[i] * (t * R - arrivals[i]),
                    type=highspy.HighsVarType.kInteger,
                )
                var_map[(i, t)] = var

        patient_vars: list[list[highspy.highs_var]] = [[] for _ in range(n)]
        for (i, t), var in var_map.items():
            patient_vars[i].append(var)
        for i, vars_i in enumerate(patient_vars):
            if vars_i:
                h.addConstr(sum(vars_i) == 1)

        T_checks = T + int(np.ceil(max_service / R))
        for k in range(T_checks):
            active: list[highspy.highs_var] = []
            for i in range(n):
                t_min = max(earliest_slot[i], k - service_slots[i] + 1)
                t_max = min(k, T - 1)
                for t in range(t_min, t_max + 1):
                    key = (i, t)
                    if key in var_map:
                        active.append(var_map[key])
            if active:
                h.addConstr(sum(active) <= self.B_max)

        h.run()
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(
                f"HiGHS did not find optimal solution: {h.getModelStatus()}"
            )

        return h.getObjectiveValue()


def compute_v_star_bucketed(
    patients: list[Patient],
    B_max: int,
    total_time: float,
    time_resolution: float = 1.0,
) -> float:
    if not patients:
        return 0.0

    n_hours = int(np.ceil(total_time / time_resolution))
    acuity_order = sorted(ACUITY_WEIGHTS.keys(), key=lambda a: ACUITY_WEIGHTS[a], reverse=True)

    arrivals = {
        a: np.zeros(n_hours, dtype=int) for a in acuity_order
    }
    service_times: dict = {a: [] for a in acuity_order}

    for p in patients:
        hour = min(int(p.arrival_time / time_resolution), n_hours - 1)
        arrivals[p.acuity][hour] += 1
        service_times[p.acuity].append(p.service_time)

    avg_service = {
        a: float(np.mean(service_times[a])) if service_times[a] else 1.0
        for a in acuity_order
    }

    return float(_solve_bucketed(arrivals, avg_service, B_max, n_hours, time_resolution))


def _solve_bucketed(
    arrivals: dict,
    avg_service: dict,
    B_max: int,
    n_hours: int,
    time_resolution: float,
) -> float:
    acuity_order = sorted(arrivals.keys(), key=lambda a: ACUITY_WEIGHTS[a], reverse=True)
    queue = {a: 0 for a in acuity_order}
    total_cost = 0.0
    beds_free = B_max

    for t in range(n_hours):
        for a in acuity_order:
            queue[a] += arrivals[a][t]

        for a in acuity_order:
            serve = min(queue[a], beds_free)
            if serve > 0:
                queue[a] -= serve
                beds_free -= serve
                total_cost += ACUITY_WEIGHTS[a] * serve * (time_resolution / 2.0)

        beds_free = B_max

        for a in acuity_order:
            total_cost += ACUITY_WEIGHTS[a] * queue[a] * time_resolution

    return total_cost
