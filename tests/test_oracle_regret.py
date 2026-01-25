
from src.oracle import HindsightOracle
from src.patient import Acuity, Patient
from src.regret import compute_cumulative_regret, compute_regret


def _make_patient(pid, acuity, arrival, service, start=None):
    p = Patient(
        id=pid,
        acuity=acuity,
        arrival_time=arrival,
        service_time=service,
    )
    p.service_start = start
    return p


def test_oracle_empty_patients():
    oracle = HindsightOracle(B_max=5)
    assert oracle.compute_optimal([], total_time=10.0) == 0.0


def test_oracle_zero_total_time():
    oracle = HindsightOracle(B_max=5)
    patients = [_make_patient(0, Acuity.ROUTINE, 0.0, 1.0)]
    assert oracle.compute_optimal(patients, total_time=0.0) == 0.0


def test_oracle_trivial_solves():
    oracle = HindsightOracle(B_max=1)
    patients = [
        _make_patient(0, Acuity.CRITICAL, 0.0, 1.0),
        _make_patient(1, Acuity.ROUTINE, 0.0, 1.0),
    ]
    V_star = oracle.compute_optimal(patients, total_time=5.0)
    assert V_star > 0.0


def test_oracle_better_than_naive_baseline():
    oracle = HindsightOracle(B_max=1)
    patients = [
        _make_patient(0, Acuity.CRITICAL, 0.0, 1.0),
        _make_patient(1, Acuity.ROUTINE, 0.5, 1.0),
    ]
    V_star = oracle.compute_optimal(patients, total_time=5.0)
    p0, p1 = patients
    p0.service_start = 0.0
    p0.service_end = 1.0
    p1.service_start = 1.0
    p1.service_end = 2.0
    wait0 = max(0.0, p0.service_start - p0.arrival_time)
    wait1 = max(0.0, p1.service_start - p1.arrival_time)
    naive_cost = 100 * wait0 + 1 * wait1
    assert V_star <= naive_cost + 1e-3


def test_oracle_infeasible_falls_back():
    oracle = HindsightOracle(B_max=1)
    patients = [_make_patient(0, Acuity.CRITICAL, 0.0, 100.0)]
    V_star = oracle.compute_optimal(patients, total_time=1.0)
    assert V_star >= 0.0


def test_regret_sign_convention():
    regret = compute_regret(V_star=50.0, realized_cost=100.0)
    assert regret == 50.0
    assert regret > 0


def test_regret_zero_when_optimal():
    assert compute_regret(V_star=100.0, realized_cost=100.0) == 0.0


def test_regret_negative_when_better():
    assert compute_regret(V_star=100.0, realized_cost=50.0) == -50.0


def test_cumulative_regret_monotonic():
    costs = [10.0, 12.0, 15.0]
    cum = compute_cumulative_regret(costs, V_star=10.0)
    assert cum == [0.0, 2.0, 5.0]


def test_cumulative_regret_empty():
    assert compute_cumulative_regret([], V_star=10.0) == []


def test_milp_produces_lower_bound():
    oracle = HindsightOracle(B_max=2)
    patients = [
        _make_patient(0, Acuity.CRITICAL, 0.0, 2.0),
        _make_patient(1, Acuity.URGENT, 0.0, 1.0),
        _make_patient(2, Acuity.CRITICAL, 1.0, 1.0),
    ]
    V_star = oracle.compute_optimal(patients, total_time=5.0)
    greedy_cost = oracle._greedy_heuristic(patients, 5.0)
    assert V_star <= greedy_cost + 1e-9, f"MILP {V_star} > greedy {greedy_cost}"


def test_milp_empty_patients():
    oracle = HindsightOracle(B_max=5)
    assert oracle.compute_optimal([], total_time=10.0) == 0.0


def test_milp_zero_time():
    oracle = HindsightOracle(B_max=5)
    patients = [_make_patient(0, Acuity.ROUTINE, 0.0, 1.0)]
    assert oracle.compute_optimal(patients, total_time=0.0) == 0.0


def test_milp_fallback_on_too_many_variables():
    oracle = HindsightOracle(B_max=10)
    patients = [
        _make_patient(i, Acuity.ROUTINE, 0.0, 0.1) for i in range(200)
    ]
    V_star = oracle.compute_optimal(patients, total_time=100.0)
    assert V_star >= 0.0
