from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st

from src.environment import EDEnvironment
from src.oracle import HindsightOracle
from src.patient import ACUITY_WEIGHTS
from src.policies import AdaptivePolicy, RobustPolicy

st.set_page_config(page_title="ARBA Dashboard", layout="wide")
st.title("Adaptive-Regret Bed Allocation")


@st.cache_data
def load_results_csv(name: str):
    path = Path(__file__).resolve().parent.parent / "results" / name
    if path.exists():
        return pd.read_csv(path)
    return None


def run_simulation(lambda_base, surge_mult, H, B_max, duration, seed):
    env_rob = EDEnvironment(
        B_max=B_max,
        lambda_base=lambda_base,
        surge_start=int(duration * 0.5),
        surge_mult=surge_mult,
        surge_duration=int(duration * 0.25),
        seed=seed,
    )
    env_adp = EDEnvironment(
        B_max=B_max,
        lambda_base=lambda_base,
        surge_start=int(duration * 0.5),
        surge_mult=surge_mult,
        surge_duration=int(duration * 0.25),
        seed=seed,
    )

    robust = RobustPolicy(lambda_adv=lambda_base, B_max=B_max)
    adaptive = AdaptivePolicy(lambda_adv=lambda_base, B_max=B_max, threshold=H)

    pat_r = env_rob.run(duration=duration, allocation_policy=robust)
    pat_a = env_adp.run(duration=duration, allocation_policy=adaptive)

    return pat_r, pat_a, env_rob, env_adp


def compute_occupancy_timeseries(patients, duration, resolution=0.5):
    times = np.arange(0, duration, resolution)
    occ = np.zeros(len(times))
    for p in patients:
        if p.service_start is not None and p.service_end is not None:
            start_idx = int(p.service_start / resolution)
            end_idx = int(min(p.service_end, duration) / resolution)
            occ[start_idx:end_idx] += 1
    return times, occ


def compute_cumulative_regret(patients, oracle_cost):
    cumulative = []
    running = 0.0
    for p in patients:
        cost = ACUITY_WEIGHTS[p.acuity] * p.wait_time
        running += cost
        cumulative.append(running - oracle_cost)
    return running - oracle_cost, cumulative


def safe_cost(p, total_time):
    if p.service_start is not None:
        return ACUITY_WEIGHTS[p.acuity] * p.wait_time
    return ACUITY_WEIGHTS[p.acuity] * total_time


tab1, tab2 = st.tabs(["Live Battle", "Regret Analysis"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        lambda_base = st.slider("Base Arrival Rate λ₀", 5, 30, 10)
    with col2:
        surge_mult = st.slider("Surge Magnitude", 1.0, 5.0, 3.0)
    with col3:
        h_threshold = st.slider("CUSUM Threshold H", 1.0, 20.0, 5.0)
    with col4:
        duration = st.slider("Simulation Duration (hrs)", 24, 168, 72)

    if st.button("Run Simulation"):
        with st.spinner("Running simulation..."):
            B_max = 20
            pat_r, pat_a, env_rob, env_adp = run_simulation(
                lambda_base, surge_mult, h_threshold, B_max, duration, 42
            )

        costs_r = [safe_cost(p, duration) for p in pat_r]
        costs_a = [safe_cost(p, duration) for p in pat_a]
        total_r = sum(costs_r)
        total_a = sum(costs_a)

        oracle = HindsightOracle(B_max=20)
        V_star = oracle.compute_optimal(pat_r, total_time=duration)

        regret_r = total_r - V_star
        regret_a = total_a - V_star

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Robust Cost", f"{total_r:.1f}")
        with kpi2:
            st.metric("Adaptive Cost", f"{total_a:.1f}",
                      delta=f"{total_r - total_a:.1f}" if total_r != total_a else None)
        with kpi3:
            st.metric("Regret (Robust)", f"{regret_r:.1f}")
        with kpi4:
            st.metric("Regret (Adaptive)", f"{regret_a:.1f}")

        st.subheader("Cumulative Regret Over Time")
        cum_regret_r = [sum(costs_r[: i + 1]) - V_star for i in range(len(costs_r))]
        cum_regret_a = [sum(costs_a[: i + 1]) - V_star for i in range(len(costs_a))]
        times_r = [p.arrival_time for p in pat_r]
        times_a = [p.arrival_time for p in pat_a]

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=times_r, y=cum_regret_r, mode="lines",
            name="Robust", line=dict(color="red", width=2)
        ))
        fig1.add_trace(go.Scatter(
            x=times_a, y=cum_regret_a, mode="lines",
            name="Adaptive", line=dict(color="green", width=2)
        ))
        fig1.update_layout(
            xaxis_title="Time (hours)", yaxis_title="Cumulative Regret",
            height=350, hovermode="x unified"
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Bed Occupancy Over Time")
        cap = 20
        t_occ, occ_r = compute_occupancy_timeseries(pat_r, duration)
        _, occ_a = compute_occupancy_timeseries(pat_a, duration)

        fig2 = go.Figure()
        colors_r = [f"rgba({int(255 * o / cap) if o > 0 else 0}, {int(255 * (1 - o / cap))}, 0, 0.7)" for o in occ_r]
        colors_a = [f"rgba({int(255 * o / cap) if o > 0 else 0}, {int(255 * (1 - o / cap))}, 0, 0.7)" for o in occ_a]

        fig2.add_trace(go.Scatter(
            x=t_occ, y=occ_r, mode="lines+markers",
            name="Robust", marker=dict(color=colors_r, size=3),
            line=dict(color="red", width=1),
        ))
        fig2.add_trace(go.Scatter(
            x=t_occ, y=occ_a, mode="lines+markers",
            name="Adaptive", marker=dict(color=colors_a, size=3),
            line=dict(color="green", width=1),
        ))
        fig2.add_hline(y=cap, line_dash="dash", line_color="black",
                       annotation_text=f"Capacity ({cap})")
        fig2.update_layout(
            xaxis_title="Time (hours)", yaxis_title="Occupied Beds",
            height=350, hovermode="x unified"
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("True λ vs Policy Estimates")
        true_t = [e[0] for e in env_rob.arrival_log]
        true_lam = [e[1] for e in env_rob.arrival_log]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=true_t, y=true_lam, mode="lines",
            name="True λ(t)", line=dict(color="black", width=2)
        ))
        fig3.add_hline(y=lambda_base, line_dash="dot", line_color="red",
                       annotation_text="λ₀ (baseline)")
        fig3.update_layout(
            xaxis_title="Time (hours)", yaxis_title="Arrival Rate λ",
            height=350, hovermode="x unified"
        )
        st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.subheader("Regret Analysis from Saved Results")

    col_a, col_b = st.columns(2)
    with col_a:
        metric_choice = st.selectbox(
            "Metric", ["total_cost", "regret", "n_patients"],
            format_func=lambda x: {"total_cost": "Total Cost",
                                    "regret": "Regret",
                                    "n_patients": "Patients Seen"}[x],
        )
    with col_b:
        scenario_filter = st.multiselect(
            "Scenarios",
            ["baseline", "flash_flood", "creeping_crisis", "acuity_flip"],
            default=["baseline", "flash_flood"],
        )

    df = load_results_csv("multi_rep_summary.csv")
    if df is not None:
        valid_cols = {"scenario", "policy", "total_cost", "V_star", "regret", "n_patients"}
        missing = valid_cols - set(df.columns)
        if missing:
            st.warning(f"CSV missing columns: {', '.join(missing)}. Run stress tests first.")
        elif metric_choice not in df.columns:
            st.warning(f"Column '{metric_choice}' not found in CSV.")
        else:
            df_filt = df[df["scenario"].isin(scenario_filter)] if scenario_filter else df

            st.subheader(f"{metric_choice.replace('_', ' ').title()} by Policy")
            plot_df = df_filt.groupby(["scenario", "policy"]).agg(
                mean=(metric_choice, "mean"),
                std=(metric_choice, "std"),
            ).reset_index()

            fig_bar = go.Figure()
            for policy in ["robust", "adaptive"]:
                sub = plot_df[plot_df["policy"] == policy]
                fig_bar.add_trace(go.Bar(
                    name=policy.title(),
                    x=sub["scenario"],
                    y=sub["mean"],
                    error_y=dict(type="data", array=sub["std"], visible=True),
                ))
            fig_bar.update_layout(
                barmode="group", xaxis_title="Scenario",
                yaxis_title=metric_choice.replace("_", " ").title(),
                height=400,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Calibrated Parameters")
        cal_df = load_results_csv("calibration_frontier.csv")
        if cal_df is not None:
            st.dataframe(cal_df.head(10), use_container_width=True)
        else:
            st.info("Run experiments/run_calibration.py to generate calibration data.")
    else:
        st.info("Run experiments/run_stress_tests.py to generate result CSVs.")
