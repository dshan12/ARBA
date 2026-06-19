# ARBA: Adaptive-Regret Bed Allocation

**A CUSUM-triggered robust optimization framework for non-stationary healthcare resource allocation**

[![CI](https://github.com/dshan12/arba/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/dshan12/arba/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-b31b1b.svg)](https://github.com/dshan12/arba)

---

## Abstract

Robust optimization is a leading framework for decision-making under uncertainty, but it assumes a fixed uncertainty set over the planning horizon. This assumption fails in dynamic environments where the underlying distribution shifts abruptly. We address this limitation by coupling a Poisson Cumulative Sum (CUSUM) sequential change detector with an online robust allocation policy. When the CUSUM statistic crosses a calibrated threshold, the policy re-centers its box uncertainty set via sliding-window maximum likelihood estimation and expands its robustness budget. The detection delay is bounded by Lorden's inequality, and the post-detection regret follows from standard online convex optimization bounds. In a SimPy discrete-event simulation of a 20-bed emergency department calibrated to NHAMCS parameters, the adaptive policy achieves 17--29% cumulative regret reduction over a static robust baseline across four non-stationary scenarios (all p < 0.001). A MIMIC-IV validation pipeline confirms 21.5% regret reduction under real-data-calibrated parameters.

---

## Problem Statement

Emergency department (ED) crowding is a well-documented public health crisis with measurable consequences for patient outcomes: higher mortality, prolonged inpatient stays, increased ambulance diversion, and higher rates of patients leaving without being seen [1, 2]. At its operational core, ED crowding reflects a resource allocation problem with non-stationary demand - arrivals follow diurnal patterns, weekly seasonality, and are punctuated by shock events such as mass casualty incidents, seasonal outbreaks, or severe weather.

Standard robust optimization [3] hedges against uncertainty by optimizing for worst-case realizations within a prescribed set. The limitation is fundamental: a static uncertainty set tuned for normal conditions is dangerously under-protective during surges, while one tuned for surges is wastefully conservative during normal operation. The resulting excess cost - regret - grows linearly with the post-shift horizon.

## Approach

The framework consists of three components:

**1. Sequential Change Detection (CUSUM)** - A Poisson log-likelihood ratio CUSUM [4, 5] monitors the arrival stream. The statistic accumulates positive evidence for a rate increase and triggers when it crosses threshold *h*. Lorden's inequality guarantees that the expected detection delay is bounded by *O(h / D_KL)*, where *D_KL* is the Kullback–Leibler divergence between pre- and post-shift distributions.

**2. Robust Allocation Policy** - A gamma-robust [3] bed reservation heuristic: a fraction *Gamma* of available beds is reserved for critical patients. The reservation fraction scales with the estimated arrival rate, interpolating between nominal and worst-case operation.

**3. Adaptive Switching** - Upon CUSUM detection, the policy re-estimates the arrival rate via sliding-window maximum likelihood, re-centers its box uncertainty set around the new estimate, and expands the robustness budget. When the rate returns to baseline, the policy reverts to standard robust mode.

The total expected regret is bounded by:

$$R(T) \leq B_{\max} \left[ \frac{h + D_{\text{KL}}}{D_{\text{KL}}} + \frac{\lambda_1}{2 D_{\text{KL}}} \log T \right] = \mathcal{O}(\log T)$$

in contrast to the linear regret *O(T)* incurred by a static robust policy.

## Key Results

| Scenario | Static Robust | Adaptive | Reduction | p-value |
|----------|:------------:|:--------:|:---------:|:-------:|
| Baseline (normal diurnal variation) | 31,013 ± 6,960 | 21,903 ± 6,099 | **29.4%** | <0.001 |
| Flash Flood (step-shift 3× surge) | 53,607 ± 3,882 | 44,205 ± 3,650 | **17.5%** | <0.001 |
| Creeping Crisis (gradual 2.5× drift) | 46,978 ± 4,481 | 37,738 ± 4,339 | **19.7%** | <0.001 |
| Acuity Flip (composition shift) | 42,613 ± 4,471 | 30,694 ± 4,075 | **28.0%** | <0.001 |

*Table: Cumulative regret (mean ± 95% CI, 30 replications). All comparisons significant by paired t-test and Wilcoxon signed-rank test.*

**Detection delay.** The CUSUM detects step shifts in 1.13 hours (Lorden bound: 3.17 h) and gradual drifts in 4.83 hours (bound: 11.42 h). Zero false alarms were recorded across all baseline replications.

**MIMIC-IV validation.** Under parameters calibrated from real ED data (42% critical acuity), the adaptive policy achieves 21.5% regret reduction (p < 0.001), confirming generalizability beyond synthetic calibration.

---

## Repository Structure

```
├── src/                  Core library (12 modules)
│   ├── cusum.py          Poisson CUSUM with Monte Carlo calibration
│   ├── environment.py    SimPy discrete-event ED simulation
│   ├── oracle.py         Hindsight MILP oracle via HiGHS
│   ├── policies.py       Robust and adaptive allocation policies
│   ├── simulation.py     Simulation runner and stress test framework
│   ├── theory.py         Regret bounds and sublinearity testing
│   ├── online_learning.py    Online convex optimization policy
│   ├── calibration.py    NHAMCS-based parameter estimation
│   ├── patient.py        Patient data model and acuity taxonomy
│   ├── regret.py         Regret computation utilities
│   ├── mimic_loader.py   MIMIC-IV data loading
│   ├── mimic_calibrator.py   MIMIC-IV parameter calibration
│   └── plotting.py       Publication-quality matplotlib style
├── tests/                71 tests (unit, integration, stress, edge cases)
├── experiments/          Multi-replication experiment scripts
│   ├── run_stress_tests.py     30-rep multi-scenario comparison
│   ├── run_calibration.py      20×15 CUSUM threshold grid
│   ├── run_factorial.py        3^5 factorial design
│   └── run_mimic_validation.py MIMIC-IV validation pipeline
├── dashboard/            Streamlit interactive visualization
├── paper/                Manuscript and bibliography
├── data/                 Calibrated parameters and MIMIC-IV demo
└── results/              Generated experimental output
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
git clone https://github.com/dshan12/arba.git
cd arba

# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Verifying the Installation

```bash
pytest tests/ -q
# Expected: 69 passed
```

---

## Usage

### Single Simulation

Compare robust and adaptive policies under a step-shift surge:

```bash
python main.py --B_max 20 --duration 168 --policy compare
```

Output includes average wait, max wait, P95 wait, total cost, oracle cost (V*), and regret for both policies.

### Reproducing Experiments

```bash
# Full 30-replication stress test suite (4 scenarios)
python -m experiments.run_stress_tests

# CUSUM threshold calibration (20×15 grid)
python -m experiments.run_calibration

# 3^5 factorial design sensitivity analysis
python -m experiments.run_factorial

# MIMIC-IV validation
python -m experiments.run_mimic_validation
```

### Interactive Dashboard

```bash
streamlit run dashboard/app.py
```

Opens an interactive visualization with live policy comparison, regret analysis, and parameter sensitivity.

---

## Documentation

- **[Paper](paper/paper.tex)** - Full manuscript with theoretical analysis and empirical results

---

## Citation

```bibtex
@techreport{kumar2026arba,
  title     = {Adaptive-Regret Bed Allocation: A CUSUM-Triggered Robust
               Optimization Framework for Non-Stationary Healthcare Environments},
  author    = {Sathish Kumar, Darshan},
  year      = {2026},
  institution = {University of Texas at San Antonio, Honors College},
  url       = {https://github.com/dshan12/arba}
}
```

---

## Related Work

This project builds on foundational work in three areas:

- **Robust optimization:** Bertsimas & Sim (2004), *The Price of Robustness*; Ben-Tal, El Ghaoui & Nemirovski (2009), *Robust Optimization*
- **Sequential change detection:** Page (1954), *Continuous Inspection Schemes*; Lorden (1971), *Procedures for Reacting to a Change in Distribution*; Moustakides (1986), *Optimal Stopping Times for Detecting Changes*
- **Online convex optimization:** Zinkevich (2003), *Online Convex Programming*; Hazan, Agarwal & Kale (2007), *Logarithmic Regret Algorithms*
- **Healthcare operations:** Chan, Farias & Reyes (2016), *Robust Optimization for ED Staffing*; Nohadani & Roy (2017), *Robust Optimization with Time-Dependent Uncertainty in Radiation Therapy*

A complete bibliography is available in [`paper/references.bib`](paper/references.bib).

---

## License

This project is distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

## Contact

Darshan Sathish Kumar - darshan.sathishkumar@my.utsa.edu  
UTSA Honors College, Department of Mathematics and Computer Science

---

## References

1. Sridhar, S., Mark, C., Wiler, J., & Pines, J. (2013). Patient flow in emergency departments: A review and research agenda. *International Journal of Healthcare Management*, 6(4), 197–208.
2. Golshani, F., Taghizadeh, G., & Yazdi, H. S. (2019). Bed allocation in hospitals: A comprehensive review. *Journal of Healthcare Management*, 64(3), 189–204.
3. Bertsimas, D. & Sim, M. (2004). The price of robustness. *Operations Research*, 52(1), 35–53.
4. Page, E. S. (1954). Continuous inspection schemes. *Biometrika*, 41(1/2), 100–115.
5. Lorden, G. (1971). Procedures for reacting to a change in distribution. *The Annals of Mathematical Statistics*, 42(6), 1897–1908.
