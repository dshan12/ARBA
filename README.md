# ARBA: Adaptive-Regret Bed Allocation

**CUSUM-Triggered Robust Optimization for Non-Stationary Healthcare Environments**

[![CI](https://github.com/dshan12/arba/actions/workflows/ci.yml/badge.svg)](https://github.com/dshan12/arba/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

ARBA addresses a fundamental limitation of robust optimization: its inability to adapt to distribution shifts in dynamic environments. When an Emergency Department (ED) experiences a sudden surge in patient arrivals, a static robust policy tuned for normal conditions becomes dangerously inadequate.

The framework couples a **Poisson CUSUM sequential change detector** with an **online robust allocation policy**. When the CUSUM statistic crosses a calibrated threshold, the policy re-centers its box uncertainty set via sliding-window maximum likelihood estimation and expands its robustness budget. This yields provably sublinear regret — an improvement over the linear regret incurred by static robust policies.

This repository contains:

- **src/**: Core library (CUSUM, policies, oracle, simulation, theory)
- **tests/**: 71 pytest tests (unit, integration, stress, edge cases)
- **experiments/**: Multi-replication calibration and stress test scripts
- **dashboard/**: Interactive Streamlit visualization
- **paper.tex**: Full manuscript with theoretical results and empirical evaluation

## Key Results

| Scenario | Robust Regret | Adaptive Regret | Reduction | p-value |
|----------|:------------:|:---------------:|:---------:|:-------:|
| Baseline | 31,013 ± 6,960 | 21,903 ± 6,099 | 29.4% | <0.001 |
| Flash Flood (Step-Shift) | 53,607 ± 3,882 | 44,205 ± 3,650 | 17.5% | <0.001 |
| Creeping Crisis (Drift) | 46,978 ± 4,481 | 37,738 ± 4,339 | 19.7% | <0.001 |
| Acuity Flip (Composition) | 42,613 ± 4,471 | 30,694 ± 4,075 | 28.0% | <0.001 |

Mean ± 95% CI across 30 stochastic replications. All comparisons significant by paired t-test and Wilcoxon signed-rank test.

MIMIC-IV validation confirms 21.5% regret reduction (p < 0.001) under real-data-calibrated parameters.

## Theoretical Contribution

The total expected regret of the adaptive policy is bounded by:

$$R(T) \leq B_{\max} \left[ \frac{h + D_{\text{KL}}}{D_{\text{KL}}} + \frac{\lambda_1}{2 D_{\text{KL}}} \log T \right] = \mathcal{O}(\log T)$$

where $h$ is the CUSUM threshold, $D_{\text{KL}}$ is the Kullback-Leibler divergence between pre- and post-shift Poisson distributions, and $B_{\max}$ is the bed capacity. The detection delay is bounded by Lorden's inequality ($\mathbb{E}[\tau] \leq 1 + h / D_{\text{KL}}$), and the post-detection regret follows from standard online convex optimization bounds.

## Quick Start

```bash
# Install
pip install -e .

# Run test suite
pytest tests/

# Run a single simulation
python main.py --B_max 20 --duration 168 --policy compare

# Run stress test experiments
python -m experiments.multi_rep

# Launch interactive dashboard
streamlit run dashboard/app.py
```

## Repository Structure

```
├── src/                # Core library
│   ├── cusum.py        # Poisson CUSUM module
│   ├── environment.py  # SimPy ED simulation
│   ├── oracle.py       # Hindsight oracle (MILP)
│   ├── policies.py     # Robust & Adaptive policies
│   ├── simulation.py   # Simulation runner
│   ├── theory.py       # Regret bounds
│   ├── online_learning.py  # Online convex optimization
│   ├── calibration.py  # NHAMCS parameter estimation
│   ├── plotting.py     # Publication-quality figures
│   ├── patient.py      # Patient data model
│   ├── regret.py       # Regret computation
│   ├── mimic_loader.py       # MIMIC-IV data loading
│   └── mimic_calibrator.py   # MIMIC-IV calibration
├── tests/              # 71 pytest tests
├── experiments/        # Experiment scripts
├── dashboard/          # Streamlit visualization
├── docs/               # Documentation
├── paper/              # Manuscript and references
├── data/               # Parameters and MIMIC data
└── results/            # Generated output
```

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

## References

Key works this project builds upon:

- Bertsimas & Sim (2004). *The Price of Robustness.* Operations Research.
- Page (1954). *Continuous Inspection Schemes.* Biometrika.
- Lorden (1971). *Procedures for Reacting to a Change in Distribution.* Annals of Mathematical Statistics.
- Hazan, Agarwal & Kale (2007). *Logarithmic Regret Algorithms for Online Convex Optimization.* Machine Learning.
- Golrezaei, Jaillet & Zhou (2026). *Online Robust Resource Allocation with Adaptive Regret.* Operations Research (Forthcoming).

See `references.bib` for the complete bibliography.
