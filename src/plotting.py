from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def set_pub_style():
    mpl.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    sns.set_style("whitegrid")

    # Colorblind-friendly palette (Wong 2011)
    CB_PALETTE = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#F0E442",
        "#56B4E9",
        "#CC79A7",
        "#E69F00",
        "#000000",
    ]
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=CB_PALETTE)
    return CB_PALETTE


def save_fig(fig, name: str, dpi: int = 300):
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / name
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
