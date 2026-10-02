from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.paths import FIGURES

# Project palette — steel-blue / grey scheme used across Notebook 01
PALETTE = {
    "primary": "#1f4e79",
    "secondary": "#2b5c8f",
    "missing": "#f0f2f5",
    "accent": "#d9534f",
    "grid": "#cccccc",
}


def apply_style() -> None:
    """Apply consistent rcParams across all notebooks."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": PALETTE["grid"],
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "grid.color": PALETTE["grid"],
    })


def save_figure(fig, filename: str, dpi: int = 300) -> Path:
    """Save a matplotlib figure to reports/figures/ at the given dpi.

    Path resolution is anchored to the project root (via src.paths),
    so it works regardless of the kernel's current working directory.
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / filename
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    return out
