#!/usr/bin/env python3
"""Build the evidence figures used by the public reproduction report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/ecl-covariate-shift"
DATA = json.loads((REPORT / "data/summary.json").read_text(encoding="utf-8"))
IMAGES = REPORT / "images"

COLORS = {
    "navy": "#17324D",
    "blue": "#377EB8",
    "orange": "#FF9F1C",
    "red": "#D1495B",
    "green": "#2A9D8F",
    "gray": "#708090",
}


def finish(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(IMAGES / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def exact_witnesses() -> None:
    rows = DATA["claim3_and_6_exact_witnesses"]
    labels = [row["name"] for row in rows]
    y = np.arange(len(rows))
    full = [row["full"] for row in rows]
    mini = [row["claimed_minibatch"] for row in rows]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.hlines(y, mini, full, color=COLORS["gray"], linewidth=3)
    ax.scatter(full, y, s=95, color=COLORS["navy"], label="Full / true quantity", zorder=3)
    ax.scatter(mini, y, s=95, color=COLORS["red"], label="Claimed mini-batch / profiled quantity", zorder=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Exact rational value (shown as decimal)")
    ax.set_title("The mini-batch identity fails on four exact witnesses", loc="left", weight="bold")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    finish(fig, "headline-exact-counterexamples.png")


def simulation() -> None:
    data = DATA["claim6_simulation"]
    paradigms = list(data)
    methods = ["Uncalibrated", "Soft-ECE", "ECL"]
    x = np.arange(len(paradigms))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for index, method in enumerate(methods):
        values = [data[p][method] for p in paradigms]
        ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=method,
            color=[COLORS["gray"], COLORS["orange"], COLORS["blue"]][index],
        )
    ax.set_xticks(x, paradigms)
    ax.set_ylabel("Mean target calibration error (5 seeds)")
    ax.set_title("Paper-scale simulation: ECL did not improve any paradigm", loc="left", weight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    finish(fig, "claim6-simulation.png")


def theorem_budget() -> None:
    data = DATA["claim2"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    axes[0].bar(
        ["Derived proof\nbudget", "Certified absolute\nconstant"],
        [data["derived_budget"], data["absolute_constant"]],
        color=[COLORS["green"], COLORS["navy"]],
    )
    axes[0].set_ylim(0, 17)
    axes[0].set_ylabel("Constant")
    axes[0].set_title("Corrected soft-bin proof closes", loc="left", weight="bold")
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].bar(
        ["Largest observed\nerror / radius", "Certified C"],
        [data["maximum_error_over_radius"], data["absolute_constant"]],
        color=[COLORS["orange"], COLORS["navy"]],
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Log scale")
    axes[1].set_title("1,024 diagnostics stay far inside", loc="left", weight="bold")
    axes[1].spines[["top", "right"]].set_visible(False)
    finish(fig, "claim2-proof-budget.png")


def implementation_agreement() -> None:
    rows = DATA["claim4"]
    labels = [row["paradigm"] for row in rows]
    loss = [row["loss_error"] for row in rows]
    gradient = [row["gradient_error"] for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - 0.18, loss, 0.36, label="Loss absolute error", color=COLORS["green"])
    ax.bar(x + 0.18, gradient, 0.36, label="Gradient absolute error", color=COLORS["blue"])
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("PyTorch vs independent NumPy error (log scale)")
    ax.set_title("All three Appendix F implementations agree independently", loc="left", weight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    finish(fig, "claim4-independent-agreement.png")


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.titlepad": 12})
    exact_witnesses()
    simulation()
    theorem_budget()
    implementation_agreement()
    print(f"wrote 4 figures to {IMAGES}")


if __name__ == "__main__":
    main()
