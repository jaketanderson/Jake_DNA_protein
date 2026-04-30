"""Analyze crossval_conditions.m output.

For each experimental condition:
  - convert the trained transition probability matrix P to a rate matrix
    K = logm(P) / dt (continuous-time rate constants, units 1/s),
  - plot every trajectory (train + test) with the original 3-color FRET
    signal, the Viterbi-fitted means, and a colored bar beneath the plot
    showing the inferred state vs time,
  - write everything to a single PDF per condition, with a final page
    containing the rate-constant matrix.

Written by Claude to not waste time. It's been reviewed by me, a human.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.linalg import logm

DT = 0.05  # seconds between frames; matches crossval_conditions.m
N_STATES = 6
STATE_COLORS = [
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#9c755f",
]
CHANNEL_COLORS = [(0.47, 0.67, 0.19), (0.07, 0.62, 1.0), (0.64, 0.08, 0.18)]
CHANNEL_NAMES = ["green", "blue", "red"]


def transition_to_rates(P: np.ndarray, dt: float) -> np.ndarray:
    """Continuous-time rate matrix K such that P = expm(K*dt)."""
    K = logm(P) / dt
    K = np.real_if_close(K, tol=1e6)
    return np.real(K)


def load_trace(trace_dir: Path):
    orig = np.loadtxt(trace_dir / "originalData.txt", delimiter=",")
    fit = np.loadtxt(trace_dir / "fittedData.txt", delimiter=",")
    states = np.loadtxt(trace_dir / "stateSequence.txt", delimiter=",")
    time = orig[0]
    ydata = orig[1:4]
    yfit = fit[1:4]
    state_seq = states[1].astype(int)
    return time, ydata, yfit, state_seq


def plot_trace(fig, name: str, time, ydata, yfit, state_seq):
    gs = fig.add_gridspec(2, 1, height_ratios=[6, 1], hspace=0.05)
    ax_top = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1], sharex=ax_top)

    for i in range(3):
        ax_top.plot(time, ydata[i], "-", color=CHANNEL_COLORS[i],
                    linewidth=1.0, alpha=0.35, label=f"{CHANNEL_NAMES[i]} (raw)")
        ax_top.plot(time, yfit[i], "-", color=CHANNEL_COLORS[i],
                    linewidth=1.8, label=f"{CHANNEL_NAMES[i]} (fit)")
    ax_top.set_ylabel("intensity (counts)")
    ax_top.set_title(name)
    ax_top.legend(loc="upper right", fontsize=7, ncol=3)
    ax_top.tick_params(labelbottom=False)
    ax_top.set_xlim(time[0], time[-1])

    cmap = ListedColormap(STATE_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, N_STATES + 0.5, 1.0), cmap.N)
    bar = state_seq.reshape(1, -1)
    ax_bar.imshow(
        bar, aspect="auto", cmap=cmap, norm=norm,
        extent=(time[0], time[-1], 0, 1), interpolation="nearest",
    )
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("time (s)")
    ax_bar.set_ylabel("state", rotation=0, ha="right", va="center")


def plot_rates_page(fig, condition: str, K: np.ndarray, P: np.ndarray):
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")

    lines = [f"Condition: {condition}", f"dt = {DT} s", "",
             "Transition probability matrix P (rows sum to 1):"]
    lines.append(_format_matrix(P, fmt="{: .4f}"))
    lines.append("")
    lines.append("Rate matrix K = logm(P)/dt   (units: 1/s)")
    lines.append("K[i,j] for i!=j is the rate from state i to state j;")
    lines.append("diagonal entries are negative total exit rates.")
    lines.append(_format_matrix(K, fmt="{: .4f}"))
    lines.append("")
    lines.append("Off-diagonal rate constants (state i -> state j, 1/s):")
    for i in range(N_STATES):
        for j in range(N_STATES):
            if i == j:
                continue
            lines.append(f"  k({i} -> {j}) = {K[i, j]: .5f}")

    ax.text(0.0, 1.0, "\n".join(lines), family="monospace",
            fontsize=8, va="top", ha="left", transform=ax.transAxes)


def _format_matrix(M: np.ndarray, fmt: str) -> str:
    rows = []
    for row in M:
        rows.append("  " + "  ".join(fmt.format(v) for v in row))
    return "\n".join(rows)


def collect_traces(condition_dir: Path):
    """Yield (display_name, trace_dir) sorted, train then test."""
    for split in ("train", "test"):
        split_dir = condition_dir / split
        if not split_dir.is_dir():
            continue
        for trace_dir in sorted(split_dir.iterdir()):
            if not trace_dir.is_dir():
                continue
            yield f"{split}/{trace_dir.name}", trace_dir


def process_condition(condition_dir: Path, out_pdf: Path):
    P = np.loadtxt(condition_dir / "transitionProbability.txt", delimiter=",")
    K = transition_to_rates(P, DT)

    with PdfPages(out_pdf) as pdf:
        rate_fig = plt.figure(figsize=(8.5, 11))
        plot_rates_page(rate_fig, condition_dir.name, K, P)
        pdf.savefig(rate_fig)
        plt.close(rate_fig)

        for name, trace_dir in collect_traces(condition_dir):
            time, ydata, yfit, states = load_trace(trace_dir)
            fig = plt.figure(figsize=(11, 5.5))
            plot_trace(fig, f"{condition_dir.name}  —  {name}",
                       time, ydata, yfit, states)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"wrote {out_pdf}")


def main():
    repo = Path(__file__).resolve().parent
    crossval_root = (repo / "GAFsmFRETdata"
                     / "matlabCode_constraintEM_HMM_for_3colorFRET"
                     / "crossval_output")
    if not crossval_root.is_dir():
        sys.exit(f"missing crossval output: {crossval_root}")

    out_root = repo / "analyze_output"
    out_root.mkdir(exist_ok=True)

    condition_dirs = sorted(d for d in crossval_root.iterdir()
                            if d.is_dir() and (d / "transitionProbability.txt").exists())
    if not condition_dirs:
        sys.exit(f"no condition directories under {crossval_root}")

    for cond in condition_dirs:
        process_condition(cond, out_root / f"{cond.name}.pdf")


if __name__ == "__main__":
    main()
