"""Render the measured thread-scaling and two learning curves as a static SVG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from matplotlib import pyplot as plt

    plt.rcParams["svg.hashsalt"] = "bcir-pr816-2026-10-13"
    data = json.loads(args.results.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    colors = {"bcir": "#087f8c", "torch-compiled": "#c05b22"}
    series = (
        ("bcir", "BCIR native kblock+exp", "after-native-kblock-exp-t"),
        ("torch-compiled", "PyTorch compiled + fused AdamW", "torch-compiled-t"),
    )
    for backend, label, prefix in series:
        threads = (1, 4, 8)
        rows = [data["timing_summary"][prefix + str(n)] for n in threads]
        values = [r["median_ms"] for r in rows]
        axes[0].errorbar(
            threads,
            values,
            yerr=(
                [r["median_ms"] - r["min_ms"] for r in rows],
                [r["max_ms"] - r["median_ms"] for r in rows],
            ),
            marker="o",
            capsize=4,
            label=label,
            color=colors[backend],
        )
        for task, axis in (("lm", axes[1]), ("copy", axes[2])):
            [record] = [
                r
                for r in data["full_training"].values()
                if r["task"] == task and r["backend"] == backend
            ]
            metric = "cross_entropy" if task == "lm" else "exact_match"
            axis.plot(
                [r["step"] for r in record["evals"]],
                [r[metric] for r in record["evals"]],
                color=colors[backend],
                linestyle="-" if backend == "bcir" else "--",
                linewidth=2,
                label=label,
            )
    axes[0].set(
        title="Matched warm training times",
        xlabel="Compute threads",
        ylabel="Milliseconds per update",
        xticks=(1, 4, 8),
    )
    axes[1].set(title="Language-model validation", xlabel="Updates", ylabel="Nats per byte")
    axes[2].set(
        title="Held-out eight-letter copy",
        xlabel="Updates",
        ylabel="Teacher-forced exact match",
        ylim=(-0.03, 1.03),
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("PR #816 · 9.84M FP32 decoder · AMD EPYC VM · no BLAS on native path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"Date": None})
    plt.close(fig)
    if args.output.suffix == ".svg":
        args.output.write_text(
            "\n".join(line.rstrip() for line in args.output.read_text().splitlines()) + "\n"
        )


if __name__ == "__main__":
    main()
