"""Plot cholqr (default call, with breakdown check) speed-up over torch.linalg.qr.

Reads benchmarks/results/overhead_2026-11/*.json (written by benchmarks/bench_overhead.py) and
writes docs/images/speedup.png. Needs matplotlib (not a dependency of cholqr).

    python benchmarks/plot_speedup.py
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "results", "overhead_2026-11")
OUT = os.path.join(HERE, "..", "docs", "images", "speedup.png")

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FASTER, SLOWER = "#2a78d6", "#eb6834"  # diverging pair: blue = cholqr faster, orange = slower
SHAPES = [(40, 768, 210), (64, 1024, 32), (16, 2048, 128), (8, 4096, 276), (1, 4096, 276), (1, 1024, 64)]

NAMES = {"apple_m4_mps": "Apple M4 (MPS)", "mi350x": "AMD MI350X", "rtxpro6000": "RTX PRO 6000 Blackwell",
         "rtx5090": "RTX 5090", "rtx4090": "RTX 4090", "h100": "H100", "a100": "A100", "a40": "A40", "l4": "L4"}
ORDER = ["h100", "a100", "mi350x", "rtxpro6000", "rtx5090", "rtx4090", "a40", "l4", "apple_m4_mps"]

data = {}
for f in glob.glob(os.path.join(SRC, "*.json")):
    d = json.load(open(f, encoding="utf-8"))
    key = os.path.splitext(os.path.basename(f))[0]
    data[key] = {tuple(r["shape"]): r["torch_qr_ms"] / r["fallback_ms"] for r in d["rows"]}

fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharex=True, sharey=True, facecolor=SURFACE)
ys = list(range(len(ORDER)))[::-1]
for ax, shape in zip(axes.flat, SHAPES):
    ax.set_facecolor(SURFACE)
    ax.axvline(1, color=INK2, lw=1, zorder=1)
    for y, key in zip(ys, ORDER):
        v = data[key].get(shape)
        if v is None:
            ax.text(1.05, y, "not measured", va="center", ha="left", fontsize=7.5, color=INK2, style="italic")
            continue
        c = FASTER if v >= 1 else SLOWER
        ax.plot([1, v], [y, y], color=c, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([v], [y], s=42, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.text(v * (1.14 if v >= 1 else 0.88), y, f"{v:.1f}x", va="center", ha="left" if v >= 1 else "right",
                fontsize=8, color=INK)
    b, n, r = shape
    ax.set_title(f"{b} x {n} x {r}" + ("  (batch)" if b > 1 else "  (single matrix)"), fontsize=10, color=INK,
                 loc="left")
    ax.set_xscale("log")
    ax.set_xlim(0.14, 40)
    ax.set_xticks([0.25, 1, 4, 16])
    ax.set_xticklabels(["0.25x", "1x", "4x", "16x"], fontsize=8, color=INK2)
    ax.minorticks_off()
    ax.set_yticks(ys)
    ax.set_yticklabels([NAMES[k] for k in ORDER], fontsize=8.5, color=INK2)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)

fig.suptitle("cholqr speed-up over torch.linalg.qr (FP32, precision \"highest\")", x=0.01, ha="left", fontsize=13,
             color=INK, fontweight="semibold")
fig.legend(handles=[plt.Line2D([], [], color=FASTER, lw=2, marker="o", label="cholqr faster"),
                    plt.Line2D([], [], color=SLOWER, lw=2, marker="o", label="cholqr slower")],
           loc="upper right", ncol=2, frameon=False, fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.99, 0.985))
fig.text(0.01, 0.005, "Default call incl. breakdown check, ratio of median ms. cholqr 0.5.0. torch 2.8.0+cu128 on NVIDIA, 2.10.0 (ROCm 7.1.1) on "
         "MI350X, 2.15.0 on Apple M4.\nSource: benchmarks/results/overhead_2026-11/*.json (benchmarks/bench_overhead.py).",
         fontsize=7.5, color=INK2, va="bottom")
fig.tight_layout(rect=(0, 0.045, 1, 0.95))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=170, facecolor=SURFACE)
print("wrote", os.path.normpath(OUT))
