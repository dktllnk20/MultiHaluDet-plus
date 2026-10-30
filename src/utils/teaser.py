"""Horizontal grouped bar chart: Base vs Ours AUC on 4 languages (en, fr, vi, bn).

Location : root/src/utils/compare_bar.py
Output   : root/results/img/compare_<model>.png

y axis -> languages (en, fr, vi, bn), x axis -> AUC, 2 bars per language:
Base (blue) and Ours (orange), value printed at the end of each bar.
Bar value = mean over seeds (computed from the per-seed numbers below).

Usage (from root):
    python src/utils/compare_bar.py                       # one figure per model
    python src/utils/compare_bar.py --combined            # + a side-by-side figure
    python src/utils/compare_bar.py --models mistral-7b   # only one model
    python src/utils/compare_bar.py --xmin 0.9 --err      # wider x range + std error bars
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "results" / "img"

BASE_COLOR = "#4F72AF"  #blue 
OURS_COLOR = "#DC8455"  #orange
LANGS = ["en", "fr", "vi", "bn"]  # top -> bottom

# Per-seed AUC, seeds = (1, 11, 29, 37, 43)
RESULTS = {
    "mistral-7b": {
        "en": {"base": [0.9836, 0.9837, 0.9900, 0.9825, 0.9846], "ours": [0.9837, 0.9861, 0.9912, 0.9851, 0.9884]},
        "fr": {"base": [0.9734, 0.9729, 0.9761, 0.9757, 0.9747], "ours": [0.9752, 0.9784, 0.9780, 0.9797, 0.9757]},
        "vi": {"base": [0.9683, 0.9680, 0.9702, 0.9680, 0.9669], "ours": [0.9724, 0.9731, 0.9750, 0.9719, 0.9726]},
        "bn": {"base": [0.9560, 0.9621, 0.9617, 0.9607, 0.9628], "ours": [0.9592, 0.9665, 0.9675, 0.9670, 0.9669]},
    },
    "llama2-7b": {
        "en": {"base": [0.9812, 0.9848, 0.9868, 0.9838, 0.9862], "ours": [0.9841, 0.9857, 0.9902, 0.9869, 0.9851]},
        "fr": {"base": [0.9729, 0.9750, 0.9751, 0.9736, 0.9726], "ours": [0.9750, 0.9790, 0.9809, 0.9784, 0.9746]},
        "vi": {"base": [0.9722, 0.9701, 0.9729, 0.9689, 0.9682], "ours": [0.9727, 0.9743, 0.9765, 0.9745, 0.9726]},
        "bn": {"base": [0.9513, 0.9565, 0.9553, 0.9527, 0.9557], "ours": [0.9544, 0.9628, 0.9640, 0.9597, 0.9615]},
    },
}


def mean_std(values: list[float]) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    return a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0


def draw_model(ax, model: str, xmin: float, xmax: float, err: bool = False, title: bool = True) -> None:
    data = RESULTS[model]
    h = 0.36
    y = np.arange(len(LANGS))
    base = [mean_std(data[l]["base"]) for l in LANGS]
    ours = [mean_std(data[l]["ours"]) for l in LANGS]
    kw = dict(capsize=3, error_kw=dict(ecolor="#333", lw=1)) if err else {}

    # y inverted below, so the "base" bar (offset -h/2) sits above the "ours" bar
    b1 = ax.barh(y - h / 2, [m for m, _ in base], h, color=BASE_COLOR, label="Base",
                 xerr=[s for _, s in base] if err else None, **kw)
    b2 = ax.barh(y + h / 2, [m for m, _ in ours], h, color=OURS_COLOR, label="Ours",
                 xerr=[s for _, s in ours] if err else None, **kw)

    pad = (xmax - xmin) * 0.012
    for bars, stats in ((b1, base), (b2, ours)):
        for rect, (m, s) in zip(bars, stats):
            ax.text(m + (s if err else 0) + pad, rect.get_y() + rect.get_height() / 2,
                    f"{m:.4f}", va="center", ha="left", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(LANGS, fontsize=11)
    ax.invert_yaxis()  # en on top
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("AUC")
    # if title:
    #     ax.set_title(f"{model} — HaluEval")
    ax.grid(alpha=0.3, axis="x")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)


def plot_compare(models: list[str], out_dir: Path, xmin: float, xmax: float, err: bool,
                 combined: bool, dpi: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for m in models:
        fig, ax = plt.subplots(figsize=(12, 4))
        draw_model(ax, m, xmin, xmax, err)
        fig.tight_layout()
        p = out_dir / f"compare_{m}.png"
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        paths.append(p)
        print(f"saved -> {p}")

    if combined and len(models) > 1:
        fig, axes = plt.subplots(1, len(models), figsize=(7.5 * len(models), 4.6), sharex=True)
        for ax, m in zip(axes, models):
            draw_model(ax, m, xmin, xmax, err)
        fig.tight_layout()
        p = out_dir / f"compare_{'_'.join(models)}.png"
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        paths.append(p)
        print(f"saved -> {p}")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(RESULTS), choices=list(RESULTS))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--xmin", type=float, default=0.94, help="left x limit (AUC axis is truncated)")
    ap.add_argument("--xmax", type=float, default=1.0)
    ap.add_argument("--err", action="store_true", help="add std (over seeds) error bars")
    ap.add_argument("--combined", action="store_true", help="also save a side-by-side figure of all models")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()
    plot_compare(a.models, Path(a.out_dir), a.xmin, a.xmax, a.err, a.combined, a.dpi)


if __name__ == "__main__":
    main()
