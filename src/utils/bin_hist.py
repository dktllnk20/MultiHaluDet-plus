"""Surface-shortcut figures on HaluEval: hallucination rate by answer length (Fig 4a)
and by "ends with period" (Fig 4b).

Location : root/src/utils/bin_hist.py
Input    : root/data/halueval_en.json  (JSON list or JSONL)
Output   : root/results/img/*.png

Fig 4a: x axis -> answer-length bin, y axis -> % of answers in that bin that are
        hallucinated, with the percentage printed on top of each bar (n under the label).
Fig 4b: x axis -> "Ends with period" (Yes / No), same y axis, same style, n per group.

Usage (from root):
    python src/utils/bin_hist.py                      # both figures (4a: 5 quantile bins, word length; 4b)
    python src/utils/bin_hist.py --fig a --n-bins 7   # only Fig 4a, 7 bins
    python src/utils/bin_hist.py --fig b              # only Fig 4b (period Yes/No)
    python src/utils/bin_hist.py --n-bins 8
    python src/utils/bin_hist.py --n-bins 10 --mode uniform --unit char
    python src/utils/bin_hist.py --unit token --tokenizer mistralai/Mistral-7B-v0.1
    python src/utils/bin_hist.py --theme orange          # solid color theme
    python src/utils/bin_hist.py --theme YlOrRd          # gradient: color follows the rate
    python src/utils/bin_hist.py --theme "#8E44AD"       # any hex color
    python src/utils/bin_hist.py --horizontal            # horizontal bars (both figures)
    python src/utils/bin_hist.py --style dark --theme plasma
    python src/utils/bin_hist.py --horizontal --style paper --theme navy --title

Look & feel = two independent knobs:
    --theme  bar colors (solid name | matplotlib colormap | hex)
    --style  figure look: default | clean | paper | soft | dark

From other code:
    from src.utils.bin_hist import plot_length_bin_hist, plot_period_hist
    plot_length_bin_hist(n_bins=6, horizontal=True, style="clean")
    plot_period_hist(horizontal=True, style="clean")

Supported record layouts (auto-detected, or force with --answer-key/--label-key):
  1. Paired  : {"right_answer": ..., "hallucinated_answer": ...}
               -> each record yields 2 samples (label 0 and label 1)
  2. Flat    : {"answer": ..., "label": 0/1 | "yes"/"no" | true/false}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "halueval_en.json"
DEFAULT_OUT_DIR = ROOT / "results" / "img"

# (correct_key, hallucinated_key) candidates for the paired layout
PAIR_KEYS = [
    ("right_answer", "hallucinated_answer"),
    ("correct_answer", "hallucinated_answer"),
    ("right_response", "hallucinated_response"),
    ("right_summary", "hallucinated_summary"),
]
ANSWER_KEYS = ["answer", "response", "output", "text", "summary"]
LABEL_KEYS = ["label", "hallucination", "is_hallucination", "hallucinated", "ground_truth"]

_POS = {"1", "yes", "true", "hallucination", "hallucinated"}
_NEG = {"0", "no", "false", "correct", "right", "factual", "non-hallucination"}


# --------------------------------------------------------------------------- #
# Themes
# --------------------------------------------------------------------------- #
# Solid themes: every bar has the same color.
SOLID_THEMES = {
    "blue": "#3B6FB6",
    "orange": "#D9822B",
    "green": "#4C9F70",
    "red": "#C8433F",
    "purple": "#7B5EA7",
    "teal": "#2A9D8F",
    "gray": "#6B7280",
    "black": "#222222",
    "navy": "#1F3A5F",
    "indigo": "#4F46E5",
    "coral": "#E76F51",
    "rose": "#D6537A",
    "emerald": "#2BA37F",
    "gold": "#E0A526",
}
# Gradient themes: any matplotlib colormap name also works (e.g. viridis, magma,
# coolwarm, YlOrRd, Blues...). Bar color follows its hallucination rate (0-100%).
GRADIENT_EXAMPLES = ["viridis", "magma", "plasma", "coolwarm", "YlOrRd", "Blues", "RdYlGn_r"]
DEFAULT_THEME = "blue"


# --------------------------------------------------------------------------- #
# Styles: figure look & feel presets (independent from the bar color theme)
# --------------------------------------------------------------------------- #
_DESPINE = {"axes.spines.top": False, "axes.spines.right": False}
STYLES = {
    "default": {"grid.alpha": 0.3},
    "clean": {  # minimal, no top/right frame, softer text
        **_DESPINE, "axes.edgecolor": "#888888", "axes.labelcolor": "#222222",
        "xtick.color": "#444444", "ytick.color": "#444444", "grid.alpha": 0.25,
    },
    "paper": {  # serif font, no top/right frame: fits LaTeX / conference papers
        **_DESPINE, "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 12, "grid.alpha": 0.2,
    },
    "soft": {  # warm off-white background, dotted grid
        **_DESPINE, "figure.facecolor": "#FAF7F2", "axes.facecolor": "#FAF7F2",
        "savefig.facecolor": "#FAF7F2", "axes.edgecolor": "#B8B2A7", "grid.color": "#D9D3C7",
        "grid.linestyle": ":", "grid.alpha": 0.9,
    },
    "dark": {  # dark slide / poster look; pairs well with --theme plasma|viridis|coral
        **_DESPINE, "figure.facecolor": "#1B1D2A", "axes.facecolor": "#1B1D2A",
        "savefig.facecolor": "#1B1D2A", "text.color": "#E8E8F0", "axes.labelcolor": "#E8E8F0",
        "xtick.color": "#C9CBD8", "ytick.color": "#C9CBD8", "axes.edgecolor": "#5A5D72",
        "grid.color": "#3A3D52", "grid.alpha": 0.9,
    },
}
DEFAULT_STYLE = "default"


def resolve_colors(theme: str, rates: np.ndarray) -> list:
    """Return one color per bar for the given theme."""
    if theme.startswith("#"):
        return [theme] * len(rates)
    if theme.lower() in SOLID_THEMES:
        return [SOLID_THEMES[theme.lower()]] * len(rates)
    try:
        cmap = matplotlib.colormaps[theme]
    except KeyError:
        raise ValueError(
            f"Unknown theme '{theme}'. Solid: {list(SOLID_THEMES)}; "
            f"gradient (matplotlib colormap), e.g. {GRADIENT_EXAMPLES}; or a hex like '#8E44AD'."
        ) from None
    # keep the very light end of sequential maps readable on a white background
    return [cmap(0.15 + 0.85 * float(r) / 100) for r in rates]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_records(path: Path) -> list[dict]:
    """Load a JSON list, a JSON dict wrapping a list, or JSONL."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(obj, dict):
        lists = [v for v in obj.values() if isinstance(v, list)]
        if not lists:
            raise ValueError(f"{path}: JSON is a dict with no list inside.")
        obj = lists[0]
    if not obj or not isinstance(obj[0], dict):
        raise ValueError(f"{path}: expected a non-empty list of objects.")
    return obj


def _to_label(v) -> int:
    if isinstance(v, (bool, int, float)):
        return int(bool(v))
    s = str(v).strip().lower()
    if s in _POS:
        return 1
    if s in _NEG:
        return 0
    raise ValueError(f"Cannot interpret label value: {v!r}")


def extract_samples(
    records: list[dict],
    answer_key: str | None = None,
    label_key: str | None = None,
    pair_keys: tuple[str, str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Return (answer_texts, labels) with label 1 = hallucinated."""
    first = records[0]
    texts: list[str] = []
    labels: list[int] = []

    # --- paired layout -------------------------------------------------------
    pair = pair_keys
    if pair is None and answer_key is None:
        pair = next((p for p in PAIR_KEYS if p[0] in first and p[1] in first), None)
    if pair is not None:
        good, bad = pair
        for r in records:
            texts += [str(r[good]), str(r[bad])]
            labels += [0, 1]
        return texts, np.array(labels)

    # --- flat layout ---------------------------------------------------------
    answer_key = answer_key or next((k for k in ANSWER_KEYS if k in first), None)
    label_key = label_key or next((k for k in LABEL_KEYS if k in first), None)
    if answer_key is None or label_key is None:
        raise KeyError(
            "Could not detect the schema. Keys of the first record: "
            f"{list(first.keys())}. Pass --answer-key/--label-key "
            "(flat) or --pair-keys GOOD BAD (paired)."
        )
    for r in records:
        texts.append(str(r[answer_key]))
        labels.append(_to_label(r[label_key]))
    return texts, np.array(labels)


# --------------------------------------------------------------------------- #
# Length + binning
# --------------------------------------------------------------------------- #
def measure_length(texts: list[str], unit: str = "word", tokenizer: str | None = None) -> np.ndarray:
    if unit == "word":
        return np.array([len(t.split()) for t in texts])
    if unit == "char":
        return np.array([len(t) for t in texts])
    if unit == "token":
        if not tokenizer:
            raise ValueError("--unit token needs --tokenizer <hf-name>")
        from transformers import AutoTokenizer  # lazy import

        tok = AutoTokenizer.from_pretrained(tokenizer)
        return np.array([len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts])
    raise ValueError(f"Unknown unit: {unit}")


def make_edges(lengths: np.ndarray, n_bins: int, mode: str = "quantile") -> np.ndarray:
    """Bin edges e_0..e_k. Bin i = (e_{i-1}, e_i]; the first bin includes e_0."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if mode == "quantile":  # ~equal samples per bin
        q = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(lengths, q, method="higher"))
    elif mode == "uniform":  # equal-width bins
        edges = np.linspace(lengths.min(), lengths.max(), n_bins + 1)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    if len(edges) - 1 < n_bins:
        print(f"[warn] only {len(edges) - 1} distinct bins possible (ties in length); asked for {n_bins}.")
    return edges


def assign_bins(lengths: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges[1:-1], lengths, side="left")


def bin_labels(edges: np.ndarray) -> list[str]:
    integer = np.allclose(edges, np.round(edges))
    out = []
    for i in range(len(edges) - 1):
        if integer:
            lo = int(edges[i]) if i == 0 else int(edges[i]) + 1
            hi = int(edges[i + 1])
            out.append(f"{lo}" if lo == hi else f"{lo}–{hi}")
        else:
            out.append(f"{edges[i]:.1f}–{edges[i + 1]:.1f}")
    return out


def hallucination_rate_per_bin(lengths: np.ndarray, labels: np.ndarray, edges: np.ndarray):
    b = assign_bins(lengths, edges)
    k = len(edges) - 1
    n = np.array([(b == i).sum() for i in range(k)])
    pos = np.array([labels[b == i].sum() for i in range(k)])
    rate = np.divide(pos, n, out=np.zeros(k), where=n > 0) * 100
    return n, pos, rate


# --------------------------------------------------------------------------- #
# Shared bar drawing (vertical or horizontal)
# --------------------------------------------------------------------------- #
def _suffix(horizontal: bool, style: str, theme: str) -> str:
    """Suffix for auto file names so different looks never overwrite each other."""
    out = "_h" if horizontal else ""
    if style != DEFAULT_STYLE:
        out += f"_{style}"
    if theme != DEFAULT_THEME:
        out += f"_{theme.lstrip('#')}"
    return out


def _bar_figure(
    names: list[str],
    rate: np.ndarray,
    n: list[int],
    *,
    category_label: str,
    out_path: Path,
    theme: str,
    style: str,
    horizontal: bool,
    show_n: bool,
    title: str | None,
    figsize: tuple[float, float],
    bar_size: float,
    dpi: int,
) -> None:
    if style not in STYLES:
        raise ValueError(f"Unknown style '{style}'. Available: {list(STYLES)}")
    k = len(names)
    colors = resolve_colors(theme, rate)
    pos = np.arange(k)

    with plt.rc_context(STYLES[style]):
        fig, ax = plt.subplots(figsize=figsize)
        edge = plt.rcParams["axes.facecolor"]

        if horizontal:  # categories on y (first bin on top), rate on x
            ax.barh(pos, rate, height=bar_size, color=colors, edgecolor=edge)
            for i, r in enumerate(rate):
                ax.text(r + 1.2, i, f"{r:.1f}%", ha="left", va="center", fontsize=9)
            ax.set_yticks(pos)
            ax.set_yticklabels([f"{nm}  (n={ni})" if show_n else nm for nm, ni in zip(names, n)])
            ax.set_xlim(0, 115)
            ax.set_xticks(range(0, 101, 20))
            ax.set_xlabel("Hallucination rate (%)")
            ax.set_ylabel(category_label)
            ax.set_ylim(k - 0.5, -0.5)
            ax.grid(True, axis="x")
        else:
            ax.bar(pos, rate, width=bar_size, color=colors, edgecolor=edge)
            for i, r in enumerate(rate):
                ax.text(i, r + 1.5, f"{r:.1f}%", ha="center", va="bottom", fontsize=9)
            ax.set_xticks(pos)
            ax.set_xticklabels([f"{nm}\n(n={ni})" if show_n else nm for nm, ni in zip(names, n)])
            if k <= 3:  # few bars: keep them from stretching across the whole axis
                ax.set_xlim(-0.6, k - 0.4)
            ax.set_ylim(0, 112)
            ax.set_yticks(range(0, 101, 20))
            ax.set_xlabel(category_label)
            ax.set_ylabel("Hallucination rate (%)")
            ax.grid(True, axis="y")

        ax.set_axisbelow(True)
        if title:
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 4a: hallucination rate by answer-length bin
# --------------------------------------------------------------------------- #
def plot_length_bin_hist(
    data_path: Path | str = DEFAULT_DATA,
    out_dir: Path | str = DEFAULT_OUT_DIR,
    n_bins: int = 5,
    mode: str = "quantile",
    unit: str = "word",
    tokenizer: str | None = None,
    answer_key: str | None = None,
    label_key: str | None = None,
    pair_keys: tuple[str, str] | None = None,
    out_name: str | None = None,
    show_n: bool = True,
    theme: str = DEFAULT_THEME,
    style: str = DEFAULT_STYLE,
    horizontal: bool = False,
    title: bool = False,
    dpi: int = 200,
) -> Path:
    records = load_records(Path(data_path))
    texts, labels = extract_samples(records, answer_key, label_key, pair_keys)
    lengths = measure_length(texts, unit, tokenizer)

    edges = make_edges(lengths, n_bins, mode)
    n, pos, rate = hallucination_rate_per_bin(lengths, labels, edges)
    names = bin_labels(edges)
    k = len(names)

    print(f"{len(labels)} samples | overall hallucination rate {labels.mean() * 100:.1f}% | unit={unit} mode={mode}")
    print(f"{'bin':>12} {'n':>7} {'halluc':>7} {'rate':>7}")
    for nm, ni, pi, ri in zip(names, n, pos, rate):
        print(f"{nm:>12} {ni:>7d} {pi:>7d} {ri:>6.1f}%")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = out_name or f"{Path(data_path).stem}_len_hist_{k}bins_{unit}_{mode}{_suffix(horizontal, style, theme)}.png"
    out_path = out_dir / out_name
    figsize = (10.0, max(3.4, 0.6 * k + 1.8)) if horizontal else (max(6.5, 1.0 * k + 3), 4.8)
    _bar_figure(
        names, rate, [int(x) for x in n],
        category_label=f"Answer length bin ({unit}s, {mode} bins)",
        out_path=out_path, theme=theme, style=style, horizontal=horizontal, show_n=show_n,
        title=f"Hallucination rate by answer length — {Path(data_path).stem}" if title else None,
        figsize=figsize, bar_size=0.8 if horizontal else 0.95, dpi=dpi,
    )
    print(f"saved -> {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# Fig 4b: hallucination rate by "ends with period"
# --------------------------------------------------------------------------- #
GROUP_ORDER = ["Yes", "No"]  # first -> last (left -> right, or top -> bottom if horizontal)


def ends_with_period(text: str) -> bool:
    """True if the answer (ignoring trailing whitespace) ends with a '.'."""
    return text.rstrip().endswith(".")


def hallucination_rate_by_period(texts: list[str], labels: np.ndarray):
    """Return {group: (n, n_halluc, rate_pct)} for group in GROUP_ORDER."""
    flag = np.array([ends_with_period(t) for t in texts])
    out = {}
    for name, mask in (("Yes", flag), ("No", ~flag)):
        n = int(mask.sum())
        pos = int(labels[mask].sum())
        out[name] = (n, pos, pos / n * 100 if n else 0.0)
    return out


def plot_period_hist(
    data_path: Path | str = DEFAULT_DATA,
    out_dir: Path | str = DEFAULT_OUT_DIR,
    answer_key: str | None = None,
    label_key: str | None = None,
    pair_keys: tuple[str, str] | None = None,
    out_name: str | None = None,
    show_n: bool = True,
    theme: str = DEFAULT_THEME,
    style: str = DEFAULT_STYLE,
    horizontal: bool = False,
    title: bool = False,
    dpi: int = 200,
) -> Path:
    records = load_records(Path(data_path))
    texts, labels = extract_samples(records, answer_key, label_key, pair_keys)
    stats = hallucination_rate_by_period(texts, labels)

    print(f"{len(labels)} samples | overall hallucination rate {labels.mean() * 100:.1f}%")
    print(f"{'ends with .':>12} {'n':>7} {'halluc':>7} {'rate':>7}")
    for g in GROUP_ORDER:
        n, pos, r = stats[g]
        print(f"{g:>12} {n:>7d} {pos:>7d} {r:>6.1f}%")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = out_name or f"{Path(data_path).stem}_period_hist{_suffix(horizontal, style, theme)}.png"
    out_path = out_dir / out_name
    _bar_figure(
        list(GROUP_ORDER), np.array([stats[g][2] for g in GROUP_ORDER]), [stats[g][0] for g in GROUP_ORDER],
        category_label="Ends with period",
        out_path=out_path, theme=theme, style=style, horizontal=horizontal, show_n=show_n,
        title=f"Hallucination rate by ending period — {Path(data_path).stem}" if title else None,
        figsize=(6.5, 3.0) if horizontal else (4.2, 4.8),  # 4a/4b share the same height when vertical
        bar_size=0.55 if horizontal else 0.6, dpi=dpi,
    )
    print(f"saved -> {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(DEFAULT_DATA), help="path to halueval_en.json")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--fig", choices=["a", "b", "both"], default="both",
                   help="a = length-bin histogram, b = ends-with-period Yes/No, both = default")
    p.add_argument("--out-name", default=None, help="Fig 4a output file name (default: auto from settings)")
    p.add_argument("--out-name-b", default=None, help="Fig 4b output file name (default: auto)")
    p.add_argument("--n-bins", type=int, default=5, help="number of length bins (Fig 4a)")
    p.add_argument("--mode", choices=["quantile", "uniform"], default="quantile",
                   help="quantile = ~equal samples per bin, uniform = equal-width bins")
    p.add_argument("--unit", choices=["word", "char", "token"], default="word")
    p.add_argument("--tokenizer", default=None, help="HF tokenizer name, only for --unit token")
    p.add_argument("--answer-key", default=None, help="flat layout: field holding the answer text")
    p.add_argument("--label-key", default=None, help="flat layout: field holding the hallucination label")
    p.add_argument("--pair-keys", nargs=2, metavar=("GOOD", "BAD"), default=None,
                   help="paired layout: field names of correct / hallucinated answer")
    p.add_argument("--theme", default=DEFAULT_THEME,
                   help=f"bar colors: {list(SOLID_THEMES)} | any matplotlib colormap (gradient by rate) | hex like '#8E44AD'")
    p.add_argument("--style", default=DEFAULT_STYLE, choices=list(STYLES),
                   help="figure look: default | clean | paper | soft | dark")
    p.add_argument("--horizontal", action="store_true", help="horizontal bars (categories on the y axis)")
    p.add_argument("--title", action="store_true", help="add an automatic title (off by default)")
    p.add_argument("--hide-n", action="store_true", help="do not show n in the bin labels")
    p.add_argument("--dpi", type=int, default=200)
    a = p.parse_args()

    common = dict(
        data_path=a.data, out_dir=a.out_dir, answer_key=a.answer_key, label_key=a.label_key,
        pair_keys=tuple(a.pair_keys) if a.pair_keys else None, show_n=not a.hide_n,
        theme=a.theme, style=a.style, horizontal=a.horizontal, title=a.title, dpi=a.dpi,
    )
    if a.fig in ("a", "both"):
        plot_length_bin_hist(n_bins=a.n_bins, mode=a.mode, unit=a.unit, tokenizer=a.tokenizer,
                             out_name=a.out_name, **common)
    if a.fig in ("b", "both"):
        plot_period_hist(out_name=a.out_name_b, **common)


if __name__ == "__main__":
    main()
