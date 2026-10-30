"""Evaluates probe-only baselines using cached RQ1 hidden-state probe scores.
It reuses the feature splits from run_pipeline.py and writes per-seed and summary CSV files.
Run this module directly after v2 probes have been generated.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.config import get_config                                   # noqa: E402
from src.data.probe_extractor import load_v2                        # noqa: E402
from src.data.splits import outer_split, pair_accuracy, rows_to_keep  # noqa: E402
from src.rq.rq1 import (DATASETS, FEAT2_DIR, LANGS, MODELS, RES_ROOT, SEEDS,   # noqa: E402
                        fkey2, probe_path)

OUT = RES_ROOT.parent / "rq1_probe_only"


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 and np.std(s) > 0 else float("nan")


def one_run(ds, lang, model, pm, views, seed):
    pp = probe_path(ds, lang, model, pm, views, seed)
    if not pp.exists():
        print(f"[skipped] missing {pp.name}")
        return None
    cfg = get_config()
    d = load_v2(str(FEAT2_DIR / fkey2(ds, lang, model, pm)))
    keep = rows_to_keep(d["valid"], d["groups"], False)
    y, g = d["y"][keep], d["groups"][keep]
    tr, te = outer_split(y, g, cfg.test_size, seed, True)
    z = np.load(pp)
    P = z["probe"]                                   # [Nk, L, V]
    pviews = [str(v) for v in z["views"]]
    if P.shape[0] != len(y) or pviews != views:
        print(f"[skipped] {pp.name}: shape {P.shape} / views {pviews} do not match "
              f"the feature data ({len(y)} rows, {views})")
        return None
    _, L, V = P.shape
    ytr, yte, gte = y[tr], y[te], g[te]

    # Normalize each layer/view column using training-set statistics.
    mu, sd = P[tr].mean(0), P[tr].std(0) + 1e-6
    Z = (P - mu) / sd

    oof = np.array([[_auc(ytr, P[tr, l, v]) for v in range(V)] for l in range(L)])
    tst = np.array([[_auc(yte, P[te, l, v]) for v in range(V)] for l in range(L)])   # Report only; never use for selection.
    layer_rows = [dict(dataset=ds, lang=lang, model=model, prompt=pm, seed=seed, view=views[v], layer=l + 1,
                       oof_auc=oof[l, v], test_auc=tst[l, v]) for l in range(L) for v in range(V)]

    scores = {}
    best_l = np.nanargmax(oof, axis=0)               # [V]
    for v in range(V):
        scores[f"best_layer_{views[v]}"] = (Z[te, best_l[v], v], f"layer {best_l[v] + 1}")
    li, vi = np.unravel_index(np.nanargmax(oof), oof.shape)
    scores["best_cell"] = (Z[te, li, vi], f"layer {li + 1} view {views[vi]}")
    scores["mean_best_views"] = (np.mean([Z[te, best_l[v], v] for v in range(V)], axis=0),
                                 ",".join(f"{views[v]}:{best_l[v] + 1}" for v in range(V)))
    flat = np.argsort(-np.nan_to_num(oof, nan=-1).ravel())[:5]
    top = [np.unravel_index(i, oof.shape) for i in flat]
    scores["mean_top5"] = (np.mean([Z[te, l, v] for l, v in top], axis=0),
                           ",".join(f"{l + 1}{views[v][0]}" for l, v in top))
    scores["mean_all"] = (Z[te].reshape(len(te), -1).mean(1), f"{L}x{V} cells")
    lr = LogisticRegression(C=0.05, max_iter=2000).fit(Z[tr].reshape(len(tr), -1), ytr)
    scores["lr_all"] = (lr.decision_function(Z[te].reshape(len(te), -1)), "C=0.05")

    rows = []
    for name, (s, info) in scores.items():
        rows.append(dict(dataset=ds, lang=lang, model=model, prompt=pm, seed=seed, method=name,
                         auc=_auc(yte, s), pair_acc=pair_accuracy(s, yte, gte)[0], n_test=len(te), selected=info))
    return rows, layer_rows


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--langs", nargs="+", default=["en"], choices=LANGS)
    ap.add_argument("--datasets", nargs="+", default=["triviaqa"], choices=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--prompt_modes", nargs="+", default=["verify", "plain"], choices=["verify", "plain"])
    ap.add_argument("--views", default="last,amean")
    a = ap.parse_args()
    views = [v.strip() for v in a.views.split(",") if v.strip()]

    rows, layer_rows = [], []
    for ds in a.datasets:
        for lang in a.langs:
            for model in a.models:
                for pm in a.prompt_modes:
                    for seed in a.seeds:
                        r = one_run(ds, lang, model, pm, views, seed)
                        if r:
                            rows += r[0]
                            layer_rows += r[1]
    if not rows:
        sys.exit("No probe files could be read.")
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_seed.csv", index=False)
    pd.DataFrame(layer_rows).to_csv(OUT / "layer_auc.csv", index=False)
    keys = ["dataset", "lang", "model", "prompt", "method"]
    summ = df.groupby(keys).agg(auc_mean=("auc", "mean"), auc_std=("auc", "std"), pair_acc=("pair_acc", "mean"),
                                n_seeds=("seed", "nunique")).reset_index()
    summ.to_csv(OUT / "summary.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4f}".format)
    print("\n=== Probe-only (test AUROC, mean/std across seeds) ===")
    print(summ.to_string(index=False))

    runs = RES_ROOT / "all_runs.csv"
    if runs.exists():
        r = pd.read_csv(runs)
        r = r[r["cond"].isin(["base", "full", "add_B", "loo_B"]) & r["dataset"].isin(a.datasets)]
        t = r.groupby(["dataset", "lang", "model", "cond"])[["deep_test_auc", "ens_auc"]].mean().reset_index()
        print("\n=== RQ1 deep/ensemble AUROC (for comparison with probe-only results) ===")
        print(t.to_string(index=False))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
