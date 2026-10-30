"""Out-of-fold linear probes on hidden-state vectors, one per (view, layer).

Each probe is StandardScaler -> PCA(pca_dim) -> LogisticRegression. Train rows get out-of-fold
decision values (folds are grouped by question), test rows the average over the fold probes, so no
row is scored by a probe that saw its label or its sibling answer.
"""
import os

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.splits import center_by_group, inner_splitter


def _probe_one(path, layer, keep, y, groups, tr, te, pair_center, n_folds, pca_dim, C, seed):
    X = np.asarray(np.load(path, mmap_mode="r")[layer])[keep].astype(np.float32)     # [Nk, D]
    if pair_center:
        X = center_by_group(X, groups)
    Xtr, Xte, ytr, gtr = X[tr], X[te], y[tr], groups[tr]
    oof = np.zeros(len(tr), np.float32)
    test = np.zeros(len(te), np.float32)
    for a, b in inner_splitter(n_folds, seed + 101, True).split(Xtr, ytr, gtr):
        pipe = make_pipeline(
            StandardScaler(),
            PCA(n_components=min(pca_dim, len(a) - 1, X.shape[1]), svd_solver="randomized", random_state=seed),
            LogisticRegression(C=C, max_iter=300))
        pipe.fit(Xtr[a], ytr[a])
        oof[b] = pipe.decision_function(Xtr[b])
        test += pipe.decision_function(Xte) / n_folds
    return oof, test


def run_layer_probes(feat_dir, views, n_layers, keep, y, groups, tr, te, *, pair_center=False,
                     n_folds=5, pca_dim=256, C=0.02, seed=42, n_jobs=4):
    """Returns (probe [Nk, L, V] float32, report text). Rows are the kept rows, in kept order."""
    tasks = [(vi, l) for vi in range(len(views)) for l in range(n_layers)]
    res = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_probe_one)(os.path.join(feat_dir, f"H_{views[vi]}.npy"), l, keep, y, groups, tr, te,
                            pair_center, n_folds, pca_dim, C, seed)
        for vi, l in tasks)
    out = np.zeros((len(y), n_layers, len(views)), np.float32)
    auc_oof = np.zeros((n_layers, len(views)))
    auc_te = np.zeros((n_layers, len(views)))
    for (vi, l), (oof, test) in zip(tasks, res):
        out[tr, l, vi], out[te, l, vi] = oof, test
        auc_oof[l, vi] = roc_auc_score(y[tr], oof)
        auc_te[l, vi] = roc_auc_score(y[te], test)
    lines = ["layer  " + "  ".join(f"{v:>7s}(oof/test)" for v in views)]
    for l in range(n_layers):
        lines.append(f"{l + 1:5d}  " + "  ".join(f"{auc_oof[l, i]:.3f}/{auc_te[l, i]:.3f}   " for i in range(len(views))))
    best = [f"{v}: best oof layer {auc_oof[:, i].argmax() + 1} ({auc_oof[:, i].max():.3f})" for i, v in enumerate(views)]
    return out, "\n".join(lines + [""] + best)
