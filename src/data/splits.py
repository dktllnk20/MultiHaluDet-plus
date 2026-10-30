"""Group-aware splits and pair utilities.

The hard-negative TriviaQA file holds exactly two rows per question (one correct answer = label 0,
one wrong answer = label 1). A row-level split puts the two siblings on different sides, so the
split has to be made per question ("group").
"""
import numpy as np
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold,
                                     train_test_split)


def make_groups(samples):
    """Integer id per distinct question text, in order of first appearance."""
    ids = {}
    return np.array([ids.setdefault(s["question"], len(ids)) for s in samples], dtype=np.int64)


def rows_to_keep(valid, groups, pair_center):
    """Raw row indices used in an experiment: rows whose extraction succeeded; with pair centering
    also only rows whose question still has at least two rows (a lone row cannot be centred)."""
    keep = np.flatnonzero(valid)
    if pair_center:
        g = groups[keep]
        counts = np.bincount(g, minlength=int(g.max()) + 1)
        keep = keep[counts[g] >= 2]
    return keep


def outer_split(y, groups, test_size, seed, group_split):
    """(train_idx, test_idx). Without group_split this reproduces the original
    train_test_split(..., stratify=y, random_state=seed) partition exactly."""
    idx = np.arange(len(y))
    if group_split:
        if groups is None:
            raise ValueError("group_split needs group ids")
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        return next(splitter.split(idx, y, groups))
    return train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)


def inner_splitter(n_splits, seed, group_split):
    """Use as splitter.split(X, y, groups); groups is ignored when group_split is False."""
    if group_split:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def center_by_group(x, groups):
    """Subtract the per-group mean (over the rows of the same question) from every row.

    For a pair this keeps only the difference between the two answers, which removes everything
    that depends on the question alone (its difficulty, topic, length). It uses no labels, but at
    test time it needs all rows of a question to be present in the test set.
    """
    x = np.asarray(x, dtype=np.float32)
    g = np.unique(groups, return_inverse=True)[1]
    flat = x.reshape(len(x), -1)
    order = np.argsort(g, kind="stable")
    gs = g[order]
    starts = np.r_[0, np.flatnonzero(np.diff(gs)) + 1]
    sums = np.add.reduceat(flat[order].astype(np.float64), starts, axis=0)
    counts = np.diff(np.r_[starts, len(gs)])[:, None]
    means = sums / counts
    return (flat - means[g]).astype(np.float32).reshape(x.shape)


def pair_accuracy(scores, y, groups):
    """Share of questions where the wrong answer (label 1) scores higher than the correct one
    (label 0). Questions that do not have both labels are skipped. Returns (accuracy, n_pairs)."""
    scores, y, groups = map(np.asarray, (scores, y, groups))
    hit = n = 0
    for g in np.unique(groups):
        m = groups == g
        if y[m].min() == y[m].max():
            continue
        s1, s0 = scores[m & (y == 1)].max(), scores[m & (y == 0)].min()
        hit += (s1 > s0) + 0.5 * (s1 == s0)
        n += 1
    return (hit / n if n else float("nan")), n
