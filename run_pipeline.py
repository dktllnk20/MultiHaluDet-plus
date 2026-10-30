"""Runs feature extraction, optional v2 probes, model training, and ensembling.
Uses configuration and stages from src/; run from any directory with Python.
Creates feature outputs, result plots, and per-model logs under the project root.
"""
import os

# Set before torch initializes CUDA to reduce memory fragmentation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import gc
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.config import get_config, MODEL_REGISTRY, PIPELINE_MODELS
from src.data.loader import load_halueval, load_triviaqa
from src.data.feature_extractor import extract_dataset, SEQ_DIM, GLOB_BASE_DIM
from src.data.probe_extractor import extract_dataset_v2, load_v2
from src.data.splits import (center_by_group, inner_splitter, make_groups, outer_split,
                             pair_accuracy, rows_to_keep)
from src.probe.layer_probe import run_layer_probes
from src.training.trainer import train_deep_model_fold, extract_features_batch, extract_logits_batch
from src.ensemble.meta_learner import get_ensemble, calibrate_ensemble
from src.utils.metrics import find_best_thresholds, evaluate_all, compute_uncertainty_metrics
from src.utils.visualization import plot_publication_results
from src.settings.runtime import ROOT_DIR, capture_run_log, ensure_directory


GLOB_GROUPS = {"base": "base_", "ll": "ans_", "yn": "v_"}


def _select_glob(X_glob, names, use):
    """Keep the selected v2 global-feature groups."""
    groups = [g.strip() for g in use.split(",") if g.strip()]
    bad = [g for g in groups if g not in GLOB_GROUPS]
    if bad:
        raise SystemExit(f"Invalid --glob_use groups {bad}; choose from {list(GLOB_GROUPS)}")
    prefixes = tuple(GLOB_GROUPS[g] for g in groups)
    cols = [i for i, n in enumerate(names) if n.startswith(prefixes)]
    if not cols:
        raise SystemExit(f"--glob_use {use} selected no columns")
    return X_glob[:, cols]


def _opt(args, name, default):
    """args.<name>, or default when the caller (e.g. an src/rq script) built args without it."""
    return getattr(args, name, default)


def _key(args, tag=True):
    """Build the feature or run output key, including the selected feature version and record limit."""
    base = f"{args.dataset}_{args.lang}_{args.model}"
    if _opt(args, "limit", None) is not None:
        base += f"_limit{args.limit}"
    feat = "v2" if _opt(args, "v2", False) else _opt(args, "feat", "v1")
    if tag and feat == "v2":
        base += f"_v2{_opt(args, 'prompt_mode', 'verify')}"
    tag_value = _opt(args, "tag", "")
    return f"{base}_{tag_value}" if tag and tag_value else base


def _fkey2(args):
    limit = f"_limit{args.limit}" if _opt(args, "limit", None) is not None else ""
    return f"{args.dataset}_{args.lang}_{args.model}{limit}_{_opt(args, 'prompt_mode', 'verify')}"


def _feat_dir2(args):
    return f"results/features_v2/{_fkey2(args)}"


def _probe_path(args, config):
    views = "-".join(v.strip() for v in _opt(args, "views", "last,amean").split(","))
    tag = (f"{views}_s{config.seed}_pca{config.probe_pca_dim}_C{config.probe_C}"
           + ("_pc" if _opt(args, "pair_center", False) else ""))
    return f"results/probe/{_fkey2(args)}_{tag}.npz"


def _load_samples(args, config):
    if args.dataset == "halueval":
        samples = load_halueval(lang=args.lang)
    else:
        samples = load_triviaqa(lang=args.lang, seed=config.seed)
    limit = _opt(args, "limit", None)
    return samples[:limit] if limit is not None else samples


def _load_llm(args, config):
    config.model_name = MODEL_REGISTRY[args.model]
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    quant = _opt(args, "quant", "4bit")
    if torch.cuda.is_available():
        if quant == "fp16":
            kwargs = {}
        elif quant == "8bit":
            kwargs = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
        else:
            # 4-bit NF4: Mistral-7B takes ~4 GB and fits entirely on a 6 GB GPU.
            kwargs = {"quantization_config": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)}
        # device_map={"": 0} forces everything onto GPU 0 (no silent CPU/disk offload).
        model_llm = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=config.model_dtype, device_map={"": 0}, **kwargs)
    else:
        model_llm = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=config.model_dtype)
    model_llm.eval()
    return tokenizer, model_llm


def stage_1_extract(args, config):
    print("\n--- Stage 1: feature extraction ---")
    print(f"Dataset: {args.dataset} ({args.lang}), Model: {args.model}")

    samples = _load_samples(args, config)
    tokenizer, model_llm = _load_llm(args, config)

    key = _key(args, tag=False)
    ensure_directory("results/features")
    ckpt_path = f'results/features/partial_{key}.npz'
    X_seq, X_glob, y = extract_dataset(samples, tokenizer, model_llm, config, ckpt_path,
                                       batch_size=args.extract_batch_size)
    X_seq = np.nan_to_num(X_seq, nan=0.0)
    X_glob = np.nan_to_num(X_glob, nan=0.0)

    np.save(f'results/features/X_seq_{key}.npy', X_seq)
    np.save(f'results/features/X_glob_{key}.npy', X_glob)
    np.save(f'results/features/y_{key}.npy', y)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    del model_llm
    torch.cuda.empty_cache()
    gc.collect()
    print("Stage 1 completed. Features saved.")


def stage_1_extract_v2(args, config):
    print(f"\n--- Stage 1 (v2): feature extraction, prompt={_opt(args, 'prompt_mode', 'verify')} ---")
    print(f"Dataset: {args.dataset} ({args.lang}), Model: {args.model}")
    samples = _load_samples(args, config)
    tokenizer, model_llm = _load_llm(args, config)
    views = tuple(v.strip() for v in args.views.split(",") if v.strip())
    out_dir = _feat_dir2(args)
    extract_dataset_v2(samples, tokenizer, model_llm, config, out_dir, mode=_opt(args, "prompt_mode", "verify"),
                       views=views, batch_size=args.extract_batch_size)
    del model_llm
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Stage 1 (v2) completed. Features saved in {out_dir}")


def _v2_context(args, config):
    """Rows and the outer split shared by the probe stage and the training stage."""
    try:
        d = load_v2(_feat_dir2(args))
    except FileNotFoundError:
        raise SystemExit(f"V2 features not found in {_feat_dir2(args)}. Run with --stage extract first.")
    keep = rows_to_keep(d["valid"], d["groups"], _opt(args, "pair_center", False))
    y, groups = d["y"][keep], d["groups"][keep]
    tr, te = outer_split(y, groups, config.test_size, config.seed, True)
    return d, keep, y, groups, tr, te


def stage_2_probe(args, config):
    print(f"\n--- V2 linear probes (seed {config.seed}) ---")
    path = _probe_path(args, config)
    if os.path.exists(path) and not _opt(args, "redo_probe", False):
        print(f"{path} exists; reusing the cached probes")
        return
    d, keep, y, groups, tr, te = _v2_context(args, config)
    views = d["meta"]["views"]
    wanted = [v.strip() for v in args.views.split(",") if v.strip()]
    if wanted != views:
        raise SystemExit(f"--views {wanted} differs from the extracted views {views}")
    probe, report = run_layer_probes(
        _feat_dir2(args), views, d["meta"]["L"], keep, y, groups, tr, te,
        pair_center=_opt(args, "pair_center", False),
        n_folds=config.probe_folds, pca_dim=config.probe_pca_dim, C=config.probe_C,
        seed=config.seed, n_jobs=config.probe_jobs)
    print(report)
    ensure_directory("results/probe")
    np.savez(path, probe=probe, views=np.array(views))
    with open(path.replace(".npz", "_auc.txt"), "w") as f:
        f.write(report + "\n")
    print(f"Saved probe logits to {path}")


def _load_training_data(args, config):
    """Load feature arrays and their train/test indices."""
    feat = "v2" if _opt(args, "v2", False) else _opt(args, "feat", "v1")
    pair_center = _opt(args, "pair_center", False)
    group_split = bool(_opt(args, "group_split", False)) or feat == "v2"
    if pair_center and not group_split:
        raise SystemExit("--pair_center requires grouped splitting (enabled automatically for v2)")

    if feat == "v2":
        d, keep, y, groups, tr, te = _v2_context(args, config)
        X_seq, X_glob = d["seq"][keep], d["glob"][keep]
        X_glob = _select_glob(X_glob, d["meta"]["glob_names"], _opt(args, "glob_use", "base,ll,yn"))
        if pair_center:
            X_seq, X_glob = center_by_group(X_seq, groups), center_by_group(X_glob, groups)
        use = args.probe_use
        if use != "none":
            path = _probe_path(args, config)
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} not found. Run with --stage extract to create v2 probes.")
            probe = np.load(path)["probe"]                                        # [N, L, V]
            if use in ("both", "seq"):
                X_seq = np.concatenate([X_seq, probe], axis=-1)
            if use in ("both", "glob"):
                half = probe.shape[1] // 2
                X_glob = np.concatenate([X_glob, probe.mean(1), probe[:, half:].mean(1)], axis=1)
        return X_seq, X_glob, y, groups, tr, te, True

    feat_key = _key(args, tag=False)
    try:
        X_seq = np.load(f'results/features/X_seq_{feat_key}.npy')
        X_glob = np.load(f'results/features/X_glob_{feat_key}.npy')
        y = np.load(f'results/features/y_{feat_key}.npy')
    except FileNotFoundError:
        raise FileNotFoundError("Features not found. Run with --stage extract first.")
    # Added columns come after the original ones, so dropping them is just slicing. This also lets
    # feature files that still carry the old logit-lens channels (seq 14) be reused as they are.
    X_seq = X_seq[:, :, :SEQ_DIM]
    if not config.use_answer_ll:
        X_glob = X_glob[:, :GLOB_BASE_DIM]

    groups = None
    if group_split:
        groups = make_groups(_load_samples(args, config))
        if len(groups) != len(y):
            raise SystemExit(f"{len(y)} feature rows but {len(groups)} samples; grouped splitting requires aligned rows")
        if pair_center:
            if (np.bincount(groups)[groups] < 2).any():
                raise SystemExit("--pair_center needs every question to have at least two rows")
            X_seq, X_glob = center_by_group(X_seq, groups), center_by_group(X_glob, groups)
    tr, te = outer_split(y, groups, config.test_size, config.seed, group_split)
    return X_seq, X_glob, y, groups, tr, te, group_split


def _train_fold(fold, tr_idx, val_idx, X_seq, X_glob, y, X_seq_test, X_glob_test, config):
    """Train one inner fold; return its out-of-fold and test outputs.

    Runs in its own process when config.fold_workers > 1, so it seeds itself.
    """
    seed = config.seed + fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config.fold_workers > 1 and not torch.cuda.is_available():
        # CPU only: share the cores between the fold processes.
        torch.set_num_threads(max(1, torch.get_num_threads() // config.fold_workers))
    model, best_auc = train_deep_model_fold(
        X_seq[tr_idx], X_glob[tr_idx], y[tr_idx],
        X_seq[val_idx], X_glob[val_idx], y[val_idx],
        config, log_prefix=f"[fold {fold + 1}] "
    )
    return (val_idx,
            extract_features_batch(model, X_seq[val_idx], X_glob[val_idx]),
            extract_features_batch(model, X_seq_test, X_glob_test),
            extract_logits_batch(model, X_seq[val_idx], X_glob[val_idx]),
            extract_logits_batch(model, X_seq_test, X_glob_test),
            best_auc)


def stage_2_3_train_oof(args, config):
    print("\n--- Detector training and out-of-fold feature generation ---")
    print(f"seed={config.seed} mixup={config.use_mixup} cutmix={config.use_cutmix} "
          f"layer_delta={config.use_layer_delta} layer_scale={config.scale_layer_weights} "
          f"rank_weight={config.rank_loss_weight} answer_ll={config.use_answer_ll} "
          f"stack={args.stack_mode}")
    data = _load_training_data(args, config)
    X_seq, X_glob, y, groups, tr, te, group_split = data
    print(f"Input features: seq {X_seq.shape[1:]}, global {X_glob.shape[1:]} | "
          f"group_split={group_split} pair_center={_opt(args, 'pair_center', False)}")

    X_seq_train, X_seq_test, X_glob_train, X_glob_test = X_seq[tr], X_seq[te], X_glob[tr], X_glob[te]
    y_train, y_test = y[tr], y[te]
    groups_train = groups[tr] if groups is not None else None
    groups_test = groups[te] if groups is not None else None

    N, L, F_seq = X_seq_train.shape
    scaler_seq = RobustScaler()
    X_seq_train = scaler_seq.fit_transform(X_seq_train.reshape(-1, F_seq)).reshape(N, L, F_seq)
    X_seq_test = scaler_seq.transform(X_seq_test.reshape(-1, F_seq)).reshape(X_seq_test.shape[0], L, F_seq)

    scaler_glob = RobustScaler()
    X_glob_train = scaler_glob.fit_transform(X_glob_train)
    X_glob_test = scaler_glob.transform(X_glob_test)

    skf = inner_splitter(config.n_inner_folds, config.seed, group_split)
    feat_dim = config.hidden_dim + (config.hidden_dim // 2)
    oof_features = np.zeros((len(y_train), feat_dim))
    test_features_accum = np.zeros((len(y_test), feat_dim))
    oof_logits = np.zeros(len(y_train))
    test_logits = np.zeros((config.n_inner_folds, len(y_test)))

    jobs = [(fold, tr_idx, val_idx, X_seq_train, X_glob_train, y_train, X_seq_test, X_glob_test, config)
            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_seq_train, y_train, groups_train))]
    if config.fold_workers > 1:
        # The model is small, so the folds share one GPU; "spawn" is required with CUDA.
        with ProcessPoolExecutor(config.fold_workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(_train_fold, *zip(*jobs)))
    else:
        results = [_train_fold(*job) for job in jobs]

    for fold, (val_idx, feats_val, feats_test, logit_val, logit_test, best_auc) in enumerate(results):
        print(f"Fold {fold+1}/{config.n_inner_folds}: best validation AUC {best_auc:.4f}")
        oof_features[val_idx] = feats_val
        test_features_accum += feats_test
        oof_logits[val_idx] = logit_val
        test_logits[fold] = logit_test

    # Deep-network-only reference, useful when comparing ablations. The OOF number is slightly
    # optimistic because each fold's best epoch was picked on that same validation fold.
    print(f"Deep-only AUC: OOF logit (train) {roc_auc_score(y_train, oof_logits):.4f} | "
          f"fold-averaged logit (test) {roc_auc_score(y_test, test_logits.mean(axis=0)):.4f}")

    key = _key(args)
    ensure_directory("results/oof")
    # "vector" stack mode: 576-d features (test = average over the 5 unrelated feature spaces)
    np.save(f'results/oof/X_train_deep_{key}.npy', oof_features)
    np.save(f'results/oof/X_test_deep_{key}.npy', test_features_accum / config.n_inner_folds)
    # "logit" stack mode: one logit per fold network + the shared-space global features
    np.save(f'results/oof/oof_logits_{key}.npy', oof_logits)
    np.save(f'results/oof/test_logits_{key}.npy', test_logits)
    np.save(f'results/oof/X_glob_train_{key}.npy', X_glob_train)
    np.save(f'results/oof/X_glob_test_{key}.npy', X_glob_test)
    np.save(f'results/oof/y_train_{key}.npy', y_train)
    np.save(f'results/oof/y_test_{key}.npy', y_test)
    for name, arr in (("g_train", groups_train), ("g_test", groups_test)):
        path = f'results/oof/{name}_{key}.npy'
        if arr is not None:
            np.save(path, arr)          # question ids, for the grouped stacker folds and pair accuracy
        elif os.path.exists(path):
            os.remove(path)             # stale file of an earlier grouped run
    print("Training and OOF generation completed.")


def stage_4_ensemble(args, config):
    print(f"\n--- Ensemble evaluation (stack mode: {args.stack_mode}) ---")
    key = _key(args)
    try:
        y_train = np.load(f'results/oof/y_train_{key}.npy')
        y_test = np.load(f'results/oof/y_test_{key}.npy')
        if args.stack_mode == "vector":
            X_train = np.load(f'results/oof/X_train_deep_{key}.npy')
            X_tests = [np.load(f'results/oof/X_test_deep_{key}.npy')]
        else:
            oof_logits = np.load(f'results/oof/oof_logits_{key}.npy')
            test_logits = np.load(f'results/oof/test_logits_{key}.npy')
            glob_train = np.load(f'results/oof/X_glob_train_{key}.npy')
            glob_test = np.load(f'results/oof/X_glob_test_{key}.npy')
            X_train = np.column_stack([oof_logits, glob_train])
            # One test matrix per fold network: the stacker is trained on rows that each carry a
            # single network's logit, so it is also applied that way; the outputs are averaged below.
            X_tests = [np.column_stack([test_logits[k], glob_test]) for k in range(len(test_logits))]
    except FileNotFoundError:
        raise FileNotFoundError("OOF outputs not found. Run with --stage train_oof first.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_tests = [scaler.transform(x) for x in X_tests]

    g_train = g_test = None
    if os.path.exists(f'results/oof/g_train_{key}.npy'):
        g_train = np.load(f'results/oof/g_train_{key}.npy')
        g_test = np.load(f'results/oof/g_test_{key}.npy')

    ensemble = get_ensemble(config.seed, _opt(args, "stacker", "full"))
    calibrated_model, _ = calibrate_ensemble(ensemble, X_train, y_train, g_train)

    probs = np.mean([calibrated_model.predict_proba(x)[:, 1] for x in X_tests], axis=0)

    thresholds = find_best_thresholds(probs, y_test)
    results = evaluate_all(probs, y_test, thresholds['youden'])
    uncertainty = compute_uncertainty_metrics(probs, y_test, (probs >= thresholds['youden']).astype(int))

    print("\nResults Summary:")
    print(f"AUC: {results['auc']:.4f}, F1: {results['f1']:.4f}, Acc: {results['acc']:.4f}")
    if g_test is not None:
        pair_acc, n_pairs = pair_accuracy(probs, y_test, g_test)
        print(f"Pair accuracy (wrong answer scored above the correct one, {n_pairs} questions): {pair_acc:.4f}")

    ensure_directory("results/plots")
    plot_path = f"results/plots/results_{key}.png"
    plot_publication_results(
        probs, y_test, (probs >= thresholds['youden']).astype(int),
        uncertainty, results['auc'], results['ap'], results['cm'],
        plot_path
    )
    print(f"Saved results to {plot_path}")


def run_model(args):
    args.views = "last,amean"
    args.prompt_mode = "verify"
    args.probe_use = "both"
    args.glob_use = "base,ll,yn"
    args.pair_center = False
    args.stack_mode = "vector"
    args.extract_batch_size = 4

    config = get_config()
    config.seed = args.seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if args.stage in ("extract", "all"):
        (stage_1_extract_v2 if args.v2 else stage_1_extract)(args, config)
        if args.v2:
            stage_2_probe(args, config)
    if args.stage in ("train_oof", "all"):
        stage_2_3_train_oof(args, config)
    if args.stage in ("ensemble", "all"):
        stage_4_ensemble(args, config)


def main():
    parser = argparse.ArgumentParser(
        description="Extract features, train the detector, and evaluate its ensemble.",
        allow_abbrev=False,
    )
    parser.add_argument("--v2", action="store_true",
                        help="Use v2 hidden-state features and probes (default: Base features).")
    parser.add_argument("--models", nargs="+", choices=PIPELINE_MODELS, default=["mistral-7b"],
                        help="One or both supported models (default: mistral-7b).")
    parser.add_argument("--dataset", choices=["halueval", "triviaqa"], default="halueval",
                        help="Dataset to process (default: halueval).")
    parser.add_argument("--lang", choices=["en", "fr", "bn", "vi"], default="en",
                        help="Dataset language (default: en).")
    parser.add_argument("--stage", choices=["extract", "train_oof", "ensemble", "all"], default="all",
                        help="extract includes v2 probes; all runs every stage (default: all).")
    parser.add_argument("--limit", type=int, help="Process at most this many dataset records (positive integer).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer.")

    os.chdir(ROOT_DIR)
    for model in args.models:
        args.model = model
        family = "mistral" if model.startswith("mistral") else "llama"
        variant = "v2" if args.v2 else "base"
        log_path = ROOT_DIR / "logs" / family / (
            f"{args.dataset}_{args.lang}_{variant}_s{args.seed}.log"
        )
        with capture_run_log(log_path):
            print(f"Model: {model}")
            print(f"Dataset: {args.dataset} | language: {args.lang} | features: {variant}")
            limit_text = args.limit if args.limit is not None else "all"
            print(f"Stage: {args.stage} | seed: {args.seed} | limit: {limit_text}")
            run_model(args)
        print(f"Log saved to {log_path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
