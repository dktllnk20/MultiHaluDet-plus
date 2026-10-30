import os

# Must be set before torch initialises CUDA; reduces memory fragmentation on small GPUs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import multiprocessing
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import torch
import numpy as np
import random
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.config import get_config, MODEL_REGISTRY, PIPELINE_MODELS
from src.data.loader import load_halueval, load_triviaqa
from src.data.feature_extractor import extract_dataset, SEQ_DIM, GLOB_BASE_DIM
from src.data.probe_extractor import extract_dataset_v2, load_v2, VIEWS
from src.data.splits import (center_by_group, inner_splitter, make_groups, outer_split,
                             pair_accuracy, rows_to_keep)
from src.probe.layer_probe import run_layer_probes
from src.training.trainer import train_deep_model_fold, extract_features_batch, extract_logits_batch
from src.ensemble.meta_learner import get_ensemble, calibrate_ensemble, temperature_scale
from src.utils.metrics import find_best_thresholds, evaluate_all, compute_uncertainty_metrics
from src.utils.visualization import plot_publication_results

RQ_STEPS = ["run", "collect", "diagnostics"]
# Lưới đầy đủ khi chạy --rq N --all (khớp với mặc định của src/rq/rq1_common.py).
RQ_GRID = {
    "models": ["mistral-7b", "llama2-7b"],
    "langs": ["en", "fr", "vi", "bn"],
    "datasets": ["halueval", "triviaqa"],
    "seeds": [1, 11, 29, 37, 43],
}


GLOB_GROUPS = {"base": "base_", "ll": "ans_", "yn": "v_"}   # nhóm cột global v2 -> tiền tố trong glob_names


def _select_glob(X_glob, names, use):
    """Giữ các nhóm cột global v2 trong `use` (base: 30 cột v1; ll: 13 cột answer-LL/PMI; yn: 3 cột Yes/No)."""
    groups = [g.strip() for g in use.split(",") if g.strip()]
    bad = [g for g in groups if g not in GLOB_GROUPS]
    if bad:
        raise SystemExit(f"--glob_use: nhóm không hợp lệ {bad}; chọn từ {list(GLOB_GROUPS)}")
    prefixes = tuple(GLOB_GROUPS[g] for g in groups)
    cols = [i for i, n in enumerate(names) if n.startswith(prefixes)]
    if not cols:
        raise SystemExit(f"--glob_use {use} không chọn được cột nào")
    return X_glob[:, cols]


def _opt(args, name, default):
    """args.<name>, or default when the caller (e.g. an src/rq script) built args without it."""
    return getattr(args, name, default)


def _key(args, tag=True):
    """File-name suffix. --tag only applies to run-specific outputs (OOF, plots), never to the
    extracted LLM features, so one extraction can serve many seeds / ablations."""
    base = f"{args.dataset}_{args.lang}_{args.model}"
    if tag and _opt(args, "feat", "v1") == "v2":
        base += f"_v2{args.prompt_mode}"
    return f"{base}_{args.tag}" if tag and args.tag else base


def _fkey2(args):
    return f"{args.dataset}_{args.lang}_{args.model}_{args.prompt_mode}"


def _feat_dir2(args):
    return f"results/features_v2/{_fkey2(args)}"


def _probe_path(args, config):
    views = "-".join(v.strip() for v in args.views.split(","))
    tag = f"{views}_s{config.seed}_pca{config.probe_pca_dim}_C{config.probe_C}" + ("_pc" if args.pair_center else "")
    return f"results/probe/{_fkey2(args)}_{tag}.npz"


def _load_samples(args, config):
    if args.dataset == "halueval":
        return load_halueval(lang=args.lang)
    return load_triviaqa(lang=args.lang, seed=config.seed)


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
    print(f"\n--- Stage 1: Dynamic Layer Probing & Feature Extraction ---")
    print(f"Dataset: {args.dataset} ({args.lang}), Model: {args.model}")

    samples = _load_samples(args, config)
    tokenizer, model_llm = _load_llm(args, config)

    key = _key(args, tag=False)
    os.makedirs('results/features', exist_ok=True)
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
    print(f"\n--- Stage 1 (v2): hidden-state vectors, prompt={args.prompt_mode}, views={args.views} ---")
    print(f"Dataset: {args.dataset} ({args.lang}), Model: {args.model}, quant: {args.quant}")
    samples = _load_samples(args, config)
    tokenizer, model_llm = _load_llm(args, config)
    views = tuple(v.strip() for v in args.views.split(",") if v.strip())
    out_dir = _feat_dir2(args)
    extract_dataset_v2(samples, tokenizer, model_llm, config, out_dir, mode=args.prompt_mode,
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
        raise SystemExit(f"v2 features not found in {_feat_dir2(args)}. Run --feat v2 --stage extract first.")
    keep = rows_to_keep(d["valid"], d["groups"], args.pair_center)
    y, groups = d["y"][keep], d["groups"][keep]
    tr, te = outer_split(y, groups, config.test_size, config.seed, True)
    return d, keep, y, groups, tr, te


def stage_2_probe(args, config):
    print(f"\n--- Stage 2a (v2): out-of-fold linear probes per layer (seed {config.seed}) ---")
    path = _probe_path(args, config)
    if os.path.exists(path) and not args.redo_probe:
        print(f"{path} exists; reusing it (--redo_probe to recompute)")
        return
    d, keep, y, groups, tr, te = _v2_context(args, config)
    views = d["meta"]["views"]
    wanted = [v.strip() for v in args.views.split(",") if v.strip()]
    if wanted != views:
        raise SystemExit(f"--views {wanted} differs from the extracted views {views}")
    probe, report = run_layer_probes(
        _feat_dir2(args), views, d["meta"]["L"], keep, y, groups, tr, te, pair_center=args.pair_center,
        n_folds=config.probe_folds, pca_dim=config.probe_pca_dim, C=config.probe_C,
        seed=config.seed, n_jobs=config.probe_jobs)
    print(report)
    os.makedirs("results/probe", exist_ok=True)
    np.savez(path, probe=probe, views=np.array(views))
    with open(path.replace(".npz", "_auc.txt"), "w") as f:
        f.write(report + "\n")
    print(f"Saved probe logits to {path}")


def _load_training_data(args, config):
    """(X_seq, X_glob, y, groups, train_idx, test_idx, group_split), or None if inputs are missing."""
    feat = _opt(args, "feat", "v1")
    pair_center = _opt(args, "pair_center", False)
    group_split = bool(_opt(args, "group_split", False)) or feat == "v2"
    if pair_center and not group_split:
        raise SystemExit("--pair_center needs --group_split (or --feat v2): both answers of a question must be in the same split")

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
                print(f"{path} not found. Run --feat v2 --stage probe first (same --seed, --views, --pair_center).")
                return None
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
        print("Features not found. Please run --stage extract first.")
        return None
    # Added columns come after the original ones, so dropping them is just slicing. This also lets
    # feature files that still carry the old logit-lens channels (seq 14) be reused as they are.
    X_seq = X_seq[:, :, :SEQ_DIM]
    if not config.use_answer_ll:
        X_glob = X_glob[:, :GLOB_BASE_DIM]

    groups = None
    if group_split:
        groups = make_groups(_load_samples(args, config))
        if len(groups) != len(y):
            raise SystemExit(f"{len(y)} feature rows but {len(groups)} samples (failed rows were skipped at "
                             f"extraction); a group split needs them aligned")
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
    print(f"\n--- Stage 2 & 3: Multi-Scale Modeling & Out-of-Fold Generation ---")
    print(f"seed={config.seed} mixup={config.use_mixup} cutmix={config.use_cutmix} "
          f"layer_delta={config.use_layer_delta} layer_scale={config.scale_layer_weights} "
          f"rank_weight={config.rank_loss_weight} answer_ll={config.use_answer_ll} "
          f"stack={args.stack_mode}")
    data = _load_training_data(args, config)
    if data is None:
        return
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
    os.makedirs('results/oof', exist_ok=True)
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
    print("Stage 2 & 3 completed. OOF outputs saved.")


def stage_4_ensemble(args, config):
    print(f"\n--- Stage 4: Log-Odds Ensemble Meta-Learning (stack mode: {args.stack_mode}) ---")
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
        print("OOF outputs not found. Please run --stage train_oof first.")
        return

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

    os.makedirs('results/plots', exist_ok=True)
    plot_path = f"results/plots/results_{key}.png"
    plot_publication_results(
        probs, y_test, (probs >= thresholds['youden']).astype(int),
        uncertainty, results['auc'], results['ap'], results['cm'],
        plot_path
    )
    print(f"Saved results to {plot_path}")


def run_rq(args, extra):
    """Chuyển sang src/rq/rq<N>_<step>.py và ghi MỘT log tổng logs/rq<N>_<YYYYmmdd_HHMMSS>.log.

    --all            : chạy full lưới RQ_GRID (2 model x 4 lang x 2 dataset x 5 seed).
    không có --all   : chỉ chạy đúng --model / --lang (và --dataset / --seed nếu có truyền);
                       dataset và seed không truyền thì lấy đủ như mặc định của script RQ.
    --models/--langs/--datasets/--seeds đưa thẳng vào `extra` luôn được ưu tiên hơn.
    """
    root = Path(__file__).resolve().parent
    # RQ viết trong MỘT file src/rq/rq<N>.py (nhận --step) thì ưu tiên; không có thì dùng rq<N>_<step>.py như cũ.
    single = root / "src" / "rq" / f"rq{args.rq}.py"
    script = single if single.exists() else root / "src" / "rq" / f"rq{args.rq}_{args.rq_step}.py"
    if not script.exists():
        sys.exit(f"Chưa có {script.relative_to(root)} (RQ{args.rq} bước '{args.rq_step}').")

    if args.all:
        grid = {k: list(v) for k, v in RQ_GRID.items()}
    else:
        grid = {"models": [args.model], "langs": [args.lang]}
        if args.dataset is not None:
            grid["datasets"] = [args.dataset]
        if args.seed is not None:
            grid["seeds"] = [args.seed]
    if args.rq_step == "collect":
        grid.pop("seeds", None)  # rq*_collect không có --seeds

    cmd = [sys.executable, str(script)]
    if script == single:
        cmd += ["--step", args.rq_step]
    cmd += extra
    for key, vals in grid.items():
        if f"--{key}" not in extra:
            cmd += [f"--{key}", *map(str, vals)]
    if script == single:
        # run_pipeline đã tự parse các flag này nên không nằm trong `extra`; file RQ đơn cần cả ba (collect cần --views).
        cmd += ["--views", args.views, "--quant", args.quant, "--stacker", args.stacker,
                "--extract_batch_size", str(args.extract_batch_size)]
    elif args.rq_step == "run":
        # Hai flag này run_pipeline đã tự parse nên không nằm trong `extra`; chỉ rq*_run mới nhận.
        cmd += ["--extract_batch_size", str(args.extract_batch_size), "--views", args.views,
                "--quant", args.quant, "--stacker", args.stacker]
        if args.stack_mode is not None:
            cmd += ["--stack_mode", args.stack_mode]

    # Cái gì đang chạy (để ghi vào log): flag tường minh trong `extra` thắng, rồi tới grid, rồi mặc định.
    def shown(key):
        flag = f"--{key}"
        if flag in extra:
            i = extra.index(flag) + 1
            vals = []
            while i < len(extra) and not extra[i].startswith("--"):
                vals.append(extra[i])
                i += 1
            return vals
        return [str(v) for v in grid.get(key, RQ_GRID[key])]

    dry = "--dry_run" in extra
    log = None
    if not dry:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = root / "logs" / f"rq{args.rq}_{stamp}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "w", encoding="utf-8")

    def emit(text):
        sys.stdout.write(text)
        sys.stdout.flush()
        if log:
            log.write(text)
            log.flush()

    emit(f"######## RQ{args.rq} | step={args.rq_step} | mode={'ALL' if args.all else 'PARTIAL'} | "
         f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    emit(f"models   : {' '.join(shown('models'))}\n")
    emit(f"langs    : {' '.join(shown('langs'))}\n")
    if args.rq_step != "collect":
        emit(f"seeds    : {' '.join(shown('seeds'))}\n")
    emit(f"datasets : {' '.join(shown('datasets'))}\n")
    emit("$ " + " ".join(cmd) + "\n\n")

    t0 = time.perf_counter()
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    try:
        proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            emit(line)
        rc = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        rc = 130
        emit("\n[interrupted]\n")
    emit(f"\n######## done rc={rc} in {time.perf_counter() - t0:.0f}s\n")
    if log:
        log.close()
        print(f"Log tổng: {log_path.relative_to(root)}")
    sys.exit(rc)


def main():
    # allow_abbrev=False: otherwise a grid flag could be silently matched to a single-run flag by prefix.
    parser = argparse.ArgumentParser(description="MultiHaluDet 4-Stage Execution Pipeline", allow_abbrev=False)
    parser.add_argument("--dataset", type=str, default=None, choices=["halueval", "triviaqa"],
                        help="Dataset (mặc định halueval; với --rq thì không truyền = chạy cả 2 dataset)")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "fr", "bn", "am", "vi"], help="Language variant of the dataset")
    parser.add_argument("--model", type=str, default="mistral-7b", choices=PIPELINE_MODELS)
    parser.add_argument("--stage", type=str, default="all", choices=["all", "extract", "probe", "train_oof", "ensemble"],
                        help="all: Run full pipeline; extract: Feature Extraction; probe: per-layer linear probes (--feat v2 only); train_oof: Multi-Scale Modeling & OOF; ensemble: Ensemble Meta-Learning")
    parser.add_argument("--extract_batch_size", type=int, default=4,
                        help="Batch size for LLM feature extraction; lower it (e.g. 2) if you hit CUDA OOM")

    # Improvements over the base pipeline. With none of these flags the base pipeline runs.
    parser.add_argument("--improved", action="store_true",
                        help="Apply everything: --layer_delta --layer_scale --rank_weight 0.1 --answer_ll --stack_mode logit")
    parser.add_argument("--layer_delta", action="store_true",
                        help="Add layer-to-layer delta channels to the sequential input")
    parser.add_argument("--layer_scale", action="store_true",
                        help="Scale the softmax layer weights by the number of layers")
    parser.add_argument("--rank_weight", type=float, default=None,
                        help="Weight of the pairwise AUC loss (default 0 = off)")
    parser.add_argument("--answer_ll", action="store_true",
                        help="Use the 4 answer log-likelihood features (global 30 -> 34)")
    parser.add_argument("--stack_mode", type=str, default=None, choices=["logit", "vector"],
                        help="vector (default): stack the 576-d features; logit: stack fold-network logits + global features")

    # Feature set v2 (hidden-state probes), split and stacker options. Defaults keep the v1 behaviour.
    parser.add_argument("--feat", type=str, default="v1", choices=["v1", "v2"],
                        help="v1: scalar statistics only (original); v2: also keep hidden-state vectors, "
                             "train per-layer linear probes and feed their logits to the model")
    parser.add_argument("--prompt_mode", type=str, default="verify", choices=["plain", "verify"],
                        help="v2 prompt: plain = 'Question/Answer'; verify = adds 'Is the proposed answer correct?' "
                             "and reads the hidden state / Yes-No logits at the last token")
    parser.add_argument("--views", type=str, default="last,amean",
                        help=f"v2 hidden-state views to store and probe, comma separated, from {','.join(VIEWS)} "
                             f"(each view costs ~2.6 GB per run for a 32-layer 4096-d model)")
    parser.add_argument("--quant", type=str, default="4bit", choices=["4bit", "8bit", "fp16"],
                        help="Precision of the LLM during extraction (8bit/fp16 need more GPU memory)")
    parser.add_argument("--probe_use", type=str, default="both", choices=["both", "seq", "glob", "none"],
                        help="v2: where the probe logits go: sequential channels, global summaries, both, or nowhere (ablation)")
    parser.add_argument("--glob_use", type=str, default="base,ll,yn",
                        help="v2: nhóm cột global đưa vào mô hình, cách nhau bằng dấu phẩy, từ base (30 cột v1), "
                             "ll (13 cột answer-LL/PMI), yn (3 cột Yes/No, chỉ có với --prompt_mode verify); dùng cho ablation RQ1")
    parser.add_argument("--pca_dim", type=int, default=None, help="PCA dimension of each probe (default config: 256)")
    parser.add_argument("--probe_C", type=float, default=None, help="Inverse L2 strength of each probe (default config: 0.02)")
    parser.add_argument("--probe_jobs", type=int, default=None, help="Probes fitted in parallel (default config: 4)")
    parser.add_argument("--redo_probe", action="store_true", help="Recompute the probes even if their file exists")
    parser.add_argument("--group_split", action="store_true",
                        help="Split by question so both answers of a question stay together (always on with --feat v2)")
    parser.add_argument("--pair_center", action="store_true",
                        help="Subtract the per-question mean from every feature (and from the probe inputs). Needs both "
                             "answers of a question at test time, so it changes what is being evaluated")
    parser.add_argument("--stacker", type=str, default="full", choices=["full", "lite"],
                        help="full: the 6 base classifiers; lite: logistic regression + LightGBM only")
    parser.add_argument("--hidden_dim", type=int, default=None, help="Override config.hidden_dim (must divide by num_heads=8)")
    parser.add_argument("--num_layers", type=int, default=None, help="Override the number of transformer layers")
    parser.add_argument("--dropout", type=float, default=None, help="Override config.dropout")

    # Run management
    parser.add_argument("--seed", type=int, default=None, help="Override config.seed (changes the split and the folds)")
    parser.add_argument("--tag", type=str, default="",
                        help="Suffix for OOF/plot files so several runs (seeds, variants) do not overwrite each other")
    parser.add_argument("--no_mixup", action="store_true", help="Disable Mixup")
    parser.add_argument("--no_cutmix", action="store_true", help="Disable CutMix")

    # Research-question grids: delegate to src/rq/rq<N>_<step>.py
    parser.add_argument("--rq", type=int, choices=[1, 2], default=None,
                        help="Chạy lưới thí nghiệm RQ<N> (model x lang x dataset x seed) thay vì pipeline đơn lẻ; "
                             "các flag còn lại (--models, --langs, --seeds, --dry_run, ...) được chuyển cho script RQ")
    parser.add_argument("--all", action="store_true",
                        help="Với --rq: chạy full lưới 2 model x 4 lang x 2 dataset x 5 seed "
                             "(không có --all thì chỉ chạy --model/--lang/--dataset/--seed đã truyền)")
    parser.add_argument("--rq_step", type=str, default="run", choices=RQ_STEPS,
                        help="run: chạy lưới; collect: gom log thành CSV; diagnostics: chẩn đoán answer-LL")
    args, extra = parser.parse_known_args()

    # Must come before the --improved block below: that block defaults stack_mode to "vector",
    # whereas rq*_run.py defaults to "logit".
    if args.rq is not None:
        run_rq(args, extra)  # never returns
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")

    if args.all:
        parser.error("--all chỉ dùng cùng --rq")
    if args.dataset is None:
        args.dataset = "halueval"

    if args.improved:
        args.layer_delta = args.layer_scale = args.answer_ll = True
        if args.rank_weight is None:
            args.rank_weight = 0.1
        if args.stack_mode is None:
            args.stack_mode = "logit"
    if args.stack_mode is None:
        args.stack_mode = "vector"

    config = get_config()
    if args.seed is not None:
        config.seed = args.seed
    if args.no_mixup:
        config.use_mixup = False
    if args.no_cutmix:
        config.use_cutmix = False
    config.use_layer_delta = args.layer_delta
    config.scale_layer_weights = args.layer_scale
    config.use_answer_ll = args.answer_ll
    if args.rank_weight is not None:
        config.rank_loss_weight = args.rank_weight

    if args.hidden_dim is not None:
        if args.hidden_dim % config.num_heads:
            parser.error(f"--hidden_dim must be a multiple of {config.num_heads}")
        config.hidden_dim = args.hidden_dim
    if args.num_layers is not None:
        config.num_layers = args.num_layers
    if args.dropout is not None:
        config.dropout = args.dropout
    if args.pca_dim is not None:
        config.probe_pca_dim = args.pca_dim
    if args.probe_C is not None:
        config.probe_C = args.probe_C
    if args.probe_jobs is not None:
        config.probe_jobs = args.probe_jobs

    v2 = args.feat == "v2"
    if args.stage == "probe" and not v2:
        parser.error("--stage probe needs --feat v2")
    if args.pair_center and not (v2 or args.group_split):
        parser.error("--pair_center needs --group_split (or --feat v2)")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if args.stage in ["extract", "all"]:
        (stage_1_extract_v2 if v2 else stage_1_extract)(args, config)
    if v2 and args.probe_use != "none" and args.stage in ["probe", "all"]:
        stage_2_probe(args, config)
    if args.stage in ["train_oof", "all"]:
        stage_2_3_train_oof(args, config)
    if args.stage in ["ensemble", "all"]:
        stage_4_ensemble(args, config)

if __name__ == "__main__":
    main()
