"""RQ1: mỗi nhóm thành phần của kiến trúc mới đóng góp bao nhiêu AUROC?

Một file duy nhất, chạy qua run_pipeline.py (cờ --rq / --rq_step):
    python run_pipeline.py --rq 1 --all --dry_run               # xem kế hoạch, không chạy gì
    python run_pipeline.py --rq 1 --all --keep_going            # lưới đầy đủ (mức nhóm)
    python run_pipeline.py --rq 1 --model llama2-7b --lang vi --dataset triviaqa --seed 1
    python run_pipeline.py --rq 1 --rq_step collect --all       # gom log/OOF thành CSV + bảng đóng góp
    python run_pipeline.py --rq 1 --all --level fine            # thêm từng thành phần riêng lẻ
    python run_pipeline.py --rq 1 --model llama2-7b --lang en --dataset triviaqa --conditions full loo_A add_A

Ba nhóm thành phần (tất cả chạy trong pipeline --feat v2, nên MỌI điều kiện dùng chung group split):
    A  answer-conditional : A_prompt (verify prompt) | A_ll (13 cột answer-LL/PMI) | A_yn (3 cột Yes/No)
    B  hidden-state probes: B_seq (logit probe làm kênh seq) | B_glob (4 tóm tắt probe nối vào global)
    C  training/stacking  : C_delta (layer-delta) | C_scale (layer-scale) | C_rank (rank loss 0.1) | C_logit (stack logit)

Điều kiện mức nhóm:  base (tắt hết) | full (bật hết) | loo_<G> = full bỏ nhóm G | add_<G> = base chỉ bật nhóm G
Đóng góp của nhóm G:  LOO:G = full - loo_G  và  ADD:G = add_G - base  (ghép cặp theo seed).
Mức fine thêm loo_<thành phần> / add_<thành phần>.

Lưu ý thiết kế:
  * Group split LUÔN bật ở --feat v2 nên "group folds" không ablate được trong pipeline này (cố định, không thuộc nhóm C).
  * A_yn cần verify prompt: loo_A_prompt bỏ luôn A_yn; add_A_yn không tồn tại, thay bằng add_A_prompt_yn so với add_A_prompt.
  * Probe logit được cache theo (dataset, lang, model, prompt_mode, views, seed) bởi run_pipeline.py --stage probe,
    nên bật/tắt B_seq/B_glob (--probe_use) không tính lại probe.
  * Điều kiện có cùng cấu hình train (chỉ khác C_logit) dùng chung một lần train_oof (OOF được hard-link sang tag của từng điều kiện).
  * collect chỉ ghép cặp những seed mà hai điều kiện có cùng n_test (cùng tập test).

Log:     logs/rq1_ablation/<model>/<lang>/rq1_<dataset>_<cond>_s<seed>.log  (ensemble), kèm log extract/probe/train. Log tổng: logs/rq1_ablation/master.log
Kết quả: results/rq1_ablation/all_{runs,summary,paired_per_seed,paired_summary}.csv và results/rq1_ablation/<model>/<lang>/*.csv
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FEAT2_DIR = ROOT / "results" / "features_v2"
OOF_DIR = ROOT / "results" / "oof"
PROBE_DIR = ROOT / "results" / "probe"
LOG_ROOT = ROOT / "logs" / "rq1_ablation"
RES_ROOT = ROOT / "results" / "rq1_ablation"

MODELS = ["mistral-7b", "llama2-7b"]
LANGS = ["en", "fr", "vi", "bn"]
DATASETS = ["triviaqa", "halueval"]       # TriviaQA là kết quả chính nên chạy trước
SEEDS = [1, 11, 29, 37, 43]
STAGES = ["extract", "probe", "train_oof", "ensemble"]

# ------------------------------------------------------------------ điều kiện
COMPONENTS = ["A_prompt", "A_ll", "A_yn", "B_seq", "B_glob", "C_delta", "C_scale", "C_rank", "C_logit"]
GROUPS = {"A": COMPONENTS[0:3], "B": COMPONENTS[3:5], "C": COMPONENTS[5:9]}
N_BASE, N_LL, N_YN = 30, 13, 3     # số cột global: base_*, ans_*, v_*
RANK_WEIGHT = 0.1


def _norm(comps):
    comps = set(comps)
    if "A_prompt" not in comps:
        comps.discard("A_yn")      # Yes/No chỉ tồn tại với verify prompt
    return frozenset(comps)


def build_conditions():
    full = set(COMPONENTS)
    group = {"base": set(), "full": full}
    for g, cs in GROUPS.items():
        group[f"loo_{g}"] = full - set(cs)
        group[f"add_{g}"] = set(cs)
    fine = {}
    for k in COMPONENTS:
        fine[f"loo_{k}"] = full - {k}
        if k != "A_yn":
            fine[f"add_{k}"] = {k}
    fine["add_A_prompt_yn"] = {"A_prompt", "A_yn"}
    return ({k: _norm(v) for k, v in group.items()}, {k: _norm(v) for k, v in fine.items()})


GROUP_CONDS, FINE_CONDS = build_conditions()
ALL_CONDS = {**GROUP_CONDS, **FINE_CONDS}


def build_pairs():
    """pair -> (điều kiện 'có', điều kiện 'không'); delta = có - không."""
    group = {"ALL": ("full", "base")}
    for g in GROUPS:
        group[f"LOO:{g}"] = ("full", f"loo_{g}")
        group[f"ADD:{g}"] = (f"add_{g}", "base")
    fine = {}
    for k in COMPONENTS:
        fine[f"LOO:{k}"] = ("full", f"loo_{k}")      # LOO:A_prompt gồm cả A_yn (bị bỏ theo)
        if k != "A_yn":
            fine[f"ADD:{k}"] = (f"add_{k}", "base")
    fine["ADD:A_yn|A_prompt"] = ("add_A_prompt_yn", "add_A_prompt")
    return group, fine


GROUP_PAIRS, FINE_PAIRS = build_pairs()
ALL_PAIRS = {**GROUP_PAIRS, **FINE_PAIRS}


def cond_spec(comps, a):
    """Cờ run_pipeline.py cho một điều kiện + chữ ký cấu hình train (không tính stack_mode)."""
    views = [v.strip() for v in a.views.split(",") if v.strip()]
    pm = "verify" if "A_prompt" in comps else "plain"
    glob = ["base"] + [g for g, k in (("ll", "A_ll"), ("yn", "A_yn")) if k in comps]
    probe = {(True, True): "both", (True, False): "seq", (False, True): "glob",
             (False, False): "none"}[("B_seq" in comps, "B_glob" in comps)]
    train = ["--feat", "v2", "--prompt_mode", pm, "--views", ",".join(views),
             "--glob_use", ",".join(glob), "--probe_use", probe]
    if "C_delta" in comps:
        train.append("--layer_delta")
    if "C_scale" in comps:
        train.append("--layer_scale")
    if "C_rank" in comps:
        train += ["--rank_weight", str(RANK_WEIGHT)]
    c_flags = "".join(c for c, k in (("d", "C_delta"), ("s", "C_scale"), ("r", "C_rank")) if k in comps) or "none"
    glob_dim = (N_BASE + (N_LL if "ll" in glob else 0) + (N_YN if "yn" in glob else 0)
                + (2 * len(views) if probe in ("both", "glob") else 0))
    return {"pm": pm, "glob": glob, "probe": probe, "train": train,
            "stack": "logit" if "C_logit" in comps else "vector",
            "sig": f"{pm}-{'_'.join(glob)}-{probe}-{c_flags}", "glob_dim": glob_dim, "views": views}


# ------------------------------------------------------------------ đường dẫn
def log_dir(model, lang):
    return LOG_ROOT / model / lang


def out_dir(model, lang):
    return RES_ROOT / model / lang


def fkey2(ds, lang, model, pm):
    return f"{ds}_{lang}_{model}_{pm}"


def okey(ds, lang, model, pm, tag):
    return f"{ds}_{lang}_{model}_v2{pm}_{tag}"


def feat_state(ds, lang, model, pm, views):
    d = FEAT2_DIR / fkey2(ds, lang, model, pm)
    if not ((d / "y.npy").exists() and (d / "meta.json").exists()):
        return "missing"
    return "ok" if json.loads((d / "meta.json").read_text()).get("views") == views else "old"


def probe_path(ds, lang, model, pm, views, seed):
    from src.config import get_config      # import muộn: kéo theo torch
    cfg = get_config()
    return PROBE_DIR / f"{fkey2(ds, lang, model, pm)}_{'-'.join(views)}_s{seed}_pca{cfg.probe_pca_dim}_C{cfg.probe_C}.npz"


OOF_FILES = ["X_train_deep", "X_test_deep", "oof_logits", "test_logits", "X_glob_train", "X_glob_test",
             "y_train", "y_test", "g_train", "g_test"]


def alias_oof(src_key, dst_key):
    """OOF của train_oof nằm dưới tag train; ensemble đọc dưới tag điều kiện -> hard-link (không tốn đĩa)."""
    for name in OOF_FILES:
        s, d = OOF_DIR / f"{name}_{src_key}.npy", OOF_DIR / f"{name}_{dst_key}.npy"
        if d.exists() or d.is_symlink():
            d.unlink()
        if s.exists():
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)


# ------------------------------------------------------------------ tiện ích chung
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, s):
        for f in self.files:
            f.write(s)

    def flush(self):
        for f in self.files:
            f.flush()


def run_logged(cmd, log_path, mode):
    """Chạy cmd tại ROOT, vừa in ra màn hình vừa ghi log. Trả về return code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with open(log_path, mode, encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def merge_write(path, new, models, langs, datasets):
    """Ghi all_*.csv kiểu GỘP: bỏ các dòng cũ thuộc (model, lang, dataset) vừa xử lý, giữ phần còn lại."""
    merged = new
    if path.exists():
        old = pd.read_csv(path)
        if {"model", "lang", "dataset"} <= set(old.columns):
            drop = old["model"].isin(models) & old["lang"].isin(langs) & old["dataset"].isin(datasets)
            old = old[~drop]
            merged = pd.concat([old, new], ignore_index=True) if len(old) else new
    if len(merged) or path.exists():
        merged.to_csv(path, index=False)
    return merged


_DATA_CACHE = {}


def check_data(ds, lang):
    """(ok, thông báo) cho data/<ds>_<lang>.json; in thống kê nhãn một lần cho mỗi (ds, lang)."""
    if (ds, lang) in _DATA_CACHE:
        return _DATA_CACHE[(ds, lang)]
    path = ROOT / "data" / f"{ds}_{lang}.json"
    rel = path.relative_to(ROOT)
    if not path.exists():
        if ds == "halueval" and lang == "en":
            res = (True, "không có file local, loader sẽ lấy từ HuggingFace (pminervini/HaluEval)")
        elif ds == "triviaqa" and lang == "en":
            res = (False, f"thiếu {rel} (loader sẽ rơi về negative = đáp án câu hỏi ngẫu nhiên)")
        else:
            res = (False, f"thiếu {rel}")
    else:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            missing = [k for k in ("question", "answer", "is_hallucination") if k not in data[0]]
            if missing:
                res = (False, f"{rel} thiếu key {missing}")
            else:
                labels = np.array([int(d["is_hallucination"]) for d in data])
                msg = f"{rel}: {len(data)} mẫu, {labels.sum()} hallucination ({labels.mean():.1%})"
                if abs(labels.mean() - 0.5) > 0.1:
                    msg += "  [cảnh báo: nhãn không cân bằng]"
                if ds == "triviaqa" and len(data) > 10000:
                    msg += "  [cảnh báo: loader chỉ lấy 10000 dòng ngẫu nhiên -> có thể làm lẻ cặp]"
                res = (True, msg)
        except Exception as e:
            res = (False, f"{rel} không đọc được: {type(e).__name__}: {e}")
    print(f"[data] {ds}/{lang}: {res[1]}")
    _DATA_CACHE[(ds, lang)] = res
    return res


# ------------------------------------------------------------------ bước run
def build_jobs(a, conds, specs):
    views = [v.strip() for v in a.views.split(",") if v.strip()]
    jobs, skipped = [], []
    for model in a.models:
        for lang in a.langs:
            for ds in a.datasets:
                data_ok, msg = check_data(ds, lang)
                base = [sys.executable, "run_pipeline.py", "--dataset", ds, "--lang", lang, "--model", model]
                ld = log_dir(model, lang)
                usable = {}
                for pm in sorted({specs[c]["pm"] for c in conds}):
                    st = feat_state(ds, lang, model, pm, views)
                    if st == "old" and not a.force_extract:
                        skipped.append((model, lang, ds, pm, "feature v2 cũ (views khác): xóa hoặc dùng --force_extract"))
                    elif st != "ok" and not data_ok:
                        skipped.append((model, lang, ds, pm, msg))
                    else:
                        usable[pm] = st
                for pm, st in usable.items():
                    if "extract" in a.stages:
                        jobs.append({"id": f"ext|{model}|{lang}|{ds}|{pm}", "kind": "extract", "model": model,
                                     "lang": lang, "ds": ds, "pm": pm, "deps": [],
                                     "pending": st != "ok" or a.force_extract,
                                     "cmd": base + ["--stage", "extract", "--feat", "v2", "--prompt_mode", pm,
                                                    "--views", ",".join(views), "--quant", a.quant,
                                                    "--extract_batch_size", str(a.extract_batch_size)],
                                     "log": ld / f"rq1_extract_{ds}_{pm}.log", "mode": "w"})
                for seed in a.seeds:
                    for pm in usable:
                        if not any(specs[c]["pm"] == pm and specs[c]["probe"] != "none" for c in conds):
                            continue
                        if "probe" in a.stages:
                            pp = probe_path(ds, lang, model, pm, views, seed)
                            jobs.append({"id": f"probe|{model}|{lang}|{ds}|{pm}|{seed}", "kind": "probe",
                                         "model": model, "lang": lang, "ds": ds, "pm": pm, "seed": seed,
                                         "deps": [f"ext|{model}|{lang}|{ds}|{pm}"],
                                         "pending": a.force_extract or not pp.exists(),
                                         "cmd": base + ["--stage", "probe", "--seed", str(seed), "--feat", "v2",
                                                        "--prompt_mode", pm, "--views", ",".join(views)]
                                                + (["--redo_probe"] if a.force_extract else []),
                                         "log": ld / f"rq1_probe_{ds}_{pm}_s{seed}.log", "mode": "w"})
                    seen = set()
                    for cond in conds:
                        sp = specs[cond]
                        if sp["pm"] not in usable:
                            continue
                        pm = sp["pm"]
                        train_tag = f"tr_{sp['sig']}_s{seed}"
                        src_key = okey(ds, lang, model, pm, train_tag)
                        train_id = f"train|{model}|{lang}|{ds}|{train_tag}"
                        if sp["sig"] not in seen and "train_oof" in a.stages:
                            seen.add(sp["sig"])
                            deps = [f"ext|{model}|{lang}|{ds}|{pm}"]
                            if sp["probe"] != "none":
                                deps.append(f"probe|{model}|{lang}|{ds}|{pm}|{seed}")
                            jobs.append({"id": train_id, "kind": "train", "model": model, "lang": lang, "ds": ds,
                                         "seed": seed, "pm": pm, "deps": deps, "src_key": src_key,
                                         "pending": a.force or not (OOF_DIR / f"y_test_{src_key}.npy").exists(),
                                         "cmd": base + ["--stage", "train_oof", "--seed", str(seed),
                                                        "--tag", train_tag] + sp["train"],
                                         "log": ld / f"rq1_{ds}_{train_tag}.log", "mode": "w"})
                        if "ensemble" in a.stages:
                            tag = f"{cond}_s{seed}"
                            lp = ld / f"rq1_{ds}_{tag}.log"
                            done = lp.exists() and "Results Summary" in lp.read_text(encoding="utf-8", errors="replace")
                            jobs.append({"id": f"ens|{model}|{lang}|{ds}|{tag}", "kind": "ensemble", "model": model,
                                         "lang": lang, "ds": ds, "seed": seed, "cond": cond, "pm": pm,
                                         "deps": [train_id], "src_key": src_key,
                                         "dst_key": okey(ds, lang, model, pm, tag),
                                         "pending": a.force or not done,
                                         "cmd": base + ["--stage", "ensemble", "--seed", str(seed), "--feat", "v2",
                                                        "--prompt_mode", pm, "--stack_mode", sp["stack"],
                                                        "--stacker", a.stacker, "--tag", tag],
                                         "log": lp, "mode": "w"})
    return jobs, skipped


def finished_ok(j):
    if j["kind"] == "train":
        return (OOF_DIR / f"y_test_{j['src_key']}.npy").exists()
    if j["kind"] == "ensemble":
        return "Results Summary" in j["log"].read_text(encoding="utf-8", errors="replace")
    return True


def step_run(a):
    conds = {c: ALL_CONDS[c] for c in (a.conditions or (ALL_CONDS if a.level == "fine" else GROUP_CONDS))}
    specs = {c: cond_spec(comps, a) for c, comps in conds.items()}
    if not a.dry_run:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        sys.stdout = Tee(sys.__stdout__, open(LOG_ROOT / "master.log", "a", encoding="utf-8"))
        print(f"\n######## {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(sys.argv)}")

    print("\n--- Điều kiện ---")
    for c, sp in specs.items():
        print(f"  {c:16s} prompt={sp['pm']:6s} glob={'+'.join(sp['glob']):10s} probe={sp['probe']:5s} "
              f"stack={sp['stack']:6s} train={sp['sig']}  (global {sp['glob_dim']} cột)")
    print("\n--- Kiểm tra dữ liệu ---")
    jobs, skipped = build_jobs(a, conds, specs)

    print("\n--- Kế hoạch (cần chạy / tổng) ---")
    combos = sorted({(j["model"], j["lang"], j["ds"]) for j in jobs})
    for model, lang, ds in combos:
        cells = []
        for kind in ("extract", "probe", "train", "ensemble"):
            sel = [j for j in jobs if (j["model"], j["lang"], j["ds"]) == (model, lang, ds) and j["kind"] == kind]
            cells.append(f"{kind}: {sum(j['pending'] for j in sel)}/{len(sel)}")
        print(f"  {model:11s} {lang} {ds:9s} " + "  ".join(cells))
    print(f"Tổng: {len(combos)} tổ hợp | " + " | ".join(
        f"{k} {sum(j['pending'] for j in jobs if j['kind'] == k)}" for k in ("extract", "probe", "train", "ensemble")))
    for model, lang, ds, pm, why in skipped:
        print(f"  [bỏ qua] {model} {lang} {ds} ({pm}): {why}")
    if a.dry_run:
        return

    failures, failed = [], set()
    for j in jobs:
        if not j["pending"]:
            continue
        label = f"{j['kind']} | {j['model']} {j['lang']} {j['ds']} | " + str(j.get("cond") or j.get("pm"))
        if j.get("seed") is not None:
            label += f" s{j['seed']}"
        if any(d in failed for d in j["deps"]):
            print(f"\n[bỏ qua] {label}: bước phụ thuộc đã lỗi")
            failed.add(j["id"])
            continue
        print(f"\n=== {label} ===")
        if j["kind"] == "ensemble":
            alias_oof(j["src_key"], j["dst_key"])
        t0 = time.perf_counter()
        rc = run_logged(j["cmd"], j["log"], j["mode"])
        print(f"[{j['kind']}] {time.perf_counter() - t0:.0f}s")
        if rc != 0 or not finished_ok(j):
            msg = f"[fail] {label} rc={rc}, xem {j['log']}"
            failures.append(msg)
            failed.add(j["id"])
            if a.keep_going:
                print(msg)
                continue
            sys.exit(msg)
    print("\nXong.")
    for msg in failures:
        print(msg)
    for model, lang, ds, pm, why in skipped:
        print(f"[đã bỏ qua] {model} {lang} {ds} ({pm}): {why}")
    print("Gom kết quả:  python run_pipeline.py --rq 1 --rq_step collect --all")


# ------------------------------------------------------------------ bước collect
RESULT = re.compile(r"^AUC: ([\d.]+), F1: ([\d.]+), Acc: ([\d.]+)", re.M)
PAIRACC = re.compile(r"^Pair accuracy \(.*?\): ([\d.]+)", re.M)
GROUP_KEYS = ["model", "lang", "dataset"]
METRICS = ["deep_test_auc", "deep_oof_auc", "ens_auc", "ens_f1", "ens_acc", "ens_pair_acc", "deep_pair_acc"]
PAIRED_METRICS = ["ens_auc", "deep_test_auc", "ens_pair_acc"]


def collect_one(model, lang, ds, cond, seed, spec):
    lp = log_dir(model, lang) / f"rq1_{ds}_{cond}_s{seed}.log"
    if not lp.exists():
        return None
    from sklearn.metrics import roc_auc_score
    from src.data.splits import pair_accuracy
    text = lp.read_text(encoding="utf-8", errors="replace")
    row = {"model": model, "lang": lang, "dataset": ds, "cond": cond, "seed": seed, "log": lp.name}
    m = RESULT.findall(text)
    if m:
        row["ens_auc"], row["ens_f1"], row["ens_acc"] = map(float, m[-1])
    m = PAIRACC.findall(text)
    if m:
        row["ens_pair_acc"] = float(m[-1])
    problems = []
    first = text.splitlines()[0] if text else ""
    for flag, want in (("--prompt_mode", spec["pm"]), ("--stack_mode", spec["stack"])):
        if first.startswith("$ ") and f"{flag} {want}" not in first:
            problems.append(f"{flag} != {want}")
    key = okey(ds, lang, model, spec["pm"], f"{cond}_s{seed}")
    try:
        y_te, z_te = np.load(OOF_DIR / f"y_test_{key}.npy"), np.load(OOF_DIR / f"test_logits_{key}.npy").mean(0)
        y_tr, z_tr = np.load(OOF_DIR / f"y_train_{key}.npy"), np.load(OOF_DIR / f"oof_logits_{key}.npy")
        row["glob_dim"] = int(np.load(OOF_DIR / f"X_glob_train_{key}.npy", mmap_mode="r").shape[1])
        row["n_test"] = int(len(y_te))
        row["deep_test_auc"] = float(roc_auc_score(y_te, z_te))
        row["deep_oof_auc"] = float(roc_auc_score(y_tr, z_tr))
        gp = OOF_DIR / f"g_test_{key}.npy"
        if gp.exists():
            row["deep_pair_acc"] = float(pair_accuracy(z_te, y_te, np.load(gp))[0])
        if row["glob_dim"] != spec["glob_dim"]:
            problems.append(f"glob_dim={row['glob_dim']} (cần {spec['glob_dim']})")
    except FileNotFoundError:
        problems.append("thiếu file OOF")
    row["config_problem"] = "; ".join(problems)
    row["complete"] = "ens_auc" in row and "deep_test_auc" in row
    return row


def analyze(ok, pairs):
    metrics = [m for m in METRICS if m in ok]
    summary = ok.groupby(GROUP_KEYS + ["cond"])[metrics].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    per_seed, paired = [], []
    for keys, g in ok.groupby(GROUP_KEYS):
        ident = dict(zip(GROUP_KEYS, keys))
        for pname, (w, wo) in pairs.items():
            sub = g[g["cond"].isin([w, wo])]
            if not {w, wo} <= set(sub["cond"]):
                continue
            n_piv = sub.pivot(index="seed", columns="cond", values="n_test")
            for metric in PAIRED_METRICS:
                if metric not in sub or sub[metric].isna().all():
                    continue
                piv = sub.pivot(index="seed", columns="cond", values=metric)[(n_piv[w] == n_piv[wo])].dropna()
                if piv.empty:
                    continue
                delta = piv[w] - piv[wo]
                for seed, d in delta.items():
                    per_seed.append({**ident, "pair": pname, "with": w, "without": wo, "metric": metric,
                                     "seed": seed, "value_with": piv.loc[seed, w],
                                     "value_without": piv.loc[seed, wo], "delta": d})
                row = {**ident, "pair": pname, "with": w, "without": wo, "metric": metric, "n_seeds": len(delta),
                       "with_mean": piv[w].mean(), "without_mean": piv[wo].mean(), "delta_mean": delta.mean(),
                       "delta_std": delta.std(ddof=1) if len(delta) > 1 else np.nan,
                       "n_with_better": int((delta > 0).sum()), "wilcoxon_p": np.nan, "ttest_p": np.nan}
                if len(delta) >= 2:
                    try:
                        row["wilcoxon_p"] = stats.wilcoxon(delta).pvalue
                    except ValueError:
                        pass
                    row["ttest_p"] = stats.ttest_rel(piv[w], piv[wo]).pvalue
                paired.append(row)
    return summary, pd.DataFrame(per_seed), pd.DataFrame(paired)


def step_collect(a):
    conds = a.conditions or list(ALL_CONDS)
    specs = {c: cond_spec(ALL_CONDS[c], a) for c in conds}
    pairs = ALL_PAIRS if a.level == "fine" else GROUP_PAIRS
    rows = []
    for model in a.models:
        for lang in a.langs:
            for ds in a.datasets:
                for cond in conds:
                    for seed in a.seeds:
                        r = collect_one(model, lang, ds, cond, seed, specs[cond])
                        if r:
                            rows.append(r)
    if not rows:
        sys.exit("Không thấy log rq1_<dataset>_<cond>_s<seed>.log nào.")
    runs = pd.DataFrame(rows)
    for _, r in runs[runs["config_problem"] != ""].iterrows():
        print(f"[cảnh báo] {r['model']}/{r['lang']}/{r['log']}: {r['config_problem']}")
    for _, r in runs[~runs["complete"]].iterrows():
        print(f"[chưa xong] {r['model']}/{r['lang']}/{r['log']}")

    RES_ROOT.mkdir(parents=True, exist_ok=True)
    runs_all = merge_write(RES_ROOT / "all_runs.csv", runs, a.models, a.langs, a.datasets)
    ok = runs[runs["complete"] & (runs["config_problem"] == "")]
    if ok.empty:
        sys.exit("Không có run hợp lệ để tổng hợp.")
    summary, per_seed, paired = analyze(ok, pairs)
    summary = merge_write(RES_ROOT / "all_summary.csv", summary, a.models, a.langs, a.datasets)
    per_seed = merge_write(RES_ROOT / "all_paired_per_seed.csv", per_seed, a.models, a.langs, a.datasets)
    paired = merge_write(RES_ROOT / "all_paired_summary.csv", paired, a.models, a.langs, a.datasets)
    for model in a.models:
        for lang in a.langs:
            d = out_dir(model, lang)
            for df, name in ((runs_all, "runs.csv"), (summary, "summary.csv"),
                             (per_seed, "paired_per_seed.csv"), (paired, "paired_summary.csv")):
                if len(df) and ((df["model"] == model) & (df["lang"] == lang)).any():
                    d.mkdir(parents=True, exist_ok=True)
                    df[(df["model"] == model) & (df["lang"] == lang)].drop(columns=["model", "lang"]).to_csv(d / name, index=False)

    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 300,
                  "display.float_format", "{:.4f}".format)
    print(f"\n{len(ok)}/{len(runs)} run hợp lệ. CSV: {RES_ROOT}/all_*.csv")
    if paired.empty:
        print("Chưa có cặp điều kiện đầy đủ để so sánh.")
        return
    order = list(pairs)
    for ds in DATASETS:
        p = paired[(paired["dataset"] == ds) & paired["pair"].isin(order)]
        if p.empty:
            continue
        agg = {}
        for metric, short in (("ens_auc", "d_ens"), ("deep_test_auc", "d_deep"), ("ens_pair_acc", "d_pairacc")):
            q = p[p["metric"] == metric].groupby("pair").agg(
                **{short: ("delta_mean", "mean"), short + "_min": ("delta_mean", "min"),
                   short + "_max": ("delta_mean", "max")})
            agg[short] = q
        t = agg["d_ens"].join(agg["d_deep"][["d_deep"]]).join(agg["d_pairacc"][["d_pairacc"]])
        e = p[p["metric"] == "ens_auc"].groupby("pair").agg(
            combos=("delta_mean", "size"), seeds_better=("n_with_better", "sum"), seeds=("n_seeds", "sum"))
        t = t.join(e).reindex([x for x in order if x in t.index])
        t["better"] = t["seeds_better"].astype(int).astype(str) + "/" + t["seeds"].astype(int).astype(str)
        print(f"\n=== {ds}: delta = có - không, trung bình trên {int(t['combos'].max())} tổ hợp model x lang ===")
        print(t[["d_ens", "d_ens_min", "d_ens_max", "d_deep", "d_pairacc", "better"]].to_string())
    print("\nLưu ý: với n=5 seed, Wilcoxon có p nhỏ nhất 0.0625; hãy nhìn hướng, số seed 'có' > 'không' và độ lớn hiệu ứng.")


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    p.add_argument("--step", default="run", choices=["run", "collect", "diagnostics"])
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--langs", nargs="+", default=LANGS, choices=LANGS)
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--level", default="group", choices=["group", "fine"],
                   help="group: base/full/loo_A,B,C/add_A,B,C; fine: thêm từng thành phần riêng lẻ")
    p.add_argument("--conditions", nargs="+", default=None, choices=list(ALL_CONDS),
                   help="Chỉ chạy/gom các điều kiện này (mặc định theo --level)")
    p.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    p.add_argument("--views", default="last,amean", help="Phải khớp --views của extraction v2")
    p.add_argument("--quant", default="4bit", choices=["4bit", "8bit", "fp16"])
    p.add_argument("--stacker", default="full", choices=["full", "lite"])
    p.add_argument("--extract_batch_size", type=int, default=4)
    p.add_argument("--force", action="store_true", help="Chạy lại train/ensemble kể cả khi đã có kết quả")
    p.add_argument("--force_extract", action="store_true", help="Trích xuất lại feature (và probe)")
    p.add_argument("--keep_going", action="store_true", help="Bước lỗi thì ghi nhận rồi chạy tiếp")
    p.add_argument("--dry_run", action="store_true", help="Chỉ in kế hoạch")
    a = p.parse_args()
    if a.step == "diagnostics":
        sys.exit("RQ1 mới không có bước diagnostics; dùng --rq_step run hoặc collect.")
    (step_run if a.step == "run" else step_collect)(a)


if __name__ == "__main__":
    main()
