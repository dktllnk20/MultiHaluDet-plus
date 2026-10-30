"""Feature extraction v2: keeps the hidden-state VECTORS (for linear probes) next to the old scalar
statistics, adds a verification prompt and richer answer log-likelihood features.

One forward pass per batch (plus one short pass on "Answer: <a>" for the PMI baseline) gives:
  H_<view>.npy  float16 [L, N, D]   hidden state per sampled layer, one file per view
      last   final token of the prompt (verify mode: the ':' after "(Yes or No)"; plain mode: last answer token)
      alast  last answer token
      afirst first answer token
      amean  mean over the answer tokens
  seq.npy       float32 [N, L, 12]  same 12 statistics per layer as v1 (the mean-pool pair is over the answer span)
  glob.npy      float32 [N, G]      30 v1 global features + 13 answer-LL/PMI features (+ 3 Yes/No features in verify mode)
  y.npy, groups.npy, valid.npy, meta.json
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from tqdm import tqdm

from src.data.feature_extractor import _layer_stats, get_anchor_stats, get_sampled_layer_indices
from src.data.splits import make_groups

VIEWS = ("last", "alast", "afirst", "amean")
MAX_LEN = 384
ANS_KEYS = ["sum", "mean", "first", "min", "std", "ent_mean", "ent_first", "rank_first", "rank_mean", "n"]
ANS_NAMES = [f"ans_{k}" for k in ANS_KEYS] + ["ans_pmi_sum", "ans_pmi_mean", "ans_pmi_first"]
VERIFY_NAMES = ["v_lp_yes", "v_lp_no", "v_yes_minus_no"]
BASE_NAMES = [f"base_{i}" for i in range(30)]


def build_texts(questions, answers, mode):
    """Prompts plus the character span of the answer inside each prompt."""
    texts, spans = [], []
    for q, a in zip(questions, answers, strict=True):
        a = str(a).strip()
        if mode == "verify":
            prefix = f"Question: {q}\nProposed answer:"
            suffix = "\nIs the proposed answer correct? Answer (Yes or No):"
        else:
            prefix, suffix = f"Question: {q}\nAnswer:", ""
        texts.append(prefix + " " + a + suffix)
        spans.append((len(prefix) + 1, len(prefix) + 1 + len(a)))
    return texts, spans


def build_null_texts(answers):
    """The same answer without the question; its log-likelihood is the PMI baseline."""
    texts, spans = [], []
    for a in answers:
        a = str(a).strip()
        prefix = "Answer:"
        texts.append(prefix + " " + a)
        spans.append((len(prefix) + 1, len(prefix) + 1 + len(a)))
    return texts, spans


def _encode(tokenizer, texts, spans, device):
    """Tokenise (left padding) and return the mask of answer tokens found through char offsets."""
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"   # never cut the answer or the verification suffix
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                    max_length=MAX_LEN, return_offsets_mapping=True)
    off = enc["offset_mapping"]
    a0 = torch.tensor([s[0] for s in spans])[:, None]
    a1 = torch.tensor([s[1] for s in spans])[:, None]
    in_ans = (off[..., 0] < a1) & (off[..., 1] > a0) & (enc["attention_mask"] == 1)
    ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    return ids, mask, in_ans.to(device)


def _answer_token_stats(hidden, head, ids, in_ans):
    """Per-sample statistics of log p(answer token | everything before it).

    hidden: [B, T, H] final-layer states (after the last norm). The state at position p-1 predicts
    the token at p, so the answer tokens are scored by the states one position earlier.
    """
    B = ids.shape[0]
    out = {k: torch.zeros(B, device=ids.device) for k in ANS_KEYS}
    bi, pi = in_ans.nonzero(as_tuple=True)
    if bi.numel() == 0:
        return out
    lp = F.log_softmax(head(hidden[bi, (pi - 1).clamp_min(0)]).float(), dim=-1)    # [M, V]
    tok_lp = lp.gather(1, ids[bi, pi][:, None])[:, 0]
    ent = -(lp.exp() * lp).sum(-1)
    log_rank = ((lp > tok_lp[:, None]).sum(-1) + 1).float().log()                  # log of 1-based rank
    for b in bi.unique().tolist():
        m = bi == b
        t, e, r = tok_lp[m], ent[m], log_rank[m]
        for k, v in (("sum", t.sum()), ("mean", t.mean()), ("first", t[0]), ("min", t.min()),
                     ("std", t.std(correction=0)), ("ent_mean", e.mean()), ("ent_first", e[0]),
                     ("rank_first", r[0]), ("rank_mean", r.mean()), ("n", torch.tensor(float(t.numel())))):
            out[k][b] = v
    return out


def _ans_matrix(ctx, null):
    cols = [ctx[k] for k in ANS_KEYS]
    pmi_sum = ctx["sum"] - null["sum"]
    cols += [pmi_sum, pmi_sum / ctx["n"].clamp_min(1), ctx["first"] - null["first"]]
    return torch.stack(cols, dim=1)


def _yes_no_ids(tokenizer):
    def ids(words):
        found = set()
        for w in words:
            t = tokenizer.encode(w, add_special_tokens=False)
            if t:
                found.add(t[-1])
        return sorted(found)
    yes, no = ids(["Yes", " Yes", "yes", " yes"]), ids(["No", " No", "no", " no"])
    both = set(yes) & set(no)
    return [i for i in yes if i not in both], [i for i in no if i not in both]


@torch.inference_mode()
def extract_batch(questions, answers, tokenizer, model_llm, config, mode, views, yes_no):
    """Returns seq [B, L, 12], glob [B, G] and {view: [L, B, D] float16}."""
    dev = model_llm.device
    decoder, head = model_llm.get_decoder(), model_llm.get_output_embeddings()

    texts, spans = build_texts(questions, answers, mode)
    ids, mask, in_ans = _encode(tokenizer, texts, spans, dev)
    out = decoder(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    hidden, last_h = out.hidden_states, out.last_hidden_state
    n_total = len(hidden) - 1
    sampled = get_sampled_layer_indices(n_total, config.n_sample_layers)

    B, T = ids.shape
    ar = torch.arange(T, device=dev)[None].expand(B, T)
    has = in_ans.any(1)
    last_pos = torch.where(has, torch.where(in_ans, ar, -1).max(1).values, T - 1)
    first_pos = torch.where(has, torch.where(in_ans, ar, T).min(1).values, T - 1)
    n_ans = in_ans.sum(1).clamp_min(1).float()[:, None]
    w = in_ans.float().unsqueeze(-1)
    bidx = torch.arange(B, device=dev)

    view_out = {v: [] for v in views}
    stats = []
    for li in sampled:
        hs = hidden[li].float()
        last = hs[:, -1]
        amean = torch.where(has[:, None], (hs * w).sum(1) / n_ans, last)
        cache = {"last": last, "alast": hs[bidx, last_pos], "afirst": hs[bidx, first_pos], "amean": amean}
        for v in views:
            view_out[v].append(cache[v])
        stats.append(torch.stack(_layer_stats(last) + [amean.norm(dim=-1), amean.std(-1)], dim=-1))

    seq = torch.nan_to_num(torch.stack(stats, dim=1), nan=0.0, posinf=0.0, neginf=0.0)
    seq = seq.cpu().numpy().astype(np.float32)                                   # [B, L, 12]
    view_np = {v: torch.stack(view_out[v], 0).half().cpu().numpy() for v in views}

    # v1 global features, computed from the next-token distribution at the last position
    logits = head(last_h[:, -1]).float()
    probs = F.softmax(logits, dim=-1)
    top = torch.topk(probs, k=3).values.cpu().numpy()
    logit_entropy = -(probs * torch.log(probs + 1e-10)).sum(-1).cpu().numpy()
    logit_std = logits.std(-1).cpu().numpy()
    logit_max = logits.amax(-1).cpu().numpy()
    norms = seq[:, :, 0]
    norm_diffs = np.diff(norms, axis=1) if norms.shape[1] > 1 else np.zeros_like(norms)
    anchor = get_anchor_stats(seq.transpose(1, 0, 2), sampled, n_total, config.anchor_fractions)
    base = [
        top[:, 0], top[:, 1], top[:, 0] - top[:, 1], logit_entropy, logit_std, logit_max,
        top[:, 2], top[:, 0] - top[:, 2],
        norm_diffs.mean(1), norm_diffs.std(1), norm_diffs.max(1), norm_diffs.min(1),
        norms[:, -1] / (norms[:, 0] + 1e-6), norms[:, -1] - norms[:, 0],
    ]
    for pos in range(len(config.anchor_fractions)):
        base.extend(anchor[pos][:, k] for k in range(3))
    base.extend([anchor[3][:, 0] - anchor[1][:, 0], anchor[3][:, 1] * logit_entropy,
                 logit_std * norm_diffs.mean(1), logit_entropy * logit_std])
    parts = [np.stack(base, axis=1)]

    # answer log-likelihood, rank and PMI against the question-free prompt
    ctx = _answer_token_stats(last_h, head, ids, in_ans)
    n_texts, n_spans = build_null_texts(answers)
    n_ids, n_mask, n_in = _encode(tokenizer, n_texts, n_spans, dev)
    null = _answer_token_stats(decoder(input_ids=n_ids, attention_mask=n_mask).last_hidden_state,
                               head, n_ids, n_in)
    parts.append(_ans_matrix(ctx, null).cpu().numpy())

    if mode == "verify":
        lp = F.log_softmax(logits, dim=-1)
        yes, no = yes_no
        lp_yes, lp_no = lp[:, yes].logsumexp(-1), lp[:, no].logsumexp(-1)
        parts.append(torch.stack([lp_yes, lp_no, lp_yes - lp_no], 1).cpu().numpy())

    glob = np.nan_to_num(np.concatenate(parts, axis=1).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return seq, glob, view_np


def glob_names(mode):
    return BASE_NAMES + ANS_NAMES + (VERIFY_NAMES if mode == "verify" else [])


def _atomic_save(path, arr):
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def extract_dataset_v2(samples, tokenizer, model_llm, config, out_dir, mode="verify",
                       views=("last", "amean"), batch_size=4, checkpoint_every=500, max_fail_frac=0.05):
    """Extract all samples into out_dir, resuming from the last checkpoint if there is one."""
    bad = [v for v in views if v not in VIEWS]
    if bad:
        raise ValueError(f"unknown views {bad}; choose from {VIEWS}")
    os.makedirs(out_dir, exist_ok=True)
    N = len(samples)
    meta_path, prog_path = os.path.join(out_dir, "meta.json"), os.path.join(out_dir, "progress.json")
    p = lambda name: os.path.join(out_dir, name)       # noqa: E731
    yes_no = _yes_no_ids(tokenizer) if mode == "verify" else None
    if mode == "verify" and not (yes_no[0] and yes_no[1]):
        raise RuntimeError("could not find Yes/No token ids for this tokenizer")

    seq = glob = valid = None
    mm = {}
    start = n_failed = 0
    if os.path.exists(meta_path) and os.path.exists(prog_path):
        meta = json.load(open(meta_path))
        prog = json.load(open(prog_path))
        if meta["N"] == N and meta["mode"] == mode and meta["views"] == list(views):
            seq, glob, valid = np.load(p("seq.npy")), np.load(p("glob.npy")), np.load(p("valid.npy"))
            mm = {v: open_memmap(p(f"H_{v}.npy"), mode="r+") for v in views}
            start, n_failed = prog["next"], prog["n_failed"]
            print(f"Resuming from sample {start}/{N}")
        else:
            print(f"Ignoring checkpoint in {out_dir}: different samples, prompt mode or views")

    def checkpoint(next_idx):
        for m in mm.values():
            m.flush()
        _atomic_save(p("seq.npy"), seq)
        _atomic_save(p("glob.npy"), glob)
        _atomic_save(p("valid.npy"), valid)
        json.dump({"next": next_idx, "n_failed": n_failed}, open(prog_path, "w"))

    last_saved = start
    for i in tqdm(range(start, N, batch_size)):
        batch = samples[i:i + batch_size]
        try:
            s, g, hv = extract_batch([b["question"] for b in batch], [b["answer"] for b in batch],
                                     tokenizer, model_llm, config, mode, views, yes_no)
        except Exception as e:      # a failed batch is skipped; its rows stay invalid
            n_failed += len(batch)
            print(f"[extract2] samples {i}-{i + len(batch) - 1} failed: {type(e).__name__}: {e}")
            if n_failed > max_fail_frac * N:
                raise RuntimeError(f"{n_failed}/{N} samples failed; aborting extraction") from e
            continue
        if seq is None:     # first good batch fixes the shapes
            L, D, G = s.shape[1], next(iter(hv.values())).shape[2], g.shape[1]
            seq = np.zeros((N, L, s.shape[2]), np.float32)
            glob = np.zeros((N, G), np.float32)
            valid = np.zeros(N, bool)
            mm = {v: open_memmap(p(f"H_{v}.npy"), mode="w+", dtype=np.float16, shape=(L, N, D)) for v in views}
            json.dump({"N": N, "L": L, "D": D, "G": G, "mode": mode, "views": list(views),
                       "glob_names": glob_names(mode)}, open(meta_path, "w"))
        n = len(batch)
        seq[i:i + n], glob[i:i + n], valid[i:i + n] = s, g, True
        for v in views:
            mm[v][:, i:i + n] = hv[v]
        if i + n - last_saved >= checkpoint_every:
            checkpoint(i + n)
            last_saved = i + n
        if dev_is_cuda(model_llm) and (i // batch_size) % 50 == 49:
            torch.cuda.empty_cache()

    if seq is None:
        raise RuntimeError("No features extracted")
    checkpoint(N)
    np.save(p("y.npy"), np.array([s["is_hallucination"] for s in samples], dtype=np.int64))
    np.save(p("groups.npy"), make_groups(samples))
    if n_failed:
        print(f"[extract2] {n_failed}/{N} samples failed and were skipped (valid.npy marks them)")
    return out_dir


def dev_is_cuda(model):
    return model.device.type == "cuda"


def load_v2(feat_dir):
    """Everything extract_dataset_v2 wrote, except the big H files (those are opened per layer)."""
    with open(os.path.join(feat_dir, "meta.json")) as f:
        meta = json.load(f)
    if not os.path.exists(os.path.join(feat_dir, "y.npy")):
        raise FileNotFoundError(f"{feat_dir} is incomplete; rerun --stage extract")
    ld = lambda n: np.load(os.path.join(feat_dir, n))      # noqa: E731
    return {"meta": meta, "seq": ld("seq.npy"), "glob": ld("glob.npy"), "y": ld("y.npy"),
            "groups": ld("groups.npy"), "valid": ld("valid.npy")}
