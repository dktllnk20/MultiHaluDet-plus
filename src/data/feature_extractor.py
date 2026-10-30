import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Feature layout. The first GLOB_BASE_DIM global columns are the original features; the last 4 are
# the answer log-likelihood statistics, which run_pipeline.py drops unless --answer_ll is given.
SEQ_DIM = 12                    # 10 last-token statistics + 2 mean-over-tokens statistics
GLOB_BASE_DIM = 30
GLOB_DIM = GLOB_BASE_DIM + 4    # + answer-span log-likelihood statistics

def get_sampled_layer_indices(n_total_layers: int, n_sample: int) -> list[int]:
    if n_total_layers <= 0:
        raise ValueError(f"Model reported {n_total_layers} transformer layers")

    if n_sample == 1:
        return [n_total_layers]

    if n_total_layers <= n_sample:
        indices = list(range(1, n_total_layers + 1))
        while len(indices) < n_sample:
            indices.append(indices[-1])
        return indices
    else:
        return [
            max(1, min(n_total_layers,
                       round(1 + (n_total_layers - 1) * i / (n_sample - 1))))
            for i in range(n_sample)
        ]

def get_anchor_stats(seq_feats: list, sampled_indices: list[int],
                     n_total_layers: int, anchor_fractions: list[float]) -> dict:
    sampled_arr = np.array(sampled_indices)
    anchor_stats = {}
    for pos, frac in enumerate(anchor_fractions):
        target_layer = max(1, min(n_total_layers, round(frac * n_total_layers)))
        closest_rank = int(np.argmin(np.abs(sampled_arr - target_layer)))
        anchor_stats[pos] = seq_feats[closest_rank]
    return anchor_stats

def _layer_stats(x):
    """10 statistics of the last token per row of x [B, H]; the caller appends 2 more."""
    mu = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    p = F.softmax(x, dim=-1)
    med = x.median(-1, keepdim=True).values
    kurtosis = (((x - mu) / std.clamp_min(1e-9)) ** 4).mean(-1)
    return [
        x.norm(dim=-1), mu.squeeze(-1), std.squeeze(-1), x.amin(-1), x.amax(-1),
        (x > 0).float().mean(-1),
        (x.abs() < 0.1).float().mean(-1),
        -(p * torch.log(p + 1e-9)).sum(-1),
        torch.where(std.squeeze(-1) < 1e-9, 0.0, kurtosis),
        (x - med).abs().median(-1).values,
    ]

def _answer_lengths(questions, tokenizer, attention_mask):
    """Number of answer tokens per prompt, i.e. the tokens after 'Answer:'.

    With left padding they are always the last n tokens of the row. A prompt truncated inside
    the question has 0 answer tokens.
    """
    prefixes = [f"Question: {q}\nAnswer:" for q in questions]
    prefix_len = [len(ids) for ids in tokenizer(prefixes, truncation=True, max_length=256)["input_ids"]]
    n_real = attention_mask.sum(1).tolist()
    return [max(0, int(n) - min(p, int(n))) for n, p in zip(n_real, prefix_len)]

def _answer_ll_stats(logits_kept, input_ids, n_ans):
    """4 statistics of log p(answer token | question, earlier answer tokens) per prompt, [B, 4]:
    mean negative log-likelihood, minimum token log-prob, std of token log-probs, mean token entropy.

    logits_kept holds the logits of the last K positions (K >= max(n_ans) + 1). The logits at
    position j-1 predict the token at position j, so the n answer tokens are predicted by the
    n positions that end one before the last.
    """
    B, K, _ = logits_kept.shape
    T = input_ids.shape[1]
    out = torch.zeros(B, 4, device=logits_kept.device)
    for b, n in enumerate(n_ans):
        if n == 0:
            continue
        lp = F.log_softmax(logits_kept[b, K - 1 - n:K - 1].float(), dim=-1)      # [n, V]
        tok_lp = lp.gather(-1, input_ids[b, T - n:, None]).squeeze(-1)            # [n]
        entropy = -(lp.exp() * lp).sum(-1)
        out[b] = torch.stack([-tok_lp.mean(), tok_lp.min(), tok_lp.std(correction=0), entropy.mean()])
    return out

@torch.inference_mode()
def extract_features(questions, answers, tokenizer, model_llm, config):
    """Features for a batch of question/answer pairs.

    Returns (seq [B, n_sample_layers, 12], glob [B, 34]) as float32 arrays.
    """
    prompts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(questions, answers, strict=True)]
    # Left padding keeps every prompt's last real token at index -1.
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=256).to(model_llm.device)
    n_ans = _answer_lengths(questions, tokenizer, inputs["attention_mask"])
    # Logits are only needed for the answer span plus the last position, which is right-aligned.
    outputs = model_llm(**inputs, output_hidden_states=True, logits_to_keep=max(n_ans) + 1)

    hidden_states = outputs.hidden_states
    n_total_layers  = len(hidden_states) - 1
    sampled_indices = get_sampled_layer_indices(n_total_layers,
                                                config.n_sample_layers)

    logits = outputs.logits[:, -1].float()
    probs = F.softmax(logits, dim=-1)

    mask = inputs["attention_mask"].unsqueeze(-1).float()
    n_tokens = mask.sum(1)
    layers = []
    for layer_idx in sampled_indices:
        hs = hidden_states[layer_idx].float()
        mean_hs = (hs * mask).sum(1) / n_tokens          # mean over real tokens only

        layers.append(torch.stack(
            _layer_stats(hs[:, -1]) + [mean_hs.norm(dim=-1), mean_hs.std(-1)], dim=-1))
    seq = torch.nan_to_num(torch.stack(layers, dim=1), nan=0.0, posinf=0.0, neginf=0.0)
    seq = seq.cpu().numpy().astype(np.float32)           # [B, L, 12]

    top = torch.topk(probs, k=3).values.cpu().numpy()
    logit_entropy = -(probs * torch.log(probs + 1e-10)).sum(-1).cpu().numpy()
    logit_std = logits.std(-1).cpu().numpy()
    logit_max = logits.amax(-1).cpu().numpy()

    norms = seq[:, :, 0]
    # One sampled layer has no layer-to-layer difference; use 0 so the statistics stay defined.
    norm_diffs = np.diff(norms, axis=1) if norms.shape[1] > 1 else np.zeros_like(norms)
    # [L, B, C] so that get_anchor_stats picks one layer for the whole batch
    anchor_stats = get_anchor_stats(seq.transpose(1, 0, 2), sampled_indices,
                                    n_total_layers, config.anchor_fractions)

    glob = [
        top[:, 0], top[:, 1], top[:, 0] - top[:, 1],
        logit_entropy, logit_std, logit_max,
        top[:, 2], top[:, 0] - top[:, 2],
        norm_diffs.mean(1), norm_diffs.std(1), norm_diffs.max(1), norm_diffs.min(1),
        norms[:, -1] / (norms[:, 0] + 1e-6),
        norms[:, -1] - norms[:, 0],
    ]
    for pos in range(len(config.anchor_fractions)):
        glob.extend(anchor_stats[pos][:, k] for k in range(3))
    glob.extend([
        anchor_stats[3][:, 0] - anchor_stats[1][:, 0],
        anchor_stats[3][:, 1] * logit_entropy,
        logit_std * norm_diffs.mean(1),
        logit_entropy * logit_std,
    ])
    answer_ll = _answer_ll_stats(outputs.logits, inputs["input_ids"], n_ans).cpu().numpy()   # [B, 4]
    glob.extend(answer_ll[:, k] for k in range(4))
    return seq, np.stack(glob, axis=1).astype(np.float32)

def _save_checkpoint(path, seq, glob, labels, next_idx, n_failed, n_total):
    tmp = path + ".tmp.npz"
    np.savez(tmp, seq=np.concatenate(seq), glob=np.concatenate(glob), y=np.array(labels),
             next=next_idx, n_failed=n_failed, n_total=n_total)
    os.replace(tmp, path)  # atomic: a kill during the write keeps the old checkpoint

def extract_dataset(samples, tokenizer, model_llm, config, ckpt_path,
                    batch_size=16, checkpoint_every=500, max_fail_frac=0.05):
    """Extract features for all samples, resuming from ckpt_path if it exists.

    Returns (X_seq, X_glob, y). A failed batch is reported and skipped; more than
    max_fail_frac failed samples aborts the run. The caller deletes ckpt_path
    after it has saved the result.
    """
    all_seq, all_glob, all_labels = [], [], []
    start = n_failed = 0
    if os.path.exists(ckpt_path):
        with np.load(ckpt_path) as ckpt:
            if (int(ckpt["n_total"]) == len(samples)
                    and ckpt["seq"].shape[2] == SEQ_DIM and ckpt["glob"].shape[1] == GLOB_DIM):
                all_seq, all_glob = [ckpt["seq"]], [ckpt["glob"]]
                all_labels = list(ckpt["y"])
                start, n_failed = int(ckpt["next"]), int(ckpt["n_failed"])
                print(f"Resuming from sample {start}/{len(samples)}")
            else:
                print(f"Ignoring {ckpt_path}: it belongs to a different sample set or feature layout")

    last_saved = start
    for i in tqdm(range(start, len(samples), batch_size)):
        batch = samples[i:i + batch_size]
        try:
            seq, glob = extract_features([b["question"] for b in batch],
                                         [b["answer"] for b in batch],
                                         tokenizer, model_llm, config)
        # ponytail: a failed batch drops all its samples; retry them one by one if that costs data
        except Exception as e:
            n_failed += len(batch)
            print(f"[extract] samples {i}-{i + len(batch) - 1} failed: {type(e).__name__}: {e}")
            if n_failed > max_fail_frac * len(samples):
                raise RuntimeError(
                    f"{n_failed}/{len(samples)} samples failed; aborting extraction") from e
            continue
        all_seq.append(seq)
        all_glob.append(glob)
        all_labels.extend(b["is_hallucination"] for b in batch)
        if i + len(batch) - last_saved >= checkpoint_every:
            _save_checkpoint(ckpt_path, all_seq, all_glob, all_labels,
                             i + len(batch), n_failed, len(samples))
            last_saved = i + len(batch)
        if model_llm.device.type == "cuda" and (i // batch_size) % 50 == 49:
            torch.cuda.empty_cache()

    if not all_seq:
        raise RuntimeError("No features extracted")
    if n_failed:
        print(f"[extract] {n_failed}/{len(samples)} samples failed and were skipped")
    return np.concatenate(all_seq), np.concatenate(all_glob), np.array(all_labels)
