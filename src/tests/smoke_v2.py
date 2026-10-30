"""CPU smoke test of the v2 pipeline on a tiny random Llama and a planted signal.

    python tests/smoke_v2.py

Answers of label 0 come from word list A and answers of label 1 from list B, so a probe on the
answer-token hidden states can separate them; this checks the plumbing (answer positions, probes,
group split, pair centering), not the quality of any real model.
"""
import json
import os
import random
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

import run_pipeline as rp
from src.data import probe_extractor as pe

random.seed(0)
torch.manual_seed(0)
A = ["Paris", "Everest", "Nile", "oxygen", "Mozart", "Titanic", "Mars", "violin", "Tokyo", "copper"]
B = ["Gandalf", "Zebra", "Quartz", "Lisbon", "Hamlet", "Cobalt", "Sahara", "Vienna", "Ottawa", "Tulip"]

qs = list(dict.fromkeys(x["question"] for x in json.load(open("data/triviaqa_en.json"))))[:500]
samples = []
for q in qs:
    samples += [{"question": q, "answer": random.choice(A) + " " + random.choice(A), "is_hallucination": 0},
                {"question": q, "answer": random.choice(B) + " " + random.choice(B), "is_hallucination": 1}]
random.shuffle(samples)

tk = Tokenizer(models.BPE(unk_token="<unk>"))
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tk.decoder = decoders.ByteLevel()
corpus = [s["question"] for s in samples] + [s["answer"] for s in samples] + \
         ["Question: Proposed answer: Answer: Is the proposed answer correct? (Yes or No) Yes No yes no"]
tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=1500, special_tokens=["<unk>", "<s>", "</s>"],
                                                   initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
tk.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", tk.token_to_id("<s>"))])
tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", unk_token="<unk>")
tok.pad_token = tok.eos_token
model = LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                     num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512)).eval()

# --- unit checks of the extractor ------------------------------------------------------------
cfg = rp.get_config(); cfg.n_sample_layers = 4
for mode in ("plain", "verify"):
    qb, ab = [s["question"] for s in samples[:6]], [s["answer"] for s in samples[:6]]
    texts, spans = pe.build_texts(qb, ab, mode)
    ids, mask, in_ans = pe._encode(tok, texts, spans, "cpu")
    for b in range(6):
        dec = tok.decode(ids[b][in_ans[b]]).strip()
        assert dec == ab[b], (mode, dec, ab[b])
    # answer log-likelihood against a direct computation with the full LM head
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask).logits.float()
        ctx = pe._answer_token_stats(model.get_decoder()(input_ids=ids, attention_mask=mask).last_hidden_state,
                                     model.get_output_embeddings(), ids, in_ans)
    lp = torch.log_softmax(logits, -1)
    for b in range(6):
        pos = in_ans[b].nonzero()[:, 0]
        want = lp[b, pos - 1].gather(1, ids[b, pos][:, None]).sum().item()
        assert abs(ctx["sum"][b].item() - want) < 1e-3, (ctx["sum"][b].item(), want)
    seq, glob, hv = pe.extract_batch(qb, ab, tok, model, cfg, mode, ("last", "alast", "afirst", "amean"),
                                     pe._yes_no_ids(tok) if mode == "verify" else None)
    assert seq.shape == (6, 4, 12) and glob.shape == (6, len(pe.glob_names(mode))), (seq.shape, glob.shape)
    assert hv["last"].shape == (4, 6, 64) and np.isfinite(glob).all()
print("extractor unit checks passed (answer spans, log-likelihoods, shapes)")

# --- end to end -------------------------------------------------------------------------------
work = tempfile.mkdtemp()
os.chdir(work)
rp._load_llm = lambda args, config: (tok, model)
rp._load_samples = lambda args, config: samples
base_get = rp.get_config
def small_config():
    c = base_get()
    c.n_sample_layers, c.epochs, c.patience, c.warmup_epochs = 4, 6, 3, 1
    c.batch_size, c.fold_workers, c.probe_jobs, c.n_inner_folds = 64, 1, 1, 3
    c.use_amp = False
    return c
rp.get_config = small_config

def run(*argv):
    sys.argv = ["run_pipeline.py", "--dataset", "triviaqa", "--model", "mistral-7b", "--seed", "1",
                "--hidden_dim", "64", "--num_layers", "1", "--extract_batch_size", "8", *argv]
    rp.main()

run("--feat", "v2", "--stage", "extract", "--views", "last,alast,afirst,amean", "--prompt_mode", "verify")
run("--feat", "v2", "--stage", "probe", "--views", "last,alast,afirst,amean", "--prompt_mode", "verify", "--pca_dim", "16")
run("--feat", "v2", "--stage", "train_oof", "--views", "last,alast,afirst,amean", "--prompt_mode", "verify",
    "--pca_dim", "16", "--improved", "--stack_mode", "logit", "--tag", "t1")
run("--feat", "v2", "--stage", "ensemble", "--views", "last,alast,afirst,amean", "--prompt_mode", "verify",
    "--pca_dim", "16", "--improved", "--stack_mode", "logit", "--stacker", "lite", "--tag", "t1")

# group split really separates questions
y_tr = np.load(glob_ := [f for f in os.listdir("results/oof") if f.startswith("g_train")][0] and "results/oof/" + [f for f in os.listdir("results/oof") if f.startswith("g_train")][0])
y_te = np.load("results/oof/" + [f for f in os.listdir("results/oof") if f.startswith("g_test")][0])
assert not set(y_tr) & set(y_te), "a question appears in both train and test"
print("group split ok:", len(set(y_tr)), "train questions,", len(set(y_te)), "test questions")

# pair centering on the v2 features, plain prompt, and the v1 path with group split + pair centering
run("--feat", "v2", "--stage", "all", "--views", "last,amean", "--prompt_mode", "plain", "--pca_dim", "16",
    "--pair_center", "--stack_mode", "logit", "--stacker", "lite", "--tag", "pc")
run("--stage", "extract", "--tag", "v1")
run("--stage", "train_oof", "--group_split", "--pair_center", "--stack_mode", "logit", "--tag", "v1pc")
run("--stage", "ensemble", "--group_split", "--pair_center", "--stack_mode", "logit", "--stacker", "lite", "--tag", "v1pc")
# base v1 path unchanged
run("--stage", "train_oof", "--stack_mode", "vector", "--tag", "v1base")
run("--stage", "ensemble", "--stack_mode", "vector", "--tag", "v1base")
print("SMOKE TEST PASSED")
shutil.rmtree(work, ignore_errors=True)
