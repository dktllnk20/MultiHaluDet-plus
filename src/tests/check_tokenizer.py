"""Check that the answer-token detection works with a REAL tokenizer (no model weights needed).

    python tests/check_tokenizer.py mistral-7b
    python tests/check_tokenizer.py llama2-7b

Run it once per model before the v2 extraction. It must print 0 mismatches.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

from src.config import MODEL_REGISTRY
from src.data import probe_extractor as pe
from src.data.loader import load_triviaqa

name = sys.argv[1] if len(sys.argv) > 1 else "mistral-7b"
tok = AutoTokenizer.from_pretrained(MODEL_REGISTRY[name])
tok.pad_token = tok.eos_token
samples = load_triviaqa(lang="en")[:300]
bad = 0
for mode in ("plain", "verify"):
    for i in range(0, len(samples), 8):
        b = samples[i:i + 8]
        texts, spans = pe.build_texts([s["question"] for s in b], [s["answer"] for s in b], mode)
        ids, _, in_ans = pe._encode(tok, texts, spans, "cpu")
        for j, s in enumerate(b):
            got = tok.decode(ids[j][in_ans[j]]).strip()
            if got != str(s["answer"]).strip():
                bad += 1
                if bad <= 5:
                    print(f"[{mode}] mismatch: decoded {got!r} vs answer {s['answer']!r}")
yes, no = pe._yes_no_ids(tok)
print("Yes ids:", yes, [tok.convert_ids_to_tokens(i) for i in yes])
print("No ids: ", no, [tok.convert_ids_to_tokens(i) for i in no])
print(f"{bad} mismatches out of {2 * len(samples)} prompts")
