# Hidden-State Probes, Answer Likelihood, and Surface Shortcuts in White-Box Hallucination Detection

White-box hallucination detection for LLMs from their internal states, evaluated on English HaluEval QA and a TriviaQA hard-negative set.

![Pipeline](./figures/pipeline.png)

## Overview

1. **Feature extraction** (frozen LLM, 4-bit NF4 by default). Two feature sets are available:
   - **Base** (default): per-layer statistics of hidden states from 32 sampled layers (12 statistics x L), plus 30 global features from the output logits.
   - **v2** (`--v2`): keeps the full **hidden-state vectors** (views `last`, `amean`) and uses a **verify prompt** ("Is the proposed answer correct? Answer (Yes or No):"). Global features grow to 46 (30 base + 13 answer log-likelihood / PMI + 3 Yes/No log-probs).
2. **Layer probes** (v2 only): for every (view, layer), an out-of-fold linear probe `StandardScaler -> PCA(256) -> LogisticRegression(C=0.02)`. Folds are grouped by question. Probes run automatically after extraction when `--v2` is enabled. Their logits are appended to the network input.
3. **MultiHaluDet** (trained per inner fold): a sequential branch (multi-scale attention, Transformer, attention pooling) and a global branch (MLP), merged by gated fusion. Loss: BCE + Focal + Asymmetric + Contrastive, AdamW + EMA, Mixup/CutMix. v2 can optionally enable delta channels, layer-weight scaling and a pairwise AUC rank loss; all three are off by default.
4. **Stacking**: out-of-fold outputs of the 5 fold networks feed the base classifiers (RF, HistGB, LR, SVM, XGB, LGBM); a logistic-regression meta-learner combines their log-odds. The final probability is averaged over the 5 test sets.

### What v2 adds over Base

| | Base | v2 |
|---|---|---|
| LLM features | 12 scalar statistics per sampled layer | same scalar statistics; hidden-state vectors are extracted to train probes, whose per-layer logits are fed to the detector |
| Prompt | `Question: ... Answer: ...` | verify prompt by default; adds answer-token likelihood/PMI features and verify-only Yes/No log-probs; plain prompt is also supported |
| Global features | 30 | 43 with plain prompt or 46 with verify prompt, plus 2 values per probe view when global probes are enabled (default: 50 total with verify and 2 views) |
| Stages | extract, train_oof, ensemble | extract (includes **probe**), train_oof, ensemble |
| Network | 12 sequential channels per layer | probe-logit channels are enabled by default; delta channels, layer-weight scaling and pairwise AUC rank loss are optional (off by default) |
| Train/test split | by row | **by question** (both answers of a question stay together) |
| Stacker | 5-fold by row; 576-d vector features and 6 classifiers by default | 5-fold by question; 576-d vector features and 6 classifiers by default; optional OOF-logit input (`--stack_mode logit`) or 2-classifier `lite` stacker |
| Extra metric | - | **pair accuracy** (wrong answer scored above the correct one) |
| Optional | - | `--pair_center` (subtract per-question mean) |

### Results summary

| Dataset          | Model      | Base AUROC | V2 AUROC  | Δ AUROC   |
| ---------------- | ---------- | ---------- | --------- | --------- |
|  HaluEval | Mistral-7B | 0.9849     | **0.9931** | +0.0082 |
|  HaluEval | Llama-2-7B | 0.9846     | **0.9917** | +0.0071 |
|  TriviaQA | Mistral-7B | 0.6452     | **0.8226** | +0.1774 |
|  TriviaQA | Llama-2-7B | 0.5931     | **0.7650** | +0.1719 |

## Installation

```bash
git clone https://github.com/dktllnk20/MultiHaluDet-plus.git && cd MultiHaluDet-plus
```
then

```bash
pip install -r requirements.txt
```

Gated models (e.g. Llama-2) need a Hugging Face token. Put it in a `.env` file at the repository root; `src/config.py` loads it automatically:

```bash
echo "HF_TOKEN=hf_xxxxxxxx" > .env
```


Data notes:
- HaluEval uses English data from `data/halueval.json`, falling back to HuggingFace if the file is missing.
- TriviaQA expects `data/triviaqa.json` with exactly two rows per question (one correct answer = label 0, one hard negative = label 1). Without it, it falls back to a TriviaQA version with random wrong answers.

> [!NOTE]
> For TriviaQA, the dataset was edited and relabeled from lucadiliello/triviaqa on Hugging Face. If you would like to reproduce our experiments, please contact us using the Gmail address below.

Hardware: All experiments were conducted on a system equipped with an **AMD Ryzen 7 7435HS CPU** and an **NVIDIA GeForce RTX 4050 GPU** with 6 GB of VRAM.

## Usage

### Pipeline
```bash
# all pipeline
python run_pipeline.py --models mistral-7b llama2-7b --dataset triviaqa halueval --stage all --seed 42

# stage by stage
python run_pipeline.py --models mistral-7b --dataset halueval --stage extract
python run_pipeline.py --models mistral-7b --dataset halueval --stage train_oof
python run_pipeline.py --models mistral-7b --dataset halueval --stage ensemble

```

### Research question
```bash
python run_pipeline.py --rq 1
python run_pipeline.py --rq 2
```

### supported models
- mistral-7b
- llama2-7b


## Folder Structure

```
src/
├── data/
├── ensemble/
├── models/
├── probe/
├── rq/
├── settings/
├── training/
└── utils/
```

## 🔗contact

Have questions about this project, want to reproduce our experiments, or just want to know more? Feel free to reach out! We'd be happy to hear from you.

- ✉️**Email:** dktlln.k20@gmail.com

Whether it's a question, feedback, or a collaboration idea, don't hesitate to drop us an email. We're always happy to connect!
