# MultiHaluDet plus

White-box hallucination detection for LLMs from their internal states, evaluated on HaluEval QA (English, French, Vietnamese, Bangali) and on a TriviaQA hard-negative set.

![Pipeline](./figures/pipeline-EN.png)

## Overview

1. **Translation**: HaluEval is translated into other languages with `translategemma-12b-it` (`data/halueval_{lang}.json`).
2. **Feature extraction** (frozen LLM, 4-bit NF4 by default). Two feature sets are available:
   - **Base** (default): per-layer statistics of hidden states from 32 sampled layers (12 statistics x L), plus 30 global features from the output logits.
   - **v2** (`--v2`): keeps the full **hidden-state vectors** (views `last`, `amean`) and uses a **verify prompt** ("Is the proposed answer correct? Answer (Yes or No):"). Global features grow to 46 (30 base + 13 answer log-likelihood / PMI + 3 Yes/No log-probs).
3. **Layer probes** (v2 only): for every (view, layer), an out-of-fold linear probe `StandardScaler -> PCA(256) -> LogisticRegression(C=0.02)`. Folds are grouped by question. Probes run automatically after extraction when `--v2` is enabled. Their logits are appended to the network input.
4. **MultiHaluDet** (trained per inner fold): a sequential branch (multi-scale attention, Transformer, attention pooling) and a global branch (MLP), merged by gated fusion. Loss: BCE + Focal + Asymmetric + Contrastive, AdamW + EMA, Mixup/CutMix. v2 additionally uses delta channels, layer-weight scaling and a pairwise AUC rank loss.
5. **Stacking**: out-of-fold outputs of the 5 fold networks feed the base classifiers (RF, HistGB, LR, SVM, XGB, LGBM); a logistic-regression meta-learner combines their log-odds. The final probability is averaged over the 5 test sets.

### What v2 adds over Base

| | Base | v2 |
|---|---|---|
| LLM features | scalar statistics only | scalar statistics + full hidden-state vectors + per-layer probe logits |
| Prompt | `Question: ... Answer: ...` | verify prompt, Yes/No log-probs, PMI against a question-free prompt |
| Global features | 30 | 46 (+ 2V probe summaries) |
| Stages | extract, train_oof, ensemble | extract (includes **probe**), train_oof, ensemble |
| Network | sequential input of 12 channels | + probe channels, delta channels, layer-weight scaling, pairwise AUC rank loss |
| Train/test split | by row | **by question** (both answers of a question stay together) |
| Stacker | 5-fold by row, 6 classifiers, 576-d features | folds by question, logit stacking, optional `lite` stacker |
| Extra metric | - | **pair accuracy** (wrong answer scored above the correct one) |
| Optional | - | `--pair_center` (subtract per-question mean) |

## Setup

```bash
git clone https://github.com/Danglee1107/MultiHaluDet-exhance.git && cd MultiHaluDet-exhance
```
then

```bash
uv sync
```
or

```bash
pip install -r requirements.txt
```

Gated models (e.g. Llama-2) need a Hugging Face token. Put it in a `.env` file at the repository root; `src/config.py` loads it automatically:

```bash
echo "HF_TOKEN=hf_xxxxxxxx" > .env
```

## Structure

```
.
├── run_pipeline.py
├── run_all.sh                 # Base + v2, several seeds, resumable
├── figures/pipeline.svg
├── data/                      # halueval_{en,fr,bn,vi}.json, triviaqa_{lang}.json
├── src/
│   ├── config.py
│   ├── data/        loader.py, feature_extractor.py (Base), probe_extractor.py (v2), splits.py
│   ├── probe/       layer_probe.py
│   ├── models/      multihaludet.py, losses.py
│   ├── training/    trainer.py, augmentations.py
│   ├── ensemble/    meta_learner.py
│   └── utils/       metrics.py, visualization.py
└── results/                   # created automatically
    ├── features/              # Base LLM features
    ├── features_v2/           # v2 hidden-state vectors and scalar features
    ├── probe/                 # v2 probe logits (.npz) and per-layer AUC report (_auc.txt)
    ├── oof/                   # out-of-fold outputs
    └── plots/
```

Data notes:
- `en` HaluEval falls back to HuggingFace if `data/halueval_en.json` is missing; other languages need a local translated file.
- TriviaQA expects `data/triviaqa_{lang}.json` with exactly two rows per question (one correct answer = label 0, one hard negative = label 1). Without it, `en` falls back to a TriviaQA version with random wrong answers.

Hardware: a CUDA GPU is expected (the 4-bit LLM takes ~4 GB, so a 6 GB GPU is enough). Each v2 view takes about 2.6 GB of disk per run for a 32-layer, 4096-d model.

## Run

```bash
python run_pipeline.py --v2 --models mistral-7b --dataset halueval --lang en --stage extract
```

`--stage extract` performs feature extraction and, with `--v2`, layer probing. `train_oof` trains the detector, `ensemble` evaluates saved OOF outputs, and `all` runs every stage. The default feature set is Base, the default language is English, and the default seed is 42.

Run both supported models:

```bash
python run_pipeline.py --models mistral-7b llama2-7b --dataset triviaqa --lang en --stage all
```

Use `--limit` to process only the first N records, and `--seed` to change the random seed. Logs are written to `logs/mistral/` or `logs/llama/` as `<dataset>_<language>_<base|v2>_s<seed>.log`.

### Supported options

| Flag | Values | Default |
|---|---|---|
| `--v2` | Boolean flag; enables v2 features and probes | off (Base) |
| `--models` | `mistral-7b`, `llama2-7b` (one or both) | `mistral-7b` |
| `--lang` | `en`, `fr`, `bn`, `am`, `vi` | `en` |
| `--dataset` | `halueval`, `triviaqa` | `halueval` |
| `--stage` | `extract`, `train_oof`, `ensemble`, `all` | `all` |
| `--limit` | Positive record count | No limit |
| `--seed` | integer | 42 |

### Outputs

`results/features/` or `results/features_v2/` (LLM features), `results/probe/` (probe logits and a per-layer AUC table), `results/oof/` (out-of-fold outputs), `results/plots/results_*.png` (ROC, PR, calibration, confusion matrix, ...). AUC, F1 and accuracy are printed to the terminal; pair accuracy is printed too when the split is by question.

## Notes

- The seed controls the dataset split, inner folds, and (for v2) probe folds.
- Bangla (`bn`) and Vietnamese (`vi`) require a local translated file `data/halueval_{lang}.json`.
- Base uses a row-level split, while v2 splits by question; results are not directly comparable.
