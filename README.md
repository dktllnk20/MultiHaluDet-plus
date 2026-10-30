# MultiHaluDet plus

White-box hallucination detection for LLMs from their internal states, evaluated on HaluEval QA (English, French, Vietnamese, Bangali) and on a TriviaQA hard-negative set.

![Pipeline](./figures/pipeline.png)

## Overview

1. **Translation**: HaluEval is translated into other languages with `translategemma-12b-it` (`data/halueval_{lang}.json`).
2. **Feature extraction** (frozen LLM, 4-bit NF4 by default). Two feature sets are available:
   - **Base** (`--feat v1`, default): per-layer statistics of hidden states from 32 sampled layers (12 statistics x L), plus 30 global features from the output logits.
   - **v2** (`--feat v2`): keeps the full **hidden-state vectors** (views `last`, `amean`) and uses a **verify prompt** ("Is the proposed answer correct? Answer (Yes or No):"). Global features grow to 46 (30 base + 13 answer log-likelihood / PMI + 3 Yes/No log-probs).
3. **Layer probes** (v2 only): for every (view, layer), an out-of-fold linear probe `StandardScaler -> PCA(256) -> LogisticRegression(C=0.02)`. Folds are grouped by question. The probe logits are appended to the network input: sequential input becomes 12 + V channels and global input becomes 46 + 2V features (V = number of views).
4. **MultiHaluDet** (trained per inner fold): a sequential branch (multi-scale attention, Transformer, attention pooling) and a global branch (MLP), merged by gated fusion. Loss: BCE + Focal + Asymmetric + Contrastive, AdamW + EMA, Mixup/CutMix. v2 additionally uses delta channels, layer-weight scaling and a pairwise AUC rank loss.
5. **Stacking**: out-of-fold outputs of the 5 fold networks feed the base classifiers (RF, HistGB, LR, SVM, XGB, LGBM; or only LR + LGBM with `--stacker lite`); a logistic-regression meta-learner combines their log-odds. The final probability is averaged over the 5 test sets. Base stacks the 576-d network features; v2 stacks one logit per fold network together with the global features.

### What v2 adds over Base

| | Base | v2 |
|---|---|---|
| LLM features | scalar statistics only | scalar statistics + full hidden-state vectors + per-layer probe logits |
| Prompt | `Question: ... Answer: ...` | verify prompt, Yes/No log-probs, PMI against a question-free prompt |
| Global features | 30 | 46 (+ 2V probe summaries) |
| Stages | extract, train_oof, ensemble | extract, **probe**, train_oof, ensemble |
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

Hardware: a CUDA GPU is expected (the 4-bit LLM takes ~4 GB, so a 6 GB GPU is enough). On CUDA OOM during extraction, use `--extract_batch_size 2`. Each v2 view takes about 2.6 GB of disk per run for a 32-layer, 4096-d model.

## Run

The default seed is **42** (`Config.seed`); omit `--seed` to use it.

### Base

```bash
python run_pipeline.py --model mistral-7b --lang en --tag base
```

Stage by stage (extracted LLM features are reused across seeds):

```bash
python run_pipeline.py --model mistral-7b --lang vi --stage extract
python run_pipeline.py --model mistral-7b --lang vi --stage train_oof --tag base
python run_pipeline.py --model mistral-7b --lang vi --stage ensemble  --tag base
```

### v2 (hidden-state probes)

v2 runs pass `--improved`, which switches on the v2 model components (delta channels, layer-weight scaling, pairwise AUC rank loss, answer log-likelihood features, logit stacking). Run the stages in this order:

```bash
B="python run_pipeline.py --model mistral-7b --lang vi --feat v2"

$B --stage extract   --prompt_mode verify --views last,amean --extract_batch_size 2
$B --stage probe     --views last,amean
$B --stage train_oof --improved --tag v2
$B --stage ensemble  --improved --stacker lite --tag v2
```

Rules to keep in mind:
- `--views` must be the same in `extract` and `probe`.
- The probe file depends on `--seed`, `--views`, `--pair_center`, `--pca_dim` and `--probe_C`. Re-run `--stage probe` when any of them changes (use `--redo_probe` to overwrite).
- `train_oof` and `ensemble` must use the same flags and the same `--tag`.
- `--feat v2` always splits by question. For a fair Base vs v2 comparison, run Base with `--group_split` too.
- Extraction writes a checkpoint every 500 samples; if it is interrupted, run the same command again to resume.

### Many seeds / languages / models

`run_all.sh` runs Base and v2 for every seed in `SEEDS` (1, 11, 29, 37, 43), extracts features once per (language, model), skips parts that already finished, and resumes if interrupted:

```bash
bash run_all.sh                      # HaluEval
DATASET=triviaqa bash run_all.sh     # TriviaQA
```

Edit `LANGS`, `MODELS` and `SEEDS` at the top of the script to change the grid. Logs go to `logs/multi/`.

Manual loop for Base only:

```bash
for s in 1 11 29 37 43; do
  python run_pipeline.py --model mistral-7b --lang en --stage train_oof --seed $s --tag base_s$s
  python run_pipeline.py --model mistral-7b --lang en --stage ensemble  --seed $s --tag base_s$s
done
```

### Main options

| Flag | Values | Default |
|---|---|---|
| `--model` | `mistral-7b`, `llama2-7b` | `mistral-7b` |
| `--lang` | `en`, `fr`, `bn`, `am`, `vi` | `en` |
| `--dataset` | `halueval`, `triviaqa` | `halueval` |
| `--stage` | `all`, `extract`, `probe`, `train_oof`, `ensemble` | `all` |
| `--seed` | integer | 42 |
| `--tag` | suffix for OOF/plot files | empty |
| `--feat` | `v1` (Base), `v2` | `v1` |
| `--stacker` | `full`, `lite` | `full` |

v2 options:

| Flag | Values | Default |
|---|---|---|
| `--improved` | v2 model components (use with `--feat v2`) | off |
| `--prompt_mode` | `plain`, `verify` | `verify` |
| `--views` | any of `last`, `alast`, `afirst`, `amean` (comma separated) | `last,amean` |
| `--quant` | `4bit`, `8bit`, `fp16` | `4bit` |
| `--probe_use` | `both`, `seq`, `glob`, `none` (where probe logits go; `none` = ablation) | `both` |
| `--pca_dim` | PCA components per probe | 256 |
| `--probe_C` | inverse L2 strength of the probe | 0.02 |
| `--probe_jobs` | probes fitted in parallel | 4 |
| `--redo_probe` | recompute probes even if the file exists | off |
| `--group_split` | split by question (always on with v2) | off |
| `--pair_center` | subtract the per-question mean from every feature | off |

### Outputs

`results/features/` or `results/features_v2/` (LLM features), `results/probe/` (probe logits and a per-layer AUC table), `results/oof/` (out-of-fold outputs), `results/plots/results_*.png` (ROC, PR, calibration, confusion matrix, ...). AUC, F1 and accuracy are printed to the terminal; pair accuracy is printed too when the split is by question.

## Notes

- Speed settings: `batch_size=128`, `lr=4e-4`, `ema_decay=0.995`. The paper's values are 28, 2e-4 and 0.999 (see `src/config.py`).
- The seed changes both the train/test split and the inner folds (and, for v2, the probe folds).
- Bangla (`bn`) and Vietnamese (`vi`) require a local translated file `data/halueval_{lang}.json`.
- The Base numbers above use a row-level split, while v2 splits by question, so they are not directly comparable. Run Base with `--group_split` to compare on equal terms.
