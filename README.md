# Hidden-State Probes, Answer Likelihood, and Surface Shortcuts in White-Box Hallucination Detection

White-box hallucination detection for large language models (LLMs) using internal representations, evaluated on English HaluEval QA and a curated TriviaQA hard-negative dataset.

![Pipeline](./figures/pipeline.png)

## Overview

The project consists of four main stages:

1. **Feature Extraction** (frozen LLM, 4-bit NF4 quantization by default).

   Two feature configurations are available:

   - **Base** (default): Extracts 12 statistical features from each of 32 sampled hidden-state layers, along with 30 global features derived from the output logits.
   - **v2** (`--v2`): Retains the full hidden-state vectors using two views (`last`, `amean`) and introduces a verification prompt: *"Is the proposed answer correct? Answer (Yes or No):"*. Global features increase to 46 (30 base features + 13 answer log-likelihood / PMI features + 3 Yes/No log-probability features).

2. **Layer Probes** (v2 only).

   For each (view, layer) pair, an out-of-fold linear probe is trained using the following pipeline:

   `StandardScaler -> PCA(256) -> LogisticRegression(C=0.02)`

   Folds are grouped by question. Probe logits are automatically appended to the network input after feature extraction when `--v2` is enabled.

3. **MultiHaluDet** (trained independently within each inner fold).

   The architecture consists of:
   - A sequential branch with multi-scale attention, a Transformer, and attention pooling.
   - A global branch implemented as an MLP.
   - A gated fusion mechanism that combines both branches.

   The training objective combines BCE, Focal, Asymmetric, and Contrastive losses. Optimization uses AdamW with Exponential Moving Average (EMA), Mixup, and CutMix.

   In v2, delta channels, layer-weight scaling, and pairwise AUC ranking loss are also available as optional components. All three are disabled by default.

4. **Stacking**.

   Out-of-fold predictions from the five fold-specific networks are passed to the base classifiers (RF, HistGB, LR, SVM, XGB, and LGBM). A logistic regression meta-learner then combines their log-odds. Final predictions are obtained by averaging across the five test sets.

### What v2 Adds Over Base

| Component | Base | v2 |
|---|---|---|
| LLM features | 12 scalar statistics per sampled layer | Same scalar statistics, with additional hidden-state vectors used to train layer probes. Their per-layer logits are fed into the detector. |
| Prompt | `Question: ... Answer: ...` | Verification prompt by default, with answer-token likelihood/PMI features and verification-only Yes/No log-probabilities. Plain prompting is also supported. |
| Global features | 30 | 43 with the plain prompt or 46 with the verification prompt, plus 2 values per probe view when global probes are enabled (50 by default with verification prompting and two views). |
| Pipeline stages | `extract`, `train_oof`, `ensemble` | `extract` (including probe training), `train_oof`, `ensemble` |
| Network | 12 sequential channels per layer | Probe-logit channels enabled by default. Delta channels and layer-weight scaling are optional. Pairwise AUC ranking loss is also optional. |
| Train/test split | By row | **By question** (both answers belonging to the same question remain in the same split). |
| Stacker | 5-fold splitting by row; 576-dimensional vector features and 6 classifiers by default | 5-fold splitting by question; 576-dimensional vector features and 6 classifiers by default. Supports optional OOF-logit input (`--stack_mode logit`) and a lightweight 2-classifier stacker (`lite`). |
| Additional metric | - | **Pair accuracy** (whether the incorrect answer receives a higher score than the correct answer). |
| Optional configuration | - | `--pair_center` (subtracts the per-question mean). |

### Results Summary

The following table reports the AUROC achieved by Base and v2 across the two evaluation datasets.

| Dataset | Model | Base AUROC | v2 AUROC | Δ AUROC |
|---|---|---:|---:|---:|
| HaluEval | Mistral-7B | 0.9849 | **0.9931** | +0.0082 |
| HaluEval | Llama-2-7B | 0.9846 | **0.9917** | +0.0071 |
| TriviaQA | Mistral-7B | 0.6452 | **0.8226** | +0.1774 |
| TriviaQA | Llama-2-7B | 0.5931 | **0.7650** | +0.1719 |

## Installation

Clone the repository:

```bash
git clone https://github.com/dktllnk20/MultiHaluDet-plus.git
cd MultiHaluDet-plus
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Hugging Face Authentication

Gated models (e.g., Llama-2) require a Hugging Face access token.

Create a `.env` file in the repository root and add your token:

```bash
echo "HF_TOKEN=hf_xxxxxxxx" > .env
```

The token is loaded automatically by `src/config.py`.

### Dataset Information

- **HaluEval:** Uses the English dataset located at `data/halueval.json`. If the file is unavailable, the pipeline automatically falls back to Hugging Face.
- **TriviaQA:** Expects `data/triviaqa.json`, containing exactly two records per question: one correct answer (label `0`) and one hard negative (label `1`). If the file is unavailable, the pipeline falls back to a version of TriviaQA with randomly generated incorrect answers.

> [!NOTE]
> The TriviaQA hard-negative dataset was derived from `lucadiliello/triviaqa` on Hugging Face and subsequently edited, manually verified, and relabeled. See [Dataset Information](DATASET.md) for further details and access instructions.

### Hardware

All experiments were conducted on a system equipped with:

- **CPU:** AMD Ryzen 7 7435HS
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)

## Usage

### Running the Pipeline

Run the complete pipeline:

```bash
python run_pipeline.py --models mistral-7b llama2-7b --dataset triviaqa halueval --stage all --seed 42
```

Alternatively, execute each stage independently:

```bash
# Stage 1: Feature extraction
python run_pipeline.py --models mistral-7b --dataset halueval --stage extract

# Stage 2: Out-of-fold training
python run_pipeline.py --models mistral-7b --dataset halueval --stage train_oof

# Stage 3: Ensemble evaluation
python run_pipeline.py --models mistral-7b --dataset halueval --stage ensemble
```

### Research Questions

Run the experiments associated with each research question:

```bash
python run_pipeline.py --rq 1
python run_pipeline.py --rq 2
```

### Supported Models

- Mistral-7B (`mistral-7b`)
- Llama-2-7B (`llama2-7b`)

## Folder Structure

```text
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

## Contact

Have questions about the project, want to reproduce the experiments, or are interested in learning more? Feel free to reach out!

- **Email:** [dktlln.k20@gmail.com](mailto:dktlln.k20@gmail.com)

Whether you have a question, feedback, or a collaboration idea, we'd be happy to hear from you.
