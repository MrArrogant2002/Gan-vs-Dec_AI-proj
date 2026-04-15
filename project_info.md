# Project Information

## Project Name

**Deepfake Robustness Pipeline**  
This repository studies adversarial robustness for **medical-text deepfake detection**.

## Project Goal

The main research goal is to test whether an adversarial training loop can improve the robustness of a medical-text detector against machine-generated fake content.

The project combines:

- a **generator** that produces fake medical-style text
- a **detector** that classifies text as real or fake
- an **adversarial agent** that rewrites fake text to evade the detector
- an iterative **round-based retraining loop**

## Important Project Variants

This repository now has **two closely related versions** of the pipeline:

### 1. Original modular codebase design

The main Python project in this repo is built around:

- **SeqGAN** as the fake-text generator
- **BioBERT** as the detector
- **BioGPT** as the adversarial rewrite agent
- a modular training pipeline in `training/adversarial_loop.py`

This design is reflected in:

- `configs/config.yaml`
- `models/seqgan/`
- `models/detector/`
- `agents/`
- `training/`
- `evaluation/`

### 2. Colab-trained standalone notebook variant

To make the project runnable on Colab with lighter operational requirements, `agent-run.ipynb` implements a standalone end-to-end notebook that uses:

- **GPT-2 Small** instead of SeqGAN for the generator role
- **BioBERT** as the detector
- **BioGPT + LoRA** as the adversarial agent
- Google Drive for checkpoints and plots
- Hugging Face Hub export/import cells

The Hugging Face models listed below were uploaded from this **Colab notebook variant**, not from the original full SeqGAN pipeline.

## Current Hugging Face Model Repositories

The trained artifacts pushed to Hugging Face are:

| Component | Hugging Face Repository |
| --- | --- |
| GPT-2 generator | https://huggingface.co/Mr-Arr0gant/gan-vs-det-ai-gpt2-generator |
| BioBERT detector | https://huggingface.co/Mr-Arr0gant/gan-vs-det-ai-biobert-detector |
| BioGPT agent | https://huggingface.co/Mr-Arr0gant/gan-vs-det-ai-biogpt-agent |

## Core Research Question

Can iterative adversarial training improve a detector's robustness on medical-text deepfakes by exposing it to progressively harder generated and rewritten attacks?

## Dataset Sources

### Real data

- **PubMed Abstracts Subset**
- Source: Hugging Face dataset `slinusc/PubMedAbstractsSubset`
- In the Colab notebook variant, the dataset is streamed and capped at **2000 usable rows** by default
- Label used for real samples: **0**

### Fake data

- **Med-MMHL / uploaded fake-news folders**
- In the Colab notebook flow, the notebook first checks for uploaded folders such as:
  - `fakenews_article`
  - `sentence`
- If those are not present, it falls back to the public Med-MMHL repository
- Label used for fake samples: **1**

## Data Preparation

The project includes a reusable preprocessing pipeline in `data/prepare_data.py`.

Implemented preprocessing steps include:

- HTML unescaping
- HTML tag stripping
- Unicode normalization
- whitespace cleanup
- empty-row filtering
- minimum word-count filtering
- duplicate removal using SHA-256 hashing
- token-aware truncation when a tokenizer is available
- stratified train/validation/test split creation

The default split ratios used across the project are:

- train: **70%**
- validation: **15%**
- test: **15%**

## Model Roles

### Generator

There are two generator paths in the repo:

- **Original repo design:** SeqGAN
- **Current Colab-trained artifact path:** GPT-2 Small fine-tuned for fake medical-style text generation

### Detector

- **Model family:** BioBERT
- **Base model:** `dmis-lab/biobert-base-cased-v1.2`
- **Task:** binary classification of medical text as real or fake

### Adversarial agent

- **Model family:** BioGPT
- **Base model:** `microsoft/biogpt`
- **Fine-tuning method:** LoRA adapters via `peft`
- **Task:** rewrite high-confidence fake samples so they are harder for the detector to catch

## Original Repository Structure

### Configuration

- `configs/config.yaml`

### Data

- `data/download_pubmed.py`
- `data/download_medfake.py`
- `data/prepare_data.py`

### Detector

- `models/detector/train_detector.py`

### SeqGAN

- `models/seqgan/generator.py`
- `models/seqgan/discriminator.py`
- `models/seqgan/rollout.py`
- `models/seqgan/train_seqgan.py`

### Adversarial agent

- `agents/prompts.py`
- `agents/adversarial_agent.py`

### Training loop and utilities

- `training/adversarial_loop.py`
- `training/utils.py`
- `training/experiment_logger.py`

### Evaluation

- `evaluation/metrics.py`
- `evaluation/visualization.py`
- `evaluation/eval_pipeline.py`

## Colab Notebook Pipeline Used For Current Uploaded Models

The current exported models are aligned with `agent-run.ipynb`, whose default settings are:

### Data settings

- PubMed real data path: `data/raw/pubmed_real.csv`
- Fake data path: `data/raw/med_mmhl`
- max real samples: **2000**
- max sequence length: **256**
- minimum word count: **50**

### Generator settings

- model: **gpt2**
- batch size: **8**
- fine-tuning epochs: **3**
- learning rate: **2e-5**
- max new tokens: **150**

### Detector settings

- model: **dmis-lab/biobert-base-cased-v1.2**
- batch size: **8**
- max length: **256**
- epochs per round: **3**
- learning rate: **2e-5**

### Agent settings

- model: **microsoft/biogpt**
- LoRA rank: **8**
- LoRA alpha: **16**
- LoRA dropout: **0.05**
- batch size: **4**
- fine-tuning epochs: **2**
- learning rate: **1e-4**
- evasion threshold: **0.5**
- high-confidence threshold: **0.8**

### Adversarial loop settings

- number of rounds: **5**
- fake pool size per round: **200**
- top hard samples selected per round: **50**

## Adversarial Training Workflow

The implemented workflow is:

1. Prepare real and fake text data
2. Fine-tune the baseline detector
3. Fine-tune the generator
4. Generate a fake text pool
5. Score generated texts with the detector
6. Select hard samples with high fake confidence
7. Rewrite hard samples with the BioGPT agent
8. Re-score rewritten texts
9. Add successful evasions back into training
10. Retrain the detector
11. Repeat for multiple rounds

## Evaluation And Metrics

The project supports both training-time and post-run evaluation.

Implemented metrics/utilities in the modular codebase include:

- accuracy
- precision
- recall
- specificity
- F1
- AUC
- PR-AUC
- balanced accuracy
- MCC
- Brier score
- log loss
- TP / TN / FP / FN
- evasion rate
- robustness delta
- confidence-shift tracking
- rewrite quality scoring
- confusion matrix generation
- ROC curve generation
- precision-recall plotting
- confidence histograms

## Experiment Tracking

The modular codebase includes experiment-logging hooks in `training/experiment_logger.py`.

Supported tracking features include:

- optional Weights & Biases initialization
- no-op fallback if W&B is unavailable
- metric logging
- table logging
- plot/image logging
- distinct run types for detector, SeqGAN, agent, adversarial loop, and evaluation

## Result Summary From The Provided Plots

The following summary is based on the two metric plots you provided:

- `auc_f1_vs_round.png`
- `evasion_rate_vs_round.png`

### AUC and F1 trend

- **AUC** stays essentially flat at about **1.00** from round 0 through round 5
- **F1** stays around **0.985** from round 0 through round 3
- **F1** drops to about **0.9705** at rounds 4 and 5

### Evasion-rate trend

- round 1: about **0.02**
- round 2: about **0.02**
- round 3: **0.00**
- round 4: **0.00**
- round 5: about **0.02**

### Practical interpretation

Based on these plots alone:

- the detector remained extremely strong in terms of ranking quality, since AUC stayed near perfect
- the adversarial agent achieved only a **low evasion rate overall**
- rounds 3 and 4 appear to be the strongest from an evasion-resistance perspective
- the late-round F1 drop suggests some threshold-level classification behavior changed even though overall separability remained very high

This suggests that the trained detector remained highly robust on the evaluated split, while the rewrite agent had only limited success in consistently fooling it.

## Current Project Status

### Implemented

- modular repo structure
- configuration system
- data pipeline
- BioBERT detector code
- SeqGAN code
- BioGPT adversarial-agent code
- adversarial loop
- evaluation utilities
- Colab standalone notebook
- Hugging Face Hub upload and reload cells

### Already completed in the Colab workflow

- standalone Colab notebook implementation
- dataset acquisition helpers
- metrics plotting
- Hugging Face upload cells
- Hugging Face reload/test cells
- model export to the three Hugging Face repos listed above

## Reproduction Paths

### Local modular-code path

1. Create a Python environment
2. Install `requirements.txt`
3. Review `configs/config.yaml`
4. Prepare datasets
5. Train the baseline detector
6. Train SeqGAN
7. Run `training/adversarial_loop.py`
8. Evaluate with the tools in `evaluation/`

### Colab notebook path

1. Open `agent-run.ipynb`
2. Upload the fake-data folders if needed
3. Stream PubMed subset data
4. Run generator, detector, and BioGPT sections
5. Run the adversarial loop
6. Save plots and artifacts to Drive
7. Upload trained components to Hugging Face Hub
8. Reload them in the notebook for sanity-check inference

## Important Notes

- The **original repo design** and the **current uploaded notebook artifacts** are related but not identical
- The original codebase uses **SeqGAN**, but the current uploaded Hugging Face generator is **GPT-2 Small**
- The current public model repos therefore reflect the **Colab-optimized implementation path**
- Result interpretation here is based on the two uploaded metric plots and the repository configuration/code, not on a full external benchmark report

## Related Files In This Repository

- `README.md`
- `implementation-complete.md`
- `agent-build-plan.md`
- `colab-build-plan.md`
- `agent-run.ipynb`
- `configs/config.yaml`

