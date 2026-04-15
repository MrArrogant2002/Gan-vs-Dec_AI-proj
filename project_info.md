# Project Information

## Project Name

**Deepfake Robustness Pipeline**  
This repository studies adversarial robustness for **medical-text deepfake detection**.

## Source Of Truth For The Reported Results

The **reported results in this document are from [agent-run.ipynb](agent-run.ipynb)**.

That notebook is the authoritative pipeline for:

- the training run summarized in this document
- the saved metric plots
- the Google Drive artifacts
- the Hugging Face model uploads

Within the notebook:

- the main experiment is run end-to-end in the Colab workflow cells
- the final plots are produced in **Cell 14 — Results And Plots**
- the Hugging Face model export is handled in **Cell 17 — Hugging Face Hub Export**
- the Hub reload/sanity-check inference is handled in **Cell 18 — Load HF Models And Test Outputs**

## Project Goal

The main research goal is to test whether an adversarial training loop can improve the robustness of a medical-text detector against machine-generated fake content.

The project combines:

- a **generator** that produces fake medical-style text
- a **detector** that classifies text as real or fake
- an **adversarial agent** that rewrites fake text to evade the detector
- an iterative **round-based retraining loop**

## Primary Experimental Pipeline

The primary experimental pipeline for this project is the standalone Colab notebook [agent-run.ipynb](agent-run.ipynb).

That notebook uses:

- **GPT-2 Small** instead of SeqGAN for the generator role
- **BioBERT** as the detector
- **BioGPT + LoRA** as the adversarial agent
- Google Drive for checkpoints and plots
- Hugging Face Hub export/import cells

The Hugging Face models listed below were uploaded from this notebook pipeline.

## Repository Background

The repository also contains the original modular codebase design, which is useful as a reference and for future extension.

That background codebase is built around:

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

The Colab notebook contains its own inline data-preparation flow, and the repository also includes a reusable preprocessing pipeline in `data/prepare_data.py`.

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

## Model Roles In agent-run.ipynb

### Generator

- **Notebook run used for results:** GPT-2 Small fine-tuned for fake medical-style text generation

### Detector

- **Model family:** BioBERT
- **Base model:** `dmis-lab/biobert-base-cased-v1.2`
- **Task:** binary classification of medical text as real or fake

### Adversarial agent

- **Model family:** BioGPT
- **Base model:** `microsoft/biogpt`
- **Fine-tuning method:** LoRA adapters via `peft`
- **Task:** rewrite high-confidence fake samples so they are harder for the detector to catch

## Supporting Repository Code

The repository still includes the broader modular implementation for future non-notebook workflows.

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

## Notebook Outputs And Saved Artifacts

The `agent-run.ipynb` workflow saves and uses:

- Drive-backed checkpoints for generator, detector, and agent artifacts
- `metrics_log.csv` for round-level metrics
- `plots/evasion_rate_vs_round.png`
- `plots/auc_f1_vs_round.png`
- `round_artifacts/round_*/` outputs such as rewrites, predictions, and per-round summaries
- Hugging Face Hub uploads for the final generator, detector, and agent repositories

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

The reported experiment from `agent-run.ipynb` uses round-level evaluation and post-run plot generation.

Metrics and utilities implemented across the repository include:

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

For the reported notebook run, the primary tracking outputs are:

- Drive-saved checkpoints
- `metrics_log.csv`
- saved PNG plots
- per-round JSON and CSV artifacts
- Hugging Face Hub model repos

The repository also contains optional Weights & Biases hooks in `training/experiment_logger.py` for the modular Python workflow.

## Result Summary From agent-run.ipynb

The following summary is based on the two metric plots produced by `agent-run.ipynb` and shared here:

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

This suggests that, in the `agent-run.ipynb` experiment, the trained detector remained highly robust on the evaluated split, while the rewrite agent had only limited success in consistently fooling it.

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

### Primary notebook path

1. Open `agent-run.ipynb`
2. Upload the fake-data folders if needed
3. Stream PubMed subset data
4. Run generator, detector, and BioGPT sections
5. Run the adversarial loop
6. Save plots and artifacts to Drive
7. Upload trained components to Hugging Face Hub
8. Reload them in the notebook for sanity-check inference

### Secondary modular-code path

1. Create a Python environment
2. Install `requirements.txt`
3. Review `configs/config.yaml`
4. Prepare datasets
5. Train the baseline detector
6. Train SeqGAN
7. Run `training/adversarial_loop.py`
8. Evaluate with the tools in `evaluation/`

## Important Notes

- `agent-run.ipynb` is the source of truth for the results described in this file
- the current public Hugging Face repos reflect the **notebook pipeline**, not the original SeqGAN training path
- the original modular codebase remains important as project scaffolding and a broader research implementation
- result interpretation here is based on the notebook outputs and the two uploaded metric plots, not on a separate external benchmark report

## Related Files In This Repository

- `README.md`
- `implementation-complete.md`
- `agent-build-plan.md`
- `colab-build-plan.md`
- `agent-run.ipynb`
- `configs/config.yaml`
