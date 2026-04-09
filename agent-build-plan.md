# Agent Build Plan: Adversarial Training for Deepfake Detector Robustness

**Research Question:** Does adversarial training (LLM-generated attacks + iterative retraining)
improve a GAN-based deepfake text detector's robustness on medical text corpora?

**Final Goal:** A robust adversarial agent pipeline where:
- SeqGAN generates fake medical abstracts from scratch
- BioGPT (fine-tuned) acts as the adversarial agent — generates and rewrites fake text to evade detection
- BioBERT (fine-tuned) acts as the detector — iteratively retrained to resist adversarial attacks
- The loop runs for N rounds, measuring robustness improvement per round

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Adversarial Loop (N rounds)              │
│                                                              │
│  [MedFake / PubMed Real Abstracts]                           │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐  generates fake text   ┌────────────────┐  │
│  │   SeqGAN    │ ──────────────────────▶ │  Fake Text     │  │
│  │ (from       │                         │  Pool          │  │
│  │  scratch)   │                         └──────┬─────────┘  │
│  └─────────────┘                                │            │
│                                                 │            │
│  ┌─────────────┐  rewrites to evade  ┌──────────▼─────────┐ │
│  │  BioGPT     │ ◀─────────────────▶ │  BioBERT Detector  │ │
│  │ Adversarial │   confidence score  │  (fine-tuned       │ │
│  │ Agent       │   as reward signal  │   classifier)      │ │
│  └──────┬──────┘                     └──────────┬─────────┘ │
│         │ adversarial samples                   │           │
│         │ added to training set        retrain  │           │
│         └───────────────────────────────────────┘           │
└──────────────────────────────────────────────────────────────┘
```

---

## Project Folder Structure

```
deepfake_robustness/
├── data/
│   ├── raw/
│   │   ├── pubmed_abstracts/          # HuggingFace: slinusc/PubMedAbstractsSubset
│   │   ├── medfake/                   # Zenodo (request access) OR Med-MMHL fallback
│   │   └── med_mmhl/                  # GitHub fallback: Yanshen-Sun/Med-MMHL
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── models/
│   ├── seqgan/
│   │   ├── generator.py               # LSTM generator
│   │   ├── discriminator.py           # LSTM discriminator
│   │   ├── rollout.py                 # Monte Carlo rollout
│   │   └── train_seqgan.py
│   ├── detector/
│   │   ├── train_detector.py          # BioBERT fine-tuning
│   │   └── checkpoints/               # saved per round
│   └── biogpt_agent/
│       ├── adversarial_agent.py       # BioGPT attack + rewrite logic
│       └── checkpoints/
├── agents/
│   ├── adversarial_agent.py           # main agent class
│   └── prompts.py                     # prompt templates for BioGPT
├── training/
│   ├── adversarial_loop.py            # main N-round loop
│   └── utils.py
├── evaluation/
│   ├── metrics.py                     # AUC, F1, evasion_rate
│   └── eval_pipeline.py
├── configs/
│   └── config.yaml                    # all hyperparameters
├── experiments/                       # W&B / MLflow logs
├── requirements.txt
└── README.md
```

---

## Datasets

### Primary: Real Abstracts
| Dataset | Source | Access | How to Download |
|---------|--------|--------|-----------------|
| PubMed Abstracts Subset | HuggingFace | Instant | `load_dataset("slinusc/PubMedAbstractsSubset")` |
| PubMed Full | HuggingFace | Instant (stream) | `load_dataset("pubmed", streaming=True)` |

### Primary: Fake / Misinformation Text
| Dataset | Source | Access | Link |
|---------|--------|--------|------|
| Monant MedFake | Zenodo | Request via institutional email | https://github.com/kinit-sk/medical-misinformation-dataset |
| Med-MMHL (fallback) | GitHub | Public | https://github.com/Yanshen-Sun/Med-MMHL |

> **Recommendation:** Submit Zenodo access request for MedFake immediately (takes 1-3 days).
> Use Med-MMHL + PubMed in the meantime to build the pipeline.

### Download Scripts

**PubMed (run this first — instant):**
```python
# data/download_pubmed.py
from datasets import load_dataset
import pandas as pd

ds = load_dataset("slinusc/PubMedAbstractsSubset")
df = pd.DataFrame(ds["train"])
df = df[["title", "abstract"]].dropna()
df["label"] = 0  # 0 = real
df.to_csv("data/raw/pubmed_abstracts/pubmed_real.csv", index=False)
print(f"Saved {len(df)} real abstracts")
```

**Med-MMHL (public fallback):**
```bash
cd data/raw
git clone https://github.com/Yanshen-Sun/Med-MMHL.git med_mmhl
```

> **Storage-constrained development note:** the codebase can be scaffolded and validated
> without downloading datasets locally. Data download and training runs can be deferred to
> the target machine where storage is available.

---

## Environment Setup

### Requirements
```
# requirements.txt
torch>=2.0.0
transformers>=4.38.0
datasets>=2.18.0
accelerate>=0.27.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
wandb>=0.16.0
sentencepiece>=0.1.99
protobuf>=3.20.0
pyyaml>=6.0.1
peft>=0.11.0
evaluate>=0.4.1
bert-score>=0.3.13
safetensors>=0.4.2
# Optional for QLoRA / 4-bit loading on supported Linux setups:
# bitsandbytes>=0.43.1
```

### Setup Commands
```bash
# 1. Create conda env
conda create -n deepfake_rob python=3.10 -y
conda activate deepfake_rob

# 2. Set HuggingFace cache to your preferred storage path
export HF_HOME=/path/to/your/storage/hf_cache
export TRANSFORMERS_CACHE=/path/to/your/storage/hf_cache
export TMPDIR=/path/to/your/storage/tmp
export PIP_CACHE_DIR=/path/to/your/storage/pip_cache
mkdir -p $HF_HOME $TMPDIR $PIP_CACHE_DIR

# Add to ~/.bashrc to persist across sessions
echo "export HF_HOME=$HF_HOME" >> ~/.bashrc
echo "export TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE" >> ~/.bashrc
echo "export TMPDIR=$TMPDIR" >> ~/.bashrc
echo "export PIP_CACHE_DIR=$PIP_CACHE_DIR" >> ~/.bashrc

# 3. Install PyTorch with CUDA 11.8 (for GPU training)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install remaining packages
pip install -r requirements.txt

# 4b. Optional: install bitsandbytes if you want QLoRA / 4-bit loading
# pip install bitsandbytes

# 5. Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

> **Disk requirement:** ~20GB free minimum.
> BioGPT: ~1.4GB, BioBERT: ~400MB, PyTorch CUDA wheel: ~2GB, datasets: ~5GB.

---

## Component Build Plan

---

### COMPONENT 1: Data Pipeline
**File:** `data/prepare_data.py`

**What it does:**
- Loads PubMed real abstracts (label=0) and MedFake/Med-MMHL fake text (label=1)
- Cleans: removes empty rows, strips HTML, normalizes whitespace, truncates to 512 tokens
- Splits into train/val/test (70/15/15, stratified)
- Saves as CSV

**Key cleaning steps:**
1. Remove abstracts shorter than 50 words (too short to train on)
2. Remove duplicates by text hash
3. Normalize unicode characters
4. Truncate at 512 tokens (BioBERT max)

```python
# configs/config.yaml
data:
  pubmed_path: "data/raw/pubmed_abstracts/pubmed_real.csv"
  fake_path: "data/raw/med_mmhl/"
  processed_path: "data/processed/"
  max_length: 512
  train_split: 0.70
  val_split: 0.15
  test_split: 0.15
  min_words: 50
  seed: 42
```

---

### COMPONENT 2: SeqGAN (Fake Text Generator — from scratch)
**Files:** `models/seqgan/generator.py`, `discriminator.py`, `rollout.py`, `train_seqgan.py`

**Architecture:**
- Generator: 2-layer LSTM, hidden_dim=512, embedding_dim=64, vocab from training corpus
- Discriminator: 2-layer LSTM binary classifier
- Training: Pre-train generator with MLE on real abstracts → adversarial training with policy gradient (REINFORCE) + Monte Carlo rollout for reward estimation

**Training stages:**
1. Pre-train generator: standard cross-entropy on real PubMed abstracts (~100 epochs)
2. Pre-train discriminator: on real vs pre-train-generated fake samples (~50 epochs)
3. Adversarial training loop: alternate G and D updates (~200 adversarial epochs)

```python
# configs/config.yaml (seqgan section)
seqgan:
  vocab_size: 10000          # built from training corpus
  embed_dim: 64
  hidden_dim: 512
  num_layers: 2
  seq_len: 200               # token length per generated abstract
  pretrain_epochs_g: 100
  pretrain_epochs_d: 50
  adversarial_epochs: 200
  rollout_num: 16            # MC rollout samples
  batch_size: 64
  lr_g: 0.0001
  lr_d: 0.0001
  generated_pool_size: 10000 # samples to generate per round
```

---

### COMPONENT 3: BioBERT Detector (Fine-tuned Classifier)
**File:** `models/detector/train_detector.py`

**Model:** `dmis-lab/biobert-base-cased-v1.2` from HuggingFace
Fine-tuned as binary sequence classifier (real=0, fake=1)

**Training:**
- Input: tokenized abstracts (max 512 tokens)
- Output: binary label + confidence score (softmax probability)
- Loss: cross-entropy
- Optimizer: AdamW, lr=2e-5, weight_decay=0.01
- Scheduler: linear warmup (10% steps) + linear decay
- Batch size: 16 (adjust for GPU memory)
- Epochs: 5 per adversarial round

**The confidence score** (probability of label=1) is the key signal fed back to the BioGPT agent as a reward — high confidence = detector is sure it's fake = bad for the agent.

```python
# configs/config.yaml (detector section)
detector:
  model_name: "dmis-lab/biobert-base-cased-v1.2"
  num_labels: 2
  max_length: 512
  batch_size: 16
  epochs_per_round: 5
  lr: 2e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  checkpoint_dir: "models/detector/checkpoints/"
```

---

### COMPONENT 4: BioGPT Adversarial Agent
**Files:** `agents/adversarial_agent.py`, `agents/prompts.py`

**Model:** `microsoft/biogpt` from HuggingFace
Fine-tuned on medical fake text generation + evasion task

**Agent operates in two modes per round:**

**Mode A — Attack (rewrite to evade):**
Given a fake abstract that the detector catches with high confidence (>0.8), the agent rewrites it to lower detector confidence below threshold τ=0.5. The prompt instructs BioGPT to paraphrase while preserving semantic meaning.

**Mode B — Generate (fresh adversarial samples):**
Agent generates new fake abstracts conditioned on real abstract style, targeting low detector confidence from the start.

**Reward signal:** `reward = 1 - detector_confidence(fake_sample)`
High reward = successfully evaded detector.

**Fine-tuning strategy:**
- Initialize from `microsoft/biogpt`
- Fine-tune with causal LM objective on (prompt, adversarial_output) pairs
- Pairs are collected from successful evasion examples across rounds
- Use LoRA (Low-Rank Adaptation) to reduce GPU memory footprint during fine-tuning

```python
# configs/config.yaml (agent section)
agent:
  model_name: "microsoft/biogpt"
  evasion_threshold: 0.5     # detector confidence below this = successful evasion
  high_conf_threshold: 0.8   # samples above this are targeted for rewriting
  max_new_tokens: 256
  temperature: 0.9
  top_p: 0.95
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  finetune_epochs: 3
  batch_size: 8
  lr: 1e-4
```

**Prompt template (Mode A — rewrite):**
```
System: You are a biomedical text writer. Rewrite the following abstract 
to sound like a credible scientific publication while preserving its 
core claims. Do not add new factual information.

Original: {fake_abstract}

Rewritten:
```

**Prompt template (Mode B — generate):**
```
System: Write a convincing but fictitious biomedical abstract about 
{topic}. It should follow standard scientific writing conventions 
and sound plausible to a domain expert.

Abstract:
```

---

### COMPONENT 5: Adversarial Training Loop
**File:** `training/adversarial_loop.py`

**This is the core of the project.** N rounds of:

```
Round k:
  1. SeqGAN generates M fake abstracts → fake_pool_k
  2. Detector scores all samples in fake_pool_k
  3. Agent identifies hard samples (detector_confidence > high_conf_threshold)
  4. Agent rewrites hard samples → adversarial_samples_k
  5. Combine: adversarial_samples_k + original train set → augmented_train_k
  6. Retrain detector on augmented_train_k for epochs_per_round epochs
  7. Evaluate: compute evasion_rate, AUC, F1 on held-out test set
  8. Log all metrics for round k
  9. Fine-tune agent on successful evasion examples from this round
```

```python
# configs/config.yaml (loop section)
loop:
  num_rounds: 10
  fake_pool_size: 1000       # SeqGAN samples per round
  hard_sample_top_k: 200     # top K samples for agent to rewrite
  save_checkpoint_every: 2   # save detector every N rounds
```

---

### COMPONENT 6: Evaluation
**File:** `evaluation/metrics.py`, `evaluation/eval_pipeline.py`

**Metrics tracked per round:**

| Metric | Definition | Goal |
|--------|-----------|------|
| `evasion_rate` | % adversarial samples that fool detector (conf < 0.5) | Should decrease over rounds |
| `AUC` | Detector AUROC on test set | Should increase over rounds |
| `F1` | Detector F1 on test set | Should increase over rounds |
| `robustness_delta` | AUC[round_k] - AUC[round_0] | Key result metric |
| `rewrite_quality` | BERTScore between original and rewritten fake | Tracks semantic preservation |

**Ablation experiments (run after main loop):**
1. Baseline detector (no adversarial training) — round 0 only
2. SeqGAN only (no BioGPT agent) — standard GAN training
3. BioGPT agent only (no SeqGAN) — LLM rewrites only
4. Full pipeline (SeqGAN + BioGPT + adversarial loop) — main result

---

## Build Order (Sequential)

```
Phase 1 — Data [~1 day]
  □ data/download_pubmed.py          → download + save PubMed real abstracts
  □ data/download_medfake.py         → download Med-MMHL fake texts
  □ data/prepare_data.py             → clean, merge, split → train/val/test CSVs

Phase 2 — Baseline Detector [~1 day]
  □ models/detector/train_detector.py → fine-tune BioBERT on train split
  □ evaluation/eval_pipeline.py       → evaluate baseline AUC/F1 (round 0)
  □ Save baseline checkpoint

Phase 3 — SeqGAN [~2-3 days]
  □ models/seqgan/generator.py        → LSTM generator
  □ models/seqgan/discriminator.py    → LSTM discriminator
  □ models/seqgan/rollout.py          → Monte Carlo rollout
  □ models/seqgan/train_seqgan.py     → full pretrain + adversarial train
  □ Evaluate: BLEU, perplexity of generated text

Phase 4 — BioGPT Adversarial Agent [~2 days]
  □ agents/prompts.py                 → prompt templates
  □ agents/adversarial_agent.py       → rewrite + generate logic
  □ Test: manually verify agent can lower detector confidence on examples

Phase 5 — Adversarial Loop [~2 days]
  □ training/adversarial_loop.py      → N-round loop
  □ Run 10 rounds, log all metrics

Phase 6 — Evaluation + Ablations [~1 day]
  □ Run all 4 ablation conditions
  □ Generate plots: evasion_rate vs round, AUC vs round
  □ evaluation/eval_pipeline.py → final results table
```

**Total estimate: ~10 days**

---

## Key Files to Build First

When starting, build in this exact order:

1. `configs/config.yaml` — central config, everything reads from here
2. `data/prepare_data.py` — nothing works without clean data
3. `models/detector/train_detector.py` — establishes baseline
4. `models/seqgan/generator.py` + `discriminator.py` + `rollout.py`
5. `models/seqgan/train_seqgan.py`
6. `agents/adversarial_agent.py`
7. `training/adversarial_loop.py`
8. `evaluation/eval_pipeline.py`

---

## Hardware Requirements

**Target Hardware: RTX 3060 (12GB VRAM) + 16GB RAM**

### VRAM Usage Per Component

| Component | Default VRAM | RTX 3060 Status | Adjustment |
|-----------|-------------|-----------------|------------|
| SeqGAN training | 4GB | ✅ Fits fine | None needed |
| BioBERT fine-tuning | 8GB | ✅ Fits | fp16 + gradient accumulation |
| BioGPT inference | 6GB | ✅ Fits | None needed |
| BioGPT fine-tuning (LoRA) | 12GB | ⚠️ Tight | LoRA/QLoRA + gradient checkpointing |
| Full loop (all at once) | 16GB | ❌ Too much | Must load/unload per step (see below) |

### Disk Space

| What | Size |
|------|------|
| PyTorch CUDA wheel | ~2GB |
| BioBERT weights | ~400MB |
| BioGPT weights | ~1.4GB |
| PubMed abstracts subset | ~500MB |
| Med-MMHL dataset | ~200MB |
| SeqGAN checkpoints (10 rounds) | ~500MB |
| BioBERT checkpoints (10 rounds) | ~4GB |
| BioGPT LoRA checkpoints | ~200MB |
| HF cache overhead | ~3GB |
| **Total** | **~12GB minimum, 20GB comfortable** |

### RAM
16GB RAM is sufficient for all data preprocessing and training steps.

---

### RTX 3060 Config Overrides

These override the default hyperparameters defined in each component section above.
They are chosen to preserve detector robustness as much as possible on a 12GB GPU:

```yaml
# configs/config.yaml — RTX 3060 12GB overrides

detector:
  batch_size: 4              # lower micro-batch for 12GB VRAM
  grad_accum_steps: 4        # keeps effective batch reasonable
  max_length: 384            # preferred compromise; use 256 only if you still OOM
  fp16: true
  gradient_checkpointing: true

agent:
  batch_size: 2              # lower micro-batch for LoRA fine-tuning
  grad_accum_steps: 4
  lora_r: 8                  # smaller LoRA rank = less VRAM
  max_new_tokens: 192        # preserve rewrite quality better than 128
  gradient_checkpointing: true
  use_4bit: true             # enable QLoRA when bitsandbytes is available

seqgan:
  batch_size: 32             # reduce batch first
  hidden_dim: 512            # keep default capacity; file already notes it fits on 3060
```

> Prefer reducing micro-batch size and using accumulation/checkpointing before shrinking
> sequence length or model capacity. Only drop the detector to `max_length: 256` if the
> `384`-token setting still exceeds VRAM on the target system.

### Critical: Load/Unload Pattern for the Adversarial Loop

12GB VRAM cannot hold BioBERT + BioGPT simultaneously.
The adversarial loop **must** load, run, and unload each model per step:

```python
# training/adversarial_loop.py — RTX 3060 pattern

import torch

# Step 1: score with detector
detector = load_detector().cuda()
scores = detector.score(fake_pool)
del detector
torch.cuda.empty_cache()

# Step 2: agent rewrites hard samples
agent = load_biogpt_agent().cuda()
adversarial_samples = agent.rewrite(hard_samples)
del agent
torch.cuda.empty_cache()

# Step 3: retrain detector on augmented data
detector = load_detector().cuda()
detector.train(augmented_data)
detector.save_checkpoint()
del detector
torch.cuda.empty_cache()
```

> This adds ~30 seconds overhead per round but keeps VRAM safely under 12GB throughout all 10 rounds.

---

## Models Summary

| Model | Role | Type | HuggingFace ID |
|-------|------|------|----------------|
| SeqGAN | Fake text generator | From scratch (PyTorch LSTM) | N/A |
| BioBERT | Detector / classifier | Fine-tuned | `dmis-lab/biobert-base-cased-v1.2` |
| BioGPT | Adversarial agent | Fine-tuned (LoRA) | `microsoft/biogpt` |

---

## Expected Results

The pipeline should demonstrate:
- Round 0: Baseline detector AUC ~0.85–0.90 on clean fake vs real
- After 10 adversarial rounds: AUC improves to ~0.92–0.95 (robustness_delta > 0)
- Evasion rate drops from ~40% (round 1) to <10% (round 10)
- Ablation: full pipeline outperforms SeqGAN-only and BioGPT-only conditions

---

## Next Steps After Build

1. Submit Zenodo request for Monant MedFake dataset (institutional email required)
2. Replace Med-MMHL with MedFake once access is granted — rerun pipeline
3. Write up results as research paper (IEEE/ACL format)
4. Consider extending: multi-round curriculum attack, larger BioGPT variant
