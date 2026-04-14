# Adversarial Training Pipeline — Colab Free Tier Build Plan
> **Target:** Google Colab Free (T4 15GB VRAM, ~12GB RAM, ~75GB ephemeral disk)
> **Goal:** Full proof-of-concept pipeline at reduced scale
> **Generator:** GPT-2 Small (replaces SeqGAN — faster, lighter, same role)

---

## Constraints Summary

| Resource | Limit | Strategy |
|---|---|---|
| VRAM | 15GB | Load/unload each model per step |
| RAM | ~12GB | Stream + cap datasets at 2000 samples |
| Session | ~12hrs | Save every round to Google Drive |
| Disk | ~75GB ephemeral | Mount Drive as persistent storage |

---

## Deliverable

A single Jupyter notebook: **`adversarial_pipeline.ipynb`**

All code lives in one notebook with clearly separated cells per section. No separate Python files. Checkpoints and outputs save to Google Drive.

---

## Scaled-Down Config

```yaml
# configs/config.yaml

colab:
  drive_mount: "/content/drive"
  save_dir: "/content/drive/MyDrive/deepfake_robustness/"
  checkpoint_every_round: true

data:
  pubmed_path: "data/raw/pubmed_real.csv"
  fake_path: "data/raw/med_mmhl/"
  processed_path: "data/processed/"
  max_samples: 2000          # hard cap — prevents RAM OOM
  max_length: 256            # down from 512
  train_split: 0.70
  val_split: 0.15
  test_split: 0.15
  min_words: 50
  seed: 42

generator:
  model_name: "gpt2"
  batch_size: 8
  max_new_tokens: 150
  finetune_epochs: 3
  lr: 2e-5
  checkpoint_path: "models/generator/checkpoints/"

detector:
  model_name: "dmis-lab/biobert-base-cased-v1.2"
  num_labels: 2
  batch_size: 8              # down from 16
  max_length: 256            # down from 512
  epochs_per_round: 3        # down from 5
  lr: 2e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  checkpoint_dir: "models/detector/checkpoints/"

agent:
  model_name: "microsoft/biogpt"
  evasion_threshold: 0.5
  high_conf_threshold: 0.8
  max_new_tokens: 128        # down from 256
  temperature: 0.9
  top_p: 0.95
  lora_r: 8                  # down from 16
  lora_alpha: 16
  lora_dropout: 0.05
  finetune_epochs: 2         # down from 3
  batch_size: 4              # down from 8
  lr: 1e-4
  checkpoint_dir: "models/agent/checkpoints/"

loop:
  num_rounds: 5              # down from 10
  fake_pool_size: 200        # down from 1000
  hard_sample_top_k: 50      # down from 200
  save_checkpoint_every: 1   # every round (critical for Colab)
```

---

## Build Order (Sequential)

---

### PHASE 1 — Setup & Data `[~30 min]`

#### Cell 1 — Install dependencies
`!pip install` all required packages: `transformers>=4.38.0`, `datasets>=2.18.0`, `accelerate>=0.27.0`, `scikit-learn>=1.3.0`, `peft>=0.9.0`, `bert-score>=0.3.13`, `tqdm`

#### Cell 2 — Config block
Define a single `CFG` dataclass or dict with all hyperparameters from the config section above. All subsequent cells read from `CFG`. No YAML file — inline dict is sufficient for a notebook.

#### Cell 3 — Drive mount + helper functions
- Mount Google Drive
- Define `save_checkpoint(obj, name, round_num)` → saves to `CFG.save_dir`
- Define `load_checkpoint(name, round_num)` → loads from Drive
- Define `free_gpu(model)` → `del model; torch.cuda.empty_cache()`
- Define `log_metrics(metrics_dict, round_num)` → appends row to `metrics_log.csv` on Drive

#### Cell 4 — Download data
- Stream `load_dataset("slinusc/PubMedAbstractsSubset", split="train", streaming=True)` and keep the first 2000 usable rows → save as CSV, `label=0`
- If uploaded fake folders like `fakenews_article` / `sentence` are present in Colab, use them first; otherwise clone `https://github.com/styxsys0927/Med-MMHL.git` → parse fake texts → save as CSV, `label=1`
- Print row counts

#### Cell 5 — Prepare data
- Load both CSVs, merge on `[text, label]`
- Clean: drop < 50 words, deduplicate by text hash, strip HTML, normalize unicode, truncate at `CFG.max_length`
- Stratified 70/15/15 split → save `train.csv`, `val.csv`, `test.csv` to Drive
- Print class distribution per split

---

### PHASE 2 — GPT-2 Fake Text Generator `[~1 hr]`

#### Cell 6 — Fine-tune GPT-2
- Load `gpt2`, fine-tune as causal LM on label=1 rows from `train.csv`
- Prepend `"[FAKE] "` prefix to each text during training
- `finetune_epochs=3`, `batch_size=8`, `lr=2e-5`, `max_length=256`
- Save checkpoint to Drive on completion

#### Cell 7 — Generator inference function
- Define `generate_fake_batch(n, model, tokenizer)` → prompt `"[FAKE] "`, returns list of `n` strings
- Smoke test: generate 5 samples, print them
- Call `free_gpu(model)` after smoke test

---

### PHASE 3 — BioBERT Baseline Detector `[~1.5 hrs]`

#### Cell 8 — Fine-tune BioBERT
- Load `dmis-lab/biobert-base-cased-v1.2` as `AutoModelForSequenceClassification`, `num_labels=2`
- Tokenize: `max_length=256`, `padding="max_length"`, `truncation=True`
- Train on `train.csv`, evaluate on `val.csv` each epoch
- AdamW + linear warmup scheduler. Save best checkpoint (by val F1) to Drive

#### Cell 9 — Scorer function + round 0 baseline
- Define `score_batch(texts, model, tokenizer)` → returns list of confidence scores (softmax prob of label=1)
- Evaluate on `test.csv` → log AUC, F1 as `round_0_metrics`
- Call `free_gpu(model)`

---

### PHASE 4 — BioGPT Adversarial Agent `[~1 hr]`

#### Cell 10 — Prompt templates
Define two string constants in the cell:
- `REWRITE_PROMPT`: takes `{fake_abstract}`, instructs BioGPT to paraphrase to evade detection while preserving meaning
- `GENERATE_PROMPT`: takes `{topic}`, instructs BioGPT to write a fictitious but plausible biomedical abstract

#### Cell 11 — BioGPT + LoRA setup and agent functions
- Load `microsoft/biogpt` with LoRA: `lora_r=8`, `lora_alpha=16`, `lora_dropout=0.05`, target modules `["q_proj", "v_proj"]`
- Define `rewrite_batch(texts, scores, model, tokenizer)`:
  - Only process texts where score > `CFG.high_conf_threshold`
  - Apply `REWRITE_PROMPT`, generate with `max_new_tokens=128`, `temperature=0.9`, `top_p=0.95`
  - Return rewritten texts
- Define `finetune_on_successes(success_pairs, model, tokenizer)`:
  - Causal LM fine-tune on `(prompt, rewrite)` pairs for `finetune_epochs=2`
  - Save only LoRA adapter weights to Drive

---

### PHASE 5 — Adversarial Loop `[~2 hrs to build, ~5 hrs to run]`

#### Cell 12 — Adversarial loop

Implement the full loop in one cell. **Critical:** follow load/unload pattern strictly.

```
for round k in range(1, num_rounds+1):

  STEP 1 — Generate fake pool
    load GPT-2 generator → GPU
    generate fake_pool_size=200 fake abstracts
    free_gpu(generator)

  STEP 2 — Score fake pool
    load BioBERT detector (latest checkpoint) → GPU
    score all 200 samples → confidence scores
    identify hard_samples: top 50 by score
    free_gpu(detector)

  STEP 3 — Agent rewrites hard samples
    load BioGPT agent (latest LoRA checkpoint) → GPU
    rewrite_batch(hard_samples)
    re-score rewrites with detector (reload → free)
    record successful_evasions: confidence < evasion_threshold
    free_gpu(agent)

  STEP 4 — Retrain detector
    load BioBERT → GPU
    augmented_train = train.csv + successful_evasions
    retrain for epochs_per_round=3
    save checkpoint to Drive as round_k_detector
    free_gpu(detector)

  STEP 5 — Fine-tune agent
    load BioGPT agent → GPU
    finetune_on_successes(successful_evasions)
    save LoRA adapter to Drive as round_k_agent
    free_gpu(agent)

  STEP 6 — Evaluate + log
    load BioBERT → GPU, evaluate on test.csv
    compute AUC, F1, evasion_rate, robustness_delta
    free_gpu(detector)
    log_metrics(metrics, round_num)
    print round summary
```

- Wrap each round in `try/except` — log error, continue next round
- Resume logic: check Drive for last completed round, skip done rounds

---

### PHASE 6 — Evaluation `[~30 min]`

#### Cell 13 — Metrics functions
Define inline: `compute_evasion_rate`, `compute_auc_f1`, `compute_robustness_delta`, `compute_bertscore`

#### Cell 14 — Results + plots
- Load `metrics_log.csv` from Drive
- Plot `evasion_rate_vs_round` and `auc_f1_vs_round` using matplotlib, save PNGs to Drive
- Print final results table
- Print ablation summary if multiple checkpoint sets exist

---

## Session Plan (Colab Free Tier)

```
Session 1 (~2 hrs): Run Cells 1–7  (setup + data + GPT-2 generator)
Session 2 (~2 hrs): Run Cells 8–9  (BioBERT baseline)
Session 3 (~6 hrs): Run Cells 10–12 (agent + adversarial loop, rounds 1–5)
Session 4 (~30 min): Run Cells 13–14 (evaluation + plots)
```

Each session mounts Drive first (Cell 3) and resumes from last saved checkpoint. All cells are re-runnable independently.

---

## Load/Unload Pattern (Mandatory)

Every model interaction in the loop must follow this pattern:

```python
# Load
model = load_model_from_checkpoint(path).to("cuda")

# Use
outputs = model(inputs)

# Unload — always both lines
del model
torch.cuda.empty_cache()
```

Never hold two large models on GPU simultaneously. BioBERT ≈ 3GB VRAM, BioGPT ≈ 6GB VRAM, GPT-2 ≈ 1GB VRAM.

---

## VRAM Budget Per Step

| Step | Model | VRAM | Fits on T4? |
|---|---|---|---|
| Generate fakes | GPT-2 | ~1GB | ✅ |
| Score samples | BioBERT | ~3GB | ✅ |
| Agent rewrite | BioGPT + LoRA | ~6–7GB | ✅ |
| Retrain detector | BioBERT | ~6GB (batch_size=8) | ✅ |
| Fine-tune agent | BioGPT + LoRA | ~10–11GB | ⚠️ tight — use gradient_checkpointing=True |
| All at once | — | ~16GB+ | ❌ never do this |

---

## Key Implementation Notes for Codex

1. **Single file: `adversarial_pipeline.ipynb`** — all code in one notebook, no external `.py` files
2. **All config in one `CFG` dict** at the top — no YAML, no file I/O for config
3. **Always use `AutoTokenizer` and `AutoModel` classes** — no hardcoded model-specific classes
4. **Use `accelerate` for training loops** — handles mixed precision on T4 automatically
5. **Use `peft` for LoRA** — do not implement from scratch
6. **Dataset loading must cap samples**: `split="train[:2000]"` syntax
7. **All persistent I/O uses Drive paths** (`/content/drive/MyDrive/deepfake_robustness/`)
8. **Every training loop must have `tqdm` progress bars**
9. **Use `sklearn.metrics` for AUC and F1**
10. **BioGPT fine-tuning must call `model.gradient_checkpointing_enable()`** before training
11. **Save only LoRA adapter weights** — `model.save_pretrained()` on the PEFT-wrapped model (~5MB vs 1.4GB per round)
12. **Metrics CSV must append per round** — never overwrite

---

## Expected Output After All Sessions

| Metric | Round 0 | Round 5 Target |
|---|---|---|
| AUC | ~0.85–0.90 | ~0.91–0.94 |
| F1 | ~0.82–0.88 | ~0.88–0.93 |
| Evasion Rate | ~35–45% | <15% |
| Robustness Delta | 0.0 | >0.05 |
