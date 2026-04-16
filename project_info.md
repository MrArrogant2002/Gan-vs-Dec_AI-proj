# Project Information

## Project Name

**Deepfake Robustness Pipeline**

This project studies how to detect fake medical text and how to make that detector stronger against adversarial attacks.

## Source Of Truth

The main results in this repository come from [agent-run.ipynb](agent-run.ipynb).

That notebook is the source of truth for:

- dataset loading in Colab
- model training
- adversarial rounds
- final metric plots
- saved Drive artifacts
- Hugging Face model uploads

The modular Python files in `training/`, `evaluation/`, `models/`, and `agents/` are still useful, but the reported run in this project is the notebook run.

## Main Idea

The project has three main models:

- a **generator** that writes fake medical-style text
- a **detector** that decides whether a text is real or fake
- an **adversarial agent** that rewrites fake text to make it harder for the detector to catch

The goal is not just to build a detector once. The real goal is to repeat this process for multiple rounds so the detector keeps seeing stronger attacks and becomes more robust.

## Simple End-To-End Flow

1. Load **real medical text** from the PubMed subset.
2. Let the user upload **fake medical-news text** folders such as `fakenews_article` and `sentence`.
3. Clean both real and fake text.
4. Give labels:
   `0` = real
   `1` = fake
5. Split the data into train, validation, and test sets.
6. Fine-tune GPT-2 on fake training text so it can generate more fake medical-style samples.
7. Fine-tune BioBERT on labeled real + fake text so it can classify text as real or fake.
8. Generate a pool of fake samples with GPT-2.
9. Score those samples with the detector.
10. Pick the fake samples the detector is most confident are fake.
11. Rewrite those samples with BioGPT + LoRA so they sound more natural and are harder to catch.
12. Score the rewritten texts again.
13. Add successful rewritten fake samples back into detector training.
14. Retrain the detector.
15. Repeat for about 5 rounds.

## How Real PubMed Data Is Used

Real PubMed data is very important in this project.

It is used in these ways:

- It provides the **real class** for the detector.
- It teaches the detector what real medical writing looks like.
- It gives the project domain-specific language, such as medical terms, abstract structure, and scientific style.
- It is included in validation and test evaluation, so the detector is checked against real medical text and fake medical text together.

In simple words:

- PubMed data shows the detector what genuine medical writing looks like.
- Without it, the detector would not know what “real” means.

## How Fake Data Enters The Project

Fake data comes from two places.

### 1. User-uploaded fake dataset

The notebook first looks for uploaded folders such as:

- `fakenews_article`
- `sentence`

These files act as the starting fake dataset. They are labeled as fake and used in training.

### 2. Model-generated fake data

After the first dataset is prepared, GPT-2 is fine-tuned on the fake training text and then used to generate new fake samples.

Later, the adversarial BioGPT agent rewrites some of those fake samples again, producing even stronger fake examples.

So fake data in the full pipeline is:

- original uploaded fake text
- GPT-2 generated fake text
- BioGPT rewritten adversarial fake text

## How The Fake Text Becomes More Realistic

The fake text becomes more realistic step by step.

- The uploaded fake dataset already has medical-domain wording, so it gives the project a useful starting style.
- GPT-2 learns sentence flow, wording patterns, and structure from the fake training texts.
- GPT-2 then generates fresh fake medical-style text instead of only reusing the uploaded examples.
- BioGPT rewrites selected fake text to make it smoother and less obviously fake.
- Those rewrites often reduce the detector's confidence, which means the text looks more convincing to the model.

In simple words:

- GPT-2 creates new fake text.
- BioGPT polishes some of that fake text to make it harder to detect.

## How GPT-2 Is Trained In This Project

In the notebook run, GPT-2 is fine-tuned on the **fake training texts**, not on PubMed.

The training idea is simple:

- GPT-2 reads fake text token by token.
- At each step, it tries to predict the next token.
- The notebook compares GPT-2’s prediction with the real next token from the training text.
- The prediction error becomes the loss.
- Backpropagation updates the model weights.

After many training steps, GPT-2 learns the writing style of the fake medical dataset.

Then, during generation:

- the notebook gives GPT-2 a short prompt
- GPT-2 continues the text
- this produces a new fake medical-style sample

So GPT-2 is learning:

- how fake medical text is usually written
- how to produce new text in a similar style

## How BioBERT Is Trained In This Project

BioBERT is the detector.

Its job is different from GPT-2.

- GPT-2 learns how to **write**
- BioBERT learns how to **judge**

BioBERT training works like this:

- each text has a label
- `0` means real
- `1` means fake
- BioBERT reads the text and outputs scores for the two classes
- those scores become probabilities
- the predicted result is compared with the true label
- the difference becomes the classification loss
- backpropagation updates the detector weights

After training, BioBERT learns patterns such as:

- what real PubMed-style text looks like
- what fake medical-news text looks like
- what machine-generated text tends to look like

## How The Adversarial Agent Works

The adversarial agent uses **BioGPT + LoRA**.

It does not create the first fake text from scratch. Instead, it rewrites selected fake samples.

The notebook does this:

- it scores generated fake texts with the current detector
- it chooses the samples the detector is most confident are fake
- it sends those samples to BioGPT
- BioGPT rewrites them so they sound more natural and less suspicious
- the detector scores the rewritten versions again

If a rewritten sample drops below the fake threshold, it counts as a successful evasion.

Those successful evasions are then added back into detector training. This is what makes the loop adversarial.

## Why The Loop Improves Robustness

The loop helps because the detector is not trained only on easy fake examples.

Instead, it keeps seeing harder cases:

- generated fake text
- rewritten fake text
- successful evasions from earlier rounds

This forces the detector to adapt to stronger attacks over time.

## Essential Metrics Used Now

The project now uses a very small metric set on purpose.

The main metrics are:

- **AUC**
- **F1**
- **evasion_rate**

### What each metric means

- **AUC** tells us how well the detector separates real and fake texts across thresholds.
- **F1** tells us how well the detector balances precision and recall at the chosen decision threshold.
- **evasion_rate** tells us how often rewritten fake samples successfully slipped past the detector.

These are the files aligned to this smaller metric policy:

- [agent-run.ipynb](agent-run.ipynb)
- [evaluation/metrics.py](evaluation/metrics.py)
- [evaluation/visualization.py](evaluation/visualization.py)
- [evaluation/eval_pipeline.py](evaluation/eval_pipeline.py)
- [training/adversarial_loop.py](training/adversarial_loop.py)

The notebook still keeps `status` internally for resume and error handling, but the user-facing results focus on the three metrics above.

## Datasets Used In The Notebook

### Real dataset

- Source: `slinusc/PubMedAbstractsSubset`
- Accessed from Hugging Face in the notebook
- Used as the real class

### Fake dataset

- Main user flow: upload fake folders in Colab
- Expected folders include `fakenews_article` and `sentence`
- Fallback source: Med-MMHL-style fake medical misinformation content
- Used as the fake class

## Default Notebook Setup

The main notebook run is designed for Colab and uses a lighter setup.

### Generator

- model: `gpt2`
- role: generate fake medical-style text

### Detector

- model: `dmis-lab/biobert-base-cased-v1.2`
- role: classify real vs fake

### Adversarial agent

- model: `microsoft/biogpt`
- tuning method: LoRA
- role: rewrite fake text to evade detection

### Adversarial loop

- default rounds: 5
- fake pool per round: 200
- top selected hard samples: 50

## Hugging Face Model Repositories

The current notebook-trained models were pushed to these repositories:

- GPT-2 generator: https://huggingface.co/Mr-Arr0gant/gan-vs-det-ai-gpt2-generator
- BioBERT detector: https://huggingface.co/Mr-Arr0gant/gan-vs-det-ai-biobert-detector
- BioGPT agent: https://huggingface.co/Mr-Arr0gant/gan-vs-det-ai-biogpt-agent

## Results From The Notebook Run

The result summary below comes from the plots saved by `agent-run.ipynb`:

- `auc_f1_vs_round.png`
- `evasion_rate_vs_round.png`

### AUC trend

- AUC stays essentially at **1.00** across the rounds.
- This means the detector keeps very strong ranking power between real and fake samples.

### F1 trend

- F1 stays around **0.985** from round 0 to round 3.
- F1 drops to about **0.9705** at rounds 4 and 5.
- This suggests the detector stays very strong overall, but threshold-based decisions become a little less clean in later rounds.

### Evasion-rate trend

- round 1: about **0.02**
- round 2: about **0.02**
- round 3: **0.00**
- round 4: **0.00**
- round 5: about **0.02**

### Simple interpretation

- The detector remained very strong during the run.
- The adversarial agent had only limited success.
- Some rewritten samples did fool the detector, but the success rate stayed low.
- The detector adapted well enough that evasion never became a large problem in this experiment.

## Main Files In The Repository

### Notebook path used for the reported run

- [agent-run.ipynb](agent-run.ipynb)
- [colab-build-plan.md](colab-build-plan.md)

### Shared evaluation and training files

- [evaluation/metrics.py](evaluation/metrics.py)
- [evaluation/visualization.py](evaluation/visualization.py)
- [evaluation/eval_pipeline.py](evaluation/eval_pipeline.py)
- [training/adversarial_loop.py](training/adversarial_loop.py)

### Background modular code

- `models/seqgan/`
- `models/detector/`
- `agents/`
- `data/`
- `configs/config.yaml`

## Important Notes

- The notebook run is the main experimental path for this project.
- The uploaded Hugging Face models come from the notebook pipeline, not from a separate SeqGAN training run.
- The repository still contains the older modular SeqGAN-based structure, but the current reported results are from the Colab notebook workflow.
