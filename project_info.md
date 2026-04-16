# Project Information

## Project Name

**Deepfake Robustness Pipeline**

This project studies how to detect fake medical text and how to make that detector stronger against adversarial attacks.

## What Changed In The Current Implementation

The current checked-in code now uses a cleaner dataset strategy than before.

The main change is this:

- the **detector** is no longer trained on `PubMed real` vs `fakenews_article fake`
- the **detector** is now trained on the labeled `fakenews_article` dataset itself
- `PubMed` is now kept as a **separate real-domain reference set**

This matters because the earlier setup risked teaching the detector:

- scientific abstract style vs news/article style

instead of the actual target:

- real medical article vs fake medical article

The new strategy is much closer to the real goal of the project.

## Source Of Truth

The main experimental workflow is [agent-run.ipynb](agent-run.ipynb).

The shared Python files support the same strategy:

- [data/prepare_data.py](data/prepare_data.py)
- [data/download_pubmed.py](data/download_pubmed.py)
- [data/download_medfake.py](data/download_medfake.py)
- [models/seqgan/train_seqgan.py](models/seqgan/train_seqgan.py)
- [training/adversarial_loop.py](training/adversarial_loop.py)
- [evaluation/metrics.py](evaluation/metrics.py)
- [evaluation/visualization.py](evaluation/visualization.py)
- [evaluation/eval_pipeline.py](evaluation/eval_pipeline.py)

## Main Idea

The project has three core model roles:

- a **generator** that produces fake medical-style text
- a **detector** that classifies text as real or fake
- an **adversarial agent** that rewrites fake text to make it harder for the detector to catch

The full pipeline is round-based:

1. prepare the training data
2. train the generator
3. train the detector
4. generate fake samples
5. rewrite the strongest fake samples
6. retrain the detector on harder fake examples
7. repeat for several rounds

The purpose is not only to detect fake medical text once, but to improve robustness over time.

## The Dataset Strategy Used Now

### 1. Main detector dataset: `fakenews_article`

This is now the main dataset for the detector.

Important detail:

- `fakenews_article` is **already labeled**
- it contains both classes
- `det_fake_label == 0` means real
- `det_fake_label == 1` means fake

This makes it a much better fit for the detector than treating the whole folder as fake.

### 2. Optional auxiliary dataset: `sentence`

The `sentence` dataset is still supported, but it is **disabled by default** in the current strategy.

Reason:

- it is much noisier
- it contains many short fragments
- it is not ideal for article-level generation or detection

So the current code keeps it for later ablations, not as the primary training source.

### 3. Separate real-domain reference set: PubMed

PubMed is still part of the project, but its role has changed.

It is now used as a **reference real medical corpus**, not as the main detector label source.

Why:

- PubMed text is scientific abstract style
- `fakenews_article` is article/news style
- mixing them directly as detector labels can create a style mismatch

So the current implementation keeps PubMed as:

- a clean real-domain reference set
- a domain anchor for the project
- a useful auxiliary dataset for later evaluation or expansion

## Simple End-To-End Flow

1. Download a capped PubMed subset from Hugging Face.
2. Upload `fakenews_article` and optionally `sentence` in Colab, or fall back to the public Med-MMHL repository.
3. Preserve the label column already present in `fakenews_article`.
4. Build detector train/validation/test splits from the labeled article dataset.
5. Save a fake-only generator training file from the detector training split.
6. Save a cleaned PubMed reference CSV separately.
7. Fine-tune GPT-2 on fake article rows only.
8. Fine-tune BioBERT on the labeled article dataset.
9. Generate fake article-style samples with GPT-2.
10. Score those samples with BioBERT.
11. Select the samples the detector is most confident are fake.
12. Rewrite those samples with BioGPT + LoRA.
13. Score the rewrites again.
14. Add successful rewrites back into detector training as fake adversarial examples.
15. Retrain the detector.
16. Repeat the adversarial loop for multiple rounds.

## How Each Dataset Is Used

### `fakenews_article`

This is the most important dataset in the current implementation.

It is used in two ways:

- for **detector training**, using both real and fake labels
- for **generator training**, using only the fake rows

This makes the overall setup much more internally consistent.

### `sentence`

This dataset is optional.

It is currently **not used by default** because:

- it has many short rows
- many rows are fragments rather than full article text
- it is less suitable for GPT-2 generation and article-level detection

### PubMed

PubMed is used as:

- a separate cleaned real reference set
- a biomedical writing reference corpus
- future support for extra evaluation or domain-specific comparisons

## Source-Specific Cleaning Rules

The current preprocessing no longer uses one hard filter for every dataset.

Instead, it uses source-specific thresholds:

- `fakenews_article` keeps article-style rows with a lower minimum length than before
- `sentence` can use a much smaller threshold when enabled
- `PubMed` keeps a stricter threshold because it is used as a cleaner reference corpus

This helps avoid throwing away too much useful article data while still filtering out obvious junk.

## How GPT-2 Is Trained Now

GPT-2 is trained only on the **fake rows** from the article dataset.

In simple words:

- the model reads fake article text token by token
- it learns to predict the next token
- its prediction error becomes the loss
- training updates the weights so it gets better at continuing fake medical-style article text

So GPT-2 learns:

- how fake medical articles are written
- how to generate new fake medical article-style samples

This is much better than training GPT-2 on mixed real + fake rows.

## How BioBERT Is Trained Now

BioBERT is trained on the labeled `fakenews_article` dataset.

That means:

- the model sees article-style real rows
- the model also sees article-style fake rows
- it learns the difference between the two within the same text domain

This is better than comparing:

- PubMed abstracts
- against fake health-news articles

because now the detector has to learn a cleaner real-vs-fake decision instead of mostly learning a style difference.

## How The Adversarial Agent Works

The adversarial agent uses **BioGPT + LoRA**.

It rewrites fake texts that the detector currently flags with high confidence.

The loop is:

1. GPT-2 generates fake article-style samples.
2. BioBERT scores them.
3. The highest-confidence fake samples are selected.
4. BioGPT rewrites them to sound more natural and less suspicious.
5. The detector scores the rewritten versions again.
6. Successful evasions are added back into detector training.

This makes the detector progressively harder to fool.

The current loop now keeps those successful adversarial rewrites **cumulatively across rounds** instead of only using the latest round’s successful examples.

## Why This New Strategy Is Better

The new strategy is better because it keeps the main detector task inside one consistent text domain.

Before, the model risked learning:

- abstract vs article

Now it is much closer to learning:

- real medical article vs fake medical article

This makes the experiment more honest and more useful for the real project goal.

## Files Updated For This Strategy

### Shared data preparation

- [data/prepare_data.py](data/prepare_data.py)
  - now preserves label columns from the detector source
  - builds detector splits from labeled article data
  - saves a fake-only generator training CSV
  - saves a cleaned PubMed reference CSV

### PubMed download

- [data/download_pubmed.py](data/download_pubmed.py)
  - now streams the Hugging Face dataset
  - caps the number of saved rows

### Med-MMHL fallback download

- [data/download_medfake.py](data/download_medfake.py)
  - now points to the correct public repository
  - reports discovered dataset components

### SeqGAN path

- [models/seqgan/train_seqgan.py](models/seqgan/train_seqgan.py)
  - now defaults to the fake-only generator training file
  - no longer assumes the generator should read every row in the detector train split

### Colab notebook

- [agent-run.ipynb](agent-run.ipynb)
  - now stages `fakenews_article` as the main labeled detector source
  - optionally stages `sentence`
  - prepares `train.csv`, `val.csv`, `test.csv` from article labels
  - prepares `generator_train_fake.csv`
  - prepares `pubmed_reference.csv`
  - aligns the BioGPT prompts with medical-news-style text instead of abstract-style text

## Metrics Used Now

The user-facing metrics remain intentionally minimal:

- **AUC**
- **F1**
- **evasion_rate**

Why these three:

- **AUC** shows how well the detector separates real and fake text overall
- **F1** shows how well the detector performs at the decision threshold
- **evasion_rate** shows how often rewritten fake text gets past the detector

These metrics are enough for the main research question without overloading the analysis.

## Important Note About Existing Artifacts

Some existing notebook outputs, plots, or uploaded Hugging Face models may have been produced before this dataset-strategy update.

So the safest interpretation is:

- the **code now reflects the cleaner strategy**
- the **artifacts only fully match the new strategy after you rerun the notebook**

## Hugging Face Model Repositories

These are the current model repositories connected to the notebook workflow:

- GPT-2 generator: https://huggingface.co/Mr-Arr0gant/gan-vs-det-ai-gpt2-generator
- BioBERT detector: https://huggingface.co/Mr-Arr0gant/gan-vs-det-ai-biobert-detector
- BioGPT agent: https://huggingface.co/Mr-Arr0gant/gan-vs-det-ai-biogpt-agent

## Final Summary In Simple Words

The current implementation now works like this:

- use `fakenews_article` as the main labeled dataset
- use only its fake rows to train GPT-2
- use its real and fake rows to train BioBERT
- keep `sentence` optional and disabled by default
- keep PubMed as a separate real medical reference set
- run the adversarial loop on article-style fake text

This is a cleaner and more useful version of the project plan for the goal:

**detect fake medical text and make the detector stronger against adversarial attacks**
