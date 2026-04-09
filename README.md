# Deepfake Robustness Pipeline

This repository implements an adversarial training pipeline for medical-text deepfake detection.
The code is scaffolded so it can be developed without local datasets and trained later on a machine
with enough storage and GPU memory.

## Project Layout

- `configs/config.yaml`: central configuration
- `data/prepare_data.py`: dataset cleaning and split generation
- `models/detector/train_detector.py`: BioBERT fine-tuning and inference helpers
- `models/seqgan/`: SeqGAN generator, discriminator, rollout, and trainer
- `agents/adversarial_agent.py`: BioGPT-based rewrite/generation agent
- `training/adversarial_loop.py`: round-based adversarial training loop
- `evaluation/`: metrics and end-to-end evaluation entry points

## Quick Start

1. Create a Python 3.10 environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Review `configs/config.yaml`.
4. Prepare data on the target machine with enough storage.
5. Train the baseline detector and SeqGAN.
6. Run the adversarial loop.

## Notes

- The current defaults target an RTX 3060 with 12GB VRAM.
- Dataset download is intentionally decoupled from local development so this codebase can be
  transferred to another system for actual training.
