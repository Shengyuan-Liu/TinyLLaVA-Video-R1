# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TinyLLaVA-Video-R1 is a small-scale (<4B parameters) video reasoning model built on TinyLLaVA-Video. It uses GRPO (Group Relative Policy Optimization) reinforcement learning to enhance reasoning abilities, producing "aha moments" where the model self-corrects its reasoning. Based on arXiv paper 2504.09641.

## Setup

```bash
conda create -n tinyllava_video python=3.10 -y
conda activate tinyllava_video
pip install -e .
pip install flash-attn==2.7.3 --no-build-isolation
```

## Key Commands

### Training (two-stage process)
```bash
# Stage 1: Cold-start supervised fine-tuning
bash scripts/train/train_qwen2_coldstart.sh

# Stage 2: GRPO reasoning training (main training)
bash scripts/train/train_qwen2_reason_nextqa.sh
```

Both scripts require editing data/model paths before running. Training uses DeepSpeed ZeRO-3 across 8 GPUs by default.

### Evaluation
```bash
# Single benchmark (single-GPU, set MODEL_PATH/EVAL_DIR in script first)
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/videomme.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mvbench.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mlvu.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mmvu.sh

# Quick single-sample inference (edit model_path/prompt/video_file in eval.py)
CUDA_VISIBLE_DEVICES=0 python eval.py
```

## Architecture

Three-component pipeline: **Vision Tower -> Connector -> LLM**

- **Vision Tower** (frozen): SigLIP encoder extracts visual features from video frames (`tinyllava/model/vision_tower/siglip.py`)
- **Connector** (trainable): Group Perceiver Resampler maps vision features to 512 query tokens (`tinyllava/model/connector/groupresampler.py`)
- **LLM** (trainable): Qwen2.5-3B generates text with reasoning (`tinyllava/model/llm/qwen2.py`)

Orchestrated by `TinyLlavaForConditionalGeneration` in `tinyllava/model/modeling_tinyllava.py`.

## Code Structure

- `tinyllava/model/` - Model definition: main model (`modeling_tinyllava.py`), config (`configuration_tinyllava.py`), loading (`load_model.py`)
- `tinyllava/train/` - Training: GRPO training (`train.py`, `tinyllava_trainer_reason.py`), cold-start SFT (`train_coldstart.py`, `tinyllava_trainer.py`)
- `tinyllava/data/` - Data loading and preprocessing: datasets (`dataset_coldstart.py`), video/image/text preprocessing, conversation templates in `template/`
- `tinyllava/eval/` - Inference (`run_tiny_llava.py`) and benchmark-specific evaluation scripts
- `tinyllava/utils/` - CLI arguments (`arguments.py`), constants, utilities
- `scripts/` - Shell scripts for training and evaluation, plus DeepSpeed configs (`zero2.json`, `zero3.json`)

## Training Data Format

Training expects `<think>` and `<answer>` tags in model outputs:
```json
{"video": "path/to/video.mp4", "conversations": [
  {"from": "human", "value": "<image>\nQuestion"},
  {"from": "gpt", "value": "<think>reasoning</think>\n<answer>A</answer>"}
]}
```

The `<image>` token is the placeholder for video frame embeddings. GRPO training uses accuracy reward functions that extract and validate the `<answer>` tag content.

## Key Dependencies

- PyTorch 2.5.1, Transformers 4.49.0, TRL 0.14.0 (GRPO), DeepSpeed 0.15.3
- Flash-Attention 2.7.3 (installed separately)
- Decord/PyTorchVideo for video processing
