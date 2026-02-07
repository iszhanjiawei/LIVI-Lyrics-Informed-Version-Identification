# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LIVI (Lyrics-Informed Version Identification) is a music cover retrieval system. It trains a lightweight audio encoder to produce lyrics-informed embeddings, using supervision from Whisper transcription and text embedding models. At inference time, no ASR is needed — only the audio encoder runs, making retrieval efficient.

## Build & Development Commands

### Installation
```bash
poetry install
poetry shell
```

### Docker (alternative)
```bash
make build        # build image
make run-bash     # run container with interactive shell
make stop         # stop container
make logs         # view container logs
```

### Code Quality
```bash
# Lint + type check (runs inside Docker)
make check        # mypy, ruff check, ruff format --check

# Auto-format (runs inside Docker)
make clean        # ruff check --fix, ruff format

# Without Docker (inside poetry shell)
poetry run mypy --show-error-codes src
poetry run ruff check --no-fix src
poetry run ruff format --check src
poetry run ruff check --fix src
poetry run ruff format src
```

### CLI Applications
All CLIs use Typer. Use `--help` on any command for options.
```bash
# Frozen encoder: vocal detection -> Whisper transcription -> text embedding
poetry run livi-frozen-encoder inference --audio-dir data/raw/audio --out-path text.npz

# Audio encoder: inference with trained model
poetry run livi-audio-encoder inference --audio-dir data/raw/audio --out-path audio.npz
poetry run livi-audio-encoder infer-one --audio-path file.mp3
poetry run livi-audio-encoder launch-training

# Retrieval evaluation
poetry run livi-retrieval-eval evaluate \
  --path-metadata data/raw/metadata/benchmark.csv \
  --path-embeddings audio.npz \
  --col-id version_id --text-id lyrics --k 100 \
  --path-metrics results/metrics/metrics.csv

# Additional CLIs
poetry run livi-audio-baselines  # audio baseline models
poetry run livi-data             # data utilities
```

## Architecture

### Three-Stage Pipeline

**Stage 1 — Frozen Encoder** (`src/livi/apps/frozen_encoder/`): Generates training targets. Audio is split into 30s chunks, transcribed with Whisper (large-v3-turbo), then lyrics are embedded with a sentence-transformer (Alibaba-NLP/gte-multilingual-base). Output: 768-dim lyrics embeddings.

**Stage 2 — Audio Encoder Training** (`src/livi/apps/audio_encoder/`): Trains a lightweight head on top of a frozen Whisper encoder. The trainable components are:
- `AttentionPooling` — learnable [CLS] token with RoPE, aggregates Whisper frame-level states (B, T, 1280) into a single vector (B, 1280)
- `Projection` — MLP (1280 -> [3072, 2048, 2048, 1536] -> 768) with LayerNorm + ReLU
- Output is L2-normalized

Loss function (`MSECosineLoss`): `L = α * L_cos + (1-α) * L_mse` where L_cos aligns audio-lyrics pairs and L_mse preserves similarity geometry across the batch. α=0.5 by default.

**Stage 3 — Inference & Retrieval** (`src/livi/apps/retrieval_eval/`): At inference, only the audio encoder is needed (no transcription). Audio embeddings are compared via cosine similarity. Metrics: MR1, HR@k, MAP@k.

### Key Source Layout
- `src/livi/apps/audio_encoder/models/` — `LiviAudioEncoder`, `AttentionPooling`, `Projection`, `WhisperEncoder`
- `src/livi/apps/audio_encoder/train/` — `Trainer` (PyTorch Lightning), `MSECosineLoss`, validation metrics, LR scheduler
- `src/livi/apps/audio_encoder/data/` — WebDataset pipeline for sharded .tar archives
- `src/livi/apps/frozen_encoder/models/` — `Transcriber` (Whisper wrapper), `TextEncoder` (sentence-transformers wrapper)
- `src/livi/apps/retrieval_eval/` — `Ranker` for cosine similarity search and metric computation
- `src/livi/core/data/preprocessing/` — vocal detection (30s chunking), Whisper feature extraction (mel spectrograms)
- `src/livi/core/data/utils/` — audio loading/resampling, embedding I/O (NPZ, PKL)

### Configuration
Hydra YAML configs live alongside each app:
- `src/livi/apps/audio_encoder/config/livi.yaml` — training hyperparameters (batch 128, lr 1e-4, warmup 10k steps, 3 epochs)
- `src/livi/apps/audio_encoder/config/infer.yaml` — inference settings
- `src/livi/apps/frozen_encoder/config/infer.yaml` — transcription + text encoding settings

Model checkpoints go in `src/livi/apps/audio_encoder/checkpoints/livi.pth`.

### Data Format
- Training data: WebDataset shards (.tar) containing precomputed mel spectrograms + lyrics embeddings
- Embeddings: .npz files mapping IDs to vectors
- Metadata: CSV files with `version_id`, `clique_id`, `lyrics`, `md5_encoded` columns
- Audio: 16kHz sample rate, 30-second chunks

## Code Style
- Python >=3.11, <3.12
- Ruff for linting (line-length 120, rules: E741/E742/E743/F/I)
- Black for formatting (via ruff)
- mypy in strict mode
- Pre-commit hooks: ruff + black
- Experiment tracking: Weights & Biases

## Running Long Tasks
For long-running training or inference, use `nohup` with unbuffered Python output and the conda env Python path directly:
```bash
nohup /path/to/env/bin/python -u script.py > log 2>&1 &
```
Avoid wrapping with `conda run` for background tasks as it adds buffering layers.


1、执行命令脚本的话，请在 LIVI 虚拟环境下面运行，使用命令 conda activate LIVI，激活 LIVI 虚拟环境。

2、模型训练的运行启动命令，请使用后台nohup运行，避免关掉终端后，训练被中断

3、多层包装会导致多层缓冲，使日志无法实时查看，Python 默认缓冲会延迟日志输出，长时间运行的任务必须使用 -u 参数

# 避免：多层嵌套
nohup conda run -n env bash -c "python script.py" > log 2>&1 &
# 推荐：直接调用
nohup /path/to/env/bin/python -u script.py > log 2>&1 &

4、后台任务应该直接使用 conda 环境的 Python 路径，conda run 适合临时命令，不适合长时间后台任务

5、复杂的启动命令容易忘记，写成脚本或文档，方便复用
