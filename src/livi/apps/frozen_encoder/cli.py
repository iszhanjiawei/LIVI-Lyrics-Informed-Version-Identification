# src/livi/apps/training/cli.py
import typer
from pathlib import Path
from livi.utils.paths import p

from typing import Optional, List
from livi.config import settings
from livi.apps.frozen_encoder.infer.session import run_inference_single, run_estimate_time, run_inference

app = typer.Typer(help="Frozen Encoder")


# ------------------------------------------------------------
# Command: infer-one
#
# Purpose:
#   Run inference on a single audio file using the frozen encoder:
#   Audio -> vocal segments extraction -> Whisper transcription -> text encoding.
#
# Typical usage:
#   poetry run livi-frozen-encoder infer-one \
#       --audio-path   src/livi/test_data/test.mp3 \
#
# Requirements:
# - Audio file must exist and be readable.
# - Config YAML must define preprocessing and model setup (default parameters are provided in this file)
#
# Arguments:
#   config_path : Path
#       Path to YAML config file controlling preprocessing and encoder setup.
#   audio_path : Path
#       Path to input audio file (.mp3, .wav, etc.).
#
# Output:
#   - Prints the shape of the resulting embedding (tuple).
# ------------------------------------------------------------
@app.command("infer-one")
def cli_infer_one(
    config_path: Path = typer.Option(None, help="Path to model/config YAML (used for data/preproc)."),
    audio_path: Path = typer.Option(..., help="Path to an audio file (.mp3, .wav, ...)"),
):
    """
    Encode a single audio file with the frozen encoder pipeline.
    """
    config_path = config_path or Path("src/livi/apps/frozen_encoder/config/infer.yaml")
    emb = run_inference_single(
        config_path=config_path,
        audio_path=audio_path,
    )
    typer.echo(f"Shape: {tuple(emb.shape)}")


# ------------------------------------------------------------
# Command: inference
#
# Purpose:
#   Run frozen-encoder inference on all audio files in a directory.
#   Extracts vocal segments, transcribes with Whisper, encodes with
#   a multilingual text encoder, and saves embeddings to disk.
#
# Typical usage:
#   poetry run livi-frozen-encoder inference \
#       --audio-dir src/livi/test_data/audio \
#       --out-path src/livi/test_data/covers80_lyrics_embeddings.npz
#
# Requirements:
# - `config_path` must point to a valid infer.yaml (defines transcriber,
#   text encoder, and vocal detection settings).
# - `audio_dir` must contain .mp3 files.
# - `out_path` should finish with .npz
#
# Arguments:
#   config_path : Path to inference config (YAML).
#   audio_dir   : Directory containing audio files.
#   out_path    : Optional destination for embeddings (.npz).
#                 Defaults to <audio_dir>/<audio_dir.name>_embeddings.npz.
# ------------------------------------------------------------
@app.command("inference")
def cli_infer_dir(
    config_path: Path = typer.Option(None, exists=True, readable=True, help="Path to infer.yaml."),
    audio_dir: Path = typer.Option(..., exists=True, file_okay=False, help="Directory containing audio files."),
    out_path: Optional[Path] = typer.Option(..., help="Output .npz path"),
):
    """
    Batch inference over a directory of audio files and save embeddings as a .npz.
    """
    config_path = config_path or Path("src/livi/apps/frozen_encoder/config/infer.yaml")
    embeddings = run_inference(
        config_path=config_path,
        audio_dir=audio_dir,
        path_out=out_path,
    )
    typer.echo(f"Done. Wrote {len(embeddings)} embeddings.")


# ------------------------------------------------------------
# Command: estimate-time
#
# Purpose:
#   Benchmark the frozen-encoder inference pipeline by estimating
#   average runtime for preprocessing, transcription, text encoding,
#   and overall end-to-end inference on a sample of audio files.
#
# Typical usage:
#   poetry run livi-frozen-encoder estimate-time \
#       --audio-dir src/livi/test_data/audio \
#
# Requirements:
# - `config_path` must point to a valid infer.yaml (defines all
#   transcriber, text encoder, and vocal detection settings).
# - `audio_dir` must contain audio files (.mp3).
#
# Arguments:
#   config_path : Path to inference config (YAML).
#   audio_dir   : Directory containing audio files (recursively scanned).
#   sample_size : Number of audio files to randomly sample (default: 200).
#   start_after : Number of warm-up iterations to skip (default: 5).
#   seed        : RNG seed for reproducibility (default: 42).
#
# Output:
#   - Logs mean and standard deviation for each stage:
#       * Preprocessing (load + vocal detection)
#       * Transcription (Whisper)
#       * Text encoding
#       * Total time
#
# Notes:
#   - Skips warm-up iterations before timing to avoid torch.compile overhead.
#   - Results are printed in seconds (mean ± std).
# ------------------------------------------------------------
@app.command("estimate-time")
def cli_estimate_time(
    config_path: Path = typer.Option(None, help="Path to model/config YAML."),
    audio_dir: Path = typer.Option(..., help="Directory to search recursively for *.mp3"),
    sample_size: int = typer.Option(200, help="Number of files to sample."),
    start_after: int = typer.Option(5, help="Warm-up iterations to skip."),
    seed: int = typer.Option(42, help="Sampling seed."),
):
    """
    Estimate average preprocessing, transcription, text-encoding, and total time.
    """
    config_path = config_path or Path("src/livi/apps/frozen_encoder/config/infer.yaml")

    run_estimate_time(
        config_path=config_path,
        audio_dir=audio_dir,
        sample_size=sample_size,
        start_after=start_after,
        seed=seed,
    )
