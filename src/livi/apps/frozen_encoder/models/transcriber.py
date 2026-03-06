import re
from typing import List, Optional, Union, Tuple, Dict, Set
import ast

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from functools import lru_cache
from tqdm import tqdm
import os
from pathlib import Path
import pandas as pd
from loguru import logger

from collections import defaultdict

from livi.core.data.utils.audio_toolbox import load_audio, split_audio_30s, split_audio_predefined


class Transcriber:
    """
    Wrapper around a Hugging Face Whisper model for transcribing
    a list of audio chunks into cleaned text.

    Workflow
    --------
    1. Load model and processor with chosen dtype/device.
    2. Batch audio chunks → run `.generate()` → decode to strings.
    3. Post-process:
    - remove unwanted phrases,
    - collapse repeated words,
    - drop very short/empty outputs.
    4. Join remaining segments with double newlines.

    Parameters
    ----------
    model_name : str
        Hugging Face model ID (e.g., "openai/whisper-large-v3-turbo").
    device : str, optional
        Device to run inference on ("cuda" or "cpu"). Default: auto-detect ("cuda" if available).
    dtype_fp16_on_cuda : bool, default=True
        Use float16 precision on CUDA if available, else float32.
    sampling_rate : int, default=16000
        Expected input audio sampling rate (Hz).

    Generation parameters
    ---------------------
    num_beams : int
    condition_on_prev_tokens : bool
    compression_ratio_threshold : float
    temperature : tuple[float, ...]
    logprob_threshold : float
    return_timestamps : bool

    Cleaning parameters
    -------------------
    remove_phrases : list[str]
        Exact phrases to remove (case-insensitive, whole words).
    repeat_threshold : int
        Collapse words repeated at least this many times.
    min_words_per_chunk : int
        Discard chunk-level text shorter than this.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3-turbo",
        *,
        device: Optional[str] = None,
        dtype_fp16_on_cuda: bool = True,
        sampling_rate: int = 16000,
        num_beams: int = 1,
        condition_on_prev_tokens: bool = False,
        compression_ratio_threshold: float = 1.35,
        temperature: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        logprob_threshold: float = -1.0,
        return_timestamps: bool = True,
        remove_phrases: Optional[List[str]] = None,
        repeat_threshold: int = 3,
        min_words_per_chunk: int = 4,
    ) -> None:
        # -------- device / dtype ----------
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        use_fp16 = dtype_fp16_on_cuda and torch.cuda.is_available()
        self.torch_dtype = torch.float16 if use_fp16 else torch.float32
        self.sampling_rate = int(sampling_rate)

        # -------- model / processor -------
        self.model: AutoModelForSpeechSeq2Seq = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.device)
        self.processor: AutoProcessor = AutoProcessor.from_pretrained(model_name)

        # -------- generation args --------
        self.num_beams = int(num_beams)
        self.condition_on_prev_tokens = bool(condition_on_prev_tokens)
        self.compression_ratio_threshold = float(compression_ratio_threshold)
        self.temperature = tuple(temperature)
        self.logprob_threshold = float(logprob_threshold)
        self.return_timestamps = bool(return_timestamps)

        # -------- cleaning args ----------
        self.remove_phrases = remove_phrases if remove_phrases is not None else ["Thank you.", "music"]
        self.repeat_threshold = int(repeat_threshold)
        self.min_words_per_chunk = int(min_words_per_chunk)

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------
    def _build_gen_kwargs(self, translate: bool) -> dict:
        """
        Build generation kwargs from the object attributes.
        If `translate` is True, add `language="en"` to the kwargs.
        """
        gen_kwargs = {
            "num_beams": self.num_beams,
            "condition_on_prev_tokens": self.condition_on_prev_tokens,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "temperature": self.temperature,
            "logprob_threshold": self.logprob_threshold,
            "return_timestamps": self.return_timestamps,
        }
        if translate:
            gen_kwargs["language"] = "en"
        return gen_kwargs

    def transcribe(
        self,
        audio_chunks: List[np.ndarray],
        translate: bool = False,
        max_batch_size: int = 2,
    ) -> Optional[Tuple[List[str], List[bool], str]]:
        """
        Transcribe a list of audio chunks, clean + filter results, and
        return text(s) with a boolean mask indicating which chunks succeeded.

        Parameters
        ----------
        audio_chunks : list
            List of 1D arrays at `self.sampling_rate`.
        translate : bool
            If True, set gen_kwargs["language"] = "en".
        max_batch_size : int
            Maximum number of chunks to process in a single Whisper forward pass.
            Limits peak GPU memory regardless of how many chunks a song has.

        Returns
        -------
        Optional[Tuple[List[str], List[bool], str]]
            - (texts, mask), where texts is the list of
            per-chunk strings, mask is a parallel list of booleans
        """
        # ---- gen kwargs ----
        gen_kwargs = self._build_gen_kwargs(translate)

        # ---- sub-batch forward pass to cap peak GPU memory ----
        raw_texts = []
        for batch_start in range(0, len(audio_chunks), max_batch_size):
            batch_chunks = audio_chunks[batch_start:batch_start + max_batch_size]

            # ---- feature extraction with fallback ----
            processed = self.processor(
                batch_chunks,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding="longest",
                truncation=False,
                return_attention_mask=True,
            )
            if processed["input_features"].shape[-1] < 3000:
                processed = self.processor(
                    batch_chunks,
                    return_tensors="pt",
                    sampling_rate=self.sampling_rate,
                )

            processed = {k: v.to(self.device, dtype=self.torch_dtype) for k, v in processed.items()}

            with torch.no_grad():
                pred_ids = self.model.generate(**processed, **gen_kwargs)
                batch_texts = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
                raw_texts.extend(batch_texts)

                # 显式释放GPU张量，避免累积
                del pred_ids
                del processed
            
            # 每个 sub-batch 后释放缓存，防止峰值显存累积
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ---- cleaning ----
        cleaned = self.clean_transcription(raw_texts)

        # ---- filtering: keep if not empty & has enough words/characters ----
        def has_enough_content(text: str) -> bool:
            """判断文本是否有足够内容（区分中英文）"""
            if not text.strip():
                return False
            # 检查是否包含中文字符
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
            if has_chinese:
                # 中文：直接计算字符数（去除空格）
                return len(text.replace(' ', '')) >= self.min_words_per_chunk
            else:
                # 英文：计算单词数
                return len(text.split()) >= self.min_words_per_chunk
        
        mask = [has_enough_content(t) for t in cleaned]
        texts = [t if ok else "" for t, ok in zip(cleaned, mask)]

        # joined_text = "\n\n".join([t for t, ok in zip(texts, mask) if ok])
        return texts, mask

    # ---------------------------------------------------------------------
    # Cleaning utilities
    # ---------------------------------------------------------------------
    def clean_transcription(self, pred_text: List[str]) -> List[str]:
        """
        Apply cleanup:
        1) Remove phrases in `self.remove_phrases` (whole-word, case-insensitive).
        2) Collapse repeated words past `self.repeat_threshold`.
        """

        def clean_text(text: str) -> str:
            out = text
            for phrase in self.remove_phrases:
                out = re.sub(rf"\b{re.escape(phrase)}\b", "", out, flags=re.IGNORECASE)
            return out.strip()

        def collapse_repeated_words(text: str) -> str:
            if self.repeat_threshold <= 1:
                return text
            pattern = re.compile(
                rf"\b(\w+)([,\s]+(?:\1\b[,\s]+){{{self.repeat_threshold - 1},}})",
                flags=re.IGNORECASE,
            )
            return pattern.sub(r"\1 ", text)

        return [collapse_repeated_words(clean_text(x)) for x in pred_text]


# --------------------------------------------------------------------
# Runners
# --------------------------------------------------------------------


@lru_cache(maxsize=2)
def _get_cached_transcriber(
    model_name: str = "openai/whisper-large-v3-turbo",
    device: Optional[str] = None,
    dtype_fp16_on_cuda: bool = True,
    sampling_rate: int = 16000,
    num_beams: int = 1,
    condition_on_prev_tokens: bool = False,
    compression_ratio_threshold: float = 1.35,
    temperature: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    logprob_threshold: float = -1.0,
    return_timestamps: bool = True,
    remove_phrases: Optional[tuple[str, ...]] = ("Thank you.", "music"),
    repeat_threshold: int = 3,
    min_words_per_chunk: int = 4,
) -> Transcriber:
    """
    Build (or reuse cached) Transcriber. LRU cache ensures model is
    loaded only once per unique parameter configuration.
    """
    return Transcriber(
        model_name=model_name,
        device=device,
        dtype_fp16_on_cuda=dtype_fp16_on_cuda,
        sampling_rate=sampling_rate,
        num_beams=num_beams,
        condition_on_prev_tokens=condition_on_prev_tokens,
        compression_ratio_threshold=compression_ratio_threshold,
        temperature=temperature,
        logprob_threshold=logprob_threshold,
        return_timestamps=return_timestamps,
        remove_phrases=list(remove_phrases) if remove_phrases else None,
        repeat_threshold=repeat_threshold,
        min_words_per_chunk=min_words_per_chunk,
    )


def transcribe(
    audio_chunks: List[np.ndarray],
    translate: bool = False,
    *,
    transcriber: Optional[Transcriber] = None,
    model_name: str = "openai/whisper-large-v3-turbo",
    device: Optional[str] = None,
    dtype_fp16_on_cuda: bool = True,
    sampling_rate: int = 16000,
    num_beams: int = 1,
    condition_on_prev_tokens: bool = False,
    compression_ratio_threshold: float = 1.35,
    temperature: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    logprob_threshold: float = -1.0,
    return_timestamps: bool = True,
    remove_phrases: Optional[tuple[str, ...]] = ("Thank you.", "music"),
    repeat_threshold: int = 3,
    min_words_per_chunk: int = 4,
) -> Optional[Tuple[List[str], List[bool], str]]:
    """
    High-level helper to run transcription without manually
    instantiating a Transcriber. Uses cached models to avoid reload.

    Parameters
    ----------
    audio_chunks : list
        List of 1D arrays/tensors at 16 kHz.
    translate : bool
        Force English output if True.
    All other params are passed to the underlying Transcriber.

    Returns
    -------
    Optional[Tuple[List[str], List[bool], str]]
        - (texts, mask, joined_text)
    """
    transcriber = transcriber or _get_cached_transcriber(
        model_name=model_name,
        device=device,
        dtype_fp16_on_cuda=dtype_fp16_on_cuda,
        sampling_rate=sampling_rate,
        num_beams=num_beams,
        condition_on_prev_tokens=condition_on_prev_tokens,
        compression_ratio_threshold=compression_ratio_threshold,
        temperature=temperature,
        logprob_threshold=logprob_threshold,
        return_timestamps=return_timestamps,
        remove_phrases=remove_phrases,
        repeat_threshold=repeat_threshold,
        min_words_per_chunk=min_words_per_chunk,
    )

    return transcriber.transcribe(audio_chunks, translate=translate)
