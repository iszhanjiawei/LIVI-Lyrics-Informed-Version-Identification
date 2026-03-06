from __future__ import annotations

from typing import List, Tuple, Optional
from functools import lru_cache
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F


from livi.core.data.utils.audio_toolbox import load_audio


class VocalDetector:
    """
    Vocal activity detection and fixed-length audio chunks extraction.

    Two modes:
    1. Preprocessed mode (use_preprocess=True): 
       - Load vocal segments from Musicnn preprocessing JSON files
       - Extract 30s chunks based on detected vocal segments
    2. Simple mode (use_preprocess=False):
       - Extract 30s non-overlapping chunks from raw audio (old behavior)
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        chunk_sec: float = 30.0,
        use_preprocess: bool = False,
        preprocess_dir: Optional[str] = None,
    ) -> None:
        """Initialize the VocalDetector.

        Args:
            sample_rate (int, optional): Sample rate for audio processing. Defaults to 16_000.
            chunk_sec (float, optional): Duration of each audio chunk in seconds. Defaults to 30.0.
            use_preprocess (bool, optional): Whether to use preprocessed vocal segments. Defaults to False.
            preprocess_dir (str, optional): Directory containing preprocessing JSON files. Required if use_preprocess=True.
        """
        self.sample_rate = int(sample_rate)
        self.chunk_sec = float(chunk_sec)
        self.chunk_size = int(self.sample_rate * self.chunk_sec)
        self.use_preprocess = use_preprocess
        self.preprocess_dir = Path(preprocess_dir) if preprocess_dir else None
        
        if self.use_preprocess and not self.preprocess_dir:
            raise ValueError("preprocess_dir must be specified when use_preprocess=True")

    def load_preprocess_info(self, audio_path: str) -> Optional[dict]:
        """
        Load preprocessing info for a given audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Preprocessing info dict or None if not found
        """
        if not self.use_preprocess or not self.preprocess_dir:
            return None
        
        audio_stem = Path(audio_path).stem
        json_path = self.preprocess_dir / f"{audio_stem}.json"
        
        if not json_path.exists():
            print(f"⚠️  Preprocessing file not found: {json_path}")
            return None
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查是否有明确的失败状态（如果没有status字段，说明处理成功）
            status = data.get('status')
            if status is not None and status != 'success':
                print(f"⚠️  Preprocessing failed for {audio_stem}: {status}")
                return None
            
            return data
        except Exception as e:
            print(f"⚠️  Error loading preprocessing file {json_path}: {e}")
            return None
    
    def pipeline(
        self,
        waveform: np.array,
        audio_path: Optional[str] = None,
    ) -> Tuple[
        np.array,  # chunks_audio (N, T_chunk)
    ]:
        """
        Extract audio chunks based on preprocessing info or simple chunking.
        
        Args:
            waveform (np.array): Input waveform (1D or 2D numpy array).
            audio_path (str, optional): Path to audio file (required for preprocess mode).
            
        Returns:
            chunks_audio (list): List of audio chunks (N, T_chunk).
        """
        # Ensure waveform is 2D (channels, samples)
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]
        
        T_total = waveform.shape[1]
        
        # Mode 1: Use preprocessing info
        if self.use_preprocess and audio_path:
            preprocess_info = self.load_preprocess_info(audio_path)
            
            if preprocess_info and 'chunks' in preprocess_info:
                chunks = self._extract_chunks_from_preprocess(waveform, preprocess_info['chunks'])
                if chunks:
                    return chunks
                # Fallback to simple chunking if extraction failed
                print(f"⚠️  Falling back to simple chunking for {Path(audio_path).stem}")
        
        # Mode 2: Simple non-overlapping chunking (original behavior)
        return self._extract_simple_chunks(waveform, T_total)
    
    def _extract_simple_chunks(self, waveform: np.array, T_total: int) -> List[np.ndarray]:
        """
        Simple non-overlapping 30s chunking (original behavior).
        """
        chunks: List[np.ndarray] = []
        for start in range(0, T_total, self.chunk_size):
            end = min(start + self.chunk_size, T_total)
            chunk = waveform[:, start:end]

            # pad the last chunk if it's too short
            if chunk.shape[1] < self.chunk_size:
                chunk = F.pad(chunk, (0, self.chunk_size - chunk.shape[1]))

            chunks.append(chunk.squeeze(0).numpy())

        return chunks
    
    def _extract_chunks_from_preprocess(self, waveform: np.array, chunk_infos: list) -> List[np.ndarray]:
        """
        Extract chunks based on preprocessing chunk information.
        
        Args:
            waveform: Audio waveform
            chunk_infos: List of chunk info dicts with 'start', 'end', 'needs_padding'
            
        Returns:
            List of audio chunks (each 30s)
        """
        chunks: List[np.ndarray] = []
        
        for chunk_info in chunk_infos:
            start_sec = chunk_info['start']
            end_sec = chunk_info['end']
            
            # Convert to sample indices
            start_sample = int(start_sec * self.sample_rate)
            end_sample = int(end_sec * self.sample_rate)
            
            # Ensure within bounds
            start_sample = max(0, start_sample)
            end_sample = min(waveform.shape[1], end_sample)
            
            # Extract chunk
            chunk = waveform[:, start_sample:end_sample]
            
            # Pad or truncate to 30s
            if chunk.shape[1] < self.chunk_size:
                # Zero-pad to 30s
                chunk = F.pad(chunk, (0, self.chunk_size - chunk.shape[1]))
            elif chunk.shape[1] > self.chunk_size:
                # Truncate to 30s
                chunk = chunk[:, :self.chunk_size]
            
            chunks.append(chunk.squeeze(0).numpy())
        
        return chunks


# ------------------------------- Runners -------------------------------
@lru_cache(maxsize=8)
def get_cached_vocal_detector(
    sample_rate: int = 16_000,
    chunk_sec: float = 30.0,
    use_preprocess: bool = False,
    preprocess_dir: Optional[str] = None,
) -> VocalDetector:
    """
    Cache detectors keyed by config to avoid reloading the model repeatedly.
    
    Args:
        sample_rate: Audio sample rate
        chunk_sec: Chunk duration in seconds
        use_preprocess: Whether to use preprocessed vocal segments
        preprocess_dir: Directory containing preprocessing JSON files
    """
    return VocalDetector(
        sample_rate=sample_rate,
        chunk_sec=chunk_sec,
        use_preprocess=use_preprocess,
        preprocess_dir=preprocess_dir,
    )


def extract_vocals(
    audio_path: str,
    waveform: torch.Tensor,
    vocal_detector: Optional[VocalDetector],
    sample_rate: Optional[int] = 16_000,
    chunk_sec: Optional[float] = 30.0,
) -> Tuple[
    torch.Tensor,  # chunks_audio (N, T_chunk)
]:
    """
    Extract vocal components from a waveform, on a single audio file.
    
    Args:
        audio_path: Path to the audio file (needed for preprocessing mode)
        waveform: Audio waveform tensor
        vocal_detector: VocalDetector instance
        sample_rate: Audio sample rate
        chunk_sec: Chunk duration in seconds
        
    Returns:
        List of audio chunks
    """

    vocal_detector = vocal_detector or get_cached_vocal_detector(
        sample_rate=sample_rate,
        chunk_sec=chunk_sec,
    )

    return vocal_detector.pipeline(waveform=waveform, audio_path=audio_path)