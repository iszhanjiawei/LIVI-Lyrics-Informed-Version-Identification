"""
Audio toolbox utilities.

Reusable helpers for loading, resampling, and preprocessing audio...
"""

import torch
import torchaudio
import torchaudio.transforms as T


def load_audio(
    path: str,
    target_sample_rate: int = 16_000,
    mono: bool = True,
) -> torch.Tensor:
    """
    Load an audio file, convert to mono, and resample.

    Args:
        path (str): Path to the audio file (e.g. .mp3, .wav).
        target_sample_rate (int): Desired sample rate (Hz). Default = 16k.
        mono (bool): If True, convert multi-channel audio to mono.

    Returns:
        torch.Tensor: Waveform tensor of shape (T,), resampled and optionally mono.
    """
    # Load waveform
    waveform, sr = torchaudio.load_with_torchcodec(path)  # shape (C, T)

    # Convert to mono if multiple channels
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)

    # Resample if needed
    if sr != target_sample_rate:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    return waveform


def split_audio_30s(
    waveform: torch.Tensor,
    sample_rate: int = 16_000,
    chunk_sec: float = 30.0,
) -> list[torch.Tensor]:
    """
    将音频分割成 30 秒的块（非重叠）。
    
    Args:
        waveform: 音频张量，形状 (T,)
        sample_rate: 采样率
        chunk_sec: 块的长度（秒）
    
    Returns:
        音频块列表
    """
    chunk_size = int(sample_rate * chunk_sec)
    chunks = []
    
    for start in range(0, len(waveform), chunk_size):
        chunk = waveform[start:start + chunk_size]
        # 如果块小于 chunk_size，用零填充
        if len(chunk) < chunk_size:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_size - len(chunk)))
        chunks.append(chunk)
    
    return chunks


def split_audio_predefined(
    waveform: torch.Tensor,
    segments: list[tuple[float, float]],
    sample_rate: int = 16_000,
    chunk_sec: float = 30.0,
) -> list[torch.Tensor]:
    """
    根据预定义的时间段分割音频。
    
    Args:
        waveform: 音频张量，形状 (T,)
        segments: 时间段列表 [(start_sec, end_sec), ...]
        sample_rate: 采样率
        chunk_sec: 目标块长度（秒）
    
    Returns:
        音频块列表
    """
    chunk_size = int(sample_rate * chunk_sec)
    chunks = []
    
    for start_sec, end_sec in segments:
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)
        
        chunk = waveform[start_sample:end_sample]
        
        # 截断或填充到 chunk_size
        if len(chunk) > chunk_size:
            chunk = chunk[:chunk_size]
        elif len(chunk) < chunk_size:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_size - len(chunk)))
        
        chunks.append(chunk)
    
    return chunks
