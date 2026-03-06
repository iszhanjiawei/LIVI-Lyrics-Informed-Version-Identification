#!/usr/bin/env python3
"""
准备 WebDataset 格式的训练数据

核心逻辑：
  1. 读取 JSON 预处理文件获取每首歌的 chunk 边界（start/end 时间）
  2. 读取 chunk 级别的歌词嵌入（key 格式：songname_chunkindex）
  3. 对于每个 chunk，只有当它在歌词嵌入中存在时才构成训练样本
  4. mel 频谱图根据 JSON 中的 chunk 时间信息从原始音频中提取

用法:
    python scripts/prepare_web_dataset.py \
        --audio-list /path/to/training_audio_paths.txt \
        --lyrics-embeddings /path/to/lyrics_embeddings.npz \
        --preprocess-dir /path/to/vocal_segments \
        --output-dir /path/to/webdataset \
        --samples-per-shard 1000
"""

import argparse
import io
import json
import numpy as np
import torch
import torch.nn.functional as F
import tarfile
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import random

from livi.core.data.utils.audio_toolbox import load_audio
from livi.core.data.preprocessing.whisper_feature_extractor import get_cached_feature_extractor, extract_mel


def load_lyrics_embeddings(npz_paths: List[Path]) -> Dict[str, np.ndarray]:
    """
    加载歌词嵌入文件（支持多个 .npz 文件）
    
    歌词嵌入的 key 格式为 chunk 级别：songname_chunkindex
    例如："信仰 - 张信哲_0", "信仰 - 张信哲_1", "信仰 - 张信哲_5"
    """
    all_embeddings = {}
    for npz_path in npz_paths:
        data = np.load(npz_path)
        all_embeddings.update({key: data[key] for key in data.files})
        data.close()
    print(f"✓ 加载了 {len(all_embeddings)} 个 chunk 级别歌词嵌入（来自 {len(npz_paths)} 个文件）")
    return all_embeddings


def load_preprocess_json(preprocess_dir: Path, song_stem: str) -> Optional[dict]:
    """
    加载歌曲的预处理 JSON 文件，获取 chunk 边界信息
    
    Returns:
        dict with 'chunks' list, each containing 'start', 'end', 'needs_padding' etc.
        None if JSON file not found
    """
    json_path = preprocess_dir / f"{song_stem}.json"
    if not json_path.exists():
        return None
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_chunk_audio(
    waveform: np.ndarray,
    chunk_info: dict,
    sr: int = 16000,
    chunk_sec: float = 30.0
) -> np.ndarray:
    """
    根据 JSON 中的 chunk 时间信息从原始音频中提取单个 chunk
    
    Args:
        waveform: 原始音频波形 (samples,)
        chunk_info: JSON 中的 chunk 信息 dict，包含 'start', 'end', 'needs_padding'
        sr: 采样率
        chunk_sec: chunk 目标长度（秒）
    
    Returns:
        np.ndarray: 30 秒的音频 chunk（如有需要会 zero-pad）
    """
    chunk_size = int(sr * chunk_sec)
    start_sample = int(chunk_info['start'] * sr)
    end_sample = int(chunk_info['end'] * sr)
    
    # 确保在范围内
    start_sample = max(0, start_sample)
    end_sample = min(len(waveform), end_sample)
    
    chunk = waveform[start_sample:end_sample]
    
    # Pad 或 truncate 到 30 秒
    if len(chunk) < chunk_size:
        chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
    elif len(chunk) > chunk_size:
        chunk = chunk[:chunk_size]
    
    return chunk


def process_audio_file(
    audio_path: Path,
    preprocess_dir: Path,
    lyrics_embeddings: Dict[str, np.ndarray],
    feature_extractor,
    sr: int = 16000
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """
    处理单个音频文件，基于 JSON 预处理信息和 chunk 级别歌词嵌入生成训练样本
    
    流程：
      1. 读取 JSON → 获取每个 chunk 的起止时间
      2. 遍历每个 chunk，检查 chunk 级别 key 是否存在于歌词嵌入中
      3. 仅为有匹配歌词嵌入的 chunk 生成 mel 频谱图 → 构成样本对
    
    Returns:
        List[(sample_id, mel_spectrogram, lyrics_embedding)]
    """
    samples = []
    stem = audio_path.stem
    
    try:
        # 1. 读取 JSON 预处理信息
        preprocess_info = load_preprocess_json(preprocess_dir, stem)
        if preprocess_info is None or 'chunks' not in preprocess_info:
            return samples
        
        chunk_infos = preprocess_info['chunks']
        
        # 2. 找出哪些 chunk 有对应的歌词嵌入
        matched_chunks = []
        for chunk_idx, chunk_info in enumerate(chunk_infos):
            chunk_key = f"{stem}_{chunk_idx}"
            if chunk_key in lyrics_embeddings:
                matched_chunks.append((chunk_idx, chunk_info, lyrics_embeddings[chunk_key]))
        
        if not matched_chunks:
            return samples
        
        # 3. 加载音频（只在有匹配的 chunk 时才加载，避免浪费）
        waveform = load_audio(str(audio_path), target_sample_rate=sr)
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()
        if waveform.ndim == 2:
            waveform = waveform[0]  # 取第一个通道
        
        # 4. 对每个匹配的 chunk 提取音频并计算 mel 频谱图
        chunk_audios = []
        chunk_keys = []
        chunk_embs = []
        
        for chunk_idx, chunk_info, chunk_emb in matched_chunks:
            chunk_audio = extract_chunk_audio(waveform, chunk_info, sr=sr)
            chunk_audios.append(chunk_audio)
            chunk_keys.append(f"{stem}_{chunk_idx:03d}")
            chunk_embs.append(chunk_emb)
        
        # 批量计算 mel 频谱图
        mel = extract_mel(chunk_audios, feature_extractor)
        
        # 构建样本对
        for j in range(len(chunk_keys)):
            mel_chunk = mel[j].cpu().numpy()  # (128, time_frames)
            samples.append((chunk_keys[j], mel_chunk, chunk_embs[j]))
        
    except Exception as e:
        print(f"\n  ✗ {audio_path.name}: 处理失败 - {e}")
    
    return samples


def create_webdataset_shard(
    samples: List[Tuple[str, np.ndarray, np.ndarray]],
    output_path: Path,
    shard_id: int
):
    """
    创建一个 WebDataset shard (.tar 文件)
    
    Args:
        samples: List of (sample_id, mel_spectrogram, lyrics_embedding)
        output_path: 输出目录
        shard_id: 分片编号
    """
    shard_name = f"shard-{shard_id:06d}.tar"
    shard_path = output_path / shard_name
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    with tarfile.open(shard_path, "w") as tar:
        for sample_id, mel, target in samples:
            # 保存 mel 频谱图
            mel_bytes = io.BytesIO()
            np.save(mel_bytes, mel)
            mel_bytes.seek(0)
            
            mel_info = tarfile.TarInfo(name=f"{sample_id}.features.npy")
            mel_info.size = len(mel_bytes.getvalue())
            tar.addfile(mel_info, mel_bytes)
            
            # 保存歌词嵌入
            target_bytes = io.BytesIO()
            np.save(target_bytes, target)
            target_bytes.seek(0)
            
            target_info = tarfile.TarInfo(name=f"{sample_id}.text.npy")
            target_info.size = len(target_bytes.getvalue())
            tar.addfile(target_info, target_bytes)
    
    print(f"✓ 创建分片: {shard_name} ({len(samples)} 个样本)")


def main():
    parser = argparse.ArgumentParser(description="准备 LIVI WebDataset 训练数据")
    parser.add_argument("--audio-list", type=Path, required=True,
                        help="音频文件路径列表（.txt，每行一个路径）")
    parser.add_argument("--lyrics-embeddings", type=Path, nargs='+', required=True,
                        help="歌词嵌入 .npz 文件（支持多个，会自动合并）")
    parser.add_argument("--preprocess-dir", type=Path, required=True,
                        help="预处理 JSON 文件目录（vocal_segments）")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="输出 WebDataset 目录")
    parser.add_argument("--sr", type=int, default=16000, help="音频采样率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--samples-per-shard", type=int, default=1000,
                        help="每个 shard 的样本数量")
    parser.add_argument("--whisper-model", type=str,
                        default="/home/zjw524/projects/LIVI-Lyrics-Informed-Version-Identification/pretrained_models/whisper-large-v3-turbo",
                        help="Whisper 模型路径（用于特征提取）")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LIVI WebDataset 数据准备（基于 JSON 预处理 + chunk 级别对齐）")
    print("=" * 60)
    
    if not args.preprocess_dir.exists():
        print(f"❌ 错误: 预处理目录不存在: {args.preprocess_dir}")
        return
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # [1/5] 加载歌词嵌入（chunk 级别 key）
    print("\n[1/5] 加载歌词嵌入...")
    lyrics_embeddings = load_lyrics_embeddings(args.lyrics_embeddings)
    
    # [2/5] 读取音频文件列表
    print("\n[2/5] 读取音频文件列表...")
    with open(args.audio_list, 'r') as f:
        audio_paths = [Path(line.strip()) for line in f if line.strip()]
    print(f"✓ 共 {len(audio_paths)} 个音频文件")
    
    # [3/5] 初始化 Whisper 特征提取器
    print("\n[3/5] 初始化 Whisper 特征提取器...")
    feature_extractor = get_cached_feature_extractor(
        sample_rate=args.sr,
        model_name=args.whisper_model
    )
    print(f"✓ 特征提取器加载完成: {args.whisper_model}")
    print(f"✓ 预处理目录: {args.preprocess_dir}")
    
    # [4/5] 处理所有音频文件
    print("\n[4/5] 处理音频文件（基于 JSON chunk 信息对齐）...")
    all_samples = []
    matched_songs = 0
    skipped_songs = 0
    
    for audio_path in tqdm(audio_paths, desc="处理音频"):
        samples = process_audio_file(
            audio_path,
            args.preprocess_dir,
            lyrics_embeddings,
            feature_extractor,
            sr=args.sr
        )
        
        if samples:
            all_samples.extend(samples)
            matched_songs += 1
        else:
            skipped_songs += 1
    
    print(f"\n✓ 成功匹配: {matched_songs} 首歌")
    print(f"  跳过（无匹配/无JSON）: {skipped_songs} 首歌")
    print(f"  总样本数: {len(all_samples)}")
    
    if len(all_samples) == 0:
        print("\n❌ 没有生成任何样本，退出")
        return
    
    # [5/5] 划分并创建 WebDataset 分片
    print("\n[5/5] 划分数据集并创建 WebDataset 分片...")
    random.shuffle(all_samples)
    
    # 80/10/10 分割
    total = len(all_samples)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    
    splits = {
        "train": all_samples[:train_end],
        "val": all_samples[train_end:val_end],
        "test": all_samples[val_end:],
    }
    
    for split_name, split_samples in splits.items():
        if not split_samples:
            continue
        print(f"\n  [{split_name}] {len(split_samples)} 样本 ({len(split_samples)/total*100:.1f}%)")
        
        # 按 samples_per_shard 分片
        split_dir = args.output_dir / split_name
        for shard_id in range(0, len(split_samples), args.samples_per_shard):
            shard_samples = split_samples[shard_id:shard_id + args.samples_per_shard]
            create_webdataset_shard(shard_samples, split_dir, shard_id // args.samples_per_shard)
    
    print("\n" + "=" * 60)
    print("✅ WebDataset 创建完成！")
    print("=" * 60)
    print(f"\n输出目录: {args.output_dir}")
    for split_name, split_samples in splits.items():
        n_shards = (len(split_samples) + args.samples_per_shard - 1) // args.samples_per_shard if split_samples else 0
        print(f"  {split_name}/: {len(split_samples)} 样本, {n_shards} 个 shard")
    print(f"\n下一步: 修改训练配置文件中的 data_dir, total_*_samples, last_shard_*")


if __name__ == "__main__":
    main()
