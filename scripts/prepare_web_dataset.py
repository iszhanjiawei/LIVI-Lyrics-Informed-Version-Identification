#!/usr/bin/env python3
"""
准备 WebDataset 格式的训练数据

用法:
    python scripts/prepare_test_dataset.py \
        --audio-dir data/test_experiment/raw/audio \
        --lyrics-embeddings data/test_experiment/processed/lyrics_embeddings.npz \
        --output-dir data/test_experiment/webdataset \
        --train-ratio 0.8
"""

import argparse
import io
import numpy as np
import torch
import tarfile
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
import random

from livi.core.data.utils.audio_toolbox import load_audio
from livi.core.data.preprocessing.vocal_detector import get_cached_vocal_detector, extract_vocals
from livi.core.data.preprocessing.whisper_feature_extractor import get_cached_feature_extractor, extract_mel


def load_lyrics_embeddings(npz_path: Path) -> Dict[str, np.ndarray]:
    """加载歌词嵌入文件"""
    data = np.load(npz_path)
    embeddings = {key: data[key] for key in data.files}
    print(f"✓ 加载了 {len(embeddings)} 个歌词嵌入")
    return embeddings


def process_audio_file(
    audio_path: Path,
    lyrics_embedding: np.ndarray,
    vocal_detector,
    feature_extractor,
    sr: int = 16000
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    处理单个音频文件，生成训练样本
    
    返回: List[(mel_spectrogram, lyrics_embedding)]
    """
    samples = []
    
    try:
        # 加载音频
        waveform = load_audio(str(audio_path), target_sample_rate=sr)
        
        # 提取人声片段（30秒 chunks）
        chunks_audio = extract_vocals(str(audio_path), waveform, vocal_detector)
        
        if chunks_audio is None or len(chunks_audio) == 0:
            print(f"  ⚠️  {audio_path.name}: 未检测到人声片段")
            return samples
        
        # 为每个 chunk 生成 mel 频谱图
        mel = extract_mel(chunks_audio, feature_extractor)
        
        # mel shape: (num_chunks, 80, time_frames)
        num_chunks = mel.shape[0]
        
        # 如果 lyrics_embedding 是 (1, 768)，复制到所有 chunks
        if lyrics_embedding.shape[0] == 1:
            target_embeddings = np.repeat(lyrics_embedding, num_chunks, axis=0)
        else:
            # 如果已经有多个 chunk 的嵌入，直接使用
            target_embeddings = lyrics_embedding
        
        # 创建样本对
        for i in range(num_chunks):
            mel_chunk = mel[i].cpu().numpy()  # (80, time_frames)
            target_chunk = target_embeddings[min(i, len(target_embeddings)-1)]  # (768,)
            samples.append((mel_chunk, target_chunk))
        
        print(f"  ✓ {audio_path.name}: 生成 {num_chunks} 个训练样本")
        
    except Exception as e:
        print(f"  ✗ {audio_path.name}: 处理失败 - {e}")
    
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
    parser.add_argument("--audio-dir", type=Path, required=True, help="音频文件目录")
    parser.add_argument("--lyrics-embeddings", type=Path, required=True, help="歌词嵌入 .npz 文件")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出 WebDataset 目录")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--sr", type=int, default=16000, help="音频采样率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LIVI WebDataset 数据准备")
    print("=" * 60)
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 加载歌词嵌入
    print("\n[1/5] 加载歌词嵌入...")
    lyrics_embeddings = load_lyrics_embeddings(args.lyrics_embeddings)
    
    # 获取音频文件列表
    print("\n[2/5] 扫描音频文件...")
    audio_files = list(args.audio_dir.glob("*.mp3")) + list(args.audio_dir.glob("*.wav"))
    audio_files = sorted(audio_files)
    print(f"✓ 找到 {len(audio_files)} 个音频文件")
    
    # 初始化预处理模型
    print("\n[3/5] 初始化预处理模型...")
    vocal_detector = get_cached_vocal_detector(sample_rate=args.sr, chunk_sec=30.0)
    feature_extractor = get_cached_feature_extractor(
        sample_rate=args.sr,
        model_name="openai/whisper-large-v3-turbo"
    )
    print("✓ 模型加载完成")
    
    # 处理所有音频文件
    print("\n[4/5] 处理音频文件...")
    all_samples = []
    
    for audio_path in tqdm(audio_files, desc="处理音频"):
        # 获取对应的歌词嵌入
        stem = audio_path.stem
        if stem not in lyrics_embeddings:
            print(f"  ⚠️  {audio_path.name}: 找不到对应的歌词嵌入，跳过")
            continue
        
        lyrics_emb = lyrics_embeddings[stem]
        
        # 处理音频，生成训练样本
        samples = process_audio_file(
            audio_path,
            lyrics_emb,
            vocal_detector,
            feature_extractor,
            sr=args.sr
        )
        
        # 添加样本 ID
        for i, (mel, target) in enumerate(samples):
            sample_id = f"{stem}_{i:03d}"
            all_samples.append((sample_id, mel, target))
    
    print(f"\n✓ 总共生成 {len(all_samples)} 个训练样本")
    
    # 划分训练集和验证集
    print("\n[5/5] 创建 WebDataset 分片...")
    random.shuffle(all_samples)
    
    split_idx = int(len(all_samples) * args.train_ratio)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    print(f"  训练样本: {len(train_samples)}")
    print(f"  验证样本: {len(val_samples)}")
    
    # 创建训练集分片
    if train_samples:
        create_webdataset_shard(
            train_samples,
            args.output_dir / "train",
            shard_id=0
        )
    
    # 创建验证集分片
    if val_samples:
        create_webdataset_shard(
            val_samples,
            args.output_dir / "val",
            shard_id=0
        )
    
    print("\n" + "=" * 60)
    print("✓ 数据准备完成！")
    print("=" * 60)
    print(f"\n输出目录: {args.output_dir}")
    print(f"  train/shard-000000.tar ({len(train_samples)} 样本)")
    print(f"  val/shard-000000.tar ({len(val_samples)} 样本)")
    print("\n下一步: 修改训练配置文件中的 data_dir 和 total_*_samples")


if __name__ == "__main__":
    main()
