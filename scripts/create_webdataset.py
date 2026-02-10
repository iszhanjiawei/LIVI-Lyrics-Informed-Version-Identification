#!/usr/bin/env python3
"""
创建 WebDataset 格式的训练数据

用法:
    python scripts/create_webdataset.py \
        --audio-dir data/test_experiment/audio_links \
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
import librosa

def load_audio(audio_path, sample_rate=16000):
    """加载音频文件"""
    waveform_np, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    waveform = torch.from_numpy(waveform_np).float()
    return waveform

def split_audio_30s(waveform, sample_rate=16000, chunk_sec=30.0):
    """将音频分割成30秒块"""
    chunk_size = int(sample_rate * chunk_sec)
    chunks = []
    
    for start in range(0, len(waveform), chunk_size):
        chunk = waveform[start:start + chunk_size]
        if len(chunk) < chunk_size:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_size - len(chunk)))
        chunks.append(chunk)
    
    return chunks

def compute_mel_spectrogram(waveform, n_mels=128, n_fft=400, hop_length=160):
    """计算 Mel 频谱图（与 Whisper 一致）"""
    # 使用 torchaudio 计算 mel 频谱图
    import torchaudio
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )(waveform)
    
    # Log scale
    mel_spec = torch.log10(torch.clamp(mel_spec, min=1e-10))
    
    # Whisper 期望固定长度 3000 帧
    # 30秒音频 @ 16kHz = 480000 samples
    # 480000 / 160 (hop_length) = 3000 帧
    target_length = 3000
    current_length = mel_spec.shape[-1]
    
    if current_length > target_length:
        # 截断
        mel_spec = mel_spec[:, :target_length]
    elif current_length < target_length:
        # 填充
        pad_length = target_length - current_length
        mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_length))
    
    return mel_spec

def load_lyrics_embeddings(npz_path: Path) -> Dict[str, np.ndarray]:
    """加载歌词嵌入文件"""
    data = np.load(npz_path)
    embeddings = {key: data[key] for key in data.files}
    print(f"✓ 加载了 {len(embeddings)} 个歌词嵌入")
    return embeddings

def create_webdataset_shard(
    samples: List[Tuple[str, np.ndarray, np.ndarray]],
    output_path: Path,
    shard_id: int
):
    """创建一个 WebDataset shard (.tar 文件)"""
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
    return len(samples)

def main():
    parser = argparse.ArgumentParser(description="创建 LIVI WebDataset")
    parser.add_argument("--audio-dir", type=Path, required=True, help="音频文件目录")
    parser.add_argument("--lyrics-embeddings", type=Path, required=True, help="歌词嵌入 .npz 文件")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出 WebDataset 目录")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("📦 创建 LIVI WebDataset")
    print("=" * 80)
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 加载歌词嵌入
    print("\n[1/4] 加载歌词嵌入...")
    lyrics_embeddings = load_lyrics_embeddings(args.lyrics_embeddings)
    
    # 获取音频文件列表
    print("\n[2/4] 扫描音频文件...")
    audio_files = sorted(list(args.audio_dir.glob("*.mp3")))
    print(f"✓ 找到 {len(audio_files)} 个音频文件")
    
    # 处理所有音频文件
    print("\n[3/4] 处理音频文件并生成训练样本...")
    all_samples = []
    processed_count = 0
    skipped_count = 0
    
    for audio_path in tqdm(audio_files, desc="处理音频"):
        song_id = audio_path.stem
        
        try:
            # 加载音频
            waveform = load_audio(str(audio_path))
            
            # 分割成 30 秒块
            chunks = split_audio_30s(waveform)
            
            # 为每个 chunk 生成样本
            for i, chunk in enumerate(chunks):
                chunk_key = f"{song_id}_{i}"
                
                # 检查是否有对应的歌词嵌入
                if chunk_key not in lyrics_embeddings:
                    continue
                
                # 计算 Mel 频谱图
                mel = compute_mel_spectrogram(chunk)  # shape: (n_mels, time_frames)
                mel_np = mel.numpy()
                
                # 获取歌词嵌入
                lyrics_emb = lyrics_embeddings[chunk_key]
                
                # 添加样本
                sample_id = f"{song_id}_{i:03d}"
                all_samples.append((sample_id, mel_np, lyrics_emb))
            
            processed_count += 1
            
        except Exception as e:
            print(f"\n  ✗ {audio_path.name}: {e}")
            skipped_count += 1
            continue
    
    print(f"\n✓ 成功处理: {processed_count} 首歌")
    print(f"✓ 跳过: {skipped_count} 首歌")
    print(f"✓ 总样本数: {len(all_samples)}")
    
    if len(all_samples) == 0:
        print("\n❌ 没有生成任何样本，退出")
        return
    
    # 划分训练集、验证集和测试集 (80/10/10)
    print("\n[4/4] 创建 WebDataset 分片...")
    random.shuffle(all_samples)
    
    # 按照论文配置：80% 训练，10% 验证，10% 测试
    total_samples = len(all_samples)
    train_split_idx = int(total_samples * 0.8)
    val_split_idx = int(total_samples * 0.9)
    
    train_samples = all_samples[:train_split_idx]
    val_samples = all_samples[train_split_idx:val_split_idx]
    test_samples = all_samples[val_split_idx:]
    
    print(f"  训练样本: {len(train_samples)} ({len(train_samples)/total_samples*100:.1f}%)")
    print(f"  验证样本: {len(val_samples)} ({len(val_samples)/total_samples*100:.1f}%)")
    print(f"  测试样本: {len(test_samples)} ({len(test_samples)/total_samples*100:.1f}%)")
    
    # 创建训练集分片
    train_count = 0
    if train_samples:
        train_count = create_webdataset_shard(
            train_samples,
            args.output_dir / "train",
            shard_id=0
        )
    
    # 创建验证集分片
    val_count = 0
    if val_samples:
        val_count = create_webdataset_shard(
            val_samples,
            args.output_dir / "val",
            shard_id=0
        )
    
    # 创建测试集分片
    test_count = 0
    if test_samples:
        test_count = create_webdataset_shard(
            test_samples,
            args.output_dir / "test",
            shard_id=0
        )
    
    print("\n" + "=" * 80)
    print("✅ WebDataset 创建完成！")
    print("=" * 80)
    print(f"\n输出目录: {args.output_dir}")
    print(f"  train/shard-000000.tar: {train_count} 样本")
    print(f"  val/shard-000000.tar: {val_count} 样本")
    if test_count > 0:
        print(f"  test/shard-000000.tar: {test_count} 样本")
    
    print(f"\n📊 数据集统计:")
    print(f"  总样本数: {total_samples}")
    print(f"  训练集: {train_count} ({train_count/total_samples*100:.1f}%)")
    print(f"  验证集: {val_count} ({val_count/total_samples*100:.1f}%)")
    print(f"  测试集: {test_count} ({test_count/total_samples*100:.1f}%)")
    print(f"\n符合论文配置 (80/10/10 分割)")

if __name__ == "__main__":
    main()
