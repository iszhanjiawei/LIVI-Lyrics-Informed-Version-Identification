#!/usr/bin/env python3
"""
并行生成歌词嵌入 - 支持指定GPU和数据分片
复用 run_inference 函数，支持断点续传
"""

import sys
import os
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from livi.apps.frozen_encoder.infer.session import run_inference_from_list

def main():
    parser = argparse.ArgumentParser(description='生成歌词嵌入（支持GPU分片）')
    parser.add_argument('--gpu-id', type=int, required=True, help='GPU ID (0 or 1)')
    parser.add_argument('--start-idx', type=int, required=True, help='开始索引')
    parser.add_argument('--end-idx', type=int, required=True, help='结束索引（不包含）')
    args = parser.parse_args()
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    # 减少 CUDA 内存碎片化，使各进程显存更一致
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 配置文件（使用训练数据准备配置）
    config_path = project_root / "src/livi/apps/frozen_encoder/config/infer_for_training.yaml"
    
    # 读取音频路径列表（中文歌曲）
    audio_list_file = Path("/home/zjw524/projects/LIVI_project_data/data/train_data/training_audio_paths_chinese.txt")
    with open(audio_list_file, 'r', encoding='utf-8') as f:
        all_audio_paths = [line.strip() for line in f if line.strip()]
    
    # 分片数据
    audio_paths = all_audio_paths[args.start_idx:args.end_idx]
    
    print(f"=" * 80)
    print(f"GPU {args.gpu_id} 开始处理")
    print(f"=" * 80)
    print(f"总数据量: {len(all_audio_paths)}")
    print(f"当前分片: [{args.start_idx}:{args.end_idx}]")
    print(f"处理数量: {len(audio_paths)}")
    print(f"配置文件: {config_path}")
    print("=" * 80)
    
    # 输出目录（中文歌曲嵌入）
    output_dir = Path("/home/zjw524/projects/LIVI_project_data/data/train_data/chinese_embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"segment_embeddings_gpu{args.gpu_id}_{args.start_idx}_{args.end_idx}.npz"
    
    print(f"\n输出文件: {output_file}")
    print("\n" + "=" * 80)
    print("开始生成歌词嵌入...")
    print("=" * 80 + "\n")
    
    # 直接使用文件路径列表运行推理（避免创建符号链接）
    audio_path_objects = [Path(p) for p in audio_paths]
    run_inference_from_list(
        config_path=config_path,
        audio_paths=audio_path_objects,
        path_out=output_file
    )
    
    print("\n" + "=" * 80)
    print(f"✅ GPU {args.gpu_id} 处理完成！")
    print(f"📁 输出文件: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
