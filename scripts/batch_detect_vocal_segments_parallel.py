#!/usr/bin/env python3
"""
并行批量人声检测预处理脚本（优化版，支持10万+歌曲）
复用 batch_detect_vocal_segments.py 的核心功能，只添加并行处理逻辑

特性：
- 多进程并行处理
- 断点续传（跳过已处理文件）
- 实时进度监控

用法:
    conda activate musicnn_env
    python scripts/batch_detect_vocal_segments_parallel.py \
        --audio-list data/train_data/audio_links/train_songs_audio_paths.txt \
        --output-dir data/train_data/vocal_segments \
        --num-workers 96 \
        --min-global-vocalness 0.3 \
        --vocal-threshold 0.5 \
        --padding-sec 1.0
"""

import sys
import json
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time
from datetime import datetime
import argparse

# 导入已有脚本的函数（复用代码，不重复实现）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "scripts"))

# 直接导入原脚本的核心处理函数
from batch_detect_vocal_segments import process_single_audio as _process_audio_core


def process_task_wrapper(task):
    """
    多进程任务包装器
    负责断点续传检查，然后调用原脚本的 process_single_audio 函数
    """
    audio_path, output_dir, args = task
    audio_file = Path(audio_path)
    
    # 检查输出文件是否已存在（断点续传）
    output_file = output_dir / f"{audio_file.stem}.json"
    if output_file.exists():
        return {
            'audio_name': audio_file.stem,
            'status': 'skipped'
        }
    
    # 检查文件是否存在
    if not audio_file.exists():
        return {
            'audio_name': audio_file.stem,
            'status': 'file_not_found'
        }
    
    # 调用原脚本的核心处理函数
    result = _process_audio_core(audio_file, args)
    
    # 如果处理成功，保存结果到 JSON
    if result['status'] == 'success':
        output_data = {
            'global_vocalness': result.get('global_vocalness'),
            'segments': result.get('segments', []),
            'chunks': result.get('chunks', []),
            'config': result.get('config', {})
        }
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return {
        'audio_name': audio_file.stem,
        'status': result['status'],
        'global_vocalness': result.get('global_vocalness'),
        'num_segments': len(result.get('segments', [])),
        'num_chunks': len(result.get('chunks', []))
    }


def main():
    parser = argparse.ArgumentParser(description="并行批量人声检测预处理")
    parser.add_argument('--audio-dir', type=str, required=False,
                        help='音频文件目录')
    parser.add_argument('--audio-list', type=str, required=False,
                        help='包含音频文件路径的文本文件（每行一个路径）')
    parser.add_argument("--output-dir", type=str, required=True,
                       help="输出目录（每个音频生成一个 JSON）")
    parser.add_argument("--num-workers", type=int, default=None,
                       help=f"并行进程数（默认: CPU核心数={cpu_count()}）")
    parser.add_argument("--min-global-vocalness", type=float, default=0.3,
                       help="最小全局人声分数阈值")
    parser.add_argument("--vocal-threshold", type=float, default=0.5,
                       help="窗口人声判定阈值")
    parser.add_argument("--padding-sec", type=float, default=1.0,
                       help="对称填充秒数（论文最多 10 秒）")
    parser.add_argument("--summary-json", type=str,
                       help="可选：汇总结果 JSON 文件")
    
    args = parser.parse_args()
    
    if not args.audio_dir and not args.audio_list:
        print("❌ 必须指定 --audio-dir 或 --audio-list 之一")
        return
    
    if args.audio_dir and args.audio_list:
        print("❌ --audio-dir 和 --audio-list 不能同时指定")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置并行进程数
    num_workers = args.num_workers or cpu_count()
    
    print("=" * 80)
    print("🎵 并行批量人声检测预处理（Musicnn）")
    print("=" * 80)
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 获取所有音频文件
    if args.audio_list:
        list_file = Path(args.audio_list)
        if not list_file.exists():
            print(f"❌ 文件列表不存在: {list_file}")
            return
        
        print(f"\n📄 音频列表文件: {list_file}")
        with open(list_file, 'r', encoding='utf-8') as f:
            audio_paths = [line.strip() for line in f if line.strip()]
        audio_files = [Path(p) for p in audio_paths]
        existing_files = [f for f in audio_files if f.exists()]
        missing_count = len(audio_files) - len(existing_files)
        if missing_count > 0:
            print(f"⚠️  {missing_count} 个文件不存在，将标记为错误")
    else:
        audio_dir = Path(args.audio_dir)
        if not audio_dir.exists():
            print(f"❌ 目录不存在: {audio_dir}")
            return
        print(f"\n📁 音频目录: {audio_dir}")
        audio_files = (list(audio_dir.glob("*.mp3")) + 
                       list(audio_dir.glob("*.wav")) + 
                       list(audio_dir.glob("*.m4a")))
        audio_files = sorted(audio_files)
    
    print(f"📁 输出目录: {output_dir}")
    print(f"\n⚙️  参数:")
    print(f"   - 并行进程数: {num_workers}")
    print(f"   - 最小全局人声分数: {args.min_global_vocalness}")
    print(f"   - 窗口人声阈值: {args.vocal_threshold}")
    print(f"   - 对称填充: {args.padding_sec} 秒")
    
    print(f"\n✓ 总共 {len(audio_files)} 个音频文件")
    
    if len(audio_files) == 0:
        print("❌ 没有找到音频文件")
        return
    
    # 检查已处理的文件
    existing_outputs = set(p.stem for p in output_dir.glob("*.json"))
    print(f"✓ 已处理 {len(existing_outputs)} 个文件（将跳过）")
    
    print("\n" + "=" * 80)
    print("开始并行处理...")
    print("=" * 80)
    
    # 准备任务
    tasks = [(str(audio_file), output_dir, args) for audio_file in audio_files]
    
    # 统计信息
    stats = {
        'total': len(audio_files),
        'success': 0,
        'low_vocalness': 0,
        'no_segments': 0,
        'musicnn_error': 0,
        'file_not_found': 0,
        'skipped': 0,
        'error': 0
    }
    
    results_detail = {}
    start_time = time.time()
    
    # 多进程处理
    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(tasks), desc="处理进度", unit="首") as pbar:
            for result in pool.imap_unordered(process_task_wrapper, tasks):
                # 更新统计
                status = result['status']
                stats[status] = stats.get(status, 0) + 1
                
                if status == 'success':
                    results_detail[result['audio_name']] = {
                        'status': 'success',
                        'global_vocalness': result.get('global_vocalness'),
                        'num_segments': result.get('num_segments'),
                        'num_chunks': result.get('num_chunks')
                    }
                else:
                    results_detail[result['audio_name']] = {
                        'status': status
                    }
                
                pbar.update(1)
                
                # 每1000个更新一次描述
                if pbar.n % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = pbar.n / elapsed
                    eta = (len(tasks) - pbar.n) / rate if rate > 0 else 0
                    pbar.set_postfix({
                        '成功': stats['success'],
                        '速度': f'{rate:.1f}首/秒',
                        'ETA': f'{eta/3600:.1f}h'
                    })
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("✅ 处理完成！")
    print("=" * 80)
    print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  总耗时: {elapsed_time/3600:.2f} 小时")
    print(f"⚡ 平均速度: {len(tasks)/elapsed_time:.2f} 首/秒")
    print(f"\n📊 统计:")
    print(f"   - 总数: {stats['total']}")
    print(f"   - 成功: {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
    print(f"   - 人声分数过低: {stats.get('low_vocalness', 0)}")
    print(f"   - 无人声片段: {stats.get('no_segments', 0)}")
    print(f"   - Musicnn 错误: {stats.get('musicnn_error', 0)}")
    print(f"   - 文件未找到: {stats.get('file_not_found', 0)}")
    print(f"   - 跳过（已处理）: {stats.get('skipped', 0)}")
    print(f"   - 其他错误: {stats.get('error', 0)}")
    
    # 保存汇总
    if args.summary_json:
        summary = {
            'stats': stats,
            'config': {
                'min_global_vocalness': args.min_global_vocalness,
                'vocal_threshold': args.vocal_threshold,
                'padding_sec': args.padding_sec,
                'num_workers': num_workers
            },
            'time': {
                'start': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'elapsed_hours': elapsed_time / 3600,
                'rate_per_second': len(tasks) / elapsed_time
            },
            'results': results_detail
        }
        
        summary_file = Path(args.summary_json)
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 汇总文件: {summary_file}")


if __name__ == "__main__":
    main()
