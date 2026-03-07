#!/usr/bin/env python3
"""
批量人声检测预处理脚本
在 musicnn_env 环境下运行
基于 step1_detect_vocal_segments.py 的逻辑

用法:
    conda activate musicnn_env
    python scripts/batch_detect_vocal_segments.py \
        --audio-dir data/test_experiment/audio_links \
        --output-dir data/test_experiment/vocal_segments \
        --min-global-vocalness 0.3 \
        --vocal-threshold 0.5 \
        --padding-sec 1.0
"""

import sys
import argparse
import json
from pathlib import Path
import numpy as np
import librosa
from tqdm import tqdm

# 添加 musicnn 到路径
project_root = Path(__file__).parent.parent
musicnn_path = project_root / "musicnn"
sys.path.insert(0, str(musicnn_path))

from musicnn.extractor import extractor


def estimate_vocalness_musicnn(audio_path, window_sec=3.0):
    """
    使用 Musicnn 估计每个窗口的人声概率
    参考自 step1_detect_vocal_segments.py
    """
    try:
        taggram, tags = extractor(
            audio_path,
            model='MTT_musicnn',
            input_length=window_sec,
            input_overlap=0,
            extract_features=False
        )
        
        vocal_tags = ['vocal', 'voice', 'singer', 'singing', 'vocals', 
                      'female voice', 'male voice', 'speech', 'choir']
        no_vocal_tags = ['no vocal', 'no vocals', 'instrumental']
        
        vocal_indices = [i for i, tag in enumerate(tags) if any(vt in tag.lower() for vt in vocal_tags)]
        no_vocal_indices = [i for i, tag in enumerate(tags) if any(nvt in tag.lower() for nvt in no_vocal_tags)]
        
        window_vocalness = []
        for i in range(taggram.shape[0]):
            if vocal_indices:
                vocal_prob = np.sum(taggram[i, vocal_indices])
            else:
                if no_vocal_indices:
                    vocal_prob = 1.0 - np.sum(taggram[i, no_vocal_indices])
                else:
                    vocal_prob = 0.5
            
            vocal_prob = np.clip(vocal_prob, 0, 1)
            
            window_vocalness.append({
                'window_id': i,
                'time_start': i * window_sec,
                'time_end': (i + 1) * window_sec,
                'vocalness': float(vocal_prob)
            })
        
        global_vocalness = float(np.mean([w['vocalness'] for w in window_vocalness]))
        
        return window_vocalness, global_vocalness
        
    except Exception as e:
        print(f"    ❌ Musicnn 处理失败: {e}")
        return None, None


def extract_vocal_segments(window_vocalness, vocal_threshold=0.5, padding_sec=1.0):
    """
    从窗口人声概率中提取连续的人声片段
    增加了对称填充（论文 4.1 节要求）
    """
    if not window_vocalness:
        return []
    
    vocal_windows = [w for w in window_vocalness if w['vocalness'] >= vocal_threshold]
    
    if not vocal_windows:
        return []
    
    segments = []
    current_start = vocal_windows[0]['time_start']
    current_end = vocal_windows[0]['time_end']
    current_vocalness = [vocal_windows[0]['vocalness']]
    
    for i in range(1, len(vocal_windows)):
        if vocal_windows[i]['window_id'] == vocal_windows[i-1]['window_id'] + 1:
            current_end = vocal_windows[i]['time_end']
            current_vocalness.append(vocal_windows[i]['vocalness'])
        else:
            segments.append({
                'start': current_start,
                'end': current_end,
                'vocalness': float(np.mean(current_vocalness))
            })
            current_start = vocal_windows[i]['time_start']
            current_end = vocal_windows[i]['time_end']
            current_vocalness = [vocal_windows[i]['vocalness']]
    
    segments.append({
        'start': current_start,
        'end': current_end,
        'vocalness': float(np.mean(current_vocalness))
    })
    
    # 对称填充（论文 4.1 节：最多填充 10 秒）
    padded_segments = []
    for seg in segments:
        padded_start = max(0, seg['start'] - padding_sec)
        padded_end = seg['end'] + padding_sec
        padded_segments.append({
            'start': padded_start,
            'end': padded_end,
            'vocalness': seg['vocalness'],
            'original_start': seg['start'],
            'original_end': seg['end']
        })
    
    return padded_segments


def compute_30s_chunks(duration, segments, chunk_sec=30.0):
    """
    计算 30 秒 chunks 的时间范围（论文要求固定 30 秒）
    """
    chunks = []
    
    for seg_idx, seg in enumerate(segments):
        segment_duration = seg['end'] - seg['start']
        
        if segment_duration >= chunk_sec:
            # 切成多个 30 秒 chunks
            num_chunks = int(np.ceil(segment_duration / chunk_sec))
            for i in range(num_chunks):
                chunk_start = seg['start'] + i * chunk_sec
                chunk_end = min(chunk_start + chunk_sec, seg['end'])
                
                # 只保留至少一半长度的 chunk
                if chunk_end - chunk_start >= chunk_sec / 2:
                    chunks.append({
                        'segment_id': seg_idx,
                        'chunk_id': i,
                        'start': chunk_start,
                        'end': chunk_end,
                        'duration': chunk_end - chunk_start,
                        'vocalness': seg['vocalness'],
                        'needs_padding': chunk_end - chunk_start < chunk_sec
                    })
        else:
            # 整个片段作为一个 chunk（会被零填充到 30 秒）
            chunks.append({
                'segment_id': seg_idx,
                'chunk_id': 0,
                'start': seg['start'],
                'end': seg['end'],
                'duration': segment_duration,
                'vocalness': seg['vocalness'],
                'needs_padding': True
            })
    
    return chunks


def process_single_audio(audio_path, args):
    """处理单个音频文件"""
    result = {
        'status': 'error',
        'audio_path': str(audio_path),
        'audio_name': audio_path.name,
    }
    
    try:
        # 1. 获取音频时长
        duration = librosa.get_duration(path=str(audio_path))
        result['duration'] = duration
        
        # 2. 使用 Musicnn 检测人声窗口
        window_vocalness, global_vocalness = estimate_vocalness_musicnn(str(audio_path))
        
        if window_vocalness is None:
            result['status'] = 'musicnn_error'
            return result
        
        result['global_vocalness'] = global_vocalness
        result['window_vocalness'] = window_vocalness
        
        # 3. 检查全局人声分数
        if global_vocalness < args.min_global_vocalness:
            result['status'] = 'low_vocalness'
            return result
        
        # 4. 提取人声片段
        segments = extract_vocal_segments(
            window_vocalness,
            vocal_threshold=args.vocal_threshold,
            padding_sec=args.padding_sec
        )
        
        if not segments:
            result['status'] = 'no_segments'
            return result
        
        result['segments'] = segments
        
        # 5. 计算 30 秒 chunks
        chunks = compute_30s_chunks(duration, segments)
        result['chunks'] = chunks
        
        result['status'] = 'success'
        result['config'] = {
            'min_global_vocalness': args.min_global_vocalness,
            'vocal_threshold': args.vocal_threshold,
            'padding_sec': args.padding_sec,
            'chunk_sec': 30.0,
            'window_sec': 3.0
        }
        
    except Exception as e:
        result['status'] = 'error'
        result['error_message'] = str(e)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="批量人声检测预处理")
    parser.add_argument('--audio-dir', type=str, required=False,
                        help='音频文件目录')
    parser.add_argument('--audio-list', type=str, required=False,
                        help='包含音频文件路径的文本文件（每行一个路径）')
    parser.add_argument("--output-dir", type=str, required=True,
                       help="输出目录（每个音频生成一个 JSON）")
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
    
    print("=" * 80)
    print("🎵 批量人声检测预处理（Musicnn）")
    print("=" * 80)
    
    # 获取所有音频文件
    if args.audio_list:
        # 从文本文件读取路径
        list_file = Path(args.audio_list)
        if not list_file.exists():
            print(f"❌ 文件列表不存在: {list_file}")
            return
        
        print(f"\n📄 音频列表文件: {list_file}")
        with open(list_file, 'r', encoding='utf-8') as f:
            audio_paths = [line.strip() for line in f if line.strip()]
        audio_files = [Path(p) for p in audio_paths if Path(p).exists()]
        missing_count = len(audio_paths) - len(audio_files)
        if missing_count > 0:
            print(f"⚠️  {missing_count} 个文件不存在，已跳过")
    else:
        # 从目录扫描
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
    print(f"   - 最小全局人声分数: {args.min_global_vocalness}")
    print(f"   - 窗口人声阈值: {args.vocal_threshold}")
    print(f"   - 对称填充: {args.padding_sec} 秒")
    
    print(f"\n✓ 找到 {len(audio_files)} 个音频文件")
    
    if len(audio_files) == 0:
        print("❌ 没有找到音频文件")
        return
    
    print("\n" + "=" * 80)
    print("开始处理...")
    print("=" * 80)
    
    # 统计信息
    stats = {
        'total': len(audio_files),
        'success': 0,
        'low_vocalness': 0,
        'no_segments': 0,
        'musicnn_error': 0,
        'other_error': 0
    }
    
    summary = {}
    
    # 批量处理
    for audio_path in tqdm(audio_files, desc="处理音频"):
        result = process_single_audio(audio_path, args)
        
        # 更新统计
        status = result['status']
        if status == 'success':
            stats['success'] += 1
        elif status == 'low_vocalness':
            stats['low_vocalness'] += 1
        elif status == 'no_segments':
            stats['no_segments'] += 1
        elif status == 'musicnn_error':
            stats['musicnn_error'] += 1
        else:
            stats['other_error'] += 1
        
        # 保存单个结果
        stem = audio_path.stem
        output_path = output_dir / f"{stem}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # 添加到汇总（简化版）
        summary[stem] = {
            'status': result['status'],
            'global_vocalness': result.get('global_vocalness'),
            'num_segments': len(result.get('segments', [])),
            'num_chunks': len(result.get('chunks', []))
        }
    
    # 显示统计
    print("\n" + "=" * 80)
    print("处理完成！")
    print("=" * 80)
    print(f"\n📊 统计:")
    print(f"   ✅ 成功: {stats['success']} / {stats['total']}")
    print(f"   ⚠️  人声过低: {stats['low_vocalness']}")
    print(f"   ⚠️  无人声片段: {stats['no_segments']}")
    print(f"   ❌ Musicnn 错误: {stats['musicnn_error']}")
    print(f"   ❌ 其他错误: {stats['other_error']}")
    
    print(f"\n💾 详细结果保存在: {output_dir}")
    print(f"   每个音频对应一个 JSON 文件")
    
    # 保存汇总（如果指定）
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary_data = {
            'stats': stats,
            'results': summary,
            'config': {
                'min_global_vocalness': args.min_global_vocalness,
                'vocal_threshold': args.vocal_threshold,
                'padding_sec': args.padding_sec,
            }
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 汇总结果保存到: {summary_path}")
    
    print("\n✅ 批量预处理完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
