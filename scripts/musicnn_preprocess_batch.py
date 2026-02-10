#!/usr/bin/env python3
"""
使用 Musicnn 对批量音频进行预处理
在 musicnn_env 环境下运行

功能：
1. 使用 Musicnn 检测人声片段
2. 提取人声区域，过滤纯器乐段
3. 保存预处理信息到 JSON 文件

输出格式：
{
  "song_id": {
    "original_path": "原始音频路径",
    "duration": 总时长(秒),
    "global_vocalness": 全局人声分数,
    "vocal_segments": [
      {
        "start": 开始时间(秒),
        "end": 结束时间(秒),
        "vocalness": 人声概率
      }
    ],
    "vocal_ratio": 人声比例,
    "status": "success" / "low_vocalness" / "error"
  }
}
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
    
    Returns:
        window_vocalness: List[Dict] - 每个窗口的人声信息
        global_vocalness: float - 全局人声分数
    """
    try:
        # 使用 musicnn 提取标签
        taggram, tags = extractor(
            audio_path,
            model='MTT_musicnn',
            input_length=window_sec,
            input_overlap=0,
            extract_features=False
        )
        
        # 查找人声相关标签
        vocal_tags = ['vocal', 'voice', 'singer', 'singing', 'vocals', 
                      'female voice', 'male voice', 'speech', 'choir']
        no_vocal_tags = ['no vocal', 'no vocals', 'instrumental']
        
        vocal_indices = [i for i, tag in enumerate(tags) if any(vt in tag.lower() for vt in vocal_tags)]
        no_vocal_indices = [i for i, tag in enumerate(tags) if any(nvt in tag.lower() for nvt in no_vocal_tags)]
        
        # 计算每个窗口的人声概率
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
        print(f"  ❌ Musicnn 处理失败: {e}")
        return None, None


def extract_vocal_segments(window_vocalness, vocal_threshold=0.5):
    """
    从窗口人声概率中提取连续的人声片段
    """
    if not window_vocalness:
        return []
    
    # 找到人声窗口
    vocal_windows = [w for w in window_vocalness if w['vocalness'] >= vocal_threshold]
    
    if not vocal_windows:
        return []
    
    # 合并连续窗口
    segments = []
    current_start = vocal_windows[0]['time_start']
    current_end = vocal_windows[0]['time_end']
    current_vocalness = [vocal_windows[0]['vocalness']]
    
    for i in range(1, len(vocal_windows)):
        # 检查是否连续
        if vocal_windows[i]['window_id'] == vocal_windows[i-1]['window_id'] + 1:
            # 连续，扩展当前片段
            current_end = vocal_windows[i]['time_end']
            current_vocalness.append(vocal_windows[i]['vocalness'])
        else:
            # 不连续，保存当前片段并开始新片段
            segments.append({
                'start': current_start,
                'end': current_end,
                'vocalness': float(np.mean(current_vocalness))
            })
            current_start = vocal_windows[i]['time_start']
            current_end = vocal_windows[i]['time_end']
            current_vocalness = [vocal_windows[i]['vocalness']]
    
    # 保存最后一个片段
    segments.append({
        'start': current_start,
        'end': current_end,
        'vocalness': float(np.mean(current_vocalness))
    })
    
    return segments


def process_song(audio_path, song_id, min_global_vocalness=0.3, vocal_threshold=0.5):
    """
    处理单首歌曲
    """
    result = {
        'song_id': song_id,
        'original_path': str(audio_path),
        'status': 'processing'
    }
    
    try:
        # 获取音频时长
        duration = librosa.get_duration(path=audio_path)
        result['duration'] = float(duration)
        
        # 使用 Musicnn 估计人声
        window_vocalness, global_vocalness = estimate_vocalness_musicnn(str(audio_path))
        
        if window_vocalness is None:
            result['status'] = 'error'
            result['error'] = 'musicnn_failed'
            return result
        
        result['global_vocalness'] = global_vocalness
        
        # 检查全局人声分数
        if global_vocalness < min_global_vocalness:
            result['status'] = 'low_vocalness'
            result['vocal_segments'] = []
            result['vocal_ratio'] = 0.0
            return result
        
        # 提取人声片段
        segments = extract_vocal_segments(window_vocalness, vocal_threshold)
        result['vocal_segments'] = segments
        
        # 计算人声比例
        vocal_duration = sum(seg['end'] - seg['start'] for seg in segments)
        result['vocal_ratio'] = float(vocal_duration / duration if duration > 0 else 0)
        
        result['status'] = 'success'
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="批量 Musicnn 预处理")
    parser.add_argument("--metadata-json", type=str, required=True, 
                       help="元数据 JSON 文件路径")
    parser.add_argument("--output-json", type=str, required=True,
                       help="输出 JSON 文件路径")
    parser.add_argument("--min-global-vocalness", type=float, default=0.3,
                       help="最小全局人声分数阈值")
    parser.add_argument("--vocal-threshold", type=float, default=0.5,
                       help="窗口人声判定阈值")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎵 Musicnn 批量预处理")
    print("=" * 80)
    
    # 加载元数据
    print(f"\n📁 加载元数据: {args.metadata_json}")
    with open(args.metadata_json, 'r', encoding='utf-8') as f:
        songs = json.load(f)
    
    print(f"✓ 找到 {len(songs)} 首歌曲")
    
    # 处理每首歌
    print(f"\n🔄 开始处理...")
    print(f"  - 最小全局人声分数: {args.min_global_vocalness}")
    print(f"  - 窗口人声阈值: {args.vocal_threshold}")
    print()
    
    results = {}
    stats = {
        'success': 0,
        'low_vocalness': 0,
        'error': 0
    }
    
    for song in tqdm(songs, desc="处理进度"):
        song_id = song['song_id']
        audio_path = song['audio_path']
        
        if not Path(audio_path).exists():
            results[song_id] = {
                'song_id': song_id,
                'original_path': audio_path,
                'status': 'error',
                'error': 'file_not_found'
            }
            stats['error'] += 1
            continue
        
        result = process_song(audio_path, song_id, 
                            args.min_global_vocalness, 
                            args.vocal_threshold)
        results[song_id] = result
        stats[result['status']] += 1
    
    # 保存结果
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 处理完成!")
    print(f"\n📊 统计:")
    print(f"  - 成功: {stats['success']} 首")
    print(f"  - 人声过低: {stats['low_vocalness']} 首")
    print(f"  - 错误: {stats['error']} 首")
    print(f"\n💾 结果保存到: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
