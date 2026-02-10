#!/usr/bin/env python3
"""
使用 musicnn 测试人声检测
直接导入 musicnn 代码，不需要安装包
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# 添加 musicnn 到路径
project_root = Path(__file__).parent.parent
musicnn_path = project_root / "musicnn"
sys.path.insert(0, str(musicnn_path))

# 导入 musicnn
from musicnn.extractor import extractor


def test_musicnn_on_song(audio_path, window_sec=3.0, vocal_threshold=0.5):
    """
    使用 musicnn 测试歌曲的人声检测
    """
    print("=" * 70)
    print("🎵 Musicnn 人声检测测试")
    print("=" * 70)
    print(f"\n📁 音频文件: {audio_path}")
    
    # 检查文件
    if not Path(audio_path).exists():
        print(f"❌ 错误: 文件不存在")
        return
    
    print(f"\n[1/3] 使用 Musicnn 提取特征...")
    print(f"  - 模型: MTT_musicnn")
    print(f"  - 窗口长度: {window_sec} 秒")
    
    try:
        # 使用 musicnn 提取标签
        taggram, tags = extractor(
            audio_path,
            model='MTT_musicnn',
            input_length=window_sec,
            input_overlap=0,  # 非重叠窗口
            extract_features=False
        )
        
        print(f"✓ 提取完成")
        print(f"  - 窗口数量: {taggram.shape[0]}")
        print(f"  - 标签数量: {taggram.shape[1]}")
        print(f"  - 总时长: {taggram.shape[0] * window_sec:.1f} 秒")
        
    except Exception as e:
        print(f"❌ Musicnn 提取失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 分析人声相关标签
    print(f"\n[2/3] 分析人声标签...")
    
    # 查找人声相关的标签索引
    vocal_tags = ['vocal', 'voice', 'singer', 'singing', 'vocals', 
                  'female voice', 'male voice', 'speech', 'choir']
    no_vocal_tags = ['no vocal', 'no vocals', 'instrumental']
    
    vocal_indices = [i for i, tag in enumerate(tags) if any(vt in tag.lower() for vt in vocal_tags)]
    no_vocal_indices = [i for i, tag in enumerate(tags) if any(nvt in tag.lower() for nvt in no_vocal_tags)]
    
    print(f"  - 找到 {len(vocal_indices)} 个人声相关标签")
    print(f"  - 找到 {len(no_vocal_indices)} 个非人声标签")
    
    if vocal_indices:
        print(f"\n  人声相关标签:")
        for idx in vocal_indices[:5]:  # 显示前5个
            print(f"    - {tags[idx]}")
    
    if no_vocal_indices:
        print(f"\n  非人声标签:")
        for idx in no_vocal_indices[:3]:
            print(f"    - {tags[idx]}")
    
    # 计算每个窗口的人声概率
    print(f"\n[3/3] 计算人声概率...")
    
    window_vocalness = []
    for i in range(taggram.shape[0]):
        if vocal_indices:
            # 人声概率 = 人声标签概率之和
            vocal_prob = np.sum(taggram[i, vocal_indices])
        else:
            # 如果没有人声标签，用非器乐的逆概率
            if no_vocal_indices:
                vocal_prob = 1.0 - np.sum(taggram[i, no_vocal_indices])
            else:
                vocal_prob = 0.5  # 默认值
        
        # 归一化到 [0, 1]
        vocal_prob = np.clip(vocal_prob, 0, 1)
        window_vocalness.append(vocal_prob)
    
    window_vocalness = np.array(window_vocalness)
    global_vocalness = np.mean(window_vocalness)
    
    print(f"✓ 计算完成")
    print(f"  - 全局人声分数: {global_vocalness:.3f}")
    print(f"  - 人声概率范围: [{np.min(window_vocalness):.3f}, {np.max(window_vocalness):.3f}]")
    
    # 显示窗口详情
    print(f"\n📊 窗口详细分析（前10个窗口）:")
    print(f"{'窗口':<8} {'时间(秒)':<12} {'人声概率':<10} {'判定'}")
    print("-" * 70)
    
    for i in range(min(10, len(window_vocalness))):
        time_start = i * window_sec
        time_end = (i + 1) * window_sec
        vocal_mark = "✅ 人声" if window_vocalness[i] >= vocal_threshold else "❌ 非人声"
        print(f"窗口 {i+1:<3}  {time_start:>5.1f} - {time_end:>5.1f}  "
              f"{window_vocalness[i]:.3f}      {vocal_mark}")
    
    if len(window_vocalness) > 10:
        print(f"... (还有 {len(window_vocalness) - 10} 个窗口)")
    
    # 统计
    vocal_windows = np.sum(window_vocalness >= vocal_threshold)
    total_windows = len(window_vocalness)
    vocal_ratio = vocal_windows / total_windows if total_windows > 0 else 0
    
    print(f"\n✅ 分析完成")
    print(f"  - 人声窗口数: {vocal_windows} / {total_windows}")
    print(f"  - 人声比例: {vocal_ratio * 100:.1f}%")
    print(f"  - 人声总时长: {vocal_windows * window_sec:.1f} 秒")
    
    # 显示 Top 标签
    print(f"\n🏆 Top 10 标签（平均概率）:")
    tags_mean = np.mean(taggram, axis=0)
    top_indices = np.argsort(tags_mean)[-10:][::-1]
    
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank:2d}. {tags[idx]:<20} {tags_mean[idx]:.3f}")
    
    print("\n" + "=" * 70)
    print("✅ 测试完成")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Musicnn 人声检测测试")
    parser.add_argument("--audio-path", type=str, required=True, help="音频文件路径")
    parser.add_argument("--window-sec", type=float, default=3.0, help="窗口长度（秒）")
    parser.add_argument("--vocal-threshold", type=float, default=0.5, help="人声阈值")
    
    args = parser.parse_args()
    
    test_musicnn_on_song(
        args.audio_path,
        window_sec=args.window_sec,
        vocal_threshold=args.vocal_threshold
    )


if __name__ == "__main__":
    main()
