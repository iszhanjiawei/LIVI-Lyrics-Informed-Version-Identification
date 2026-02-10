"""
基于 Musicnn 的人声检测器
按照论文 4.1 节实现：
1. 使用 Musicnn 提取特征
2. 添加线性层进行二分类
3. 每 3 秒窗口估计人声概率
4. 过滤低人声歌曲，提取人声片段
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
import librosa

# 添加 musicnn 到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
musicnn_path = project_root / "musicnn"
if str(musicnn_path) not in sys.path:
    sys.path.insert(0, str(musicnn_path))

from musicnn.extractor import extractor


class MusicnnVocalDetector:
    """
    基于 Musicnn 的人声检测器
    
    论文 4.1 节描述：
    - Musicnn 架构 + 单个线性层（维度 2）进行二分类
    - 估计每个 3 秒窗口的人声概率 v
    - 全局人声分数 = 所有窗口的平均值
    - 过滤人声分数 < λ 的歌曲
    - 保留 v ≥ 0.5 的窗口，连接成连续区域
    - 对称填充最多 10 秒，截断/零填充到 30 秒
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_sec: float = 30.0,
        window_sec: float = 3.0,
        vocal_threshold: float = 0.5,
        min_global_vocalness: float = 0.3,
        padding_sec: float = 10.0,
        musicnn_model: str = 'MTT_musicnn',
        use_simple_heuristic: bool = True,
    ):
        """
        Args:
            sample_rate: 采样率
            chunk_sec: 输出 chunk 的长度（秒）
            window_sec: 检测窗口大小（秒），论文中为 3 秒
            vocal_threshold: 人声概率阈值，保留 v ≥ threshold 的窗口
            min_global_vocalness: 最小全局人声分数 λ，低于此值的歌曲会被跳过
            padding_sec: 对称填充长度（秒），论文中为最多 10 秒
            musicnn_model: Musicnn 模型名称
            use_simple_heuristic: 是否使用简单启发式（因为没有训练好的分类器）
        """
        self.sample_rate = sample_rate
        self.chunk_sec = chunk_sec
        self.chunk_size = int(sample_rate * chunk_sec)
        self.window_sec = window_sec
        self.window_size = int(sample_rate * window_sec)
        self.vocal_threshold = vocal_threshold
        self.min_global_vocalness = min_global_vocalness
        self.padding_sec = padding_sec
        self.padding_size = int(sample_rate * padding_sec)
        self.musicnn_model = musicnn_model
        self.use_simple_heuristic = use_simple_heuristic
        
        # 注意：论文中使用的是专有模型（Musicnn + 线性层 for binary classification）
        # 由于我们没有预训练的分类器，这里使用启发式方法
        # 实际生产环境需要训练一个二分类器
    
    def estimate_vocalness_musicnn(self, audio_path: str) -> Tuple[np.ndarray, float]:
        """
        使用 Musicnn 估计每个窗口的人声概率
        
        Args:
            audio_path: 音频文件路径
        
        Returns:
            window_vocalness: 每个窗口的人声概率 (N_windows,)
            global_vocalness: 全局人声分数（所有窗口的平均值）
        """
        try:
            # 使用 musicnn 提取特征和标签
            taggram, tags = extractor(
                audio_path,
                model=self.musicnn_model,
                input_length=self.window_sec,
                input_overlap=0,  # 非重叠窗口
                extract_features=False
            )
            
            # taggram shape: (n_windows, n_tags)
            # tags: 标签列表
            
            # 使用启发式规则估计人声
            # 查找与人声相关的标签
            vocal_tags = ['vocal', 'voice', 'singer', 'singing', 'vocals', 
                         'female voice', 'male voice', 'speech']
            no_vocal_tags = ['no vocal', 'no vocals', 'instrumental']
            
            vocal_indices = [i for i, tag in enumerate(tags) if any(vt in tag.lower() for vt in vocal_tags)]
            no_vocal_indices = [i for i, tag in enumerate(tags) if any(nvt in tag.lower() for nvt in no_vocal_tags)]
            
            if vocal_indices:
                # 人声概率 = 人声标签的概率总和
                window_vocalness = np.sum(taggram[:, vocal_indices], axis=1)
            else:
                # 如果没有人声标签，使用逆向逻辑（非器乐）
                if no_vocal_indices:
                    window_vocalness = 1.0 - np.sum(taggram[:, no_vocal_indices], axis=1)
                else:
                    # 默认假设有人声
                    window_vocalness = np.ones(taggram.shape[0]) * 0.5
            
            # 归一化到 [0, 1]
            window_vocalness = np.clip(window_vocalness, 0, 1)
            
            # 全局人声分数 = 平均值
            global_vocalness = float(np.mean(window_vocalness))
            
            return window_vocalness, global_vocalness
            
        except Exception as e:
            print(f"  ⚠️  Musicnn 处理失败: {e}，使用简单启发式")
            return self._estimate_vocalness_heuristic(audio_path)
    
    def _estimate_vocalness_heuristic(self, audio_path: str) -> Tuple[np.ndarray, float]:
        """
        简单的启发式人声检测（备用方案）
        基于频谱特征
        """
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # 计算窗口数量
        n_windows = max(1, len(audio) // self.window_size)
        window_vocalness = []
        
        for i in range(n_windows):
            start = i * self.window_size
            end = min(start + self.window_size, len(audio))
            window = audio[start:end]
            
            if len(window) < self.window_size // 2:
                # 窗口太短，跳过
                continue
            
            # 特征 1: 频谱重心（人声主要在 200-4000 Hz）
            spectral_centroid = librosa.feature.spectral_centroid(y=window, sr=sr)[0]
            mean_centroid = np.mean(spectral_centroid)
            
            # 人声频率范围
            if 200 <= mean_centroid <= 4000:
                centroid_score = 1.0
            elif mean_centroid < 200:
                centroid_score = mean_centroid / 200
            else:
                centroid_score = max(0, 1.0 - (mean_centroid - 4000) / 4000)
            
            # 特征 2: 能量
            energy = np.sqrt(np.mean(window ** 2))
            energy_score = min(energy / 0.1, 1.0)
            
            # 特征 3: 过零率
            zcr = librosa.feature.zero_crossing_rate(window)[0]
            zcr_mean = np.mean(zcr)
            zcr_score = 1.0 if 0.1 <= zcr_mean <= 0.3 else 0.5
            
            # 综合得分
            vocalness = 0.5 * centroid_score + 0.3 * energy_score + 0.2 * zcr_score
            window_vocalness.append(vocalness)
        
        window_vocalness = np.array(window_vocalness)
        global_vocalness = float(np.mean(window_vocalness)) if len(window_vocalness) > 0 else 0.0
        
        return window_vocalness, global_vocalness
    
    def extract_vocal_segments(
        self,
        waveform: torch.Tensor,
        audio_path: Optional[str] = None
    ) -> Tuple[List[np.ndarray], float]:
        """
        提取人声片段
        
        按照论文 4.1 节：
        1. 估计每个 3 秒窗口的人声概率
        2. 计算全局人声分数，低于 min_global_vocalness 则跳过
        3. 保留 v ≥ vocal_threshold 的窗口
        4. 连接成连续区域
        5. 对称填充最多 padding_sec 秒
        6. 截断/零填充到 chunk_sec 秒
        
        Args:
            waveform: 输入波形 (1, T) or (T,)
            audio_path: 音频文件路径（用于 musicnn）
        
        Returns:
            chunks: 人声片段列表
            global_vocalness: 全局人声分数
        """
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)
        
        # 1. 估计人声概率
        if audio_path and not self.use_simple_heuristic:
            window_vocalness, global_vocalness = self.estimate_vocalness_musicnn(audio_path)
        else:
            # 使用简单启发式或没有音频路径
            if audio_path:
                window_vocalness, global_vocalness = self._estimate_vocalness_heuristic(audio_path)
            else:
                # 无法估计，假设全部是人声
                n_windows = max(1, len(waveform) // self.window_size)
                window_vocalness = np.ones(n_windows) * 0.8
                global_vocalness = 0.8
        
        # 2. 检查全局人声分数
        if global_vocalness < self.min_global_vocalness:
            print(f"  ⚠️  全局人声分数过低 ({global_vocalness:.3f} < {self.min_global_vocalness})，跳过")
            return [], global_vocalness
        
        # 3. 提取 v ≥ threshold 的窗口
        vocal_windows = window_vocalness >= self.vocal_threshold
        
        if not vocal_windows.any():
            print(f"  ⚠️  没有找到人声窗口")
            return [], global_vocalness
        
        # 4. 找到连续的人声区域
        vocal_regions = self._find_continuous_regions(vocal_windows)
        
        if len(vocal_regions) == 0:
            print(f"  ⚠️  没有连续的人声区域")
            return [], global_vocalness
        
        # 5. 对每个区域进行对称填充并提取 30 秒片段
        chunks = []
        waveform_np = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform
        
        for start_win, end_win in vocal_regions:
            # 将窗口索引转换为采样点索引
            start_sample = start_win * self.window_size
            end_sample = min(end_win * self.window_size, len(waveform_np))
            
            # 对称填充（最多 padding_sec 秒）
            pad_start = max(0, start_sample - self.padding_size)
            pad_end = min(len(waveform_np), end_sample + self.padding_size)
            
            region_audio = waveform_np[pad_start:pad_end]
            
            # 截断或零填充到 chunk_size
            if len(region_audio) >= self.chunk_size:
                # 如果区域很长，可能需要切成多个 chunk
                for chunk_start in range(0, len(region_audio), self.chunk_size):
                    chunk = region_audio[chunk_start:chunk_start + self.chunk_size]
                    
                    if len(chunk) >= self.chunk_size // 2:  # 至少一半长度
                        if len(chunk) < self.chunk_size:
                            # 零填充
                            chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))
                        chunks.append(chunk)
            else:
                # 零填充到 chunk_size
                chunk = np.pad(region_audio, (0, self.chunk_size - len(region_audio)))
                chunks.append(chunk)
        
        return chunks, global_vocalness
    
    def _find_continuous_regions(self, binary_array: np.ndarray) -> List[Tuple[int, int]]:
        """
        找到二进制数组中的连续 True 区域
        
        Args:
            binary_array: 布尔数组
        
        Returns:
            regions: [(start, end), ...] 区域列表
        """
        regions = []
        in_region = False
        start = 0
        
        for i, val in enumerate(binary_array):
            if val and not in_region:
                # 进入新区域
                start = i
                in_region = True
            elif not val and in_region:
                # 离开区域
                regions.append((start, i))
                in_region = False
        
        # 处理最后一个区域
        if in_region:
            regions.append((start, len(binary_array)))
        
        return regions
    
    def pipeline(
        self,
        waveform: torch.Tensor,
        audio_path: Optional[str] = None
    ) -> List[np.ndarray]:
        """
        完整的人声检测和提取流程
        
        Args:
            waveform: 输入波形
            audio_path: 音频文件路径（可选，用于 musicnn）
        
        Returns:
            chunks: 人声片段列表
        """
        chunks, global_vocalness = self.extract_vocal_segments(waveform, audio_path)
        
        if len(chunks) == 0:
            # 如果没有检测到人声，降级到简单切片
            print(f"  ⚠️  未检测到人声，使用简单切片作为后备")
            if waveform.dim() == 2:
                waveform = waveform.squeeze(0)
            waveform_np = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform
            
            # 简单地切成 30 秒块
            chunks = []
            for start in range(0, len(waveform_np), self.chunk_size):
                chunk = waveform_np[start:start + self.chunk_size]
                if len(chunk) < self.chunk_size:
                    chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))
                chunks.append(chunk)
        
        return chunks


# ------------------------------- Runners -------------------------------
@lru_cache(maxsize=8)
def get_cached_musicnn_vocal_detector(
    sample_rate: int = 16000,
    chunk_sec: float = 30.0,
    window_sec: float = 3.0,
    vocal_threshold: float = 0.5,
    min_global_vocalness: float = 0.3,
) -> MusicnnVocalDetector:
    """
    获取缓存的 Musicnn 人声检测器
    """
    return MusicnnVocalDetector(
        sample_rate=sample_rate,
        chunk_sec=chunk_sec,
        window_sec=window_sec,
        vocal_threshold=vocal_threshold,
        min_global_vocalness=min_global_vocalness,
    )


def extract_vocals_musicnn(
    waveform: torch.Tensor,
    audio_path: Optional[str] = None,
    vocal_detector: Optional[MusicnnVocalDetector] = None,
    sample_rate: int = 16000,
    chunk_sec: float = 30.0,
) -> List[np.ndarray]:
    """
    使用 Musicnn 提取人声片段
    
    Args:
        waveform: 输入波形
        audio_path: 音频文件路径（可选）
        vocal_detector: 人声检测器实例（可选）
        sample_rate: 采样率
        chunk_sec: chunk 长度
    
    Returns:
        chunks: 人声片段列表
    """
    vocal_detector = vocal_detector or get_cached_musicnn_vocal_detector(
        sample_rate=sample_rate,
        chunk_sec=chunk_sec,
    )
    
    return vocal_detector.pipeline(waveform, audio_path)
