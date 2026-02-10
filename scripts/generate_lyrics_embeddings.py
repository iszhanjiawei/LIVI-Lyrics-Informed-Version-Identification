#!/usr/bin/env python3
"""
生成歌词嵌入
使用 Whisper 转录 + 文本编码器生成歌词嵌入

按照论文 4.1 节流程：
1. 使用 Musicnn 检测人声片段
2. 过滤低人声分数的歌曲
3. 保留 v ≥ 0.5 的窗口，连接成连续区域
4. 对称填充（优化后为 1 秒）
5. 截断/零填充到 30 秒
6. 转录人声片段
7. 生成文本嵌入
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoModel,
    AutoTokenizer,
)
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# 添加 musicnn 路径
musicnn_path = project_root / "musicnn"
if str(musicnn_path) not in sys.path:
    sys.path.insert(0, str(musicnn_path))

from livi.core.data.preprocessing.musicnn_vocal_detector import MusicnnVocalDetector


def transcribe_chunks(chunks, processor, model, device):
    """使用 Whisper 转录音频块（论文 4.1 节）"""
    transcriptions = []
    
    for chunk in chunks:
        # 准备输入
        inputs = processor(
            chunk,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(device)
        
        # 生成转录（使用 return_timestamps=True 确保捕获完整内容）
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_length=448,
                num_beams=1,
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                return_timestamps=True,
            )
        
        # 解码
        transcription = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        transcriptions.append(transcription)
    
    return " ".join(transcriptions)

def mean_pooling(model_output, attention_mask):
    """Mean pooling"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_text(text, model, tokenizer, device):
    """使用文本编码器生成嵌入"""
    if not text or len(text.strip()) == 0:
        # 空文本返回零向量
        return np.zeros(768, dtype=np.float32)
    
    # Tokenize
    encoded_input = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=8192
    ).to(device)
    
    # 生成嵌入
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Mean pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings.cpu().numpy().squeeze().astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="生成歌词嵌入（论文 4.1 节流程）")
    parser.add_argument("--audio-dir", type=str, required=True,
                       help="音频目录")
    parser.add_argument("--output", type=str, required=True,
                       help="输出 .npz 文件路径")
    parser.add_argument("--whisper-model", type=str,
                       default="pretrained_models/whisper-large-v3-turbo",
                       help="Whisper 模型路径")
    parser.add_argument("--text-model", type=str,
                       default="pretrained_models/gte-multilingual-base",
                       help="文本编码器路径")
    parser.add_argument("--min-global-vocalness", type=float, default=0.3,
                       help="最小全局人声分数阈值 λ（论文 4.1 节）")
    parser.add_argument("--vocal-threshold", type=float, default=0.5,
                       help="窗口人声判定阈值（论文 4.1 节）")
    parser.add_argument("--padding-sec", type=float, default=1.0,
                       help="对称填充秒数（论文中为 10 秒，优化后为 1 秒）")
    
    args = parser.parse_args()
    
    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    print(f"\n=" * 80)
    print("论文 4.1 节预处理流程")
    print("=" * 80)
    print(f"- Musicnn 窗口: 3 秒")
    print(f"- 人声阈值: {args.vocal_threshold}")
    print(f"- 最小全局人声分数 λ: {args.min_global_vocalness}")
    print(f"- 对称填充: ±{args.padding_sec} 秒")
    print(f"- 输出片段长度: 30 秒")
    
    # 初始化 Musicnn 人声检测器（论文 4.1 节）
    print(f"\n初始化 Musicnn 人声检测器...")
    vocal_detector = MusicnnVocalDetector(
        sample_rate=16000,
        chunk_sec=30.0,
        window_sec=3.0,
        vocal_threshold=args.vocal_threshold,
        min_global_vocalness=args.min_global_vocalness,
        padding_sec=args.padding_sec,
        use_simple_heuristic=True
    )
    print("✓ Musicnn 人声检测器初始化完成")
    
    # 加载 Whisper 模型
    print(f"\n加载 Whisper 模型: {args.whisper_model}")
    whisper_processor = AutoProcessor.from_pretrained(args.whisper_model)
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.whisper_model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    print("✓ Whisper 模型加载完成")
    
    # 加载文本编码器
    print(f"\n加载文本编码器: {args.text_model}")
    text_tokenizer = AutoTokenizer.from_pretrained(
        args.text_model,
        local_files_only=True,
        trust_remote_code=True
    )
    text_model = AutoModel.from_pretrained(
        args.text_model,
        local_files_only=True,
        trust_remote_code=True
    ).to(device)
    text_model.eval()
    print("✓ 文本编码器加载完成")
    
    # 获取音频文件
    audio_dir = Path(args.audio_dir)
    audio_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.m4a")) + list(audio_dir.glob("*.wav"))
    audio_files = sorted(audio_files)
    print(f"\n找到 {len(audio_files)} 个音频文件")
    
    # 处理每个文件
    embeddings = {}
    skipped_count = 0
    
    print("\n" + "=" * 80)
    print("开始处理音频文件（论文 4.1 节流程）")
    print("=" * 80)
    
    for audio_path in tqdm(audio_files, desc="生成歌词嵌入"):
        try:
            song_id = audio_path.stem
            
            # 步骤 1-5: 使用 Musicnn 进行人声检测和预处理（论文 4.1 节）
            chunks, global_vocalness = vocal_detector.extract_vocal_segments(str(audio_path))
            
            # 检查全局人声分数（论文 4.1 节：过滤低于 λ 的歌曲）
            if global_vocalness < args.min_global_vocalness:
                skipped_count += 1
                tqdm.write(f"⊘ 跳过 {audio_path.name} (全局人声分数 {global_vocalness:.3f} < {args.min_global_vocalness})")
                continue
            
            # 检查是否有人声片段
            if chunks is None or len(chunks) == 0:
                skipped_count += 1
                tqdm.write(f"⊘ 跳过 {audio_path.name} (无人声片段)")
                continue
            
            # 步骤 6: 转录人声片段
            transcription = transcribe_chunks(chunks, whisper_processor, whisper_model, device)
            
            # 步骤 7: 生成文本嵌入
            embedding = encode_text(transcription, text_model, text_tokenizer, device)
            
            # 为每个 chunk 保存嵌入（论文中每个 30 秒片段独立处理）
            for i in range(len(chunks)):
                key = f"{song_id}_{i}"
                embeddings[key] = embedding.astype(np.float32)
        
        except Exception as e:
            tqdm.write(f"\n❌ 处理失败 {audio_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(output_path, **embeddings)
    
    print(f"\n" + "=" * 80)
    print("✅ 处理完成！")
    print("=" * 80)
    print(f"保存到: {output_path}")
    print(f"处理文件: {len(audio_files)} 个")
    print(f"跳过文件: {skipped_count} 个（低人声分数或无人声片段）")
    print(f"成功文件: {len(audio_files) - skipped_count} 个")
    print(f"生成嵌入: {len(embeddings)} 个")
    if len(embeddings) > 0:
        print(f"嵌入维度: {list(embeddings.values())[0].shape}")

if __name__ == "__main__":
    main()
