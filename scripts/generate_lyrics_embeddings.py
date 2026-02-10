#!/usr/bin/env python3
"""
生成歌词嵌入
使用 Whisper 转录 + 文本编码器生成歌词嵌入
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
import torchaudio
import warnings
warnings.filterwarnings("ignore")

def load_audio(audio_path, sample_rate=16000):
    """加载音频文件"""
    # 使用 librosa 加载（更稳定）
    import librosa
    waveform_np, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    # 转换为 torch tensor
    waveform = torch.from_numpy(waveform_np).float()
    
    return waveform

def split_audio_30s(waveform, sample_rate=16000, chunk_sec=30.0):
    """将音频分割成30秒块"""
    chunk_size = int(sample_rate * chunk_sec)
    chunks = []
    
    for start in range(0, len(waveform), chunk_size):
        chunk = waveform[start:start + chunk_size]
        # 零填充
        if len(chunk) < chunk_size:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_size - len(chunk)))
        chunks.append(chunk)
    
    return chunks

def transcribe_audio(audio_chunks, processor, model, device):
    """使用 Whisper 转录音频"""
    transcriptions = []
    
    for chunk in audio_chunks:
        # 准备输入
        inputs = processor(
            chunk.numpy(),
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
    parser = argparse.ArgumentParser(description="生成歌词嵌入")
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
    
    args = parser.parse_args()
    
    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载 Whisper 模型
    print(f"\n加载 Whisper 模型: {args.whisper_model}")
    whisper_processor = AutoProcessor.from_pretrained(args.whisper_model)
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.whisper_model,
        torch_dtype=torch.float32,  # 使用 float32 避免类型不匹配
        low_cpu_mem_usage=True,
    ).to(device)
    
    # 加载文本编码器
    print(f"加载文本编码器: {args.text_model}")
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
    
    # 获取音频文件
    audio_dir = Path(args.audio_dir)
    audio_files = sorted(audio_dir.glob("*.mp3"))
    print(f"\n找到 {len(audio_files)} 个音频文件")
    
    # 处理每个文件
    embeddings = {}
    
    print("\n开始处理...")
    for audio_path in tqdm(audio_files, desc="生成歌词嵌入"):
        try:
            song_id = audio_path.stem
            
            # 加载音频
            waveform = load_audio(str(audio_path))
            
            # 分割成30秒块
            chunks = split_audio_30s(waveform)
            
            # 转录
            transcription = transcribe_audio(chunks, whisper_processor, whisper_model, device)
            
            # 生成文本嵌入
            embedding = encode_text(transcription, text_model, text_tokenizer, device)
            
            # 为每个块保存嵌入（这里简化为每首歌一个嵌入）
            for i, chunk in enumerate(chunks):
                key = f"{song_id}_{i}"
                embeddings[key] = embedding.astype(np.float32)
        
        except Exception as e:
            print(f"\n❌ 处理失败 {audio_path.name}: {e}")
            continue
    
    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(output_path, **embeddings)
    
    print(f"\n✅ 完成！")
    print(f"保存到: {output_path}")
    print(f"生成嵌入数量: {len(embeddings)}")
    print(f"嵌入维度: {list(embeddings.values())[0].shape}")

if __name__ == "__main__":
    main()
