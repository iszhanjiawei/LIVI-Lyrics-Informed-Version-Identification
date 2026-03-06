import torch
from omegaconf import OmegaConf
from loguru import logger
from livi.core.data.preprocessing.vocal_detector import get_cached_vocal_detector, extract_vocals
from livi.core.data.utils.audio_toolbox import load_audio

from livi.apps.frozen_encoder.models.text_encoder import _get_cached_text_encoder, encode_text
from livi.apps.frozen_encoder.models.transcriber import _get_cached_transcriber, transcribe
from typing import List, Optional

from livi.utils.time import record_time
from loguru import logger
import numpy as np
from pathlib import Path
from typing import Dict


class Session:
    """
    Main class to run inference on the frozen encoder.
    Given the path of an audio file, it :
    - loads the audio
    - extract 30s audio chunks (placeholder for vocal activity detection)
    - extract mel spectrograms via Whisper feature extractor
    - Transcribe via Whisper
    - Compute text embeddings via pre-trained multilingual text encoder
    - Outputs lyrics-informed embeddings
    """

    def __init__(self, config_path: str):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cfg = OmegaConf.load(config_path)

        # Transcriber / Text Encoder
        self.transcriber = _get_cached_transcriber(
            model_name=self.cfg.transcriber.model_name,
            device=self.device,
            dtype_fp16_on_cuda=torch.float16 if self.device.type == "cuda" else torch.float32,
            sampling_rate=self.cfg.data.sr,
            num_beams=self.cfg.transcriber.num_beams,
            condition_on_prev_tokens=self.cfg.transcriber.condition_on_prev_tokens,
            compression_ratio_threshold=self.cfg.transcriber.compression_ratio_threshold,
            temperature=self.cfg.transcriber.temperature,
            logprob_threshold=self.cfg.transcriber.logprob_threshold,
            return_timestamps=self.cfg.transcriber.return_timestamps,
            remove_phrases=self.cfg.transcriber.remove_phrases,
            repeat_threshold=self.cfg.transcriber.repeat_threshold,
            min_words_per_chunk=self.cfg.transcriber.min_words_per_chunk,
        )
        self.text_encoder = _get_cached_text_encoder(
            model_name=self.cfg.text_encoder.model_name,
            chunking=self.cfg.text_encoder.chunking,
        )
        # Vocal Segments Extraction
        # Check if preprocessing mode is enabled
        use_preprocess = self.cfg.data.get('use_preprocess', False)
        preprocess_dir = self.cfg.data.get('preprocess_dir', None)
        
        self.vocal_detector = get_cached_vocal_detector(
            sample_rate=self.cfg.data.sr,
            chunk_sec=self.cfg.data.chunk_sec,
            use_preprocess=use_preprocess,
            preprocess_dir=preprocess_dir,
        )

    def inference(self, audio_path: str) -> torch.Tensor:
        waveform = load_audio(audio_path, target_sample_rate=self.cfg.data.sr)

        # Extract 30s audio chunks using preprocessing info or simple chunking
        chunks_audio = extract_vocals(audio_path, waveform, self.vocal_detector)
        
        # 释放 waveform，避免累积
        del waveform

        with torch.no_grad():
            # Transcribe
            transcriptions = transcribe(
                chunks_audio, translate=self.cfg.transcriber.translate, transcriber=self.transcriber
            )
            
            # 释放 chunks_audio，避免累积
            del chunks_audio
            
            if self.cfg.text_encoder.chunking:
                inputs = [x for x in transcriptions[0] if x]  # 过滤掉那些无效的。
            else:
                inputs = transcriptions[-1]
            
            # 释放 transcriptions，避免累积
            del transcriptions

            embeddings = encode_text(
                inputs,
                text_encoder=self.text_encoder,
                model_name=self.cfg.text_encoder.model_name,
                chunking=self.cfg.text_encoder.chunking,
                batch_size=self.cfg.text_encoder.batch_size,
                get_single_embedding=self.cfg.text_encoder.get_single_embedding,
            )
            
            # 释放 inputs
            del inputs
            
            # 强制清理 GPU 缓存，防止显存累积
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 确保返回 numpy 数组（CPU）
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        return embeddings

    def estimate_inference_time(
        self,
        audio_paths: List[str],
        start_after: Optional[int] = 5,
    ) -> None:
        """
        Estimate the inference time for a random sample of tracks.
        Args:
            audio_paths (List[str]): List of audio file paths to estimate inference time for.
            start_after (int): Number of tracks to wait before starting the timer
            (first inference steps are longer than after with torch.compile)
        """
        pre_times, transc_times, encoding_times, total_times = [], [], [], []

        for idx, audio_path in enumerate(audio_paths):
            with record_time(total_times, idx, start_after):
                with record_time(pre_times, idx, start_after):
                    waveform = load_audio(audio_path, target_sample_rate=self.cfg.data.sr)

                    # Extract 30s audio chunks using preprocessing info or simple chunking
                    chunks_audio = extract_vocals(audio_path, waveform, self.vocal_detector)

                with record_time(transc_times, idx, start_after):
                    # Transcribe with Whisper
                    transcriptions = transcribe(
                        chunks_audio, translate=self.cfg.transcriber.translate, transcriber=self.transcriber
                    )
                    if self.cfg.text_encoder.chunking:
                        inputs = [x for x in transcriptions[0] if x]
                    else:
                        inputs = transcriptions[-1]

                with record_time(encoding_times, idx, start_after):
                    # Encode with text encoder
                    embeddings = encode_text(
                        inputs,
                        text_encoder=self.text_encoder,
                        model_name=self.cfg.text_encoder.model_name,
                        chunking=self.cfg.text_encoder.chunking,
                        batch_size=self.cfg.text_encoder.batch_size,
                        get_single_embedding=self.cfg.text_encoder.get_single_embedding,
                    )

        def mean_std(xs):
            return (float(np.mean(xs)), float(np.std(xs))) if xs else (float("nan"), float("nan"))

        pre_mean, pre_std = mean_std(pre_times)
        transc_mean, transc_std = mean_std(transc_times)
        encoding_mean, encoding_std = mean_std(encoding_times)
        tot_mean, tot_std = mean_std(total_times)

        logger.info(f"Preproc: {pre_mean:.4f}s (±{pre_std:.4f})")
        logger.info(f"Transc : {transc_mean:.4f}s (±{transc_std:.4f})")
        logger.info(f"Encoding: {encoding_mean:.4f}s (±{encoding_std:.4f})")
        logger.info(f"Total  : {tot_mean:.4f}s (±{tot_std:.4f})")


# ---------------------------------
# High-level runner helper methods
# ---------------------------------
def run_inference(
    config_path: Path,
    audio_dir: Path,
    path_out: Optional[Path] = None,
) -> np.ndarray:
    """
    Run frozen-encoder inference on all audio files in a directory and save
    a mapping {basename -> embedding} to disk.

    Parameters
    ----------
    config_path : Path
        Path to the OmegaConf YAML
    audio_dir : Path
        Directory to scan recursively for audio files.
    path_out : Path, optional
        Destination .npz path.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from file stem (basename without extension) to its embedding array.
        Files for which no embedding is produced (e.g., no vocals) are skipped.
    """
    session = Session(str(config_path))
    path_out.parent.mkdir(parents=True, exist_ok=True)

    embeddings: Dict[str, np.ndarray] = {}
    
    # 扫描多种音频格式
    audio_files = (list(audio_dir.glob("**/*.mp3")) +
                   list(audio_dir.glob("**/*.wav")) +
                   list(audio_dir.glob("**/*.m4a")))
    
    for i, audio_path in enumerate(sorted(audio_files), 1):
        print(f"\r处理进度: {i}/{len(audio_files)} ({i*100//len(audio_files)}%)", end='', flush=True)
        emb = session.inference(str(audio_path))
        filename = audio_path.stem
        embeddings[filename] = emb
    print(f"\n✅ 完成！生成了 {len(embeddings)} 个歌词嵌入")

    # Save in .npz
    np.savez(path_out, **embeddings)
    return embeddings


def run_inference_from_list(
    config_path: Path,
    audio_paths: List[Path],
    path_out: Optional[Path] = None,
    save_interval: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Run frozen-encoder inference on a list of audio files.
    
    Parameters
    ----------
    config_path : Path
        Path to the OmegaConf YAML
    audio_paths : List[Path]
        List of audio file paths to process
    path_out : Path, optional
        Destination .npz path
    save_interval : int, default=10
        Save intermediate results and clear GPU cache every N files to prevent memory leak
        
    Returns
    -------
    dict[str, np.ndarray]
        Mapping from file stem to embedding array
    """
    session = Session(str(config_path))
    if path_out:
        path_out.parent.mkdir(parents=True, exist_ok=True)
    
    embeddings: Dict[str, np.ndarray] = {}
    temp_files = []  # 临时文件列表
    batch_count = 0
    
    for i, audio_path in enumerate(audio_paths, 1):
        print(f"\r处理进度: {i}/{len(audio_paths)} ({i*100//len(audio_paths)}%)", end='', flush=True)
        
        emb = session.inference(str(audio_path))
        filename = Path(audio_path).stem
        
        # 确保 embedding 在 CPU 上（numpy 数组）
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        embeddings[filename] = emb
        
        # 每首歌处理完后立即清理，防止累积
        del emb
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 定期保存到临时文件并清理内存
        if i % save_interval == 0:
            # 保存到临时文件
            if path_out:
                temp_file = path_out.parent / f"{path_out.stem}_temp_{batch_count}.npz"
                np.savez(temp_file, **embeddings)
                temp_files.append(temp_file)
                batch_count += 1
            
            embeddings.clear()
            
            # 清理GPU缓存
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f" [已处理 {i}/{len(audio_paths)}, 清理缓存]", end='')
    
    # 保存剩余的到临时文件
    if embeddings and path_out:
        temp_file = path_out.parent / f"{path_out.stem}_temp_{batch_count}.npz"
        np.savez(temp_file, **embeddings)
        temp_files.append(temp_file)
    
    total_count = len(embeddings)
    print(f"\n✅ 完成！生成了歌词嵌入，合并临时文件...")
    
    # 合并所有临时文件到最终文件
    if path_out and temp_files:
        all_embeddings = {}
        for temp_file in temp_files:
            data = np.load(temp_file)
            all_embeddings.update(dict(data.items()))
            total_count = len(all_embeddings)
            data.close()
            temp_file.unlink()  # 删除临时文件
        
        np.savez(path_out, **all_embeddings)
        print(f"✅ 合并完成！总计 {total_count} 个歌词嵌入")
        return all_embeddings
    
    return embeddings


def run_inference_single(
    config_path: Path,
    audio_path: Path,
) -> np.ndarray:
    """
    Run frozen-encoder inference on a single audio file.

    Parameters
    ----------
    config_path : Path
        Path to Hydra/OmegaConf config (contains data/preproc settings).
    audio_path : Path
        Path to the input audio file.

    Returns
    -------
    np.ndarray
        Embedding array.
    """
    session = Session(str(config_path))
    return session.inference(str(audio_path))


def run_estimate_time(
    config_path: Path,
    audio_dir: Path,
    *,
    sample_size: int = 200,
    start_after: int = 5,
    seed: int = 42,
) -> None:
    """
    Estimate average preprocessing/transcription/text-encoding/total times.

    Parameters
    ----------
    checkpoint_path : Path
        Placeholder for symmetry (unused).
    config_path : Path
        Path to Hydra/OmegaConf config.
    audio_dir : Path
        Directory where audio files live; recursively scans for *.mp3.
    sample_size : int, default 200
        Random sample size.
    start_after : int, default 5
        Warm-up iterations to skip.
    seed : int, default 42
        RNG seed for sampling.
    """
    files = sorted(audio_dir.glob("**/*.mp3"))
    if not files:
        raise FileNotFoundError(f"No files found under: {audio_dir}")

    rng = np.random.default_rng(seed)
    if len(files) > sample_size:
        files = list(rng.choice(files, size=sample_size, replace=False))
    else:
        files = list(files)

    logger.info(f"Timing on {len(files)} files (warm-up skip: {start_after}).")
    session = Session(str(config_path))
    session.estimate_inference_time([str(p) for p in files], start_after=start_after)
