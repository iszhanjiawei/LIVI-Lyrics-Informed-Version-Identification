from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Dict, Union, Sequence

import numpy as np
import pandas as pd
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from livi.apps.retrieval_eval.data import convert_to_dict
from livi.core.data.utils.io_toolbox import save_embeddings
from loguru import logger

# -------------- Internal Helpers --------------


class TextEncoder:
    """
    Minimal wrapper around `SentenceTransformer` with optional chunking.

    Chunking behavior
    -----------------
        - If an input is a `str` and chunking is enabled, we construct chunks via:
        `self.chunker.create_documents([text])` and take the unique
        `doc.page_content` values.
        - If an input is `list[str]`, it is treated as pre-chunked and used as-is.
    """

    def __init__(self, model_name: str, chunking: bool = False):
        """
        Initialize the encoder and (optionally) a text chunker.

        Parameters
        ----------
        model_name : str
            A Sentence-Transformers model identifier.
        chunking : bool, default=False
            Whether to enable automatic text chunking.
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

        self.chunker: Optional[RecursiveCharacterTextSplitter] = None
        if chunking:
            self.chunker = RecursiveCharacterTextSplitter(
                chunk_size=200,
                chunk_overlap=0,
                length_function=len,
                is_separator_regex=True,
            )

    def encode(self, inputs: List[str], batch_size: Optional[int] = 8) -> np.ndarray:
        """
        Encode a batch of strings.

        Parameters
        ----------
        inputs : list of str
            Raw text strings to encode.
        batch_size : int, optional
            Batch size for model inference.

        Returns
        -------
        np.ndarray
            Array of shape (N, D) with L2-normalized embeddings.
        """
        kwargs = {"task": "retrieval.query", "prompt": "retrieval.query"} if "jina" in self.model_name else {}

        embeddings = self.model.encode(inputs, normalize_embeddings=True, device=self.device, batch_size=batch_size, **kwargs)
        
        # 确保返回numpy数组（在CPU上），避免GPU张量累积
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        # 强制清理 GPU 缓存，防止 Text Encoder 的张量累积
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return embeddings

    def encode_chunks(
        self,
        inputs: Sequence[Union[str, List[str]]],
        get_single_embedding: Optional[bool] = False,
        batch_size: Optional[int] = 8,
    ) -> np.ndarray:
        """
        Encode chunked input.
        - If the item is a `str`, it is chunked using `self.chunker`
        - If the item is `list[str]`, it is treated as already chunked.

        Parameters
        ----------
        inputs : Sequence[Union[str, list[str]]]
            Texts to encode; either raw strings or pre-chunked lists of strings.
        get_single_embedding : bool, optional
            If True, mean-pool chunk embeddings into a single (1, D) vector
            per ID.
        batch_size : int, optional
            Batch size for encoding.

        Returns
        -------
        np.ndarray
            Array of shape (N, D) with L2-normalized embeddings, with N the number of chunks.
        """
        try:
            # Need to chunk input text if it's a string
            if isinstance(inputs, str):
                if self.chunker is None:
                    raise ValueError(
                        "Chunker is not defined. Re-create TextEncoder with chunking=True "
                        "or pass pre-chunked lists instead of strings."
                    )
                docs = self.chunker.create_documents([inputs])
                chunks = list({doc.page_content for doc in docs})
            elif isinstance(inputs, list):
                chunks = inputs
            else:
                raise ValueError(f"Unsupported input type: {type(inputs)}")

            # handle empty-chunk edge case
            if not chunks:
                logger.warning(f"Empty chunks.")
                return

            # Encode chunks and optionally mean-pool
            embs = self.encode(chunks, batch_size=batch_size).astype(np.float32)
            if get_single_embedding:
                embs = np.mean(embs, axis=0, keepdims=True)
            
            # 显式释放chunks，避免累积
            del chunks
            
            # 强制清理 GPU 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception:
            logger.warning(f"Encoding failed.")
            return
        
        # 确保返回numpy数组（已经在CPU上）
        return embs


# --------------------------------------------------------------------
# Runners
# --------------------------------------------------------------------


@lru_cache(maxsize=2)
def _get_cached_text_encoder(
    model_name: str = "Alibaba-NLP/gte-multilingual-base", chunking: bool = False
) -> TextEncoder:
    """
    Return a cached `TextEncoder` instance.

    Parameters
    ----------
    model_name : str, default="Alibaba-NLP/gte-multilingual-base"
        Sentence-Transformers model name to load.
    chunking : bool, default=False
        Whether the cached encoder should be initialized with chunking.

    Returns
    -------
    TextEncoder
        Cached encoder instance (per unique argument combination).
    """
    return TextEncoder(model_name=model_name, chunking=chunking)


def encode_text(
    inputs: Sequence[Union[str, List[str]]],
    text_encoder: Optional[TextEncoder],
    model_name: Optional[str],
    chunking: Optional[bool] = False,
    batch_size: Optional[int] = 8,
    get_single_embedding: Optional[bool] = False,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Encode texts with optional chunking.

    Parameters
    ----------
    inputs : Sequence[Union[str, list[str]]]
        Raw strings or pre-chunked lists of strings.
    ids : list of str
        Unique IDs associated with `inputs` (required if `chunking=True`).
    text_encoder : TextEncoder, optional
        Pre-initialized encoder to reuse. If None, a cached encoder is built.
    model_name : str, optional
        Model name used when constructing a cached encoder (if `text_encoder` is None).
    chunking : bool, default=False
        If True, return per-ID chunk embeddings as a dict; otherwise return a single array.
    batch_size : int, optional
        Batch size for inference.
    get_single_embedding : bool, optional
        If True (and `chunking=True`), mean-pool chunk embeddings into (1, D) per ID.

    Returns
    -------
    np.ndarray or dict[str, np.ndarray]
        - If `chunking=False`: array of shape (N, D).
        - If `chunking=True`: mapping ID -> (num_chunks, D) or (1, D) if pooled.
    """
    enc = text_encoder or _get_cached_text_encoder(
        model_name=model_name or "Alibaba-NLP/gte-multilingual-base", chunking=chunking
    )

    # 检查 chunking 配置是否匹配（避免重复创建模型）
    if bool(enc.chunker) != chunking:
        logger.warning(f"TextEncoder chunking mismatch: expected {chunking}, got {bool(enc.chunker)}. This should not happen if using cached encoder.")
        enc = TextEncoder(model_name=model_name or enc.model_name, chunking=chunking)

    if chunking:
        return enc.encode_chunks(inputs, batch_size=batch_size, get_single_embedding=get_single_embedding)
    else:
        return enc.encode(list(inputs), batch_size=batch_size)
