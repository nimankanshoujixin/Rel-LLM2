from __future__ import annotations

import os
from typing import List, Optional, Sequence

import torch
from torch import Tensor

# Please run `pip install -U sentence-transformers`
from sentence_transformers import SentenceTransformer


class TextEmbedding:
    def __init__(
        self,
        embed_type,
        modelPath=None,
        device: Optional[torch.device | str] = None,
        devices: Optional[Sequence[str]] = None,
        encode_batch_size: Optional[int] = None,
        multi_process_chunk_size: Optional[int] = None,
        show_progress_bar: bool = False,
    ):
        if embed_type == 'mpnet':
            embed_type = "all-mpnet-base-v2"
        else:
            embed_type = "average_word_embeddings_glove.6B.300d"

        if modelPath is not None:
            modelPath = os.path.join(modelPath, embed_type)

        if modelPath is not None and os.path.exists(modelPath):
            self.model = SentenceTransformer(modelPath, device=device)
        else:
            # download from https://hf-mirror.com/ if unable to connect to huggingface
            self.model = SentenceTransformer(f"sentence-transformers/{embed_type}", device=device)
            if modelPath is not None:
                self.model.save(modelPath)

        normalized_devices = [str(item) for item in devices] if devices else []
        self.devices = [item for item in normalized_devices if item]
        self.encode_batch_size = encode_batch_size
        self.multi_process_chunk_size = multi_process_chunk_size
        self.show_progress_bar = show_progress_bar
        self._pool = None
        if len(self.devices) > 1:
            self._pool = self.model.start_multi_process_pool(target_devices=self.devices)

    def close(self) -> None:
        if self._pool is not None:
            self.model.stop_multi_process_pool(self._pool)
            self._pool = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __call__(self, sentences: List[str]) -> Tensor:
        if self._pool is not None:
            values = self.model.encode_multi_process(
                sentences,
                pool=self._pool,
                batch_size=self.encode_batch_size or 32,
                chunk_size=self.multi_process_chunk_size,
                show_progress_bar=self.show_progress_bar,
            )
            return torch.from_numpy(values)
        encoded = self.model.encode(
            sentences,
            convert_to_tensor=True,
            batch_size=self.encode_batch_size,
            show_progress_bar=self.show_progress_bar,
        )
        if isinstance(encoded, torch.Tensor):
            return encoded.cpu()
        return torch.as_tensor(encoded)
