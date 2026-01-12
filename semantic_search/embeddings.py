# semantic_search/embeddings.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# FastEmbed provides local ONNX-based embedding models
# pip install fastembed
from fastembed import TextEmbedding


@dataclass
class ModelInfo:
    """
    Stores metadata about the embedding model.
    """
    name: str
    dimensions: Optional[int] = None


class EmbeddingGenerator:
    """
    Generates embeddings using FastEmbed.

    Requirements satisfied:
    - __init__(model_name): initialize with model choice
    - Lazy-load the model (only load when first used)
    - embed_single(text) -> returns embedding array
    - embed_batch(texts) -> returns list of embeddings
    - Store model info (name, dimensions)
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        """
        Initializes the generator with a model choice.
        Does NOT load the model yet (lazy-loading).
        """
        self.model_info = ModelInfo(name=model_name, dimensions=None)

        # This will store the actual FastEmbed model object once loaded.
        self._model: Optional[TextEmbedding] = None

    def _load_model(self) -> None:
        """
        Loads the FastEmbed model if it hasn't been loaded already.
        This method enforces lazy-loading.
        """
        if self._model is None:
            self._model = TextEmbedding(self.model_info.name)

    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single text string into a vector.

        Returns:
            np.ndarray: embedding vector
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        # Remove extra spaces; helps avoid embedding weird blank inputs
        text = text.strip()
        if not text:
            raise ValueError("text cannot be empty")

        # Lazy-load the model ONLY when embedding is requested
        self._load_model()
        assert self._model is not None  # for type checker

        # FastEmbed's embed() returns an iterator/generator of embeddings
        embedding_list = list(self._model.embed([text]))

        # We embedded 1 text so we should get exactly 1 embedding
        embedding = np.array(embedding_list[0], dtype=np.float32)

        # Set dimensions once (first time we embed anything)
        if self.model_info.dimensions is None:
            self.model_info.dimensions = int(embedding.shape[0])

        return embedding

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed a batch/list of texts.

        Returns:
            List[np.ndarray]: list of embeddings
        """
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")

        # Validate every text
        cleaned_texts: List[str] = []
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise TypeError(f"texts[{i}] must be a string")
            t = t.strip()
            if not t:
                raise ValueError(f"texts[{i}] cannot be empty")
            cleaned_texts.append(t)

        if len(cleaned_texts) == 0:
            raise ValueError("texts list cannot be empty")

        # Lazy-load model
        self._load_model()
        assert self._model is not None

        # Embed all texts at once for efficiency
        embeddings = list(self._model.embed(cleaned_texts))

        # Convert embeddings into numpy arrays
        vectors = [np.array(e, dtype=np.float32) for e in embeddings]

        # Set dimensions once
        if self.model_info.dimensions is None and len(vectors) > 0:
            self.model_info.dimensions = int(vectors[0].shape[0])

        return vectors
