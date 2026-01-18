import numpy as np

from semantic_search.embeddings import EmbeddingGenerator


def test_embedding_returns_numpy_array():
    gen = EmbeddingGenerator("BAAI/bge-small-en-v1.5")
    vec = gen.embed_single("hello")

    arr = np.array(vec)
    assert isinstance(arr, np.ndarray)


def test_embedding_dimensions_384():
    gen = EmbeddingGenerator("BAAI/bge-small-en-v1.5")
    vec = gen.embed_single("hello")
    assert len(vec) == 384


def test_same_text_same_embedding():
    gen = EmbeddingGenerator("BAAI/bge-small-en-v1.5")
    v1 = gen.embed_single("same text")
    v2 = gen.embed_single("same text")

    # compare as numpy arrays for equality
    a1 = np.array(v1, dtype=np.float32)
    a2 = np.array(v2, dtype=np.float32)

    assert np.allclose(a1, a2, atol=1e-6)
