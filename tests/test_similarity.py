import pytest
import numpy as np

from semantic_search.similarity import cosine_similarity
from semantic_search.embeddings import EmbeddingGenerator


def test_cosine_identical_vectors_is_one():
    a = [1, 2, 3]
    b = [1, 2, 3]
    assert cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-6)


def test_cosine_orthogonal_vectors_is_zero():
    a = [1, 0]
    b = [0, 1]
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_similarity_mass_mass_vs_ill_do_it_later_high():
    gen = EmbeddingGenerator("BAAI/bge-small-en-v1.5")
    v1 = gen.embed_single("I mass mass")
    v2 = gen.embed_single("I'll do it later")

    score = cosine_similarity(v1, v2)
    assert score > 0.55


def test_similarity_fix_bug_vs_pizza_low():
    gen = EmbeddingGenerator("BAAI/bge-small-en-v1.5")
    v1 = gen.embed_single("Fix the bug")
    v2 = gen.embed_single("Best pizza recipe")

    score = cosine_similarity(v1, v2)
    assert score < 0.50

