import os
from semantic_search.index import DocumentIndex


def test_build_and_load_preserves_documents(tmp_path):
    idx = DocumentIndex("test_index")

    docs = ["apple is fruit", "python is language", "pizza is tasty"]
    idx.add_documents(docs)

    out_path = tmp_path / "my_index"
    idx.save(str(out_path))

    loaded = DocumentIndex.load(str(out_path))

    assert loaded.documents == docs
    assert loaded.embeddings is not None
    assert len(loaded.documents) == 3


def test_search_returns_correct_top_k(tmp_path):
    idx = DocumentIndex("test_index")

    docs = ["FastAPI is for APIs", "Pizza recipe", "Debugging in Python"]
    idx.add_documents(docs)

    out_path = tmp_path / "search_index"
    idx.save(str(out_path))

    loaded = DocumentIndex.load(str(out_path))

    results = loaded.search("python debugging", top_k=2)
    assert len(results) == 2


def test_top_result_has_highest_score(tmp_path):
    idx = DocumentIndex("test_index")

    docs = ["Debugging tips for Python", "Best pizza in Mumbai", "Football match results"]
    idx.add_documents(docs)

    out_path = tmp_path / "rank_index"
    idx.save(str(out_path))

    loaded = DocumentIndex.load(str(out_path))

    results = loaded.search("python debugging", top_k=3)

    # ensure sorted by highest score
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
