
import os
import numpy as np

from semantic_search.embeddings import EmbeddingGenerator
from semantic_search.similarity import find_top_k


class DocumentIndex:
    def __init__(self, name: str, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.name = name
        self.model_name = model_name

        self.documents: list[str] = []
        self.embeddings: np.ndarray | None = None  # will store a 2D numpy matrix

        self.embedder = EmbeddingGenerator(model_name)

    def add_documents(self, docs: list[str]):
        """
        Add documents and create embeddings.
        """
        if not isinstance(docs, list) or len(docs) == 0:
            raise ValueError("docs must be a non-empty list of strings")

        # store documents
        self.documents.extend(docs)

        # embed documents
        vectors = self.embedder.embed_batch(docs)  # list of vectors

        # convert to numpy matrix
        new_matrix = np.array(vectors, dtype=np.float32)

        # if embeddings already exist, append vertically
        if self.embeddings is None:
            self.embeddings = new_matrix
        else:
            self.embeddings = np.vstack([self.embeddings, new_matrix])

    def search(self, query: str, top_k: int = 5):
        """
        Search the stored documents.
        Returns list of dicts: {"document": ..., "score": ...}
        """
        if len(self.documents) == 0 or self.embeddings is None:
            return []

        # embed query
        query_vec = self.embedder.embed_single(query)

        # find top matches using similarity.py
        matches = find_top_k(query_vec, self.embeddings, k=top_k)

        results = []
        for idx, score in matches:
            results.append({
                "document": self.documents[idx],
                "score": score
            })

        return results

    def save(self, path: str):
        """
        Save index to .npz file.
        """
        if self.embeddings is None or len(self.documents) == 0:
            raise ValueError("Index is empty. Add documents before saving.")

        # ensure extension
        if not path.endswith(".npz"):
            path = path + ".npz"

        np.savez(
            path,
            name=self.name,
            model_name=self.model_name,
            documents=np.array(self.documents, dtype=object),
            embeddings=self.embeddings,
        )

    @classmethod
    def load(cls, path: str):
        """
        Load index from .npz file and return DocumentIndex object.
        """
        if not path.endswith(".npz"):
            path = path + ".npz"

        if not os.path.exists(path):
            raise FileNotFoundError(f"Index file not found: {path}")

        data = np.load(path, allow_pickle=True)

        name = str(data["name"])
        model_name = str(data["model_name"])
        documents = data["documents"].tolist()
        embeddings = data["embeddings"]

        idx = cls(name=name, model_name=model_name)
        idx.documents = documents
        idx.embeddings = embeddings

        return idx
