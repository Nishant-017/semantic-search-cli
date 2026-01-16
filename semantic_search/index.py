
import pickle

from semantic_search.embeddings import EmbeddingGenerator
from semantic_search.similarity import find_top_k


class DocumentIndex:
    def __init__(self, name: str, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.name = name
        self.model_name = model_name

        self.documents = []
        self.embeddings = []

        self.embedder = EmbeddingGenerator(model_name)

    def add_documents(self, docs: list[str]):
        if not isinstance(docs, list) or len(docs) == 0:
            raise ValueError("docs must be a non-empty list of strings")

        self.documents.extend(docs)
        vectors = self.embedder.embed_batch(docs)
        self.embeddings.extend(vectors)

    def search(self, query: str, top_k: int = 5):
        if len(self.documents) == 0:
            return []

        query_vec = self.embedder.embed_single(query)

        top_matches = find_top_k(query_vec, self.embeddings, k=top_k)

        results = []
        for idx, score in top_matches:
            results.append({
                "document": self.documents[idx],
                "score": score
            })

        return results

    def save(self, path: str):
        """
        Save the index to disk.
        """
        data = {
            "name": self.name,
            "model_name": self.model_name,
            "documents": self.documents,
            "embeddings": self.embeddings,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str):
        """
        Load an index from disk and return a DocumentIndex object.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        idx = cls(name=data["name"], model_name=data["model_name"])
        idx.documents = data["documents"]
        idx.embeddings = data["embeddings"]

        return idx
