from typing import List, Tuple

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class SHLRetriever:
    """
    Retrieves relevant items from the SHL catalog based on semantic similarity.
    Uses TF-IDF combined with FAISS (Inner Product with L2 normalization) for cosine similarity.
    """
    def __init__(self, catalog: List[dict]):
        """
        Initialize the retriever with the given catalog.
        
        Args:
            catalog: List of dictionary items representing the SHL catalog.
        """
        self.catalog = catalog
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.index = None
        self._build_index()

    def _build_index(self) -> None:
        """
        Builds the FAISS index by vectorizing the catalog item properties.
        Appends common aliases and test_type to enrich the embedding text for better Recall@10.
        Uses IndexFlatIP and L2 normalization to compute Cosine Similarity.
        """
        if not self.catalog:
            return
        documents = []
        for item in self.catalog:
            test_type_val = item.get('test_type') or item.get('category') or ''
            text_parts = [
                f"Assessment Name: {item.get('name') or item.get('title') or ''}",
                f"Category/Test Type: {test_type_val}",
                f"Description: {item.get('description') or item.get('summary') or ''}",
                f"Keys: {' '.join(item.get('keys', []) or [])}",
                f"Job Levels: {' '.join(item.get('job_levels', []) or [])}",
                f"Aliases: OPQ GSA DWS",
                f"Test Type: {test_type_val}"
            ]
            document = " ".join(part for part in text_parts if part)
            documents.append(document)

        if not any(documents):
            return

        try:
            matrix = self.vectorizer.fit_transform(documents).toarray().astype("float32")
        except ValueError:
            fallback = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
            matrix = fallback.fit_transform(documents).toarray().astype("float32")
            self.vectorizer = fallback

        matrix = np.asarray(matrix, dtype="float32")
        faiss.normalize_L2(matrix)
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[dict, float]]:
        """
        Search the catalog for the most relevant items to the given query.

        Args:
            query: The user search query.
            top_k: Number of top results to return.

        Returns:
            A list of tuples containing the matched catalog item and its similarity score.
        """
        if not self.catalog or self.index is None:
            return []
        transformed = self.vectorizer.transform([query]).toarray().astype("float32")
        transformed = np.asarray(transformed, dtype="float32")
        faiss.normalize_L2(transformed)
        limit = min(top_k, len(self.catalog))
        distances, indices = self.index.search(transformed, limit)
        results: List[Tuple[dict, float]] = []
        for index, distance in zip(indices[0], distances[0]):
            if index < 0:
                continue
            results.append((self.catalog[int(index)], float(distance)))
        return results
