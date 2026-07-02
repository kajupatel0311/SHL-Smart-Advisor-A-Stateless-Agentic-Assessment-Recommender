from typing import List, Tuple

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class SHLRetriever:
    """
    Retrieves relevant SHL catalog items based on query similarity.
    Uses TF-IDF with FAISS (Inner Product + L2 normalization = Cosine Similarity).
    The catalog uses the fields: 'name', 'link', 'keys', 'description', 'job_levels'.
    """

    def __init__(self, catalog: List[dict]):
        """
        Initialize the retriever with the given catalog.

        Args:
            catalog: List of SHL catalog items.
        """
        self.catalog = catalog
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.index = None
        self._build_index()

    def _build_index(self) -> None:
        """Builds the FAISS index by vectorizing catalog items for semantic search."""
        if not self.catalog:
            return

        documents = []
        for item in self.catalog:
            # Note: our catalog uses 'name'/'link'/'keys'/'description'/'job_levels'
            name = item.get("name") or item.get("title") or ""
            keys_list = item.get("keys") or []
            keys_str = " ".join(keys_list)
            description = item.get("description") or item.get("summary") or ""
            job_levels = " ".join(item.get("job_levels") or [])
            languages = " ".join(item.get("languages") or [])
            duration = item.get("duration") or ""

            text_parts = [
                f"Assessment Name: {name}",
                f"Test Type / Keys: {keys_str}",
                f"Description: {description[:500]}",
                f"Job Levels: {job_levels}",
                f"Languages: {languages}",
                f"Duration: {duration}",
                # Common abbreviation aliases to improve recall
                "OPQ OPQ32r GSA DWS Verify MQ SJT",
            ]
            documents.append(" ".join(part for part in text_parts if part))

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

    def search(self, query: str, top_k: int = 10) -> List[Tuple[dict, float]]:
        """
        Search the catalog for the most relevant items to the given query.

        Args:
            query: The user search query (all user messages combined for best context).
            top_k: Number of top results to return.

        Returns:
            A list of (catalog_item, similarity_score) tuples.
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
