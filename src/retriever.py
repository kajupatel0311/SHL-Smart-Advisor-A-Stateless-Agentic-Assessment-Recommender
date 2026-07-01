from typing import List, Tuple

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class SHLRetriever:
    def __init__(self, catalog: List[dict]):
        self.catalog = catalog
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.index = None
        self._build_index()

    def _build_index(self) -> None:
        if not self.catalog:
            return
        documents = []
        for item in self.catalog:
            text_parts = [
                f"Assessment Name: {item.get('name') or item.get('title') or ''}",
                f"Category/Test Type: {item.get('test_type') or item.get('category') or ''}",
                f"Description: {item.get('description') or item.get('summary') or ''}",
                f"Keys: {' '.join(item.get('keys', []) or [])}",
                f"Job Levels: {' '.join(item.get('job_levels', []) or [])}",
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
