"""Dense Vector 검색 인덱스 (FAISS + sentence-transformers).

BM25가 놓치는 의미적 유사(오탈자, 어순 변형, 이표기 미등록)를 보완하기 위한 축.
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline.config import DENSE_EMBEDDING_MODEL
from pipeline.retrieval.db import Ingredient, load_all


class VectorIndex:
    def __init__(self, ingredients: list[Ingredient] | None = None, model_name: str = DENSE_EMBEDDING_MODEL):
        self.ingredients = ingredients if ingredients is not None else load_all()
        self._model = SentenceTransformer(model_name)
        texts = [ing.search_text() for ing in self.ingredients]
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype="float32")
        self._index = faiss.IndexFlatIP(embeddings.shape[1])  # 정규화된 벡터 → 내적 = 코사인 유사도
        self._index.add(embeddings)

    def search(self, query: str, top_k: int) -> list[tuple[Ingredient, float]]:
        query_vec = self._model.encode([query], normalize_embeddings=True)
        query_vec = np.asarray(query_vec, dtype="float32")
        scores, indices = self._index.search(query_vec, top_k)
        result = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            result.append((self.ingredients[idx], float(score)))
        return result
