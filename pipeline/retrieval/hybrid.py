"""Stage 2. 하이브리드 검색 — BM25 + Dense Vector 결합, OCR 토큰별 Top-5 후보 반환.

두 검색기의 점수 스케일이 다르므로(BM25는 비음수 비정규화 점수, Dense는 코사인 유사도)
후보 집합 내에서 각각 min-max 정규화한 뒤 가중합으로 결합한다.
"""

from dataclasses import dataclass

from pipeline.config import BM25_WEIGHT, DENSE_WEIGHT, TOP_K_CANDIDATES
from pipeline.retrieval.bm25_index import Bm25Index
from pipeline.retrieval.db import Ingredient, load_all
from pipeline.retrieval.vector_index import VectorIndex


@dataclass
class Candidate:
    ingredient: Ingredient
    score: float  # 0~1, 결합 후 점수 (신뢰도 산출에 사용 — Stage 4/5 참고)


def _min_max_normalize(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}  # 전부 동점이면 만점 처리
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRetriever:
    def __init__(self, ingredients: list[Ingredient] | None = None):
        ingredients = ingredients if ingredients is not None else load_all()
        self._bm25 = Bm25Index(ingredients)
        self._vector = VectorIndex(ingredients)
        self._by_id = {ing.id: ing for ing in ingredients}

    def search(self, query: str, top_k: int = TOP_K_CANDIDATES) -> list[Candidate]:
        # 후보 폭을 top_k보다 넓게 가져가야 두 검색기의 합집합에서 재정렬이 의미가 있다.
        pool_k = max(top_k * 3, 10)
        bm25_hits = self._bm25.search(query, pool_k)
        dense_hits = self._vector.search(query, pool_k)

        bm25_scores = {ing.id: score for ing, score in bm25_hits}
        dense_scores = {ing.id: score for ing, score in dense_hits}
        bm25_norm = _min_max_normalize(bm25_scores)
        dense_norm = _min_max_normalize(dense_scores)

        all_ids = set(bm25_norm) | set(dense_norm)
        combined = {
            id_: BM25_WEIGHT * bm25_norm.get(id_, 0.0) + DENSE_WEIGHT * dense_norm.get(id_, 0.0)
            for id_ in all_ids
        }
        ranked_ids = sorted(combined, key=lambda i: combined[i], reverse=True)[:top_k]
        return [Candidate(ingredient=self._by_id[i], score=combined[i]) for i in ranked_ids]
