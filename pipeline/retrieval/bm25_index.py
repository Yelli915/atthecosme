"""BM25 희소 검색 인덱스.

성분명(INCI + 한글 + 이표기)을 토큰화해 BM25 인덱스를 구축하고,
OCR 토큰 문자열에 대해 상위 후보 성분을 반환한다.
"""

import re

from rank_bm25 import BM25Okapi

from pipeline.retrieval.db import Ingredient, load_all


def _tokenize(text: str) -> list[str]:
    """공백·구두점 기준 단순 토큰화 + 소문자화.

    한글/영문 혼용 성분명이라 형태소 분석기 없이 시작. 검색 실패(오탈자)가
    많이 관찰되면 자모 n-gram 토큰화로 교체 검토 (계획.md 2단계 오탈자 집계 이후 결정).
    """
    text = text.lower()
    return re.findall(r"[a-z0-9가-힣]+", text)


class Bm25Index:
    def __init__(self, ingredients: list[Ingredient] | None = None):
        self.ingredients = ingredients if ingredients is not None else load_all()
        corpus = [_tokenize(ing.search_text()) for ing in self.ingredients]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int) -> list[tuple[Ingredient, float]]:
        """query와 가장 유사한 상위 top_k 성분을 (성분, 점수) 튜플로 반환한다."""
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(self.ingredients, scores), key=lambda p: p[1], reverse=True)
        return ranked[:top_k]
