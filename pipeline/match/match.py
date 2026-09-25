"""Stage 4. 대조 — 라벨에서 추출한 성분 ID 집합 vs 브랜드 제출 데이터 성분 ID 집합 비교.

기획서 3장 기준 미구현 항목이었던 부분. `계획.md` 3단계 대응.

브랜드 제출 데이터는 자유 텍스트 성분명 목록으로 들어온다고 가정하고,
먼저 DB에 매핑(resolve_submitted_names)한 뒤 ID 집합으로 비교한다.
"""

from dataclasses import dataclass, field

from pipeline.retrieval.db import Ingredient, get_by_id, load_all
from pipeline.retrieval.hybrid import HybridRetriever

# 제출 텍스트 하나가 정확 매칭 후보가 없을 때, 하이브리드 검색 1위를 채택할지 판단하는 하한선.
# TODO(records/04-신뢰도임계치.md): 실측 후 확정. 우선 보수적으로 0.7 유지.
SUBMITTED_FUZZY_ACCEPT_THRESHOLD = 0.7


@dataclass
class MatchResult:
    missing: list[Ingredient] = field(default_factory=list)  # 제출 데이터엔 있으나 라벨에서 못 찾음 (재현율 리스크)
    extra: list[Ingredient] = field(default_factory=list)  # 라벨엔 있으나 제출 데이터엔 없음
    matched: list[Ingredient] = field(default_factory=list)  # 양쪽 다 있음
    unresolved_submitted: list[str] = field(default_factory=list)  # DB 매핑 자체에 실패한 제출 텍스트

    @property
    def has_discrepancy(self) -> bool:
        return bool(self.missing or self.extra or self.unresolved_submitted)


def resolve_submitted_names(names: list[str], retriever: HybridRetriever | None = None) -> dict[str, int | None]:
    """브랜드 제출 성분명(자유 텍스트) 목록을 DB ingredient id로 매핑한다.

    1) INCI명/한글명/이표기 완전 일치(대소문자 무시) 우선
    2) 실패 시 하이브리드 검색 1위를 임계치 이상일 때만 채택
    3) 그래도 없으면 None (unresolved)
    """
    retriever = retriever or HybridRetriever()
    ingredients = load_all()
    exact_index: dict[str, int] = {}
    for ing in ingredients:
        exact_index[ing.inci_name.strip().lower()] = ing.id
        if ing.kr_name:
            exact_index[ing.kr_name.strip().lower()] = ing.id
        for alias in ing.aliases:
            exact_index[alias.strip().lower()] = ing.id

    result: dict[str, int | None] = {}
    for name in names:
        key = name.strip().lower()
        if key in exact_index:
            result[name] = exact_index[key]
            continue
        candidates = retriever.search(name, top_k=1)
        if candidates and candidates[0].score >= SUBMITTED_FUZZY_ACCEPT_THRESHOLD:
            result[name] = candidates[0].ingredient.id
        else:
            result[name] = None
    return result


def match(label_ingredient_ids: set[int], submitted_ingredient_ids: dict[str, int | None]) -> MatchResult:
    """라벨에서 정규화된 성분 ID 집합과, 제출 데이터를 매핑한 ID 딕셔너리를 대조한다."""
    result = MatchResult()

    resolved_submitted_ids = {v for v in submitted_ingredient_ids.values() if v is not None}
    result.unresolved_submitted = [name for name, id_ in submitted_ingredient_ids.items() if id_ is None]

    missing_ids = resolved_submitted_ids - label_ingredient_ids
    extra_ids = label_ingredient_ids - resolved_submitted_ids
    matched_ids = label_ingredient_ids & resolved_submitted_ids

    result.missing = [get_by_id(i) for i in missing_ids]
    result.extra = [get_by_id(i) for i in extra_ids]
    result.matched = [get_by_id(i) for i in matched_ids]
    return result
