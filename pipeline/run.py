"""Stage 1~5 전체 파이프라인 오케스트레이션.

    라벨 이미지 + 제출 전성분
        → Stage 1 OCR
        → Stage 2 하이브리드 검색 (토큰별 Top-5 후보)
        → Stage 3 구조화 출력 (후보 중 선택 또는 NOT_FOUND)
        → Stage 4 대조 (누락/추가/불일치)
        → Stage 5 신뢰도 분기 (원클릭 승인 vs 집중 검수)
"""

from dataclasses import dataclass, field

from pipeline.config import CONFIDENCE_THRESHOLD
from pipeline.match.match import MatchResult, match, resolve_submitted_names
from pipeline.ocr.engine import OcrEngine, OcrToken, split_ingredient_tokens
from pipeline.retrieval.hybrid import HybridRetriever
from pipeline.structured_output.normalize import NormalizationResult, normalize_token


@dataclass
class TokenNormalization:
    result: NormalizationResult
    retrieval_confidence: float  # Stage 2 top-1 후보 점수
    combined_confidence: float  # min(검색 신뢰도, 토큰 신뢰도) — 결합 방식은 임시, records/04에서 실측 후 확정


@dataclass
class PipelineResult:
    ocr_tokens: list[OcrToken]
    normalizations: list[TokenNormalization]
    match_result: MatchResult
    min_confidence: float
    auto_approved: bool


def run_pipeline(image_path: str, submitted_names: list[str], retriever: HybridRetriever | None = None) -> PipelineResult:
    retriever = retriever or HybridRetriever()

    # Stage 1
    ocr_tokens = OcrEngine().extract(image_path)
    ingredient_texts = split_ingredient_tokens(ocr_tokens)

    # Stage 2 + Stage 3 (토큰마다 검색 → 정규화)
    normalizations: list[TokenNormalization] = []
    for text in ingredient_texts:
        candidates = retriever.search(text)
        norm_result = normalize_token(text, candidates)
        top_score = candidates[0].score if candidates else 0.0
        combined = min(top_score, norm_result.token_confidence)
        normalizations.append(
            TokenNormalization(result=norm_result, retrieval_confidence=top_score, combined_confidence=combined)
        )

    label_ingredient_ids = {
        n.result.chosen_ingredient_id for n in normalizations if n.result.chosen_ingredient_id is not None
    }

    # Stage 4
    submitted_map = resolve_submitted_names(submitted_names, retriever)
    match_result = match(label_ingredient_ids, submitted_map)

    # Stage 5 — 신뢰도 결합: 검색 신뢰도(Stage 2)와 구조화 출력 토큰 신뢰도(Stage 3, logprobs) 중
    # 최솟값을 사용 (약한 고리 기준). Ollama가 logprobs를 반환하는 것은 확인됨(정상 동작).
    # 다만 가중 평균 등 다른 결합 방식과의 비교·최종 확정은 records/04-신뢰도임계치.md에서
    # 실측 후 진행 — 지금은 계획.md 4단계가 언급한 후보 중 하나를 임시로 쓰는 상태.
    confidences = [n.combined_confidence for n in normalizations]
    min_confidence = min(confidences) if confidences else 0.0
    has_not_found = any(n.result.chosen_ingredient_id is None for n in normalizations)
    auto_approved = (
        min_confidence >= CONFIDENCE_THRESHOLD and not has_not_found and not match_result.has_discrepancy
    )

    return PipelineResult(
        ocr_tokens=ocr_tokens,
        normalizations=normalizations,
        match_result=match_result,
        min_confidence=min_confidence,
        auto_approved=auto_approved,
    )
