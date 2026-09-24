"""기획서 5.2 추출 지표 계산.

모든 함수는 순수 함수로 작성해 실제 벤치마크 실행 없이도 단위 테스트할 수 있게 한다.
"""


def ingredient_recall(predicted_ids: set[int], gold_ids: set[int]) -> float:
    """성분 재현율 = 정답 성분 중 추출된 비율. 핵심 지표 (누락 방지)."""
    if not gold_ids:
        return 1.0
    return len(predicted_ids & gold_ids) / len(gold_ids)


def ingredient_precision(predicted_ids: set[int], gold_ids: set[int]) -> float:
    """성분 정밀도 = 추출 성분 중 정답인 비율. 검수자 부담 관리 지표."""
    if not predicted_ids:
        return 1.0 if not gold_ids else 0.0
    return len(predicted_ids & gold_ids) / len(predicted_ids)


def exact_match(predicted: str, gold: str) -> bool:
    """정규화된 문자열 완전 일치 (제품명 EM 등에 사용)."""
    return predicted.strip().lower() == gold.strip().lower()


def aggregate_recall(per_sample_predicted: list[set[int]], per_sample_gold: list[set[int]]) -> float:
    """전체 벤치마크셋 기준 마이크로 평균 재현율 (분모 = 전체 정답 성분 개수 합).

    계획.md 1단계 "벤치마크셋 전체 정답 성분 개수(재현율 분모)" 확인 항목과 연결된다.
    """
    total_gold = sum(len(g) for g in per_sample_gold)
    if total_gold == 0:
        return 1.0
    total_hit = sum(len(p & g) for p, g in zip(per_sample_predicted, per_sample_gold))
    return total_hit / total_gold
