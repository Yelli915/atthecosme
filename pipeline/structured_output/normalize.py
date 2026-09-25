"""Stage 3. 구조화 출력 — 요청마다 동적으로 만든 Enum(후보 ID + NOT_FOUND)으로
출력을 제약해, DB에 없는 성분명을 모델이 생성할 수 없게 한다.

Ollama(로컬 LLM)의 `format` 파라미터에 JSON Schema를 넘겨 구조적으로 강제한다.
모델이 Enum 밖의 값을 반환하면(제약이 완벽히 지켜지지 않는 모델도 있음) NOT_FOUND로 처리한다
— 이 안전장치가 없으면 환각 차단이 모델의 스키마 준수 능력에만 의존하게 된다.
"""

import json
import math
from dataclasses import dataclass

import ollama

from pipeline.config import NOT_FOUND_LABEL, OLLAMA_MODEL
from pipeline.retrieval.hybrid import Candidate

CANDIDATE_PREFIX = "cand_"


@dataclass
class NormalizationResult:
    ocr_text: str
    chosen_ingredient_id: int | None  # None이면 NOT_FOUND
    raw_model_output: str
    token_confidence: float = 1.0  # 생성된 토큰 중 최저 확률 (exp(min(logprob))). 후보가 없어 모델을 호출하지 않은 경우 1.0(확정적 NOT_FOUND)


def _candidate_enum_value(candidate: Candidate) -> str:
    return f"{CANDIDATE_PREFIX}{candidate.ingredient.id}"


def _build_schema(candidates: list[Candidate]) -> dict:
    enum_values = [_candidate_enum_value(c) for c in candidates] + [NOT_FOUND_LABEL]
    return {
        "type": "object",
        "properties": {
            "ingredient_id": {"type": "string", "enum": enum_values},
        },
        "required": ["ingredient_id"],
    }


def _build_prompt(ocr_text: str, candidates: list[Candidate]) -> str:
    lines = [
        "화장품 라벨에서 OCR로 추출한 성분 표기를 성분 마스터 DB 후보 중 하나로 매핑하세요.",
        "후보 중 일치하는 것이 없으면 반드시 NOT_FOUND를 선택하세요. 후보 목록에 없는 값을 만들어내지 마세요.",
        f"\nOCR 텍스트: {ocr_text!r}",
        "\n후보:",
    ]
    for c in candidates:
        ing = c.ingredient
        alias_part = f" (이표기: {', '.join(ing.aliases)})" if ing.aliases else ""
        kr_part = f" / {ing.kr_name}" if ing.kr_name else ""
        lines.append(f"- {_candidate_enum_value(c)}: {ing.inci_name}{kr_part}{alias_part}")
    return "\n".join(lines)


def _value_token_confidence(raw: str, token_logprobs: list[dict], value: str) -> float:
    """`raw` 전체가 아니라 실제 선택값(`value`) 문자열에 해당하는 토큰들의 최저 확률만 사용한다.

    `{"ingredient_id": ...}` 중 키 이름·중괄호 등 스키마 고정 부분은 토크나이저 특성상
    확률이 낮게 나올 수 있는데, 이는 모델의 '선택'과 무관한 노이즈라 신뢰도 계산에서 제외해야 한다.
    """
    if not token_logprobs:
        return 1.0
    value_start = raw.find(value)
    if value_start == -1:
        return math.exp(min(t["logprob"] for t in token_logprobs))
    value_end = value_start + len(value)

    offset = 0
    relevant_logprobs = []
    for t in token_logprobs:
        start, end = offset, offset + len(t["token"])
        offset = end
        if start < value_end and end > value_start:  # 값 구간과 겹치는 토큰만
            relevant_logprobs.append(t["logprob"])
    if not relevant_logprobs:
        relevant_logprobs = [t["logprob"] for t in token_logprobs]
    return math.exp(min(relevant_logprobs))


def normalize_token(ocr_text: str, candidates: list[Candidate], model: str = OLLAMA_MODEL) -> NormalizationResult:
    """OCR 텍스트 하나를 후보 중 하나(또는 NOT_FOUND)로 정규화한다."""
    if not candidates:
        return NormalizationResult(ocr_text=ocr_text, chosen_ingredient_id=None, raw_model_output="")

    schema = _build_schema(candidates)
    prompt = _build_prompt(ocr_text, candidates)

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format=schema,
        logprobs=True,
    )
    raw = response["message"]["content"]
    token_logprobs = response.get("logprobs") or []

    valid_ids = {c.ingredient.id for c in candidates}
    chosen_id: int | None = None
    value = NOT_FOUND_LABEL
    try:
        parsed = json.loads(raw)
        value = parsed.get("ingredient_id", NOT_FOUND_LABEL)
        if value.startswith(CANDIDATE_PREFIX):
            candidate_id = int(value[len(CANDIDATE_PREFIX) :])
            # 스키마 제약이 이론상 이 범위를 보장하지만, 모델이 스키마를 어길 가능성에 대비해 재검증한다.
            if candidate_id in valid_ids:
                chosen_id = candidate_id
    except (json.JSONDecodeError, ValueError, AttributeError):
        chosen_id = None  # 파싱 실패 시 안전하게 NOT_FOUND로 처리

    token_confidence = _value_token_confidence(raw, token_logprobs, value)

    return NormalizationResult(
        ocr_text=ocr_text, chosen_ingredient_id=chosen_id, raw_model_output=raw, token_confidence=token_confidence
    )
