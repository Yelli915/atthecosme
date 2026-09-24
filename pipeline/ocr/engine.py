"""Stage 1. OCR — 라벨 이미지에서 텍스트 + Bounding Box 좌표 추출.

PaddleOCR(로컬, 오픈소스)을 사용한다. 곡면 용기·반사광·소형 후면 폰트에서
텍스트를 놓치는 경우(OCR 누락)는 이후 어떤 단계로도 복구할 수 없으므로,
이 모듈의 재현율이 파이프라인 전체 재현율의 상한을 결정한다.
(기획서 3장 "이 구조가 막지 못하는 오류" 참고)
"""

from dataclasses import dataclass

from paddleocr import PaddleOCR

from pipeline.config import OCR_LANG, OCR_USE_ANGLE_CLS


@dataclass
class OcrToken:
    text: str
    confidence: float
    # 4점 폴리곤 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] — PaddleOCR 원본 좌표계
    bbox: list[tuple[float, float]]


class OcrEngine:
    """PaddleOCR 래퍼. 모델 로딩 비용이 크므로 인스턴스를 재사용한다."""

    def __init__(self, lang: str = OCR_LANG, use_angle_cls: bool = OCR_USE_ANGLE_CLS):
        self._ocr = PaddleOCR(lang=lang, use_angle_cls=use_angle_cls, show_log=False)

    def extract(self, image_path: str) -> list[OcrToken]:
        """이미지에서 텍스트 라인을 추출한다.

        반환 순서는 PaddleOCR이 반환하는 순서(대체로 위→아래, 왼쪽→오른쪽)를 그대로 따른다.
        성분표는 쉼표로 구분된 한 문단인 경우가 많아, 라인 단위 결과를 이후
        (Stage 2 이전) 쉼표 기준으로 토큰화하는 전처리가 필요할 수 있다 — 실제 라벨
        샘플로 검증 필요.
        """
        raw = self._ocr.ocr(image_path, cls=True)
        tokens: list[OcrToken] = []
        if not raw or raw[0] is None:
            return tokens
        for line in raw[0]:
            bbox, (text, confidence) = line
            tokens.append(OcrToken(text=text, confidence=float(confidence), bbox=bbox))
        return tokens


def split_ingredient_tokens(tokens: list[OcrToken]) -> list[str]:
    """OCR 라인 텍스트를 쉼표/가운데점 등 구분자로 쪼개 성분 후보 문자열 목록을 만든다.

    실제 라벨 포맷(구분자, 줄바꿈 처리)에 맞춰 조정 필요 — records/01-사실정리.md 참고.
    """
    separators = [",", "、", "·"]
    result: list[str] = []
    for token in tokens:
        text = token.text
        for sep in separators:
            text = text.replace(sep, ",")
        for part in text.split(","):
            part = part.strip()
            if part:
                result.append(part)
    return result
