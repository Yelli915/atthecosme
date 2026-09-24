"""파이프라인 전역 설정.

경로·모델명·임계치를 한 곳에 모아둔다. 실제 값이 바뀌면(엔진 교체, DB 위치 이동 등)
여기만 고치면 되도록 각 Stage 모듈은 이 파일의 상수만 참조한다.
"""

from pathlib import Path

# 경로 ----------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
INGREDIENT_DB_PATH = DATA_DIR / "ingredient_db" / "ingredients.db"
INGREDIENT_SEED_CSV = DATA_DIR / "ingredient_db" / "seed_ingredients.csv"
BENCHMARK_DIR = DATA_DIR / "benchmark"
RECORDS_DIR = ROOT_DIR / "records"

# Stage 1. OCR ----------------------------------------------------------
OCR_LANG = "korean"  # PaddleOCR 언어 모델. 라벨에 한글/영문이 섞여 있어 korean 모델이 영문도 함께 인식.
OCR_USE_ANGLE_CLS = True  # 곡면 용기 등 기울어진 텍스트 보정

# Stage 2. 하이브리드 검색 -------------------------------------------
DENSE_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 한글 성분명 + 영문 INCI명 모두 지원
TOP_K_CANDIDATES = 5
BM25_WEIGHT = 0.5
DENSE_WEIGHT = 0.5

# Stage 3. 구조화 출력 ---------------------------------------------
OLLAMA_MODEL = "qwen2.5:7b"  # 로컬 실행, 한국어 성능과 JSON/구조화 출력 준수도를 고려해 선택
NOT_FOUND_LABEL = "NOT_FOUND"

# Stage 5. 신뢰도 분기 -------------------------------------------------
# TODO(records/04-신뢰도임계치.md): 벤치마크로 실측 후 확정. 우선 기획서 3안 중 중간값으로 시작.
CONFIDENCE_THRESHOLD = 0.90
