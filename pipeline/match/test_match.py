"""Stage 4 대조 로직 회귀 테스트 — records/03-대조구현.md에서 스크래치 스크립트로
검증했던 시나리오를 정식 테스트로 고정한다. 샘플 성분 DB(15건, seed_ingredients.csv)만
있으면 되고 라벨 이미지는 필요 없다.

사전 조건: `python -m pipeline.retrieval.db`로 data/ingredient_db/ingredients.db가
생성돼 있어야 한다 (README "실행 방법" 참고).
"""

import unittest

from pipeline.match.match import match, resolve_submitted_names
from pipeline.retrieval.hybrid import HybridRetriever

# 라벨에서 추출됐다고 가정: Water, Glycerin, Niacinamide, Sodium Hyaluronate, Panthenol
LABEL_IDS = {1, 2, 5, 7, 13}


class TestMatchScenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = HybridRetriever()  # 임베딩 모델 로딩 비용이 커서 클래스당 1회만

    def _resolve_and_match(self, submitted_names: list[str]):
        submitted_map = resolve_submitted_names(submitted_names, self.retriever)
        return match(LABEL_IDS, submitted_map)

    def test_정상_변형없음_불일치_없음(self):
        result = self._resolve_and_match(
            ["Water", "Glycerin", "Niacinamide", "Sodium Hyaluronate", "Panthenol"]
        )
        self.assertFalse(result.has_discrepancy)
        self.assertEqual({i.id for i in result.matched}, LABEL_IDS)

    def test_누락_제출에만_있는_성분_검출(self):
        result = self._resolve_and_match(
            ["Water", "Glycerin", "Niacinamide", "Sodium Hyaluronate", "Panthenol", "Tocopherol"]
        )
        self.assertTrue(result.has_discrepancy)
        self.assertEqual({i.id for i in result.missing}, {12})  # Tocopherol

    def test_추가_라벨에만_있는_성분_검출(self):
        result = self._resolve_and_match(["Water", "Glycerin", "Niacinamide", "Sodium Hyaluronate"])
        self.assertTrue(result.has_discrepancy)
        self.assertEqual({i.id for i in result.extra}, {13})  # Panthenol

    def test_유사_성분_치환_missing과_extra_동시_검출(self):
        result = self._resolve_and_match(
            ["Water", "Glycerin", "Titanium Dioxide", "Sodium Hyaluronate", "Panthenol"]
        )
        self.assertTrue(result.has_discrepancy)
        self.assertEqual({i.id for i in result.missing}, {9})  # Titanium Dioxide
        self.assertEqual({i.id for i in result.extra}, {5})  # Niacinamide

    def test_오탈자는_하이브리드_검색으로_정상_resolve(self):
        result = self._resolve_and_match(
            ["Water", "Glycerin", "Niacinmide", "Sodium Hyaluronate", "Panthenol"]  # Niacinamide 오탈자
        )
        self.assertFalse(result.has_discrepancy)

    def test_완전_미등록_성분명은_임계치가_관대해_오매핑될_수_있음(self):
        """알려진 위험(records/03-대조구현.md): DB에 전혀 없는 성분명이 unresolved로
        빠지지 않고 SUBMITTED_FUZZY_ACCEPT_THRESHOLD(0.7)를 넘겨버려 엉뚱한 실제 성분으로
        매핑되는 현재 동작을 문서화한다. 이 테스트는 '바람직한 동작'이 아니라 '현재 동작'을
        고정해 임계치를 실측 후 고칠 때 이 테스트도 같이 업데이트하라는 표식이다.
        """
        submitted_map = resolve_submitted_names(
            ["Water", "Glycerin", "Niacinamide", "Sodium Hyaluronate", "Panthenol", "Retinal Propionate XYZ123"],
            self.retriever,
        )
        # 기대(이상적)했다면 None(unresolved)이어야 하지만, 현재는 엉뚱한 성분 id로 매핑된다.
        self.assertIsNotNone(submitted_map["Retinal Propionate XYZ123"])


if __name__ == "__main__":
    unittest.main()
