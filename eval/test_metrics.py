"""eval/metrics.py 단위 테스트.

metrics.py는 순수 함수로 작성돼 실제 벤치마크(이미지) 없이도 검증 가능하다는 게
설계 의도(모듈 docstring 참고) — 그 의도대로 되는지 확인한다.
"""

import unittest

from eval.metrics import aggregate_recall, exact_match, ingredient_precision, ingredient_recall


class TestIngredientRecall(unittest.TestCase):
    def test_부분_재현(self):
        self.assertAlmostEqual(ingredient_recall({1, 2}, {1, 2, 3}), 2 / 3)

    def test_완전_재현(self):
        self.assertEqual(ingredient_recall({1, 2, 3}, {1, 2, 3}), 1.0)

    def test_전혀_못_찾음(self):
        self.assertEqual(ingredient_recall(set(), {1, 2, 3}), 0.0)

    def test_정답이_없으면_1(self):
        self.assertEqual(ingredient_recall(set(), set()), 1.0)
        self.assertEqual(ingredient_recall({1, 2}, set()), 1.0)


class TestIngredientPrecision(unittest.TestCase):
    def test_부분_정밀도(self):
        self.assertAlmostEqual(ingredient_precision({1, 2, 3}, {1, 2}), 2 / 3)

    def test_추출값_없고_정답도_없으면_1(self):
        self.assertEqual(ingredient_precision(set(), set()), 1.0)

    def test_추출값_없는데_정답은_있으면_0(self):
        self.assertEqual(ingredient_precision(set(), {1, 2}), 0.0)

    def test_전혀_겹치지_않으면_0(self):
        self.assertEqual(ingredient_precision({9, 10}, {1, 2}), 0.0)


class TestExactMatch(unittest.TestCase):
    def test_공백_대소문자_무시하고_일치(self):
        self.assertTrue(exact_match("  Niacinamide ", "niacinamide"))

    def test_불일치(self):
        self.assertFalse(exact_match("Niacinamide", "Glycerin"))


class TestAggregateRecall(unittest.TestCase):
    def test_마이크로_평균(self):
        # 샘플1: 정답 2개 중 1개 히트, 샘플2: 정답 3개 중 3개 히트 -> 전체 4/5
        predicted = [{1}, {4, 5, 6}]
        gold = [{1, 2}, {4, 5, 6}]
        self.assertAlmostEqual(aggregate_recall(predicted, gold), 4 / 5)

    def test_정답_전체가_0건이면_1(self):
        self.assertEqual(aggregate_recall([set(), set()], [set(), set()]), 1.0)

    def test_샘플_수와_무관하게_분모는_전체_정답_개수_합(self):
        # 계획.md 1단계 "벤치마크셋 전체 정답 성분 개수(재현율 분모)"와 연결되는 부분 —
        # 샘플 개수가 아니라 정답 성분 총 개수가 분모여야 한다.
        predicted = [{1}, set(), {2}]
        gold = [{1}, {2, 3}, {2}]  # 분모 = 1 + 2 + 1 = 4, 히트 = 1 + 0 + 1 = 2
        self.assertAlmostEqual(aggregate_recall(predicted, gold), 2 / 4)


if __name__ == "__main__":
    unittest.main()
