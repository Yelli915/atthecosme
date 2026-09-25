"""pipeline/structured_output/normalize.py의 순수 함수 단위 테스트.

`_value_token_confidence`는 실제로 Ollama 로컬 모델을 호출하지 않고도(=사진·GPU 없이도)
검증할 수 있는 순수 함수라 여기서 고정한다. 아래 로그 확률 값은 실제 로컬 모델
(hf.co/Qwen/Qwen2.5-7B-Instruct-GGUF, 2026-09-25) 호출 결과를 그대로 가져온 것으로,
"JSON 키 이름 토큰의 낮은 확률이 신뢰도를 왜곡하던" 실제 버그를 재현한다 (records/07 참고).
"""

import math
import unittest

from pipeline.structured_output.normalize import _value_token_confidence

# 실제 호출 결과: {"ingredient_id": "cand_5"} 를 생성할 때의 토큰별 logprob.
# 'ingredient'/'_id' 는 스키마가 강제하는 키 이름이라 확률이 낮아도(-10.8 등)
# 모델의 실제 '선택'과는 무관한 토큰이다.
_REAL_CAND_5_TOKENS = [
    {"token": "{\n", "logprob": -0.0475199930369854},
    {"token": " ", "logprob": -0.0331069752573967},
    {"token": ' "', "logprob": -1.1920930376163597e-07},
    {"token": "ingredient", "logprob": -10.79673957824707},
    {"token": "_id", "logprob": -9.135128021240234},
    {"token": '":', "logprob": -7.391003236989491e-06},
    {"token": ' "', "logprob": -0.0008728139800950885},
    {"token": "c", "logprob": -0.12289987504482269},
    {"token": "and", "logprob": -5.3287971240933985e-05},
    {"token": "_", "logprob": -6.198902156029362e-06},
    {"token": "5", "logprob": -0.06311652064323425},
    {"token": '"\n', "logprob": -0.10838999599218369},
    {"token": "}", "logprob": -0.00012994656572118402},
]
_REAL_CAND_5_RAW = "".join(t["token"] for t in _REAL_CAND_5_TOKENS)


class TestValueTokenConfidence(unittest.TestCase):
    def test_키_이름_토큰의_낮은_확률에_왜곡되지_않는다(self):
        confidence = _value_token_confidence(_REAL_CAND_5_RAW, _REAL_CAND_5_TOKENS, "cand_5")

        # 'ingredient' 토큰(logprob -10.8)까지 포함했다면 confidence는 2e-5 수준으로 떨어졌을 것.
        # 값(cand_5)에 해당하는 토큰('c','and','_','5')만의 최솟값('c', -0.1229)을 써야 한다.
        expected = math.exp(-0.12289987504482269)
        self.assertAlmostEqual(confidence, expected, places=6)
        self.assertGreater(confidence, 0.1)  # 왜곡됐다면 이 값보다 한참 작았을 것

    def test_값_문자열이_raw에_없으면_전체_최솟값으로_폴백(self):
        tokens = [{"token": "a", "logprob": -0.1}, {"token": "b", "logprob": -5.0}]
        confidence = _value_token_confidence("ab", tokens, "존재하지않는값")
        self.assertAlmostEqual(confidence, math.exp(-5.0))

    def test_토큰_로그확률이_없으면_1(self):
        self.assertEqual(_value_token_confidence("아무거나", [], "아무거나"), 1.0)

    def test_값_토큰_자체가_불확실하면_confidence도_낮다(self):
        # 값 구간(마지막 토큰)의 확률이 실제로 낮은 경우 -> 왜곡 보정 없이도 낮게 나와야 함
        tokens = [
            {"token": '{"v":"', "logprob": -0.01},
            {"token": "NOT_FOUND", "logprob": -2.0},
            {"token": '"}', "logprob": -0.01},
        ]
        raw = "".join(t["token"] for t in tokens)
        confidence = _value_token_confidence(raw, tokens, "NOT_FOUND")
        self.assertAlmostEqual(confidence, math.exp(-2.0))


if __name__ == "__main__":
    unittest.main()
