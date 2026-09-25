"""벤치마크 실행 스크립트.

사용법:
    python -m eval.run_benchmark --manifest data/benchmark/manifest.json

manifest.json 포맷은 data/benchmark/manifest.example.json 참고.
실제 100건/150건 벤치마크셋(이미지 + 정답 라벨)은 아직 이 저장소에 없음
— records/01-사실정리.md 에서 소재/라벨링 방식을 확인한 뒤 준비.
"""

import argparse
import json
import sys
from pathlib import Path

from eval.metrics import aggregate_recall, exact_match
from pipeline.retrieval.hybrid import HybridRetriever
from pipeline.run import run_pipeline


def load_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run(manifest_path: Path) -> None:
    if not manifest_path.exists():
        print(
            f"[오류] {manifest_path} 없음. data/benchmark/manifest.example.json 형식으로 "
            "실제 벤치마크셋 manifest를 준비하세요 (records/01-사실정리.md 참고).",
            file=sys.stderr,
        )
        sys.exit(1)

    samples = load_manifest(manifest_path)
    retriever = HybridRetriever()  # 샘플마다 재구축하지 않도록 공유

    per_sample_predicted: list[set[int]] = []
    per_sample_gold: list[set[int]] = []
    em_hits = 0

    for sample in samples:
        result = run_pipeline(
            image_path=sample["image_path"],
            submitted_names=sample["submitted_ingredients"],
            retriever=retriever,
        )
        predicted_ids = {
            n.result.chosen_ingredient_id for n in result.normalizations if n.result.chosen_ingredient_id is not None
        }
        gold_ids = set(sample["gold_ingredient_ids"])
        per_sample_predicted.append(predicted_ids)
        per_sample_gold.append(gold_ids)

        if "product_name_gold" in sample and "product_name_predicted" in sample:
            if exact_match(sample["product_name_predicted"], sample["product_name_gold"]):
                em_hits += 1

        print(
            f"{sample['sample_id']}: 재현율={len(predicted_ids & gold_ids)}/{len(gold_ids)}"
            f", 자동승인={result.auto_approved}, 불일치={result.match_result.has_discrepancy}"
        )

    recall = aggregate_recall(per_sample_predicted, per_sample_gold)
    print(f"\n전체 성분 재현율(마이크로 평균): {recall:.1%} (N={len(samples)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/benchmark/manifest.json"))
    args = parser.parse_args()
    run(args.manifest)
