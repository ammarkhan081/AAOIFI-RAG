"""Full evaluation: hard set (n=7) + probes (n=25).

Tests RQ2/H2 (selective prediction, coverage vs residual error) and RQ3/H3
(disagreement/escalation behavior).

Requires:
- Hard set with gold annotations (data/private/hard_set.jsonl)
- Unanswerable probes (data/probes/unanswerable_probes.jsonl)
- Retrieval backend (BM25 + reranker or SAC)
- Generator backend (Jais-2, GPU-required)

Run from project root after retrieval indexes and generation backend are ready.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Add src to path for direct script execution
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aaoifi_rag.orchestration.pipeline import AnswerPipeline
from aaoifi_rag.reporting.evaluation import build_evaluation_label
from aaoifi_rag.reporting.metrics import (
    compute_combined_metrics,
    compute_hard_set_metrics,
    compute_probe_metrics,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Save JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    """Run full evaluation."""

    # Load evaluation sets
    hard_set_path = REPO_ROOT / "data" / "private" / "hard_set.jsonl"
    probes_path = REPO_ROOT / "data" / "probes" / "unanswerable_probes.jsonl"

    if not hard_set_path.exists():
        print(f"ERROR: Hard set not found: {hard_set_path}")
        print("Hard set is Git-ignored. Ensure it exists before running.")
        sys.exit(1)

    if not probes_path.exists():
        print(f"ERROR: Probes not found: {probes_path}")
        sys.exit(1)

    hard_set = load_jsonl(hard_set_path)
    probes = load_jsonl(probes_path)

    print(f"Loaded {len(hard_set)} hard-set items")
    print(f"Loaded {len(probes)} probes")
    print()

    # 🛑 MANUAL ACTION REQUIRED: Inject retriever and generator
    print("=" * 70)
    print("🛑 MANUAL ACTION REQUIRED")
    print("=" * 70)
    print()
    print("This script requires:")
    print("  1. A Retriever implementation (GPU-required for reranker)")
    print("  2. A Generator implementation (GPU-required for Jais-2)")
    print()
    print("These backends are not available in the local sandbox.")
    print()
    print("OPTIONS:")
    print("  A. Run this script in Google Colab with GPU")
    print("  B. Use the replay mode (run on stored n=7 results only)")
    print("  C. Inject scripted backends for testing")
    print()
    print("To proceed:")
    print("  1. Set up retriever and generator backends")
    print("  2. Uncomment the pipeline construction below")
    print("  3. Run: python scripts/run_full_evaluation.py")
    print()
    print("=" * 70)
    sys.exit(0)

    # Uncomment when backends are ready:
    # from your_retrieval_module import create_retriever
    # from your_generation_module import create_generator
    #
    # retriever = create_retriever()
    # generator = create_generator()
    # pipeline = AnswerPipeline(retriever, generator)
    #
    # # Run hard set
    # hard_results = pipeline.run_batch(hard_set)
    #
    # # Run probes (adapt field names)
    # probe_items = [
    #     {
    #         "item_id": p["probe_id"],
    #         "question_text": p["question_text"],
    #         "expected_behavior": p["expected_behavior"],
    #     }
    #     for p in probes
    # ]
    # probe_results = pipeline.run_batch(probe_items, id_field="item_id")
    #
    # # Build evaluation labels
    # hard_labels = [
    #     build_evaluation_label(
    #         item, result.decision.value, result.retrieval.records
    #     )
    #     for item, result in zip(hard_set, hard_results)
    # ]
    #
    # probe_labels = [
    #     build_evaluation_label(
    #         {"item_id": p["probe_id"], "expected_behavior": p["expected_behavior"]},
    #         result.decision.value,
    #         None,  # No gold clauses for probes
    #     )
    #     for p, result in zip(probes, probe_results)
    # ]
    #
    # # Compute metrics
    # hard_metrics = compute_hard_set_metrics(hard_labels)
    # probe_metrics = compute_probe_metrics(probe_labels, probes)
    # combined_metrics = compute_combined_metrics(hard_labels, probe_labels)
    #
    # # Save results
    # output_dir = REPO_ROOT / "runs" / "full_evaluation"
    # output_dir.mkdir(parents=True, exist_ok=True)
    #
    # with (output_dir / "metrics.json").open("w") as f:
    #     json.dump({
    #         "hard_set": hard_metrics,
    #         "probes": probe_metrics,
    #         "combined": combined_metrics,
    #     }, f, indent=2)
    #
    # print("Evaluation complete!")
    # print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
