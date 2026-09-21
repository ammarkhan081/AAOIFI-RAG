"""Evaluate hard set + probes to test H2/H3 (selective prediction & escalation).

Combines:
- n=7 hard set (stored run replay, like replay_n7_router.py)
- n=25 probes (unanswerable questions: 19 abstain, 6 escalate)

Total n=32. Tests RQ2/H2 (abstention precision/recall, unsafe answer rate) and
RQ3/H3 (escalation precision/recall).

GPU-free when using stored hard-set results. Probes require either:
1. Live pipeline run (GPU-required for Jais-2)
2. Mock decisions for diagnostic testing
3. Stored probe results (if available)

Run from repo root:
    python scripts/evaluate_with_probes.py --mode replay  # Hard set only
    python scripts/evaluate_with_probes.py --mode mock    # Mock probe decisions
    python scripts/evaluate_with_probes.py --mode live    # Full GPU run (Colab)
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aaoifi_rag.reporting.metrics import compute_selective_risk  # noqa: E402

# File paths
STORED_RUN = REPO_ROOT / "reports" / "e2e_batch_smoke_results.json"
CLAUSE_CHUNKS = REPO_ROOT / "data" / "private" / "extracted" / "clause_chunks.jsonl"
HARD_SET = REPO_ROOT / "data" / "private" / "hard_set.jsonl"
PROBES = REPO_ROOT / "data" / "probes" / "unanswerable_probes.jsonl"
GATES_CONFIG = REPO_ROOT / "configs" / "reliability" / "gates_v1.json"

DEFAULT_METRICS_OUT = REPO_ROOT / "reports" / "combined_evaluation_n32.json"


def require(path: Path, name: str) -> Path:
    """Require a file exists, exit with error message if not."""
    if not path.exists():
        print(
            f"ERROR: Missing required input: {path.relative_to(REPO_ROOT).as_posix()}\n"
            f"This script requires {name}.",
            file=sys.stderr,
        )
        sys.exit(2)
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def replay_hard_set_decisions() -> list[dict[str, Any]]:
    """Replay stored n=7 run through routing policy to get decisions.

    Uses the same approach as scripts/replay_n7_router.py but returns just
    the decision outcomes, not full traces.
    """
    # Import here to avoid circular dependencies
    from aaoifi_rag.generation.types import GeneratedAnswer, GenerationConfig
    from aaoifi_rag.orchestration import AnswerPipeline, RetrievalResult
    from aaoifi_rag.reliability.policy import (
        PolicyConfig,
        SelectivePredictionPolicy,
    )

    # Load inputs
    stored = json.loads(require(STORED_RUN, "stored hard-set run").read_text(encoding="utf-8"))
    chunks = {
        record["chunk_id"]: record
        for record in read_jsonl(require(CLAUSE_CHUNKS, "clause chunks"))
    }
    hard_set = read_jsonl(require(HARD_SET, "hard set"))
    hard_by_id = {item["item_id"]: item for item in hard_set}

    # Rejoin contexts
    contexts = {}
    for item in stored["items"]:
        item_id = item["item_id"]
        top5 = sorted(item["top5"], key=lambda x: x["rank"])
        records = []
        for entry in top5:
            chunk = chunks[entry["chunk_id"]]
            records.append({**chunk, "reranker_score": entry.get("reranker_score")})
        contexts[item_id] = records

    # Replay retriever and generator
    class ReplayRetriever:
        def __init__(self, contexts):
            self._contexts = contexts
            self._item_id = None
        def for_item(self, item_id):
            self._item_id = item_id
            return self
        def retrieve(self, query, k):
            records = self._contexts[self._item_id]
            return RetrievalResult(
                records=tuple(records[:k]),
                reranker_signal=None,
                retrieval_policy={"source": "replay"},
            )

    class ReplayGenerator:
        def __init__(self, responses, config):
            self._responses = responses
            self._config = config
            self._item_id = None
        def for_item(self, item_id):
            self._item_id = item_id
            return self
        def generate(self, prompt):
            return GeneratedAnswer(
                text=self._responses[self._item_id],
                config=self._config,
            )

    # Set up pipeline
    responses = {item["item_id"]: item.get("model_response", "") for item in stored["items"]}
    generation_config = GenerationConfig(max_new_tokens=512)

    policy = SelectivePredictionPolicy(
        PolicyConfig.from_json_file(GATES_CONFIG) if GATES_CONFIG.exists() else PolicyConfig()
    )

    retriever = ReplayRetriever(contexts)
    generator = ReplayGenerator(responses, generation_config)
    pipeline = AnswerPipeline(retriever, generator, policy, generation_config=generation_config)

    # Run pipeline on each item
    results = []
    for item_id in sorted(hard_by_id.keys()):
        hard_item = hard_by_id[item_id]
        retriever.for_item(item_id)
        generator.for_item(item_id)

        result = pipeline.run(
            item_id=item_id,
            query_text=hard_item["question_text"],
            gold_answer=hard_item.get("gold_answer"),
        )

        results.append({
            "item_id": item_id,
            "expected_behavior": hard_item["expected_behavior"],
            "decision": result.decision.value,
            "triggering_gate": result.routing.triggering_gate,
            "source": "hard_set_replay",
        })

    return results


def generate_mock_probe_decisions(probes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate mock decisions for probes for testing purposes.

    Simulates realistic pipeline behavior:
    - Abstain probes: 80% correctly abstain, 15% incorrectly answer, 5% escalate
    - Escalate probes: 67% correctly escalate, 33% incorrectly abstain
    """
    import random
    random.seed(42)  # Reproducible

    results = []
    for probe in probes:
        expected = probe["expected_behavior"]

        if expected == "abstain":
            # Simulate: most correctly abstain, some slip through as answers
            rand = random.random()
            if rand < 0.80:
                decision = "abstain"
            elif rand < 0.95:
                decision = "answer"  # Unsafe!
            else:
                decision = "escalate"
        else:  # expected == "escalate"
            # Simulate: most correctly escalate, some incorrectly abstain
            decision = "escalate" if random.random() < 0.67 else "abstain"

        results.append({
            "item_id": probe["probe_id"],
            "expected_behavior": expected,
            "decision": decision,
            "triggering_gate": None,  # Mock doesn't track gates
            "source": "mock_simulation",
        })

    return results


def print_summary(items: list[dict[str, Any]]) -> None:
    """Print per-item decisions and aggregate metrics."""
    print(f"\n{'Item ID':<15} {'Expected':<10} {'Decision':<10} {'Gate':<24} {'Source':<20}")
    print("=" * 85)

    for item in items:
        gate = item.get("triggering_gate") or "-"
        source = item.get("source", "unknown")
        print(
            f"{item['item_id']:<15} "
            f"{item['expected_behavior']:<10} "
            f"{item['decision']:<10} "
            f"{gate:<24} "
            f"{source:<20}"
        )

    print("=" * 85)
    print()

    # Compute metrics
    metrics = compute_selective_risk(items)

    print("=== SELECTIVE RISK METRICS (n={}) ===".format(metrics.n_items))
    print()

    print("--- Routing Breakdown ---")
    decisions = Counter(item["decision"] for item in items)
    for decision, count in sorted(decisions.items()):
        pct = 100.0 * count / metrics.n_items if metrics.n_items else 0
        print(f"  {decision:<10}: {count:3d} ({pct:5.1f}%)")
    print()

    print("--- ASYMMETRIC RISK (The Core Question) ---")
    print(f"  Unsafe answers: {metrics.n_unsafe_answers}/{metrics.n_abstain_expected}")
    if metrics.unsafe_answer_rate is not None:
        print(f"  Unsafe answer rate: {metrics.unsafe_answer_rate:.1%}")
        print(f"  (corpus-absent questions that were incorrectly served an answer)")
    print()

    print("--- ABSTENTION (Corpus-Absent Questions) ---")
    print(f"  Expected to abstain: {metrics.n_abstain_expected}")
    print(f"  True abstain (correct): {metrics.n_true_abstain}")
    print(f"  False abstain (over-abstention on answerable): {metrics.n_false_abstain}")
    if metrics.abstention_precision is not None:
        print(f"  Precision: {metrics.abstention_precision:.1%}")
    if metrics.abstention_recall is not None:
        print(f"  Recall: {metrics.abstention_recall:.1%}")
    print()

    print("--- ESCALATION (Cross-Standard Comparisons, STIPULATED) ---")
    print(f"  Expected to escalate: {metrics.n_escalation_expected}")
    print(f"  True escalate: {metrics.n_true_escalate}")
    if metrics.escalation_precision is not None:
        print(f"  Precision: {metrics.escalation_precision:.1%}")
    if metrics.escalation_recall is not None:
        print(f"  Recall: {metrics.escalation_recall:.1%}")
    print()
    print("  WARNING: Escalation labels are stipulated_definition, not scholar review.")
    print("  Reject the definition and these numbers disappear; abstention (mechanical")
    print("  corpus absence) is unaffected.")
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["replay", "mock", "live"],
        default="replay",
        help=(
            "replay: Hard set from stored run, probes skipped (n=7). "
            "mock: Hard set replay + simulated probe decisions (n=32). "
            "live: Full GPU pipeline run (requires Colab, not implemented yet)."
        ),
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=DEFAULT_METRICS_OUT,
        help="Where to write JSON metrics output.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print summary only, write nothing.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print(f"Mode: {args.mode}")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()

    # Load probes
    probes = read_jsonl(require(PROBES, "unanswerable probes"))
    print(f"Loaded {len(probes)} probes")
    probe_counts = Counter(p["expected_behavior"] for p in probes)
    for behavior, count in sorted(probe_counts.items()):
        print(f"  {behavior}: {count}")
    print()

    # Get decisions based on mode
    if args.mode == "replay":
        print("Replaying hard set (n=7) from stored run...")
        hard_set_results = replay_hard_set_decisions()
        print(f"Replayed {len(hard_set_results)} hard-set items")
        print()
        print("NOTE: Probes not evaluated in replay mode (GPU required).")
        print("Use --mode mock for diagnostic testing or --mode live for full evaluation.")
        items = hard_set_results

    elif args.mode == "mock":
        print("Replaying hard set + generating mock probe decisions (n=32)...")
        hard_set_results = replay_hard_set_decisions()
        probe_results = generate_mock_probe_decisions(probes)
        items = hard_set_results + probe_results
        print(f"Combined: {len(hard_set_results)} hard set + {len(probe_results)} probes = {len(items)} total")
        print()
        print("NOTE: Probe decisions are SIMULATED. Use --mode live for real evaluation.")

    elif args.mode == "live":
        print("ERROR: Live mode not yet implemented.")
        print("This requires GPU access for Jais-2 generation.")
        print("Run in Google Colab using scripts/colab_e2e_batch_test.py as template.")
        return 1

    else:
        print(f"Unknown mode: {args.mode}")
        return 1

    # Print summary
    print()
    print_summary(items)

    # Write metrics
    if not args.no_write:
        metrics = compute_selective_risk(items)
        output = {
            "run_id": f"eval_with_probes_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
            "mode": args.mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_items": len(items),
            "n_hard_set": len([i for i in items if i.get("source") == "hard_set_replay"]),
            "n_probes": len([i for i in items if i.get("source") in ("mock_simulation", "live_pipeline")]),
            "metrics": metrics.as_dict(),
            "verification_note": (
                "Hard-set items: corpus_cross_reference only, not qualified_scholar_review. "
                "Probe labels: mechanical_corpus_absence (abstain) and stipulated_definition (escalate)."
            ),
        }

        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(
            json.dumps(output, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote metrics -> {args.metrics_out}")
    else:
        print("\n--no-write: nothing written")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
