"""Quantify the retrieval-stage blind spot. GPU-free, fully local.

The finding
-----------
Both shipped retrieval-stage gates are structurally incapable of firing on a question
the corpus does not cover:

* ``retrieval_non_empty`` fails only when ``retrieved_count == 0``. BM25 and a dense
  index return *k* records for **any** query, so it never fails in practice.
* ``retrieval_normative`` fails only when *every* retrieved record looks like a
  section heading. A corpus-absent question still retrieves five ordinary normative
  clauses - about something else.

So the pre-generation check cannot abstain on the 25 probes in
``data/probes/unanswerable_probes.jsonl``, whose labels are facts about the corpus.
This script measures that rather than asserting it, over both classes at once, and
prints the counterfactual for the rejected lexical-anchoring gate at the one
threshold that has a zero false-positive rate on the answerable items.

Why the measurement matters more than the gates
-----------------------------------------------
It localises the architecture's abstention capability. If the retrieval layer cannot
contribute to abstention on out-of-corpus questions, then **every** such abstention
must come from the generator's own behaviour, and the reliability layer's job on
those items is auditing a model decision rather than making its own. That is a claim
about where the risk sits, and it is the reason the response-stage gates and the
model-mediated abstention path carry the weight in this design.

``decision == "answer"`` here means only "no retrieval-stage gate objected; proceed to
generation". It does **not** mean an answer was served. Nothing in this script calls a
model; the post-generation decision needs a GPU and is out of scope.

Run: ``python scripts/audit_retrieval_stage_blind_spot.py``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aaoifi_rag.reliability import (  # noqa: E402
    AnchorSpec,
    CorpusVocabulary,
    PolicyConfig,
    SelectivePredictionPolicy,
    compute_anchoring_signal,
    compute_signals,
    describe_gates,
)
from aaoifi_rag.reporting import (  # noqa: E402
    compute_selective_risk,
    load_probes,
)
from aaoifi_rag.retrieval.bm25_baseline import search_bm25  # noqa: E402

CLAUSE_CHUNKS = REPO_ROOT / "data" / "private" / "extracted" / "clause_chunks.jsonl"
BM25_INDEX = REPO_ROOT / "data" / "private" / "extracted" / "bm25_clause_level.pkl"
HARD_SET = REPO_ROOT / "data" / "private" / "hard_set.jsonl"
PROBES = REPO_ROOT / "data" / "probes" / "unanswerable_probes.jsonl"
OUT_JSON = REPO_ROOT / "reports" / "retrieval_stage_blind_spot.json"

TOP_K = 5

#: The counterfactual threshold. ``reports/anchoring_gate_calibration.md`` shows 0.25
#: is the largest IDF-weighted coverage threshold with 0/7 false positives on the
#: answerable items. Running the audit with the gate forced on at 0.25 answers "would
#: the rejected gate have closed the blind spot?" with a number instead of a guess.
COUNTERFACTUAL_THRESHOLD = 0.25


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(
            f"missing required input: {path.relative_to(REPO_ROOT).as_posix()}\n"
            "This audit reads the licensed corpus and the persisted BM25 index; both "
            "are Git-ignored. See data/README.md.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_items() -> list[dict[str, str]]:
    """The two classes in one list, each carrying its own provenance labels."""
    items: list[dict[str, str]] = []
    for row in read_jsonl(HARD_SET):
        items.append(
            {
                "id": str(row["item_id"]),
                "question_text": str(row["question_text"]),
                "expected_behavior": str(row["expected_behavior"]),
                "klass": "answerable_hard_set",
                "basis": str(row.get("answerability", "")),
                "verification_basis": str(row.get("verification_basis", "")),
            }
        )
    for probe in load_probes(PROBES):
        items.append(
            {
                "id": probe.probe_id,
                "question_text": probe.question_text,
                "expected_behavior": probe.expected_behavior,
                "klass": "unanswerable_probe",
                "basis": probe.basis.value,
                "verification_basis": probe.verification_basis,
            }
        )
    return items


def audit(
    items: Sequence[Mapping[str, str]],
    chunks: Sequence[Mapping[str, Any]],
    vocabulary: CorpusVocabulary,
    policy: SelectivePredictionPolicy,
) -> list[dict[str, Any]]:
    """One pre-generation routing decision per item, with its evidence."""
    rows: list[dict[str, Any]] = []
    for item in items:
        context = search_bm25(list(chunks), BM25_INDEX, item["question_text"], top_k=TOP_K)
        anchoring = compute_anchoring_signal(
            item["question_text"], context, vocabulary
        )
        signals = compute_signals(context, anchoring=anchoring)
        outcome = policy.decide_pre_generation(signals)
        rows.append(
            {
                "id": item["id"],
                "klass": item["klass"],
                "basis": item["basis"],
                "expected_behavior": item["expected_behavior"],
                "decision": outcome.decision.value,
                "triggering_gate": outcome.triggering_gate,
                "retrieved_count": signals.retrieval.retrieved_count,
                "all_heading_like": signals.retrieval.all_heading_like,
                "heading_like_top_1": signals.retrieval.heading_like_top_1,
                "idf_weighted_coverage": anchoring.idf_weighted_coverage,
                "n_anchors": anchoring.n_anchors,
            }
        )
    return rows


def summarise(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    """Counts per class, plus the selective-risk cross-tabulation."""
    probes = [r for r in rows if r["klass"] == "unanswerable_probe"]
    answerable = [r for r in rows if r["klass"] == "answerable_hard_set"]
    risk = compute_selective_risk(rows)
    return {
        "label": label,
        "n_items": len(rows),
        "probes": {
            "n": len(probes),
            "n_routed_answer": sum(1 for r in probes if r["decision"] == "answer"),
            "n_routed_abstain": sum(1 for r in probes if r["decision"] == "abstain"),
            "n_routed_escalate": sum(1 for r in probes if r["decision"] == "escalate"),
        },
        "answerable": {
            "n": len(answerable),
            "n_routed_answer": sum(1 for r in answerable if r["decision"] == "answer"),
            "n_routed_abstain": sum(
                1 for r in answerable if r["decision"] == "abstain"
            ),
        },
        "selective_risk": risk.as_dict(),
        "stage_note": (
            "pre-generation only; decision 'answer' means no retrieval-stage gate "
            "objected, NOT that an answer was served"
        ),
    }


def print_summary(summary: Mapping[str, Any]) -> None:
    print(f"\n=== {summary['label']} ===")
    probes = summary["probes"]
    answerable = summary["answerable"]
    print(
        f"probes n={probes['n']}: proceed-to-generation {probes['n_routed_answer']}, "
        f"abstained {probes['n_routed_abstain']}, escalated {probes['n_routed_escalate']}"
    )
    print(
        f"answerable n={answerable['n']}: proceed-to-generation "
        f"{answerable['n_routed_answer']}, abstained {answerable['n_routed_abstain']} "
        "(any abstention here is a false positive)"
    )
    risk = summary["selective_risk"]
    unsafe = risk["asymmetric_risk"]
    print(
        f"unsafe-proceed cell: {unsafe['n_unsafe_answers']}/"
        f"{risk['abstention']['n_abstain_expected']} unanswerable items reached "
        "generation unchallenged"
    )
    interval = unsafe["interval"]
    if interval:
        print(
            f"  exact 95% CI {interval['lower']:.3f}-{interval['upper']:.3f} "
            f"(width {interval['width']:.3f}, informative={interval['is_informative']})"
        )
    abstention = risk["abstention"]
    print(
        f"abstention recall {abstention['recall']}  precision {abstention['precision']}"
    )
    print(
        "escalation (STIPULATED, never pooled with the above): recall "
        f"{risk['escalation_stipulated']['recall']}  precision "
        f"{risk['escalation_stipulated']['precision']}"
    )


def print_gate_table(rows: Sequence[Mapping[str, Any]]) -> None:
    print("\nWhy no retrieval-stage gate fires on the probes:")
    print(f"{'':<4}{'retrieved==0':>14}{'all headings':>14}{'heading top-1':>15}")
    probes = [r for r in rows if r["klass"] == "unanswerable_probe"]
    print(
        f"{'':<4}{sum(1 for r in probes if r['retrieved_count'] == 0):>14}"
        f"{sum(1 for r in probes if r['all_heading_like']):>14}"
        f"{sum(1 for r in probes if r['heading_like_top_1']):>15}"
        f"   of {len(probes)} probes"
    )
    print(
        "  retrieval_non_empty and retrieval_normative are the only enabled "
        "retrieval-stage gates; neither has a condition a corpus-absent question "
        "satisfies."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    parser.add_argument(
        "--no-write", action="store_true", help="Print only; write nothing."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not BM25_INDEX.exists():
        print(f"missing BM25 index: {BM25_INDEX}", file=sys.stderr)
        raise SystemExit(2)
    chunks = read_jsonl(CLAUSE_CHUNKS)
    spec = AnchorSpec()
    vocabulary = CorpusVocabulary.from_chunks(chunks, spec)
    items = load_items()
    print(
        f"corpus {len(chunks)} chunks | items {len(items)} "
        f"({sum(1 for i in items if i['klass'] == 'answerable_hard_set')} answerable + "
        f"{sum(1 for i in items if i['klass'] == 'unanswerable_probe')} probes)"
    )

    shipped = SelectivePredictionPolicy()
    enabled = [g["name"] for g in describe_gates(shipped.config) if g["enabled"]]
    print(f"shipped policy: {len(enabled)} gates enabled")

    rows_shipped = audit(items, chunks, vocabulary, shipped)
    summary_shipped = summarise(rows_shipped, "shipped policy (anchoring gate OFF)")
    print_summary(summary_shipped)
    print_gate_table(rows_shipped)

    counterfactual = SelectivePredictionPolicy(
        PolicyConfig(
            enable_anchoring_gate=True,
            min_anchor_coverage=COUNTERFACTUAL_THRESHOLD,
        )
    )
    rows_cf = audit(items, chunks, vocabulary, counterfactual)
    summary_cf = summarise(
        rows_cf,
        f"counterfactual: anchoring gate ON at {COUNTERFACTUAL_THRESHOLD} "
        "(REJECTED config, not shipped)",
    )
    print_summary(summary_cf)

    print(
        "\nInterpretation: the shipped retrieval-stage check cannot abstain on a "
        "corpus-absent question, and the one candidate gate that could have does not "
        "separate the classes (AUC 0.513, reports/anchoring_gate_calibration.md). "
        "Abstention on these items therefore has to be model-mediated and then "
        "audited at the response stage - which is where this architecture puts it."
    )

    if args.no_write:
        print("\n--no-write: nothing written.")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "source": "scripts/audit_retrieval_stage_blind_spot.py",
                "top_k": TOP_K,
                "retriever": "bm25_clause_level (probes cannot be retrieved locally "
                "with the dense index; no GPU in this environment)",
                "anchor_spec": spec.as_dict(),
                "counterfactual_threshold": COUNTERFACTUAL_THRESHOLD,
                "shipped": {"summary": summary_shipped, "items": rows_shipped},
                "counterfactual": {"summary": summary_cf, "items": rows_cf},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {args.out.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
