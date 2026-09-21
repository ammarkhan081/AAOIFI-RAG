"""Calibrate the lexical-anchoring gate. GPU-free, fully local.

Produces the numbers that decide whether ``enable_anchoring_gate`` ships on or
off, and at what threshold. Nothing here is authored: every rate is computed from
``data/private/extracted/clause_chunks.jsonl``, the persisted BM25 index, the
seven hard-set items and the 25 probes.

Reading the output
------------------
The table splits the true-positive rate by probe basis, because the three bases
are **not** equally good evidence for this gate:

* ``term_absent_from_corpus`` - **circular.** Those probes were selected by having
  a term with zero corpus occurrences, and the gate keys on rare-or-absent query
  terms. A high rate here is a tautology and is printed only so it is not mistaken
  for a finding.
* ``standard_absent_from_corpus`` and ``clause_absent_from_standard`` - **valid.**
  Selected by standard/clause membership, a criterion unrelated to term frequency.
* the seven answerable items - **the number that matters.** Authored
  independently by the researcher against real clauses. Any firing here is a
  false positive, and a gate with a non-trivial false-positive rate on n=7
  answerable items should not ship enabled.

Two context sources are measured for the seven answerable items: the persisted
BM25 top-5 (comparable with the probes, which can only be retrieved locally with
BM25) and the stored hybrid+rerank top-5 the model actually saw in the n=7 Colab
run. If those disagree the gate is sensitive to the retriever and the report says
so.

Run: ``python scripts/calibrate_anchoring_gate.py``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aaoifi_rag.reliability.anchoring import (  # noqa: E402
    AnchorSpec,
    CorpusVocabulary,
    compute_anchoring_signal,
)
from aaoifi_rag.reporting.probes import load_probes  # noqa: E402
from aaoifi_rag.retrieval.bm25_baseline import search_bm25  # noqa: E402

CLAUSE_CHUNKS = REPO_ROOT / "data" / "private" / "extracted" / "clause_chunks.jsonl"
BM25_INDEX = REPO_ROOT / "data" / "private" / "extracted" / "bm25_clause_level.pkl"
HARD_SET = REPO_ROOT / "data" / "private" / "hard_set.jsonl"
PROBES = REPO_ROOT / "data" / "probes" / "unanswerable_probes.jsonl"
STORED_RUN = REPO_ROOT / "reports" / "e2e_batch_smoke_results.json"
OUT_JSON = REPO_ROOT / "reports" / "anchoring_gate_calibration.json"

TOP_K = 5
THRESHOLDS = (0.10, 0.20, 0.25, 0.30, 0.34, 0.40, 0.50, 0.60, 0.67, 0.75)

#: Probe bases whose true-positive rate is real evidence for this gate.
NON_CIRCULAR_BASES = (
    "standard_absent_from_corpus",
    "clause_absent_from_standard",
)
CIRCULAR_BASES = ("term_absent_from_corpus",)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(
            f"missing required input: {path.relative_to(REPO_ROOT).as_posix()}\n"
            "Calibration reads the licensed corpus and the persisted BM25 index; "
            "both are Git-ignored. See data/README.md.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def bm25_context(
    chunks: list[dict[str, Any]], query: str
) -> list[dict[str, Any]]:
    return search_bm25(chunks, BM25_INDEX, query, top_k=TOP_K)


def stored_contexts(
    chunks: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """The hybrid+rerank top-5 the model actually saw, rejoined to clause text."""
    if not STORED_RUN.exists():
        return {}
    by_id = {record["chunk_id"]: record for record in chunks}
    payload = json.loads(STORED_RUN.read_text(encoding="utf-8"))
    contexts: dict[str, list[dict[str, Any]]] = {}
    for item in payload["items"]:
        ordered = sorted(item["top5"], key=lambda entry: entry["rank"])
        contexts[item["item_id"]] = [by_id[entry["chunk_id"]] for entry in ordered]
    return contexts


def measure(
    rows: Sequence[Mapping[str, Any]],
    contexts: Mapping[str, Sequence[Mapping[str, Any]]],
    vocabulary: CorpusVocabulary,
    spec: AnchorSpec,
) -> list[dict[str, Any]]:
    """One record per item: coverage, oov ratio and the anchor terms behind them."""
    out: list[dict[str, Any]] = []
    for row in rows:
        key = str(row["id"])
        signal = compute_anchoring_signal(
            str(row["question_text"]), list(contexts[key]), vocabulary, spec
        )
        out.append(
            {
                "id": key,
                "group": row["group"],
                "coverage": signal.idf_weighted_coverage,
                "unweighted_coverage": signal.context_anchor_coverage,
                "oov_ratio": signal.oov_ratio,
                "max_uncovered_idf": signal.max_uncovered_idf,
                "n_anchors": signal.n_anchors,
                "anchors": list(signal.anchors),
                "uncovered": list(signal.uncovered_anchors),
                "oov": list(signal.oov_anchors),
            }
        )
    return out


def rate(records: Sequence[Mapping[str, Any]], threshold: float) -> tuple[int, int]:
    """``(n_fired, n_evaluable)``. A ``None`` coverage cannot fire; see the module."""
    evaluable = [r for r in records if r["coverage"] is not None]
    fired = sum(1 for r in evaluable if r["coverage"] < threshold)
    return fired, len(evaluable)


def sweep(groups: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        entry: dict[str, Any] = {"threshold": threshold}
        for name, records in groups.items():
            fired, total = rate(records, threshold)
            entry[name] = {
                "fired": fired,
                "of": total,
                "rate": (fired / total) if total else None,
            }
        rows.append(entry)
    return rows


def auc(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    """Mann-Whitney AUC: P(a probe scores lower than an answerable item).

    Computed by hand rather than pulled from scipy so calibration has no extra
    dependency. Ties count a half. 0.5 is chance; the gate needs AUC well above
    0.5 to be worth enabling, and at 7 vs 11 items the sampling error on this
    statistic is enormous - it is reported to make a *lack* of separation
    unmistakable, not to claim a small one.
    """
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive < negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def scores(records: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    return [r[key] for r in records if r[key] is not None]


def print_auc(
    answerable: Sequence[Mapping[str, Any]],
    probes: Sequence[Mapping[str, Any]],
    label: str,
) -> None:
    print(f"\nSeparability, {label} (probe should score LOWER than answerable):")
    for key, name in (
        ("coverage", "idf_weighted_coverage"),
        ("unweighted_coverage", "context_anchor_coverage"),
        ("oov_ratio", "oov_ratio (inverted)"),
    ):
        if key == "oov_ratio":
            value = auc(
                [-s for s in scores(probes, key)],
                [-s for s in scores(answerable, key)],
            )
        else:
            value = auc(scores(probes, key), scores(answerable, key))
        rendered = "n/a" if value is None else f"{value:.3f}"
        print(f"  AUC {name:<28} {rendered}")


def print_sweep(title: str, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    print(f"\n{title}")
    header = f"{'thresh':>7}  " + "  ".join(f"{name:>26}" for name in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = []
        for name in columns:
            block = row[name]
            ratio = "  n/a" if block["rate"] is None else f"{block['rate']:.3f}"
            cells.append(f"{block['fired']:>3}/{block['of']:<3} {ratio:>7}".rjust(26))
        print(f"{row['threshold']:>7.2f}  " + "  ".join(cells))


def per_item(title: str, records: Sequence[Mapping[str, Any]]) -> None:
    print(f"\n{title}")
    print(f"{'id':<12} {'wcov':>6} {'ucov':>6} {'oov':>6} {'maxU':>6} {'#anc':>5}  uncovered anchors")
    print("-" * 92)
    for record in sorted(records, key=lambda r: (r["coverage"] is None, r["coverage"])):
        cov = "  none" if record["coverage"] is None else f"{record['coverage']:.3f}"
        ucov = (
            "  none"
            if record["unweighted_coverage"] is None
            else f"{record['unweighted_coverage']:.3f}"
        )
        oov = "  none" if record["oov_ratio"] is None else f"{record['oov_ratio']:.3f}"
        maxu = (
            "  none"
            if record["max_uncovered_idf"] is None
            else f"{record['max_uncovered_idf']:.2f}"
        )
        shown = ", ".join(record["uncovered"][:6]) or "-"
        print(
            f"{record['id']:<12} {cov:>6} {ucov:>6} {oov:>6} {maxu:>6} "
            f"{record['n_anchors']:>5}  {shown}"
        )


def build_rows(
    hard_set: Sequence[Mapping[str, Any]], probes: Sequence[Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    answerable = [
        {
            "id": str(item["item_id"]),
            "question_text": str(item["question_text"]),
            "group": "answerable_n7",
        }
        for item in hard_set
    ]
    probe_rows = [
        {
            "id": probe.probe_id,
            "question_text": probe.question_text,
            "group": probe.basis.value,
        }
        for probe in probes
    ]
    return answerable, probe_rows


def run(spec: AnchorSpec) -> dict[str, Any]:
    chunks = read_jsonl(CLAUSE_CHUNKS)
    if not BM25_INDEX.exists():
        print(f"missing BM25 index: {BM25_INDEX}", file=sys.stderr)
        raise SystemExit(2)
    vocabulary = CorpusVocabulary.from_chunks(chunks, spec)
    hard_set = read_jsonl(HARD_SET)
    probes = load_probes(PROBES)
    answerable, probe_rows = build_rows(hard_set, probes)

    bm25 = {
        row["id"]: bm25_context(chunks, row["question_text"])
        for row in [*answerable, *probe_rows]
    }
    stored = stored_contexts(chunks)

    measured_bm25 = measure([*answerable, *probe_rows], bm25, vocabulary, spec)
    measured_stored = (
        measure(answerable, stored, vocabulary, spec) if stored else []
    )
    return {
        "spec": spec.as_dict(),
        "vocabulary": {
            "n_documents": vocabulary.n_documents,
            "n_terms": vocabulary.n_terms,
        },
        "bm25": measured_bm25,
        "stored_hybrid_rerank": measured_stored,
    }


def group_by(records: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record["group"]), []).append(record)
    return groups


def report(outcome: Mapping[str, Any]) -> None:
    spec = outcome["spec"]
    vocab = outcome["vocabulary"]
    print(
        f"\n=== spec fold_suffixes={spec['fold_suffixes']} "
        f"drop_frame_terms={spec['drop_frame_terms']} min_idf={spec['min_idf']} "
        f"min_term_length={spec['min_term_length']} ==="
    )
    print("coverage column = IDF-weighted; ucov = unweighted, for comparison")
    print(f"vocabulary: {vocab['n_terms']} folded terms over {vocab['n_documents']} chunks")

    groups = group_by(outcome["bm25"])
    answerable = groups.get("answerable_n7", [])
    non_circular = [r for base in NON_CIRCULAR_BASES for r in groups.get(base, [])]
    circular = [r for base in CIRCULAR_BASES for r in groups.get(base, [])]
    cross = groups.get("cross_standard_comparison", [])

    columns = ["answerable_n7 (FP)", "valid probes (TP)", "term probes (circular)", "cross-standard"]
    rows = sweep(
        {
            "answerable_n7 (FP)": answerable,
            "valid probes (TP)": non_circular,
            "term probes (circular)": circular,
            "cross-standard": cross,
        }
    )
    print_sweep("BM25 top-5 context, threshold sweep:", rows, columns)
    print_auc(answerable, non_circular, "answerable n=7 vs valid probes n=11")
    print_auc(answerable, cross, "answerable n=7 vs cross-standard n=6")

    if outcome["stored_hybrid_rerank"]:
        stored_rows = sweep({"answerable_n7 (FP)": outcome["stored_hybrid_rerank"]})
        print_sweep(
            "Stored hybrid+rerank top-5 (the context the model saw), n=7:",
            stored_rows,
            ["answerable_n7 (FP)"],
        )

    per_item("Per-item, answerable n=7, BM25 top-5:", answerable)
    per_item("Per-item, probes, BM25 top-5:", [*non_circular, *circular, *cross])


def redact_for_publication(outcome: Mapping[str, Any]) -> dict[str, Any]:
    """Strip anchor term lists for the private hard set; keep them for probes.

    ``reports/`` is tracked. Probe questions are authored and published in
    ``data/probes/``, so their anchors are already public and are the most useful
    thing a reader can audit. Hard-set questions are Git-ignored, so their terms
    are reduced to counts - the rates in the sweep are unaffected either way.
    """
    def scrub(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        cleaned: list[dict[str, Any]] = []
        for record in records:
            entry = dict(record)
            if entry["group"] == "answerable_n7":
                for key in ("anchors", "uncovered", "oov"):
                    entry[f"n_{key}"] = len(entry.pop(key))
                entry["terms_withheld"] = "hard-set question text is Git-ignored"
            cleaned.append(entry)
        return cleaned

    return {
        **outcome,
        "bm25": scrub(outcome["bm25"]),
        "stored_hybrid_rerank": scrub(outcome["stored_hybrid_rerank"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    parser.add_argument(
        "--no-write", action="store_true", help="Print only; write nothing."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    variants = [
        AnchorSpec(),
        AnchorSpec(drop_frame_terms=False),
        AnchorSpec(fold_suffixes=False),
    ]
    outcomes = []
    for spec in variants:
        outcome = run(spec)
        report(outcome)
        outcomes.append(outcome)

    print(
        "\nInterpretation guard: 'term probes (circular)' is NOT this gate's "
        "true-positive rate. See the module docstring."
    )
    if args.no_write:
        print("\n--no-write: nothing written.")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "source": "scripts/calibrate_anchoring_gate.py",
                "top_k": TOP_K,
                "thresholds": list(THRESHOLDS),
                "circular_bases": list(CIRCULAR_BASES),
                "non_circular_bases": list(NON_CIRCULAR_BASES),
                "variants": [redact_for_publication(o) for o in outcomes],
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






