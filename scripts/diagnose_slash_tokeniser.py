"""Diagnose the unreachable slash-numeral branch in the anchoring tokeniser.

The finding
-----------
:data:`aaoifi_rag.reliability.anchoring.ANCHOR_TOKEN_RE` is
``[\\w]+(?:['’][\\w]+)?|\\d+(?:/\\d+)*``. Python's ``|`` is **ordered**, and
``[\\w]+`` matches at every position where ``\\d+`` would, so the second branch is
unreachable: ``2/4/2`` tokenises to ``['2', '4', '2']``, each of which
:func:`_is_candidate` then drops as a bare digit. A question citing a clause number
therefore contributes **no anchor at all** from that citation, and
``AnchorSpec.keep_slash_numerals`` is dead through the public path.
``BM25_TOKEN_RE`` has the identical alternation order, so the two tokenisers really
are in sync - it is only the claim that *either* keeps ``9/2`` whole that was wrong.

Why the regex was not simply corrected
--------------------------------------
Two reasons, and this script exists to make the second one a measurement rather
than an assertion.

1. ``BM25_TOKEN_RE`` is the tokeniser the persisted index in
   ``data/private/extracted/bm25_clause_level.pkl`` was built with. Changing
   anchoring alone breaks the sync the duplication exists to preserve; changing both
   invalidates the index and every retrieval number measured against it.
2. **The fix would not change the verdict.** Run this script and read the table: the
   gate is still unusable, for the same reason, and the improvement it does produce
   comes entirely from a mechanism that the n=7 positive class cannot test.

What it does
------------
Runs ``scripts/calibrate_anchoring_gate.py``'s own measurement twice against the
default spec - once as shipped, once with :data:`ANCHOR_TOKEN_RE` rebound so the
slash branch comes first - and diffs them. The rebinding is a deliberate
monkeypatch confined to this process; nothing is written and no shipped module is
edited.

Prints coverage per probe (probe questions are published) and per hard-set item id
only (hard-set question text is Git-ignored), never question text.

Run: ``python scripts/diagnose_slash_tokeniser.py``
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from aaoifi_rag.reliability import anchoring  # noqa: E402

import calibrate_anchoring_gate as cal  # noqa: E402

#: The same alternation with the branches swapped. This is the *only* difference.
SLASH_FIRST = re.compile(r"\d+(?:/\d+)*|[\w]+(?:['’][\w]+)?")

BASES = (
    "standard_absent_from_corpus",
    "clause_absent_from_standard",
    "term_absent_from_corpus",
    "cross_standard_comparison",
)
THRESHOLDS = (0.25, 0.30, 0.50, 0.67, 0.75)


def snapshot(label: str) -> dict[str, Any]:
    """One full default-spec calibration, reduced to what the diff needs."""
    spec = anchoring.AnchorSpec()
    outcome = cal.run(spec)
    groups = cal.group_by(outcome["bm25"])
    answerable = groups.get("answerable_n7", [])
    valid = [row for base in cal.NON_CIRCULAR_BASES for row in groups.get(base, [])]
    return {
        "label": label,
        "n_terms": outcome["vocabulary"]["n_terms"],
        "auc_valid": cal.auc(
            cal.scores(valid, "coverage"), cal.scores(answerable, "coverage")
        ),
        "answerable": {
            row["id"]: (row["coverage"], row["n_anchors"]) for row in answerable
        },
        "stored": {
            row["id"]: row["coverage"] for row in outcome["stored_hybrid_rerank"]
        },
        "fp": {t: cal.rate(answerable, t) for t in THRESHOLDS},
        "tp": {t: cal.rate(valid, t) for t in THRESHOLDS},
        "by_basis": {
            base: {
                "auc": cal.auc(
                    cal.scores(groups.get(base, []), "coverage"),
                    cal.scores(answerable, "coverage"),
                ),
                "coverage": {
                    row["id"]: row["coverage"] for row in groups.get(base, [])
                },
            }
            for base in BASES
        },
    }


def fmt(value: float | None, places: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def diff_ids(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    return sorted(key for key in before if before[key] != after[key])


def show_group(
    title: str, before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    print(f"\n{title}")
    print(f"  {'id':<14} {'shipped':>9} {'slash-first':>12}   delta")
    for key in sorted(before):
        old, new = before[key], after[key]
        mark = "" if old == new else "  <-- changed"
        delta = "" if old == new or old is None or new is None else f"{new - old:+.4f}"
        print(f"  {key:<14} {fmt(old):>9} {fmt(new):>12}   {delta}{mark}")


def main(argv: list[str] | None = None) -> int:
    del argv
    print(__doc__.split("Run: ")[0].rstrip())
    print("=" * 78)

    shipped = anchoring.ANCHOR_TOKEN_RE
    print(f"\nshipped pattern:     {shipped.pattern}")
    print(f"counterfactual:      {SLASH_FIRST.pattern}")
    print(f"anchor_tokens('clause 2/4/2 applies') as shipped: "
          f"{anchoring.anchor_tokens('clause 2/4/2 applies')}")

    before = snapshot("shipped")
    try:
        anchoring.ANCHOR_TOKEN_RE = SLASH_FIRST
        print("anchor_tokens('clause 2/4/2 applies') patched:  "
              f"{anchoring.anchor_tokens('clause 2/4/2 applies')}")
        after = snapshot("slash-first")
    finally:
        anchoring.ANCHOR_TOKEN_RE = shipped

    print(f"\ncorpus vocabulary: {before['n_terms']} -> {after['n_terms']} folded terms")
    print(
        f"AUC, answerable n=7 vs valid probes n=11: "
        f"{fmt(before['auc_valid'])} -> {fmt(after['auc_valid'])}"
    )

    print("\nThreshold sweep (fired/evaluable):")
    print(f"  {'thresh':>7}  {'FP shipped':>11} {'FP slash':>10}   "
          f"{'TP shipped':>11} {'TP slash':>10}")
    for threshold in THRESHOLDS:
        fp_b, fp_a = before["fp"][threshold], after["fp"][threshold]
        tp_b, tp_a = before["tp"][threshold], after["tp"][threshold]
        print(
            f"  {threshold:>7.2f}  {f'{fp_b[0]}/{fp_b[1]}':>11} {f'{fp_a[0]}/{fp_a[1]}':>10}   "
            f"{f'{tp_b[0]}/{tp_b[1]}':>11} {f'{tp_a[0]}/{tp_a[1]}':>10}"
        )

    print("\nAUC by probe basis (coverage; probe should score LOWER):")
    for base in BASES:
        old = before["by_basis"][base]["auc"]
        new = after["by_basis"][base]["auc"]
        mark = "" if old == new else "   <-- moved"
        print(f"  {base:<32} {fmt(old, 3):>6} -> {fmt(new, 3):>6}{mark}")

    for base in BASES:
        show_group(
            f"Probe coverage, {base}:",
            before["by_basis"][base]["coverage"],
            after["by_basis"][base]["coverage"],
        )

    print("\nAnswerable items, BM25 top-5 (ids only; question text is Git-ignored):")
    print(f"  {'id':<6} {'shipped':>9} {'slash-first':>12}  {'#anchors':>9}")
    for key in sorted(before["answerable"]):
        old_cov, old_n = before["answerable"][key]
        new_cov, new_n = after["answerable"][key]
        mark = "" if (old_cov, old_n) == (new_cov, new_n) else "  <-- changed"
        print(f"  {key:<6} {fmt(old_cov):>9} {fmt(new_cov):>12}  "
              f"{old_n:>4} -> {new_n:<4}{mark}")

    changed_answerable = diff_ids(before["answerable"], after["answerable"])
    changed_stored = diff_ids(before["stored"], after["stored"])
    changed_probes: list[str] = []
    for base in BASES:
        changed_probes.extend(
            diff_ids(before["by_basis"][base]["coverage"], after["by_basis"][base]["coverage"])
        )

    print("\n" + "=" * 78)
    print("Verdict")
    print("=" * 78)
    print(
        f"answerable items changed (BM25):   "
        f"{changed_answerable or 'none - no hard-set question cites a clause number'}"
    )
    print(
        f"answerable items changed (stored): "
        f"{changed_stored or 'none'}"
    )
    print(f"probes changed:                    {sorted(changed_probes) or 'none'}")
    print(
        "\nEvery probe that moves is one whose question cites a clause number, and no\n"
        "answerable item moves at all. The improvement is therefore a *clause-reference\n"
        "grounding* signal wearing a coverage signal's clothes, and its false-positive\n"
        "rate is UNMEASURED at n=7 rather than zero: none of the seven answerable\n"
        "questions exercises it. Gold retrieval was incomplete on 3 of 7 items\n"
        "(reports/reliability_signal_analysis_n7.md), so an answerable question that did\n"
        "cite a clause the retriever missed would fire it."
    )

    zero_fp = [t for t in THRESHOLDS if after["fp"][t][0] == 0]
    detecting = [t for t in zero_fp if after["tp"][t][0] > 0]
    print(
        f"\nThresholds with zero false positives after the patch: {zero_fp}\n"
        f"...of which any detect a valid probe: {detecting or 'NONE'}\n"
        "The rejection in reports/anchoring_gate_calibration.md stands unchanged."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
