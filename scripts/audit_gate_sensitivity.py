"""Threshold sensitivity and gate ablation over the stored n=7 run. GPU-free.

Two questions, both answered by re-routing **frozen** signals rather than by generating
anything:

1. ``max_echo_ratio = 0.85`` was set from three observed echo ratios. How far can it move
   before a decision changes, and which decision changes first?
2. Of the 9 enabled gates, how many actually change a decision at n=7, and what is the
   smallest subset that reproduces the shipped routing exactly?

Both are honest-limitation instruments. A wide stable interval in (1) is *not* evidence
that 0.85 is well chosen - it is evidence that three points cannot constrain a threshold.
A small sufficient subset in (2) is *not* evidence that the other gates are redundant - it
is evidence that their failure modes did not occur in seven items.

Run: ``.venv/Scripts/python.exe scripts/audit_gate_sensitivity.py``
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
    PolicyConfig,
    SelectivePredictionPolicy,
    compute_signals,
)
from aaoifi_rag.reliability.sensitivity import (  # noqa: E402
    SensitivityItem,
    compute_ablation_lattice,
    sweep_threshold,
)

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from replay_n7_router import (  # noqa: E402
    _StoredRerankerSignal,
    load_inputs,
    rejoin_context,
)

GATES_CONFIG = REPO_ROOT / "configs" / "reliability" / "gates_v1.json"
OUT_JSON = REPO_ROOT / "reports" / "gate_sensitivity_n7.json"


def build_items() -> list[SensitivityItem]:
    """Frozen signals for the seven stored items, exactly as the router saw them."""
    stored, chunks, hard_set = load_inputs()
    expected = {
        item["item_id"]: str(item.get("expected_behavior", "")) for item in hard_set
    }
    items: list[SensitivityItem] = []
    for entry in sorted(stored["items"], key=lambda row: row["item_id"]):
        item_id = entry["item_id"]
        context = rejoin_context(entry["top5"], chunks)
        signals = compute_signals(
            context,
            entry.get("model_response") or "",
            reranker_signal=_StoredRerankerSignal.from_records(context),
        )
        items.append(
            SensitivityItem(
                item_id=item_id,
                signals=signals,
                expected_behavior=expected.get(item_id) or None,
            )
        )
    return items


def print_sweep(sensitivity: Any) -> None:
    print(f"\n=== threshold sensitivity: {sensitivity.field_name} ===")
    print(
        f"shipped value {sensitivity.shipped_value} | items {sensitivity.n_items} "
        f"| evaluable {sensitivity.n_items_evaluable} "
        f"| distinct routings across the whole range {sensitivity.n_distinct_routings}"
    )
    observed = ", ".join(f"{value:.4f}" for value in sensitivity.observed_values)
    print(f"observed signal values (the only possible transition points): {observed}")
    low, high = sensitivity.stable_interval
    print(
        f"stable interval around the shipped value: [{low:.6f}, {high:.6f}] "
        f"width {sensitivity.stable_interval_width:.6f}"
    )
    if not sensitivity.is_decision_relevant:
        print("  NOTE: no threshold in [0,1] changes any decision on this evidence.")
    print("\n  threshold      answer abstain escalate  agree   shipped")
    for point in sensitivity.points:
        mark = "  <-- shipped" if point.is_shipped_value else ""
        agree = "-" if point.n_agree is None else str(point.n_agree)
        print(
            f"  {point.threshold:>12.9f}   {point.n_answer:>6} {point.n_abstain:>7} "
            f"{point.n_escalate:>8}  {agree:>5}{mark}"
        )
    print(f"\n  decision flips ({len(sensitivity.flips)}):")
    for flip in sensitivity.flips:
        print(
            f"    {flip.item_id}: {flip.decision_below} -> {flip.decision_at_or_above} "
            f"as the threshold crosses {flip.threshold_at_or_above:.6f}"
        )
    if not sensitivity.flips:
        print("    none")


def print_lattice(lattice: Any) -> None:
    print("\n=== gate ablation lattice ===")
    print(
        f"{len(lattice.enabled_gate_names)} enabled gates -> "
        f"{lattice.n_subsets} subsets evaluated over {lattice.n_items} items; "
        f"{lattice.n_distinct_routings} distinct routings reachable"
    )
    print(f"{'gate':<26}{'trig':>5}{'changed':>9}  shifts when removed")
    for contribution in lattice.contributions:
        shifts = "; ".join(contribution.decision_shifts) or "-"
        print(
            f"{contribution.gate_name:<26}{contribution.n_items_triggering:>5}"
            f"{contribution.n_items_changed:>9}  {shifts}"
        )
    print(
        f"\nload-bearing at n=7 ({len(lattice.load_bearing_gates)}): "
        f"{', '.join(lattice.load_bearing_gates) or 'none'}"
    )
    print(
        f"unexercised at n=7 ({len(lattice.unexercised_gates)}): "
        f"{', '.join(lattice.unexercised_gates) or 'none'}"
    )
    print(
        f"smallest subset reproducing the shipped routing: "
        f"{lattice.minimal_sufficient_size} gate(s); "
        f"{len(lattice.minimal_sufficient_subsets)} such subset(s)"
    )
    for subset in lattice.minimal_sufficient_subsets[:5]:
        print(f"  {{{', '.join(subset) or 'empty set'}}}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    parser.add_argument(
        "--no-write", action="store_true", help="Print only; write nothing."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = (
        PolicyConfig.from_json_file(GATES_CONFIG)
        if GATES_CONFIG.exists()
        else PolicyConfig()
    )
    policy = SelectivePredictionPolicy(config)
    items = build_items()
    print(
        f"items {len(items)} | policy {config.policy_version} | "
        f"max_echo_ratio {config.max_echo_ratio}"
    )
    print("baseline routing: " + ", ".join(
        f"{item.item_id}={policy.decide(item.signals).decision.value}" for item in items
    ))

    echo = sweep_threshold(items, field_name="max_echo_ratio", policy=policy)
    print_sweep(echo)
    lattice = compute_ablation_lattice(items, policy)
    print_lattice(lattice)

    if args.no_write:
        print("\n--no-write: nothing written.")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "source": "scripts/audit_gate_sensitivity.py",
                "evidence": "stored n=7 Colab run, replayed with frozen signals",
                "policy": config.as_dict(),
                "threshold_sensitivity": {"max_echo_ratio": echo.as_dict()},
                "ablation_lattice": lattice.as_dict(),
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
