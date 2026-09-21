"""Regression checks for known AAOIFI PDF extraction boundary cases.

This check intentionally uses only record identifiers, page numbers, and
reference identifiers. It does not embed AAOIFI clause prose in repository
code.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def run_regression_checks(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Raise on a known false split or lost legitimate record."""
    by_clause: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_clause[(record["standard_id"], record["clause_id"])].append(record)

    failures: list[str] = []
    false_split_cases = (
        ("SS8", "3/1/5", 13, "SS8", "3/1/3", "3/1/5"),
        ("SS8", "5/6", 20, "SS8", "5/8", "5/6"),
        ("SS9", "5/2/2", 13, "SS9", "4/1/1", "5/2/2"),
        ("SS9", "5/2/2", 13, "SS9", "6/2", "5/2/2"),
        ("SS17", "5/2/8", 18, "SS17", "5/2/11", "5/2/8"),
    )
    for standard_id, clause_id, expected_page, parent_standard, parent_clause, reference_id in false_split_cases:
        matching = by_clause[(standard_id, clause_id)]
        case_name = f"{standard_id}:{clause_id}-> {parent_standard}:{parent_clause}"
        if len(matching) != 1 or matching[0]["source_page"] != expected_page:
            failures.append(f"false-split case not collapsed: {case_name}")
        parent = by_clause[(parent_standard, parent_clause)]
        if len(parent) != 1 or reference_id not in parent[0]["text"]:
            failures.append(f"reference was not retained in parent: {case_name}")

    arboun = by_clause[("SS9", "4/1/4")]
    if len(arboun) != 1 or arboun[0]["source_page"] != 10 or not arboun[0]["text"].startswith("\u2019"):
        failures.append("legitimate SS9:4/1/4 record was not preserved")

    ss26_agency = by_clause[("SS26", "3")]
    if (
        len(ss26_agency) != 1
        or ss26_agency[0]["source_page"] != 6
        or ss26_agency[0]["sub_clause_id"] is not None
    ):
        failures.append("SS26:3 parenthetical prose was incorrectly split as a sub-clause")

    ss13_lettered_items = [
        record
        for record in by_clause[("SS13", "9/1/6")]
        if record["sub_clause_id"] in {"a)", "b)"}
    ]
    if (
        [record["sub_clause_id"] for record in ss13_lettered_items] != ["a)", "b)"]
        or any(record["source_page"] != 12 for record in ss13_lettered_items)
    ):
        failures.append("legitimate SS13:9/1/6 lettered items were not preserved")

    ss17_duplicate = by_clause[("SS17", "3/6/2")]
    if (
        len(ss17_duplicate) != 2
        or [record["source_page"] for record in ss17_duplicate] != [8, 9]
        or [record["occurrence_index"] for record in ss17_duplicate] != [0, 1]
    ):
        failures.append("source-faithful SS17:3/6/2 duplicate was not preserved")

    if failures:
        raise AssertionError("; ".join(failures))

    return {
        "passed": True,
        "known_false_split_cases_checked": len(false_split_cases),
        "known_parenthetical_prose_case_checked": 1,
        "legitimate_cases_checked": 5,
        "preserved_duplicate_clause_ids": [
            {
                "standard_id": "SS17",
                "clause_id": "3/6/2",
                "occurrence_indexes": [0, 1],
                "source_pages": [8, 9],
            }
        ],
    }


def _load_private_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    extracted_dir = ROOT / "data" / "private" / "extracted"
    for path in sorted(extracted_dir.glob("ss*_clauses.jsonl")):
        with path.open("r", encoding="utf-8") as stream:
            records.extend(json.loads(line) for line in stream if line.strip())
    return records


if __name__ == "__main__":
    print(json.dumps(run_regression_checks(_load_private_records()), ensure_ascii=False, indent=2))
