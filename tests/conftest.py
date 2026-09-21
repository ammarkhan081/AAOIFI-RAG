"""Shared fixtures. Real corpus where it exists, synthetic where it must not matter.

Two rules govern everything in ``tests/``:

1. **No test may require a GPU, torch, transformers, rank_bm25 or a network call.** The
   reliability and reporting layers are specified to be importable and testable without
   them, and a test suite that quietly depends on them would stop enforcing that.
   ``test_import_boundaries.py`` asserts it directly.
2. **Tests that need the licensed corpus are marked and skipped, never faked.** A
   synthetic stand-in for AAOIFI clause text would let a corpus-dependent assertion pass
   against data the project does not have, which is the failure mode
   ``verify_probe`` exists to prevent. Use the ``requires_private_data`` marker and the
   ``clause_chunks`` fixture, which skips rather than substitutes.

No fixture in this file contains AAOIFI clause prose. The synthetic records use obviously
invented text.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # belt and braces alongside pyproject's pythonpath
    sys.path.insert(0, str(SRC))

CLAUSE_CHUNKS = REPO_ROOT / "data" / "private" / "extracted" / "clause_chunks.jsonl"
BM25_INDEX = REPO_ROOT / "data" / "private" / "extracted" / "bm25_clause_level.pkl"
HARD_SET = REPO_ROOT / "data" / "private" / "hard_set.jsonl"
PROBES = REPO_ROOT / "data" / "probes" / "unanswerable_probes.jsonl"
GATES_CONFIG = REPO_ROOT / "configs" / "reliability" / "gates_v1.json"
STORED_RUN = REPO_ROOT / "reports" / "e2e_batch_smoke_results.json"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def clause_chunks() -> list[dict[str, Any]]:
    """The real 362-clause corpus, or a skip. Never a substitute."""
    if not CLAUSE_CHUNKS.exists():
        pytest.skip(
            f"{CLAUSE_CHUNKS.relative_to(REPO_ROOT).as_posix()} absent "
            "(licensed corpus, Git-ignored - see data/README.md)"
        )
    return _read_jsonl(CLAUSE_CHUNKS)


@pytest.fixture(scope="session")
def hard_set() -> list[dict[str, Any]]:
    if not HARD_SET.exists():
        pytest.skip(
            f"{HARD_SET.relative_to(REPO_ROOT).as_posix()} absent (Git-ignored)"
        )
    return _read_jsonl(HARD_SET)


@pytest.fixture(scope="session")
def probes_path() -> Path:
    """The tracked probe file. Its absence is a real failure, not a skip."""
    assert PROBES.exists(), (
        f"{PROBES.relative_to(REPO_ROOT).as_posix()} is tracked and must be present"
    )
    return PROBES


@pytest.fixture(scope="session")
def stored_run() -> dict[str, Any]:
    if not STORED_RUN.exists():
        pytest.skip(
            f"{STORED_RUN.relative_to(REPO_ROOT).as_posix()} absent (Git-ignored run "
            "artifact carrying clause text)"
        )
    return json.loads(STORED_RUN.read_text(encoding="utf-8"))


@pytest.fixture
def synthetic_context() -> list[dict[str, Any]]:
    """Five invented normative-looking records in the shape retrieval emits.

    The text is deliberately not AAOIFI prose. Every assertion built on this fixture is
    about the *mechanics* of a check - shingle overlap, citation ranks, heading detection
    - none of which depends on the content being real.
    """
    return [
        {
            "chunk_id": f"clause:XX{index}:1/{index}:occurrence:1",
            "standard_id": f"XX{index}",
            "clause_id": f"1/{index}",
            "sub_clause_id": None,
            "occurrence_index": 1,
            "page_number": 10 + index,
            "text": (
                f"The institution shall record the widget number {index} in its books "
                f"at the value agreed between the parties, and shall disclose that "
                f"value in the notes to the financial statements for period {index}."
            ),
            "reranker_score": round(0.9 - 0.1 * index, 4),
        }
        for index in range(1, 6)
    ]


@pytest.fixture
def heading_context() -> list[dict[str, Any]]:
    """Records that all look like clause headings rather than normative sentences."""
    return [
        {
            "chunk_id": f"clause:XX:{index}:occurrence:1",
            "standard_id": "XX",
            "clause_id": str(index),
            "sub_clause_id": None,
            "occurrence_index": 1,
            "page_number": index,
            "text": f"Section {index} Scope And Definitions",
            "reranker_score": 0.5,
        }
        for index in range(1, 4)
    ]
