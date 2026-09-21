"""Build the private extraction output and Phase-2 lexical retrieval baselines.

Run from the repository root after installing the limited Phase 1-2 baseline
dependencies. This script does not download models or make network calls.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aaoifi_rag.data.extraction import STANDARD_SOURCES, extract_standard, write_jsonl
from aaoifi_rag.retrieval.bm25_baseline import (
    FIXED_CHUNK_OVERLAP,
    FIXED_CHUNK_SIZE,
    FIXED_TOKENIZER_NAME,
    build_clause_chunks,
    build_fixed_token_chunks,
    persist_bm25_index,
)
from check_extraction_regressions import run_regression_checks


def main() -> None:
    source_dir = ROOT / "data" / "private"
    output_dir = source_dir / "extracted"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    all_pages: list[dict] = []
    per_standard: dict[str, dict] = {}
    for standard_id, filename in STANDARD_SOURCES:
        pdf_path = source_dir / filename
        if not pdf_path.is_file():
            raise FileNotFoundError(f"Missing expected source PDF: {pdf_path}")
        print(f"Starting {standard_id}...", flush=True)
        extraction_started = time.perf_counter()
        try:
            records, pages, diagnostics = extract_standard(
                pdf_path,
                standard_id,
                progress_callback=lambda current, total: print(
                    f"{standard_id} processed page {current}/{total}", flush=True
                ),
            )
        except Exception:
            elapsed = time.perf_counter() - extraction_started
            error_path = output_dir / "error_log.txt"
            with error_path.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(traceback.format_exc())
            print(f"{standard_id} failed after {elapsed:.1f}s", flush=True)
            raise
        elapsed = time.perf_counter() - extraction_started
        if not diagnostics["report_gate"]["passed"]:
            raise RuntimeError(
                f"Extraction report gate failed for {standard_id}: "
                f"{', '.join(diagnostics['report_gate']['failures'])}"
            )
        print(f"Finished {standard_id}: {len(records)} records", flush=True)
        print(f"{standard_id} took {elapsed:.1f}s", flush=True)
        write_started = time.perf_counter()
        write_jsonl(output_dir / f"{standard_id.lower()}_clauses.jsonl", records)
        print(f"{standard_id} clause JSONL write took {time.perf_counter() - write_started:.1f}s", flush=True)
        write_started = time.perf_counter()
        write_jsonl(output_dir / f"{standard_id.lower()}_pages.jsonl", pages)
        print(f"{standard_id} page JSONL write took {time.perf_counter() - write_started:.1f}s", flush=True)
        all_records.extend(records)
        all_pages.extend(pages)
        per_standard[standard_id] = diagnostics

    regression_checks = run_regression_checks(all_records)
    print(json.dumps({"extraction_regression_checks": regression_checks}, ensure_ascii=False), flush=True)

    fixed_chunks = build_fixed_token_chunks(all_pages)
    clause_chunks = build_clause_chunks(all_records)
    write_started = time.perf_counter()
    write_jsonl(output_dir / "fixed_token_chunks.jsonl", fixed_chunks)
    print(f"Fixed-token chunk JSONL write took {time.perf_counter() - write_started:.1f}s", flush=True)
    write_started = time.perf_counter()
    write_jsonl(output_dir / "clause_chunks.jsonl", clause_chunks)
    print(f"Clause chunk JSONL write took {time.perf_counter() - write_started:.1f}s", flush=True)
    persist_bm25_index(fixed_chunks, output_dir / "bm25_fixed_token.pkl", "bm25_fixed_token")
    persist_bm25_index(clause_chunks, output_dir / "bm25_clause_level.pkl", "bm25_clause_level")

    report = {
        "extraction_library": "pdfplumber",
        "character_deduplication": "page.dedupe_chars(tolerance=1)",
        "schema_output": "one JSONL record per clause/sub-clause, conforming to schemas/clause_schema.json",
        "standards": per_standard,
        "regression_checks": regression_checks,
        "totals": {
            "records": len(all_records),
            "pages": len(all_pages),
            "fixed_token_chunks": len(fixed_chunks),
            "clause_chunks": len(clause_chunks),
        },
        "fixed_token_baseline": {
            "tokenizer": FIXED_TOKENIZER_NAME,
            "chunk_size": FIXED_CHUNK_SIZE,
            "overlap": FIXED_CHUNK_OVERLAP,
            "note": "This tokenizer is a deterministic baseline choice, not the future Jais tokenizer.",
        },
        "blocked": {
            "dense_embeddings": "Deferred: BGE-M3 model access/download is out of scope.",
            "reranking": "Deferred: BGE-reranker model access/download is out of scope.",
            "sac": "Blocked: summary generation requires the future Jais-2 access; no rule-based approximation was used.",
            "generation_evaluation": "Out of scope for this prompt.",
        },
    }
    with (output_dir / "extraction_report.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    # The caller selects and labels throwaway manual sanity queries after
    # inspecting the extracted corpus. No evaluation questions are generated here.
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
