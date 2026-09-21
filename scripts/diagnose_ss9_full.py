"""Run the real clause-extraction function for SS9 only, with progress."""

from __future__ import annotations

from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aaoifi_rag.data.extraction import extract_standard


def main() -> None:
    pdf_path = ROOT / "data" / "private" / "SS-9-Ijarah-and-Ijarah-Muntahia-Bittamleek.pdf"
    print(f"Starting full SS9 extraction: {pdf_path}", flush=True)
    started = time.perf_counter()
    try:
        records, pages, diagnostics = extract_standard(
            pdf_path,
            "SS9",
            progress_callback=lambda current, total: print(
                f"SS9 processed page {current}/{total}", flush=True
            ),
        )
    except Exception:
        elapsed = time.perf_counter() - started
        print(f"SS9 extraction failed after {elapsed:.1f}s", flush=True)
        traceback.print_exc()
        raise
    elapsed = time.perf_counter() - started
    print(f"SS9 extraction completed: {len(records)} records, {len(pages)} pages", flush=True)
    print(f"SS9 took {elapsed:.1f}s", flush=True)
    print(f"SS9 sub-clause records: {diagnostics['sub_clause_records']}", flush=True)


if __name__ == "__main__":
    main()
