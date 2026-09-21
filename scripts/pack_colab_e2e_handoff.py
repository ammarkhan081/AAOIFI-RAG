"""Build a single private zip for the Colab hard-set smoke test.

Run from the repository root:

    python scripts/pack_colab_e2e_handoff.py

Writes data/private/extracted/colab_e2e_handoff.zip

The zip contains clause chunks, the current hard set (all items in
data/private/hard_set.jsonl), the Chroma store, and the retrieval Python
package. Re-run this packer after hard-set edits so Colab sees H01–H07.
It does not include AAOIFI PDFs or other source scans. Keep the zip
private; do not commit or publish it.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private"
EXTRACTED = PRIVATE / "extracted"
OUT_PATH = EXTRACTED / "colab_e2e_handoff.zip"

# Explicit whitelist only. PDFs and page/clause dumps are not included.
FILES: list[tuple[Path, str]] = [
    (EXTRACTED / "clause_chunks.jsonl", "data/private/extracted/clause_chunks.jsonl"),
    (PRIVATE / "hard_set.jsonl", "data/private/hard_set.jsonl"),
    (EXTRACTED / "retrieval_ablation_results.json", "data/private/extracted/retrieval_ablation_results.json"),
    # Stored e2e results (for context reconstruction in prompt ablation)
    (ROOT / "reports" / "e2e_batch_smoke_results.json", "reports/e2e_batch_smoke_results.json"),
    # Gates config
    (ROOT / "configs" / "reliability" / "gates_v1.json", "configs/reliability/gates_v1.json"),
    # --- src package: root ---
    (ROOT / "src" / "aaoifi_rag" / "__init__.py", "src/aaoifi_rag/__init__.py"),
    # --- src package: retrieval ---
    (
        ROOT / "src" / "aaoifi_rag" / "retrieval" / "__init__.py",
        "src/aaoifi_rag/retrieval/__init__.py",
    ),
    (
        ROOT / "src" / "aaoifi_rag" / "retrieval" / "bm25_baseline.py",
        "src/aaoifi_rag/retrieval/bm25_baseline.py",
    ),
    (
        ROOT / "src" / "aaoifi_rag" / "retrieval" / "bge_reranker.py",
        "src/aaoifi_rag/retrieval/bge_reranker.py",
    ),
    # --- src package: generation (for prompt ablation) ---
    (
        ROOT / "src" / "aaoifi_rag" / "generation" / "__init__.py",
        "src/aaoifi_rag/generation/__init__.py",
    ),
    (
        ROOT / "src" / "aaoifi_rag" / "generation" / "prompt.py",
        "src/aaoifi_rag/generation/prompt.py",
    ),
    (
        ROOT / "src" / "aaoifi_rag" / "generation" / "prompt_config.py",
        "src/aaoifi_rag/generation/prompt_config.py",
    ),
    (
        ROOT / "src" / "aaoifi_rag" / "generation" / "types.py",
        "src/aaoifi_rag/generation/types.py",
    ),
    # --- src package: orchestration ---
    (
        ROOT / "src" / "aaoifi_rag" / "orchestration" / "__init__.py",
        "src/aaoifi_rag/orchestration/__init__.py",
    ),
    (
        ROOT / "src" / "aaoifi_rag" / "orchestration" / "protocols.py",
        "src/aaoifi_rag/orchestration/protocols.py",
    ),
    (
        ROOT / "src" / "aaoifi_rag" / "orchestration" / "pipeline.py",
        "src/aaoifi_rag/orchestration/pipeline.py",
    ),
    # --- src package: reliability ---
    (
        ROOT / "src" / "aaoifi_rag" / "reliability" / "__init__.py",
        "src/aaoifi_rag/reliability/__init__.py",
    ),
    (
        ROOT / "src" / "aaoifi_rag" / "reliability" / "anchoring.py",
        "src/aaoifi_rag/reliability/anchoring.py",
    ),
    (
        ROOT / "src" / "aaoifi_rag" / "reliability" / "citations.py",
        "src/aaoifi_rag/reliability/citations.py",
    ),
    (
        ROOT / "src" / "aaoifi_rag" / "reliability" / "grounding.py",
        "src/aaoifi_rag/reliability/grounding.py",
    ),
    (
        ROOT / "src" / "aaoifi_rag" / "reliability" / "policy.py",
        "src/aaoifi_rag/reliability/policy.py",
    ),
    (
        ROOT / "src" / "aaoifi_rag" / "reliability" / "response_class.py",
        "src/aaoifi_rag/reliability/response_class.py",
    ),
    (
        ROOT / "src" / "aaoifi_rag" / "reliability" / "sensitivity.py",
        "src/aaoifi_rag/reliability/sensitivity.py",
    ),
    (
        ROOT / "src" / "aaoifi_rag" / "reliability" / "signals.py",
        "src/aaoifi_rag/reliability/signals.py",
    ),
    (
        ROOT / "src" / "aaoifi_rag" / "reliability" / "text_checks.py",
        "src/aaoifi_rag/reliability/text_checks.py",
    ),
]


def _add_chroma(archive: zipfile.ZipFile) -> int:
    chroma_dir = EXTRACTED / "chroma_bge_m3"
    sqlite = chroma_dir / "chroma.sqlite3"
    if not sqlite.is_file():
        raise FileNotFoundError(f"Missing Chroma store: {sqlite}")
    added = 0
    for path in chroma_dir.rglob("*"):
        if path.is_file():
            archive.write(path, Path("chroma_bge_m3") / path.relative_to(chroma_dir))
            added += 1
    return added


def main() -> None:
    missing = [str(src) for src, _ in FILES if not src.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n  " + "\n  ".join(missing))
    forbidden = list(PRIVATE.glob("*.pdf")) + list(PRIVATE.glob("**/*.pdf"))
    # Presence of PDFs in data/private is expected locally; they must not enter the zip.
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for src, arcname in FILES:
            archive.write(src, arcname)
        chroma_files = _add_chroma(archive)
        names = archive.namelist()
    pdf_in_zip = [name for name in names if name.lower().endswith(".pdf")]
    if pdf_in_zip:
        OUT_PATH.unlink(missing_ok=True)
        raise RuntimeError(f"Refusing to write a zip that contains PDFs: {pdf_in_zip}")
    print(f"Wrote {OUT_PATH}")
    print(f"  members: {len(names)} (chroma files: {chroma_files})")
    print(f"  bytes:   {OUT_PATH.stat().st_size}")
    print("Upload this one file to Colab as /content/colab_e2e_handoff.zip")
    print(f"Local PDFs were not packed ({len(forbidden)} pdf path(s) exist under data/private).")


if __name__ == "__main__":
    main()
