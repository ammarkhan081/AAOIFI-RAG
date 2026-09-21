"""Conservative, layout-aware extraction of private AAOIFI source PDFs.

This module intentionally does not generate summaries, embeddings, answers, or
evaluation data. It records only source-derived clause records and extraction
diagnostics in Git-ignored storage.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable

import pdfplumber


STANDARD_SOURCES: tuple[tuple[str, str], ...] = (
    ("SS8", "SS-8-Murabahah.pdf"),
    ("SS9", "SS-9-Ijarah-and-Ijarah-Muntahia-Bittamleek.pdf"),
    ("SS13", "SS-13-Mudarabah.pdf"),
    ("SS17", "SS-17-Investment-Sukuk.pdf"),
    ("SS26", "SS-26-Islamic-Insurance.pdf"),
)

SECTION_RE = re.compile(r"^(?P<section>\d+)\.\s+(?P<text>.+)$")
CLAUSE_RE = re.compile(r"^(?P<clause>\d+(?:/\d+)+)\s*(?P<text>.*)$")
# Enumerations in the normative source pages use ``a)`` / ``b)``. Single-letter
# and Roman-numeral parenthetical forms are retained for source compatibility;
# full parenthetical words such as ``(Agency)`` are continuous prose, not labels.
BULLET_RE = re.compile(
    r"^(?P<bullet>(?:[a-z]\)|\((?:[a-z]|[ivxlcdm]+)\)))\s+(?P<text>.+)$"
)
HEADER_RE = re.compile(r"^Shari[’']ah Standard No\.\s*\(\d+\)", re.IGNORECASE)
PAGE_NUMBER_RE = re.compile(r"^\d{1,4}$")
STATEMENT_OF_STANDARD_RE = re.compile(r"^Statement of the Standard$", re.IGNORECASE)
DATE_OF_ISSUANCE_RE = re.compile(
    r"^(?:\d+\.\s+)?Date of Issuance of (?:the )?Standard\b", re.IGNORECASE
)
TOC_DOT_LEADER_RE = re.compile(r"\.{3,}\s*\d+\s*$")
REFERENCE_CUE_RE = re.compile(
    r"\b(?:see|items?|clauses?|paras?\.?|paragraphs?|sections?)\s*$",
    re.IGNORECASE,
)
CONTINUATION_MARKER_RE = re.compile(
    r"^(?:[,.;:\]\)]|(?:and|or|above|below|should)\b)"
)
CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
NORMAL_TYPOGRAPHIC_CODEPOINTS = frozenset("–‘’“”…")


@dataclass
class _OpenRecord:
    standard_id: str
    section_id: str
    clause_id: str
    sub_clause_id: str | None
    source_page: int
    fragments: list[str]

    def append(self, fragment: str) -> None:
        if not fragment:
            return
        if self.fragments and self.fragments[-1].endswith("-") and fragment[:1].islower():
            self.fragments[-1] = self.fragments[-1][:-1] + fragment
        else:
            self.fragments.append(fragment)

    def as_schema_record(self, updated_at: str) -> dict[str, Any]:
        return {
            "standard_id": self.standard_id,
            "section_id": self.section_id,
            "clause_id": self.clause_id,
            "occurrence_index": 0,
            "sub_clause_id": self.sub_clause_id,
            "text": " ".join(self.fragments).strip(),
            "source_page": self.source_page,
            "disagreement_tag": None,
            "last_updated": updated_at,
        }


def normalize_line(text: str) -> str:
    """Normalise PDF text without changing semantic content."""
    text = unicodedata.normalize("NFC", text).replace("\u00ad", "")
    return re.sub(r"\s+", " ", text).strip()


def clean_page_text(page: pdfplumber.page.Page) -> str:
    """Remove overlapping duplicate glyphs before reading text order.

    These AAOIFI PDFs contain duplicated character layers. `dedupe_chars` is
    necessary to avoid doubled words such as ``MMuurraabbaahhaahh``.
    """
    deduped = page.dedupe_chars(tolerance=1)
    return deduped.extract_text(x_tolerance=2, y_tolerance=3) or ""


def _is_ignored_line(line: str) -> bool:
    return (
        not line
        or HEADER_RE.match(line) is not None
        or PAGE_NUMBER_RE.match(line) is not None
        or line in {"Statement of the Standard", "Contents"}
    )


def _is_contents_page(lines: list[str]) -> bool:
    """Identify a table-of-contents page from its structure, not its number."""
    folded_lines = [line.casefold() for line in lines]
    has_contents_heading = "contents" in folded_lines
    has_subject_page_heading = any(line.startswith("subject page") for line in folded_lines)
    dot_leader_count = sum(TOC_DOT_LEADER_RE.search(line) is not None for line in lines)
    numbered_entry_count = sum(
        SECTION_RE.match(line) is not None and CLAUSE_RE.match(line) is None
        for line in lines
    )
    return has_contents_heading and (
        has_subject_page_heading or dot_leader_count >= 2 or numbered_entry_count >= 4
    )


def _record_reference(record: dict[str, Any]) -> dict[str, Any]:
    """Return compact record metadata for diagnostics and report gates."""
    return {
        "section_id": record["section_id"],
        "clause_id": record["clause_id"],
        "occurrence_index": record["occurrence_index"],
        "sub_clause_id": record["sub_clause_id"],
        "source_page": record["source_page"],
    }


def _reference_continuation_reason(
    content_after_identifier: str,
    square_bracket_depth: int,
    previous_context_line: str | None,
) -> str | None:
    """Identify a wrapped citation that must not open a numbered record."""
    if square_bracket_depth > 0:
        return "inside_unclosed_square_bracket"
    if previous_context_line and REFERENCE_CUE_RE.search(previous_context_line):
        return "preceding_line_ends_with_citation_cue"
    if CONTINUATION_MARKER_RE.match(content_after_identifier):
        return "continuation_marker_after_identifier"
    return None


def _assign_occurrence_indexes(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign source-order indexes without altering source clause identifiers."""
    occurrence_counts: Counter[tuple[str, str]] = Counter()
    for record in records:
        key = (record["standard_id"], record["clause_id"])
        record["occurrence_index"] = occurrence_counts[key]
        occurrence_counts[key] += 1

    return [
        {
            "standard_id": standard_id,
            "clause_id": clause_id,
            "occurrence_count": count,
        }
        for (standard_id, clause_id), count in sorted(occurrence_counts.items())
        if count > 1
    ]


def _unicode_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate literal replacement characters from legitimate typography."""
    replacement_records = [record for record in records if "\ufffd" in record["text"]]
    control_records = [record for record in records if CONTROL_RE.search(record["text"])]
    typographic_counts: Counter[str] = Counter()
    other_non_ascii_counts: Counter[str] = Counter()

    for record in records:
        for character in record["text"]:
            if character in NORMAL_TYPOGRAPHIC_CODEPOINTS:
                typographic_counts[f"U+{ord(character):04X}"] += 1
            elif ord(character) > 127 and character != "\ufffd":
                other_non_ascii_counts[
                    f"U+{ord(character):04X} {unicodedata.name(character, '<unnamed>')}"
                ] += 1

    return {
        "literal_replacement_character_count": sum(
            record["text"].count("\ufffd") for record in replacement_records
        ),
        "records_with_literal_replacement_character": [
            _record_reference(record) for record in replacement_records
        ],
        "control_character_record_count": len(control_records),
        "records_with_control_characters": [
            _record_reference(record) for record in control_records
        ],
        "normal_typographic_unicode_counts": dict(sorted(typographic_counts.items())),
        "other_non_ascii_codepoint_counts": dict(sorted(other_non_ascii_counts.items())),
    }


def extract_standard(
    pdf_path: Path,
    standard_id: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Extract section/clause records and cleaned pages from one source PDF.

    A record is emitted for every numbered section, slash-numbered clause, or
    lettered sub-clause detected after the first numbered section. This is a
    conservative parser: unfamiliar numeric forms are logged as suspicious,
    never guessed into a new identifier convention.
    """
    updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    records: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    suspicious_lines: list[dict[str, Any]] = []
    empty_pages: list[int] = []
    active: _OpenRecord | None = None
    current_section: str | None = None
    normative_started = False
    statement_of_standard_seen = False
    normative_start_page: int | None = None
    normative_start_reason: str | None = None
    back_matter_start_page: int | None = None
    skipped_toc_pages: list[int] = []
    excluded_toc_candidate_record_count = 0
    previous_context_line: str | None = None
    square_bracket_depth = 0
    excluded_reference_continuation_candidates: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal active
        if active is None:
            return
        record = active.as_schema_record(updated_at)
        if record["text"]:
            records.append(record)
        active = None

    def note_context_line(line: str) -> None:
        """Retain line context across page breaks for wrapped citations."""
        nonlocal previous_context_line, square_bracket_depth
        previous_context_line = line
        square_bracket_depth = max(
            0,
            square_bracket_depth + line.count("[") - line.count("]"),
        )

    with pdfplumber.open(pdf_path, unicode_norm="NFC") as pdf:
        total_pages = len(pdf.pages)
        for page in pdf.pages:
            source_page = page.page_number
            raw_text = clean_page_text(page)
            clean_lines = [normalize_line(line) for line in raw_text.splitlines()]
            clean_lines = [line for line in clean_lines if line]
            if not clean_lines:
                empty_pages.append(source_page)

            if not normative_started and _is_contents_page(clean_lines):
                skipped_toc_pages.append(source_page)
                excluded_toc_candidate_record_count += sum(
                    SECTION_RE.match(line) is not None and CLAUSE_RE.match(line) is None
                    for line in clean_lines
                )
                if progress_callback is not None and (
                    source_page % 5 == 0 or source_page == total_pages
                ):
                    progress_callback(source_page, total_pages)
                continue

            normative_page_lines: list[str] = []

            for line in clean_lines:
                if not normative_started:
                    if STATEMENT_OF_STANDARD_RE.match(line):
                        statement_of_standard_seen = True
                        continue
                    section_match = SECTION_RE.match(line)
                    if (
                        section_match
                        and section_match.group("section") == "1"
                        and CLAUSE_RE.match(line) is None
                    ):
                        normative_started = True
                        normative_start_page = source_page
                        normative_start_reason = (
                            "statement_of_standard_then_section_1"
                            if statement_of_standard_seen
                            else "section_1_fallback"
                        )
                    else:
                        continue

                if DATE_OF_ISSUANCE_RE.match(line):
                    flush()
                    back_matter_start_page = source_page
                    break

                if _is_ignored_line(line):
                    continue

                normative_page_lines.append(line)

                section_match = SECTION_RE.match(line)
                clause_match = CLAUSE_RE.match(line)
                bullet_match = BULLET_RE.match(line)
                candidate_match = clause_match or section_match
                reference_continuation_reason: str | None = None
                if candidate_match is not None:
                    candidate_identifier = (
                        candidate_match.group("clause")
                        if clause_match is not None
                        else candidate_match.group("section")
                    )
                    candidate_content = candidate_match.group("text")
                    reference_continuation_reason = _reference_continuation_reason(
                        candidate_content,
                        square_bracket_depth,
                        previous_context_line,
                    )
                    if reference_continuation_reason is not None:
                        excluded_reference_continuation_candidates.append(
                            {
                                "source_page": source_page,
                                "candidate_identifier": candidate_identifier,
                                "reason": reference_continuation_reason,
                            }
                        )
                        section_match = None
                        clause_match = None

                if section_match and not clause_match:
                    flush()
                    current_section = section_match.group("section")
                    active = _OpenRecord(
                        standard_id=standard_id,
                        section_id=current_section,
                        clause_id=current_section,
                        sub_clause_id=None,
                        source_page=source_page,
                        fragments=[section_match.group("text")],
                    )
                    note_context_line(line)
                    continue

                if clause_match:
                    flush()
                    clause_id = clause_match.group("clause")
                    current_section = clause_id.split("/", maxsplit=1)[0]
                    active = _OpenRecord(
                        standard_id=standard_id,
                        section_id=current_section,
                        clause_id=clause_id,
                        sub_clause_id=None,
                        source_page=source_page,
                        fragments=[clause_match.group("text")] if clause_match.group("text") else [],
                    )
                    note_context_line(line)
                    continue

                if bullet_match and active is not None:
                    parent_clause = active.clause_id
                    parent_section = active.section_id
                    flush()
                    active = _OpenRecord(
                        standard_id=standard_id,
                        section_id=parent_section,
                        clause_id=parent_clause,
                        sub_clause_id=bullet_match.group("bullet"),
                        source_page=source_page,
                        fragments=[bullet_match.group("text")],
                    )
                    note_context_line(line)
                    continue

                if active is not None:
                    active.append(line)
                elif re.match(r"^\d", line):
                    suspicious_lines.append({"source_page": source_page, "reason": "unparsed_numeric_line"})
                note_context_line(line)

            if normative_page_lines:
                pages.append(
                    {
                        "standard_id": standard_id,
                        "source_page": source_page,
                        "text": "\n".join(normative_page_lines),
                    }
                )

            if back_matter_start_page is not None:
                if progress_callback is not None and source_page != total_pages:
                    progress_callback(total_pages, total_pages)
                break

            if progress_callback is not None and (source_page % 5 == 0 or source_page == total_pages):
                progress_callback(source_page, total_pages)

    flush()
    duplicate_clause_id_groups = _assign_occurrence_indexes(records)
    short_records = [record for record in records if len(record["text"].split()) < 5]
    long_records = [record for record in records if len(record["text"].split()) > 700]
    unicode_diagnostics = _unicode_diagnostics(records)
    toc_gate_violations = [
        record
        for record in records
        if record["source_page"] in skipped_toc_pages
        or TOC_DOT_LEADER_RE.search(record["text"]) is not None
    ]
    back_matter_gate_violations = [
        record for record in records if DATE_OF_ISSUANCE_RE.search(record["text"]) is not None
    ]
    report_gate_failures: list[str] = []
    if normative_start_page is None:
        report_gate_failures.append("normative_body_start_not_detected")
    if toc_gate_violations:
        report_gate_failures.append("toc_like_record_emitted")
    if back_matter_gate_violations:
        report_gate_failures.append("back_matter_record_emitted")
    sections = sorted({record["section_id"] for record in records}, key=lambda value: int(value))
    skipped_back_matter_pages = (
        list(range(back_matter_start_page, total_pages + 1))
        if back_matter_start_page is not None
        else []
    )
    return records, pages, {
        "standard_id": standard_id,
        "source_pdf": pdf_path.name,
        "source_pdf_pages": total_pages,
        "records": len(records),
        "normative_pages": len(pages),
        "sub_clause_records": sum(record["sub_clause_id"] is not None for record in records),
        "sections_detected": sections,
        "empty_pages": empty_pages,
        "normative_start_page": normative_start_page,
        "normative_start_reason": normative_start_reason,
        "skipped_toc_pages": skipped_toc_pages,
        "skipped_toc_page_count": len(skipped_toc_pages),
        "excluded_toc_candidate_record_count": excluded_toc_candidate_record_count,
        "excluded_reference_continuation_candidate_count": len(
            excluded_reference_continuation_candidates
        ),
        "excluded_reference_continuation_candidates": excluded_reference_continuation_candidates,
        "back_matter_start_page": back_matter_start_page,
        "back_matter_pages_with_excluded_content": skipped_back_matter_pages,
        "skipped_back_matter_page_count": len(skipped_back_matter_pages),
        "unparsed_numeric_lines": suspicious_lines,
        "short_records": [
            {"section_id": record["section_id"], "clause_id": record["clause_id"], "source_page": record["source_page"]}
            for record in short_records
        ],
        "long_records": [
            {"section_id": record["section_id"], "clause_id": record["clause_id"], "source_page": record["source_page"]}
            for record in long_records
        ],
        "unicode_diagnostics": unicode_diagnostics,
        "source_clause_id_duplicate_groups": duplicate_clause_id_groups,
        "report_gate": {
            "passed": not report_gate_failures,
            "failures": report_gate_failures,
            "toc_like_records_emitted": [
                _record_reference(record) for record in toc_gate_violations
            ],
            "back_matter_records_emitted": [
                _record_reference(record) for record in back_matter_gate_violations
            ],
        },
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
