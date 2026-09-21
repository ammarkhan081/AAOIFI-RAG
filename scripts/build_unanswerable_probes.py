"""Build ``data/probes/unanswerable_probes.jsonl`` and verify it against the corpus.

The probe questions are authored here, in code, so the file is reproducible rather
than hand-maintained. Every ``basis_evidence`` number is computed from the live
corpus at build time and re-checked by :func:`aaoifi_rag.reporting.probes.verify_probes`
before anything is written. If a claim does not hold, the script exits non-zero and
writes nothing.

Deliberate restraint on what the questions assert
-------------------------------------------------
``standard_absent_from_corpus`` probes name a standard **by number only** ("AAOIFI
Shari'ah Standard No. 11"). They do not claim what that standard covers. The
project can verify from ``data/manifests/source_manifest.json`` that the index
holds SS8, SS9, SS13, SS17 and SS26 and nothing else; it cannot verify AAOIFI's
wider catalogue from its own files, so it does not assert one. The probe is
unanswerable because no clause of that standard is indexed - a fact - not because
of any claim about its subject matter.

``term_absent_from_corpus`` terms are restricted to subject matter plainly outside
the five indexed standards: other standards' territory (zakah, waqf, tawarruq),
financial-accounting territory (deferred tax), prudential regulation (capital
adequacy) and anachronisms (cryptocurrency, blockchain, ESG). Terms that are
foundational to all five standards were considered and rejected even at 0
occurrences - ``riba`` occurs in 0 chunks, but a reviewer could fairly argue the
concept is addressed under other wording, which would make the label wrong.

Run: ``python scripts/build_unanswerable_probes.py``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aaoifi_rag.reporting.probes import (  # noqa: E402
    Probe,
    ProbeVerificationError,
    UnanswerableBasis,
    corpus_clause_ids,
    corpus_standards,
    term_occurrences,
    verify_probes,
    write_probes,
)

CLAUSE_CHUNKS = REPO_ROOT / "data" / "private" / "extracted" / "clause_chunks.jsonl"
MANIFEST = REPO_ROOT / "data" / "manifests" / "source_manifest.json"
OUT_PATH = REPO_ROOT / "data" / "probes" / "unanswerable_probes.jsonl"

#: (term, question). Each term is verified to occur in 0 chunks at build time.
TERM_PROBES: tuple[tuple[str, str], ...] = (
    ("zakat", "How should zakat be calculated on assets held under this framework?"),
    ("waqf", "What are the requirements for establishing a waqf endowment?"),
    (
        "tawarruq",
        "Is organised tawarruq permitted as a liquidity-management instrument?",
    ),
    (
        "cryptocurrency",
        "May a cryptocurrency be used as the subject matter of the contract?",
    ),
    (
        "blockchain",
        "What controls apply when contract records are maintained on a blockchain?",
    ),
    (
        "deferred tax",
        "How is deferred tax recognised and measured for these transactions?",
    ),
    (
        "capital adequacy",
        "What capital adequacy ratio must an institution maintain against this exposure?",
    ),
    ("esg", "What ESG disclosures are required before the contract is concluded?"),
)

#: (standard_id, display number, question). Verified not indexed at build time.
#: The questions name the standard by NUMBER ONLY - see the module docstring on
#: why no subject matter is asserted for a standard the project does not hold.
STANDARD_PROBES: tuple[tuple[str, str, str], ...] = (
    (
        "SS11",
        "No. 11",
        "What does AAOIFI Shari'ah Standard No. 11 require of the two parties "
        "before the contract is signed?",
    ),
    (
        "SS12",
        "No. 12",
        "Summarise the permissibility conditions set out in AAOIFI Shari'ah "
        "Standard No. 12.",
    ),
    (
        "SS21",
        "No. 21",
        "Under AAOIFI Shari'ah Standard No. 21, what disclosures must the "
        "institution make to the client?",
    ),
    (
        "SS30",
        "No. 30",
        "Which clause of AAOIFI Shari'ah Standard No. 30 governs early "
        "termination, and what does it say?",
    ),
    (
        "SS41",
        "No. 41",
        "What does clause 4/1 of AAOIFI Shari'ah Standard No. 41 provide?",
    ),
    (
        "SS59",
        "No. 59",
        "Does AAOIFI Shari'ah Standard No. 59 permit the arrangement described "
        "in the preceding paragraph?",
    ),
)

#: (standard_id, clause_id, question). Each clause id is verified ABSENT from a
#: standard that IS indexed, so the failure mode under test is a plausible-looking
#: citation request rather than an unknown standard. Evidence carries only the two
#: fields ``verify_probe`` re-derives; a count that nothing re-checks would be
#: exactly the stale label this design is meant to prevent.
CLAUSE_PROBES: tuple[tuple[str, str, str], ...] = (
    (
        "SS8",
        "7/3",
        "What does clause 7/3 of the Murabahah standard require?",
    ),
    (
        "SS9",
        "12/1",
        "Quote clause 12/1 of the Ijarah and Ijarah Muntahia Bittamleek standard.",
    ),
    (
        "SS13",
        "14/2",
        "Under clause 14/2 of the Mudarabah standard, how are losses allocated?",
    ),
    (
        "SS17",
        "9/2",
        "What conditions does clause 9/2 of the Investment Sukuk standard impose "
        "on the issuer?",
    ),
    (
        "SS26",
        "20/5",
        "What does clause 20/5 of the Islamic Insurance standard say about "
        "surplus distribution?",
    ),
)

#: (standard_ids, question). The stipulated escalation class. Every named standard
#: must itself be indexed - otherwise the probe is merely unanswerable and the
#: escalation label would be measuring the wrong thing.
CROSS_PROBES: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("SS8", "SS9"),
        "Compare how the Murabahah standard and the Ijarah standard each treat "
        "the institution's ownership of the asset before it passes to the client.",
    ),
    (
        ("SS9", "SS17"),
        "An Ijarah-based sukuk is being structured. Reconcile the Ijarah "
        "standard's requirements on the leased asset with the Investment Sukuk "
        "standard's requirements on the underlying assets.",
    ),
    (
        ("SS13", "SS17"),
        "Where a sukuk issue is structured on a Mudarabah basis, how do the "
        "Mudarabah standard's profit-sharing rules interact with the Investment "
        "Sukuk standard's rules on returns to certificate holders?",
    ),
    (
        ("SS8", "SS13"),
        "Which is more appropriate for financing working capital under the "
        "standards indexed here - Murabahah or Mudarabah - and on what grounds "
        "in each standard?",
    ),
    (
        ("SS17", "SS26"),
        "May an Islamic insurance fund invest its reserves in investment sukuk, "
        "and which conditions from each standard apply?",
    ),
    (
        ("SS9", "SS13", "SS26"),
        "An Islamic insurance operator wants to lease its head office and invest "
        "policyholder surplus through a Mudarabah. Which provisions across the "
        "relevant standards constrain this?",
    ),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(
            f"missing required input: {path.relative_to(REPO_ROOT).as_posix()}\n"
            "Probe evidence is computed from the live corpus, so this script "
            "cannot run without data/private/. See data/README.md. Refusing to "
            "write a probe file whose counts would be guesses.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def build_probes(chunks: list[dict[str, Any]]) -> list[Probe]:
    """Author every probe, deriving each evidence figure from ``chunks``."""
    probes: list[Probe] = []
    n_chunks = len(chunks)
    indexed = sorted(corpus_standards(chunks))

    for index, (term, question) in enumerate(TERM_PROBES, start=1):
        probes.append(
            Probe(
                probe_id=f"U-TERM-{index:02d}",
                question_text=question,
                basis=UnanswerableBasis.TERM_ABSENT,
                basis_evidence={
                    "term": term,
                    "corpus_occurrences": term_occurrences(term, chunks),
                    "of_chunks": n_chunks,
                    "match_rule": "whole word, after normalise_for_match on both sides",
                },
                notes=(
                    f"{term!r} occurs in 0 of {n_chunks} clause chunks. Subject "
                    "matter lies outside the five indexed standards; the label is "
                    "corpus absence, not a Shari'ah judgement."
                ),
            )
        )

    for index, (standard_id, display, question) in enumerate(
        STANDARD_PROBES, start=1
    ):
        probes.append(
            Probe(
                probe_id=f"U-STD-{index:02d}",
                question_text=question,
                basis=UnanswerableBasis.STANDARD_ABSENT,
                basis_evidence={
                    "named_standard": standard_id,
                    "indexed_standards": indexed,
                },
                notes=(
                    f"AAOIFI Shari'ah Standard {display} is not indexed; the index "
                    f"holds {', '.join(indexed)} only. The question names the "
                    "standard by number and makes no claim about its subject "
                    "matter, which this project cannot verify from its own files."
                ),
            )
        )

    for index, (standard_id, clause_id, question) in enumerate(
        CLAUSE_PROBES, start=1
    ):
        probes.append(
            Probe(
                probe_id=f"U-CLAUSE-{index:02d}",
                question_text=question,
                basis=UnanswerableBasis.CLAUSE_ABSENT,
                basis_evidence={
                    "standard_id": standard_id,
                    "clause_id": clause_id,
                },
                notes=(
                    f"{standard_id} is indexed but has no clause {clause_id}. A "
                    "system that answers this has invented a citation inside a "
                    "standard it genuinely holds - the sharpest failure mode in "
                    "the whole negative class."
                ),
            )
        )

    for index, (standard_ids, question) in enumerate(CROSS_PROBES, start=1):
        probes.append(
            Probe(
                probe_id=f"E-CROSS-{index:02d}",
                question_text=question,
                basis=UnanswerableBasis.CROSS_STANDARD,
                basis_evidence={
                    "standard_ids": list(standard_ids),
                    "n_standards_required": len(standard_ids),
                },
                notes=(
                    "STIPULATED escalation, not a scholar judgement that human "
                    f"review is required. Requires {len(standard_ids)} indexed "
                    f"standards ({', '.join(standard_ids)}); every clause chunk "
                    "belongs to exactly one standard, so no single retrieved "
                    "clause can settle it."
                ),
            )
        )

    return probes


def summarise(probes: list[Probe], chunks: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for probe in probes:
        counts[probe.basis.value] = counts.get(probe.basis.value, 0) + 1
    print(f"corpus: {len(chunks)} chunks, standards {sorted(corpus_standards(chunks))}")
    print(f"built {len(probes)} probes")
    for basis, count in sorted(counts.items()):
        behaviour = next(p.expected_behavior for p in probes if p.basis.value == basis)
        print(f"  {basis:<32} {count:>3}  -> {behaviour}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Verify only: rebuild in memory, verify against the corpus and diff "
            "against the file on disk. Writes nothing. Exit 1 if they differ."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    chunks = read_jsonl(CLAUSE_CHUNKS)
    probes = build_probes(chunks)
    summarise(probes, chunks)

    try:
        verify_probes(probes, chunks)
    except ProbeVerificationError as error:
        print(f"\nVERIFICATION FAILED - nothing written.\n{error}", file=sys.stderr)
        return 1
    print("every basis_evidence claim re-derived from the live corpus: confirmed")

    rendered = [
        json.dumps(probe.as_dict(), ensure_ascii=False) for probe in probes
    ]
    if args.check:
        if not args.out.exists():
            print(f"\n--check: {args.out} does not exist.", file=sys.stderr)
            return 1
        on_disk = [
            line.strip()
            for line in args.out.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if on_disk != rendered:
            print(
                f"\n--check: {args.out.name} differs from a fresh build; re-run "
                "without --check to regenerate.",
                file=sys.stderr,
            )
            return 1
        print(f"--check: {args.out.name} matches a fresh build ({len(rendered)} lines)")
        return 0

    written = write_probes(probes, args.out)
    print(f"wrote {len(probes)} probes -> {written.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
