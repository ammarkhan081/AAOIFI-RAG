"""Unanswerable and escalation probes: a mechanically labelled negative class.

Why this module exists
----------------------
``data/private/hard_set.jsonl`` has seven items and every one of them is
``answerability = answerable`` with ``expected_behavior = answer``. On that set an
abstention-aware system cannot be evaluated at all: every abstention is
over-abstention by construction, and abstention precision/recall and escalation
precision/recall have no positives, so they are undefined rather than small.
Fixing that by authoring more answerable items needs qualified scholars.

Fixing the *negative* class does not. A probe's label here is a **fact about the
corpus**, not a judgement about Shari'ah:

``term_absent_from_corpus``
    The question turns on a term that occurs in 0 of the corpus chunks. Verified
    by :func:`term_occurrences` against the live corpus, not by assertion.
``standard_absent_from_corpus``
    The question names an AAOIFI standard whose clauses are not in the index at
    all. Verified by set membership against :func:`corpus_standards`.
``clause_absent_from_standard``
    The question names a clause id that does not exist in a standard that *is*
    indexed. Verified against :func:`corpus_clause_ids`.
``cross_standard_comparison``
    The question requires provisions from two or more indexed standards. Every
    chunk carries exactly one ``standard_id`` (asserted by
    :func:`assert_chunks_are_single_standard`), so no single clause can settle
    such a question. This is the **stipulated** escalation class - see
    :data:`ESCALATION_STIPULATION`.

What this buys and what it does not
-----------------------------------
It buys measurable abstention precision/recall and a true-negative rate, computed
on constructed inputs. It does **not** establish that the router behaves correctly
on natural user queries, and it is not a substitute for a scholar-validated
answerable set. A probe is an adversarial control, not a sample from a population.

Three of the four bases are facts. The fourth is a stipulation and is labelled as
one everywhere it appears, so a reader can reject the definition without the rest
of the evaluation collapsing.

This module contains no AAOIFI clause prose. The probe questions are authored by
this project and are safe to publish, which is why ``data/probes/`` is tracked
while ``data/private/`` is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from ..reliability.text_checks import normalise_for_match

#: Bump when the probe field set changes.
PROBE_SCHEMA_VERSION = "unanswerable_probe_v1"

#: Attached to every ``cross_standard_comparison`` probe. Repeated in full so a
#: stray probe file still carries its own caveat.
ESCALATION_STIPULATION = {
    "stipulation_id": "cross_standard_comparison_v1",
    "definition": (
        "A question that requires provisions from two or more indexed standards "
        "is escalate-worthy, because every clause chunk belongs to exactly one "
        "standard and so no single retrieved clause can settle the question."
    ),
    "warning": (
        "This is a STIPULATED definition, not a qualified scholar judgement that "
        "human review is required. Escalation precision and recall computed "
        "against it are conditional on accepting the definition. Reject the "
        "definition and the escalation numbers go away; nothing else does."
    ),
    "verification_basis": "stipulated_definition",
}


class UnanswerableBasis(StrEnum):
    """How a probe's label is established. Three facts and one stipulation."""

    TERM_ABSENT = "term_absent_from_corpus"
    STANDARD_ABSENT = "standard_absent_from_corpus"
    CLAUSE_ABSENT = "clause_absent_from_standard"
    CROSS_STANDARD = "cross_standard_comparison"


#: The stipulated basis routes to escalate; the factual ones route to abstain.
EXPECTED_BEHAVIOR_BY_BASIS = {
    UnanswerableBasis.TERM_ABSENT: "abstain",
    UnanswerableBasis.STANDARD_ABSENT: "abstain",
    UnanswerableBasis.CLAUSE_ABSENT: "abstain",
    UnanswerableBasis.CROSS_STANDARD: "escalate",
}

#: Verification basis recorded on each probe, by how the label was established.
VERIFICATION_BASIS_BY_BASIS = {
    UnanswerableBasis.TERM_ABSENT: "mechanical_corpus_absence",
    UnanswerableBasis.STANDARD_ABSENT: "mechanical_corpus_absence",
    UnanswerableBasis.CLAUSE_ABSENT: "mechanical_corpus_absence",
    UnanswerableBasis.CROSS_STANDARD: "stipulated_definition",
}


class ProbeVerificationError(ValueError):
    """Raised when a probe's stored evidence disagrees with the live corpus.

    This is deliberately an error and not a warning. A probe whose justification
    has gone stale - because the corpus grew, or a term was mis-transcribed - is
    a wrong label, and a wrong label in the negative class silently inflates
    abstention precision.
    """


def _normalised_texts(chunks: Iterable[Mapping[str, Any]]) -> list[str]:
    return [normalise_for_match(chunk.get("text") or "") for chunk in chunks]


def term_occurrences(term: str, chunks: Sequence[Mapping[str, Any]]) -> int:
    """Number of chunks whose text contains ``term`` as a whole word.

    Both sides go through :func:`~aaoifi_rag.reliability.text_checks.normalise_for_match`,
    so case, Unicode form and *typographic variants* are unified the same way the
    citation audit unifies them: an en-dash in the corpus matches a plain hyphen in the
    probe term, and a curly apostrophe matches an ASCII one.

    Two properties this deliberately does **not** have, because every
    ``term_absent_from_corpus`` label is exactly this function and the label has to mean
    what it says:

    * Punctuation is not *stripped*, only normalised. A probe term must be spelled the
      way the corpus spells it.
    * There is no stemming. ``tawarruq`` does not match ``tawarruqs``; a probe claiming
      absence of one makes no claim about the other. Since ``\\w`` excludes ``-``, a
      hyphen acts as a word boundary, so ``inah`` matches inside ``al-inah``.

    Counting *chunks* rather than occurrences keeps the number stable under re-chunking
    of a single clause. ``tests/test_probes.py`` pins all of the above.
    """
    needle = normalise_for_match(term)
    if not needle:
        raise ValueError("term is empty after normalisation")
    pattern = re.compile(rf"(?<!\w){re.escape(needle)}(?!\w)")
    return sum(1 for text in _normalised_texts(chunks) if pattern.search(text))


def corpus_standards(chunks: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(chunk["standard_id"]) for chunk in chunks}


def corpus_clause_ids(
    chunks: Sequence[Mapping[str, Any]], standard_id: str
) -> set[str]:
    return {
        str(chunk["clause_id"])
        for chunk in chunks
        if str(chunk["standard_id"]) == standard_id
    }


def assert_chunks_are_single_standard(chunks: Sequence[Mapping[str, Any]]) -> None:
    """The structural fact the escalation stipulation rests on.

    If a chunk ever spanned two standards, ``cross_standard_comparison`` would no
    longer imply "no single clause settles it", and the stipulation would be
    unsound. Checked rather than assumed.
    """
    for chunk in chunks:
        standard = chunk.get("standard_id")
        if not isinstance(standard, str) or not standard:
            raise ProbeVerificationError(
                f"chunk {chunk.get('chunk_id')!r} has no single standard_id; the "
                "cross-standard escalation stipulation is unsound on this corpus"
            )


@dataclass(frozen=True)
class Probe:
    """One negative-class item and the evidence for its label."""

    probe_id: str
    question_text: str
    basis: UnanswerableBasis
    #: Re-derivable justification. Shape depends on ``basis``; see :func:`verify_probe`.
    basis_evidence: dict[str, Any]
    notes: str = ""
    schema_version: str = PROBE_SCHEMA_VERSION

    @property
    def expected_behavior(self) -> str:
        return EXPECTED_BEHAVIOR_BY_BASIS[self.basis]

    @property
    def verification_basis(self) -> str:
        return VERIFICATION_BASIS_BY_BASIS[self.basis]

    @property
    def requires_scholar(self) -> bool:
        """False for every probe. That is the whole point of the negative class."""
        return False

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "probe_id": self.probe_id,
            "question_text": self.question_text,
            "expected_behavior": self.expected_behavior,
            "unanswerable_basis": self.basis.value,
            "basis_evidence": dict(self.basis_evidence),
            "verification_basis": self.verification_basis,
            "requires_scholar": self.requires_scholar,
            "notes": self.notes,
        }
        if self.basis is UnanswerableBasis.CROSS_STANDARD:
            payload["stipulation"] = dict(ESCALATION_STIPULATION)
        return payload

    def as_evaluation_item(self) -> dict[str, Any]:
        """Shaped like a hard-set item so probes flow through the same evaluation path.

        ``gold_clause_ids`` is empty on purpose: an unanswerable probe has no gold
        clause, so its per-item recall is undefined, and
        :class:`~aaoifi_rag.reporting.evaluation.RetrievalScore` must not be asked
        to average a 0/0 into the macro mean. Callers building labels for probes
        should pass ``context_records=None``.
        """
        return {
            "item_id": self.probe_id,
            "question_text": self.question_text,
            "standard_ids": list(self.basis_evidence.get("standard_ids", [])),
            "gold_clause_ids": [],
            "gold_answer": None,
            "status": "generated",
            "verification_basis": self.verification_basis,
            "answerability": (
                "requires_multiple_standards"
                if self.basis is UnanswerableBasis.CROSS_STANDARD
                else "unanswerable_from_corpus"
            ),
            "expected_behavior": self.expected_behavior,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Probe":
        return cls(
            probe_id=str(payload["probe_id"]),
            question_text=str(payload["question_text"]),
            basis=UnanswerableBasis(str(payload["unanswerable_basis"])),
            basis_evidence=dict(payload.get("basis_evidence") or {}),
            notes=str(payload.get("notes") or ""),
            schema_version=str(payload.get("schema_version", PROBE_SCHEMA_VERSION)),
        )


def verify_probe(
    probe: Probe, chunks: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Re-derive ``probe.basis_evidence`` from ``chunks``. Returns failure strings.

    An empty list means every stored number was reproduced. Nothing here trusts
    the file: the counts in ``data/probes/unanswerable_probes.jsonl`` are treated
    as claims to be checked, which is what makes the negative class auditable by
    someone who holds the corpus.
    """
    failures: list[str] = []
    evidence = probe.basis_evidence
    standards = corpus_standards(chunks)

    if probe.basis is UnanswerableBasis.TERM_ABSENT:
        term = evidence.get("term")
        if not term:
            return [f"{probe.probe_id}: term_absent probe has no 'term' in evidence"]
        actual = term_occurrences(str(term), chunks)
        if actual != 0:
            failures.append(
                f"{probe.probe_id}: term {term!r} now occurs in {actual} chunk(s); "
                "the probe is no longer unanswerable on this corpus"
            )
        claimed = evidence.get("corpus_occurrences")
        if claimed is not None and int(claimed) != actual:
            failures.append(
                f"{probe.probe_id}: stored corpus_occurrences={claimed} but "
                f"recomputed {actual}"
            )
        if int(evidence.get("of_chunks", len(chunks))) != len(chunks):
            failures.append(
                f"{probe.probe_id}: stored of_chunks="
                f"{evidence.get('of_chunks')} but corpus has {len(chunks)}"
            )

    elif probe.basis is UnanswerableBasis.STANDARD_ABSENT:
        named = str(evidence.get("named_standard", ""))
        if not named:
            failures.append(f"{probe.probe_id}: no 'named_standard' in evidence")
        elif named in standards:
            failures.append(
                f"{probe.probe_id}: standard {named} IS indexed "
                f"({sorted(standards)}); probe is answerable"
            )

    elif probe.basis is UnanswerableBasis.CLAUSE_ABSENT:
        standard = str(evidence.get("standard_id", ""))
        clause = str(evidence.get("clause_id", ""))
        if standard not in standards:
            failures.append(
                f"{probe.probe_id}: clause_absent probe names standard {standard} "
                "which is not indexed; use standard_absent_from_corpus instead"
            )
        elif clause in corpus_clause_ids(chunks, standard):
            failures.append(
                f"{probe.probe_id}: clause {standard}:{clause} EXISTS; "
                "probe is answerable"
            )

    elif probe.basis is UnanswerableBasis.CROSS_STANDARD:
        named = [str(value) for value in evidence.get("standard_ids", [])]
        if len(set(named)) < 2:
            failures.append(
                f"{probe.probe_id}: cross-standard probe names {named}; "
                "the stipulation needs at least two distinct standards"
            )
        missing = [value for value in named if value not in standards]
        if missing:
            failures.append(
                f"{probe.probe_id}: names non-indexed standard(s) {missing}; a "
                "cross-standard escalation probe must span standards that ARE "
                "indexed, or it is simply unanswerable instead"
            )

    return failures


def verify_probes(
    probes: Sequence[Probe], chunks: Sequence[Mapping[str, Any]]
) -> None:
    """Verify every probe, plus the structural fact the stipulation rests on."""
    assert_chunks_are_single_standard(chunks)
    failures = [failure for probe in probes for failure in verify_probe(probe, chunks)]
    seen: dict[str, int] = {}
    for probe in probes:
        seen[probe.probe_id] = seen.get(probe.probe_id, 0) + 1
    duplicates = sorted(key for key, count in seen.items() if count > 1)
    if duplicates:
        failures.append(f"duplicate probe_id(s): {', '.join(duplicates)}")
    if failures:
        raise ProbeVerificationError(
            f"{len(failures)} probe verification failure(s):\n  - "
            + "\n  - ".join(failures)
        )


def load_probes(path: Path | str) -> list[Probe]:
    """Read a probe JSONL file. Does not verify; call :func:`verify_probes`."""
    with Path(path).open("r", encoding="utf-8") as stream:
        return [Probe.from_dict(json.loads(line)) for line in stream if line.strip()]


def write_probes(probes: Sequence[Probe], path: Path | str) -> Path:
    """Write probes as JSONL. Safe to commit: authored questions, no clause text."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for probe in probes:
            stream.write(json.dumps(probe.as_dict(), ensure_ascii=False))
            stream.write("\n")
    return target
