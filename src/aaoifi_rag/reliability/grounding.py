"""Clause-reference grounding: did the answer cite a clause it was actually shown?

Why this exists as its own module
---------------------------------
``reports/anchoring_gate_calibration.md`` §12 rejected repairing the
retrieval-stage anchoring measure so that it could see clause numbers, and
committed to a specific alternative:

    The correct home for "did the model cite a clause it was actually shown?" is
    a response-stage grounding check on the *answer's* citations, where the
    reference set is explicit and the false-positive rate is directly
    measurable - not a repair to a bag-of-stems retrieval-stage measure. That is
    implemented separately and is not this gate.

This module is that check. It is deliberately *not* a predictive signal. Given
the context records, "does ``SS8 §2/4/1`` correspond to any clause the model was
shown?" is a decidable question with no judgement in it, in the same way that
:mod:`aaoifi_rag.reliability.citations` decides whether ``[9]`` exists in a
five-record context. There is no threshold here, and nothing to calibrate.

What is uncertain is the *extractor*, not the rule - so the extractor was
measured before it was written.

Measured behaviour of the extractor (see ``reports/clause_grounding_gate.md``)
-----------------------------------------------------------------------------
Over all 362 corpus clause texts:

* 36 references extracted, and all 36 are genuine cross-references
  ("with due consideration to item 2/2/3", "[see items 5/2/2, 7/2/1, and
  7/2/2]"). Zero ratios, fractions, dates or percentages were mis-extracted.
  This matters because SS13 is Mudarabah, whose profit-sharing ratios are
  written in the same ``x/y`` shape as a clause path.
* Zero slash-numerals fall outside every extracted span, so nothing in that
  shape is missed relative to a bare ``\\d+(?:/\\d+)+`` baseline.
* All 36 carry **no** standard token. AAOIFI's own idiom is unqualified
  (``item 3/1/5``, ``[see para. 7/1]``), so :func:`_bind_standard`'s
  unqualified path is load-bearing, not a convenience.

Deliberate limitations, each with its measured cost
--------------------------------------------------
* **Depth-1 references are not extracted** (``MIN_PATH_DEPTH``). ``item 8``
  cannot be separated from "8 years" without judgement. Measured cost on this
  corpus: 0 occurrences in clause prose and 0 in the seven gold answers.
* **Cross-standard prose references are bound, not resolved.** Seven corpus
  clauses say "item 3/1/4/3 of Shari'ah Standard No. (12)"-style, naming
  standards 5, 9, 12 and 13. :data:`STANDARD_NUMBER_RE` binds those forward, so
  such a reference is reported :attr:`ReferenceStatus.UNKNOWN_STANDARD` instead
  of being resolved against whatever standards happen to be retrieved. This
  *does* change the verdict, in the strict direction and correctly: ``3/1/4/3``
  is an SS17-shaped path, so without the forward binding a reference to SS12's
  ``3/1/4/3`` would be grounded against a retrieved SS17 clause it has nothing to
  do with. The forward binding prevents that false negative.

  When a *retrieved* clause itself quotes another standard's item, the path is
  genuinely in front of the model and the context checks - which run first -
  ground it. Both cases are pinned in ``tests/test_grounding.py``.
* **An unqualified reference is resolved against every retrieved standard.**
  This is lenient on purpose: it can only turn a would-be failure into a pass,
  never the reverse, so the gate fires only when it is certain.

The corpus index refines the diagnosis and never the decision
-------------------------------------------------------------
Grounding is decided from the context block alone, so
:attr:`ClauseGroundingAudit.passed` never depends on :class:`CorpusClauseIndex`.
The index only subdivides a failure: into "a real clause that was not retrieved"
(:attr:`ReferenceStatus.OUTSIDE_CONTEXT`), "a standard this corpus does not hold"
(:attr:`ReferenceStatus.UNKNOWN_STANDARD`) and "no such path in the standards it
could belong to" (:attr:`ReferenceStatus.UNRESOLVABLE`). All three gate. Omit the
index to run the gate without loading the corpus and get the same decisions with
a coarser explanation.

``UNRESOLVABLE`` is **not** a fabrication verdict. The corpus holds five
standards; a genuine reference to ``SS12 §3/1`` is unresolvable here and invented
nowhere. Only a qualified reviewer can call something fabricated, and this module
does not.

This module contains no AAOIFI clause prose. :class:`CorpusClauseIndex` holds
clause *identifiers* only, which tracked reports already publish.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Any, Iterable, Mapping, Sequence

#: Shallowest clause path the extractor will take. Depth 1 (``item 8``) is
#: indistinguishable from an ordinary number without judgement; measured cost of
#: excluding it on this corpus is zero occurrences.
MIN_PATH_DEPTH = 2

#: ``SS8``, ``SS 26``. The project's own idiom, used in the gold answers and in
#: the provenance header the prompt prints.
STANDARD_TOKEN_RE = re.compile(r"\bSS\s?(\d{1,2})\b", re.IGNORECASE)

#: AAOIFI's prose idiom for another standard: "of Shari'ah Standard No. (12)".
#: The apostrophe varies (``'``/``’``), so it is matched loosely.
STANDARD_NUMBER_RE = re.compile(
    r"Shari.?ah\s+Standard\s+No\.?\s*\(?(\d{1,2})\)?", re.IGNORECASE
)

#: Words AAOIFI uses to introduce a clause path. Optional: the section sign and a
#: bare path both occur.
CUE_PATTERN = (
    r"(?:§|\bitems?\b|\bclauses?\b|\bparagraphs?\b|\bparas?\b\.?|\bsections?\b)"
)

#: A clause path, optionally preceded by a standard token and/or a cue word.
CLAUSE_REFERENCE_RE = re.compile(
    rf"(?:(?P<std>SS\s?\d{{1,2}})\s*)?"
    rf"(?:{CUE_PATTERN}\s*(?:no\.?\s*)?)?"
    rf"(?P<path>\d+(?:/\d+){{1,3}})\b",
    re.IGNORECASE,
)

#: How far back to look for a standard token when the reference is unqualified.
STANDARD_LOOKBACK = 60

#: How far forward to look for "of Shari'ah Standard No. (N)".
STANDARD_LOOKAHEAD = 48


class ReferenceStatus(StrEnum):
    """Outcome of resolving one extracted clause reference."""

    #: Resolves to a clause in the context block, or its path appears verbatim in
    #: the retrieved clause text. Never a failure.
    GROUNDED_IN_CONTEXT = "grounded_in_context"
    #: Resolves to a real corpus clause that was not retrieved. The answer used
    #: material outside its context, which the prompt contract forbids.
    OUTSIDE_CONTEXT = "outside_context"
    #: Names a standard the corpus does not hold, so whether the clause is real
    #: cannot be checked here. It was still not in the context, so it still gates.
    UNKNOWN_STANDARD = "unknown_standard"
    #: No clause path match in any standard it could belong to. Not a fabrication
    #: verdict - see the module docstring.
    UNRESOLVABLE = "unresolvable"


#: Every status other than :attr:`ReferenceStatus.GROUNDED_IN_CONTEXT`.
#:
#: The gate's question is the one the calibration report set: *was this clause
#: shown to the model?* That is decidable from the context block alone. Whether
#: the clause exists in AAOIFI at all is a different question, it is the only one
#: that needs :class:`CorpusClauseIndex`, and it is not what gates - so the three
#: non-grounded statuses all fail, and they differ only in how precisely the
#: failure can be described.
#:
#: An earlier draft excused ``UNKNOWN_STANDARD`` as "unadjudicable". That
#: conflated the two questions: a reference to ``SS12 §1/1`` in a context of SS8
#: clauses demonstrably was not shown, whatever its status inside AAOIFI. The
#: mistake was caught by ``tests/test_grounding.py``'s index-independence test,
#: which no real text in this project exercises.
FAILING_STATUSES = frozenset(
    {
        ReferenceStatus.OUTSIDE_CONTEXT,
        ReferenceStatus.UNKNOWN_STANDARD,
        ReferenceStatus.UNRESOLVABLE,
    }
)


def clause_ancestors(clause_path: str) -> frozenset[str]:
    """``"2/4/4"`` -> ``{"2", "2/4", "2/4/4"}``.

    Used both ways: a reference to ``2/4`` is grounded by a retrieved ``2/4/4``
    (the parent was shown, in the sense that its child was), and the index stores
    ancestors so a shallow reference resolves against a deep corpus clause.
    """
    parts = [part for part in clause_path.split("/") if part]
    return frozenset("/".join(parts[:index]) for index in range(1, len(parts) + 1))


def normalise_standard(token: str | None) -> str | None:
    """``"SS 8"`` / ``"ss8"`` -> ``"SS8"``; a bare number -> ``"SS12"``."""
    if token is None:
        return None
    digits = re.sub(r"\D", "", token)
    return f"SS{int(digits)}" if digits else None


@dataclass(frozen=True)
class CorpusClauseIndex:
    """Clause identities present in the corpus, per standard.

    Identifiers only - no clause text - so this is safe to build, log and
    publish. :meth:`from_records` accepts the same record shape the rest of the
    pipeline uses, and ignores any ``text`` field.
    """

    #: Standard id -> every clause path in it, plus all their ancestors.
    paths_by_standard: Mapping[str, frozenset[str]]

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]]) -> "CorpusClauseIndex":
        collected: dict[str, set[str]] = {}
        for record in records:
            standard = record.get("standard_id")
            clause_path = record.get("clause_id")
            if not standard or not clause_path:
                continue
            collected.setdefault(str(standard), set()).update(
                clause_ancestors(str(clause_path))
            )
        return cls(
            paths_by_standard={
                standard: frozenset(paths) for standard, paths in collected.items()
            }
        )

    @property
    def standards(self) -> frozenset[str]:
        return frozenset(self.paths_by_standard)

    def contains(self, standard_id: str, clause_path: str) -> bool:
        return clause_path in self.paths_by_standard.get(standard_id, frozenset())

    def as_dict(self) -> dict[str, Any]:
        """Sizes and standard ids, not the full path lists."""
        return {
            "standards": sorted(self.standards),
            "n_paths": {
                standard: len(paths)
                for standard, paths in sorted(self.paths_by_standard.items())
            },
        }


@dataclass(frozen=True)
class ClauseReference:
    """One clause reference found in a response, and how it resolved.

    Carries the reference's *identifiers* and character offset. No surrounding
    prose is retained, so this is safe to serialise into a public trace.
    """

    clause_path: str
    standard_id: str | None
    #: True when a standard token was found for this reference, by either binding
    #: rule. False means it was resolved against every candidate standard.
    qualified: bool
    char_offset: int
    status: ReferenceStatus
    #: Which rule settled it: ``exact``, ``ancestor``, ``verbatim_in_context``,
    #: ``corpus_exact``, ``corpus_ancestor``, ``standard_not_in_corpus``, or
    #: ``no_match``.
    resolved_via: str
    #: Standards the reference was tried against when unqualified.
    candidate_standards: tuple[str, ...] = ()

    @property
    def failed(self) -> bool:
        return self.status in FAILING_STATUSES

    def as_dict(self) -> dict[str, Any]:
        return {
            "clause_path": self.clause_path,
            "standard_id": self.standard_id,
            "qualified": self.qualified,
            "char_offset": self.char_offset,
            "status": self.status.value,
            "resolved_via": self.resolved_via,
            "candidate_standards": list(self.candidate_standards),
        }


@dataclass(frozen=True)
class ClauseGroundingAudit:
    """Every clause reference in one response, resolved against its context."""

    references: tuple[ClauseReference, ...]
    context_standards: tuple[str, ...]
    #: Whether a :class:`CorpusClauseIndex` was available. Affects the reported
    #: status split only; :attr:`passed` is identical either way.
    corpus_index_used: bool
    min_path_depth: int = MIN_PATH_DEPTH

    @property
    def failing_references(self) -> tuple[ClauseReference, ...]:
        return tuple(reference for reference in self.references if reference.failed)

    @property
    def passed(self) -> bool:
        """No reference certainly points outside the context block."""
        return not self.failing_references

    @property
    def distinct_reference_count(self) -> int:
        """Deduplicated by resolved identity.

        The stored H01 response contains seven slash-numerals but only two
        distinct references: it transcribes the provenance header as well as
        making the claim in prose. Counting raw occurrences would report an
        answer as more heavily referenced than it is.
        """
        return len(
            {(reference.standard_id, reference.clause_path) for reference in self.references}
        )

    def counts(self) -> dict[str, int]:
        tally = {status.value: 0 for status in ReferenceStatus}
        for reference in self.references:
            tally[reference.status.value] += 1
        return tally

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_references": len(self.references),
            "n_distinct_references": self.distinct_reference_count,
            "context_standards": list(self.context_standards),
            "corpus_index_used": self.corpus_index_used,
            "min_path_depth": self.min_path_depth,
            "passed": self.passed,
            "status_counts": self.counts(),
            "references": [reference.as_dict() for reference in self.references],
        }


def _bind_standard(text: str, match: re.Match[str]) -> tuple[str | None, bool]:
    """Attach a standard to one reference match. Returns ``(standard, qualified)``.

    Three rules, in precedence order:

    1. A standard token immediately before the path (``SS8 §2/4/1``) - the
       project's own idiom and the gold answers'.
    2. "of Shari'ah Standard No. (N)" immediately after - AAOIFI's idiom for a
       *different* standard, 7 occurrences in this corpus.
    3. The nearest preceding standard token within :data:`STANDARD_LOOKBACK`
       characters, which carries an ``SS8 §2/4/4 ... §2/4/2`` run.

    Rule 2 is checked before rule 3 because a forward "Shari'ah Standard No. (12)"
    is a stronger signal than an ``SS17`` mentioned a sentence earlier.
    """
    if match.group("std"):
        return normalise_standard(match.group("std")), True

    path_end = match.end("path")
    ahead = text[path_end : path_end + STANDARD_LOOKAHEAD]
    forward = STANDARD_NUMBER_RE.search(ahead)
    if forward is not None:
        return normalise_standard(forward.group(1)), True

    behind = text[max(0, match.start() - STANDARD_LOOKBACK) : match.start()]
    previous = list(STANDARD_TOKEN_RE.finditer(behind))
    if previous:
        return normalise_standard(previous[-1].group(0)), True
    return None, False


def extract_clause_references(
    text: str, *, min_path_depth: int = MIN_PATH_DEPTH
) -> list[tuple[str, str | None, bool, int]]:
    """Find clause references in ``text``.

    Returns ``(clause_path, standard_id, qualified, char_offset)`` tuples in
    document order, including repeats - deduplication is the audit's job, because
    a repeat at a different offset is still evidence about the response.
    """
    found: list[tuple[str, str | None, bool, int]] = []
    for match in CLAUSE_REFERENCE_RE.finditer(text or ""):
        clause_path = match.group("path")
        if len(clause_path.split("/")) < min_path_depth:
            continue
        standard, qualified = _bind_standard(text, match)
        found.append((clause_path, standard, qualified, match.start("path")))
    return found


def _resolve(
    clause_path: str,
    standard_id: str | None,
    context_by_standard: Mapping[str, frozenset[str]],
    context_blob: str,
    corpus_index: CorpusClauseIndex | None,
) -> tuple[ReferenceStatus, str, tuple[str, ...]]:
    """Resolve one reference. Order is fixed and each step is documented.

    Context checks run first and unconditionally, including for a reference that
    names a standard outside the corpus: a retrieved clause may itself quote
    "item 6/4 of Shari'ah Standard No. (5)", and that path really is in front of
    the model.
    """
    candidates = (
        (standard_id,)
        if standard_id is not None
        else tuple(sorted(context_by_standard))
    )

    for candidate in candidates:
        if clause_path in context_by_standard.get(candidate or "", frozenset()):
            return ReferenceStatus.GROUNDED_IN_CONTEXT, "exact", candidates

    # A reference to ``2/4`` is grounded by a retrieved ``2/4/4``: the parent was
    # shown, in the sense that one of its children was. The reverse is not
    # grounding - a reference to ``2/4/4/1`` when only ``2/4/4`` was retrieved
    # points at a sub-clause the model never saw - so this test is one-directional.
    for candidate in candidates:
        paths = context_by_standard.get(candidate or "", frozenset())
        if any(clause_path in clause_ancestors(path) for path in paths):
            return ReferenceStatus.GROUNDED_IN_CONTEXT, "ancestor", candidates

    if clause_path in context_blob:
        return ReferenceStatus.GROUNDED_IN_CONTEXT, "verbatim_in_context", candidates

    if corpus_index is None:
        return ReferenceStatus.OUTSIDE_CONTEXT, "no_corpus_index", candidates

    if standard_id is not None and standard_id not in corpus_index.standards:
        return ReferenceStatus.UNKNOWN_STANDARD, "standard_not_in_corpus", candidates

    corpus_candidates = (
        (standard_id,) if standard_id is not None else tuple(sorted(corpus_index.standards))
    )
    for candidate in corpus_candidates:
        if candidate and corpus_index.contains(candidate, clause_path):
            return ReferenceStatus.OUTSIDE_CONTEXT, "corpus_exact", corpus_candidates
    return ReferenceStatus.UNRESOLVABLE, "no_match", corpus_candidates


def audit_clause_grounding(
    response_text: str,
    context_records: Sequence[Mapping[str, Any]],
    *,
    corpus_index: CorpusClauseIndex | None = None,
    min_path_depth: int = MIN_PATH_DEPTH,
) -> ClauseGroundingAudit:
    """Resolve every clause reference in ``response_text`` against the context.

    ``corpus_index`` is optional and never changes :attr:`ClauseGroundingAudit.passed`;
    it only splits a non-grounded reference into ``OUTSIDE_CONTEXT`` versus
    ``UNRESOLVABLE``. Omit it to run the gate without loading the corpus.
    """
    context_by_standard: dict[str, set[str]] = {}
    for record in context_records:
        standard = record.get("standard_id")
        clause_path = record.get("clause_id")
        if not standard or not clause_path:
            continue
        context_by_standard.setdefault(str(standard), set()).add(str(clause_path))
    frozen_context = {
        standard: frozenset(paths) for standard, paths in context_by_standard.items()
    }
    # Clause *body* text only, deliberately excluding the provenance header the
    # prompt prints. The header repeats every retrieved clause's own path, so
    # including it would ground a reference to ``SS12 §3/1/4/3`` against a
    # retrieved ``SS17 §3/1/4/3`` - a different standard's clause that merely
    # shares a path shape. Nothing is lost by excluding it: a response that
    # transcribes the header, as the stored H01 response does, is grounded by the
    # ``exact`` rule instead, which checks the standard too.
    #
    # What the body text is needed for is AAOIFI's own internal cross-references.
    # A retrieved clause saying "[see para. 7/1]" really does put that path in
    # front of the model, and the seven gold answers include one such quotation.
    context_blob = "\n".join(
        str(record.get("text") or "") for record in context_records
    )

    references: list[ClauseReference] = []
    for clause_path, standard_id, qualified, offset in extract_clause_references(
        response_text or "", min_path_depth=min_path_depth
    ):
        status, resolved_via, candidates = _resolve(
            clause_path, standard_id, frozen_context, context_blob, corpus_index
        )
        references.append(
            ClauseReference(
                clause_path=clause_path,
                standard_id=standard_id,
                qualified=qualified,
                char_offset=offset,
                status=status,
                resolved_via=resolved_via,
                candidate_standards=() if qualified else candidates,
            )
        )

    return ClauseGroundingAudit(
        references=tuple(references),
        context_standards=tuple(sorted(frozen_context)),
        corpus_index_used=corpus_index is not None,
        min_path_depth=min_path_depth,
    )
