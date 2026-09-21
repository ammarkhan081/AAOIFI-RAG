"""The publication boundary (plan Layer 6).

A complete trace contains verbatim AAOIFI clause text three times over: in the retrieved
records, in the built prompt, and - on H03, whose entire answer is a quotation - in the
model's response. ``docs/governance/`` and ``data/README.md`` forbid that text reaching a
public repository. So this module's job is not "log some fields"; it is to make the
licensing boundary a property of the code rather than of the author's attention. These
tests check the boundary the way it would actually fail.

Four claims carry the weight:

* **A full trace cannot be written outside a Git-ignored root, and the refusal happens at
  construction.** ``TraceWriter(full_path=...)`` calls
  :func:`assert_private_destination` in ``__init__``, so a mistyped destination raises
  before any licensed text has been serialised - not after the first ``write()``.
* **``PRIVATE_ROOTS`` is only as true as ``.gitignore``.** The tuple is a claim about
  another file. It is checked against that file here, line by line, because a root
  removed from ``.gitignore`` would leave the guard happily approving a tracked
  directory.
* **The public view is a redaction, not a rename.** Clause text, prompt text and response
  text are replaced by truncated digests. Asserted against the real corpus, with real
  clause prose as the forbidden snippets, so the guarantee is measured rather than
  declared.
* **The published artefact is checked, not just the code that could produce it.**
  ``reports/replay_n7_public_traces.jsonl`` is tracked. The last section reads that file
  and hunts for corpus text in it.

No AAOIFI clause prose is written into this file. The corpus-gated tests read real prose
at run time and use it as a needle; nothing they read is ever asserted as a literal.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from aaoifi_rag.generation.prompt import AnswerPrompt, build_answer_prompt
from aaoifi_rag.generation.types import GeneratedAnswer, GenerationConfig
from aaoifi_rag.orchestration.pipeline import AnswerPipeline, PipelineResult
from aaoifi_rag.orchestration.protocols import RetrievalResult
from aaoifi_rag.reporting.evaluation import build_evaluation_label
from aaoifi_rag.reporting.trace import (
    DIGEST_CHARS,
    PRIVATE_ROOTS,
    TRACE_SCHEMA_VERSION,
    AnswerTrace,
    PublicationBoundaryError,
    TraceWriter,
    assert_no_clause_text,
    assert_private_destination,
    read_traces,
    text_digest,
)

#: Invented prose, long enough to survive the eight-word shingle window intact.
INVENTED_CLAUSE = (
    "The institution shall record the widget at the value agreed between the parties "
    "and shall disclose that value in the notes to the financial statements."
)

PUBLIC_TRACES = "reports/replay_n7_public_traces.jsonl"


# --------------------------------------------------------------------------------------
# Scripted backends. Nothing here touches a GPU, a model or the network.
# --------------------------------------------------------------------------------------


class ScriptedRetriever:
    """Return a fixed record list. Satisfies ``Retriever``."""

    def __init__(self, records: Sequence[Mapping[str, Any]], **policy: Any) -> None:
        self._records = tuple(records)
        self._policy = policy or {"scripted": True}

    def retrieve(self, query_text: str, k: int) -> RetrievalResult:
        return RetrievalResult(
            records=self._records[:k],
            reranker_signal=None,
            retrieval_policy={**self._policy, "requested_k": k},
        )


class ScriptedGenerator:
    """Return a fixed string. Satisfies ``Generator``."""

    def __init__(self, text: str, config: GenerationConfig | None = None) -> None:
        self._text = text
        self._config = config or GenerationConfig()

    def generate(self, prompt: AnswerPrompt) -> GeneratedAnswer:
        return GeneratedAnswer(
            text=self._text,
            config=self._config,
            input_token_count=len(prompt.user_prompt.split()),
            new_tokens=len(self._text.split()),
            generation_seconds=0.0,
            truncated=False,
            extra={"source": "scripted_test_generator"},
        )


@pytest.fixture
def clause_context() -> list[dict[str, Any]]:
    """Two records whose ``text`` is invented but positioned exactly like clause prose."""
    return [
        {
            "chunk_id": f"clause:XX:1/{index}:occurrence:0",
            "standard_id": "XX",
            "clause_id": f"1/{index}",
            "sub_clause_id": None,
            "occurrence_index": 0,
            "source_page": 10 + index,
            "reranker_score": 5.5 - index,
            "retrieval_sources": ["bm25", "dense"],
            "text": f"{INVENTED_CLAUSE} Record {index}.",
        }
        for index in (1, 2)
    ]


@pytest.fixture
def result(clause_context) -> PipelineResult:
    """One real pipeline run over scripted backends, cited so the audit has work to do."""
    pipeline = AnswerPipeline(
        retriever=ScriptedRetriever(clause_context, index="scripted"),
        generator=ScriptedGenerator(
            "The institution records the widget at the agreed value [1]."
        ),
    )
    return pipeline.run(item_id="T01", query_text="How is a widget recorded?")


@pytest.fixture
def trace(result) -> AnswerTrace:
    return AnswerTrace.from_pipeline_result(
        result,
        run_id="test_run_v1",
        evaluation=build_evaluation_label(
            {
                "item_id": "T01",
                "expected_behavior": "answer",
                "answerability": "answerable",
                "verification_basis": "corpus_cross_reference",
                "status": "reviewed",
                "gold_clause_ids": [{"standard_id": "XX", "clause_id": "1/1"}],
            },
            result.decision.value,
            list(result.retrieval.records),
        ),
        environment={"scripted": True},
        created_at="2026-09-04T00:00:00+00:00",
    )


# --------------------------------------------------------------------------------------
# text_digest
# --------------------------------------------------------------------------------------


def test_the_digest_is_a_truncated_sha256_of_the_utf8_bytes() -> None:
    """Spelled out rather than compared against the function, so the algorithm is pinned.

    Anyone holding the licensed corpus must be able to recompute these from the clause
    text with a one-liner; that only works if the recipe is written down somewhere a
    reader will find it.
    """
    expected = hashlib.sha256(INVENTED_CLAUSE.encode("utf-8")).hexdigest()[:DIGEST_CHARS]
    assert text_digest(INVENTED_CLAUSE) == expected
    assert DIGEST_CHARS == 16
    assert len(text_digest(INVENTED_CLAUSE)) == 16


def test_none_digests_to_none_and_the_empty_string_does_not() -> None:
    """"No text" and "text of length zero" are different facts about a trace.

    A short-circuited run has ``response_text is None`` because generation never ran; a
    run whose model returned nothing has ``""``. Collapsing them would hide which.
    """
    assert text_digest(None) is None
    assert text_digest("") == hashlib.sha256(b"").hexdigest()[:DIGEST_CHARS]


def test_the_digest_is_sensitive_to_the_kind_of_edit_it_has_to_detect() -> None:
    """Typographic folding is a *matching* concern, not a digest concern.

    ``normalise_for_match`` deliberately folds U+2019 to an apostrophe so a restyled
    quotation still counts as echo. The digest must not: it certifies the exact bytes of
    what was sent to the model, so a trace and a corpus that differ by one character must
    produce different digests.
    """
    curly = "the institution’s books"
    straight = "the institution's books"
    assert text_digest(curly) != text_digest(straight)


# --------------------------------------------------------------------------------------
# assert_private_destination
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("root", PRIVATE_ROOTS)
def test_every_declared_private_root_is_accepted(root, repo_root) -> None:
    target = repo_root / root / "sub" / "traces_full.jsonl"
    assert assert_private_destination(target, repo_root) == target.resolve()


@pytest.mark.parametrize("root", PRIVATE_ROOTS)
def test_the_root_itself_is_accepted_as_well_as_paths_under_it(root, repo_root) -> None:
    """``parts == root_name`` is a separate branch from ``startswith(root + "/")``."""
    assert assert_private_destination(repo_root / root, repo_root)


@pytest.mark.parametrize(
    "relative",
    [
        "reports/traces_full.jsonl",
        "docs/traces.jsonl",
        "traces.jsonl",
        "src/aaoifi_rag/leak.jsonl",
        "tests/leak.jsonl",
        "data/traces.jsonl",
        "data/probes/leak.jsonl",
        "configs/leak.jsonl",
    ],
)
def test_a_tracked_destination_is_refused(relative, repo_root) -> None:
    with pytest.raises(PublicationBoundaryError, match="not under a Git-ignored root"):
        assert_private_destination(repo_root / relative, repo_root)


def test_data_alone_is_not_a_private_root_but_three_of_its_children_are(
    repo_root,
) -> None:
    """The prefix check is on whole path segments, so ``data/`` does not cover itself.

    Worth an explicit test because ``data/`` contains both Git-ignored subtrees
    (``private``, ``derived``, ``raw``) and tracked ones (``manifests``, ``probes``), and
    a guard that matched on ``data`` would approve the tracked ones.
    """
    for child in ("private", "derived", "raw"):
        assert assert_private_destination(repo_root / "data" / child / "x.jsonl", repo_root)
    for child in ("manifests", "probes", "evaluation"):
        with pytest.raises(PublicationBoundaryError):
            assert_private_destination(repo_root / "data" / child / "x.jsonl", repo_root)


def test_a_lookalike_sibling_directory_is_not_mistaken_for_a_private_root(
    repo_root,
) -> None:
    """``runs_public/`` starts with ``runs`` but is not ``runs``. Segment-wise, not
    string-prefix-wise."""
    with pytest.raises(PublicationBoundaryError):
        assert_private_destination(repo_root / "runs_public" / "x.jsonl", repo_root)
    with pytest.raises(PublicationBoundaryError):
        assert_private_destination(repo_root / "logs2" / "x.jsonl", repo_root)


def test_a_path_outside_the_repository_is_refused_rather_than_trusted(
    repo_root, tmp_path
) -> None:
    """The failure mode this closes is the tempting one: "just write it to my Desktop".

    Outside the repository there is no ``.gitignore`` to check against, so the guard
    cannot know whether the destination is safe. It refuses instead of assuming.
    """
    with pytest.raises(PublicationBoundaryError, match="outside the repository"):
        assert_private_destination(tmp_path / "traces_full.jsonl", repo_root)


def test_a_traversal_back_out_of_a_private_root_is_refused(repo_root) -> None:
    """``resolve()`` runs before the prefix check, so ``runs/../reports`` is ``reports``."""
    with pytest.raises(PublicationBoundaryError):
        assert_private_destination(
            repo_root / "runs" / ".." / "reports" / "leak.jsonl", repo_root
        )


def test_the_guard_defaults_to_the_packages_own_repository_root() -> None:
    """No ``repo_root`` argument: derived from ``trace.__file__``, four parents up.

    A relative path is resolved against the *process* cwd, which is why this asserts the
    absolute form. The default matters because production callers do not pass a root.
    """
    from aaoifi_rag.reporting import trace as trace_module

    derived = Path(trace_module.__file__).resolve().parents[3]
    assert (derived / "src" / "aaoifi_rag").is_dir(), "the four-parents walk still holds"
    assert assert_private_destination(derived / "runs" / "x.jsonl")
    with pytest.raises(PublicationBoundaryError):
        assert_private_destination(derived / "reports" / "x.jsonl")


# --------------------------------------------------------------------------------------
# PRIVATE_ROOTS against .gitignore
# --------------------------------------------------------------------------------------


def test_every_private_root_is_actually_git_ignored(repo_root) -> None:
    """The tuple is a claim about ``.gitignore``; here it is checked against it.

    Without this test the guard could keep approving a directory that had been removed
    from ``.gitignore``, and the failure would be invisible until the first commit. The
    match is on the anchored form ``/name/`` because that is how the file spells them,
    and an unanchored pattern would mean something different.
    """
    ignore_lines = {
        line.strip()
        for line in (repo_root / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    missing = [
        root
        for root in PRIVATE_ROOTS
        if f"/{root}/" not in ignore_lines and f"{root}/" not in ignore_lines
    ]
    assert missing == [], f"declared private but not Git-ignored: {missing}"


def test_reports_is_not_git_ignored_which_is_why_the_public_view_exists(
    repo_root,
) -> None:
    """The other half of the same claim. If ``reports/`` were ignored wholesale, the
    redaction would be pointless; it is not, so it is not."""
    ignore_text = (repo_root / ".gitignore").read_text(encoding="utf-8")
    assert "\n/reports/\n" not in f"\n{ignore_text}\n"
    assert "reports" not in PRIVATE_ROOTS


# --------------------------------------------------------------------------------------
# The public view is a redaction
# --------------------------------------------------------------------------------------


def test_the_public_view_carries_no_clause_prompt_or_response_text(trace) -> None:
    """One blob search over the whole payload, on the three texts that must not appear.

    The response is included deliberately: on H03 the model's entire answer was a
    quotation of a retrieved clause, so redacting the records while publishing the
    response would have leaked the clause anyway.
    """
    blob = json.dumps(trace.as_public_dict(), ensure_ascii=False)
    assert INVENTED_CLAUSE not in blob
    assert trace.user_prompt is not None and trace.user_prompt not in blob
    assert trace.response_text is not None and trace.response_text not in blob
    assert "widget at the agreed value" not in blob


def test_the_public_record_keeps_identity_and_provenance_and_swaps_text_for_a_digest(
    trace, clause_context
) -> None:
    """What a reader can still do with a redacted record: locate it and verify it."""
    records = trace.as_public_dict()["retrieval"]["records"]
    assert [record["rank"] for record in records] == [1, 2]
    first = records[0]
    assert first["chunk_id"] == clause_context[0]["chunk_id"]
    assert (first["standard_id"], first["clause_id"]) == ("XX", "1/1")
    assert first["source_page"] == 11
    assert first["retrieval_sources"] == ["bm25", "dense"]
    assert first["text_sha256_16"] == text_digest(clause_context[0]["text"])
    assert first["text_char_count"] == len(clause_context[0]["text"])
    assert "text" not in first


def test_a_record_without_text_reports_none_rather_than_zero(trace) -> None:
    """A missing ``text`` field and an empty one are different, as with the digest."""
    from aaoifi_rag.reporting.trace import _public_record

    bare = _public_record(1, {"chunk_id": "clause:XX:1/1:occurrence:0"})
    assert bare["text_sha256_16"] is None
    assert bare["text_char_count"] is None
    assert bare["occurrence_index"] == 0, "defaulted, matching clause_key's coercion"


def test_the_prompt_is_reduced_to_two_digests_and_one_length(trace) -> None:
    """Enough to prove which prompt was used; not enough to reconstruct it."""
    prompt = trace.as_public_dict()["prompt"]
    assert prompt["prompt_version"] == trace.prompt_version
    assert prompt["system_prompt_sha256_16"] == text_digest(trace.system_prompt)
    assert prompt["user_prompt_sha256_16"] == text_digest(trace.user_prompt)
    assert prompt["user_prompt_char_count"] == len(trace.user_prompt or "")
    assert set(prompt) == {
        "prompt_version",
        "system_prompt_sha256_16",
        "user_prompt_sha256_16",
        "user_prompt_char_count",
    }


def test_the_query_text_can_be_withheld_for_publication_before_the_hard_set_is_released(
    trace,
) -> None:
    """``question_text`` is authored by this project, so it defaults to included.

    ``data/private/hard_set.jsonl`` is nonetheless Git-ignored in full, so the switch
    exists. ``scripts/replay_n7_router.py`` leaves it on, which is why the tracked
    artefact carries the questions.
    """
    assert trace.as_public_dict()["query_text"] == "How is a widget recorded?"
    assert trace.as_public_dict(include_query_text=False)["query_text"] is None
    assert trace.query_text_only() == "How is a widget recorded?"


def test_the_public_view_labels_itself_and_repeats_the_verification_note(trace) -> None:
    """A stray trace file has to be self-describing, including about what it is not."""
    payload = trace.as_public_dict()
    assert payload["view"] == "public_no_clause_text"
    assert payload["schema_version"] == TRACE_SCHEMA_VERSION == "answer_trace_v1"
    assert "NOT qualified Shari'ah" in payload["verification_note"]
    assert "fatwa" in payload["verification_note"]
    assert payload["run_id"] == "test_run_v1"
    assert payload["created_at"] == "2026-09-04T00:00:00+00:00"


def test_the_public_view_keeps_every_field_a_reviewer_needs(trace) -> None:
    """Redaction is narrow: only text is removed. Decisions, gates, scores all remain."""
    payload = trace.as_public_dict()
    assert payload["routing"]["decision"] in {"answer", "abstain", "escalate"}
    assert payload["routing"]["policy"]["policy_version"] == "gates_v1"
    assert payload["signals"]["retrieval"] is not None
    assert payload["generation"]["seed"] == 42
    assert payload["generation"]["model_id"] == "inception42/Jais-2-8B-Chat"
    assert [stage["stage"] for stage in payload["stages"]] == [
        "retrieve",
        "gate_retrieval",
        "build_prompt",
        "generate",
        "signals",
        "route",
    ]
    assert payload["evaluation"]["item_id"] == "T01"
    assert payload["evaluation"]["is_scholar_validated"] is False
    assert payload["environment"] == {"scripted": True}


def test_the_response_is_reduced_to_a_digest_and_a_length(trace) -> None:
    response = trace.as_public_dict()["response"]
    assert response == {
        "sha256_16": text_digest(trace.response_text),
        "char_count": len(trace.response_text or ""),
    }


# --------------------------------------------------------------------------------------
# The full view
# --------------------------------------------------------------------------------------


def test_the_full_view_is_the_public_view_plus_the_three_texts(trace) -> None:
    """A superset, so a holder of the full trace never needs both files."""
    public = trace.as_public_dict()
    full = trace.as_full_dict()
    assert full["view"] == "full_contains_licensed_clause_text"
    assert full["prompt"]["user_prompt"] == trace.user_prompt
    assert full["prompt"]["system_prompt"] == trace.system_prompt
    assert full["response"]["text"] == trace.response_text
    assert full["retrieval"]["records"][0]["text"] == trace.context_records[0]["text"]
    # Every digest and identifier the public view carried is still there and unchanged.
    for key, value in public["prompt"].items():
        assert full["prompt"][key] == value
    assert full["response"]["sha256_16"] == public["response"]["sha256_16"]
    assert full["retrieval"]["records"][0]["text_sha256_16"] == text_digest(
        trace.context_records[0]["text"]
    )


def test_the_full_view_declares_that_it_contains_licensed_text(trace) -> None:
    """The ``view`` field is the thing a human grepping a directory will notice."""
    assert "licensed" in trace.as_full_dict()["view"]
    assert INVENTED_CLAUSE in json.dumps(trace.as_full_dict(), ensure_ascii=False)


def test_building_the_full_view_does_not_mutate_the_public_view(trace) -> None:
    """``as_full_dict`` starts from ``as_public_dict()`` and edits it in place.

    That is fine only because the public dict is rebuilt on every call. If it were ever
    cached or returned by reference, the first ``as_full_dict()`` would poison every
    later ``as_public_dict()`` - and the leak would be silent, because the poisoning
    happens after the redaction test has already passed.
    """
    before = trace.as_public_dict()
    trace.as_full_dict()
    after = trace.as_public_dict()
    assert after == before
    assert "text" not in after["retrieval"]["records"][0]
    assert "user_prompt" not in after["prompt"]
    assert "text" not in after["response"]


# --------------------------------------------------------------------------------------
# from_pipeline_result
# --------------------------------------------------------------------------------------


def test_the_trace_captures_the_generation_config_and_the_backends_own_telemetry(
    trace,
) -> None:
    """Config and measurement are merged into one ``generation`` block, config first.

    Both are needed for reproduction: the config says what was asked for, the telemetry
    says what happened. The extras a backend supplies are merged in too, which is how the
    replay script labels its timings as coming from the Colab box.
    """
    assert trace.generation["seed"] == 42
    assert trace.generation["do_sample"] is False
    assert trace.generation["max_new_tokens"] == 512
    assert trace.generation["hub_revision"] == (
        "da0e1639cd92b508b24120f7f77f5270a8465dc4"
    )
    assert trace.generation["truncated"] is False
    assert trace.generation["source"] == "scripted_test_generator"
    assert trace.generation["new_tokens"] is not None


def test_a_short_circuited_run_produces_a_trace_with_no_prompt_or_response(
    clause_context,
) -> None:
    """The empty-retrieval path. Generation never ran, so those fields are ``None``.

    This is the trace shape that must not be mistaken for "the model returned nothing":
    ``as_public_dict`` reports ``sha256_16: None``, and four of six stages record
    ``entered: False``.
    """
    pipeline = AnswerPipeline(
        retriever=ScriptedRetriever([]),
        generator=ScriptedGenerator("this generator is never called"),
    )
    result = pipeline.run(item_id="T02", query_text="anything at all")
    assert result.short_circuited is True

    trace = AnswerTrace.from_pipeline_result(result, run_id="test_run_v1")
    assert trace.system_prompt is None
    assert trace.user_prompt is None
    assert trace.prompt_version is None
    assert trace.response_text is None
    assert trace.generation == {}

    payload = trace.as_public_dict()
    assert payload["response"]["sha256_16"] is None
    assert payload["response"]["char_count"] is None
    assert payload["prompt"]["user_prompt_char_count"] is None
    assert payload["retrieval"]["context_size"] == 0
    assert payload["retrieval"]["records"] == []
    assert [stage["entered"] for stage in payload["stages"]] == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert payload["evaluation"] is None


def test_a_created_at_is_generated_when_not_supplied(result) -> None:
    """UTC and ISO-8601, because traces from two machines get concatenated."""
    from datetime import datetime

    trace = AnswerTrace.from_pipeline_result(result, run_id="r")
    parsed = datetime.fromisoformat(trace.created_at)
    assert parsed.tzinfo is not None, "naive timestamps cannot be ordered across runs"
    assert parsed.utcoffset() is not None
    assert parsed.utcoffset().total_seconds() == 0.0


def test_the_reranker_signal_is_none_when_the_retriever_reported_none(trace) -> None:
    """Distinct from "the margin was zero". The scripted retriever supplies no signal,
    and ``reliability_signal_analysis_n7.md`` found the signal uninformative anyway."""
    assert trace.reranker_signal is None
    assert trace.as_public_dict()["retrieval"]["reranker_signal"] is None


def test_the_trace_is_frozen(trace) -> None:
    with pytest.raises(Exception):
        trace.response_text = "rewritten"  # type: ignore[misc]


# --------------------------------------------------------------------------------------
# TraceWriter
# --------------------------------------------------------------------------------------


def test_a_full_destination_outside_a_private_root_raises_at_construction(
    repo_root, trace
) -> None:
    """Before ``write()``, not during it. The distinction is the whole point.

    A writer that validated on first write would already have opened the file, and a
    partially written full trace in ``reports/`` is exactly the accident this prevents.
    """
    target = repo_root / "reports" / "should_never_exist_traces_full.jsonl"
    with pytest.raises(PublicationBoundaryError):
        TraceWriter(full_path=target, repo_root=repo_root)
    assert not target.exists(), "nothing may be created on the refused path"


def test_a_writer_with_no_destination_is_rejected(repo_root) -> None:
    with pytest.raises(ValueError, match="at least one destination"):
        TraceWriter(repo_root=repo_root)


def test_a_public_only_writer_may_target_a_tracked_directory(tmp_path, trace) -> None:
    """The asymmetry: ``public_path`` is unrestricted because the view is already safe."""
    target = tmp_path / "nested" / "public.jsonl"
    counts = TraceWriter(public_path=target).write([trace])
    assert counts == {"full": 0, "public": 1}
    assert target.exists(), "parent directories are created"
    rows = read_traces(target)
    assert rows[0]["view"] == "public_no_clause_text"
    assert INVENTED_CLAUSE not in target.read_text(encoding="utf-8")


def test_a_full_writer_under_a_private_root_writes_the_licensed_view(
    repo_root, trace
) -> None:
    """Written under ``runs/``, which ``.gitignore`` covers - asserted above, not assumed.

    The file is removed afterwards regardless, because a test that leaves clause-bearing
    output behind is its own small governance failure. The text here is invented, but the
    habit should not depend on that.
    """
    target = repo_root / "runs" / "pytest_tmp" / "traces_full.jsonl"
    try:
        counts = TraceWriter(full_path=target, repo_root=repo_root).write([trace])
        assert counts == {"full": 1, "public": 0}
        rows = read_traces(target)
        assert rows[0]["view"] == "full_contains_licensed_clause_text"
        assert INVENTED_CLAUSE in rows[0]["retrieval"]["records"][0]["text"]
    finally:
        target.unlink(missing_ok=True)
        for parent in (target.parent,):
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()


def test_both_destinations_get_their_own_view_from_one_write(repo_root, tmp_path, trace) -> None:
    """The normal production call. One pass, two files, only one of them publishable."""
    full = repo_root / "runs" / "pytest_tmp" / "both_full.jsonl"
    public = tmp_path / "both_public.jsonl"
    try:
        counts = TraceWriter(
            full_path=full, public_path=public, repo_root=repo_root
        ).write([trace, trace])
        assert counts == {"full": 2, "public": 2}
        assert INVENTED_CLAUSE in full.read_text(encoding="utf-8")
        assert INVENTED_CLAUSE not in public.read_text(encoding="utf-8")
        assert len(read_traces(full)) == len(read_traces(public)) == 2
    finally:
        full.unlink(missing_ok=True)
        if full.parent.is_dir() and not any(full.parent.iterdir()):
            full.parent.rmdir()


def test_the_include_query_text_switch_reaches_the_written_file(tmp_path, trace) -> None:
    target = tmp_path / "redacted.jsonl"
    TraceWriter(public_path=target, include_query_text=False).write([trace])
    assert read_traces(target)[0]["query_text"] is None
    assert "How is a widget recorded?" not in target.read_text(encoding="utf-8")


def test_write_truncates_rather_than_appending_but_the_counter_accumulates(
    tmp_path, trace
) -> None:
    """A real asymmetry, recorded rather than defended.

    ``_write_jsonl`` opens with mode ``"w"``, so a second ``write()`` replaces the file -
    while ``_counts`` keeps climbing. A caller that batched its traces across two calls
    would end up with one batch on disk and a count claiming both. The replay script
    makes exactly one call; anyone adding a second needs to know this.
    """
    writer = TraceWriter(public_path=tmp_path / "twice.jsonl")
    assert writer.write([trace, trace]) == {"full": 0, "public": 2}
    assert writer.write([trace]) == {"full": 0, "public": 3}
    assert len(read_traces(tmp_path / "twice.jsonl")) == 1, "truncated, not appended"


def test_the_written_file_is_one_json_object_per_line_with_unix_endings(
    tmp_path, trace
) -> None:
    """JSONL, and ``newline="\\n"`` even on Windows, so the artefact hashes identically
    on every platform."""
    target = tmp_path / "lines.jsonl"
    TraceWriter(public_path=target).write([trace, trace])
    raw = target.read_bytes()
    assert b"\r\n" not in raw
    assert raw.endswith(b"\n")
    assert raw.decode("utf-8").count("\n") == 2
    for line in raw.decode("utf-8").splitlines():
        assert json.loads(line)["item_id"] == "T01"


def test_non_ascii_is_written_as_itself_not_as_an_escape(tmp_path, trace) -> None:
    """``ensure_ascii=False``. Load-bearing for the script gate's evidence.

    H05's defect was an Arabic character substituted for an apostrophe. If the writer
    escaped it to ``\\u0648`` the finding would still be recoverable, but a reader
    grepping the trace for the character would not find it - and
    ``unexpected_script_chars`` run over the file's text would see backslashes and
    digits.
    """
    target = tmp_path / "unicode.jsonl"
    arabic_trace = AnswerTrace(
        run_id="r",
        item_id="T03",
        created_at="2026-09-04T00:00:00+00:00",
        query_text="does the institutionوs record apply?",
        context_records=(),
        retrieval_policy={},
        reranker_signal=None,
        system_prompt=None,
        user_prompt=None,
        prompt_version=None,
        generation={},
        response_text=None,
        signals={},
        routing={},
        stages=(),
        pipeline_version="single_pass_v1",
    )
    TraceWriter(public_path=target).write([arabic_trace])
    assert "و" in target.read_text(encoding="utf-8")
    assert "\\u0648" not in target.read_text(encoding="utf-8")


def test_read_traces_round_trips_and_ignores_blank_lines(tmp_path, trace) -> None:
    target = tmp_path / "rt.jsonl"
    TraceWriter(public_path=target).write([trace])
    with target.open("a", encoding="utf-8") as stream:
        stream.write("\n   \n")
    rows = read_traces(target)
    assert len(rows) == 1
    assert rows[0] == trace.as_public_dict()


# --------------------------------------------------------------------------------------
# assert_no_clause_text
# --------------------------------------------------------------------------------------


def test_the_leak_check_finds_a_snippet_anywhere_in_the_payload(trace) -> None:
    """It searches the serialised blob, so a snippet hidden in a nested value is found."""
    payload = trace.as_public_dict()
    assert_no_clause_text([payload], [INVENTED_CLAUSE])  # clean
    with pytest.raises(PublicationBoundaryError, match="leaked"):
        assert_no_clause_text([trace.as_full_dict()], [INVENTED_CLAUSE])


def test_the_leak_check_reports_how_many_snippets_leaked_not_which(trace) -> None:
    """The message must not quote the clause text it just caught.

    An exception string ends up in CI logs, which are frequently public. Naming the
    snippet would republish exactly the thing the check exists to withhold.
    """
    with pytest.raises(PublicationBoundaryError) as caught:
        assert_no_clause_text(
            [trace.as_full_dict()], [INVENTED_CLAUSE, "widget at the agreed value"]
        )
    message = str(caught.value)
    assert "2 clause snippet(s) leaked" in message
    assert INVENTED_CLAUSE not in message
    assert "widget" not in message


def test_a_blank_snippet_is_skipped_rather_than_matching_everything(trace) -> None:
    """``if snippet and ...`` - an empty needle is in every haystack.

    Same fail-closed reasoning as ``contains_normalised("anything", "")`` returning
    False: a degenerate input must not manufacture a finding.
    """
    assert_no_clause_text([trace.as_full_dict()], ["", "   ", None])  # type: ignore[list-item]


def test_the_leak_check_passes_on_an_empty_payload_list(trace) -> None:
    assert_no_clause_text([], [INVENTED_CLAUSE])


# --------------------------------------------------------------------------------------
# Corpus-gated: the boundary measured against real clause text
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def corpus_snippets(clause_chunks) -> list[str]:
    """Distinctive slices of real clause prose, used only as search needles.

    Twelve words from the middle of each of thirty records: long enough to be unique to
    the corpus, short enough to survive the extraction's line wrapping. Nothing here is
    asserted as a literal, and none of it is written to disk.
    """
    snippets: list[str] = []
    for chunk in clause_chunks[::12]:
        tokens = str(chunk.get("text", "")).split()
        if len(tokens) >= 24:
            snippets.append(" ".join(tokens[6:18]))
    return snippets


@pytest.mark.requires_private_data
def test_the_snippets_really_are_present_in_the_corpus(
    clause_chunks, corpus_snippets
) -> None:
    """A negative control for the two leak tests below.

    If the needles were malformed, "no needle found in the public view" would be
    vacuously true and the guarantee would be untested. So first check the needles find
    the corpus.
    """
    assert len(corpus_snippets) >= 20
    blob = json.dumps([chunk.get("text") for chunk in clause_chunks], ensure_ascii=False)
    assert all(snippet in blob for snippet in corpus_snippets)


@pytest.mark.requires_private_data
def test_a_public_view_built_over_real_clauses_leaks_none_of_them(
    clause_chunks, corpus_snippets
) -> None:
    """The guarantee, measured against the licensed text rather than an invented stand-in.

    The context here is five real records and the response is a verbatim quotation of the
    first one - the H03 shape, which is the worst case for a redaction that only covered
    the record list.
    """
    records = clause_chunks[:5]
    pipeline = AnswerPipeline(
        retriever=ScriptedRetriever(records, index="real_corpus_slice"),
        generator=ScriptedGenerator(f"{records[0]['text']} [1]"),
    )
    result = pipeline.run(item_id="R01", query_text="an invented question")
    trace = AnswerTrace.from_pipeline_result(result, run_id="leak_check_v1")

    assert_no_clause_text([trace.as_public_dict()], corpus_snippets)
    # And the negative control: the full view does leak, so the check can see this data.
    with pytest.raises(PublicationBoundaryError):
        assert_no_clause_text([trace.as_full_dict()], corpus_snippets)


@pytest.mark.requires_private_data
def test_the_tracked_published_traces_contain_no_corpus_text(
    repo_root, corpus_snippets
) -> None:
    """The artefact itself, not the code that could produce it.

    ``reports/replay_n7_public_traces.jsonl`` is committed. This is the test that would
    have caught a redaction bug *after* the file was written, which is the only moment
    that matters. Two searches: the parsed payloads through the library check, and the
    raw bytes, in case a snippet survived somewhere the parser normalises away.
    """
    path = repo_root / PUBLIC_TRACES
    assert path.exists(), f"{PUBLIC_TRACES} is tracked and must be present"
    rows = read_traces(path)
    assert len(rows) == 7
    assert_no_clause_text(rows, corpus_snippets)
    raw = path.read_text(encoding="utf-8")
    leaked = [snippet for snippet in corpus_snippets if snippet in raw]
    assert leaked == [], f"{len(leaked)} corpus snippet(s) in the raw published file"


@pytest.mark.requires_private_data
def test_the_published_digests_match_the_licensed_corpus(
    repo_root, clause_chunks
) -> None:
    """What the redaction bought: verifiability without republication.

    Every ``text_sha256_16`` in the published traces is recomputed from the real clause
    text, which is the exact operation a reader holding the licensed standards would
    perform to confirm the traces describe the corpus they hold. 35 records across seven
    items, and the character counts too.
    """
    by_chunk = {chunk["chunk_id"]: chunk for chunk in clause_chunks}
    checked = 0
    for row in read_traces(repo_root / PUBLIC_TRACES):
        for record in row["retrieval"]["records"]:
            source = by_chunk[record["chunk_id"]]
            assert record["text_sha256_16"] == text_digest(source["text"]), record[
                "chunk_id"
            ]
            assert record["text_char_count"] == len(source["text"])
            checked += 1
    assert checked == 35, "seven items times five records"


@pytest.mark.requires_private_data
def test_the_published_traces_carry_no_text_keys_at_all(repo_root) -> None:
    """A structural check alongside the content one, and cheaper to reason about.

    A snippet search can only find text it was given a needle for. This finds any key
    that would hold text regardless of what is in it, so a future field named ``text``
    or ``user_prompt`` fails here even on a run whose corpus this test cannot see.
    """
    forbidden = {"text", "system_prompt", "user_prompt", "bm25_text", "residue_text"}

    def walk(node: Any, path: str = "") -> list[str]:
        found: list[str] = []
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key in forbidden:
                    found.append(f"{path}.{key}")
                found.extend(walk(value, f"{path}.{key}"))
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                found.extend(walk(value, f"{path}[{index}]"))
        return found

    offenders = [
        offender
        for row in read_traces(repo_root / PUBLIC_TRACES)
        for offender in walk(row)
    ]
    assert offenders == []


@pytest.mark.requires_private_data
def test_the_published_traces_are_reproduced_by_the_replay_script_they_claim(
    repo_root, stored_run, clause_chunks
) -> None:
    """Ranks, chunk ids and reranker scores, from the stored run through to the artefact.

    Not a re-run of the router - the decisions are ``test_metrics.py``'s business. This
    checks the narrower thing a trace exists to support: that the published record of
    what the model was shown matches what the GPU run recorded showing it.
    """
    stored_context = {
        row["item_id"]: sorted(row["top5"], key=lambda entry: entry["rank"])
        for row in stored_run["items"]
    }
    known_chunks = {chunk["chunk_id"] for chunk in clause_chunks}
    for row in read_traces(repo_root / PUBLIC_TRACES):
        expected = stored_context[row["item_id"]]
        published = row["retrieval"]["records"]
        assert [record["rank"] for record in published] == [1, 2, 3, 4, 5]
        assert [record["chunk_id"] for record in published] == [
            entry["chunk_id"] for entry in expected
        ]
        assert [record["reranker_score"] for record in published] == [
            entry["reranker_score"] for entry in expected
        ]
        assert all(record["chunk_id"] in known_chunks for record in published)
        assert row["environment"]["replay_is_gpu_free"] is True
        assert "not the n=25-30+ pilot" in row["environment"]["stored_run_label"]
