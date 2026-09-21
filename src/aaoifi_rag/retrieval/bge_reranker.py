"""Direct Transformers implementation for BGE reranking.

This deliberately does not use ``FlagReranker``. The Phase-2 Colab run found
that wrapper's internal method unreliable, while this AutoTokenizer plus
AutoModelForSequenceClassification pattern completed successfully.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True)
class RerankerSignal:
    """Raw reranker score signal; it is not calibrated uncertainty."""

    top_1_score: float | None
    top_2_score: float | None
    top_1_top_2_margin: float | None


class DirectBGEReranker:
    """Score query-passage pairs with BGE's documented Transformers pattern.

    Third-party imports happen at construction time so lexical-only code can
    remain importable in the local sandbox, where PyTorch is intentionally not
    installed.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_BGE_RERANKER_MODEL,
        *,
        device: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Direct BGE reranking requires PyTorch and transformers. "
                "Use the Colab environment or install the model dependencies first."
            ) from error

        self._torch = torch
        self.model_id = model_id
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str] | list[str]],
        *,
        batch_size: int = 16,
        max_length: int = 1024,
    ) -> list[float]:
        """Return one raw ranking logit per ``(query, passage)`` pair."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_length < 1:
            raise ValueError("max_length must be positive")

        scores: list[float] = []
        for offset in range(0, len(pairs), batch_size):
            batch = pairs[offset : offset + batch_size]
            if any(len(pair) != 2 for pair in batch):
                raise ValueError("Each reranker input must contain exactly a query and passage.")
            queries = [pair[0] for pair in batch]
            passages = [pair[1] for pair in batch]
            tokenized = self.tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokenized = {name: tensor.to(self.device) for name, tensor in tokenized.items()}
            with self._torch.no_grad():
                logits = self.model(**tokenized, return_dict=True).logits.view(-1)
            scores.extend(float(value) for value in logits.float().cpu().tolist())
        return scores


def rerank_records(
    reranker: DirectBGEReranker,
    query: str,
    candidates: Iterable[Mapping[str, Any]],
    *,
    text_key: str = "text",
    batch_size: int = 16,
    max_length: int = 1024,
) -> tuple[list[dict[str, Any]], RerankerSignal]:
    """Rerank records and return raw top-score/margin reliability signals."""
    records = [dict(candidate) for candidate in candidates]
    pairs: list[tuple[str, str]] = []
    for record in records:
        text = record.get(text_key)
        if not isinstance(text, str) or not text:
            raise ValueError(f"Each candidate must contain non-empty string field {text_key!r}.")
        pairs.append((query, text))

    scores = reranker.score_pairs(pairs, batch_size=batch_size, max_length=max_length)
    if len(scores) != len(records):
        raise RuntimeError("Reranker returned a score count different from the candidate count.")
    ranked = [
        {**record, "reranker_score": score}
        for record, score in zip(records, scores, strict=True)
    ]
    ranked.sort(key=lambda record: record["reranker_score"], reverse=True)

    top_1 = ranked[0]["reranker_score"] if ranked else None
    top_2 = ranked[1]["reranker_score"] if len(ranked) > 1 else None
    return ranked, RerankerSignal(
        top_1_score=top_1,
        top_2_score=top_2,
        top_1_top_2_margin=(top_1 - top_2) if top_1 is not None and top_2 is not None else None,
    )
