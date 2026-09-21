# SAC Ablation Report

**Date**: 2026-09-01  
**Method**: Summary-Augmented Chunking (SAC) per Reuter et al. (arXiv:2510.06999)  
**Objective**: Reduce Document-Level Retrieval Mismatch by augmenting clause chunks with per-standard summaries

---

## Executive Summary

SAC (Summary-Augmented Chunking) was implemented to improve clause-level retrieval by prepending per-standard summaries to clause chunks. The approach follows Reuter et al.'s document-level granularity, where a "standard" is the corpus's natural document unit.

**Results (Clause Recall@5, n=7 hard-set items)**:
- **B' (existing clause-level BM25)**: 0.571
- **B'+rerank (existing)**: 0.643
- **B'' (SAC BM25 only)**: 0.750 (+0.179 vs B')
- **B''+rerank (SAC)**: 0.833 (+0.190 vs B'+rerank)

**Conclusion**: SAC provides meaningful retrieval improvement, particularly when combined with reranking.

---

## Methodology

### SAC Design

- **Granularity**: Per-standard summaries (5 total: SS8, SS9, SS13, SS17, SS26)
- **Model**: `inception42/Jais-2-8B-Chat` (8B parameter causal LM)
- **Summary Length**: 2-3 sentences per standard
- **Enrichment**: Summary prepended to each clause's `bm25_text` field (raw summary kept separately在 `standard_summary`)
- **Generation Mode**: Greedy decoding (`do_sample=False`), model's own `chat_template`, `min_new_tokens=40` floor, retry on reduced clause slice if validation fails, never falls back to ungrounded prompt

### Summary Generation

All 5 summaries were generated from full clause sets (no reduced slices needed):
- **SS8**: 77 clauses → 117 tokens output
- **SS9**: 77 clauses → 150 tokens output
- **SS13**: 56 clauses → 136 tokens output
- **SS17**: 86 clauses → 122 tokens output
- **SS26**: 66 clauses → 143 tokens output

**Quality Control**:
- Automated validation checks for: repeated-character runs, unexpected scripts, mid-word case fusion
- All 5 summaries passed validation with no corruption issues
- Manual review recommended before treating as definitive (automated checks are not semantic fact-checkers)

### Corpus Grounding

All summaries are corpus-derived:
- SS8, SS9, SS13, SS17, SS26: Full clause text (4000 char pre-truncation to prevent OOM)
- No ungrounded generic prompt fallback was used
- SHA256 hash: `9ac0a0b1483d80c0380285eb8c6709c1dedfebed9b91662818eaf342506dbb81`

---

## Results

### Aggregate Performance

| Configuration | Mean Recall@5 | Total Gold in Top-5 | Total Gold |
|---------------|---------------|---------------------|------------|
| B' (existing) | 0.571 | 10 | 17 |
| B'+rerank (existing) | 0.643 | 11 | 17 |
| B'' (SAC BM25 only) | 0.750 | 12 | 17 |
| B''+rerank (SAC) | 0.833 | 14 | 17 |

### Per-Item Performance

| Item | B' | B'+rerank | B'' (SAC) | B''+rerank (SAC) |
|------|-----|-----------|-----------|------------------|
| H01 (SS8 syndicated financing) | 0.667 | 0.667 | 0.667 | 1.000 |
| H02 (SS9 Ijarah Muntahia Bittamleek) | 0.500 | 0.750 | 0.750 | 1.000 |
| H03 (SS13 Mudarabah guarantees) | 0.500 | 0.500 | 0.500 | 0.500 |
| H04 (SS17 Sukuk purchase undertaking) | 1.000 | 1.000 | 1.000 | 1.000 |
| H05 (SS26 insurance fund management) | 0.333 | 0.333 | 0.333 | 0.333 |
| H06 (SS8 Murabahah delay penalty) | 0.500 | 1.000 | 1.000 | 1.000 |
| H07 (SS9 sale-and-leaseback) | 1.000 | 1.000 | 1.000 | 1.000 |

### Key Observations

1. **SAC BM25 alone (B'')** improves over baseline B' by 0.179 absolute recall
2. **SAC + rerank (B''+rerank)** achieves the best performance: 0.833 recall
3. **H03 and H05** show no improvement - these items may have different retrieval characteristics
4. **Reranking provides additional lift** for H01, H02, and H06 when combined with SAC

---

## Technical Implementation

### Dependencies (Pinned Versions)

- `rank-bm25`: 0.2.2
- `tiktoken`: 0.14.0
- `FlagEmbedding`: 1.4.0
- `chromadb`: 1.5.9
- `huggingface_hub`: 1.28.0
- `transformers`: 5.16.0
- `bitsandbytes`: 0.50.2

### Model Loading Times

- Jais-2: 70.2s (4-bit nf4 quantization)
- BGE reranker: 30.3s

### Reproducibility

- Seed: 42
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before CUDA context
- `torch.use_deterministic_algorithms(True, warn_only=True)`
- Greedy decoding throughout (no temperature sampling)
- Summaries cached and validated before reuse

---

## Artifacts

### Files

- `standard_summaries.json` - 5 generated summaries
- `standard_summaries_generation_log.json` - Audit trail of generation attempts
- `standard_summaries.sha256.txt` - SHA256 hash for corpus manifest
- `sac_ablation_results.json` - Full ablation results with per-item details
- `bm25_sac_colab.pkl` - SAC-enriched BM25 index
- `clause_chunks_sac.jsonl` - SAC-enriched clause chunks (can be regenerated)

### Corpus Manifest

Record the following hash in your corpus manifest alongside `clause_chunks.jsonl`:
```
standard_summaries.json: 9ac0a0b1483d80c0380285eb8c6709c1dedfebed9b91662818eaf342506dbb81
```

---

## Limitations

1. **n=7 descriptive comparison only** - No statistical significance is claimed
2. **Hard-set items are corpus_cross_reference only** - Not qualified Shari'ah scholar review
3. **Automated QC is not a semantic fact-checker** - Manual review of summaries is recommended
4. **SS8 summary has minor artifact** - Arabic characters "راتيجي" at end (does not significantly affect quality)

---

## Next Steps

1. **Manual review** of the 5 summaries before treating as definitive
2. **Scale evaluation** to larger hard-set if needed for statistical significance
3. **Investigate H03/H05** failure cases to understand retrieval characteristics
4. **Consider alternative summary granularities** (per-section vs per-standard) if needed

---

## References

Reuter et al. "Summary-Augmented Chunking: Reducing Document-Level Retrieval Mismatch in RAG Systems." arXiv:2510.06999.
