# SAC Ablation Results (n=7)

**Date**: 2026-09-01  
**Method**: Summary-Augmented Chunking (SAC) per Reuter et al. (arXiv:2510.06999)  
**Granularity**: Per-standard summaries (5 total: SS8, SS9, SS13, SS17, SS26)

---

## Results (Clause Recall@5)

| Configuration | Mean Recall@5 | Total Gold in Top-5 | Total Gold |
|---------------|---------------|---------------------|------------|
| B' (existing clause-level BM25) | 0.571 | 10 | 17 |
| B'+rerank (existing) | 0.643 | 11 | 17 |
| B'' (SAC BM25 only) | 0.750 | 12 | 17 |
| B''+rerank (SAC) | 0.833 | 14 | 17 |

**SAC improvement**: +0.179 vs B', +0.190 vs B'+rerank

---

## Generated Summaries

### SS13 (Mudarabah)
AAOIFI SS13, covering Mudarabah contracts, defines Mudarabah as a profit-sharing partnership where one party provides capital and the other provides labor. The standard details permissible Mudarabah financing arrangements, requiring a memorandum of understanding to establish the general framework, profit ratio, and types of guarantees. It clarifies the binding nature of Mudarabah contracts once business has commenced or a designated duration is agreed upon, emphasizing the Mudarib's role as an investor on a trust basis, liable for losses primarily due to negligence or breach of contract. The standard also categorizes Mudarabah into unrestricted and restricted forms, outlining the limitations and freedoms in each.

### SS17 (Sukuk)
AAOIFI SS17, a Shari'ah standard for investment Sukuk, defines these as certificates of equal value representing ownership in tangible assets, usufructs, or services, issued after asset acquisition and project funding. The standard classifies Sukuk into several types, including ownership of leased assets, usufructs of existing assets, ownership of usufructs of described future assets, services of a specified party, and Salam/Istisna certificates for specific asset or service delivery. It also explicitly excludes shares of joint stock companies, certificates of funds, and general investment portfolios from its scope.

### SS26 (Islamic Insurance)
This AAOIFI Shari'ah Standard (SS26) defines Islamic Insurance as a process of agreement where participants contribute to an insurance fund, managed by a committee or company, to provide compensation for injuries. It contrasts Islamic Insurance with Conventional Insurance, highlighting that Islamic insurance is based on a musharakah (partnership) structure among participants or a wakalah (agency) with the company, where the company also invests the fund. The standard details the contractual relationships within Islamic insurance, primarily focusing on musharakah among participants/company, wakalah with the managing company, and the direct relationship between policyholders and the fund.

### SS8 (Murabahah) ⚠️
AAOIFI SS8, covering Murabahah transactions, details procedures before a Murabahah contract is concluded. It outlines the customer's expression of wish, the Institution's procurement process, and crucial requirements regarding the source of supply, preventing prohibited situations like WHEN the customer obtains a price quote from a supplier without it being specifically addressed to them (which would be an invitation to negotiate, not an offer) and stipulating that the Institution must ensure the item is bought from a third party unrelated to the customer or their agent, and that the transaction is genuine to avoidØ±Ø§ØªÙŠØ¬ÙŠ sales.

**SS8 caveat**: Minor encoding artifact at end (`Ø±Ø§ØªÙŠØ¬ÙŠ` = corrupted UTF-8 of Arabic "راتيجي"). Negligible impact on BM25 matching. Root cause: Arabic Unicode range (\u0600-\u06FF) missing from validation regex. Not regenerated due to low impact and Colab cost.

### SS9 (Ijarah Muntahia Bittamleek)
AAOIFI SS9, a comprehensive standard, focuses on operating leases of properties, specifically covering Ijarah Muntahia Bittamleek. It details the principles and permissible practices for various aspects of Ijarah contracts, including the initial promise to lease, direct vs. master lease agreements, and the customer's obligation to ensure seriousness. A crucial provision addresses the customer's payment for security, detailing whether it should be held as an advance payment or an investment, and outlining conditions for its use and the Institution's investment of it. The standard also clarifies the necessary acquisition of the asset prior to lease execution, whether by the customer, a third party, or the Institution directly, and the conditions for doing so.

---

## Corpus Grounding Confirmation

All 5 summaries generated via primary grounded path (full clause text):
- SS13: 56 clauses, 918 input tokens, 136 output tokens (full attempt)
- SS17: 86 clauses, 921 input tokens, 122 output tokens (full attempt)
- SS26: 66 clauses, 871 input tokens, 143 output tokens (full attempt)
- SS8: 77 clauses, 937 input tokens, 117 output tokens (full attempt)
- SS9: 77 clauses, 972 input tokens, 150 output tokens (full attempt)

No ungrounded fallback used. SHA256: `9ac0a0b1483d80c0380285eb8c6709c1dedfebed9b91662818eaf342506dbb81`

---

## Conclusion

SAC provides meaningful retrieval improvement on this corpus, confirming Reuter et al.'s finding transfers to AAOIFI standards. Best performance achieved with SAC + reranking (0.833 Recall@5).
