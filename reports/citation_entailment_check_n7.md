# Citation-Entailment Check: RAG vs. Frontier (n=7)

**Date**: 2026-08-31  
**Purpose**: Manual qualitative check of whether cited clauses genuinely entail/support the model's claims.  
**Data Source**: 
- RAG: `reports/e2e_batch_smoke_results.json` (answered responses: H01, H03, H05)
- Frontier: Claude Sonnet 5 (Extended Thinking) session responses (verbatim)
**Note**: This is a manual/qualitative check on n=7 (frontier) and n=3 (RAG answered) citations, not a statistically validated entailment classifier.

---

## RAG Response H01

**Model Claim**: "Yes, based on the provided excerpts, if a Murabahah is carried out by means of syndicated financing, the Institution which acts as the arranger of the syndicate is entitled to an arrangement fee to be paid by the participants in the syndicate (from SS8 §2/4/4). This sits with SS8's rules on a commitment fee and a fee for providing a credit facility as an arrangement fee is explicitly mentioned as something the Institution is entitled to receive in the context of syndicated financing for Murabahah (from SS8 §2/4/4). A fee for providing a credit facility is prohibited (from SS8 §2/4/2)."

**Cited Clauses**:
- **SS8 §2/4/4**: "If the Murabahah is carried out by means of syndicated financing, the Institution which acts as the arranger of the syndicate is entitled to an arrangement fee to be paid by the participants in the syndicate."
- **SS8 §2/4/2**: "It is not permissible for the Institution to receive a fee for providing a credit facility."

**Entailment Judgment**: **Entailed**

**Justification**: The model's claim that the Institution is entitled to an arrangement fee in syndicated financing is directly supported by SS8 §2/4/4, which states this explicitly. The claim that a fee for providing a credit facility is prohibited is directly supported by SS8 §2/4/2. The model correctly distinguishes between the two types of fees.

---

## RAG Response H03

**Model Claim**: "The capital provider is permitted to obtain guarantees from the Mudarib that are adequate and enforceable."

**Cited Clause**:
- **SS13 §6**: "Guarantees in a Mudarabah Contract The capital provider is permitted to obtain guarantees from the Mudarib that are adequate and enforceable. This is circumscribed by a condition that the capital provider will not enforce these guarantees except in cases of misconduct, negligence or breach of contract on the part of Mudarib."

**Entailment Judgment**: **Partial**

**Justification**: The model's claim is partially entailed by the first sentence of SS13 §6, which states "The capital provider is permitted to obtain guarantees from the Mudarib that are adequate and enforceable." However, the model omits the important condition from the second sentence: "This is circumscribed by a condition that the capital provider will not enforce these guarantees except in cases of misconduct, negligence or breach of contract on the part of Mudarib." This omission makes the model's answer incomplete and potentially misleading.

---

## RAG Response H05

**Model Claim**: "[1] The relationship between the company and the policyholders وأخرجه fund which is a Wakalah relationship in regard to management, and a Mudarabah or investment agency relationship in regard to the investment of the fundتهنئة assets. [2] The managing company is entitled to its own capital and returns on capital, the agency fee, and its specific share of the profits earned by investing the insurance assets through Mudarabah or investment agency. The company also bears all the expenses of its operations including those relating to its tasks for investing the insurance assets."

**Cited Clauses**:
- **SS26 §4/2**: "The relationship between the company and the policyholders' fund which is a Wakalah relationship in regard to management, and a Mudarabah or investment agency relationship in regard to the investment of the fund's assets."
- **SS26 §3**: "Status of Islamic Insurance According to Fiqh (Islamic Jurisprudence) Islamic insurance is based on the commitment of the participants to make donations for the sake of their own interest. The participants, therefore, protect their group by payment of contributions that constitute the resources of the insurance fund, and assign the management of that fund to a committee of policyholders, or to a joint stock company that possesses the license of practicing insurance business. In the latter case, the company assumes this job on the basis of a remunerated Wakalah (Agency) contract. In addition to managing the insurance operations, the committee of policyholders or the company also assumes the responsibility of investing the assets of the fund through Mudarabah or investment agency."

**Entailment Judgment**: **Not Entailed**

**Justification**: The model's first sentence is partially entailed by SS26 §4/2, which correctly states the Wakalah/Mudarabah relationship distinction. However, the model's second sentence ("The managing company is entitled to its own capital and returns on capital, the agency fee, and its specific share of the profits earned by investing the insurance assets through Mudarabah or investment agency") is not supported by either cited clause. SS26 §3 discusses the company's role in management and investment but does not mention entitlement to capital, returns, agency fee, or profit shares. The model appears to be hallucinating or conflating concepts not present in the retrieved context. Additionally, the response contains garbled Arabic tokens ("أخرجه", "تهنئة") which indicates generation quality issues.

---

## Frontier Response H01

**Model Claim**: "Yes. SS8 §2/4/4 states: 'If the Murabahah is carried out by means of syndicated financing, the Institution which acts as the arranger of the syndicate is entitled to an arrangement fee to be paid by the participants in the syndicate.' This sits alongside two prohibitions in the same corpus section — SS8 §2/4/1: 'It is not permissible for the Institution to receive a commitment fee from the customer,' and SS8 §2/4/2: 'It is not permissible for the Institution to receive a fee for providing a credit facility.' The corpus lists the arrangement fee as a distinct, separately-permitted item under 'Commissions and expenses'; it does not itself explain the reasoning that distinguishes an arrangement fee from the two prohibited fees, so I'm not inferring one."

**Cited Clauses**:
- **SS8 §2/4/4**: "If the Murabahah is carried out by means of syndicated financing, the Institution which acts as the arranger of the syndicate is entitled to an arrangement fee to be paid by the participants in the syndicate."
- **SS8 §2/4/1**: "It is not permissible for the Institution to receive a commitment fee from the customer."
- **SS8 §2/4/2**: "It is not permissible for the Institution to receive a fee for providing a credit facility."

**Entailment Judgment**: **Entailed**

**Justification**: The model's claim is directly supported by all three cited clauses. SS8 §2/4/4 explicitly permits arrangement fees in syndicated financing, while SS8 §2/4/1 and §2/4/2 prohibit commitment fees and credit facility fees respectively. The model correctly distinguishes these three fee types and explicitly states it is not inferring reasoning not present in the corpus.

---

## Frontier Response H02

**Model Claim**: "No. SS9 §8/1 requires that 'the method of transferring the title in the leased asset to the lessee must be evidenced in a document separate from the Ijarah contract document.' It further specifies: 'the separate document evidencing a promise of gift, promise of sale or a promise of gift contingent on a particular event, should be independent of the contract of Ijarah Muntahia Bittamleek and cannot be taken as an integral part of the contract of Ijarah.' So the two cannot be combined into one binding document."

**Cited Clauses**:
- **SS9 §8/1 (occurrence:0)**: "In Ijarah Muntahia Bittamleek, the method of transferring the title in the leased asset to the lessee must be evidenced in a document separate from the Ijarah contract document, using one of the following methods:"
- **SS9 §8/1(c) (occurrence:3)**: "A promise to give it as a gift, contingent upon the payment of the remaining instalments. In all these cases, the separate document evidencing a promise of gift, promise of sale or a promise of gift contingent on a particular event, should be independent of the contract of Ijarah Muntahia Bittamleek and cannot be taken as an integral part of the contract of Ijarah."

**Entailment Judgment**: **Entailed**

**Justification**: The model's claim is fully entailed by SS9 §8/1. The first quote matches SS9 §8/1 at occurrence_index=0 (parent clause). The second quote matches SS9 §8/1(c) at occurrence_index=3 (sub-clause c). Both quotes are verbatim-present in the corpus under different occurrence indices. The model correctly bundles these under the generic "SS9 §8/1" citation label. The conclusion that the two documents cannot be combined is directly supported by the corpus text.

**Note on Verification Method**: This classification was initially marked "Partial" due to checking only occurrence_index=0 for SS9 §8/1, missing the text in occurrence_index=3 (sub-clause c). This is the same class of error (checking only occurrence_index=0 instead of all occurrences under a clause_id) that was previously caught and fixed in H02's own gold_clause_ids earlier in this project. This is a recurring blind spot worth remembering for any future manual/automated entailment checking.

---

## Frontier Response H03

**Model Claim**: "Yes. SS13 §6 ('Guarantees in a Mudarabah Contract') states: 'The capital provider is permitted to obtain guarantees from the Mudarib that are adequate and enforceable. This is circumscribed by a condition that the capital provider will not enforce these guarantees except in cases of misconduct, negligence or breach of contract on the part of Mudarib.' So enforcement is limited to those three conditions only."

**Cited Clause**:
- **SS13 §6**: "Guarantees in a Mudarabah Contract The capital provider is permitted to obtain guarantees from the Mudarib that are adequate and enforceable. This is circumscribed by a condition that the capital provider will not enforce these guarantees except in cases of misconduct, negligence or breach of contract on the part of Mudarib."

**Entailment Judgment**: **Entailed**

**Justification**: The model's claim is fully entailed by SS13 §6. The clause explicitly states that guarantees are permitted but enforcement is limited to misconduct, negligence, or breach of contract. The model correctly summarizes this condition.

---

## Frontier Response H04

**Model Claim**: "No. SS17 §5/2/2 states: 'it is permissible for the issuer to undertake, through the prospectus of issue, to purchase at market value, after the completion of the process of issue, any certificate that may be offered to him, however, it is not permissible for the issuer to undertake to purchase the Sukuk at their nominal value.' Purchase at market value is allowed; purchase at nominal value is not."

**Cited Clause**:
- **SS17 §5/2/2**: "In the case of negotiable Sukuk, it is permissible for the issuer to undertake, through the prospectus of issue, to purchase at market value, after the completion of the process of issue, any certificate that may be offered to him, however, it is not permissible for the issuer to undertake to purchase the Sukuk at their nominal value"

**Entailment Judgment**: **Entailed**

**Justification**: The model's claim is directly supported by SS17 §5/2/2, which explicitly permits purchase at market value but prohibits purchase at nominal value. The model correctly summarizes this distinction.

---

## Frontier Response H05

**Model Claim**: "SS26 §4/2 states: 'The relationship between the company and the policyholders' fund which is a Wakalah relationship in regard to management, and a Mudarabah or investment agency relationship in regard to the investment of the fund's assets.' This is echoed in SS26 §5/3: 'The company should assume the role of the agent in managing the insurance account, and the role of the Mudarib or agent in investing the insurance assets.' So no, treating both legs as Mudarabah-only would not match: management is specifically described as a Wakalah (agency) relationship, not Mudarabah, and even the investment leg is 'Mudarabah or investment agency,' not restricted to Mudarabah alone."

**Cited Clauses**:
- **SS26 §4/2**: "The relationship between the company and the policyholders' fund which is a Wakalah relationship in regard to management, and a Mudarabah or investment agency relationship in regard to the investment of the fund's assets."
- **SS26 §5/3**: "The company should assume the role of the agent in managing the insurance account, and the role of the Mudarib or agent in investing the insurance assets."

**Entailment Judgment**: **Entailed**

**Justification**: The model's claim is fully entailed by both cited clauses. SS26 §4/2 explicitly states the management relationship is Wakalah and investment is Mudarabah or investment agency. SS26 §5/3 echoes this distinction. The model correctly infers that treating both legs as Mudarabah-only would not match the corpus description.

---

## Frontier Response H06

**Model Claim**: "No. SS8 §5/6 states: 'It is permissible that the contract of Murabahah consists of an undertaking from the customer to pay an amount of money or a percentage of the debt, on the basis of undertaking to donate it in the event of a delay on his part in paying instalments on their due date. The Shari'ah Supervisory Board of the Institution must have full knowledge that any such amount is indeed spent on charitable causes, and not for the benefit of the Institution itself.' The amount must go to charitable causes, with Shari'ah Supervisory Board oversight — not retained for the Institution's own benefit."

**Cited Clause**:
- **SS8 §5/6**: "It is permissible that the contract of Murabahah consists of an undertaking from the customer to pay an amount of money or a percentage of the debt, on the basis of undertaking to donate it in the event of a delay on his part in paying instalments on their due date. The Shari'ah Supervisory Board of the Institution must have full knowledge that any such amount is indeed spent on charitable causes, and not for the benefit of the Institution itself."

**Entailment Judgment**: **Entailed**

**Justification**: The model's claim is fully entailed by SS8 §5/6. The clause explicitly requires that any such amount be spent on charitable causes with Shari'ah Supervisory Board oversight, not for the Institution's benefit. The model correctly summarizes this requirement.

---

## Frontier Response H07

**Model Claim**: "No, this would not satisfy the standard. SS9 §8/5 states: 'If the leased asset was purchased from the lessee before it was leased back to the lessee on the basis of Ijarah Muntahia Bittamleek, a (reasonable) period of time, between the lease contract and the time of the sale of the asset to the lessee, must have expired, to avoid the contract of 'Inah. This period must be long enough so that the leased property or its value could have changed.' A same-day sale-and-leaseback with no gap would not meet this reasonable-period requirement."

**Cited Clause**:
- **SS9 §8/5**: "If the leased asset was purchased from the lessee before it was leased back to the lessee on the basis of Ijarah Muntahia Bittamleek, a (reasonable) period of time, between the lease contract and the time of the sale of the asset to the lessee, must have expired, to avoid the contract of 'Inah. This period must be long enough so that the leased property or its value could have changed. This shall also apply to the case of early ownership of the asset where a sale contract is concluded during the Ijarah. [see para. 7/1]"

**Entailment Judgment**: **Entailed**

**Justification**: The model's claim is fully entailed by SS9 §8/5. The clause explicitly requires a reasonable period of time between lease contract and sale to avoid 'Inah, and states this period must be long enough for the property or its value to change. The model correctly infers that same-day sale-and-leaseback would not meet this requirement.

---

## Summary: RAG vs. Frontier

### RAG Answered Responses (n=3)

| Item | Entailment | Justification |
|------|------------|---------------|
| H01 | Entailed | Directly supported by cited clauses SS8 §2/4/4 and §2/4/2. |
| H03 | Partial | First sentence entailed by SS13 §6, but omits important condition about enforcement circumstances. |
| H05 | Not Entailed | First sentence partially entailed by SS26 §4/2, but second sentence hallucinates concepts not in cited clauses. Also contains garbled Arabic tokens. |

**RAG Entailment Rate (answered subset)**: 1/3 fully entailed (33%), 1/3 partial (33%), 1/3 not entailed (33%).

### Frontier Responses (n=7)

| Item | Entailment | Justification |
|------|------------|---------------|
| H01 | Entailed | Directly supported by cited clauses SS8 §2/4/4, §2/4/1, §2/4/2. |
| H02 | Entailed | Both quotes verbatim-present in SS9 §8/1 at occurrence_index=0 and occurrence_index=3 (sub-clause c). Model correctly bundles these under generic "SS9 §8/1" label. |
| H03 | Entailed | Fully entailed by SS13 §6, correctly summarizes enforcement conditions. |
| H04 | Entailed | Directly supported by SS17 §5/2/2, correctly distinguishes market vs. nominal value purchase. |
| H05 | Entailed | Fully entailed by SS26 §4/2 and §5/3, correctly infers Wakalah/Mudarabah distinction. |
| H06 | Entailed | Fully entailed by SS8 §5/6, correctly summarizes charitable cause requirement. |
| H07 | Entailed | Fully entailed by SS9 §8/5, correctly infers reasonable-period requirement. |

**Frontier Entailment Rate (full set)**: 7/7 fully entailed (100%), 0/7 partial (0%), 0/7 not entailed (0%).

### Aggregate Comparison

**Important Caveat**: These percentages are not directly comparable because they use different bases:
- RAG: 3 answered responses out of 7 total items (43% answered rate)
- Frontier: 7 answered responses out of 7 total items (100% answered rate)

The RAG analysis only covers the subset of items where Jais-2 chose to answer, while the frontier analysis covers all items. This selection bias means the RAG entailment rate may not represent the system's performance on abstained items.

**Observation**: RAG's answered responses show lower entailment quality (33% fully entailed) compared to frontier (100% fully entailed). This suggests that even when Jais-2 chooses to answer, the answers may not be fully grounded in the retrieved context.

---

## RAG Entailment Pattern vs. Response Quality (n=3 Observation)

**H01 (entailed)**: Long, well-structured answer with explicit citation format. Fully grounded in retrieved context.

**H03 (partial)**: Short fragment (25 tokens). Partially grounded but omits important conditions from cited clause.

**H05 (not entailed)**: Garbled response with Arabic tokens and self-contradictory trailing abstention. Not grounded in retrieved context (hallucinates concepts).

**Interpretation**: RAG's entailment pattern tracks response length and coherence. The longer, well-structured answer was fully grounded, the short fragment was partially grounded, and the garbled response was not grounded. This is an n=3 observation, not a validated correlation, but suggests that response quality (length, coherence, structure) may correlate with grounding quality.

---

## Verification Note

Hard-set items are `corpus_cross_reference` only, **not** qualified Shari'ah scholar review. This is a manual qualitative check, not a formal entailment classifier or Shari'ah judgment.
