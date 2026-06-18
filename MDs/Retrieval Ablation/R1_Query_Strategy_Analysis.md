# VerdaSense RAG — Experiment R1: Query Formulation Strategy Ablation
## Comprehensive Analysis & Discussion

**Experiment:** R1 — Query Formulation Strategy  
**Stage:** 1 — Retrieval Ablation  
**Date:** 11 May 2026  
**Configuration:** k=6 | MedEmbed-large-v0.1 | Dense-only (ChromaDB cosine similarity) | 3 runs each  
**RAGAS Judge:** gpt-4o-mini + text-embedding-3-small (fixed across all versions)

---

## 1. Experiment Overview

Experiment R1 investigates a question central to VerdaSense's architecture: **how should structured T.I.M.E. wound assessment inputs be converted into a retrieval query?** Unlike general-purpose RAG systems where the user types a free-text question, VerdaSense receives machine-generated structured inputs (tissue percentages, infection status, moisture level, edge status) from an upstream computer vision pipeline. This structural difference makes the query formulation problem unique and is an original contribution not studied in comparable RAG-for-clinical literature (e.g., Powering & Rothgang, 2026).

Three strategies were compared under identical conditions (same KB, same embedding model, same vector store, same k):

| Version | Label | Strategy Description |
|---|---|---|
| **R1-A** | Flat label concat | Direct concatenation of TIME classification labels into a keyword string |
| **R1-B** | Narrative NL query | GPT-generated natural language question built from TIME inputs |
| **R1-C** | Multi-axis sub-queries | Three parallel focused sub-queries: wound-type algorithm, dressing mechanism, patient notes |

---

## 2. Results Summary Table

| Version | Strategy | CR ± SD | CP ± SD | HR@6 ± SD | MRR ± SD | NDCG@6 ± SD | R@6 ± SD | P@6 ± SD | Latency (ms) ± SD |
|---|---|---|---|---|---|---|---|---|---|
| **R1-A** | Flat label concat | 0.8244 ± 0.0048 | 0.9228 ± 0.0113 | 0.7812 ± 0.0000 | 0.4906 ± 0.0000 | 0.5660 ± 0.0000 | 0.2439 ± 0.0000 | 0.2188 ± 0.0000 | 40.0 ± 11.4 |
| **R1-B** | Narrative NL query | 0.8483 ± 0.0215 | 0.9137 ± 0.0113 | 0.4688 ± 0.0313 | 0.2371 ± 0.0195 | 0.2956 ± 0.0209 | 0.0921 ± 0.0046 | 0.0816 ± 0.0060 | 1969.4 ± 93.6 |
| **R1-C** | Multi-axis sub-queries | **0.8684 ± 0.0205** | **0.9838 ± 0.0035** | **0.9062 ± 0.0000** | **0.7146 ± 0.0000** | **0.7077 ± 0.0000** | **0.3241 ± 0.0000** | **0.2812 ± 0.0000** | 77.7 ± 2.0 |

> **Bold** = best performance per metric. R1-C leads on all quality metrics. R1-A is fastest.

**Winner: R1-C (Multi-axis sub-queries) — selected as the fixed query strategy for R2 onwards.**

---

## 3. Per-Metric Analysis

### 3.1 Context Recall (CR) — Primary Retrieval Quality Metric

**What it measures:** The proportion of information present in the reference contexts that is covered by the retrieved chunks, as judged by the RAGAS LLM. A high CR means the system retrieved enough of the right content to support accurate generation.

| Version | CR Mean | CR SD | Per-Run CR |
|---|---|---|---|
| R1-A | 0.8244 | 0.0048 | [0.8241, 0.8198, 0.8294] |
| R1-B | 0.8483 | 0.0215 | [0.8668, 0.8533, 0.8247] |
| R1-C | **0.8684** | 0.0205 | [0.8621, 0.8518, 0.8913] |

**Analysis:**

- R1-C achieves the highest mean CR (0.8684), followed by R1-B (0.8483) and R1-A (0.8244). The ordering confirms the hypothesis that semantically richer, decomposed queries retrieve more of the clinically relevant content.
- The gap between R1-A and R1-C is +4.4 percentage points, which is meaningful for a clinical system where missing guideline content could result in incomplete or unsafe recommendations.
- R1-B's CR (0.8483) exceeds R1-A, showing that natural language phrasing extracts more relevant content than keyword concatenation from the embedding space. However, R1-B's higher SD (0.0215 vs 0.0048) indicates greater instability — the GPT-generated query varies across runs, making the system less reproducible.
- R1-A's remarkably low SD (0.0048) is expected: since the query is fully deterministic (a fixed string), the only source of variance is minor floating-point non-determinism in the embedding model. This makes R1-A the most reproducible strategy, but its ceiling is limited by the semantic poverty of flat label concatenation.
- R1-C also has non-trivial SD (0.0205), attributable to the LLM judge's scoring variance across runs rather than retrieval variance (IR metrics in R1-C are perfectly stable across runs: HR@6=0.9062 in all 3 runs).

**Viva note:** A CR of 0.87 on a medical domain task with a domain-specific embedding model (MedEmbed-large-v0.1) and a small, manually curated KB (138 chunks) is a solid result. The 13% shortfall likely reflects cases where reference content spans multiple guideline sources and k=6 is insufficient to cover all relevant chunks in a single retrieval call — this is a known limitation of dense-only retrieval at fixed k, which R2 will address with hybrid search.

---

### 3.2 Context Precision (CP) — Retrieval Noise Metric

**What it measures:** The fraction of retrieved chunks that are actually relevant to the query, as judged by the RAGAS LLM. High CP means the system is not polluting the context window with irrelevant material.

| Version | CP Mean | CP SD | Per-Run CP |
|---|---|---|---|
| R1-A | 0.9228 | 0.0113 | [0.9107, 0.9246, 0.9330] |
| R1-B | 0.9137 | 0.0113 | [0.9266, 0.9055, 0.9089] |
| R1-C | **0.9838** | **0.0035** | [0.9879, 0.9818, 0.9818] |

**Analysis:**

- R1-C is dramatically ahead in CP (+6.1 pp over R1-A, +7.0 pp over R1-B), with an SD of only 0.0035 — the lowest of all three versions. This means R1-C retrieves highly relevant chunks with exceptional consistency.
- The mechanism behind R1-C's precision advantage: by splitting the query into semantically focused sub-queries (wound classification, dressing mechanism, patient notes), each sub-query is narrow and precise. Dense retrieval on a focused query is less likely to surface tangentially related chunks. In contrast, R1-A's flat concatenation string mixes multiple clinical dimensions into a single embedding vector, which tends to retrieve broadly relevant chunks that may not all be needed for a specific case.
- R1-B's CP (0.9137) being slightly below R1-A (0.9228) is an important finding: GPT-generated narratives, while semantically richer, introduce phrasing that can match clinically adjacent but not directly relevant material (e.g., a narrative about "infected exuding wound" might retrieve general infection management chunks not specific to the dressing choice).
- R1-C's CP of ~0.984 is very close to perfect precision — nearly every retrieved chunk is relevant. This has direct implications for generation quality: the LLM generating recommendations will see less noise in its context, reducing the risk of hallucination or recommendation drift.

**Viva note:** The CR–CP tradeoff is a classic retrieval challenge: increasing recall usually comes at the cost of precision (retrieving more chunks means retrieving some irrelevant ones). R1-C achieves **both** higher CR and dramatically higher CP simultaneously. This is unusual and deserves discussion — the multi-axis decomposition effectively addresses both dimensions because each sub-query is targeted, pulling only what it needs without broad overlap.

---

### 3.3 Hit Rate @ K (HR@6) — Binary Retrieval Success

**What it measures:** Whether at least one reference chunk appears anywhere in the top-6 retrieved results. This is the simplest binary measure of whether the retrieval pipeline is working at all.

| Version | HR@6 | SD |
|---|---|---|
| R1-A | 0.7812 | 0.0000 |
| R1-B | 0.4688 | 0.0313 |
| R1-C | **0.9062** | **0.0000** |

**Analysis:**

- R1-B's HR@6 of 0.4688 is alarming at first glance — the narrative query fails to hit any reference chunk in more than half of cases. This is a critical finding that disqualifies R1-B as a standalone retrieval strategy despite its acceptable RAGAS CR scores.
- The explanation for this apparent contradiction (decent CR but poor HR): RAGAS CR is LLM-judge based and measures semantic coverage, not exact chunk-level matching. When the GPT-generated narrative is semantically close to the reference content but uses different phrasing from the KB, the RAGAS judge may award partial credit to paraphrased content in adjacent chunks — but those chunks would not be counted as hits by the chunk-level exact-match HR metric.
- R1-A's HR@6 of 0.7812 means 7 out of 32 cases had no reference chunk in the top 6, which is a moderate performance. For a clinical decision support system, one in four cases failing to retrieve any reference material is a meaningful limitation.
- R1-C's HR@6 of 0.9062 means only 3 out of 32 cases fail to hit a reference chunk. The zero SD across all three runs (0.9062 exactly each time) confirms this is a stable property of the multi-axis query architecture, not a lucky run.

---

### 3.4 Mean Reciprocal Rank (MRR) — Ranking Quality

**What it measures:** For each query, 1/rank of the first relevant chunk. Averaged across all cases. A score of 1.0 means the top result is always relevant; 0.5 means the first relevant result is on average ranked second.

| Version | MRR | SD |
|---|---|---|
| R1-A | 0.4906 | 0.0000 |
| R1-B | 0.2371 | 0.0195 |
| R1-C | **0.7146** | **0.0000** |

**Analysis:**

- R1-C's MRR of 0.714 means the first relevant chunk is, on average, ranked between 1st and 2nd position. This is excellent ranking behaviour for a medical domain retriever using dense-only search.
- R1-A's MRR of 0.491 means the first relevant chunk is on average ranked between 2nd and 3rd. Still acceptable, but the flat query formulation means relevant chunks must compete with many similarly ranked chunks.
- R1-B's MRR of 0.237 is poor — the narrative query, despite its semantic richness, consistently fails to rank reference chunks at the top. This likely reflects a mismatch between the conversational embedding space of GPT-generated text and the clinical/technical language of the guideline chunks. MedEmbed-large-v0.1 was trained on medical literature, not patient-facing narrative queries.
- The large gap between R1-C and R1-A in MRR (+0.224) has a direct quality implication for generation: when the most relevant chunk ranks higher, it receives more weight in the context window (particularly when contexts are ordered by score), and the generation LLM is more likely to ground its output in the best evidence first.

---

### 3.5 NDCG @ K — Graded Ranking Quality

**What it measures:** Normalised Discounted Cumulative Gain — a comprehensive ranking metric that rewards relevant chunks ranked higher, with diminishing returns for lower positions. Scores range from 0 to 1, where 1 is perfect ranking.

| Version | NDCG@6 | SD |
|---|---|---|
| R1-A | 0.5660 | 0.0000 |
| R1-B | 0.2956 | 0.0209 |
| R1-C | **0.7077** | **0.0000** |

**Analysis:**

- NDCG@6 confirms the ordering established by MRR: R1-C > R1-A >> R1-B. The gap between R1-C (0.708) and R1-A (0.566) is +0.142, which is a large and practically meaningful difference in ranking quality.
- R1-B's NDCG of 0.296 — barely above random at 0.25 for a binary relevance judgement — is consistent with its poor HR and MRR. The narrative query approach fundamentally fails the ranking objective in this domain.
- R1-C's zero SD on NDCG (same score across all 3 runs) reinforces that the multi-axis retrieval architecture is deterministic at the IR metric level, even though RAGAS scores vary slightly due to LLM judge stochasticity.

---

### 3.6 Recall @ K and Precision @ K — Chunk-Level Coverage and Noise

| Version | R@6 | SD | P@6 | SD |
|---|---|---|---|---|
| R1-A | 0.2439 | 0.0000 | 0.2188 | 0.0000 |
| R1-B | 0.0921 | 0.0046 | 0.0816 | 0.0060 |
| R1-C | **0.3241** | **0.0000** | **0.2812** | **0.0000** |

**Analysis:**

- Recall@6 measures what fraction of all reference chunks are actually retrieved in the top 6. R1-C covers 32.4% of reference chunks, compared to 24.4% for R1-A. Given k=6 and a testset where reference_contexts typically contain 3 chunks per case, a theoretical maximum recall at k=6 with 3 reference chunks is 3/3 = 1.0 per case, but aggregate recall is bounded by the total number of reference chunks across all cases. R1-C's advantage suggests its sub-queries are each retrieving distinct relevant chunks rather than repeatedly retrieving the same ones.
- R1-B's R@6 of 0.092 is extremely low — confirming that narrative queries, while producing acceptable RAGAS scores, fail to directly retrieve the reference chunks used to construct those RAGAS scores.
- Precision@6 for R1-C (0.281) exceeds R1-A (0.219), meaning not only does R1-C retrieve more reference chunks, it also does so with less wasted slot capacity.

---

### 3.7 Retrieval Latency — Deployment Feasibility

| Version | Mean Latency (ms) | SD (ms) | Latency Ratio vs R1-A |
|---|---|---|---|
| R1-A | **40.0** | 11.4 | 1.0× (baseline) |
| R1-B | 1969.4 | 93.6 | **49.2×** |
| R1-C | 77.7 | 2.0 | 1.9× |

**Analysis:**

- R1-A is the fastest strategy at 40 ms mean latency, reflecting its deterministic, zero-LLM-call architecture: the TIME labels are concatenated into a string and passed directly to the vector store.
- R1-B's 1969 ms latency represents the GPT-4o-mini API call overhead for generating the narrative query. This makes R1-B unacceptable for mobile deployment — even before generation, the system already spends ~2 seconds on query formulation alone, and SD of 93.6 ms indicates significant network-dependent variability.
- R1-C at 77.7 ms is only 1.9× slower than R1-A despite issuing 3 parallel sub-queries to ChromaDB. This confirms that the latency cost of multi-axis retrieval is minimal — the parallel ChromaDB queries are fast, and no LLM call is needed at query formulation time. For a mobile wound care app targeting <10 seconds end-to-end latency, an additional 37.7 ms for substantially better retrieval quality is an excellent trade-off.
- R1-C's very low SD (2.0 ms) indicates highly stable latency — important for a patient-facing clinical app where response time predictability matters as much as mean latency.

---

## 4. Cross-Metric Comparison — Radar Summary

```
Metric              R1-A    R1-B    R1-C    Winner
─────────────────────────────────────────────────────
Context Recall      0.8244  0.8483  0.8684  R1-C  ✓
Context Precision   0.9228  0.9137  0.9838  R1-C  ✓
Hit Rate @ 6        0.7812  0.4688  0.9062  R1-C  ✓
MRR                 0.4906  0.2371  0.7146  R1-C  ✓
NDCG @ 6            0.5660  0.2956  0.7077  R1-C  ✓
Recall @ 6          0.2439  0.0921  0.3241  R1-C  ✓
Precision @ 6       0.2188  0.0816  0.2812  R1-C  ✓
Latency (ms)          40.0  1969.4    77.7  R1-A  ✗
─────────────────────────────────────────────────────
R1-C wins 7/8 metrics. R1-A wins latency only.
```

---

## 5. Hypothesis Evaluation

The pre-registered hypothesis stated:

> *"Multi-axis sub-query (R1-C) should outperform flat concat (R1-A) because each sub-query is semantically focused on a specific aspect of the clinical scenario. Narrative query (R1-B) should outperform flat concat but lag behind multi-axis."*

**Evaluation:**

| Prediction | Outcome | Confirmed? |
|---|---|---|
| R1-C > R1-A on CR | 0.8684 > 0.8244 (+4.4 pp) | ✅ |
| R1-C > R1-A on CP | 0.9838 > 0.9228 (+6.1 pp) | ✅ |
| R1-C > R1-A on ranking metrics | MRR: 0.7146 > 0.4906 | ✅ |
| R1-B > R1-A on CR | 0.8483 > 0.8244 | ✅ |
| R1-B < R1-C on CR | 0.8483 < 0.8684 | ✅ |
| R1-B ranking between R1-A and R1-C | HR@6: 0.4688 — BELOW R1-A (0.7812) | ❌ Partially refuted |

**Summary:** The hypothesis is substantially confirmed for RAGAS metrics. The partial refutation is notable: **R1-B was expected to rank between R1-A and R1-C on IR metrics, but it performs significantly worse than R1-A on Hit Rate, MRR, and NDCG.** This suggests that LLM-generated narrative queries are semantically richer in meaning captured by RAGAS CR (which measures semantic overlap at the sentence level) but create embedding vectors that are misaligned with the clinical technical language of the KB chunks. This is a domain mismatch effect — MedEmbed-large-v0.1 is tuned for medical literature language, not conversational narrative style.

---

## 6. Key Findings and Insights

### Finding 1: The CR–CP Decoupling Paradox in R1-B

R1-B achieves a reasonable CR (0.8483) but catastrophic HR@6 (0.4688) and R@6 (0.0921). This apparent paradox — good semantic coverage but poor direct hit rate — reveals a fundamental difference between how RAGAS CR and chunk-level IR metrics evaluate retrieval.

RAGAS CR uses an LLM to assess whether the claims in the reference answer are supported by the retrieved contexts, judging semantic meaning. This allows partial credit when a retrieved chunk paraphrases a reference chunk in different words. Chunk-level IR metrics require the exact reference chunk to appear in the retrieved set. R1-B's narrative queries retrieve *semantically adjacent but not reference-identical* chunks — good enough for the LLM judge to award credit, but not good enough to place the actual reference chunks in the top 6. For a clinical safety system, this distinction matters: the actual guideline content (with specific dressing names, dose thresholds, and contraindications) must be retrieved, not just semantically similar paraphrases.

### Finding 2: Multi-Axis Decomposition Resolves the Retrieval Conflict

Dense retrieval with a single embedding vector faces an inherent tension when the query encodes multiple clinical dimensions simultaneously: the vector becomes a compromise representation that is moderately similar to many chunks but maximally similar to none. R1-C resolves this by decomposing the query into three independent vectors, each optimised for one clinical dimension. The result is that each sub-query finds its own best-matching chunks, and the union of these matches is both broader (higher CR) and cleaner (higher CP) than any single-vector approach.

### Finding 3: Latency is Acceptable for Mobile Deployment

R1-C at 77.7 ms is well within feasibility for mobile deployment. Given that the generation step (GPT-4o-mini) will add ~1–3 seconds, and the pre-processing pipeline (wound detection, segmentation, classification) adds additional time, the retrieval step contributes less than 1% of the total expected response time. R1-B's 1969 ms latency, by contrast, would contribute ~20% of the total expected response time — an unacceptable overhead for a patient-facing app.

### Finding 4: R1-C's IR Metrics are Perfectly Reproducible

R1-C produces identical HR@6, MRR, NDCG, R@6, and P@6 across all 3 runs (SD = 0.0000 for all). This occurs because the multi-axis sub-queries are fully deterministic (no LLM call), and ChromaDB cosine similarity retrieval is deterministic for a fixed embedding model and fixed DB. The RAGAS CR and CP scores vary slightly across runs (due to LLM judge stochasticity), but the underlying retrieval is perfectly stable. This is an important reliability property for a clinical system.

---

## 7. Per-Case Diagnostic Patterns

### Cases with Consistently Low CR Across All Versions

Examining per-case CR data reveals cases that are systematically hard to retrieve across all strategies. These are cases where the reference contexts may be distributed across multiple guideline sources, or where the wound category is underrepresented in the KB.

**Challenging cases (R1-C Run 1 per-case CR < 0.20):**
- Case 27 (index 26): CR = 0.059 — consistently near-zero across all versions and runs. This suggests the reference content for this case is either from a guideline source with limited KB coverage, or the TIME inputs for this case produce queries that don't align well with the chunk vocabulary.
- Case 19 (index 18): CR = 0.154 in R1-C Run 1, but 0.75 in Run 3 — high variability. The RAGAS LLM judge is uncertain about this case.

**Implications for R2:** Cases with persistently low CR under dense-only retrieval are prime candidates for improvement with hybrid retrieval (BM25 can catch exact dressing name or guideline-specific term matches that semantic search misses). These hard cases should be monitored specifically in R2 analysis.

---

## 8. Discussion — Implications for VerdaSense Architecture

### 8.1 Why Multi-Axis Sub-Queries Work for Structured Clinical Inputs

The T.I.M.E. framework naturally decomposes wound assessment into independent clinical axes: tissue composition, infection status, moisture balance, and wound edge behaviour. These axes correspond to distinct knowledge dimensions in wound care guidelines: tissue management, infection control, moisture management, and wound progression assessment. By generating one sub-query per clinical axis (plus a patient notes sub-query), R1-C mirrors the epistemic structure of the guidelines it is querying.

This is an architectural choice specific to VerdaSense's structured input design. It would not be applicable to a free-text clinical Q&A system (like the German nursing paper) where the user's question is already a unified natural language expression. This specificity is part of VerdaSense's original contribution.

### 8.2 R1-B as a Warning Against LLM-in-the-Loop at Query Time

R1-B's failure at IR metrics despite acceptable RAGAS scores is a cautionary finding. Adding an LLM call to "improve" the query introduces three problems: (1) latency increases by ~50×; (2) the generated narrative may use vocabulary that creates a domain mismatch with the embedding model; and (3) the system becomes non-deterministic and harder to debug. For a clinical system where reproducibility and auditability matter, LLM-generated queries introduce an uncontrolled variable. R1-C achieves better results with zero LLM overhead.

### 8.3 Connecting to the Hybrid Retrieval Hypothesis (R2)

Despite R1-C's strong performance, a CR of 0.868 means ~13% of reference information is not retrieved. The most likely causes are: (a) reference chunks from sources with lower semantic similarity to the query embedding (vocabulary mismatch that BM25 could resolve); and (b) some reference content is in chunks ranked 7th or beyond (addressable by increasing k, which R3 will study). R2's hybrid retrieval experiment will add BM25 to R1-C's multi-axis sub-queries, testing whether lexical matching (exact guideline terminology: "alginate", "hydrofibre", "debridement") can recover the remaining 13% recall deficit.

---

## 9. Decision: R1-C Selected for R2

**R1-C (Multi-axis sub-queries) is selected as the fixed query strategy for all subsequent experiments.**

Justification:
- Best performance on 7 of 8 evaluation metrics
- Dominates on the two most critical clinical retrieval metrics: CR (+4.4 pp over R1-A) and CP (+6.1 pp over R1-A)
- Latency penalty over R1-A is negligible (37.7 ms additional) and does not affect mobile deployment feasibility
- Perfectly reproducible IR metrics (SD = 0 across 3 runs)
- Eliminates the domain mismatch risk identified in R1-B
- Architecture aligns with the T.I.M.E. framework's natural clinical decomposition

---

## 10. What to Fix in R1-C Before R2

The following implementation note should be carried forward: R1-C retrieves 5 chunks instead of 6 for one specific case category (cat_a_type8_wet_infected_necrotic, visible in run logs). This occurs when the three sub-queries return fewer than 6 unique chunks after deduplication. In R2, ensure the deduplication logic preserves k=6 by adding fallback single-query retrieval when sub-queries return fewer than k total unique chunks.

---

## 11. Summary for FYP Viva

**One-sentence answer to the R1 research question:**

> The way T.I.M.E. inputs are formulated into retrieval queries significantly affects retrieval quality: multi-axis sub-query decomposition (R1-C) outperforms both flat keyword concatenation (R1-A) and LLM-generated narrative queries (R1-B) across all quality metrics while adding only 37.7 ms retrieval overhead, making it the optimal query strategy for a structured clinical input RAG system.

**Three things to remember for the viva:**

1. **R1-B's paradox** — good RAGAS CR but terrible HR@6 — explains why LLM-judged semantic metrics and chunk-level IR metrics measure different things and why both are needed.
2. **R1-C's simultaneous CR and CP improvement** is unusual in retrieval (normally a tradeoff); the decomposition strategy achieves it because each sub-query is semantically narrow and precise.
3. **R1's original contribution** lies in studying structured TIME input query formulation, a problem not addressed by any existing RAG-for-clinical-guidelines literature.

---

*Generated: 11 May 2026 | VerdaSense FYP — Universiti Malaya*  
*Next: R2 — Retrieval Strategy Ablation (Dense vs Sparse vs Hybrid vs Hybrid+Reranking) using R1-C fixed query strategy*
