# VerdaSense RAG — Experiment R2: Retrieval Strategy Ablation
## Comprehensive Analysis & Discussion

**Experiment:** R2 — Retrieval Strategy (Dense / Sparse / Hybrid / Hybrid+Reranking)  
**Stage:** 1 — Retrieval Ablation  
**Date:** 12–13 May 2026  
**Configuration:** k=6 | MedEmbed-large-v0.1 | R1-C multi-axis sub-queries (fixed) | 3 runs each  
**RAGAS Judge:** gpt-4o-mini + text-embedding-3-small (fixed across all versions)

---

## 1. Experiment Overview

Experiment R2 investigates the core retrieval architecture question: **does combining or augmenting dense semantic search with additional retrieval signals improve the quality of wound care guideline retrieval?**

R1 established that multi-axis sub-query decomposition (R1-C) is the optimal query strategy, achieving a context recall of 0.868 but leaving ~13% of reference information unretrieved. The ablation study map hypothesised that this shortfall is attributable to vocabulary mismatch — cases where exact clinical terminology (e.g., "alginate", "hydrofibre", "debridement") is present in the reference chunks but not maximally aligned in the embedding space. BM25 lexical search should be able to recover such cases.

Five versions were tested under identical conditions (same KB, same testset, same R1-C query strategy, same k=6):

| Version | Label | Retrieval Method | Reranker |
|---|---|---|---|
| **R2-A** | Dense only | ChromaDB cosine similarity (baseline — replication of R1-C) | None |
| **R2-B** | Sparse only (BM25) | BM25Retriever on all KB documents | None |
| **R2-C** | Hybrid Dense+BM25 (RRF) | EnsembleRetriever with Reciprocal Rank Fusion | None |
| **R2-D-BGE** | Hybrid + Rerank (BGE-v2-m3) | Hybrid pool → BAAI/bge-reranker-v2-m3 | BAAI/bge-reranker-v2-m3 (multilingual) |
| **R2-D-MiniLM-L6** | Hybrid + Rerank (MiniLM-L6) | Hybrid pool → cross-encoder/ms-marco-MiniLM-L-6-v2 | MiniLM-L6 (web-domain) |
| **R2-D-MiniLM-L12** | Hybrid + Rerank (MiniLM-L12) | Hybrid pool → cross-encoder/ms-marco-MiniLM-L-12-v2 | MiniLM-L12 (web-domain, larger) |

R2-A serves as the direct baseline, reproducing the R1-C dense-only configuration. The R2-D variants test three different reranking models spanning medical-adjacent multilingual (BGE-v2-m3) and web-domain-trained cross-encoders (MiniLM-L6/L12), allowing the reranker domain mismatch hypothesis — first identified in the earlier ablation study — to be tested systematically.

---

## 2. Results Summary Table

| Version | Strategy | CR ± SD | CP ± SD | HR@6 ± SD | MRR ± SD | NDCG@6 ± SD | R@6 ± SD | P@6 ± SD | Latency (ms) ± SD |
|---|---|---|---|---|---|---|---|---|---|
| **R2-A** | Dense only | 0.8803 ± 0.0038 | 0.9720 ± 0.0035 | 0.9062 ± 0.0000 | 0.7146 ± 0.0000 | 0.7077 ± 0.0000 | 0.3241 ± 0.0000 | 0.2812 ± 0.0000 | 97.1 ± 13.3 |
| **R2-B** | Sparse only (BM25) | **0.9062 ± 0.0044** | **0.9939 ± 0.0045** | 0.7188 ± 0.0000 | 0.3109 ± 0.0000 | 0.4091 ± 0.0000 | 0.1762 ± 0.0000 | 0.1562 ± 0.0000 | **1.8 ± 0.3** |
| **R2-C** | Hybrid (Dense+BM25, RRF) | 0.8700 ± 0.0167 | 0.9797 ± 0.0023 | 0.7812 ± 0.0000 | 0.6729 ± 0.0000 | 0.6589 ± 0.0000 | 0.1969 ± 0.0000 | 0.1719 ± 0.0000 | 154.5 ± 10.6 |
| **R2-D-BGE** | Hybrid + Rerank (BGE-v2-m3) | 0.8740 ± 0.0224 | **1.0000 ± 0.0000** | **0.8125 ± 0.0000** | **0.7214 ± 0.0000** | 0.7045 ± 0.0000 | 0.2214 ± 0.0000 | 0.1927 ± 0.0000 | 2801.3 ± 31.0 |
| **R2-D-MiniLM-L6** | Hybrid + Rerank (MiniLM-L6) | 0.8523 ± 0.0329 | 0.9447 ± 0.0034 | 0.8125 ± 0.0000 | 0.6896 ± 0.0000 | **0.7146 ± 0.0000** | 0.1706 ± 0.0000 | 0.1406 ± 0.0000 | 241.0 ± 8.1 |
| **R2-D-MiniLM-L12** | Hybrid + Rerank (MiniLM-L12) | 0.8047 ± 0.0176 | 0.9359 ± 0.0053 | 0.7812 ± 0.0000 | 0.7578 ± 0.0000 | 0.7635 ± 0.0000 | 0.1562 ± 0.0000 | 0.1302 ± 0.0000 | 778.2 ± 32.0 |

> **Bold** = best performance per metric. SD = population standard deviation across 3 runs.

**Note on IR metric SD:** All IR metrics (HR@6 through P@6) show SD = 0.0000 across 3 runs for all versions. This is expected: the retrieval pipelines are fully deterministic for a fixed embedding model and KB. The only source of variance is the RAGAS LLM judge (GPT-4o-mini), which explains non-zero CR and CP standard deviations.

---

## 3. Per-Metric Analysis

### 3.1 Context Recall (CR) — Primary Retrieval Quality Metric

**What it measures:** The proportion of information in the reference contexts that is covered by the retrieved chunks, as judged by the RAGAS LLM. A high CR means the system retrieved enough of the right content to support accurate generation. This is the most clinically important retrieval metric — missing reference information at retrieval means it cannot appear in the generated recommendation.

| Version | CR Mean | CR SD | Per-Run CR |
|---|---|---|---|
| R2-A | 0.8803 | 0.0038 | [0.8815, 0.8833, 0.876] |
| R2-B | **0.9062** | 0.0044 | [0.9054, 0.9023, 0.911] |
| R2-C | 0.8700 | 0.0167 | [0.8807, 0.8507, 0.8785] |
| R2-D-BGE | 0.8740 | 0.0224 | [0.8914, 0.8487, 0.882] |
| R2-D-MiniLM-L6 | 0.8523 | 0.0329 | [0.8338, 0.8903, 0.8327] |
| R2-D-MiniLM-L12 | 0.8047 | 0.0176 | [0.8244, 0.7992, 0.7906] |

**Analysis:**

The CR ranking reveals a counter-intuitive result that is one of the most important findings in R2: **BM25 alone (R2-B, CR = 0.9062) substantially outperforms every other strategy on context recall, including the dense baseline (R2-A, CR = 0.8803) by +2.6 pp, and all hybrid and reranking variants.**

More surprisingly, hybrid retrieval (R2-C, CR = 0.8700) underperforms the dense-only baseline (R2-A, CR = 0.8803) by −1.0 pp. This directly contradicts the ablation map's hypothesis that BM25 would complement dense retrieval to recover the 13% recall deficit identified in R1. The RRF fusion mechanism, which was expected to combine the strengths of both modalities, instead produced a result worse than either modality alone on the primary recall metric.

All three reranking variants follow a clear pattern: reranking consistently depresses CR relative to the hybrid pool it operates on. The CR degradation intensifies with web-domain rerankers. R2-D-BGE (CR = 0.8740) causes the least damage, while R2-D-MiniLM-L12 (CR = 0.8047) reduces recall to the lowest value observed in the entire experiment — a −7.6 pp drop from the BM25 baseline.

Standard deviation is particularly informative: R2-A shows very low SD (0.0038), consistent with the deterministic dense retrieval producing stable chunks and only LLM judge variance contributing to CR fluctuation. R2-D-MiniLM-L6 shows the highest SD (0.0329), indicating that the MiniLM-L6 reranker's instability propagates through to the RAGAS judge's assessment — a reproducibility concern for a clinical system.

**Viva note:** A CR of 0.906 for BM25-only on a wound care knowledge base is a strong and somewhat unexpected result. The mechanism is likely that wound care guidelines are highly terminology-specific: they contain frequent exact matches for clinical terms (alginate, hydrofibre, Aquacel, NPWT, debridement, maceration) that appear verbatim in both the testset reference contexts and the KB chunks. BM25's IDF weighting makes it highly sensitive to low-frequency, high-specificity clinical terms — exactly the type of terms that are most diagnostically discriminative in wound care. This is a domain-specific finding worth discussing in the FYP context.

---

### 3.2 Context Precision (CP) — Retrieval Noise Metric

**What it measures:** The fraction of retrieved chunks that are actually relevant to the query, as judged by the RAGAS LLM. High CP means the system is not polluting the LLM context window with irrelevant material that could distract the generation model or increase hallucination risk.

| Version | CP Mean | CP SD | Per-Run CP |
|---|---|---|---|
| R2-A | 0.9720 | 0.0035 | [0.9701, 0.9699, 0.9761] |
| R2-B | 0.9939 | 0.0045 | [0.999, 0.9908, 0.9918] |
| R2-C | 0.9797 | 0.0023 | [0.9823, 0.9784, 0.9784] |
| **R2-D-BGE** | **1.0000** | **0.0000** | [1.0, 1.0, 1.0] |
| R2-D-MiniLM-L6 | 0.9447 | 0.0034 | [0.9426, 0.9428, 0.9486] |
| R2-D-MiniLM-L12 | 0.9359 | 0.0053 | [0.936, 0.9411, 0.9305] |

**Analysis:**

R2-D-BGE achieves a perfect CP of 1.0000 across all three runs (SD = 0.0000), a remarkable result indicating that after BGE-v2-m3 reranking, every single retrieved chunk is judged relevant by the RAGAS LLM with perfect consistency. This is the only version in the experiment to achieve perfect context precision.

R2-B (BM25-only) is close behind at CP = 0.9939, which is itself an impressive result for a lexical retriever with no semantic awareness. This indicates that exact keyword matches in a specialised clinical KB tend to be highly relevant — the vocabulary of wound care guidelines is specific enough that BM25 rarely retrieves off-topic chunks.

The web-domain MiniLM rerankers (L6 and L12) are the only versions that reduce CP below the dense baseline. R2-D-MiniLM-L6 (CP = 0.9447) and R2-D-MiniLM-L12 (CP = 0.9359) both score below R2-A (CP = 0.9720). This confirms the reranker domain mismatch hypothesis from the ablation map: cross-encoders trained on web-domain MS MARCO data are unreliable judges of clinical chunk relevance. Their relevance scoring pushes clinically appropriate chunks down the ranking and surfaces chunks that are relevant by web-domain criteria but not by clinical-domain criteria.

A CP below 0.94 (as seen in both MiniLM variants) means roughly 6% of retrieved context tokens contain material that is irrelevant to the query — in a k=6 context window, this corresponds to approximately one out of every 16 chunks being noise. For a patient-facing clinical system, this level of context pollution is a meaningful risk factor for hallucination or recommendation drift.

**Viva note:** The CR–CP relationship in R2 is more nuanced than the classic retrieval tradeoff. R2-B achieves simultaneously the best CR and second-best CP. R2-D-BGE achieves perfect CP while not sacrificing CR dramatically (−0.6 pp relative to the dense baseline). The only versions that degrade both CR and CP relative to the dense baseline are the web-domain MiniLM rerankers — a consistent double-failure pattern that is the clearest negative finding in R2.

---

### 3.3 Hit Rate @ 6 (HR@6) — Binary Retrieval Success

**What it measures:** Whether at least one reference chunk appears anywhere in the top-6 retrieved results. This is the simplest binary measure of whether the retrieval pipeline is working at all.

| Version | HR@6 | Per-Run HR@6 |
|---|---|---|
| **R2-A** | **0.9062** | [0.9062, 0.9062, 0.9062] |
| R2-B | 0.7188 | [0.7188, 0.7188, 0.7188] |
| R2-C | 0.7812 | [0.7812, 0.7812, 0.7812] |
| R2-D-BGE | 0.8125 | [0.8125, 0.8125, 0.8125] |
| R2-D-MiniLM-L6 | 0.8125 | [0.8125, 0.8125, 0.8125] |
| R2-D-MiniLM-L12 | 0.7812 | [0.7812, 0.7812, 0.7812] |

**Analysis:**

The HR@6 ranking is the inverse of the CR ranking, and this divergence is the key diagnostic finding of R2. Dense-only retrieval (R2-A) achieves the highest HR@6 (0.9062), meaning it successfully retrieves at least one reference chunk in 29 of 32 cases. BM25 (R2-B) has the lowest HR@6 (0.7188), successfully retrieving at least one reference chunk in only 23 of 32 cases.

How can BM25 simultaneously have the best CR (0.906) and the worst HR@6 (0.719)? This apparent paradox resolves as follows: RAGAS CR is an LLM-judged semantic metric that awards partial credit when a retrieved chunk *expresses the same clinical information* as a reference chunk, even if it is not the identical chunk. BM25 retrieves chunks that contain the same keywords as the reference but may be from different sections or sources — semantically equivalent for the purposes of answering the clinical question, but not chunk-identical. HR@6 requires an exact chunk match.

This distinction has important practical implications: if the downstream generation task only requires that the retrieved context *contains the information needed* to generate a correct recommendation, BM25's high CR (0.906) is the relevant metric. If the system is required to cite specific guideline passages by chunk ID (e.g., for regulatory traceability), HR@6 is the relevant metric and dense retrieval (R2-A) is superior.

For VerdaSense's patient-facing recommendation use case, CR is the primary metric — the patient receives a recommendation derived from the retrieved information, and the specific chunk provenance is not visible to the patient. This favours R2-B's CR-first profile.

---

### 3.4 Mean Reciprocal Rank (MRR) — Ranking Quality

**What it measures:** The average of 1/(rank of the first relevant chunk) across all queries. MRR rewards configurations where the first relevant chunk is ranked near the top of the retrieved list. An MRR of 1.0 means the first result is always relevant; 0.5 means the first relevant result is typically ranked second.

| Version | MRR | Per-Run MRR |
|---|---|---|
| R2-A | 0.7146 | [0.7146, 0.7146, 0.7146] |
| R2-B | 0.3109 | [0.3109, 0.3109, 0.3109] |
| R2-C | 0.6729 | [0.6729, 0.6729, 0.6729] |
| R2-D-BGE | 0.7214 | [0.7214, 0.7214, 0.7214] |
| R2-D-MiniLM-L6 | 0.6896 | [0.6896, 0.6896, 0.6896] |
| **R2-D-MiniLM-L12** | **0.7578** | [0.7578, 0.7578, 0.7578] |

**Analysis:**

MRR reveals a sharply different picture from CR and HR@6. BM25 (R2-B, MRR = 0.3109) is dramatically the worst performer on ranking quality, with its first relevant chunk typically appearing around rank 3 or lower. This is consistent with BM25's known weakness: it can retrieve the right information but ranks it poorly because its IDF weighting is agnostic to semantic relevance ordering. For a clinical system where the LLM reads chunks in order and may be influenced by what appears first in the context window, poor MRR is a meaningful risk.

R2-D-MiniLM-L12 achieves the best MRR (0.7578), and R2-D-BGE is close (0.7214). This is where rerankers show their design-intended value: cross-encoders perform joint relevance scoring of the query against each candidate chunk, which produces better relative ranking even when they suppress some relevant chunks from the top-K set (explaining the CR penalty).

Notably, R2-A (MRR = 0.7146) and R2-D-BGE (MRR = 0.7214) are nearly identical on ranking quality, suggesting BGE-v2-m3's reranking does not substantially improve the ordering that MedEmbed dense retrieval already achieves — but it does reduce the noise at the tail of the ranked list (as evidenced by its perfect CP).

---

### 3.5 NDCG @ 6 — Comprehensive Ranking Quality

**What it measures:** Normalised Discounted Cumulative Gain at k=6. NDCG rewards configurations where relevant chunks are ranked higher, with diminishing credit for relevant chunks ranked lower. Unlike MRR (which only considers the first relevant chunk), NDCG accounts for all relevant chunks across the ranked list. This is the most comprehensive single ranking quality metric.

| Version | NDCG@6 | Per-Run NDCG@6 |
|---|---|---|
| R2-A | 0.7077 | [0.7077, 0.7077, 0.7077] |
| R2-B | 0.4091 | [0.4091, 0.4091, 0.4091] |
| R2-C | 0.6589 | [0.6589, 0.6589, 0.6589] |
| R2-D-BGE | 0.7045 | [0.7045, 0.7045, 0.7045] |
| R2-D-MiniLM-L6 | 0.7146 | [0.7146, 0.7146, 0.7146] |
| **R2-D-MiniLM-L12** | **0.7635** | [0.7635, 0.7635, 0.7635] |

**Analysis:**

NDCG confirms the MRR findings and extends them. R2-D-MiniLM-L12 leads on NDCG (0.7635), followed by MiniLM-L6 (0.7146), then R2-A (0.7077) and R2-D-BGE (0.7045). BM25 (0.4091) is again the clear last-place finisher on ranking quality.

The near-identical NDCG of R2-D-BGE (0.7045) and R2-A (0.7077) is striking: BGE reranking achieves near-perfect CP (1.0) and marginally better MRR (0.7214 vs 0.7146) while maintaining essentially equivalent NDCG. This suggests BGE-v2-m3 is pruning irrelevant chunks from the retrieved set without substantially reorganising the relative order of the relevant ones — a precision-cleaning operation rather than a ranking-improvement operation.

The MiniLM models show a clear capacity effect on NDCG: MiniLM-L12 (0.7635) outperforms MiniLM-L6 (0.7146), and L12 also outperforms the BGE reranker (0.7045). However, this NDCG advantage for MiniLM-L12 is entirely nullified when CR is considered: MiniLM-L12's CR (0.8047) is by far the worst in the experiment, meaning it ranks retrieved chunks well but is suppressing relevant chunks from the top-6 set.

---

### 3.6 Recall @ 6 and Precision @ 6 — Chunk-Level Coverage Metrics

**What they measure:** R@6 = fraction of all reference chunks found anywhere in the top-6 retrieved results. P@6 = fraction of the top-6 retrieved chunks that are reference chunks. These are direct chunk-identity measures complementing RAGAS CR/CP.

| Version | R@6 | P@6 |
|---|---|---|
| **R2-A** | **0.3241** | **0.2812** |
| R2-B | 0.1762 | 0.1562 |
| R2-C | 0.1969 | 0.1719 |
| R2-D-BGE | 0.2214 | 0.1927 |
| R2-D-MiniLM-L6 | 0.1706 | 0.1406 |
| R2-D-MiniLM-L12 | 0.1562 | 0.1302 |

**Analysis:**

Dense retrieval (R2-A) dominates both chunk-level metrics by a large margin: R@6 = 0.324 vs 0.176 for BM25 and 0.157 for MiniLM-L12. This is the mirror image of the CR finding: dense retrieval is much better at returning the exact reference chunks, while BM25 returns equivalent-information chunks that are not chunk-identical.

All reranking variants score below R2-A on both R@6 and P@6, which confirms that rerankers suppress some exact-match reference chunks from the top-6 set. This is consistent with the CR findings — the reranker substitutes an alternative relevant chunk for the exact reference chunk, which is invisible to RAGAS CR but penalises chunk-level IR metrics.

The absolute R@6 values across all versions (0.157–0.324) are low because the testset reference contexts often comprise 3–8 reference chunks per case, while the retrieval pool covers 138 chunks across 8 guideline sources. A Recall@6 of 0.324 means R2-A retrieves approximately 2 of every 6 reference chunks — consistent with a k=6 ceiling on a multi-source reference set. This is an inherent limitation of fixed-k retrieval, which R3 will investigate by varying k.

---

### 3.7 Retrieval Latency — Mobile Deployment Feasibility

**What it measures:** Wall-clock time for the retrieval step only (not generation). For a patient-facing mobile app, the practical threshold for acceptable total response time is under 10,000 ms, of which retrieval should ideally be under 500 ms to leave sufficient budget for generation.

| Version | Latency Mean (ms) | Latency SD (ms) | Per-Run Latency (ms) |
|---|---|---|---|
| **R2-B** | **1.8** | **0.3** | [1.5, 1.7, 2.1] |
| R2-A | 97.1 | 13.3 | [111.7, 85.7, 93.8] |
| R2-C | 154.5 | 10.6 | [165.9, 145.0, 152.7] |
| R2-D-MiniLM-L6 | 241.0 | 8.1 | [239.6, 249.8, 233.7] |
| R2-D-MiniLM-L12 | 778.2 | 32.0 | [807.5, 744.0, 783.0] |
| R2-D-BGE | 2801.3 | 31.0 | [2768.9, 2804.4, 2830.6] |

**Analysis:**

BM25 is extraordinarily fast (1.8 ms), as expected for an in-memory TF-IDF index with no embedding computation. Dense retrieval (97.1 ms) is already acceptable, but BGE-v2-m3 reranking at 2801.3 ms is a serious deployment concern — nearly 3 seconds for retrieval alone, before any LLM generation call. Given that generation with GPT-4o-mini typically adds 1–3 seconds, BGE reranking would push the total expected response time to 4–6 seconds, which is within the 10-second feasibility threshold but leaves minimal budget for the CV pipeline processing upstream.

MiniLM-L6 (241 ms) and MiniLM-L12 (778 ms) are well within the retrieval latency budget. However, given that both MiniLM variants degrade CR relative to the dense baseline, their latency cost buys no recall improvement — making them poor value for the system.

**Viva note:** All versions except R2-D-BGE are within the 500 ms retrieval budget for mobile deployment. R2-D-BGE's 2.8-second retrieval latency is a practical concern but not a disqualifying one — whether it is acceptable depends on the total system latency budget once the full pipeline (CV + retrieval + generation) is measured. This should be flagged in the discussion as a deployment tradeoff rather than a hard failure.

---

## 4. Cross-Version Comparison: The Dominant Metric Divergence

The most important analytical observation in R2 is the **systematic divergence between RAGAS semantic metrics (CR, CP) and chunk-level IR metrics (HR@6, R@6)**. This divergence is not random noise — it is a structural finding with a clear mechanistic explanation.

| Metric family | Best version | Worst version | Gap |
|---|---|---|---|
| RAGAS CR (semantic recall) | R2-B (0.9062) | R2-D-MiniLM-L12 (0.8047) | −10.2 pp |
| RAGAS CP (semantic precision) | R2-D-BGE (1.000) | R2-D-MiniLM-L12 (0.9359) | −6.4 pp |
| HR@6 (binary chunk hit) | R2-A (0.9062) | R2-B (0.7188) | −18.7 pp |
| MRR (ranking, first hit) | R2-D-MiniLM-L12 (0.7578) | R2-B (0.3109) | −44.7 pp |
| NDCG@6 (comprehensive ranking) | R2-D-MiniLM-L12 (0.7635) | R2-B (0.4091) | −35.4 pp |
| R@6 (chunk recall) | R2-A (0.3241) | R2-D-MiniLM-L12 (0.1562) | −16.8 pp |

**Why BM25 scores best on RAGAS CR but worst on HR@6 and ranking:**

BM25 retrieves information-equivalent chunks — not necessarily the same chunks as the reference, but chunks containing the same clinical content (same dressing name, same guideline recommendation, different sentence structure or source section). The RAGAS LLM judge, assessing semantic coverage rather than chunk identity, awards BM25 high CR because the information is present. But HR@6, MRR, and NDCG all require the exact reference chunks to be present and highly ranked, which BM25 cannot guarantee when the reference context was authored from a specific section and BM25 finds a different section containing the same terms.

**Why MiniLM rerankers score best on ranking but worst on recall:**

Cross-encoders optimised for MS MARCO web-domain queries produce a clean, well-ordered ranking of the chunks they retain — but they are trained on a fundamentally different distribution than wound care clinical guidelines. They systematically suppress chunks with dense clinical terminology that doesn't match web-domain relevance signals, reducing the pool of clinically correct chunks. The result: excellent NDCG over a reduced, cleaner set, but poor coverage (CR, R@6) because relevant clinical chunks are excluded.

---

## 5. Persistent Hard Cases: Case 27 (Index 26)

A key diagnostic finding from R1 identified Case 27 (index 26) as a persistently near-zero CR case across all R1 strategies. R2 provides the opportunity to test whether any retrieval method resolves this case.

| Version | Case 27 CR (Run 1 / Run 2 / Run 3) |
|---|---|
| R2-A (Dense) | 0.059 / 0.000 / 0.062 |
| R2-B (BM25) | 0.053 / 0.053 / 0.062 |
| R2-C (Hybrid) | 0.000 / 0.000 / 0.059 |
| R2-D-BGE | 0.059 / 0.059 / 0.059 |
| R2-D-MiniLM-L6 | 0.059 / 0.000 / 0.000 |
| R2-D-MiniLM-L12 | 0.000 / 0.000 / 0.000 |

Case 27 is structurally irrecoverable across all retrieval strategies and all rerankers. No version achieves a CR above 0.062 for this case. This indicates that the reference content for Case 27 is not retrievable at k=6 regardless of retrieval method — the relevant chunk is either ranked below position 6 in all strategies or is not adequately represented in the current KB at the chunk level. This is a candidate for investigation in R3 (higher k) and separately for KB curation review.

Case 19 (index 18) is also diagnostically interesting: R2-B achieves CR = 1.0 in all three runs for this case, while R2-A averages ~0.38 and R2-D-MiniLM-L12 averages ~0.60. This case specifically benefits from BM25's exact-term matching, suggesting the reference content contains clinical terminology that dense embeddings do not rank optimally for this case type.

---

## 6. Key Findings

### Finding 1: BM25 Dominates on RAGAS Metrics in a Clinical-Terminology KB

The strongest result in R2 is that BM25-only retrieval achieves the highest CR (0.906) and second-highest CP (0.994) of any version tested. This is not a statistical artefact — the gap over the dense baseline (+2.6 pp CR) is consistent across all three runs. Wound care guidelines are lexically distinctive: they contain specialised vocabulary (specific dressing product names, wound management procedures, clinical decision criteria) with low cross-domain frequency, which is precisely the regime where BM25's IDF weighting is most effective. The clinical KB is a document collection where lexical specificity and semantic specificity are strongly aligned.

### Finding 2: Hybrid Retrieval with RRF Fails to Improve over Either Single Modality on CR

R2-C (Hybrid RRF) achieves CR = 0.870, which is worse than both the dense baseline (0.880) and BM25 (0.906). This negative finding contradicts the primary hypothesis of the R2 experiment and warrants discussion. The failure mechanism of RRF in this setting is likely related to the k=6 hard ceiling: when both the dense retriever and BM25 each contribute their best matches to an RRF pool, and then the pool is reduced to k=6, the fusion process selects chunks that both systems agree on (high joint rank), at the cost of chunks that only one system ranked highly. Since BM25's unique contribution (exact-match information-equivalent chunks) is only highly ranked by BM25 and not dense retrieval, these chunks tend to be deprioritised by RRF — reducing BM25's recall contribution. The implication is that RRF with a tight k ceiling acts as an intersection-biased operator rather than the union-biased operator that motivated the hybrid hypothesis.

### Finding 3: BGE-v2-m3 is the Only Reranker Compatible with Clinical Retrieval

Among the three rerankers tested, BAAI/bge-reranker-v2-m3 is the only one that does not significantly harm CR relative to the dense baseline (0.874 vs 0.880, −0.6 pp). It achieves perfect CP (1.0000), the best precision in the entire experiment. In contrast, MiniLM-L6 (−2.8 pp CR) and MiniLM-L12 (−7.6 pp CR) both substantially degrade recall, confirming the domain mismatch hypothesis. The cost of BGE-v2-m3's compatibility is a 28× latency increase over the dense baseline (2801 ms vs 97 ms).

### Finding 4: Web-Domain Rerankers Produce a Double Failure Pattern

Both MiniLM variants degrade CR below the dense baseline AND degrade CP below all non-reranking variants. This is the clearest negative finding in R2: a reranker trained on MS MARCO web-domain data is actively harmful to clinical wound care retrieval — it reduces both the quantity and the quality of retrieved information. This replicates and extends the finding from the previous ablation study that first identified this pattern.

### Finding 5: R2-A (Dense-only) Remains the Best-Balanced Configuration

When all metrics are considered together, R2-A (dense-only, replicating the R1-C baseline) provides the best overall balance: second-best CR (0.880), middle-of-range CP (0.972), best HR@6 (0.906), best chunk-level recall (R@6 = 0.324), acceptable latency (97 ms), and perfectly stable IR metrics (SD = 0). R2-B's higher CR comes at the cost of catastrophic HR@6 (0.719) and MRR (0.311), which represent meaningful risks for a system that may need to surface specific guideline passages. R2-D-BGE offers perfect CP but at 29× higher latency and no CR improvement.

---

## 7. Discussion — Implications for VerdaSense Architecture

### 7.1 Why Hybrid Retrieval Failed the Primary Hypothesis

The R2 ablation map predicted that hybrid retrieval would improve recall by combining BM25's lexical match strength with dense retrieval's semantic understanding. This hypothesis was based on the assumption that BM25 and dense retrieval retrieve complementary sets of relevant chunks. The R2 results suggest this assumption does not hold at k=6 for this specific KB.

The most likely explanation is that in a small, specialised clinical KB (138 chunks, 8 sources), the vocabulary alignment between queries and documents is already high for the dense embedding model (MedEmbed-large-v0.1, trained on medical text). The "vocabulary mismatch" problem that BM25 typically solves is less severe here because MedEmbed already understands clinical terminology. BM25's contribution at the retrieval step is therefore not additive — it retrieves many of the same chunks dense retrieval already retrieves, and the RRF fusion's union advantage is lost under the k=6 ceiling.

This finding should be discussed in the FYP as a domain-specific result: hybrid retrieval's benefit depends on the vocabulary gap between queries and documents. In a general-domain KB, dense embeddings are trained primarily on non-clinical text and vocabulary mismatch is real. In VerdaSense's clinically curated KB with a medically fine-tuned embedding model, the mismatch is already minimised, reducing the marginal benefit of BM25.

### 7.2 The Case for BM25 as a Single-Strategy Alternative

R2-B's CR = 0.906 is the highest of any version in R2, and its latency (1.8 ms) is 54× faster than dense retrieval. For deployment contexts where latency is the dominant constraint (e.g., a mobile app on a low-power network connection), BM25-only retrieval is a compelling option for this KB. Its weakness on ranking (MRR = 0.311, NDCG = 0.409) is a real limitation, but its practical impact on generation quality depends on whether the LLM generating recommendations is sensitive to chunk ordering in the context window.

### 7.3 Connecting BGE-v2-m3 Findings to Reranker Selection for Future Experiments

R2-D-BGE's perfect CP (1.0) with minimal CR cost (−0.6 pp) indicates that multilingual, medical-adjacent cross-encoders can act as effective precision cleaners without destroying recall. If precision is critical for the generation step (e.g., to minimise hallucination risk), adding BGE-v2-m3 reranking to the best retrieval strategy is a defensible architectural choice, provided the 2.8-second latency overhead is acceptable within the overall system budget.

For R3 and R4, the reranker question can be revisited if a higher-k hybrid pool allows BGE-v2-m3 to operate on a larger candidate set (which may improve its CR contribution when k is not the binding constraint).

### 7.4 The Metric Divergence as a Methodological Contribution

The finding that RAGAS CR and chunk-level IR metrics (HR@6, MRR) rank retrieval strategies in opposite orders is a methodological contribution in itself. Previous RAG evaluation work (including the German nursing paper, Powering & Rothgang 2026) primarily used RAGAS metrics or IR metrics but not both simultaneously. The R2 results demonstrate that using only RAGAS metrics would lead to selecting BM25 as the winner, while using only IR metrics would lead to selecting dense retrieval. The truth is more nuanced: the optimal strategy depends on whether the downstream task requires semantic information coverage (RAGAS CR-optimal: BM25) or exact reference chunk retrieval (HR@6-optimal: Dense).

For VerdaSense's generation task — producing a patient-facing recommendation grounded in retrieved context — the RAGAS CR metric is the more directly relevant predictor of generation quality, because the LLM synthesises information from retrieved chunks rather than citing specific chunk IDs. However, disclosing this methodological distinction in the FYP discussion demonstrates evaluative rigour and awareness of RAG evaluation limitations.

---

## 8. Decision: R2-A Selected for R3, R2-B and R2-D-BGE Carried Forward as Alternatives

**R2-A (Dense-only) is selected as the fixed retrieval strategy for R3 (top-K ablation) and beyond.**

Justification:

- Best overall balance across the full metric suite: best HR@6 (0.906), best R@6 (0.324), best P@6 (0.281), competitive CR (0.880 vs 0.906 for R2-B), acceptable CP (0.972).
- Stable, reproducible IR metrics (SD = 0 across 3 runs).
- Lowest latency of all vector-search-based strategies (97.1 ms).
- R2-B's higher CR is noted and will be discussed as a KB-domain-specific finding, but its HR@6 and ranking quality deficits make it a secondary rather than primary selection for the full Stage 1 pipeline.

**Carry-forward note for the FYP discussion:**
R2-B's CR = 0.906 is the highest single retrieval CR observed in Stage 1 so far, and this result should be presented in the Stage 1 summary table and discussed as a domain-specific observation — that lexical retrieval is particularly effective on clinically specialised, terminology-dense KBs. If the FYP includes a section on alternative deployment profiles (e.g., lightweight edge deployment), R2-B + R3 optimal-k is a viable low-cost configuration worth characterising.

R2-D-BGE's perfect CP (1.0000) is worth noting as the highest precision result in Stage 1. If Stage 2 generation experiments show that hallucination or answer drift is linked to context noise, revisiting BGE reranking as a precision module may be warranted.

---

## 9. Summary for FYP Viva

**One-sentence answer to the R2 research question:**

> Combining dense and sparse retrieval (hybrid RRF) does not improve context recall over dense-only retrieval at k=6 on VerdaSense's clinically curated KB; BM25-only retrieval achieves the highest RAGAS context recall (0.906) by exploiting clinical terminology specificity, while dense-only retrieval (0.880) remains the best-balanced strategy across the full metric suite; and web-domain cross-encoder rerankers (MiniLM-L6/L12) are actively harmful to clinical retrieval, degrading both recall and precision simultaneously.

**Three things to remember for the viva:**

1. **BM25 beats dense on CR** — not because dense embedding fails, but because the wound care KB is lexically distinctive enough that exact keyword matching surfaces information-equivalent content with high coverage. This is a domain-specific finding, not a general one.

2. **The hybrid hypothesis failed at k=6** — RRF is a union-favouring operator, but its benefit is bounded by the k ceiling. When both retrievers agree on relevant chunks (as they do in a focused clinical KB), RRF offers no additional diversity over either single strategy alone.

3. **No reranker improves on both CR and ranking simultaneously** — BGE-v2-m3 perfects precision but slows the system 29×; MiniLM variants improve ranking metrics but destroy recall. The "free lunch" scenario where a reranker improves all metrics does not materialise in this clinical domain with a medically fine-tuned embedding model.

---

*Generated: 13 May 2026 | VerdaSense FYP — Universiti Malaya*  
*Next: R3 — Top-K Retrieval Depth Ablation using R2-A (Dense-only) fixed retrieval strategy*
