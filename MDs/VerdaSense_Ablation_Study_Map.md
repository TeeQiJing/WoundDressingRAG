# VerdaSense RAG — New-Era Ablation Study Map
**Fixed Foundation: `db_wound_care_v4` (8 KB sources) + `wound_testset_v3.json` (32 cases)**
**Fixed RAGAS Judge: `gpt-4o-mini` + `text-embedding-3-small` — never changed across any version**

---

## Preface — Why Start Again

The previous 9-version study (v2/v3/v4 × 00/01/02) used `db_wound_care_v3` (4 sources, 28 cases).
That work established a strong architectural baseline in `v4_02`.
**This new study begins fresh** with an expanded gold-standard KB (8 sources, 32 cases)
and is designed as a structured academic ablation that can be written up in your FYP chapter-by-chapter.

The study is divided into two independent stages:

| Stage | What varies | What is fixed |
|---|---|---|
| **Stage 1 — Retrieval Ablation** | Retrieval strategy and embedding model | Generation: best possible prompt, `gpt-4o-mini` |
| **Stage 2 — Generation Ablation** | LLM model and prompt strategy | Retrieval: best configuration from Stage 1 |

Every version in both stages uses the **same KB and the same 32-case testset**.
This is methodologically equivalent to the German nursing paper (Powering & Rothgang, 2026),
which held all other components fixed while varying one dimension at a time.

---

## Gold Standard Fixed Components

```
KB:      db_wound_care_v4/   (8 sources: GP, WCM, AJGP, SFP, EWMA, ISTAP, ANZBA, RCH)
Testset: wound_testset_v3.json   (32 cases: Cat A×8, B×12, C×6, D×4, E×2)
Embedding (ingestion & query): abhinand/MedEmbed-large-v0.1   [FIXED for all Stage 1 versions EXCEPT R-Emb variants]
RAGAS LLM judge:  gpt-4o-mini   [FIXED — never changed]
RAGAS Embed judge: text-embedding-3-small   [FIXED — never changed]
```

---

## Evaluation Metrics — Full Suite

### Stage 1: Retrieval Metrics

These metrics measure how well the retrieval pipeline surfaces the right chunks
**before** any generation happens. They are computed by comparing `retrieved_contexts`
against `reference_contexts` from the testset.

| Metric | What it measures | Why it matters for your FYP |
|---|---|---|
| **Context Recall (CR)** | Fraction of reference context information covered by retrieved chunks (LLM judge) | Primary retrieval metric — did we retrieve everything the reference needs? |
| **Context Precision (CP)** | Fraction of retrieved chunks that are actually relevant (LLM judge) | Measures noise in retrieval — irrelevant chunks hurt generation quality |
| **Hit Rate @ K** | Whether at least 1 reference chunk appears in top-K results | Simple binary retrieval success measure; good for reporting in a table |
| **MRR (Mean Reciprocal Rank)** | Average of 1/rank of the first relevant chunk across all queries | Captures ranking quality — is the most relevant chunk ranked first? |
| **NDCG @ K** | Graded ranking quality score (rewards relevant chunks ranked higher) | Most comprehensive ranking metric; standard in IR literature |
| **Recall @ K** | Fraction of reference chunks found in top-K retrieved results | Directly comparable to the German paper's recall metric |
| **Precision @ K** | Fraction of top-K retrieved chunks that are relevant | Directly comparable to the German paper's precision metric |
| **Retrieval Latency (ms)** | Wall-clock time for the retrieval step only | Practical metric for mobile app deployment feasibility |

> **Note on BLEU / ROUGE / METEOR:** These traditional NLP metrics measure surface-level text overlap
> between generated output and a reference string. They are **not recommended** for your FYP for
> two reasons: (1) your reference answers use patient-facing plain language while the KB uses
> clinical terminology — surface overlap will be low by design; (2) RAGAS faithfulness and
> answer correctness already capture what BLEU/ROUGE attempt to measure, but using semantic
> embedding similarity instead of word overlap, which is far more appropriate for
> paraphrased clinical recommendations. Do not include these metrics.

### Stage 2: Generation Metrics

| Metric | What it measures | Why it matters |
|---|---|---|
| **Faithfulness (FA)** | Fraction of claims in the answer that can be traced to the retrieved context | Hallucination detection — are all statements grounded in the KB? |
| **Answer Relevancy (AR)** | How well the answer addresses the patient's actual question (embedding similarity) | Is the answer on-topic and complete? |
| **Safety Pass Rate (%)** | Rule-based check: are contraindicated dressings absent? Is antibiotic language present when required? Is referral language present when required? | Clinical safety — the most important non-RAGAS metric for a clinical AI FYP |
| **Generation Latency (ms)** | Wall-clock time for the LLM generation step only | Mobile app feasibility — patient-facing tools need <10s response |
| **Overall Latency (ms)** | End-to-end wall-clock time (retrieval + generation) | Real-world usability |

### Additional Metrics to Collect (Low Effort, High Value)

| Metric | How to collect | Why useful |
|---|---|---|
| **Average retrieved context length (tokens)** | `sum(len(chunk.split()) for chunk in retrieved_contexts)` | Longer context = higher cost + latency; precision-recall tradeoff |
| **Unique sources in retrieved context** | Count distinct `source` metadata values | Measures KB coverage diversity per query |
| **Chunk overlap with reference_contexts (exact match %)** | Binary check: is each `reference_context` chunk in `retrieved_contexts`? | Complement to RAGAS CR — direct chunk-level match |
| **Safety failure breakdown** | Which safety rule failed (contraindication / antibiotic / referral) | Pinpoints where the architecture fails clinically |

---

## Stage 1 — Retrieval Ablation

**Generation is held constant:** `gpt-4o-mini`, best clinical prompt from `v4_02`, no architectural changes to classifier or mandatory injection.
The goal is to isolate and measure the independent contribution of each retrieval component.

The stage is structured into four experiments, each varying one retrieval dimension.

---

### Experiment R1: Query Formulation Strategy

**Research Question:** Does the way the TIME inputs are turned into a retrieval query affect what chunks are retrieved?

This is the most important experiment for your original contribution — the German paper
did not study structured clinical query expansion at all.

| Version | Label | Query Strategy | Other Settings |
|---|---|---|---|
| **R1-A** | Flat label concat | `"sloughy fibrinous wound bed Not infected High exudate Non-advancing wound dressing"` | Dense, k=6, MedEmbed |
| **R1-B** | Narrative NL query | GPT-built natural language question from TIME inputs | Dense, k=6, MedEmbed |
| **R1-C** | Multi-axis sub-queries | 3 parallel queries: (A) wound-type algorithm, (B) dressing mechanism, (C) patient notes | Dense, k=6 total, MedEmbed |

**Hypothesis:** Multi-axis sub-query (R1-C) should outperform flat concat (R1-A) because
each sub-query is semantically focused on a specific aspect of the clinical scenario.
Narrative query (R1-B) should outperform flat concat but lag behind multi-axis.

**What to measure:** CR, CP, Hit Rate, MRR, NDCG, Recall@6, Precision@6, Retrieval Latency.

> **Your original contribution note:** The comparison R1-A vs R1-B vs R1-C on
> structured TIME inputs is not found in the German paper or other RAG-for-nursing literature.
> This is a meaningful academic contribution because your query inputs are structured
> (from a CV pipeline), not free text — the query formulation problem is different from
> general clinical Q&A RAG.

---

### Experiment R2: Retrieval Strategy (Sparse / Dense / Hybrid)

**Research Question:** Does combining dense semantic search with sparse BM25 retrieval
improve context recall for wound care guidelines?

Hold constant: Multi-axis query from best R1 result, k=6 total, MedEmbed.

| Version | Label | Retrieval Method | Notes |
|---|---|---|---|
| **R2-A** | Dense only | ChromaDB cosine similarity search | Baseline |
| **R2-B** | Sparse only (BM25) | BM25Retriever on all KB documents | Lexical keyword match |
| **R2-C** | Hybrid (Dense + BM25) | EnsembleRetriever, RRF fusion | Semantic + lexical |
| **R2-D** | Hybrid + Reranking | Hybrid retrieval → cross-encoder reranking | Ranking quality improvement |

**On the reranker choice:** Your previous study showed `ms-marco-MiniLM-L-6-v2` (trained on
web text) consistently hurt Context Recall by depressing clinical chunks. For this new study,
use a medically-appropriate reranker. Options in order of preference:
- `abhinand/MedEmbed-reranker-v0.1` — same family as your embedding model, medical domain
- `BAAI/bge-reranker-v2-m3` — strong multilingual cross-encoder, better than MiniLM on domain text
- `cross-encoder/ms-marco-MiniLM-L-12-v2` — larger version of your previous reranker (keep if no medical option available; disclose limitation)

**Hypothesis:** Hybrid (R2-C) should improve recall over dense-only (R2-A) for wound type
queries because BM25 catches exact dressing name matches (e.g., "alginate", "hydrofibre")
that semantic search may rank lower. Reranking (R2-D) may improve precision but could
hurt recall — this is an empirical question worth answering.

**What to measure:** Full retrieval metric suite + latency.

---

### Experiment R3: Top-K Retrieval Depth

**Research Question:** How does the number of retrieved chunks affect the precision-recall tradeoff?

Hold constant: Best query strategy (R1 best), best retrieval method (R2 best), MedEmbed.

| Version | Label | K value | Notes |
|---|---|---|---|
| **R3-A** | k=2 | Top 2 chunks | Minimum context, high precision |
| **R3-B** | k=4 | Top 4 chunks | Moderate |
| **R3-C** | k=6 | Top 6 chunks | Current default |
| **R3-D** | k=8 | Top 8 chunks | More coverage |
| **R3-E** | k=10 | Top 10 chunks | Maximum practical context |

**Hypothesis:** Recall increases and precision decreases monotonically as K increases.
There is a point where adding more chunks no longer improves recall but adds noise.
The German paper found k=2 optimal for their content-aware chunked dataset.
For your dataset (mixed chunk sizes from 8 sources), the optimum may differ.

**Also report:** Average context token length at each K — this is important for
generation quality and API cost, and is a metric the German paper reported.

---

### Experiment R4: Embedding Model Comparison

**Research Question:** Does the choice of embedding model affect retrieval quality for wound care guidelines?

Hold constant: Best query strategy (R1), best retrieval method (R2), best K (R3).

| Version | Label | Embedding Model | Notes |
|---|---|---|---|
| **R4-A** | MedEmbed Large | `abhinand/MedEmbed-large-v0.1` | Current model — medical domain |
| **R4-B** | BGE Large | `BAAI/bge-large-en-v1.5` | Strong general retrieval benchmark performance |
| **R4-C** | E5 Large | `intfloat/e5-large-v2` | Another strong retrieval model |
| **R4-D** | *(Optional)* MedEmbed Base | `abhinand/MedEmbed-base-v0.1` | Smaller, faster — efficiency comparison |

> **Important:** Each embedding model requires re-ingesting the KB into a separate ChromaDB
> collection (because embeddings change). This is the most expensive experiment in Stage 1.
> If HPC time is limited, run R4-A (already done) and R4-B only, and note the others as
> future work. The academic contribution here is showing whether a medical-domain embedding
> model outperforms a general embedding model on clinical guideline retrieval.

**What to measure:** Full retrieval metric suite. Note: CR and CP from RAGAS use the
FIXED judge (`gpt-4o-mini` + `text-embedding-3-small`) — they do not change with the
embedding model. The metrics that change are Hit Rate, MRR, NDCG, Recall@K, Precision@K.

---

### Stage 1 — Summary Table Template

After running all R1–R4 experiments, fill this table:

| Exp | Version | Query | Retrieval | K | Embedding | CR | CP | HR@K | MRR | NDCG@K | R@K | P@K | Lat(ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | R1-A | Flat concat | Dense | 6 | MedEmbed | | | | | | | | |
| R1 | R1-B | Narrative | Dense | 6 | MedEmbed | | | | | | | | |
| R1 | R1-C | Multi-axis | Dense | 6 | MedEmbed | | | | | | | | |
| R2 | R2-A | Best R1 | Dense | 6 | MedEmbed | | | | | | | | |
| R2 | R2-B | Best R1 | BM25 | 6 | MedEmbed | | | | | | | | |
| R2 | R2-C | Best R1 | Hybrid | 6 | MedEmbed | | | | | | | | |
| R2 | R2-D | Best R1 | Hybrid+Rerank | 6 | MedEmbed | | | | | | | | |
| R3 | R3-A | Best R1 | Best R2 | 2 | MedEmbed | | | | | | | | |
| R3 | R3-B | Best R1 | Best R2 | 4 | MedEmbed | | | | | | | | |
| R3 | R3-C | Best R1 | Best R2 | 6 | MedEmbed | | | | | | | | |
| R3 | R3-D | Best R1 | Best R2 | 8 | MedEmbed | | | | | | | | |
| R3 | R3-E | Best R1 | Best R2 | 10 | MedEmbed | | | | | | | | |
| R4 | R4-A | Best R1 | Best R2 | Best R3 | MedEmbed | | | | | | | | |
| R4 | R4-B | Best R1 | Best R2 | Best R3 | BGE Large | | | | | | | | |
| R4 | R4-C | Best R1 | Best R2 | Best R3 | E5 Large | | | | | | | | |

**Best Retrieval Configuration** = the row with the best balance of CR, CP, and Recall@K.
This configuration is then held fixed for all Stage 2 experiments.

---

## Stage 2 — Generation Ablation

**Retrieval is held constant:** Best configuration identified from Stage 1.
The goal is to isolate how prompt strategy and LLM choice affect output quality.

---

### Experiment G1: Prompt Strategy (Same LLM — GPT-4o-mini)

**Research Question:** How much does the prompt design contribute to faithfulness, safety, and answer quality?

This experiment holds the LLM fixed (`gpt-4o-mini`) and varies only the prompt.
It isolates the contribution of prompt engineering from the contribution of model capability.

| Version | Label | Prompt Strategy | Clinical Scaffolding |
|---|---|---|---|
| **G1-A** | Zero-shot | Raw TIME inputs only, no instruction | None |
| **G1-B** | Basic structured | Standard clinical sections (Primary Dressing, Rationale, etc.) | None |
| **G1-C** | Grounded system prompt | Explicit "use only retrieved sources, cite source numbers" instruction | None |
| **G1-D** | Full clinical scaffolding | Grounded prompt + Binding algorithm block (G1) + Mandatory injection (G2) + Clinical pre-classifier (R1) | Pre-classifier + mandatory blocks |

**What G1-A vs G1-B vs G1-C vs G1-D isolates:**
- G1-A → G1-B: Does structured output format help?
- G1-B → G1-C: Does grounding instruction reduce hallucination?
- G1-C → G1-D: Does clinical pre-classification and mandatory injection improve safety?

**What to measure:** FA, AR, Safety Pass Rate, Generation Latency.

> **Key hypothesis based on your previous study:** The biggest jump will be G1-B → G1-C
> (grounding instruction). Safety pass rate will jump most sharply at G1-C → G1-D
> (mandatory injection handles the antibiotic/referral language). This matches your v2→v3
> and v3→v4 findings and would be a replication of those findings on the new KB + testset.

---

### Experiment G2: LLM Model Comparison (Closed-Source)

**Research Question:** Does GPT-4o-mini perform better than other closed-source LLMs for
patient-facing wound care recommendations?

Hold constant: Best prompt from G1, best retrieval from Stage 1.

| Version | Label | LLM | Type | Notes |
|---|---|---|---|---|
| **G2-A** | GPT-4o-mini | `gpt-4o-mini` | OpenAI closed | Current baseline; fast and cheap |
| **G2-B** | GPT-4o | `gpt-4o` | OpenAI closed | Higher capability reference point |
| **G2-C** | *(Optional)* Claude Haiku | `claude-haiku-4-5` | Anthropic closed | Alternative closed-source comparison |

> **Cost note:** GPT-4o is ~5–10× more expensive than GPT-4o-mini per token.
> Run it on a subset (16 cases, the Cat A + Cat B cases) if cost is a constraint.
> Document clearly in your FYP that the full 32-case comparison was done with GPT-4o-mini
> and a 16-case subset with GPT-4o.

---

### Experiment G3: LLM Model Comparison (Open-Source — HPC)

**Research Question:** Can open-source LLMs run on UM HPC match closed-source performance
for wound care recommendations, enabling a cost-free deployment path?

Hold constant: Best prompt from G1, best retrieval from Stage 1.

| Version | Label | LLM | Parameters | Notes |
|---|---|---|---|---|
| **G3-A** | Qwen3 14B | `Qwen/Qwen3-14B` | 14B | Already used in your previous study via Ollama |
| **G3-B** | Llama 3.1 8B | `meta-llama/Llama-3.1-8B-Instruct` | 8B | Widely used instruction-tuned baseline |
| **G3-C** | Llama 3.1 70B | `meta-llama/Llama-3.1-70B-Instruct` | 70B | Larger model; check HPC VRAM availability |
| **G3-D** | Gemma 3 12B | `google/gemma-3-12b-it` | 12B | Best performer in the German paper |
| **G3-E** | *(Optional)* MedGemma | `google/medgemma-4b-it` | 4B | Medical domain fine-tune; used in German paper |

> **HPC practical advice:**
> - Run all models on the same GPU node with the same batch settings.
> - Use `vllm` or `transformers` with `torch.bfloat16` for efficient inference.
> - Set temperature=0 for all models for deterministic output.
> - If 70B is too large for available VRAM, run Llama 3.1 8B + Qwen3 14B only — two models is sufficient for an academic comparison.
> - Gemma 3 12B is recommended because it is the best-performing open-source model in the closest published study.

---

### Experiment G4: Patient Language vs Clinician Language (Prompt Framing)

**Research Question:** Does explicitly instructing the LLM to use patient-friendly language
improve Answer Relevancy for a patient-facing recommendation?

This experiment is unique to your study and not found in the German paper.
Because your system outputs recommendations directly to patients (not nurses),
the instruction framing matters differently.

Hold constant: Best retrieval, `gpt-4o-mini`.

| Version | Label | Instruction Framing | Notes |
|---|---|---|---|
| **G4-A** | Clinician framing | "Provide a clinical wound care recommendation with T.I.M.E. rationale..." | Standard clinical prompt |
| **G4-B** | Patient framing | "Provide a dressing recommendation for the patient in plain language, avoiding medical jargon..." | Patient-facing instruction |
| **G4-C** | Hybrid framing | Patient-facing output + brief clinician-readable rationale section | Best of both |

**What to measure:** FA, AR, Safety Pass Rate. Also manually review 5–10 outputs for
readability — AR score alone may not capture comprehension by a non-clinician.
This is where your subset clinician/patient review adds value beyond automated metrics.

> **Why this matters for your FYP:** The German paper evaluated nurse-facing output.
> You are evaluating patient-facing output. G4 is your clearest point of academic distinction
> from prior work.

---

### Stage 2 — Summary Table Template

| Exp | Version | LLM | Prompt | FA | AR | Safety% | Gen Lat(ms) | Total Lat(ms) |
|---|---|---|---|---|---|---|---|---|
| G1 | G1-A | GPT-4o-mini | Zero-shot | | | | | |
| G1 | G1-B | GPT-4o-mini | Basic structured | | | | | |
| G1 | G1-C | GPT-4o-mini | Grounded system prompt | | | | | |
| G1 | G1-D | GPT-4o-mini | Full scaffolding (v4_02) | | | | | |
| G2 | G2-A | GPT-4o-mini | Best G1 | | | | | |
| G2 | G2-B | GPT-4o | Best G1 | | | | | |
| G3 | G3-A | Qwen3 14B | Best G1 | | | | | |
| G3 | G3-B | Llama 3.1 8B | Best G1 | | | | | |
| G3 | G3-C | Llama 3.1 70B | Best G1 | | | | | |
| G3 | G3-D | Gemma 3 12B | Best G1 | | | | | |
| G4 | G4-A | Best G2/G3 | Clinician framing | | | | | |
| G4 | G4-B | Best G2/G3 | Patient framing | | | | | |
| G4 | G4-C | Best G2/G3 | Hybrid framing | | | | | |

---

## Stage 3 — Best Configuration (Production)

After Stage 1 and Stage 2 are complete, identify the single best configuration:

```
Best Retrieval Config  =  best (query + retrieval method + K + embedding)
Best Generation Config =  best (LLM + prompt strategy)
```

Run a final evaluation of this combined "production configuration" on all 32 cases.
This becomes your **VerdaSense v6** — the final system integrated into the mobile app.

Report this in your FYP as the "Optimal Configuration" and compare it against:
- The weakest baseline (R1-A + G1-A): shows total improvement
- The previous best from the old study (v4_02 on v3 KB): shows KB expansion impact

---

## Implementation Notes

### Notebook structure

Create one notebook per experiment group to keep evaluation reproducible:

```
ragas_ablation_R1_query_strategy.ipynb
ragas_ablation_R2_retrieval_method.ipynb
ragas_ablation_R3_topk.ipynb
ragas_ablation_R4_embedding.ipynb
ragas_ablation_G1_prompt_strategy.ipynb
ragas_ablation_G2_closed_llm.ipynb
ragas_ablation_G3_open_llm.ipynb
ragas_ablation_G4_patient_language.ipynb
ragas_ablation_FINAL_best_config.ipynb
```

Each notebook:
1. Loads `wound_testset_v3.json` (never modified)
2. Loads `db_wound_care_v4/` (never modified)
3. Runs the architecture variant
4. Saves results to `results/EXPID_VERSION_results.json`
5. Runs RAGAS with fixed judge (`gpt-4o-mini` + `text-embedding-3-small`)
6. Saves RAGAS scores to `results/EXPID_VERSION_ragas.json`
7. Prints the summary table row

### Variance: run each version 3 times

Every experiment should be run 3 times and scores averaged (mean ± standard deviation).
This accounts for RAGAS judge stochasticity. For GPT-4o-mini at temperature=0,
variance is low but not zero across runs. For open-source models, variance is higher.
Running 3 times is one line of code change and makes your results significantly more credible.

```python
N_RUNS = 3
all_scores = []
for run in range(N_RUNS):
    scores = evaluate_ragas(results, testset)
    all_scores.append(scores)
report_mean_std(all_scores)
```

### Retrieval-only metrics (Hit Rate, MRR, NDCG, Recall@K, Precision@K)

These do not require the RAGAS judge and can be computed cheaply from the
`retrieved_contexts` and `reference_contexts` fields. Implement a `compute_retrieval_metrics()`
function once in a shared utility module and call it from every notebook.

```python
def compute_retrieval_metrics(retrieved_contexts, reference_contexts, k):
    """
    retrieved_contexts: list of strings (actual retrieved chunk texts)
    reference_contexts: list of strings (gold-standard chunk texts from testset)
    k: int
    """
    hits = []
    reciprocal_ranks = []
    ndcgs = []
    recalls = []
    precisions = []

    for retrieved, references in zip(retrieved_contexts, reference_contexts):
        # Binary relevance: is reference chunk text present in retrieved?
        # Use first 100 chars as key (chunk texts are unique)
        ref_keys = {r[:100] for r in references}
        ret_keys = [r[:100] for r in retrieved[:k]]

        # Hit Rate
        hit = any(rk in ref_keys for rk in ret_keys)
        hits.append(int(hit))

        # MRR
        rr = 0.0
        for rank, rk in enumerate(ret_keys, 1):
            if rk in ref_keys:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        # NDCG@K
        relevances = [1 if rk in ref_keys else 0 for rk in ret_keys]
        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))
        ideal = sorted(relevances, reverse=True)
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        # Recall@K and Precision@K
        retrieved_relevant = sum(1 for rk in ret_keys if rk in ref_keys)
        recalls.append(retrieved_relevant / len(ref_keys) if ref_keys else 0.0)
        precisions.append(retrieved_relevant / k)

    return {
        "hit_rate": mean(hits),
        "mrr": mean(reciprocal_ranks),
        "ndcg_at_k": mean(ndcgs),
        "recall_at_k": mean(recalls),
        "precision_at_k": mean(precisions),
    }
```

### Latency measurement

Measure and record latency separately for retrieval and generation:

```python
import time

t0 = time.perf_counter()
retrieved = retrieve_chunks(...)
retrieval_latency_ms = (time.perf_counter() - t0) * 1000

t1 = time.perf_counter()
result = generate_recommendation(...)
generation_latency_ms = (time.perf_counter() - t1) * 1000

total_latency_ms = retrieval_latency_ms + generation_latency_ms
```

Report mean ± SD latency across all 32 test cases for each version.
For mobile app feasibility, a total latency under 10,000 ms (10 seconds) is a
practical threshold for an acceptable patient-facing experience.

---

## Mapping to Your FYP Chapter Structure

| FYP Chapter | Covered by |
|---|---|
| Introduction + Literature Review | German paper (Powering & Rothgang) + scoping review gap analysis |
| System Design | Fixed KB design, testset construction, evaluation framework |
| Stage 1: Retrieval Ablation | R1 (query), R2 (retrieval method), R3 (top-K), R4 (embedding) |
| Stage 2: Generation Ablation | G1 (prompt), G2 (closed LLM), G3 (open LLM), G4 (patient language) |
| Final Configuration | Stage 3 best-config evaluation + safety analysis |
| Mobile Integration | Latency results + API design for the wound tracking app |
| Discussion | Comparison with German paper + limitations + future work |

---

## Comparison with the German Paper (Powering & Rothgang, 2026)

This table shows where your study replicates, extends, or differs from the reference paper.
Use this in your FYP discussion chapter.

| Dimension | German Paper | Your Study |
|---|---|---|
| Domain | German nursing guideline (wound care) | English wound care guidelines (GP, EWMA, ANZBA, ISTAP, RCH, etc.) |
| Language | German | English |
| KB size | 1 guideline | 8 guidelines |
| Testset | 30 questions (verbatim from guideline) | 32 cases (structured TIME + clinical notes) |
| Input type | Free-text clinical question | Structured TIME assessment + optional free-text notes |
| Preprocessing ablation | Raw vs manually cleaned | Not studied (all manually curated — document this as deliberate choice) |
| Chunking ablation | Fixed-size vs content-aware vs sentence-based | Not studied (all content-aware — document why) |
| Embedding models | BGE-M3, LaBSE, distiluse | MedEmbed, BGE-Large, E5-Large |
| Retrieval methods | Dense, BM25, Hybrid, Reranking | Dense, BM25, Hybrid, Reranking (same four) |
| LLMs | Gemma 3 12B, Llama 3.1 8B, DeepSeek-R1 14B, MedGemma | GPT-4o-mini, GPT-4o, Qwen3 14B, Llama 3.1, Gemma 3 12B |
| Prompt strategies | Zero-shot vs Instruction | Zero-shot vs Basic structured vs Grounded vs Full scaffolding |
| Clinical scaffolding | None | Pre-classifier + binding block + mandatory injection (your novel contribution) |
| Output audience | Nurses (clinician-facing) | Patients (patient-facing) — your novel contribution |
| Safety evaluation | None | Rule-based safety checker (your novel contribution) |
| Evaluation metrics | Precision, Recall, F1, MRR, nDCG, BLEU, ROUGE-L, BERTScore, RAGAS | Hit Rate, MRR, NDCG, Recall@K, Precision@K, RAGAS (FA + AR), Safety Pass Rate, Latency |

Your study **replicates** the German paper's retrieval and generation comparison approach.
Your study **extends** it with: structured TIME inputs, clinical pre-classification,
patient-facing output, a safety checker, and latency measurement.

---

## Things NOT to Do

- **Do not change the KB or testset** between Stage 1 and Stage 2 experiments.
  Both are gold standard and fixed from day one of the new study.
- **Do not change the RAGAS judge** (gpt-4o-mini + text-embedding-3-small).
  Changing the judge mid-study makes all previous scores incomparable.
- **Do not run only one trial per version.** Run 3 and report mean ± SD.
- **Do not include BLEU, ROUGE, or METEOR** in your FYP metrics. They are not
  appropriate for paraphrased patient-facing clinical recommendations and will
  produce misleadingly low scores that are hard to interpret.
- **Do not evaluate open-source models using Ollama on your local machine
  and closed-source models on the API** in the same experiment — the hardware
  difference will make latency comparisons meaningless. Use HPC for all open-source
  and API for all closed-source; report latency separately for each category.
- **Do not try to run more than two open-source models if HPC time is limited.**
  Qwen3 14B (already validated) + Gemma 3 12B (best in German paper) is a
  sufficient and academically defensible open-source comparison.

---

## Recommended Execution Order

```
Phase 0:  Verify KB (db_wound_care_v4) + testset (wound_testset_v3) are locked.
          Generate chunk-level statistics (total chunks, per-source, avg token length).

Phase 1:  R1 — query strategy   (3 versions, cheap — dense only)
Phase 2:  R2 — retrieval method  (4 versions, using best R1)
Phase 3:  R3 — top-K            (5 versions, using best R1+R2)
Phase 4:  R4 — embedding        (2–3 versions, most expensive — needs KB re-ingestion)

→ Identify Best Retrieval Config

Phase 5:  G1 — prompt strategy  (4 versions, GPT-4o-mini, cheap)
Phase 6:  G2 — closed LLM      (2 versions, GPT-4o-mini + GPT-4o)
Phase 7:  G3 — open-source LLM  (2–4 versions, HPC)
Phase 8:  G4 — patient language  (3 versions, best LLM)

→ Identify Best Generation Config

Phase 9:  Final best-config evaluation (3 runs, full 32 cases)
Phase 10: Small-sample clinician/patient review (5–10 outputs)
```

Total estimated API cost (gpt-4o-mini only, 32 cases × 3 runs):
- RAGAS evaluation: ~$0.10–0.20 per version
- Generation: ~$0.05–0.10 per version
- Approximately $3–6 USD total for all closed-source versions

---

*Last updated: May 2026 | VerdaSense FYP — Universiti Malaya*
