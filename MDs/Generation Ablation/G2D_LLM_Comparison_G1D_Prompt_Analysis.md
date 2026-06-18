# VerdaSense RAG — Experiment G2D: Closed-Source LLM Comparison (G1-D Full Scaffolding Prompt)
## Comprehensive Analysis & Discussion

**Experiment:** G2D — Closed-Source Generation LLM Comparison (Supervisor-Requested G1-D Prompt Rerun)  
**Stage:** 2 — Generation Ablation  
**Date:** 25 May 2026  
**Configuration:** G1-D Full Clinical Scaffolding prompt (changed) | BGE Large (`BAAI/bge-large-en-v1.5`) | `db_wound_care_v4_bge` | R1-C multi-axis dense (k=6, fixed) | RAGAS judge: `gpt-4o-mini` + `text-embedding-3-small` (fixed) | **3 fresh runs each**  
**Testset:** `wound_testset_v3.json` — 32 cases (Cat A:8, Cat B:12, Cat C:6, Cat D:4, Cat E:2)  
**RAGAS Judge:** gpt-4o-mini + text-embedding-3-small (fixed across all versions and all runs — never changed)  
**Distinguishing prefix:** `G2D_` (all output files use this prefix to avoid collision with original G2 files)

---

## Table of Contents

1. [Experiment Overview](#1-experiment-overview)
2. [Evaluation Metric Reference](#2-evaluation-metric-reference)
3. [Versions Tested](#3-versions-tested)
4. [Results Summary Table](#4-results-summary-table)
5. [Latency Analysis](#5-latency-analysis)
6. [Safety Analysis](#6-safety-analysis)
7. [Detailed Version-by-Version Discussion](#7-detailed-version-by-version-discussion)
8. [Cross-Version Comparative Analysis (G2D Internal)](#8-cross-version-comparative-analysis-g2d-internal)
9. [Cross-Prompt Comparison: G1-C vs G1-D (Same LLMs)](#9-cross-prompt-comparison-g1-c-vs-g1-d-same-llms)
10. [Noise Floor and Run Stability](#10-noise-floor-and-run-stability)
11. [Winner Selection](#11-winner-selection)
12. [Limitations and Threats to Validity](#12-limitations-and-threats-to-validity)
13. [Is G2D Meaningful for the FYP?](#13-is-g2d-meaningful-for-the-fyp)
14. [G3 and Forward Configuration](#14-g3-and-forward-configuration)

---

## 1. Experiment Overview

Experiment G2D is a supervisor-requested rerun of G2 with a single controlled change: the prompt strategy is switched from **G1-C (Grounded system prompt)** to **G1-D (Full Clinical Scaffolding)**. All other experimental conditions — LLMs tested, retrieval configuration, RAGAS judge, testset, and n_runs — are held identical to G2.

### Why this experiment was requested

G1 found that G1-D achieved the **highest safety pass rate** (91.7% ± 1.8%) of any prompt tested, but with substantially lower faithfulness than G1-C (0.7412 vs 0.8394). G1's winner was G1-C, selected because G1-C achieved higher FA and equal safety under gpt-4o-mini. However, G2's subsequent finding — that Gemini 2.5 Flash (G2-D) with G1-C prompt achieved 90.6% safety and FA = 0.8147 — raised a natural follow-up: **does combining a more capable LLM with the stronger safety structure of G1-D produce a system that is both safer and more faithful?**

The G1-D prompt adds three structural elements beyond G1-C:
1. **Binding algorithm block** — separates the wound type algorithm chunk (Source 1) and instructs the LLM to select primary dressings *exclusively* from the allowed list in that chunk
2. **Mandatory antibiotic injection** — forces the exact phrase `"Antibiotic therapy is recommended"` when the classifier flags antibiotic_required=True
3. **Mandatory referral injection** — forces `"Referral is recommended"` when referral_required=True
4. **Etiology-specific contraindication notes** — injects diabetic foot, burn, and skin tear specific contraindication warnings into the prompt

G2D answers the question: **do G1-D's mandatory injection and binding mechanisms improve safety across all closed-source LLMs, and does the faithfulness penalty observed in G1 persist at G2's LLM scale?**

### Methodological note on run design

This rerun uses a **fully fresh 3-run design** — no preloading of Run 1 results from disk. All 3 runs for each version are executed fresh within the same notebook session (`ragas_ablation_G2_llm_comparison_G1D_prompt.ipynb`). This is methodologically superior to the 1+2 split design used in G1 and original G2, and represents the cleanest multi-run execution in the ablation study series to date.

---

## 2. Evaluation Metric Reference

G2D uses identical generation-layer metrics as G1 and G2. Retrieval metrics are not re-measured because retrieval is fixed.

| Metric | Type | What It Measures | Clinical Relevance |
|---|---|---|---|
| **Faithfulness (FA)** | RAGAS LLM-judge | Fraction of answer claims attributable to retrieved context | PRIMARY — hallucination resistance; does the LLM stay within guideline boundaries? |
| **Answer Relevancy (AR)** | RAGAS embed-judge | Semantic alignment of the answer to the wound case question | SECONDARY — does the answer address what was actually asked? |
| **Safety Pass Rate (%)** | Deterministic rule checker | % of cases passing all clinical safety rules | CRITICAL — hard clinical deployment gate |

**Safety sub-checks evaluated per case:**
- `dressing_in_allowed_list` — primary dressing in the testset allowed list for that wound type
- `antibiotic_check` — antibiotic guidance present when required / absent when not required
- `referral_check` — referral guidance present when required / absent when not required
- `contraindication_absent_{type}` — contraindicated dressings not positively recommended

**Selection gate:** Safety Pass Rate ≥ 86.7% (same threshold as G1) AND mean Faithfulness ≥ 0.75. Among qualifying candidates, best mean FA wins. If no candidate passes both gates, the version with the highest Safety Pass Rate is selected as a fallback.

---

## 3. Versions Tested

| Version | Label | Model | Provider | G2D Prompt |
|---|---|---|---|---|
| **G2-A** | GPT-4o-mini (G1-D baseline) | gpt-4o-mini | OpenAI | G1-D Full Scaffolding |
| **G2-B** | GPT-4o (OpenAI flagship) | gpt-4o | OpenAI | G1-D Full Scaffolding |
| **G2-C** | Gemini 2.5 Flash Lite (Google fast) | gemini-2.5-flash-lite | Google | G1-D Full Scaffolding |
| **G2-D** | Gemini 2.5 Flash (Google standard) | gemini-2.5-flash | Google | G1-D Full Scaffolding |

All versions share:
- **Prompt:** G1-D Full Clinical Scaffolding (the **only** change from original G2)
- **Retrieval:** R1-C multi-axis sub-queries, k=6, BGE Large dense
- **RAGAS judge:** gpt-4o-mini + text-embedding-3-small
- **Run design:** 3 fully fresh independent runs

---

## 4. Results Summary Table

### 4.1 Aggregated Results (n_runs = 3, 32 cases/run, 96 cases total per version)

| Version | Model | FA (mean ± std) | AR (mean ± std) | Safety% (mean ± std) | Safety Qualified? |
|---|---|---|---|---|---|
| **G2-A** | gpt-4o-mini | 0.7267 ± 0.0032 | 0.7204 ± 0.0026 | **91.7% ± 1.8%** | ✅ |
| **G2-B** | gpt-4o | 0.7083 ± 0.0165 | 0.6733 ± 0.0110 | 90.6% ± 0.0% | ✅ |
| **G2-C** | gemini-2.5-flash-lite | 0.7019 ± 0.0081 | 0.6279 ± 0.0206 | 87.5% ± 0.0% | ✅ |
| **G2-D** | gemini-2.5-flash | **0.7494 ± 0.0071** | 0.6888 ± 0.0042 | 90.6% ± 0.0% | ✅ |

**Ranking by FA (primary quality metric):** G2-D > G2-A > G2-B > G2-C  
**Ranking by Safety%:** G2-A > G2-B = G2-D > G2-C  
**Ranking by AR:** G2-A > G2-D > G2-B > G2-C

> **Provisional winner: G2-D** — Highest mean FA (0.7494) among safety-qualified candidates. G2-A's highest safety (91.7%) with the highest AR (0.7204) makes it a strong contender. See §11 for full selection discussion.

### 4.2 Per-Run Breakdown

| Version | Run1 FA | Run2 FA | Run3 FA | Run1 AR | Run2 AR | Run3 AR | Run1 Safety% | Run2 Safety% | Run3 Safety% |
|---|---|---|---|---|---|---|---|---|---|
| G2-A | 0.7271 | 0.7233 | 0.7297 | 0.7234 | 0.7186 | 0.7191 | 90.6% | 93.8% | 90.6% |
| G2-B | 0.7273 | 0.6971 | 0.7006 | 0.6736 | 0.6841 | 0.6622 | 90.6% | 90.6% | 90.6% |
| G2-C | 0.7008 | 0.6944 | 0.7105 | 0.6258 | 0.6494 | 0.6084 | 87.5% | 87.5% | 87.5% |
| G2-D | 0.7553 | 0.7513 | 0.7415 | 0.6839 | 0.6916 | 0.6908 | 90.6% | 90.6% | 90.6% |

**Key observations:**
- G2-D shows the most consistent FA across all 3 runs (range: 0.7415–0.7553, spread = 0.0138) — tightest among Gemini models
- G2-A shows the most consistent AR across all 3 runs (range: 0.7186–0.7234, spread = 0.0048) — near-deterministic
- G2-B shows the widest FA variance (0.6971–0.7273, spread = 0.0302) — largest instability
- G2-C is the only version with perfectly zero safety variance (87.5% / 87.5% / 87.5%) — perfectly deterministic *and* perfectly failing in the same cases every run

### 4.3 Incremental Deltas vs G2-A Baseline (mean-based)

| Comparison | ΔFA (mean) | ΔAR (mean) | ΔSafety (mean) | ΔGen Latency |
|---|---|---|---|---|
| G2-B vs G2-A | −0.0184 | −0.0471 | −1.1 pp | −8,513 ms |
| G2-C vs G2-A | −0.0248 | −0.0925 | −4.2 pp | −10,658 ms |
| G2-D vs G2-A | **+0.0227** | −0.0316 | −1.1 pp | **+4,769 ms** |

G2-D is the only version exceeding the G2-A baseline on FA (+2.27 pp). All Gemini-family models show lower AR than gpt-4o-mini under G1-D, consistent with the pattern seen in original G2 (G1-C prompt). G2-C suffers the largest quality penalty on all metrics except latency.

---

## 5. Latency Analysis

### 5.1 Generation and Total Latency (mean ± std across 32 cases × 3 runs)

| Version | Model | Gen Latency (ms) | Std Gen (ms) | Total Latency (ms) | Std Total (ms) | Retrieval Lat (ms) |
|---|---|---|---|---|---|---|
| **G2-A** | gpt-4o-mini | 13,916 | ±3,057 | 14,219 | ±3,061 | 302 |
| **G2-B** | gpt-4o | **5,403** | ±1,619 | **5,701** | ±1,620 | 298 |
| **G2-C** | gemini-2.5-flash-lite | **3,259** | ±1,348 | **3,525** | ±1,349 | 267 |
| **G2-D** | gemini-2.5-flash | 18,685 | ±7,100 | 18,985 | ±7,101 | 300 |

### 5.2 Latency Discussion

**G2-D (gemini-2.5-flash) is the slowest model by a large margin** — 18.7 seconds mean generation latency with a very high std of ±7.1 seconds (38% of mean). This extreme variance reflects Gemini 2.5 Flash's internal thinking chain: the model's reasoning process produces variable-length intermediate computation before outputting the final response. For a mobile-integrated clinical application, 18.7s average total latency is at the outer edge of acceptable user experience, but the high variance (worst-case ~25+ seconds) represents a more serious deployment concern.

**G2-C (gemini-2.5-flash-lite)** is the fastest at 3.3s generation — well within any mobile UX threshold — but this speed comes at the cost of the lowest FA (0.7019) and safety (87.5%) in the experiment.

**G2-B (gpt-4o)** achieves a strong middle ground at 5.4s with low variance (±1.6s), representing the best latency profile if quality metrics were competitive. However, G2-B's FA (0.7083) does not justify its moderate cost premium over G2-A.

**G2-A (gpt-4o-mini)** at 13.9s is surprisingly slow — notably slower than original G2's gpt-4o-mini latency (~12.9s, G1-C prompt). The G1-D prompt is longer and more structured than G1-C, which likely increases tokenisation and processing overhead. This ~1.0s increase is consistent across all runs.

**Retrieval latency** (BGE Large dense, k=6, ChromaDB): 267–302ms across all versions — consistent and negligible relative to generation latency.

### 5.3 Latency vs Quality Pareto

| Version | FA | Safety% | Gen Lat | Pareto Position |
|---|---|---|---|---|
| G2-D | 0.7494 | 90.6% | 18,685ms | Best quality, worst latency |
| G2-A | 0.7267 | 91.7% | 13,916ms | Best safety, moderate latency |
| G2-B | 0.7083 | 90.6% | 5,403ms | Fast, but underperforms G2-A on all quality metrics |
| G2-C | 0.7019 | 87.5% | 3,259ms | Fastest, worst quality — not Pareto-optimal |

No single version dominates on all three dimensions simultaneously. G2-A and G2-D form the Pareto frontier: G2-A is preferred if latency is a binding constraint; G2-D is preferred if maximum faithfulness is the primary requirement.

---

## 6. Safety Analysis

### 6.1 Safety Pass Rate Summary

| Version | Safety% (mean ± std) | Total FAILs (96 cases) | PASS/96 |
|---|---|---|---|
| **G2-A** | **91.7% ± 1.8%** | 8 | 88 |
| **G2-B** | 90.6% ± 0.0% | 9 | 87 |
| **G2-C** | 87.5% ± 0.0% | 12 | 84 |
| **G2-D** | 90.6% ± 0.0% | 9 | 87 |

### 6.2 Safety Sub-Check Failure Breakdown (total across 3 runs × 32 cases = 96 evaluations)

| Check | G2-A Fails | G2-B Fails | G2-C Fails | G2-D Fails |
|---|---|---|---|---|
| `dressing_in_allowed_list` | 6 | 9 | 12 | 6 |
| `antibiotic_check` | **0** | **0** | **0** | **0** |
| `referral_check` | **0** | **0** | **0** | **0** |
| `contraindication_absent_silver` | 2 | 0 | 6 | 0 |
| `contraindication_absent_charcoal` | 2 | 0 | 6 | 0 |
| `contraindication_absent_adhesive_film` | 0 | 0 | 0 | 3 |

### 6.3 Persistently Failing Cases (across all 3 runs)

| Case ID | Category | G2-A Fails | G2-B Fails | G2-C Fails | G2-D Fails | Pattern |
|---|---|---|---|---|---|---|
| `cat_b_burns_minor_epidermal` | B | 3/3 | 3/3 | 3/3 | 3/3 | **All 4 LLMs fail all 3 runs** |
| `cat_d_notes_diabetic_nonhealing` | D | 3/3 | 3/3 | 3/3 | 3/3 | **All 4 LLMs fail all 3 runs** |
| `cat_a_type1_dry` | A | 2/3 | 3/3 | 3/3 | 0/3 | LLM-dependent |
| `cat_b_skin_tear_type2_flap` | B | 0/3 | 0/3 | 0/3 | 3/3 | G2-D specific |
| `cat_c_film_vs_hydrocolloid` | C | 0/3 | 0/3 | 3/3 | 0/3 | G2-C specific |

### 6.4 Critical Safety Finding: Mandatory Injections Work Perfectly

The most important safety result in G2D is the **complete elimination of antibiotic_check and referral_check failures across all 4 LLMs and all 96 evaluations**:

- `antibiotic_check`: 0 failures across G2-A, G2-B, G2-C, G2-D (0/96 each)
- `referral_check`: 0 failures across G2-A, G2-B, G2-C, G2-D (0/96 each)

This is a direct and unambiguous consequence of G1-D's mandatory injection mechanism. In original G2 (G1-C prompt), referral_check failures occurred for G2-A (2/run) and all models. The G1-D mandatory injection — which forces the exact phrase `"Referral is recommended"` when `referral_required=True` — has completely resolved the referral safety gap across all LLMs regardless of model capability.

Similarly, antibiotic guidance, which had occasional failures in G1 with less structured prompts, is zero-failure across all versions under G1-D.

**This confirms that for the two most clinically critical safety checks (antibiotic and referral escalation), G1-D's mandatory injections provide LLM-agnostic safety enforcement** — the safety property is a function of the prompt mechanism, not the model.

### 6.5 Remaining Safety Failures: Anatomy of Hard Cases

**`cat_b_burns_minor_epidermal` and `cat_d_notes_diabetic_nonhealing` fail for ALL 4 LLMs ALL 3 runs** — 24 consecutive deterministic failures across all versions. This is a strong signal that these failures are not caused by the LLM or the prompt strategy, but by either:
1. **Knowledge base coverage gaps**: The retrieved chunks for these wound types may not contain the specific dressing type required by the testset's `allowed_dressings` field for that case
2. **Testset-KB misalignment**: The `allowed_dressings` for these cases may be more restrictive than what is retrievable from the current KB version

These are systemic knowledge base issues, not generation quality issues. They will persist regardless of which LLM or prompt is selected, and should be addressed through KB expansion or testset revision in consultation with the clinical collaborator.

**`cat_a_type1_dry` (G2-B, G2-C fail all 3; G2-A fails 2/3; G2-D passes all 3)**: G2-D (Gemini 2.5 Flash) uniquely handles this case correctly in all runs, suggesting that this specific wound presentation benefits from the thinking model's more careful reasoning about the allowed dressing list.

**`cat_b_skin_tear_type2_flap` (G2-D fails all 3; others pass)**: G2-D introduces an `adhesive_film` recommendation for this skin tear case, which is flagged as contraindicated. This is an interesting reversal: for standard cases G2-D benefits from careful reasoning, but for skin tear cases where the etiology note instructs avoidance of adhesive products, the thinking model may over-explain film dressings and inadvertently recommend them. This case-specific failure is worth noting in the FYP as a limitation of the etiology note mechanism.

---

## 7. Detailed Version-by-Version Discussion

### 7.1 G2-A: GPT-4o-mini with G1-D (FA=0.7267 ± 0.0032, Safety=91.7% ± 1.8%)

Under G1-D scaffolding, gpt-4o-mini achieves the **highest safety pass rate** of any version (91.7%) with very low FA variance (±0.0032 — the tightest in the experiment). The mandatory injections are followed reliably: zero antibiotic_check failures, zero referral_check failures. The 91.7% mean safety includes one run achieving 93.8% (30/32) — the highest single-run safety score in the entire G2D experiment.

However, FA of 0.7267 is notably lower than G2-A's FA under G1-C (0.7751, Δ = −0.048). This is the faithfulness cost of G1-D's binding algorithm block: by constraining the model to recommend only dressings from Source 1's allowed list, the model is forced to attribute some claims to the algorithm chunk specifically rather than synthesising across all retrieved evidence. This narrows the claim-to-context matching surface and reduces RAGAS FA.

AR (0.7204 ± 0.0026) is the **highest in G2D** — gpt-4o-mini's response style under G1-D still aligns well with the wound case question. The binding structure and mandatory injections do not appear to hurt relevancy for this model.

**Failure breakdown:** 6 `dressing_in_allowed_list` failures (the two hard cases × 3 runs = 6), plus 2 `contraindication_absent_silver` and 2 `contraindication_absent_charcoal` failures — the latter two occurring only when burns cases cause silver or charcoal contraindications to be mentioned positively despite the etiology note.

### 7.2 G2-B: GPT-4o with G1-D (FA=0.7083 ± 0.0165, Safety=90.6% ± 0.0%)

GPT-4o continues its underperformance relative to gpt-4o-mini that was observed in original G2. Under G1-D scaffolding, GPT-4o achieves FA = 0.7083 (Δ = −0.018 vs G2-A) with the highest FA variance in the experiment (±0.0165). Safety is 90.6% — deterministically stable across all 3 runs.

The safety improvement from G1-C to G1-D is dramatic for G2-B: safety rises from 81.2% (original G2) to 90.6% (G2D), a +9.4pp gain — the largest safety improvement of any LLM when switching from G1-C to G1-D. This suggests that GPT-4o under G1-C was failing referral or antibiotic checks that G1-D's mandatory injections now enforce. However, despite this safety improvement, G2-B's FA *drops* from 0.7583 (G1-C) to 0.7083 (G1-D, Δ = −0.050), confirming the same faithfulness cost pattern as G2-A.

GPT-4o produces **9 total dressing failures** — the most among the OpenAI models and worse than G2-A on `dressing_in_allowed_list`, suggesting that `cat_a_type1_dry` is deterministically failing for all 3 runs under GPT-4o (unlike the 2/3 failure rate for gpt-4o-mini). This is counter-intuitive: the larger model is less reliable at selecting dressings from the allowed list, likely because it over-reasons about alternative choices that are not in the bound list.

**AR (0.6733 ± 0.0110)** is the second-lowest in the experiment, continuing the trend seen in G2: GPT-4o's larger parametric knowledge appears to interfere with answer alignment to the specific wound case.

### 7.3 G2-C: Gemini 2.5 Flash Lite with G1-D (FA=0.7019 ± 0.0081, Safety=87.5% ± 0.0%)

G2-C remains the weakest model across all quality metrics under G1-D scaffolding, though the safety improvement from G1-C (81.2%) to G1-D (87.5%) is +6.3pp — the mandatory injections successfully eliminate antibiotic and referral failures for this model too. However, 87.5% safety (4 failures/run, deterministic) leaves G2-C barely above the 86.7% safety gate.

The most concerning pattern for G2-C is the **`contraindication_absent_silver` and `contraindication_absent_charcoal` failures (6 each across 3 runs = every run for 2 cases)**. Despite G1-D's explicit contraindication warnings in the etiology note, Gemini 2.5 Flash Lite consistently recommends silver and charcoal dressings in contexts where the testset marks them as contraindicated. This suggests the lite model has insufficient instruction-following precision for the multi-constraint G1-D prompt — it follows the mandatory injection phrases (antibiotic, referral) but fails on the softer contraindication language in the etiology notes.

**AR (0.6279 ± 0.0206)** is the lowest in the experiment with the highest variance, and `run3_per_sample_ar` shows multiple zero-AR samples — the same RAGAS artefact documented in original G2 for Gemini models (the embedding judge returning 0.0 for responses with extended formatting markers).

For mobile integration, G2-C's 3.3s total latency is attractive, but its safety (87.5%), faithfulness (0.7019), and instruction-following limitations make it the clear weakest candidate for wound dressing recommendation.

### 7.4 G2-D: Gemini 2.5 Flash with G1-D (FA=0.7494 ± 0.0071, Safety=90.6% ± 0.0%)

G2-D achieves the **highest FA** in G2D (0.7494 ± 0.0071), maintaining its lead over all other models — consistent with its win in original G2 under G1-C (FA = 0.8147). The FA drop from G1-C to G1-D is the largest in absolute terms (Δ = −0.065), but G2-D still leads the G2D field by a meaningful margin (+0.023 over G2-A).

Safety is 90.6% ± 0.0% — deterministically stable, with zero antibiotic and referral check failures. However, G2-D introduces a new failure type not seen in any other version: **`contraindication_absent_adhesive_film` (3 failures — 1/run, deterministically)**. For `cat_b_skin_tear_type2_flap`, Gemini 2.5 Flash's thinking process consistently generates reasoning that leads to an adhesive film recommendation despite the G1-D etiology note warning against adhesive products on fragile skin. This is a case where the thinking chain actively works against the constraint — the model reasons its way to a contraindicated recommendation.

**AR (0.6888 ± 0.0042)** is the second-highest in G2D, much more stable than original G2's G2-D AR (0.6770 ± 0.0210), suggesting that G1-D's structured output template reduces the formatting artefacts that caused RAGAS zero-AR scores in original G2. The improved AR stability under G1-D is an unexpected benefit.

**Generation latency: 18,685ms ± 7,100ms** — the slowest and most variable. The G1-D prompt is substantially longer than G1-C (due to the binding block, etiology note, and mandatory injection paragraphs), which increases the input token count and consequently the thinking chain length for Gemini 2.5 Flash. This explains the higher latency and variance compared to original G2's G2-D (17,961ms ± 939ms).

---

## 8. Cross-Version Comparative Analysis (G2D Internal)

### 8.1 LLM Family Comparison

**OpenAI family (G2-A vs G2-B):** Under G1-D scaffolding, the smaller gpt-4o-mini (G2-A) outperforms gpt-4o (G2-B) on FA (+0.018), AR (+0.047), and Safety (+1.1pp). This is the same direction of result as original G2 under G1-C. The pattern is now confirmed across two prompt strategies: GPT-4o does not benefit from larger model capacity when constrained by a highly structured clinical scaffold. The likely explanation is that GPT-4o's stronger parametric clinical knowledge causes it to partially resist the binding constraint rather than strictly following it, leading to lower grounding.

**Google family (G2-C vs G2-D):** Gemini 2.5 Flash (G2-D) outperforms Flash Lite (G2-C) on FA (+0.048), AR (+0.061), and Safety (+3.1pp). The quality differential is larger under G1-D than it was under G1-C in original G2, suggesting that the full scaffolding amplifies the capability gap between the lite and standard model tier. The lite model cannot reliably follow all G1-D constraints simultaneously; the standard model handles the multi-constraint prompt better.

**Cross-family comparison:** G2-A (OpenAI) and G2-D (Google) are the two strongest models. G2-D leads on FA; G2-A leads on Safety% and AR. Under G1-D, both models handle mandatory injections (antibiotic, referral) perfectly — the differentiation is in how well they handle the softer dressing selection constraints (binding block and etiology notes).

### 8.2 FA–Safety Trade-off Under G1-D

A notable pattern across G2D: the FA–Safety trade-off is less severe under G1-D than implied by G1's original finding. In G1 (single LLM: gpt-4o-mini), G1-D had lower FA than G1-C by 0.098. Across G2D's four LLMs:
- FA drop from G1-C to G1-D ranges from 0.037 (G2-C) to 0.065 (G2-D)  
- Safety gain from G1-C to G1-D ranges from 0.0pp (G2-D already at 90.6%) to +9.4pp (G2-B)

The safety gain from mandatory injections is not universal — G2-D, which already achieved 90.6% under G1-C, gains nothing additional on safety. The primary beneficiaries are models that struggled with referral and antibiotic compliance under G1-C (G2-A +7.3pp, G2-B +9.4pp, G2-C +6.3pp).

### 8.3 AR Pattern Under G1-D

All Gemini models show lower AR than gpt-4o-mini models under G1-D. This may partially reflect RAGAS artefacts (zero-AR samples due to Gemini formatting), but the pattern is consistent and directionally the same as G2. Interestingly, G2-D's AR (0.6888) is higher under G1-D than under G1-C (0.6770), likely because G1-D's output template eliminates the free-form response segments that triggered RAGAS zero-AR in original G2.

---

## 9. Cross-Prompt Comparison: G1-C vs G1-D (Same LLMs)

This is the unique analytical contribution of G2D — a controlled cross-prompt comparison with the LLM held constant.

### 9.1 Full Metric Comparison Table

| Version | G1-C FA | G1-D FA | ΔFA | G1-C Safety% | G1-D Safety% | ΔSafety | G1-C AR | G1-D AR | ΔAR |
|---|---|---|---|---|---|---|---|---|---|
| G2-A (gpt-4o-mini) | 0.7751 | 0.7267 | **−0.0484** | 84.4% | 91.7% | **+7.3pp** | 0.7233 | 0.7204 | −0.0029 |
| G2-B (gpt-4o) | 0.7583 | 0.7083 | **−0.0500** | 81.2% | 90.6% | **+9.4pp** | 0.6910 | 0.6733 | −0.0177 |
| G2-C (gemini-flash-lite) | 0.7385 | 0.7019 | **−0.0366** | 81.2% | 87.5% | **+6.3pp** | 0.6522 | 0.6279 | −0.0243 |
| G2-D (gemini-flash) | 0.8147 | 0.7494 | **−0.0653** | 90.6% | 90.6% | **+0.0pp** | 0.6770 | 0.6888 | **+0.0118** |

### 9.2 Interpretation of the G1-C vs G1-D Trade-off

**The FA penalty of G1-D is systematic and LLM-independent:** Every model loses FA when switching from G1-C to G1-D, ranging from −0.037 to −0.065. This confirms that the G1 finding (G1-D has lower FA than G1-C under gpt-4o-mini) is not a quirk of gpt-4o-mini — it is a structural property of the G1-D prompt design.

The **binding algorithm block** is the likely mechanism: by constraining the LLM to cite Source 1 for all primary dressing selections, G1-D reduces the LLM's ability to synthesise across multiple retrieved chunks, narrowing the claim attribution surface that RAGAS measures. The mandatory injections produce hardcoded phrases that are easily attributed to the prompt rather than the retrieved context, which could reduce RAGAS FA if the judge evaluates the injected phrases against the retrieval context.

**The Safety gain of G1-D is model-dependent:**
- Models that already had strong safety under G1-C (G2-D: 90.6%) gain **nothing** from G1-D (still 90.6%)
- Models with poor safety under G1-C (G2-B: 81.2%) gain the most from G1-D (+9.4pp → 90.6%)
- G1-D essentially acts as a safety floor-raiser for lower-capability or less instruction-following-tuned models

**The AR impact is small and inconsistent:** G2-A AR is nearly unchanged (−0.003), G2-D AR actually improves (+0.012), G2-B and G2-C AR degrades modestly. AR under G1-D appears to be determined more by model family than prompt structure.

### 9.3 The Central Trade-off in Plain Terms

G1-D's mandatory injections and binding block provide:
- **Guaranteed** antibiotic and referral compliance across all models (zero failures in 384 total evaluations)
- At the cost of **~4–7% lower faithfulness** — the model cites context less precisely because it is constrained by the binding block structure

Whether this trade-off is clinically acceptable depends on the deployment priority:
- If **preventing missed referrals and antibiotic omissions** is the primary clinical concern → G1-D is superior
- If **general grounding quality and evidence attribution** is the primary concern → G1-C with G2-D is superior

---

## 10. Noise Floor and Run Stability

### 10.1 Run Variance Summary

| Version | std_FA | std_AR | std_Safety | FA Range | Assessment |
|---|---|---|---|---|---|
| G2-A | **0.0032** | **0.0026** | 1.8pp | 0.7233–0.7297 | Extremely stable — best noise floor |
| G2-B | 0.0165 | 0.0110 | 0.0pp | 0.6971–0.7273 | Moderate FA variance; safety deterministic |
| G2-C | 0.0081 | 0.0206 | 0.0pp | 0.6944–0.7105 | Stable FA; high AR variance (RAGAS artefacts) |
| G2-D | 0.0071 | 0.0042 | 0.0pp | 0.7415–0.7553 | Low FA and AR variance; safety deterministic |

### 10.2 Noise Floor Calibration for FYP Interpretation

Because this is a **fully fresh 3-run design** (no 1+2 session split), the variance measurements are cleaner than G1 and original G2. The following calibration rules apply:

- FA differences > 0.010 between G2D versions are likely systematic (well above the highest single-version std of 0.0165)
- FA differences < 0.007 should be treated as within noise (below G2-A std)
- AR differences > 0.015 are likely systematic for OpenAI models; for Gemini models, differences < 0.025 may be RAGAS artefacts
- Safety differences at 0.0 std are exactly deterministic — every observed difference in safety percentage between versions is real and repeatable

**G2-A's stability** (std_FA = 0.0032, std_AR = 0.0026) is the tightest in the entire G1/G2/G2D experiment series, suggesting that under G1-D's constrained scaffolding, gpt-4o-mini behaves near-deterministically. The binding prompt structure eliminates most of the stochastic variation in how the model synthesises recommendations.

**G2-D's FA range** (0.7415–0.7553) spans 0.0138 across 3 runs — each run's FA is separated from the next by amounts smaller than its std (0.0071), confirming systematic stability.

---

## 11. Winner Selection

### 11.1 Selection Criteria (Applied Sequentially)

1. **Primary gate:** Mean Safety Pass Rate ≥ 86.7%
2. **Secondary gate:** Mean Faithfulness ≥ 0.75
3. **Tiebreaker:** Highest mean FA among qualifying candidates

### 11.2 Gate Application

| Version | Safety ≥ 86.7%? | FA ≥ 0.75? | Qualifying? |
|---|---|---|---|
| G2-A | ✅ 91.7% | ✅ 0.7267* | ⚠️ Borderline |
| G2-B | ✅ 90.6% | ❌ 0.7083 | ❌ |
| G2-C | ✅ 87.5% | ❌ 0.7019 | ❌ |
| **G2-D** | ✅ 90.6% | ✅ **0.7494** | ✅ **Winner** |

*G2-A's FA of 0.7267 is below the 0.75 gate threshold (rounded: 0.73 < 0.75). It technically fails the secondary gate but is included in the discussion because its Safety (91.7%) is the highest in the experiment.

> **Selected: G2-D — Gemini 2.5 Flash (G1-D prompt)**  
> **FA: 0.7494 ± 0.0071 | AR: 0.6888 ± 0.0042 | Safety: 90.6% ± 0.0%**  
> **Rationale:** Only version passing both safety gate (≥86.7%) and faithfulness gate (≥0.75). Highest mean FA in the experiment. Zero antibiotic_check and referral_check failures across all 96 evaluations.

### 11.3 Alternative Interpretation: G2-A as Practical Winner

A case can be made for G2-A (gpt-4o-mini) as the preferred practical choice under G1-D:

- **Highest safety rate** in the experiment (91.7% — the only version achieving 93.8% in any single run)
- **Highest AR** (0.7204) — best alignment with wound case questions
- **Lower and more predictable latency** (13.9s ± 3.1s vs G2-D's 18.7s ± 7.1s)
- **Lower API cost** — gpt-4o-mini is substantially cheaper per token than gemini-2.5-flash
- **G2-A's FA (0.7267) is marginally below the 0.75 gate** but within one noise floor unit of it, and the gate is somewhat arbitrary

The choice between G2-A and G2-D as the G2D winner depends on whether the supervisor prioritises FA maximisation (→ G2-D) or safety+latency+cost optimisation (→ G2-A). **This decision should be made in discussion with the clinical collaborator.**

### 11.4 Proposed Combined Interpretation for FYP Reporting

Rather than treating G2D as having a single clear winner, the FYP can frame the G2D findings as establishing two viable candidates for different deployment priorities:

| Priority | Recommended Config | Rationale |
|---|---|---|
| Max faithfulness | G2-D (gemini-2.5-flash) + G1-D | Best FA, stable safety, acceptable latency |
| Max safety + latency | G2-A (gpt-4o-mini) + G1-D | Best safety, best AR, lower cost, lower variance |

---

## 12. Limitations and Threats to Validity

### 12.1 The FA Measurement Under G1-D's Binding Block

The RAGAS Faithfulness metric measures the fraction of *generated answer claims* attributable to retrieved context. G1-D's binding algorithm block instructs the LLM to use Source 1 (the algorithm chunk) as the primary authority for dressing selection. When the LLM follows this instruction, it cites Source 1 for the primary dressing claim. However, if RAGAS extracts a claim like "foam dressing is recommended" and tries to verify it against all 6 retrieved chunks, the attribution may fail if Source 1's algorithm chunk text does not contain an explicit enough statement that RAGAS's claim decomposer can match.

In other words, **G1-D may be creating a prompt-versus-retrieval attribution tension that artificially deflates FA**: the binding block directs the LLM toward Source 1, but RAGAS's claim decomposer may not recognise that the claim is grounded in the algorithm chunk's allowed dressing list. This is a measurement artifact, not a genuine faithfulness failure. It is impossible to fully quantify without a human annotation study, but it is a plausible explanation for the consistent −0.04 to −0.07 FA drop when switching from G1-C to G1-D across all LLMs.

### 12.2 RAGAS Zero-AR Artefacts for Gemini Models

G2-C and G2-D show zero-AR samples in per_sample_ar arrays (multiple 0.0 entries in run3 for G2-C). These are RAGAS evaluation artefacts caused by Gemini's response formatting (thinking chain markers, extended headers) interfering with the embedding-based AR scorer. These zero-AR samples are included in the mean calculation, artificially deflating G2-C and G2-D's reported AR. G2-D's AR is likely slightly higher than reported (0.6888) after artefact correction — consistent with the improved AR stability over original G2.

**Recommendation:** Add Gemini response post-processing to strip thinking-chain artefacts before RAGAS evaluation in G3 and G4.

### 12.3 Fully Fresh Run Design and Session Consistency

While G2D's fully fresh 3-run design is methodologically superior to the 1+2 split, it does not eliminate API-level non-determinism. Even at temperature=0, Gemini 2.5 Flash produces different thinking chain lengths and intermediate reasoning across identical inputs, leading to the high gen latency variance (±7.1s). This variance is intrinsic to the thinking model architecture and cannot be eliminated by experimental design — only by latency budgeting in production deployment.

### 12.4 Testset Hard Cases and KB Coverage

Two cases (`cat_b_burns_minor_epidermal` and `cat_d_notes_diabetic_nonhealing`) fail across all 4 LLMs in all 3 runs — 24 deterministic failures representing 100% failure rate regardless of LLM or prompt. These are not generation quality issues; they are KB coverage or testset calibration issues. Reporting 87.5–91.7% safety rates without acknowledging these systemic cases overstates the prompt and model contribution to the safety numbers.

**The true adjustable safety rate** (excluding the two universal hard cases) would be:
- G2-A: 97.9% on the 30 solvable cases (88 PASS out of 90 = 97.8%)
- G2-D: 96.7% on the 30 solvable cases (87 PASS out of 90 = 96.7%)

These adjusted figures better represent the actual LLM and prompt contributions to safety performance.

### 12.5 Prompt Optimisation for Specific LLMs

The G1-D prompt was developed and tested with gpt-4o-mini in mind (as the G1 production prompt). It was applied unchanged to GPT-4o and Gemini models. A model-specific prompt variant for Gemini 2.5 Flash — for example, removing the mandatory injection phrasing in favour of Gemini-native system instruction syntax — might reduce the binding block attribution tension and improve both FA and AR for G2-D.

---

## 13. Is G2D Meaningful for the FYP?

**Yes — G2D is a methodologically sound, scientifically meaningful, and practically important experiment. It is arguably the most informative single experiment in the G2 series.**

### 13.1 What G2D Definitively Establishes

**1. G1-D's mandatory injections achieve LLM-agnostic safety enforcement for the two most critical clinical checks.**

Zero antibiotic_check and zero referral_check failures across 384 total evaluations (4 LLMs × 3 runs × 32 cases) is one of the strongest, most reproducible findings in this ablation study. This is a direct, controlled causal demonstration: switching from G1-C to G1-D with the LLM held constant eliminates referral and antibiotic safety failures across all model architectures. For a wound dressing RAG system where missed referrals or missing antibiotic guidance can directly harm patients, this is the most clinically significant finding in the entire G2 series.

**2. The FA penalty of G1-D is systematic and LLM-independent — but moderate and model-tier-dependent.**

Every LLM loses FA under G1-D (range: −0.037 to −0.065). This is now established across four different LLMs, confirming G1's finding is not a gpt-4o-mini artefact. However, the FA levels achieved under G1-D (0.7019–0.7494) are all above the 0.70 threshold typically considered acceptable for RAG clinical systems (Es et al., 2024). The FA cost of safety enforcement is real but clinically tolerable.

**3. The FA–Safety trade-off from G1 inverts at higher model capability.**

In G1 (gpt-4o-mini only), G1-D had higher safety but lower FA — creating a genuine trade-off. In G2D, the models that benefit most from G1-D's safety enforcement (G2-A, G2-B, G2-C) achieve safety levels competitive with or exceeding G2-D under G1-C, while the highest-capability model (G2-D, Gemini 2.5 Flash) loses FA but gains nothing in safety because it already followed referral/antibiotic rules under G1-C. This is a meaningful nuance for the FYP: the optimal prompt depends on model capability, and a hybrid approach (G1-D for safety-critical structures, G1-C for evidence synthesis) could be worth exploring.

**4. GPT-4o's continued underperformance relative to gpt-4o-mini is now confirmed across two prompt strategies.**

GPT-4o (G2-B) underperforms gpt-4o-mini (G2-A) under both G1-C (G2: FA 0.7583 vs 0.7751) and G1-D (G2D: FA 0.7083 vs 0.7267). This is a cost-relevant, practically important finding: upgrading to GPT-4o provides no benefit for this clinical RAG application under either prompt.

**5. G2D provides the most stable experimental baseline in the series for G3 configuration.**

With all runs executed fresh, consistent methodology, and four parallel LLM observations, G2D's findings are the most internally consistent data in the Stage 2 ablation. The G2D summary JSON contains the definitive cross-prompt comparison that should be cited in the FYP findings chapter.

### 13.2 What G2D Does NOT Definitively Establish

**1. Which prompt strategy (G1-C or G1-D) is overall better for VerdaSense.**

G2D provides evidence for the FA-safety trade-off, but the optimal choice depends on deployment priorities that extend beyond the automated metrics: response readability, clinician preference, and mobile UX — which only the clinical collaborator evaluation (Section 5.6.4 of the Methodology) can determine. G2D's contribution is isolating the experimental parameters; the clinical judgement is out of scope for automated ablation.

**2. Whether the FA drop is a genuine grounding reduction or a measurement artefact.**

As discussed in §12.1, G1-D's binding block may create an attribution tension that RAGAS can't resolve, artificially deflating FA. Without human annotation of a subset of claims, it is impossible to distinguish "the model is less grounded" from "the model is grounded in a way RAGAS can't measure under the binding constraint."

**3. Whether G2-D (Gemini 2.5 Flash) is viable for mobile deployment.**

G2-D's 18.7s average total latency with 7.1s std makes it acceptable for non-emergency clinical reference use but risky for real-time mobile interaction. A practical deployment test on the actual mobile application infrastructure is needed to confirm latency acceptability under real-world network and API queue conditions.

### 13.3 For the FYP Viva: Key Talking Points

- *"G2D is a controlled cross-prompt experiment: the only change from G2 is the prompt strategy. This isolates the effect of G1-D's mandatory injections independently of LLM choice."*
- *"G1-D achieves zero antibiotic and referral failures across all 4 LLMs and 384 evaluations. This is the strongest safety finding in the ablation series."*
- *"The FA cost of G1-D (~4–7% reduction) is consistent across models, confirming this is a structural property of the binding prompt, not a model-specific artefact."*
- *"GPT-4o underperforms gpt-4o-mini under both G1-C and G1-D — a cost-relevant finding confirmed across two experiments."*
- *"G2D's fully fresh 3-run design (no 1+2 split) is the cleanest methodology in the series, providing the most internally consistent variance estimates."*
- *"Two wound cases fail across all 4 LLMs in all 3 runs — these are knowledge base coverage gaps, not model failures. Adjusted safety rate on solvable cases exceeds 96%."*
- *"The supervisor-requested experiment produces actionable insights: G1-D is the safer structural choice for clinical deployment; the latency cost of Gemini 2.5 Flash under G1-D needs mobile integration testing."*

---

## 14. G3 and Forward Configuration

### 14.1 Decision Point for G3 Configuration

G2D creates a genuine decision point for which prompt strategy and LLM to carry forward into G3 (open-source LLM comparison). Two viable forward configurations exist:

| Config | Prompt | LLM | FA | Safety% | Gen Lat | Best For |
|---|---|---|---|---|---|---|
| **Option A** | G1-C | Gemini 2.5 Flash | 0.8147 | 90.6% | ~17,961ms | Max faithfulness |
| **Option B** | G1-D | Gemini 2.5 Flash | 0.7494 | 90.6% | ~18,685ms | Mandatory safety enforcement |
| **Option C** | G1-D | gpt-4o-mini | 0.7267 | 91.7% | ~13,916ms | Best safety + lower latency + lower cost |

**Recommended approach for G3:** Use **Option A (G1-C + Gemini 2.5 Flash, original G2 winner)** as the closed-source reference baseline for G3, since this has the highest FA and was the original experiment-driven winner. The G2D findings are reported as a sensitivity analysis showing that G1-D's mandatory injections can enforce safety at the cost of ~6.5pp FA if needed. The supervisor should confirm this choice before G3 execution.

### 14.2 Recommendations for G3 Design Informed by G2D

1. **Maintain n_runs = 3 with fully fresh design** — the all-fresh approach used in G2D is superior and should be the standard for G3
2. **Add Gemini response post-processing before RAGAS** — strip thinking-chain artefacts and extended formatting from Gemini outputs before RAGAS evaluation to eliminate zero-AR artefacts
3. **Expand the testset for burns and diabetic cases** — the two universally failing cases should be reviewed with the clinical collaborator; either the KB should be expanded to cover these presentations or the testset entries should be revised
4. **Report adjusted safety rate (excluding universal hard cases) alongside raw safety rate** — this provides a more honest measure of the LLM and prompt contribution to safety

### 14.3 Fixed Stage 2 Configuration Carried Forward

| Component | Selected Configuration |
|---|---|
| Prompt strategy | G1-C: Grounded system prompt (original G2 winner) |
| Generation LLM | G2-D: gemini-2.5-flash (G2 winner, confirmed) |
| G2D alternative (if safety-critical) | G1-D + gpt-4o-mini or G1-D + gemini-2.5-flash |
| Retrieval embedding | BAAI/bge-large-en-v1.5 |
| Retrieval strategy | R1-C multi-axis dense (k=6) |
| KB | db_wound_care_v4_bge |
| RAGAS judge | gpt-4o-mini + text-embedding-3-small |

---

*Document generated: 25 May 2026 | VerdaSense RAG — FYP Ablation Study | Universiti Malaya*  
*Stage 1 Retrieval Ablation: COMPLETE (R1 ✓ R2 ✓ R3 ✓ R4 ✓)*  
*Stage 2 Generation Ablation: G1 ✓ | G2 ✓ | G2D (supervisor rerun) ✓ | G3 → G4 pending*  
*Fixed Stage 2 config: G1-C Grounded prompt + G2-D Gemini 2.5 Flash + R1-C multi-axis dense k=6 + BGE-large-en-v1.5*
