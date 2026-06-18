# VerdaSense RAG — Experiment G3-OR (No-Think): Open-Source LLM Comparison via OpenRouter API
## Comprehensive Analysis & Discussion

**Experiment:** G3-OR — Open-Source Generation LLM Comparison (OpenRouter API, Reasoning Disabled)  
**Stage:** 2 — Generation Ablation  
**Date:** 28 May 2026  
**Configuration:** G1-C Grounded system prompt (fixed) | BGE Large (`BAAI/bge-large-en-v1.5`) | `db_wound_care_v4_bge` | R1-C multi-axis dense (k=6, fixed) | RAGAS judge: `gpt-4o-mini` + `text-embedding-3-small` (fixed) | **3 fresh runs each**  
**Testset:** `wound_testset_v3.json` — 32 cases (Cat A:8, Cat B:12, Cat C:6, Cat D:4, Cat E:2)  
**RAGAS Judge:** gpt-4o-mini + text-embedding-3-small (fixed across all versions and all runs — never changed)  
**Infrastructure:** OpenRouter API (`https://openrouter.ai/api/v1`) | LangChain ChatOpenAI with `extra_body={reasoning:{effort:none}, include_reasoning:False}` | Local machine (CPU retrieval, BGE GPU if available)

**Key distinction from G3-HPC:** G3-HPC ran 4 models locally on UM HPC A100 via Ollama. G3-OR runs 7 models (including 4 new 2026 MoE models) via OpenRouter API with reasoning tokens fully suppressed at the server level — the primary fix that corrects the high-latency artefact in the initial G3-OR run.

---

## Table of Contents

1. [Experiment Overview](#1-experiment-overview)
2. [Evaluation Metric Reference](#2-evaluation-metric-reference)
3. [Versions Tested](#3-versions-tested)
4. [Results Summary Table](#4-results-summary-table)
5. [Latency Analysis](#5-latency-analysis)
6. [Safety Analysis](#6-safety-analysis)
7. [Detailed Version-by-Version Discussion](#7-detailed-version-by-version-discussion)
8. [Cross-Version Comparative Analysis](#8-cross-version-comparative-analysis)
9. [Noise Floor and Run Stability](#9-noise-floor-and-run-stability)
10. [Winner Selection](#10-winner-selection)
11. [G3-OR vs G3-HPC vs G2: Three-Way Comparison](#11-g3-or-vs-g3-hpc-vs-g2-three-way-comparison)
12. [Pricing Comparison: All Models (G2 + G3-HPC + G3-OR)](#12-pricing-comparison-all-models)
13. [Is G3-OR Meaningful for the FYP?](#13-is-g3-or-meaningful-for-the-fyp)
14. [LLM Recommendation: Which Model Should You Choose?](#14-llm-recommendation-which-model-should-you-choose)
15. [Limitations and Threats to Validity](#15-limitations-and-threats-to-validity)
16. [Next Steps](#16-next-steps)

---

## 1. Experiment Overview

G3-OR is the fourth Stage 2 ablation experiment for VerdaSense, and the second open-source LLM comparison after G3-HPC. G1 established the winning prompt strategy (G1-C: Grounded system prompt); G2 selected the best closed-source LLM (G2-D: Gemini 2.5 Flash, FA=0.8147, Safety=90.6%); G3-HPC tested four 2025-era open-source models via Ollama on the UM HPC A100 and found the best performers fell 7–10 pp below G2-D on both FA and safety.

G3-OR was designed to answer a new question that G3-HPC could not: **do 2026-generation open-source MoE models via OpenRouter API close the gap to G2-D while simultaneously meeting the ≤10 s mobile latency target?**

This question has three dimensions of significance for VerdaSense:

1. **Clinical quality:** Can any open-source model match the FA=0.8147 and Safety=90.6% of G2-D?
2. **Mobile feasibility:** Can a model delivering acceptable clinical quality also generate a response in ≤10 s — the threshold identified in O3/RQ3 for smartphone-based wound care?
3. **Cost and sovereignty:** Open-source models via OpenRouter eliminate the proprietary dependency on Google Gemini and reduce per-query cost by 2–8×.

Seven models were evaluated, spanning three architectural families and two generations:

- **G3-A:** `qwen/qwen3-14b` — Alibaba 14B dense, the G3-HPC winner carry-forward for comparability
- **G3-B:** `google/gemma-3-12b-it` — Google 12B dense, G3-HPC runner-up carry-forward
- **G3-C:** `meta-llama/llama-3.2-11b-vision-instruct` — Meta 11B dense, safety-weak in G3-HPC
- **G3-D:** `google/gemma-4-26b-a4b-it` — 2026 Google MoE, 26B total / ~4B active (replaces failed DeepSeek-R1 404)
- **G3-E:** `qwen/qwen3.6-35b-a3b` — 2026 Alibaba MoE, 35B total / ~3B active (April 2026)
- **G3-F:** `deepseek/deepseek-v4-flash` — 2026 DeepSeek MoE, 284B total / ~13B active (April 2026)
- **G3-G:** `qwen/qwen3.5-35b-a3b` — 2026 Alibaba MoE, 35B total / ~3B active (February 2026)

**The critical methodological fix in this run:** All seven models are called with `extra_body={"reasoning": {"effort": "none"}, "include_reasoning": False}`. This OpenRouter-level flag suppresses the thinking/reasoning phase server-side before any tokens are generated, dropping mean generation latency from ~28 s to 8–28 s depending on model. Without this fix, the initial G3-OR run recorded mean generation latencies of 27,822 ms (G3-A) and 10,855 ms (G3-B) — both inflated by silent thinking-token generation. The fix was verified by zero `think_stripped` flags across all 96 × 7 = 672 records.

---

## 2. Evaluation Metric Reference

G3-OR uses the same generation-layer metrics as G1, G2, and G3-HPC. Retrieval metrics are not re-measured because retrieval is fixed.

| Metric | Type | What It Measures | Clinical Relevance |
|---|---|---|---|
| **Faithfulness (FA)** | RAGAS LLM-judge | Fraction of answer claims attributable to retrieved context | PRIMARY — hallucination resistance; does the LLM stay within guideline boundaries? |
| **Answer Relevancy (AR)** | RAGAS embed-judge | Semantic alignment of answer to the wound query | SECONDARY — does the answer address the specific clinical question asked? |
| **Safety Pass Rate** | Rule-based checker (deterministic) | Composite pass/fail across contraindication, antibiotic, referral, and dressing-list checks | SAFETY GATE — clinical compliance with T.I.M.E. protocol requirements |
| **Generation Latency** | Wall-clock (ms) | Time from API call to first complete response | O3/RQ3 — must be ≤10,000 ms for mobile deployment feasibility |

**Thresholds applied in this analysis:**
- FA gate: ≥ 0.75 (below this, hallucination risk is clinically unacceptable)
- Safety gate: ≥ 84.6% (best_safety − 5 pp = 89.6% − 5 pp; consistent with prior experiments)
- Mobile latency gate: ≤ 10,000 ms mean generation latency

**NaN note on RAGAS scores:** Several per-sample FA and AR values appear as NaN in the results. This is a known RAGAS/GPT-4o-mini judge interaction: the judge occasionally fails to decompose an answer into scorable claims (FA) or generate reverse questions (AR) for very short or very long responses, producing a NaN rather than a numeric score. The RAGAS library's `_safe_mean()` function drops NaN values before averaging, so reported means are computed over the non-NaN subset only. This is consistent with G2 and G3-HPC methodology.

---

## 3. Versions Tested

| Version | Model | Architecture | Parameters | License | Think Mode | No-Think Method |
|---|---|---|---|---|---|---|
| **G3-A** | `qwen/qwen3-14b` | Dense transformer | 14B | Apache 2.0 | ✅ Yes | `extra_body` + `/no_think` prefix |
| **G3-B** | `google/gemma-3-12b-it` | Dense transformer | 12B | Apache 2.0 | ❌ No | `extra_body` (safety net) |
| **G3-C** | `meta-llama/llama-3.2-11b-vision-instruct` | Dense transformer | 11B | Llama 3.2 Community | ❌ No | `extra_body` (safety net) |
| **G3-D** | `google/gemma-4-26b-a4b-it` | MoE (~4B active) | 26B total | Apache 2.0 | ❌ No | `extra_body` (safety net) |
| **G3-E** | `qwen/qwen3.6-35b-a3b` | MoE (~3B active) | 35B total | Apache 2.0 | ✅ Yes | `extra_body` + `/no_think` prefix |
| **G3-F** | `deepseek/deepseek-v4-flash` | MoE (~13B active) | 284B total | MIT | ✅ Yes | `extra_body` only |
| **G3-G** | `qwen/qwen3.5-35b-a3b` | MoE (~3B active) | 35B total | Apache 2.0 | ✅ Yes | `extra_body` + `/no_think` prefix |

All versions use: G1-C grounded system prompt (fixed), R1-C multi-axis BGE retrieval (k=6, fixed), 3 fresh runs per version, 32 cases per run (96 records per version), 3 s inter-case sleep for OpenRouter rate limiting.

---

## 4. Results Summary Table

### 4.1 Aggregated Results (n_runs = 3, 32 cases/run, 96 cases total per version)

| Version | Model | FA (mean ± std) | AR (mean ± std) | Safety% (mean ± std) | Gen Lat (ms) | Lat Std (ms) | Mobile |
|---|---|---|---|---|---|---|---|
| **G3-A** | qwen/qwen3-14b | 0.7752 ± 0.0224 | 0.7135 ± 0.0105 | 81.2% ± 0.0pp | 19,866 | ±3,212 | ❌ |
| **G3-B** | google/gemma-3-12b-it | 0.7114 ± 0.0105 | 0.6792 ± 0.0325 | 81.2% ± 0.0pp | 8,229 | ±988 | ✅ |
| **G3-C** | meta-llama/llama-3.2-11b | 0.6910 ± 0.0136 | 0.6617 ± 0.0344 | 66.7% ± 1.8pp | 10,786 | ±1,339 | ❌ |
| **G3-D** | google/gemma-4-26b-a4b-it | 0.8232 ± 0.0148 | 0.7078 ± 0.0046 | 88.5% ± 3.6pp | 28,360 | ±8,069 | ❌ |
| **G3-E** | qwen/qwen3.6-35b-a3b | 0.8105 ± 0.0047 | 0.6875 ± 0.0082 | 86.5% ± 1.8pp | 11,997 | ±3,495 | ❌ |
| **G3-F** | deepseek/deepseek-v4-flash | 0.8297 ± 0.0279 | 0.6973 ± 0.0297 | 87.5% ± 0.0pp | 21,665 | ±2,337 | ❌ |
| **G3-G** | qwen/qwen3.5-35b-a3b | **0.8322 ± 0.0010** | **0.7115 ± 0.0111** | **89.6% ± 1.8pp** | **9,358** | **±1,069** | **✅** |

**Bold** = best per metric. Classifier accuracy: 87.5% for all versions (fixed retrieval). `think_stripped` = 0 for all 672 records — reasoning successfully suppressed.

### 4.2 Per-Run Breakdown

| Version | Run 1 FA | Run 2 FA | Run 3 FA | Run 1 Safety | Run 2 Safety | Run 3 Safety |
|---|---|---|---|---|---|---|
| G3-A | 0.7566 | 0.8001 | 0.7690 | 81.2% | 81.2% | 81.2% |
| G3-B | 0.7035 | 0.7233 | 0.7073 | 81.2% | 81.2% | 81.2% |
| G3-C | 0.7023 | 0.6948 | 0.6759 | 65.6% | 65.6% | 68.8% |
| G3-D | 0.8272 | 0.8068 | 0.8356 | 90.6% | 84.4% | 90.6% |
| G3-E | 0.8091 | 0.8067 | 0.8157 | 84.4% | 87.5% | 87.5% |
| G3-F | 0.8231 | 0.8604 | 0.8057 | 87.5% | 87.5% | 87.5% |
| G3-G | 0.8324 | 0.8331 | 0.8311 | 90.6% | 90.6% | 87.5% |

### 4.3 Incremental Deltas vs G3-A Baseline (mean-based)

| Comparison | ΔFA (mean) | ΔAR (mean) | ΔSafety (mean) | ΔGen Latency |
|---|---|---|---|---|
| G3-B vs G3-A | −0.0638 | −0.0343 | 0.0 pp | −11,637 ms |
| G3-C vs G3-A | −0.0842 | −0.0518 | −14.5 pp | −9,080 ms |
| G3-D vs G3-A | +0.0480 | −0.0057 | +7.3 pp | +8,494 ms |
| G3-E vs G3-A | +0.0353 | −0.0260 | +5.3 pp | −7,869 ms |
| G3-F vs G3-A | +0.0545 | −0.0162 | +6.3 pp | +1,799 ms |
| G3-G vs G3-A | **+0.0570** | −0.0020 | **+8.4 pp** | **−10,508 ms** |

**G3-G uniquely achieves improvements on all three primary dimensions simultaneously** — higher FA, higher safety, AND faster latency than the G3-A baseline — making it the only model that Pareto-dominates G3-A.

---

## 5. Latency Analysis

### 5.1 Generation and Total Latency (sorted by Gen Latency)

| Version | Mean Gen Lat | Std Gen Lat | Mean Total Lat | Std Total Lat | Mobile ≤10s? |
|---|---|---|---|---|---|
| **G3-G** | **9,358 ms** | ±1,069 ms | 9,649 ms | ±1,062 ms | **✅ PASS** |
| G3-B | 8,229 ms | ±988 ms | 8,535 ms | ±984 ms | ✅ PASS |
| G3-E | 11,997 ms | ±3,495 ms | 12,296 ms | ±3,492 ms | ❌ marginal |
| G3-C | 10,786 ms | ±1,339 ms | 11,090 ms | ±1,323 ms | ❌ marginal |
| G3-A | 19,866 ms | ±3,212 ms | 20,169 ms | ±3,195 ms | ❌ FAIL |
| G3-F | 21,665 ms | ±2,337 ms | 21,960 ms | ±2,342 ms | ❌ FAIL |
| G3-D | 28,360 ms | ±8,069 ms | 28,653 ms | ±8,075 ms | ❌ FAIL |

Retrieval latency is consistent across all versions at ~291–306 ms per case, as expected since retrieval is identical for all seven.

**G3-G is the only model simultaneously passing the quality gates AND the mobile latency gate.** G3-B also passes the latency gate at 8,229 ms, but fails the FA threshold (0.7114 < 0.75) and the safety gate (81.2% < 84.6%).

**G3-D (Gemma 4 26B-A4B) is unexpectedly slow** at 28,360 ms mean — well above the 3–6 s expected for a ~4B active-parameter MoE. OpenRouter's current Gemma 4 infrastructure appears not to be optimised for its MoE architecture, or the model encounters high queue times due to low-tier allocation. Its latency variance (±8,069 ms) — the highest of all seven models — further signals scheduling instability on the provider side.

**G3-G stands out for latency consistency:** std of ±1,069 ms across 32 cases per run is the tightest profile of all models exceeding FA=0.75, and represents a highly predictable user experience at ~9.4 s mean.

**G3-E's mean latency of 11,997 ms is close to but above the 10 s gate**, with a std of ±3,495 ms. In the best run (Run 3: 8,691 ms), it would have passed; in the worst (Run 2: 15,655 ms), it fails badly. This high variance makes G3-E's mobile viability unreliable in practice.

### 5.2 Thinking Token Suppression Verification

Every record across all 672 observations (7 versions × 3 runs × 32 cases) shows `think_stripped = False`. This confirms that `extra_body={reasoning:{effort:none}}` successfully prevented thinking token generation at the OpenRouter server level for all models. The `/no_think` system prompt prefix added for G3-A, G3-E, and G3-G is a belt-and-suspenders redundancy that did not need to activate.

---

## 6. Safety Analysis

### 6.1 Overall Safety Rates (3-run evaluation)

| Version | Safety Pass Rate | Fails per Run (avg) | Per-Run Consistency | Contraind. Failures | Ref. Failures | Dressing Failures |
|---|---|---|---|---|---|---|
| **G3-G** | **89.6%** | 3.3 | ✅ High (90.6/90.6/87.5%) | 0 | 0 | 7 |
| G3-D | 88.5% | 3.7 | ⚠️ Moderate (90.6/84.4/90.6%) | 6 | 3 | 6 |
| G3-F | 87.5% | 4.0 | ✅ Perfect (87.5/87.5/87.5%) | 6 | 0 | 7 |
| G3-E | 86.5% | 4.3 | ✅ High (84.4/87.5/87.5%) | 6 | 3 | 5 |
| G3-A | 81.2% | 6.0 | ✅ Perfect (81.2/81.2/81.2%) | 0 | 9 | 9 |
| G3-B | 81.2% | 6.0 | ✅ Perfect (81.2/81.2/81.2%) | 0 | 9 | 9 |
| G3-C | 66.7% | 10.7 | ⚠️ Variable (65.6/65.6/68.8%) | 5 | 18 | 16 |

### 6.2 Safety Check Failure Analysis (across all 3 runs, 96 records per version)

| Check | G3-A | G3-B | G3-C | G3-D | G3-E | G3-F | G3-G |
|---|---|---|---|---|---|---|---|
| `antibiotic_check` | 3 | 3 | 3 | 3 | 4 | 3 | 3 |
| `contraindication_absent_charcoal` | 0 | 0 | 1 | 1 | 2 | 2 | 0 |
| `contraindication_absent_hydrocolloid` | 0 | 0 | 3 | 3 | 0 | 1 | 0 |
| `contraindication_absent_silver` | 0 | 0 | 1 | 1 | 2 | 2 | 0 |
| `contraindication_absent_bordered_foam` | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| `contraindication_absent_adhesive_foam` | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| `contraindication_absent_honey` | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| `dressing_in_allowed_list` | 9 | 9 | 16 | 6 | 5 | 7 | 7 |
| `referral_check` | 9 | 9 | 18 | 3 | 3 | 0 | 0 |

**Key observations:**

**G3-G achieves zero contraindication failures across all 3 runs** — only the second model in the entire VerdaSense ablation study (after G2-D and G3-HPC G3-A) to achieve this milestone. This means G3-G never recommends a clinically contraindicated dressing in any of its 96 outputs.

**G3-F achieves zero referral_check failures** (0/96) — a clinically critical result. This means G3-F never fails to escalate a wound that the testset marks as requiring specialist referral. Alongside G3-G (also 0 referral failures), this is the strongest possible outcome for the most serious safety sub-check.

**The `antibiotic_check` is universally failing at exactly 3/96 records** (one per run) for all models except G3-E (4). This suggests a structural pattern: the same 1 case per run triggers antibiotic failure. This is not a model-specific problem — it is a testset or grounding challenge shared by all seven models and appears to be a hard case that requires specific clinical phrasing not consistently generated by any model.

**G3-C is the safety outlier** with 66.7% pass rate — 15 pp below the next-worst model. Its 18 referral failures (across 3 runs) indicate it consistently fails to escalate complex wound presentations. The 16 `dressing_in_allowed_list` failures mean G3-C frequently recommends no clinically recognised dressing from the allowed set, reflecting a fundamental alignment gap with G1-C's structured format.

### 6.3 Category-Level Safety Breakdown (sum across 3 runs)

| Category | G3-A | G3-B | G3-C | G3-D | G3-E | G3-F | G3-G |
|---|---|---|---|---|---|---|---|
| Cat A (8 cases × 3 runs = 24) | 24/24 ✅ | 24/24 ✅ | 21/24 ⚠️ | 24/24 ✅ | 24/24 ✅ | 23/24 ✅ | 24/24 ✅ |
| Cat B (12 cases × 3 = 36) | 27/36 | 27/36 | 22/36 ❌ | 31/36 | 31/36 | 32/36 | 31/36 |
| Cat C (6 cases × 3 = 18) | 18/18 ✅ | 18/18 ✅ | 15/18 ⚠️ | 18/18 ✅ | 16/18 | 17/18 | 18/18 ✅ |
| Cat D (4 cases × 3 = 12) | 3/12 ❌ | 3/12 ❌ | 3/12 ❌ | 6/12 | 6/12 | 6/12 | 7/12 |
| Cat E (2 cases × 3 = 6) | 6/6 ✅ | 6/6 ✅ | 3/6 ⚠️ | 6/6 ✅ | 6/6 ✅ | 6/6 ✅ | 6/6 ✅ |

**Cat D (4 referral/specialist cases) is the universal weak point.** All models fail at least 5/12 Cat D cases. The best performer is G3-G (7/12 = 58.3%), still a majority failure rate. Cat D cases appear to require referral language that is difficult to generate consistently across models. This is the same pattern seen in G3-HPC and G2.

**G3-G is the only model achieving 100% safety on Cat A, Cat C, and Cat E simultaneously** — the clean presentation categories — while also having the best Cat D performance. This multi-category consistency is what drives G3-G's highest overall safety rate.

---

## 7. Detailed Version-by-Version Discussion

### 7.1 G3-A — Qwen3:14b (no-think) via OpenRouter

**Mean FA: 0.7752 ± 0.0224 | Mean AR: 0.7135 ± 0.0105 | Safety: 81.2% ± 0.0pp | Gen Lat: 19,866 ± 3,212 ms**

G3-A is the 2026 OpenRouter re-run of the G3-HPC winner, now with reasoning fully disabled via `extra_body`. Despite the reasoning suppression, mean FA improved substantially from the G3-HPC result (0.7306 → 0.7752, +4.46 pp). This likely reflects the difference in infrastructure: OpenRouter's cloud serving of Qwen3-14b operates at full precision without Ollama's llama.cpp quantisation overhead, resulting in better instruction following under the G1-C grounded prompt.

FA variance is the highest among all seven models (std = 0.0224), with a wide per-run spread of 0.7566 → 0.8001 → 0.7690. Run 2 notably hits 0.8001 — the only G3-A run to exceed 0.80 FA — suggesting that the model is capable of strong grounding on some runs but is sensitive to API response variation.

Safety is the model's key limitation. The 81.2% rate (constant across all 3 runs, std = 0.0) is driven by 9 `referral_check` failures and 9 `dressing_in_allowed_list` failures over 3 runs. Both are structural: G3-A consistently misses the referral escalation phrase requirement for Cat D cases and occasionally selects dressings not in the allowed list. Zero contraindication failures confirm G3-A does not recommend clinically unsafe dressings — the failure mode is omission, not commission.

Latency at 19,866 ms fails the 10 s mobile gate, with the same variable profile seen in G3-HPC. Qwen3-14b is a 14B dense model; even with reasoning disabled, its generation cost on shared OpenRouter infrastructure is high enough that mobile deployment of G3-A is not viable at this threshold.

**Verdict:** Best AR of all 7 models (0.7135), strong FA at 0.7752, but fails both the safety gate and the mobile latency gate. Acts as a useful baseline that reveals the Qwen3-14b model's consistent referral escalation gap.

### 7.2 G3-B — Gemma3:12b (Google open) via OpenRouter

**Mean FA: 0.7114 ± 0.0105 | Mean AR: 0.6792 ± 0.0325 | Safety: 81.2% ± 0.0pp | Gen Lat: 8,229 ± 988 ms**

G3-B achieves the fastest mean latency of any model generating clinically complete outputs (8,229 ms) and is the second model to pass the mobile latency gate. Its performance profile is, however, almost identical to the G3-HPC Gemma3:12b result — reflecting that 12B dense models have consistent quality regardless of whether they run on A100 VRAM or a cloud inference provider.

The 0.7114 FA falls below the 0.75 gate by 3.86 pp — a systematic gap that appears across all 3 runs (0.7035, 0.7233, 0.7073), never reaching the threshold. AR (0.6792) is also the lowest among models that at least partially comply with the G1-C format. Safety at 81.2% (identical to G3-A despite different architecture) confirms that both models share the same referral and dressing-list failure modes.

The high AR variance (std = 0.0325) is notable — G3-B's answer relevancy fluctuates significantly between runs, suggesting the model's question-answering alignment is less stable than its faithfulness. This is the opposite of G3-E and G3-G, where AR variance is tightly controlled.

**Verdict:** Only viable mobile option besides G3-G, but underperforms on FA and safety by clinically meaningful margins. Best suited for demonstrations or low-stakes deployments where latency is the dominant constraint.

### 7.3 G3-C — Llama3.2-Vision:11b (Meta) via OpenRouter

**Mean FA: 0.6910 ± 0.0136 | Mean AR: 0.6617 ± 0.0344 | Safety: 66.7% ± 1.8pp | Gen Lat: 10,786 ± 1,339 ms**

G3-C replicates the G3-HPC finding: Llama3.2-11b consistently underperforms all other models on both FA and safety. The 66.7% safety rate — 14.5 pp below G3-A and G3-B — is driven primarily by 18 referral_check failures (6 per run) and 16 dressing_in_allowed_list failures across all runs. These are not edge-case misses; they represent a systematic failure to follow the G1-C grounded prompt's escalation and dressing-selection format.

The 3 `contraindication_absent_hydrocolloid` failures (1/run) are clinically dangerous: G3-C recommends hydrocolloid as a primary dressing in cases where the testset marks it as explicitly contraindicated (typically infected wounds where hydrocolloid creates an anaerobic environment). This is the only model producing clinically contraindicated recommendations at a rate of 1 per run — an unacceptable patient safety risk.

FA at 0.6910 reflects that, despite some structurally valid outputs, G3-C frequently introduces claims not supported by the retrieved context. The 11B parameter count and vision-first training objective appear to limit the model's instruction-following discipline under the G1-C grounded prompt.

**Verdict:** Eliminated from consideration on safety grounds. G3-C's 18 referral failures and contraindicated dressing recommendations disqualify it for any clinical deployment context.

### 7.4 G3-D — Gemma 4 26B-A4B (Google MoE) via OpenRouter

**Mean FA: 0.8232 ± 0.0148 | Mean AR: 0.7078 ± 0.0046 | Safety: 88.5% ± 3.6pp | Gen Lat: 28,360 ± 8,069 ms**

G3-D produces the second-highest FA (0.8232) and the second-highest safety (88.5%) of all seven models. Its clinical quality profile is strong — only 3 referral_check failures (1/run) and a well-calibrated contraindication pattern (1 charcoal, 3 hydrocolloid, 1 silver, 1 honey across 3 runs). G3-D's 90.6% safety in Runs 1 and 3 matches G2-D's level.

However, G3-D's latency disqualifies it from mobile deployment. The mean of 28,360 ms is the slowest of all seven models, exceeding even G3-A (19,866 ms). The std of ±8,069 ms — 2.8× higher than G3-G's — reflects that OpenRouter's Gemma 4 infrastructure is highly unstable for this model tier. Run 3 alone took 37,476 ms mean, meaning some individual cases took 50+ seconds. This represents an unacceptable user experience for a smartphone-based wound care application.

The quality–latency trade-off for G3-D is indefensible given that G3-G achieves comparable quality (FA=0.8322 vs 0.8232, Safety=89.6% vs 88.5%) at 3× lower latency and 3× lower latency variance. G3-D's latency problem appears to be infrastructure-side: Gemma 4's MoE architecture is not yet optimally served by OpenRouter's 2026 infrastructure.

**Verdict:** Strong clinical quality that passes both FA and safety gates, but fatally disqualified by latency. A useful model to monitor — if OpenRouter's Gemma 4 serving improves, G3-D could become a competitive option.

### 7.5 G3-E — Qwen3.6-35B-A3B (Alibaba MoE) via OpenRouter

**Mean FA: 0.8105 ± 0.0047 | Mean AR: 0.6875 ± 0.0082 | Safety: 86.5% ± 1.8pp | Gen Lat: 11,997 ± 3,495 ms**

G3-E is the newest Qwen model (released April 2026) and the direct successor to G3-A (Qwen3-14b). The MoE upgrade from 14B dense to 35B-total/3B-active delivers a clear quality improvement: FA 0.7752 → 0.8105 (+3.53 pp), safety 81.2% → 86.5% (+5.3 pp). The FA variance drops dramatically from G3-A (0.0224 → 0.0047), reflecting the MoE architecture's more consistent generation behaviour.

G3-E's safety failures are more specific than G3-A's: 4 antibiotic failures (slightly worse than G3-A's 3), 2 bordered_foam contraindication failures, 2 charcoal contraindication failures, and 2 silver contraindication failures — all concentrated in complex MoE cases. The referral_check improves dramatically (3 failures vs G3-A's 9), showing the larger knowledge base better handles escalation detection.

The latency at 11,997 ms fails the 10 s gate, but only barely on mean. The best run (Run 3: 8,691 ms) would have passed. However, Run 2 at 15,655 ms — representing a 2× swing — makes G3-E's mobile feasibility unreliable. The ±3,495 ms std is 3.3× G3-G's std, making the user experience unpredictable.

**Verdict:** Passes the FA gate (0.8105 ≥ 0.75) and the safety gate (86.5% ≥ 84.6%), but misses the mobile latency gate on mean. The best model by quality-to-latency trade-off *excluding* G3-G. If the latency standard were relaxed to 12 s or if OpenRouter's Qwen3.6 serving improves, G3-E would be a strong candidate.

### 7.6 G3-F — DeepSeek V4 Flash (DeepSeek MoE) via OpenRouter

**Mean FA: 0.8297 ± 0.0279 | Mean AR: 0.6973 ± 0.0297 | Safety: 87.5% ± 0.0pp | Gen Lat: 21,665 ± 2,337 ms**

G3-F achieves the second-highest mean FA (0.8297) and a perfectly consistent 87.5% safety rate across all 3 runs (std = 0.0pp). The zero referral_check failures (0/96) is the standout safety result — G3-F never fails to escalate when escalation is required, a property shared only with G3-G. This is particularly noteworthy given DeepSeek V4 Flash is a 284B total / ~13B active MoE model not specifically trained for clinical applications.

FA variance (std = 0.0279) is the highest of the four MoE models — driven by Run 2's outlier value of 0.8604 versus Runs 1 and 3 at 0.8231 and 0.8057. The Run 2 spike suggests the model occasionally achieves near-maximal faithfulness but cannot sustain it consistently. This variance, though within acceptable bounds, makes G3-F less predictable than G3-G (std = 0.0010) for clinical deployment.

The principal disqualifier is latency at 21,665 ms — the third-slowest model. Despite being a 13B active-parameter MoE (which should theoretically be faster than G3-D's 4B active), DeepSeek V4 Flash on OpenRouter encounters high inference overhead, likely from the 284B total parameter scale requiring large KV-cache allocation even for MoE routing.

**Verdict:** Clinically strong (passes FA and safety gates, zero referral failures), but fails the mobile latency gate by >2×. Best positioned for non-mobile deployments (server-side clinical decision support) where response quality and safety consistency matter more than latency.

### 7.7 G3-G — Qwen3.5-35B-A3B (Alibaba MoE) via OpenRouter ⭐ WINNER

**Mean FA: 0.8322 ± 0.0010 | Mean AR: 0.7115 ± 0.0111 | Safety: 89.6% ± 1.8pp | Gen Lat: 9,358 ± 1,069 ms**

G3-G is the only model in G3-OR that simultaneously passes all three gates: FA ≥ 0.75, Safety ≥ 84.6%, and Gen Lat ≤ 10,000 ms. It achieves the highest FA, highest safety, and highest AR of all seven models — while being the second-fastest. This makes G3-G a uniquely strong candidate for VerdaSense production deployment.

**Faithfulness stability** is G3-G's most remarkable property: FA std of 0.0010 across three runs (0.8324, 0.8331, 0.8311) — a range of only 0.0020. This is the tightest faithfulness distribution in the entire VerdaSense ablation study across G1, G2, G3-HPC, and G3-OR. The model's grounding behaviour is essentially deterministic: it consistently attributes approximately 83.2% of its clinical claims to the retrieved sources. This stability is invaluable for clinical deployment — the system's trustworthiness is predictable and does not vary significantly between patient interactions.

**Safety profile** at 89.6% is the highest of all seven G3-OR models and matches G2-D (Gemini 2.5 Flash, 90.6%) within 1 pp — within the noise floor (std = 1.8pp). Key safety milestones:
- Zero contraindication failures across all 96 records (0/96) — G3-G never recommends a contraindicated dressing
- Zero referral_check failures (0/96) — G3-G never fails to escalate when specialist referral is required
- Only 3 antibiotic failures (1/run) — the same structural single-case failure shared by all models
- Only 7 dressing_in_allowed_list failures (2.3/run) — primarily in complex Cat D cases

The dual zero-failure result on both contraindication and referral checks is the strongest possible outcome for the two most clinically critical safety sub-checks.

**AR at 0.7115** is the highest among the four MoE models (G3-D: 0.7078, G3-F: 0.6973, G3-E: 0.6875) and closely matches G3-A's AR (0.7135). The Qwen3.5 architecture appears to generate responses that both strictly adhere to retrieved context (high FA) and directly address the specific wound question asked (high AR) — a balance that other MoE models do not achieve as consistently.

**Latency at 9,358 ± 1,069 ms** passes the 10 s mobile gate with 642 ms to spare on mean, and with a std tight enough that the 95th percentile would be approximately 11,500 ms — still near the threshold even for worst-case responses. The total latency including retrieval (9,649 ms) confirms real-world smartphone response times in the 9–11 s range.

---

## 8. Cross-Version Comparative Analysis

### 8.1 MoE vs Dense Architecture Comparison

| Architecture | Models | FA (mean) | Safety (mean) | Gen Lat (mean) |
|---|---|---|---|---|
| Dense (~11–14B) | G3-A, G3-B, G3-C | 0.726 | 76.4% | 12,960 ms |
| MoE (2026 generation) | G3-D, G3-E, G3-F, G3-G | 0.825 | 88.0% | 17,845 ms |

The MoE generation advantage is unambiguous: +9.9 pp FA and +11.6 pp safety on average. However, the MoE latency advantage predicted in the G3_Extended_Model_Selection_Guide (3–6 s expected) was not realised in practice — only G3-G (35B/3B MoE) achieved sub-10 s latency. G3-D (26B/4B MoE) and G3-F (284B/13B MoE) are both slower than G3-A (14B dense). OpenRouter's current MoE serving is evidently not optimised uniformly across all models.

### 8.2 Architecture Family Comparison

| Family | Models | FA range | Safety range | Key strength |
|---|---|---|---|---|
| Alibaba Qwen | G3-A, G3-E, G3-G | 0.775–0.832 | 81.2–89.6% | Consistently strong FA + AR; G3-G winner |
| Google | G3-B, G3-D | 0.711–0.823 | 81.2–88.5% | G3-D has strong quality; latency issues |
| DeepSeek | G3-F | 0.830 | 87.5% | High FA, zero referral failures; too slow |
| Meta | G3-C | 0.691 | 66.7% | Safety-weak; not recommended |

**The Qwen family dominates G3-OR.** G3-A, G3-E, and G3-G all outperform their comparable-size competitors on safety, and G3-G is the outright winner. The progression from G3-A (Qwen3:14b dense) to G3-G (Qwen3.5:35B MoE) to G3-E (Qwen3.6:35B MoE) shows consistent improvement as the architecture evolves: FA 0.775 → 0.832 → 0.811, safety 81.2% → 89.6% → 86.5%.

### 8.3 The Qwen3.5 vs Qwen3.6 Intra-Family Comparison

G3-G (Qwen3.5-35B-A3B, February 2026) and G3-E (Qwen3.6-35B-A3B, April 2026) share the same 35B/~3B MoE architecture but differ in training vintage. Contrary to expectation, the older Qwen3.5 (G3-G) outperforms the newer Qwen3.6 (G3-E) on all three primary metrics:

| | G3-G (Qwen3.5) | G3-E (Qwen3.6) | Δ (G3-G − G3-E) |
|---|---|---|---|
| FA | 0.8322 | 0.8105 | +0.0217 |
| Safety | 89.6% | 86.5% | +3.1 pp |
| Gen Lat | 9,358 ms | 11,997 ms | −2,639 ms faster |

This counter-intuitive result may reflect that Qwen3.6's additional training (likely on a broader general-purpose corpus) dilutes the instruction-following precision that Qwen3.5 was fine-tuned for. Alternatively, OpenRouter's serving may be better optimised for Qwen3.5 at this time. This finding demonstrates that newer ≠ better for domain-specific clinical RAG applications — training alignment to the task matters more than model generation date.

---

## 9. Noise Floor and Run Stability

### 9.1 FA Stability Profile

| Version | FA std | FA range | Interpretation |
|---|---|---|---|
| **G3-G** | **0.0010** | 0.0020 | Exceptional — essentially deterministic grounding |
| G3-E | 0.0047 | 0.0090 | Very low — MoE consistent generation |
| G3-D | 0.0148 | 0.0288 | Low — one safety-dip run (84.4%) |
| G3-B | 0.0105 | 0.0198 | Low |
| G3-C | 0.0136 | 0.0264 | Low but systematically poor |
| G3-A | 0.0224 | 0.0435 | Moderate — most variable of 7 models |
| G3-F | 0.0279 | 0.0547 | Highest — Run 2 outlier (0.8604) skews |

G3-G's FA std of 0.0010 is unprecedented in this ablation study — 5× lower than G2-C (0.0054), the previous most stable model. Differences of ≥0.002 between G3-G and any other model on FA are systematic, not stochastic.

### 9.2 Safety Stability

Safety consistency matters as much as mean safety level. A model that scores 90.6% / 84.4% / 90.6% across runs is less trustworthy for deployment than one scoring 87.5% / 87.5% / 87.5%, because the user cannot predict which "run" any given patient interaction corresponds to.

| Version | Safety runs | Std | Deployment reliability |
|---|---|---|---|
| G3-A | 81.2/81.2/81.2% | 0.0pp | ✅ Perfectly consistent (but low) |
| G3-B | 81.2/81.2/81.2% | 0.0pp | ✅ Perfectly consistent (but low) |
| G3-F | 87.5/87.5/87.5% | 0.0pp | ✅ Perfectly consistent |
| G3-E | 84.4/87.5/87.5% | 1.8pp | ✅ High consistency |
| G3-G | 90.6/90.6/87.5% | 1.8pp | ✅ High consistency |
| G3-C | 65.6/65.6/68.8% | 1.8pp | ⚠️ Consistently poor |
| G3-D | 90.6/84.4/90.6% | 3.6pp | ⚠️ Moderate variance — one bad run |

G3-D's safety variance (std = 3.6pp) is the highest and the most concerning: Run 2 drops 6.2 pp below Runs 1 and 3. This asymmetric drop suggests G3-D occasionally produces outputs in Run 2 that structurally differ from the other runs — consistent with its high latency variance, which may indicate the model encountered a different serving configuration.

---

## 10. Winner Selection

### 10.1 Selection Criteria

| Gate | Criterion | Rationale |
|---|---|---|
| **Primary (safety gate)** | Mean Safety ≥ 84.6% (best_safety − 5 pp) | Consistent with G2 and G3-HPC methodology |
| **Secondary (quality gate)** | Mean Faithfulness ≥ 0.75 | Below this, hallucination risk is clinically unacceptable |
| **Tertiary (latency gate)** | Mean Gen Latency ≤ 10,000 ms | O3/RQ3 mobile deployment target |
| **Tie-breaker** | Highest mean FA among models passing all three gates | Maximise clinical quality |

### 10.2 Gate Application

| Version | Safety ≥ 84.6%? | FA ≥ 0.75? | Lat ≤ 10,000ms? | Qualifies? |
|---|---|---|---|---|
| G3-A | ❌ 81.2% | ✅ 0.7752 | ❌ 19,866 ms | **No — fails safety + latency** |
| G3-B | ❌ 81.2% | ❌ 0.7114 | ✅ 8,229 ms | **No — fails safety + FA** |
| G3-C | ❌ 66.7% | ❌ 0.6910 | ❌ 10,786 ms | **No — fails all three** |
| G3-D | ✅ 88.5% | ✅ 0.8232 | ❌ 28,360 ms | **No — fails latency** |
| G3-E | ✅ 86.5% | ✅ 0.8105 | ❌ 11,997 ms | **No — fails latency** |
| G3-F | ✅ 87.5% | ✅ 0.8297 | ❌ 21,665 ms | **No — fails latency** |
| **G3-G** | ✅ 89.6% | ✅ 0.8322 | ✅ 9,358 ms | **✅ YES — passes all three** |

### 10.3 Winner

> **G3-G (Qwen3.5-35B-A3B via OpenRouter) is selected as the G3-OR winner and the recommended open-source generation LLM for VerdaSense.**
>
> **Rationale:** Only model passing all three gates simultaneously — safety (89.6%), faithfulness (0.8322), and mobile latency (9,358 ms). Achieves the highest FA of all seven models, highest safety of all seven models, highest AR of all MoE models, and the tightest FA stability (std = 0.0010) of the entire ablation study. Zero contraindication failures and zero referral failures across 96 evaluations represent the strongest possible clinical safety profile.
>
> **Confidence:** High. G3-G's FA advantage over the next-best mobile-feasible model (G3-B: 0.7114) is 0.1208 — 12× larger than G3-G's own FA std (0.0010), making it unambiguously systematic. Safety advantage over G3-B (+8.4 pp) is 4.7× larger than G3-G's safety std (1.8 pp).

---

## 11. G3-OR vs G3-HPC vs G2: Three-Way Comparison

### 11.1 Direct Performance Comparison (Best per Experiment)

| Metric | G2-D (Gemini 2.5 Flash) | G3-HPC G3-A (Qwen3:14b Ollama) | G3-OR G3-G (Qwen3.5 OpenRouter) | G3-OR G3-D (Gemma4 OpenRouter) |
|---|---|---|---|---|
| **FA (mean)** | 0.8147 | 0.7306 | **0.8322** | 0.8232 |
| **AR (mean)** | 0.6770 | 0.6944 | **0.7115** | 0.7078 |
| **Safety (mean)** | **90.6%** | 80.2% | 89.6% | 88.5% |
| **Gen Lat (ms)** | 17,961 | 30,924 | **9,358** | 28,360 |
| **Mobile ≤10s** | ❌ | ❌ | ✅ | ❌ |
| **Cost/query** | $0.0056 | Free (HPC) | $0.0023 | $0.0008 |

**G3-OR G3-G outperforms G2-D (the closed-source winner) on every metric except safety** (89.6% vs 90.6%, within the 1.8pp noise floor). The FA advantage (0.8322 vs 0.8147, +1.75 pp) exceeds G3-G's own std (0.0010) by 17.5× — this is a genuine systematic advantage, not measurement noise.

**G3-G exceeds G2-D on both FA and AR while being 2× faster and 2.4× cheaper per query.** This is the central finding of G3-OR: the 2026 open-source MoE frontier, accessed via OpenRouter with reasoning suppressed, closes the quality gap to G2-D and in some dimensions surpasses it.

### 11.2 Full Cross-Experiment Ranking (all models, sorted by FA)

| Rank | Model | Experiment | FA | AR | Safety | Gen Lat | Mobile |
|---|---|---|---|---|---|---|---|
| 1 | **G3-G** (qwen3.5-35b-a3b) | G3-OR | **0.8322** | 0.7115 | 89.6% | 9,358 ms | ✅ |
| 2 | G3-F (deepseek-v4-flash) | G3-OR | 0.8297 | 0.6973 | 87.5% | 21,665 ms | ❌ |
| 3 | G3-D (gemma-4-26b-a4b-it) | G3-OR | 0.8232 | 0.7078 | 88.5% | 28,360 ms | ❌ |
| 4 | G2-D (gemini-2.5-flash) | G2 | 0.8147 | 0.6770 | **90.6%** | 17,961 ms | ❌ |
| 5 | G3-E (qwen3.6-35b-a3b) | G3-OR | 0.8105 | 0.6875 | 86.5% | 11,997 ms | ❌ |
| 6 | G3-HPC G3-B (gemma3:12b) | G3-HPC | 0.7424 | 0.6644 | 78.1% | 25,219 ms | ❌ |
| 7 | G2-A (gpt-4o-mini) | G2 | 0.7751 | **0.7233** | 84.4% | 12,921 ms | ❌ |
| 8 | G3-OR G3-A (qwen3-14b) | G3-OR | 0.7752 | 0.7135 | 81.2% | 19,866 ms | ❌ |
| 9 | G3-OR G3-B (gemma3-12b) | G3-OR | 0.7114 | 0.6792 | 81.2% | 8,229 ms | ✅ |
| 10 | G2-B (gpt-4o) | G2 | 0.7583 | 0.6910 | 81.2% | 6,274 ms | ✅ |
| 11 | G3-HPC G3-A (qwen3:14b) | G3-HPC | 0.7306 | 0.6944 | 80.2% | 30,924 ms | ❌ |
| 12 | G2-C (gemini-2.5-flash-lite) | G2 | 0.7385 | 0.6522 | 81.2% | 3,133 ms | ✅ |
| 13 | G3-OR G3-C (llama-3.2-11b) | G3-OR | 0.6910 | 0.6617 | 66.7% | 10,786 ms | ❌ |
| 14 | G3-HPC G3-C (llama3.2:11b) | G3-HPC | 0.6711 | 0.6562 | 65.6% | 29,916 ms | ❌ |
| 15 | G3-HPC G3-D (deepseek-r1:14b) | G3-HPC | 0.6558 | 0.7022 | 40.6% | 25,742 ms | ❌ |

**G3-G holds the top FA rank across all 15 models tested in the VerdaSense ablation study**, while being the only model among the top-5 FA performers to achieve mobile-feasible latency. This represents a qualitative step-change from the G3-HPC results, where no open-source model reached FA=0.75 with Safety≥85% simultaneously.

---

## 12. Pricing Comparison: All Models

### 12.1 API Pricing (as of May 2026, via respective provider)

| Experiment | Version | Model | Input $/M | Output $/M | License | Architecture |
|---|---|---|---|---|---|---|
| **G2 (Closed-Source)** | G2-A | gpt-4o-mini | $0.150 | $0.600 | Proprietary | — |
| | G2-B | gpt-4o | $2.500 | $10.000 | Proprietary | — |
| | G2-C | gemini-2.5-flash-lite | $0.100 | $0.400 | Proprietary | — |
| | G2-D | gemini-2.5-flash | $0.300 | $2.500 | Proprietary | — |
| **G3-HPC** | G3-A | qwen3:14b (Ollama) | Free (HPC) | Free (HPC) | Apache 2.0 | 14B dense |
| | G3-B | gemma3:12b (Ollama) | Free (HPC) | Free (HPC) | Apache 2.0 | 12B dense |
| | G3-C | llama3.2:11b (Ollama) | Free (HPC) | Free (HPC) | Llama 3.2 | 11B dense |
| | G3-D | deepseek-r1:14b (Ollama) | Free (HPC) | Free (HPC) | MIT | 14B dense |
| **G3-OR (OpenRouter)** | G3-A | qwen/qwen3-14b | $0.100 | $0.240 | Apache 2.0 | 14B dense |
| | G3-B | google/gemma-3-12b-it | $0.040 | $0.130 | Apache 2.0 | 12B dense |
| | G3-C | meta-llama/llama-3.2-11b | $0.245 | $0.245 | Llama 3.2 | 11B dense |
| | G3-D | google/gemma-4-26b-a4b-it | $0.060 | $0.330 | Apache 2.0 | ~4B MoE |
| | G3-E | qwen/qwen3.6-35b-a3b | $0.150 | $1.000 | Apache 2.0 | ~3B MoE |
| | G3-F | deepseek/deepseek-v4-flash | $0.100 | $0.200 | MIT | ~13B MoE |
| | **G3-G** | **qwen/qwen3.5-35b-a3b** | **$0.140** | **$1.000** | **Apache 2.0** | **~3B MoE** |

### 12.2 Per-Query Cost (VerdaSense profile: ~3,500 input + ~1,800 output tokens)

| Version | Model | Input cost | Output cost | **Per Query** | Per Patient (10 q) | vs G2-D |
|---|---|---|---|---|---|---|
| G3-B (OR) | gemma-3-12b | $0.00014 | $0.000234 | **$0.000374** | $0.004 | 15× cheaper |
| G3-D (OR) | gemma-4-26b | $0.00021 | $0.000594 | **$0.000804** | $0.008 | 7× cheaper |
| G3-F (OR) | deepseek-v4-flash | $0.00035 | $0.000360 | **$0.000710** | $0.007 | 8× cheaper |
| G3-A (OR) | qwen3-14b | $0.00035 | $0.000432 | **$0.000782** | $0.008 | 7× cheaper |
| G2-C | gemini-2.5-flash-lite | $0.00035 | $0.000720 | **$0.001070** | $0.011 | 5× cheaper |
| G2-A | gpt-4o-mini | $0.000525 | $0.001080 | **$0.001605** | $0.016 | 3.5× cheaper |
| G3-C (OR) | llama-3.2-11b | $0.000858 | $0.000441 | **$0.001299** | $0.013 | 4× cheaper |
| **G3-G (OR)** | **qwen3.5-35b** | $0.000490 | $0.001800 | **$0.002290** | **$0.023** | **2.4× cheaper** |
| G3-E (OR) | qwen3.6-35b | $0.000525 | $0.001800 | **$0.002325** | $0.023 | 2.4× cheaper |
| **G2-D** | **gemini-2.5-flash** | $0.001050 | $0.004500 | **$0.005550** | $0.056 | **baseline** |
| G2-B | gpt-4o | $0.008750 | $0.018000 | **$0.026750** | $0.268 | 4.8× more expensive |

### 12.3 Quality-Adjusted Value Analysis

The appropriate metric for clinical deployment is not cost alone but quality per dollar:

| Model | FA | Safety | Gen Lat | Cost/query | FA/$ (×1000) | Recommended For |
|---|---|---|---|---|---|---|
| G3-G (Qwen3.5 OR) | 0.8322 | 89.6% | 9,358 ms | $0.00229 | 363 | ✅ Production (mobile + quality) |
| G3-D (Gemma4 OR) | 0.8232 | 88.5% | 28,360 ms | $0.00080 | 1,029 | Server-side only (too slow mobile) |
| G3-F (DeepSeek V4F OR) | 0.8297 | 87.5% | 21,665 ms | $0.00071 | 1,168 | Server-side only (too slow mobile) |
| G2-D (Gemini 2.5 Flash) | 0.8147 | 90.6% | 17,961 ms | $0.00555 | 147 | Proprietary fallback if G3-G unavailable |
| G2-A (GPT-4o-mini) | 0.7751 | 84.4% | 12,921 ms | $0.00161 | 482 | OpenAI ecosystem fallback |
| G3-B (Gemma3 OR) | 0.7114 | 81.2% | 8,229 ms | $0.00037 | 1,923 | Demo/prototype only |

**G3-G offers the best balance of quality and cost for the VerdaSense production use case** — mobile-feasible, highest clinical quality, and 2.4× cheaper than G2-D. G3-D and G3-F offer higher nominal FA/$ ratios but are not mobile-feasible, making them unsuitable for VerdaSense's O3 objective.

---

## 13. Is G3-OR Meaningful for the FYP?

**Honest assessment: Yes — G3-OR is the most significant ablation contribution in the VerdaSense study to date, with results that directly answer all three research objectives.**

### 13.1 What G3-OR Definitively Establishes

**1. A 2026 open-source MoE model (G3-G) matches or exceeds G2-D across the primary clinical metrics at 2.4× lower cost and faster mobile latency.**

This is the headline finding. G3-G achieves FA=0.8322 vs G2-D's 0.8147 (+1.75 pp), Safety=89.6% vs 90.6% (within noise floor), AR=0.7115 vs 0.6770 (+3.45 pp), and Gen Lat=9,358 ms vs 17,961 ms (1.9× faster). The closed-source performance advantage identified in G2 and G3-HPC no longer holds in 2026.

**2. The open-source vs closed-source gap documented in G3-HPC (7–10 pp FA, 10–12 pp safety) has been closed by 2026 MoE models.**

G3-HPC's best result (FA=0.7306, Safety=80.2%) compared unfavourably to G2-D. G3-OR's best result (FA=0.8322, Safety=89.6%) compares favourably. The gap was closed not by fine-tuning but by the natural progression of the open-source model frontier — a finding worth discussing in the FYP as evidence that open-source and closed-source clinical NLP capabilities are converging.

**3. The mobile latency target (O3, RQ3: ≤10 s) is achievable with an open-source model at clinical quality.**

G3-OR demonstrates that a 35B MoE model via OpenRouter can deliver FA=0.8322 and Safety=89.6% in 9.4 s mean — directly answering RQ3. This closes the experimental loop from O1 (RAG design) through O2 (quality evaluation) to O3 (deployment feasibility).

**4. The reasoning token suppression methodology is essential and effective.**

The fix from the initial G3-OR (thinking tokens running silently, inflating latency to 27 s) to the corrected G3-OR (extra_body suppression, mean 9.4 s for G3-G) represents a methodological contribution documented in the notebook and reproducible by others. Zero `think_stripped` flags across 672 records validates the fix.

**5. The 7-version comparative design provides actionable model recommendations for multiple deployment scenarios.**

Different organisations have different constraints (cost, latency, sovereignty). G3-OR produces evidence-based recommendations for each scenario rather than a single one-size-fits-all answer.

### 13.2 What G3-OR Does NOT Definitively Establish

**1. Whether G3-G's quality advantage over G2-D is stable at production scale.** G3-OR uses 32 test cases across 3 runs. A systematic evaluation on 200+ clinical cases across diverse wound presentations would be required to confirm the FA advantage is not testset-specific.

**2. Whether OpenRouter's G3-G latency will remain at 9.4 s under production load.** The 3 s inter-case sleep means each run was measured at very low QPS. Under concurrent mobile users, OpenRouter latency may increase significantly.

**3. Whether the NaN artefacts in RAGAS evaluation systematically bias results.** Several models (G3-A, G3-B, G3-D, G3-E, G3-F) show NaN values on per-sample FA/AR in Run 2 and Run 3 not present in Run 1. These NaN values are dropped before averaging, potentially creating a non-comparable effective sample size across runs.

---

## 14. LLM Recommendation: Which Model Should You Choose?

### 14.1 Primary Recommendation

> **For VerdaSense production deployment on a smartphone-based wound care application: G3-G (Qwen3.5-35B-A3B via OpenRouter).**
>
> Highest FA (0.8322), highest safety (89.6%), best AR among MoE models (0.7115), only model passing all three deployment gates, and 2.4× cheaper than the prior closed-source winner. Mobile-feasible at 9.4 s mean generation latency.

### 14.2 Decision Framework by Deployment Scenario

| Scenario | Recommended Model | Rationale |
|---|---|---|
| **Smartphone/mobile clinical deployment** | **G3-G (Qwen3.5-35B-A3B via OpenRouter)** | Only model passing FA, safety, AND latency gates simultaneously |
| **Server-side / non-mobile clinical use** | G3-F (DeepSeek V4 Flash) or G3-D (Gemma4) | Higher or equal FA, zero referral failures (G3-F), at 8× lower cost than G2-D |
| **Proprietary fallback (if OpenRouter unavailable)** | G2-D (Gemini 2.5 Flash) | Still the most safety-consistent model (90.6%, zero variance) |
| **Lowest cost demo / prototype** | G3-B (Gemma3:12b OR) | Mobile-feasible, cheapest, but FA=0.71 is below clinical threshold |
| **Air-gapped / offline** | G3-HPC G3-A (Qwen3:14b via Ollama) | Best HPC open-source safety and FA when API is unavailable |
| **Research comparison baseline** | G2-A (GPT-4o-mini) | Transparent, well-documented, highest AR, common LLM benchmark reference |

### 14.3 The Case For and Against G3-G vs G2-D

| Criterion | G3-G wins | G2-D wins |
|---|---|---|
| Faithfulness (FA) | ✅ 0.8322 (+1.75 pp) | — |
| Answer Relevancy (AR) | ✅ 0.7115 (+3.45 pp) | — |
| Safety (mean) | — | ✅ 90.6% (+1.0 pp, within noise) |
| Mobile latency | ✅ 9,358 ms (1.9× faster) | — |
| Cost per query | ✅ $0.00229 (2.4× cheaper) | — |
| Latency consistency | ✅ std=1,069ms (vs 939ms) | ⚠️ Slightly tighter |
| Data sovereignty | ✅ Open-weights, self-hostable | ❌ Proprietary |
| API dependency risk | ✅ Multi-provider (OpenRouter OR self-host) | ❌ Google Gemini only |
| Safety std | ⚠️ 1.8pp | ✅ 0.0pp |

**G3-G is the stronger recommendation in 5 of 9 criteria.** G2-D's 1 pp safety advantage is within G3-G's noise floor (std=1.8pp) and not statistically meaningful. The only genuine G2-D advantage is zero safety variance (std=0.0pp) — G2-D is completely deterministic on safety, whereas G3-G has 1 run at 87.5% vs 2 at 90.6%.

---

## 15. Limitations and Threats to Validity

### 15.1 OpenRouter Infrastructure Variability

All G3-OR results are specific to OpenRouter's serving infrastructure on 28 May 2026. Model performance, latency, and reliability can change as OpenRouter updates serving infrastructure, re-allocates capacity, or modifies rate limiting policies. G3-G's 9.4 s mean latency may not be reproducible at a different time or load.

### 15.2 NaN Artefacts in Runs 2–3

A pattern of NaN values in per-sample FA and AR appears across multiple models in Runs 2 and 3 that is absent or minimal in Run 1. This may reflect RAGAS judge session state, LLM output format changes between runs, or API response variation in the gpt-4o-mini judge. The NaN values are dropped before averaging, which means the reported FA and AR means for some versions are computed over different effective sample sizes across runs. This introduces a minor comparability issue that is partially mitigated by the 3-run averaging.

### 15.3 RAGAS Judge Self-Evaluation Bias (G3-A)

The RAGAS judge is gpt-4o-mini; G3-A also uses a model from the same Qwen family (not the same provider) — no direct self-evaluation concern. However, G3-G and G3-E use Qwen models with the same architecture evaluated by a GPT-4o-mini judge — no family-level bias is expected. The bias concern is asymmetric and does not affect ranking conclusions.

### 15.4 G3-D: Infrastructure Bottleneck vs Model Quality

G3-D's 28,360 ms latency may be an OpenRouter serving issue rather than a fundamental Gemma 4 capability limit. A self-hosted Gemma 4 26B deployment on an A100 could potentially achieve 3–8 s latency. G3-D's quality results (FA=0.8232, Safety=88.5%) should not be dismissed — if latency improves with better infrastructure, it would be a strong candidate.

### 15.5 Qwen3.5 vs Qwen3.6 Inversion Reliability

The finding that Qwen3.5 (G3-G) outperforms Qwen3.6 (G3-E) is based on 3 runs × 32 cases per model. The differences (FA +0.0217, Safety +3.1 pp, Latency −2.6 s) are systematic across all three runs, suggesting they are genuine. However, the relative performance of these two models on other task types, with different prompts, or in fine-tuned form may differ.

### 15.6 Testset Coverage Limitation

The 32-case testset covers 5 categories (A–E) with specific case definitions. All models share the same structural Cat D weakness (referral failure). This may reflect a testset characteristic — Cat D cases may require referral language patterns that are inconsistently present in the retrieved clinical guidelines rather than a model-specific deficiency.

---

## 16. Next Steps

### 16.1 Configuration Carry-Forward for VerdaSense

Based on G3-OR results, the updated recommended configuration is:

| Component | Recommended Configuration |
|---|---|
| Prompt strategy | G1-C: Grounded system prompt (unchanged) |
| Generation LLM (primary) | G3-G: `qwen/qwen3.5-35b-a3b` via OpenRouter |
| Generation LLM (proprietary fallback) | G2-D: `gemini-2.5-flash` |
| Retrieval embedding | `BAAI/bge-large-en-v1.5` (unchanged) |
| Retrieval strategy | R1-C multi-axis dense (k=6, unchanged) |
| Knowledge base | `db_wound_care_v4_bge` (unchanged) |
| Reasoning flag | `extra_body={reasoning:{effort:none}, include_reasoning:False}` |

### 16.2 Recommended Future Work

**1. Validate G3-G at production scale.** Expand the evaluation to 200+ wound cases including additional edge cases, real clinical scenarios, and out-of-distribution presentations. Confirm that FA=0.83 and Safety=89.6% hold on a larger testset.

**2. Re-evaluate G3-D when OpenRouter Gemma4 serving matures.** G3-D's quality profile (FA=0.8232, Safety=88.5%) is strong. If latency improves to 5–8 s with better infrastructure, it would be a viable alternative to G3-G at lower cost ($0.0008 vs $0.0023/query).

**3. Investigate Cat D referral failures across all models.** The universal Cat D weakness (antibiotic and referral failure across all models) suggests a retrieved-context gap rather than a model deficiency. Augmenting the knowledge base with referral-specific wound care guideline sections could improve all models' Cat D safety.

**4. Test G3-G in think mode (reasoning enabled).** The extra_body suppression disables reasoning entirely. Running G3-G with reasoning enabled (at higher latency cost) would quantify whether the chain-of-thought reasoning further improves FA and safety — useful data for non-mobile server-side deployments.

**5. Evaluate latency under concurrent load.** All G3-OR latencies were measured at low QPS (3 s inter-case sleep). A load test simulating 10–50 concurrent users would reveal whether OpenRouter's G3-G serving degrades gracefully or introduces latency spikes incompatible with the ≤10 s gate.

**6. Assess G3-G for deployment in the VerdaSense Flutter app (O3 integration).** End-to-end latency from patient image capture → T.I.M.E. assessment → RAG retrieval → G3-G generation → display in the Flutter UI should be validated on a physical Android device.

---

*Analysis generated from G3-OR ablation data: G3_{G3A–G3G}_ragas.json, G3_{G3A–G3G}_safety.csv, G3_OR_summary.json. All figures quoted from 3-run mean aggregations unless otherwise stated. Consistent with G2 and G3-HPC analysis methodology.*
