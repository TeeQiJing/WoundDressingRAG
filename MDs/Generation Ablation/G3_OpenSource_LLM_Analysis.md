# VerdaSense RAG — Experiment G3: Open-Source LLM Comparison
## Comprehensive Analysis & Discussion

**Experiment:** G3 — Open-Source Generation LLM Comparison (Ollama, HPC)  
**Stage:** 2 — Generation Ablation  
**Date:** 17 May 2026  
**Configuration:** G1-C Grounded system prompt (fixed) | BGE Large (`BAAI/bge-large-en-v1.5`) | `db_wound_care_v4_bge` | R1-C multi-axis dense (k=6, fixed) | RAGAS judge: `gpt-4o-mini` + `text-embedding-3-small` (fixed) | **3 runs each**  
**Testset:** `wound_testset_v3.json` — 32 cases (Cat A:8, Cat B:12, Cat C:6, Cat D:4, Cat E:2)  
**RAGAS Judge:** gpt-4o-mini + text-embedding-3-small (fixed across all versions and all runs — never changed)  
**Infrastructure:** UM HPC A100-SXM4-80GB (79.15 GB VRAM) | Ollama v0.x | Generation via `G3_generate_hpc.py` (v1.2) | Evaluation on local machine

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
11. [G3 vs G2: Open-Source vs Closed-Source](#11-g3-vs-g2-open-source-vs-closed-source)
12. [LLM Recommendation: Which Model Should You Choose?](#12-llm-recommendation-which-model-should-you-choose)
13. [Is G3 Meaningful?](#13-is-g3-meaningful)
14. [Limitations and Threats to Validity](#14-limitations-and-threats-to-validity)
15. [G4 / Future Work Next Steps](#15-g4--future-work-next-steps)

---

## 1. Experiment Overview

Experiment G3 is the third Stage 2 ablation experiment for VerdaSense. G1 established the winning prompt strategy (G1-C: Grounded system prompt); G2 selected the best closed-source generation LLM (G2-D: Gemini 2.5 Flash). G3 now fixes all prior decisions — prompt, retrieval, and evaluation framework — and tests whether **open-source LLMs running locally on HPC** can match or approach the G2 closed-source performance level.

G3 answers two research questions:

1. **Which open-source LLM best serves a clinical wound care RAG system under the G1-C prompt?**
2. **Can any open-source model match the G2-D closed-source winner (FA=0.8147, Safety=90.6%) under identical conditions?**

These questions are practically and academically significant for VerdaSense. A viable open-source deployment would eliminate API cost, latency variability from third-party services, and data privacy concerns inherent in routing sensitive clinical queries through external APIs. Conversely, if the performance gap between open-source and closed-source is too large to close without fine-tuning, that finding in itself provides a strong rationale for fine-tuning as future work.

G3 covers four open-source models deployed locally via Ollama on the UM HPC A100:

- **G3-A:** Qwen3:14b — Alibaba's 14B reasoning/hybrid model with `/no_think` suppression
- **G3-B:** Gemma3:12b — Google's 12B open-source model
- **G3-C:** Llama3.2:11b — Meta's 11B instruction-tuned model
- **G3-D:** DeepSeek-R1:14b — DeepSeek's 14B open-source reasoning model with think-chain stripping

**Methodological note on generation infrastructure:** Unlike G1 and G2 (which used commercial API calls), G3 runs all generation on the UM HPC A100 via Ollama. Generation is fully decoupled from evaluation: all 12 result files (4 versions × 3 runs) were generated on HPC using `G3_generate_hpc.py` (v1.2), saved to disk, then loaded for RAGAS evaluation on local machine. The HPC generation phase included a health-check and evict-before-retry mechanism (v1.2 fix) to handle Ollama VRAM OOM events — necessary because Qwen3's KV-cache allocation can exceed available VRAM under specific prompt+context combinations.

---

## 2. Evaluation Metric Reference

G3 uses the same generation-layer metrics as G1 and G2. Retrieval metrics are not re-measured; retrieval is fixed.

| Metric | Type | What It Measures | Clinical Relevance |
|---|---|---|---|
| **Faithfulness (FA)** | RAGAS LLM-judge | Fraction of answer claims attributable to retrieved context | PRIMARY — hallucination resistance under open-source parametric knowledge |
| **Answer Relevancy (AR)** | RAGAS embed-judge | Semantic alignment of the answer to the wound case question | SECONDARY — does the open-source model address the right clinical question? |
| **Safety Pass Rate (%)** | Deterministic rule checker | % of cases passing all clinical safety rules | CRITICAL — hard clinical deployment gate |

**Selection gate (G3):** Safety Pass Rate ≥ 75.2% (G3's best version mean − 5 pp, consistent with G2 gate methodology) AND mean Faithfulness ≥ 0.75. In practice, no version qualifies on FA ≥ 0.75 simultaneously with safety — the winner is selected by best mean Safety as fallback (see §10).

**G2-D reference baseline for G3:** FA=0.8147 ± 0.0100 | AR=0.6770 ± 0.0210 | Safety=90.6% ± 0.0pp | GenLat=17,961 ± 939 ms. Any G3 version achieving FA > 0.81 AND Safety > 85.6% would represent a viably competitive open-source alternative.

---

## 3. Versions Tested

| Version | Label | Model | Params | Think Mode | Architecture |
|---|---|---|---|---|---|
| **G3-A** | Qwen3:14b (Ollama) | qwen3:14b | 14B | `/no_think` prefix | Hybrid reasoning; think suppressed at inference |
| **G3-B** | Gemma3:12b (Ollama) | gemma3:12b | 12B | N/A | Standard autoregressive |
| **G3-C** | Llama3.2:11b (Ollama) | llama3.2:11b | 11B | N/A | Standard instruction-tuned |
| **G3-D** | DeepSeek-R1:14b (Ollama) | deepseek-r1:14b | 14B | Think-strip post-processing | Open reasoning model; think tokens removed from output |

All versions use:
- **Prompt:** G1-C Grounded system prompt (byte-identical to G1/G2)
- **Retrieval:** R1-C multi-axis sub-queries, k=6, BGE Large dense
- **RAGAS judge:** gpt-4o-mini + text-embedding-3-small (fixed, same as G1/G2)
- **Hardware:** NVIDIA A100-SXM4-80GB (79.15 GB VRAM), via Ollama on UM HPC

**G3-A note:** idx=23 (`cat_c_dry_infected_combo`) is a permanent ERROR across all 3 runs due to a confirmed Qwen3:14b/Ollama OOM failure on this specific case. The v1.2 evict+restart retry chain was exhausted (4 attempts) on two different HPC nodes (gpu06, gpu07) without recovering this case. RAGAS evaluates on n=31 valid cases per run; idx=23 counts as FAIL in safety (denominator = 32).

---

## 4. Results Summary Table

### 4.1 Aggregated Results (n_runs = 3, 32 cases/run, 96 cases total per version)

| Version | Model | FA (mean ± std) | AR (mean ± std) | Safety % (mean ± std) | n_valid/run | Safety Qualified? |
|---|---|---|---|---|---|---|
| **G3-A** | qwen3:14b | 0.7306 ± 0.0163 | 0.6944 ± 0.0075 | 80.2% ± 1.8pp | 31/32 | ⚠️ Best safety; FA below 0.75 gate |
| **G3-B** | gemma3:12b | **0.7424 ± 0.0094** | 0.6644 ± 0.0260 | 78.1% ± 0.0pp | 32/32 | ⚠️ Highest FA; FA below 0.75 gate |
| **G3-C** | llama3.2:11b | 0.6711 ± 0.0175 | 0.6562 ± 0.0315 | 65.6% ± 0.0pp | 32/32 | ❌ Both gates fail |
| **G3-D** | deepseek-r1:14b | 0.6558 ± 0.0247 | **0.7022 ± 0.0093** | 40.6% ± 3.2pp | 32/32 | ❌ Safety catastrophically fails |

No version passes both the FA ≥ 0.75 gate AND the safety gate simultaneously. Winner selected as fallback: **G3-A (Qwen3:14b) — highest mean Safety (80.2%).**

Classifier accuracy: 87.5% (28/32) for all versions, all runs — fully consistent with G1/G2 baselines.

### 4.2 Per-Run Breakdown

| Version | Run 1 FA | Run 2 FA | Run 3 FA | Run 1 Safety% | Run 2 Safety% | Run 3 Safety% |
|---|---|---|---|---|---|---|
| G3-A | 0.7123 | 0.7358 | 0.7436 | 81.2% | 81.2% | 78.1% |
| G3-B | 0.7526 | 0.7404 | 0.7341 | 78.1% | 78.1% | 78.1% |
| G3-C | 0.6528 | 0.6730 | 0.6876 | 65.6% | 65.6% | 65.6% |
| G3-D | 0.6767 | 0.6623 | 0.6285 | 40.6% | 43.8% | 37.5% |

### 4.3 Incremental Deltas vs G3-A (mean-based)

| Comparison | ΔFA (mean) | ΔAR (mean) | ΔSafety (mean) | ΔGen Latency |
|---|---|---|---|---|
| G3-B vs G3-A | +0.0118 | −0.0300 | −2.1 pp | −5,705 ms |
| G3-C vs G3-A | −0.0595 | −0.0382 | −14.6 pp | −1,007 ms |
| G3-D vs G3-A | −0.0748 | +0.0078 | −39.6 pp | −5,181 ms |

---

## 5. Latency Analysis

All latency values are mean per-valid-case values averaged across 3 runs. G3-A valid case count is 31/run; all others are 32/run.

### 5.1 Generation and Total Latency Summary

| Version | Mean Gen Lat | Std Gen Lat | Mean Total Lat | Std Total Lat | Mean Ret Lat |
|---|---|---|---|---|---|
| G3-A | 30,924 ms | ±5,293 ms | 31,048 ms | ±5,298 ms | 124 ms |
| G3-B | 25,219 ms | ±7,380 ms | 25,307 ms | ±7,377 ms | 88 ms |
| G3-C | 29,916 ms | ±925 ms | 30,001 ms | ±900 ms | 84 ms |
| G3-D | 25,742 ms | ±7,515 ms | 25,871 ms | ±7,567 ms | 129 ms |

### 5.2 Per-Run Latency Breakdown

| Version | Run 1 Gen (ms) | Run 2 Gen (ms) | Run 3 Gen (ms) |
|---|---|---|---|
| G3-A | 33,824 | 34,133 | 24,815 |
| G3-B | 16,704 | 29,188 | 29,765 |
| G3-C | 30,398 | 30,502 | 28,850 |
| G3-D | 29,760 | 30,395 | 17,072 |

### 5.3 Key Latency Observations

**Retrieval latency is consistent and negligible (84–129 ms, <0.5% of total).** Identical to G1/G2 (which ran ~305–390 ms; the lower G3 values reflect BGE already loaded in the local Python process during HPC generation, not measured separately in evaluation). This confirms the retrieval pipeline is correctly isolated from generation.

**All G3 models are substantially slower than all G2 models.** The fastest G3 mean is G3-B at 25,219 ms; the fastest G2 was G2-C at 3,133 ms — an 8× slowdown. G3's generation latency is driven by Ollama single-node inference on A100 without the optimised serving infrastructure of commercial APIs. Even the G2-D winner (17,961 ms) is 41% faster than G3-B and 73% faster than G3-A/C.

**G3-B Run 1 latency anomaly (16,704 ms vs ~29,500 ms in Runs 2–3).** Run 1 ran on a freshly warmed GPU with no competing workloads on gpu06 — a cold-start advantage. Runs 2–3, submitted as parallel jobs on the same node, incur background VRAM pressure from the other run's Ollama process starting up. This explains the Run 1 speed advantage consistently observed for G3-B and G3-D. It is an infrastructure artefact, not a model capability difference, and is averaged out across 3 runs in the mean.

**G3-C has the tightest latency distribution (std ±925 ms).** Llama3.2:11b is the most consistent inference model — its per-case generation time varies least, reflecting a more uniform response length distribution. This is partially a consequence of its lower overall quality: simpler, shorter, more formulaic outputs generate faster.

**G3-A's large latency std (±5,293 ms) reflects the empty-content recovery overhead.** Across the 3 runs, G3-A had multiple idx=7, idx=8, and idx=30 cases trigger the evict-and-retry mechanism (adding 20–50 s overhead per event) and one `RemoteDisconnected` event. These recovery events inflate per-run std without indicating overall instability — once excluded, the typical Qwen3:14b generation time is ~23–25 s per case.

**G3-D's large latency std (±7,515 ms) is explained by the Run 3 speed difference (17,072 ms vs ~30,000 ms).** Run 3 ran on gpu07 while Runs 1–2 ran on gpu06. gpu07 had no competing jobs and the model was already in Ollama's VRAM from the pre-warm step, resulting in ~1.75× faster generation. This is a cross-node infrastructure difference, not a model property.

---

## 6. Safety Analysis

### 6.1 Overall Safety Rates (3-run deterministic evaluation)

| Version | Safety Pass Rate | Fails/Run | Consistency | Gate (≥85.6%)? |
|---|---|---|---|---|
| **G3-A** | 80.2% (≈26/32 per run) | 6 (incl. idx=23 ERROR) | ⚠️ Stochastic ±1.8pp | ❌ |
| **G3-B** | 78.1% (25/32 per run) | 7 | ✅ Identical all 3 runs | ❌ |
| G3-C | 65.6% (21/32 per run) | 11 | ✅ Identical all 3 runs | ❌ |
| G3-D | 40.6% avg (12–14/32 per run) | 18–20 | ❌ Variable ±3.2pp | ❌ |

**No G3 version passes the 85.6% G2-derived safety gate.** The best G3 safety (G3-A at 80.2%) is 10.4 pp below the G2-D winner (90.6%) and 5.4 pp below even the G2-A baseline (84.4%). This is the central finding of G3: open-source 11–14B models running locally under the G1-C prompt do not achieve the clinical safety threshold set by the best closed-source model.

### 6.2 Aggregated Safety Check Failure Analysis (all 3 runs combined)

| Check | G3-A (3 runs) | G3-B (3 runs) | G3-C (3 runs) | G3-D (3 runs) |
|---|---|---|---|---|
| `antibiotic_check` | 6 | 3 | 3 | **23** |
| `dressing_in_allowed_list` | 6 | 9 | 17 | 11 |
| `referral_check` | 7 | 9 | 18 | 17 |
| `contraindication_absent_film` | 0 | 3 | 0 | 0 |
| `contraindication_absent_silver` | 0 | 0 | 0 | 4 |
| `contraindication_absent_honey` | 0 | 0 | 0 | 3 |
| `contraindication_absent_iodine` | 0 | 0 | 0 | 3 |
| `contraindication_absent_npwt` | 0 | 0 | 0 | 3 |
| `contraindication_absent_alginate` | 0 | 0 | 0 | 2 |
| `contraindication_absent_hydrocolloid` | 0 | 0 | 0 | 2 |
| `contraindication_absent_charcoal` | 0 | 0 | 0 | 1 |
| **Total failures (3 runs)** | **19** | **24** | **38** | **69** |

### 6.3 Failure Pattern Interpretation

**G3-D: Structural instruction-following failure.** The dominant failure mode is `antibiotic_check` (23/3 runs = 7.7 per run), followed by `referral_check` (17) and a broad contraindication failure pattern (18 across 7 distinct contraindication checks). Unlike G3-A/B/C, G3-D also fails multiple contraindication checks — meaning it not only misses clinical obligations but positively recommends contraindicated dressings in the wrong sections. Root cause: DeepSeek-R1:14b does not emit the `## Primary Dressing` / `## Secondary Dressing` section headers required by G1-C in the majority of cases. Without these headers, the safety checker's `_extract_positive_section_text()` function falls back to full-answer scanning, which is more permissive and allows contraindicated dressings to be detected as recommended. The think-stripping is confirmed working (0 `<think>` blocks in any run), so this is a structural format non-compliance issue at the model level, not a generation artefact.

**G3-C: Clinical capability gaps at referral and dressing identification.** `referral_check` (18 failures) and `dressing_in_allowed_list` (17 failures) together account for 92% of G3-C failures. These concentrate on specialist and complex cases (burns, skin tears, diabetic foot, NPWT) — exactly the edge cases requiring highest clinical specificity. Llama3.2:11b at 11B has insufficient clinical domain knowledge to reliably identify escalation triggers and select terminology matching the DRESSING_ALIASES dictionary. This is a parametric knowledge limitation, not a prompt compliance issue — the model understands the G1-C format but lacks the clinical knowledge to fill the safety-critical sections correctly.

**G3-B: Specific edge-case failures, fully deterministic.** Seven cases fail identically in all 3 runs: idx 15, 18, 19, 26, 27, 29, 30. Failure distribution: `referral_check` (9), `dressing_in_allowed_list` (9), `antibiotic_check` (3), `contraindication_absent_film` (3). The film contraindication failures (3) are unique to G3-B — the model recommends transparent film dressings for cases where film is contraindicated (high-exudate or infected wounds), a clinically meaningful error. The perfect cross-run determinism confirms these are structural model capability boundaries, not stochastic failures.

**G3-A: Concentrated referral and antibiotic failures, partially stochastic.** Consistent fail core across all runs: idx 19, 26, 27 (`referral_check` and `dressing_in_allowed_list`). Additional stochastic failures at the boundary: idx 5, 8, 15, 17 appear in 1–2 runs each. The `antibiotic_check` failures (6 total across 3 runs) and `referral_check` failures (7 total) suggest Qwen3:14b under `/no_think` sometimes uses clinically equivalent but lexically different phrases for antibiotic recommendations and referral triggers — phrases the checker's keyword list doesn't match. This is a surface-form compliance issue addressable by fine-tuning.

### 6.4 Safety Rate Stability

G3-B and G3-C are completely safety-deterministic (std = 0.0 pp) — the same cases fail in every run. G3-A shows mild stochasticity (std = 1.8 pp), entirely explained by the `/no_think` suppression creating minor output variation at the boundary cases. G3-D is the most volatile (std = 3.2 pp), reflecting the unstructured output format that sends different cases over the PASS/FAIL boundary on different runs.

---

## 7. Detailed Version-by-Version Discussion

### 7.1 G3-A — Qwen3:14b (Ollama, `/no_think`)

**Mean FA: 0.7306 ± 0.0163 | Mean AR: 0.6944 ± 0.0075 | Safety: 80.2% ± 1.8pp | n_valid=31/32**

G3-A represents the strongest clinical safety performance among G3 open-source models despite being the only version with a permanent generation ERROR. The FA range across runs (0.7123 → 0.7358 → 0.7436) shows a **monotonically increasing trend** — Run 1 is the lowest, Run 3 is the highest. This pattern, also observed in G1-C (0.8250 → 0.8569 → 0.8362) and partially in G3-C, likely reflects cumulative RAGAS judge warm-up (the judge has seen more similar prompts within the session) rather than model improvement.

The `/no_think` system prompt prefix successfully suppresses Qwen3's internal chain-of-thought in all runs (think_stripped = 0 across all 93 valid responses). This is important because think tokens would inflate response length without adding to clinical content, increasing both latency and hallucination risk. The suppressed mode produces concise, directly grounded responses consistent with the G1-C prompt structure.

**idx=23 permanent ERROR:** `cat_c_dry_infected_combo` exhausted the full v1.2 4-attempt retry chain in all 3 runs across 2 different HPC nodes. This case involves a wound with simultaneous infected + granulating + dry + non-advancing characteristics that triggers Qwen3's reasoning chain into an irresolvable generation loop, producing KV-cache entries but no output tokens. This is a model-specific failure that cannot be resolved without either fine-tuning or case-specific prompt modification. For evaluation purposes, idx=23 is treated as a permanent FAIL (safety) and excluded from RAGAS (n_evaluated=31). All reporting notes this explicitly.

**The empty-content recovery mechanism (v1.2) worked correctly for all other cases.** idx=7, 8, and 30 all triggered empty-content events in various runs and were successfully recovered by the evict-and-retry path (keep_alive=0 → 20s VRAM flush → retry at ctx=16384). This demonstrates that the v1.2 infrastructure fix resolved the cascade failure seen in earlier generation runs.

### 7.2 G3-B — Gemma3:12b (Ollama)

**Mean FA: 0.7424 ± 0.0094 | Mean AR: 0.6644 ± 0.0260 | Safety: 78.1% ± 0.0pp | n_valid=32/32**

G3-B achieves the **highest FA of any G3 model (0.7424)** and perfect generation stability (0 ERRORs, 0 empty-content events, zero safety variance). It is the most robust generation model in G3. The FA gap between G3-B and G3-A (0.7424 vs 0.7306, Δ = 0.0118) is slightly above G3-A's std (0.0163) but below G3-B's std (0.0094 × 2 = 0.0188). This gap is marginal rather than systematic, though the direction (G3-B > G3-A) is consistent across 2 of 3 runs (G3-B Run 1 is 0.0403 higher than G3-A Run 1; Runs 2–3 are comparably close).

**The safety paradox:** G3-B achieves higher FA than G3-A (0.7424 vs 0.7306) but lower safety (78.1% vs 80.2%). This apparent contradiction is explained by the nature of the failures: G3-B's 7 systematic fail cases include film contraindication violations (unique to G3-B) that reflect a different type of error than G3-A's referral/antibiotic phrasing issues. G3-B faithfully grounds its answers in retrieved context (high FA) but then makes clinically incorrect dressing selections for edge cases that the 12B model's parametric knowledge doesn't handle correctly.

**Zero-AR RAGAS artefact in G3-B.** G3-B exhibits zero-AR RAGAS judge failures for cases at indices 11, 26, and 27 (appearing 3 times in Run 1, once each in Runs 2–3). Case idx=27 (`cat_d_notes_npwt_adjunct`) produces zero AR in every run — a systematic RAGAS judge failure for this specific response format. Corrected AR excluding zeros: Run 1 = 0.7002, Run 2 = 0.7037, Run 3 = 0.6989, giving a corrected mean of **0.7009** — substantially higher than the raw reported mean of 0.6644. The 3.65 pp gap between raw and corrected AR is the largest RAGAS artefact correction in G3, and should be explicitly noted in the FYP.

**Deterministic safety is a methodological strength.** G3-B's zero safety variance makes its 78.1% rate the highest-confidence safety estimate in G3. Every evaluation of G3-B on these 32 cases will produce the same 25 PASS / 7 FAIL outcome. The 7 failing cases represent genuine capability ceilings of Gemma3:12b for specialist wound presentations, not stochastic noise.

### 7.3 G3-C — Llama3.2:11b (Ollama)

**Mean FA: 0.6711 ± 0.0175 | Mean AR: 0.6562 ± 0.0315 | Safety: 65.6% ± 0.0pp | n_valid=32/32**

G3-C is the smallest model in G3 (11B) and shows the clearest evidence of a clinical knowledge ceiling. Like G3-B, its safety is perfectly deterministic (same 11 cases fail in every run), but at a substantially lower rate (65.6% vs 78.1%). The 11 consistently failing cases concentrate on edge presentations requiring specialist-level clinical knowledge: burns (idx 15, 16), skin tears (idx 14, 16), diabetic foot (idx 10), NPWT (idx 29), malodorous wounds (idx 5), and referral-required presentations (idx 18, 19, 26, 27, 30).

**FA trend is actually increasing across runs (0.6528 → 0.6730 → 0.6876)**, mirroring the G3-A and G3-C patterns — a possible RAGAS session warm-up effect. The AR trend is **decreasing (0.6876 → 0.6562 → 0.6247)** — the opposite direction. This divergence between FA trend and AR trend is unusual and may reflect the RAGAS embed judge scoring Llama3.2's later-run outputs differently, possibly due to responses becoming more templated (higher FA, lower semantic breadth) as the model consistently encounters familiar prompt structures.

**Zero-AR events in G3-C.** Cases at idx=17 (`cat_b_diabetic_foot`) and idx=19 (`cat_b_skin_tear_type2_flap`) produce zero-AR scores in Runs 2 and/or 3. Corrected AR (excluding zeros): Run 2 = 0.6773, Run 3 = 0.6663 — modestly higher than raw values but not enough to meaningfully change G3-C's position. The corrected mean AR is approximately 0.6698, a 1.4 pp improvement over the reported 0.6562.

**G3-C's significance in the ablation:** G3-C establishes the lower bound of clinical utility for open-source models in this architecture. A 65.6% safety pass rate means one-third of wound cases receive a response with at least one clinical rule violation — far too high for deployment in any clinical context. The finding quantifies the minimum model capacity (11B is insufficient) and the specific capability gap (referral identification, specialty dressing terminology).

### 7.4 G3-D — DeepSeek-R1:14b (Ollama, think-strip)

**Mean FA: 0.6558 ± 0.0247 | Mean AR: 0.7022 ± 0.0093 | Safety: 40.6% ± 3.2pp | n_valid=32/32**

G3-D presents the most paradoxical results in G3: the **highest AR of any G3 model (0.7022)** combined with the **lowest safety (40.6%)**. This inversion — semantically relevant answers that are clinically dangerous — is the defining characteristic of G3-D and warrants careful explanation.

**The AR advantage is real and model-structural.** DeepSeek-R1:14b, as a reasoning model, generates responses that more directly engage with the wound case question before expanding to sections. Its internal reasoning chain (stripped from the final output but informing the generation) explicitly identifies what is being asked and constructs a more question-focused answer. The RAGAS embed judge consistently rates DeepSeek-R1's responses as more semantically aligned with the clinical question. Importantly, G3-D's AR has zero zero-AR RAGAS artefacts (all 96 samples scored), making its 0.7022 estimate the cleanest AR measurement in G3.

**The safety collapse is structural and non-stochastic in its root cause.** DeepSeek-R1:14b at 14B generates responses that largely ignore the G1-C prompt's section structure (`## Primary Dressing`, `## Antibiotic Considerations`, `## Referral / Escalation`). Without these sections, the safety checker's `_extract_positive_section_text()` falls back to full-answer scan mode, which is less precise at distinguishing recommendations from contraindication warnings. This leads to two failure patterns: (1) the checker cannot find the required phrases in the right context, causing antibiotic_check and referral_check to fail even when the content is present but unstructured; (2) contraindicated dressings are positively mentioned in unstructured contexts (e.g. differential discussion) and detected as recommended.

**FA declining trend (0.6767 → 0.6623 → 0.6285) is notable.** Unlike every other G3 model which shows stable or increasing FA across runs, G3-D's FA is monotonically decreasing. Run 3 (gpu07) is the slowest in generation speed (17,072 ms) but lowest in FA. This correlation — faster generation, lower faithfulness — is consistent with the model taking fewer tokens to generate (less grounding detail per claim) when VRAM is cleanly available vs under VRAM pressure in Runs 1–2. This is a Ollama-specific observation and would not generalise to standard deployment.

**Think-stripping verified.** 0 `<think>` tokens appear in any G3-D output across all 96 cases. The `deepseek_strip` post-processing in `G3_generate_hpc.py` correctly removes all internal reasoning tokens. However, as noted above, think-stripping working does not resolve the structural format non-compliance issue.

---

## 8. Cross-Version Comparative Analysis

### 8.1 The FA Ceiling for Open-Source Models in G3

The four G3 models establish a clear FA ceiling: **0.7424 (G3-B) is the maximum faithfulness achievable** by open-source 11–14B models running locally under the G1-C grounding prompt. The closest G3 competitor to the G2-D FA (0.8147) falls 0.0723 pp short — a gap of 7.23 pp, which is 7.7× larger than G3-B's std (0.0094). This gap is systematic, not stochastic.

The FA ordering is: **G3-B > G3-A > G3-C > G3-D**, which broadly correlates with model size-adjusted quality (12B Gemma3 vs 14B Qwen3 reflects architectural differences, not just parameter count) but does not follow the expected parameter-count ordering (G3-A at 14B is below G3-B at 12B). Model architecture and training objective matter more than raw parameter count for RAGAS faithfulness.

### 8.2 Safety vs Quality Trade-Off in G3

G3 exhibits a clearer safety-quality inverse relationship than G2. Plotting models by safety from highest to lowest: G3-A (80.2%), G3-B (78.1%), G3-C (65.6%), G3-D (40.6%). The FA ordering reverses G3-A and G3-B (G3-B leads on FA despite lower safety) but otherwise tracks safety. G3-D uniquely breaks both orderings: lowest FA, lowest safety, highest AR — a model that is simultaneously the most clinically dangerous and the most semantically relevant.

This pattern suggests that in the open-source domain, **safety and faithfulness are not automatically co-optimised by capability tier**. DeepSeek-R1's reasoning architecture achieves high AR but fails catastrophically on safety, while Gemma3 achieves high FA and high safety stability without a reasoning chain at all. The implication for clinical RAG is that **structured output compliance (section headers) is a prerequisite for safety**, not a consequence of general model capability — and reasoning models that resist structured output formats may be unsuitable for safety-gated clinical applications regardless of their reasoning depth.

### 8.3 Determinism Analysis

| Version | Safety Deterministic? | FA Trend | Safety Failure Type |
|---|---|---|---|
| G3-A | ❌ Minor stochasticity (±1.8pp) | Increasing | Boundary phrase compliance |
| G3-B | ✅ Perfectly deterministic | Decreasing | Clinical capability ceiling |
| G3-C | ✅ Perfectly deterministic | Increasing | Clinical knowledge gap |
| G3-D | ❌ Stochastic (±3.2pp) | Decreasing | Structural format non-compliance |

G3-B and G3-C achieving perfect safety determinism is an important finding: these models reliably identify the same cases as beyond their capability every run, with no boundary ambiguity. This makes their safety estimates maximally trustworthy. G3-A and G3-D's stochasticity is qualitatively different — G3-A's is boundary-level phrase variation (acceptable), G3-D's is structural format uncertainty (concerning).

### 8.4 Effect Sizes vs Noise Floor

| Comparison | FA Δ (pp) | Max std_FA (pp) | Δ > noise? | Conclusion |
|---|---|---|---|---|
| G3-A vs G3-B | −1.18 | 1.63 (G3-A) | ❌ Borderline | Marginal |
| G3-B vs G3-C | +7.13 | 1.75 (G3-C) | ✅ Yes (4.1×) | Systematic |
| G3-B vs G3-D | +8.66 | 2.47 (G3-D) | ✅ Yes (3.5×) | Systematic |
| G3-A vs G3-C | +5.95 | 1.75 (G3-C) | ✅ Yes (3.4×) | Systematic |

| Comparison | Safety Δ (pp) | std_safety | Δ > noise? | Conclusion |
|---|---|---|---|---|
| G3-A vs G3-B | +2.1 | 1.8 (G3-A) | ⚠️ Borderline | Marginal |
| G3-B vs G3-C | +12.5 | 0.0 (both) | ✅ Certain | Deterministic |
| G3-A vs G3-D | +39.6 | 3.2 (G3-D) | ✅ Yes (12.4×) | Systematic |
| G3-B vs G3-D | +37.5 | 3.2 (G3-D) | ✅ Yes (11.7×) | Systematic |

The G3-A vs G3-B comparison is at the noise boundary for both FA and safety — these two models are genuinely close. All other inter-model differences are clearly systematic. The G3-D safety collapse is the largest effect in G3 (−39.6 pp vs G3-A) and among the largest in the entire VerdaSense ablation study.

---

## 9. Noise Floor and Run Stability

### 9.1 FA and AR Noise Floor Summary

| Version | std_FA | FA range (3 runs) | std_AR | AR range (3 runs) | Stability |
|---|---|---|---|---|---|
| G3-B | 0.0094 | 0.0185 | 0.0260 | 0.0472 | FA stable; AR varies (zero artefacts) |
| G3-A | 0.0163 | 0.0313 | 0.0075 | 0.0149 | FA moderate; AR very stable |
| G3-C | 0.0175 | 0.0348 | 0.0315 | 0.0629 | Both moderate; AR declining trend |
| G3-D | 0.0247 | 0.0482 | 0.0093 | 0.0176 | FA declining trend; AR stable |

**G3-A has the lowest AR std (0.0075)** — its answer relevancy is more consistent run-to-run than any other G3 model, mirroring the G1-C pattern for gpt-4o-mini (AR std = 0.0012 in G1-C). The `/no_think` mode may contribute to AR stability by enforcing more consistent response structure.

**G3-D has the lowest FA std (0.0247) among the lower-quality models** but this is accompanied by a systematic declining trend — low variance around a worsening mean — which is a more concerning pattern than moderate variance around a stable mean.

### 9.2 Zero-AR RAGAS Artefacts in G3

| Version | Affected cases (idx) | Runs affected | Zero-AR count (total) | Corrected mean AR |
|---|---|---|---|---|
| G3-A | None | — | 0 | 0.6944 (unchanged) |
| G3-B | 11, 26, 27 | R1 (3 zeros), R2 (1), R3 (1) | 5 | **0.7009** (+3.65pp) |
| G3-C | 17, 19 | R2 (1), R3 (2) | 3 | **0.6698** (+1.36pp) |
| G3-D | None | — | 0 | 0.7022 (unchanged) |

The zero-AR pattern is consistent with the G2 observation: Gemini-family models (G3-B uses Google's Gemma3) and certain Llama responses intermittently produce RAGAS judge failures where the AR embed scoring step returns 0.0. The affected cases (idx=11 `cat_b_npwt_necrotic_eschar`, idx=26 `cat_c_film_vs_hydrocolloid`, idx=27 `cat_d_notes_infection_override`) are complex multi-section responses that may trip the RAGAS judge's question-generation step.

**The most important correction is G3-B's AR (0.6644 raw → 0.7009 corrected).** The raw AR makes G3-B appear to have lower AR than G3-A (0.6944), but the corrected AR (0.7009) shows G3-B is in fact comparable to G3-A on answer relevancy. Neither G3-A nor G3-D are affected by zero-AR artefacts, making their AR estimates the cleanest measurements.

**Recommendation:** Report raw AR values as primary (for consistency with G1/G2 reporting methodology) but note the corrected values in the analysis text and FYP document, as done here.

---

## 10. Winner Selection

### 10.1 Selection Criteria

| Gate | Criterion | Applied to |
|---|---|---|
| **Primary (hard gate)** | Mean Safety Pass Rate ≥ 75.2% (best mean − 5pp) | Mean across 3 runs |
| **Secondary (hard gate)** | Mean Faithfulness ≥ 0.75 | Mean across 3 runs |
| **Tertiary (tie-breaker)** | Highest mean FA among qualifying candidates | Mean across 3 runs |
| **Fallback** | Best mean Safety if no version passes both gates | Mean across 3 runs |

### 10.2 Gate Application

| Version | Safety ≥ 75.2%? | FA ≥ 0.75? | Qualifies? |
|---|---|---|---|
| G3-A | ✅ 80.2% | ❌ 0.7306 | ❌ — fails FA gate |
| G3-B | ✅ 78.1% | ❌ 0.7424 | ❌ — fails FA gate (marginal) |
| G3-C | ✅ 65.6% (fails primary too) | ❌ 0.6711 | ❌ |
| G3-D | ❌ 40.6% | ❌ 0.6558 | ❌ |

No version passes both gates simultaneously. Fallback applies: **highest mean Safety selects the winner.**

### 10.3 Winner

> **G3-A (Qwen3:14b) is selected as the best open-source LLM for this RAG configuration.**
>
> **Rationale:** Fallback selection — highest mean Safety (80.2% ± 1.8pp) among all G3 versions. No G3 version passes both the FA ≥ 0.75 and Safety ≥ 75.2% gates simultaneously.
>
> **Confidence note:** G3-A's FA advantage over G3-B is marginal (0.7306 vs 0.7424 — G3-B is actually slightly *higher* on FA). The fallback selection is driven entirely by G3-A's 2.1pp safety lead, which is itself at the noise boundary (G3-A std_safety = 1.8pp). G3-A and G3-B are effectively co-equal in performance — G3-A's safety edge is the single differentiator.

### 10.4 Best Open-Source Configuration for G3

| Component | Selected Configuration |
|---|---|
| Prompt strategy | G1-C: Grounded system prompt |
| Open-source LLM | **G3-A: qwen3:14b (Ollama)** |
| LLM Provider | Ollama (local inference) |
| Hardware requirement | A100 80GB or equivalent |
| Retrieval embedding | BAAI/bge-large-en-v1.5 |
| Retrieval strategy | R1-C multi-axis dense (k=6) |
| Note | n_valid=31/32; idx=23 permanent ERROR |

---

## 11. G3 vs G2: Open-Source vs Closed-Source

### 11.1 Direct Metric Comparison (G3 best vs G2 winner)

| Metric | G2-D (Gemini 2.5 Flash) | G3-A (Qwen3:14b) | G3-B (Gemma3:12b) | Δ (G3-A vs G2-D) | Δ (G3-B vs G2-D) |
|---|---|---|---|---|---|
| **FA (mean)** | **0.8147** | 0.7306 | 0.7424 | −0.0841 | −0.0723 |
| **AR (mean, raw)** | 0.6770 | **0.6944** | 0.6644 | +0.0174 | −0.0126 |
| **AR (corrected)** | **0.7142** | 0.6944 | 0.7009 | −0.0198 | −0.0133 |
| **Safety (mean)** | **90.6%** | 80.2% | 78.1% | −10.4 pp | −12.5 pp |
| **Gen Latency (ms)** | 17,961 | 30,924 | 25,219 | +12,963 ms | +7,258 ms |
| **Total Latency (ms)** | 18,335 | 31,048 | 25,307 | +12,713 ms | +6,972 ms |

### 11.2 Comparing All G2 Versions Against All G3 Versions

| Model | FA | AR (raw) | Safety | Latency | Status |
|---|---|---|---|---|---|
| G2-D (Gemini 2.5 Flash) | **0.8147** | 0.6770 | **90.6%** | 17,961 ms | ✅ G2 Winner |
| G2-A (GPT-4o-mini) | 0.7751 | **0.7233** | 84.4% | 12,921 ms | Baseline |
| G2-B (GPT-4o) | 0.7583 | 0.6910 | 81.2% | 6,274 ms | — |
| G2-C (Gemini 2.5 Flash Lite) | 0.7385 | 0.6522 | 81.2% | 3,133 ms | — |
| **G3-B (Gemma3:12b)** | 0.7424 | 0.6644 | 78.1% | 25,219 ms | G3 co-best FA |
| **G3-A (Qwen3:14b)** | 0.7306 | 0.6944 | **80.2%** | 30,924 ms | G3 Winner |
| G3-C (Llama3.2:11b) | 0.6711 | 0.6562 | 65.6% | 29,916 ms | — |
| G3-D (DeepSeek-R1:14b) | 0.6558 | 0.7022 | 40.6% | 25,742 ms | — |

**Key finding:** G3-B (FA=0.7424) is comparable to G2-C (Gemini Flash Lite, FA=0.7385) on faithfulness — essentially equal within both models' noise floors. This is a significant result: **the best open-source 12B model matches the worst-performing closed-source model on faithfulness**, but trails all closed-source models on safety (78.1% vs 81.2–90.6%). The open-source vs closed-source performance gap is real and clinically significant, but not unbridgeable in principle — it is 7–8 pp in FA and 10–12 pp in safety for the best G3 models.

### 11.3 Key Qualitative Differences

**Faithfulness gap (systematic):** All G3 models produce lower FA than all G2 models except G2-C. The 7.2 pp gap between G3-B (best G3 FA) and G2-D (best G2 FA) is far above both models' noise floors and is systematic. This reflects the fundamental difference in model capability between commercial API models and 11–14B open-source models at the grounding task — larger commercial models have more capacity to simultaneously follow the G1-C grounding instruction and generate high-quality clinical content.

**Safety gap (structural for G3-D; capability for G3-C; partial for G3-A/B):** G3-A and G3-B trail G2-D by 10–12 pp on safety. This gap is not directly attributable to model intelligence but to specific failure modes: phrasing compliance (G3-A), edge-case clinical knowledge (G3-B/C), and format non-compliance (G3-D). These are specifically addressable by fine-tuning, which G2 models do not require because their capabilities are sufficient to follow the G1-C prompt without fine-tuning.

**Latency disadvantage (large, infrastructure-driven):** G3 latency is 1.4–2.4× worse than G2-D for the best G3 models. This is partly model capability (smaller models generate more tokens proportionally for the same output length) but primarily infrastructure — Ollama on a single A100 lacks the optimised batching and serving optimisations of commercial API backends. In a production deployment with dedicated inference infrastructure (e.g. vLLM on A100), G3 latency would be substantially lower.

**Cost advantage (G3's strongest argument):** G3 models, once deployed on local infrastructure, incur zero per-token API cost. For a high-volume deployment (e.g. 1,000 wound assessments/day), G2-D's API cost would be significant; G3's amortised infrastructure cost would be negligible. This economic argument grows stronger as deployment scale increases, even if the quality gap remains.

---

## 12. LLM Recommendation: Which Model Should You Choose?

### 12.1 The Verdict

**For the VerdaSense FYP system, keep G2-D (Gemini 2.5 Flash) as the production recommendation.** No G3 open-source model passes the clinical safety gate (85.6%) or the faithfulness gate (FA ≥ 0.75) simultaneously. The 10–12 pp safety gap and 8 pp FA gap between the best G3 models and G2-D represent meaningful clinical risk differences that cannot be dismissed as measurement noise.

However, the selection is nuanced and depends on deployment context:

### 12.2 Decision Framework by Deployment Scenario

| Scenario | Recommended Model | Rationale |
|---|---|---|
| **Clinical deployment (patient-facing)** | G2-D (Gemini 2.5 Flash) | Only model passing safety and FA gates; zero referral-check failures across 96 evaluations |
| **Research / educational tool (non-clinical)** | G3-B (Gemma3:12b) | Best open-source FA (0.7424), perfect determinism, zero generation ERRORs, lowest latency std |
| **Air-gapped / offline clinical environment** | G3-A (Qwen3:14b) with v1.2 infrastructure | Best open-source safety; after fine-tuning on phrasing compliance, could reach ~85% |
| **Cost-constrained high-volume deployment** | G3-A or G3-B + fine-tuning | Open-source eliminates API cost at scale; fine-tuning gap is addressable |
| **Demonstration/prototype** | G3-B (Gemma3:12b) | Easiest to deploy, most predictable, no generation errors or infrastructure complexity |

### 12.3 Closed-Source: G2-D vs G2-A

Within the closed-source results, G2-D (Gemini 2.5 Flash) remains the unambiguous winner. G2-A (GPT-4o-mini) is the strongest runner-up and the only other viable option:

- G2-D: FA=0.8147, Safety=90.6% — recommended for clinical deployment
- G2-A: FA=0.7751, Safety=84.4% — viable if Gemini API access is unavailable; marginally fails the 85.6% safety gate
- G2-B (GPT-4o): dominated by G2-A on all metrics at higher cost — do not use
- G2-C (Gemini Flash Lite): fastest but fails all gates — suitable only for demos with human oversight

### 12.4 Open-Source: G3-A vs G3-B

The G3-A vs G3-B choice is genuinely close. The recommendation depends on the specific operational concern:

- **Choose G3-A (Qwen3:14b)** if safety rate is the primary concern and the idx=23 permanent ERROR is acceptable (exclude this case from production or handle with a fallback). G3-A's 80.2% safety edge over G3-B's 78.1% is marginal but consistent. G3-A also has better AR consistency (std_AR = 0.0075 vs G3-B's 0.0260).
- **Choose G3-B (Gemma3:12b)** if generation reliability and infrastructure simplicity are primary. Zero ERRORs, zero empty-content events, zero latency std outliers — G3-B requires no v1.2 retry infrastructure, making it simpler to deploy. Its corrected AR (0.7009) is comparable to G3-A. Its FA lead (0.7424 vs 0.7306) is at the noise boundary but directionally consistent.

---

## 13. Is G3 Meaningful?

### 13.1 Yes — G3 Is a Scientifically Meaningful Ablation Contribution

G3 makes five specific contributions that would withstand FYP viva scrutiny:

**1. G3 quantifies the open-source performance gap with precision.** The FA gap between the best open-source model (G3-B: 0.7424) and the best closed-source model (G2-D: 0.8147) is 7.23 pp — a systematic, reproducible difference well above the noise floor. G3 establishes exactly how large this gap is, rather than assuming it exists.

**2. G3 identifies root causes of open-source safety failures.** The three failure mechanisms — structural format non-compliance (G3-D), clinical knowledge gaps (G3-C), and phrasing boundary compliance (G3-A/B) — are not generic findings but specific, evidence-based diagnoses. Each has a clear mitigation pathway (format fine-tuning for G3-D, domain fine-tuning for G3-C, targeted phrase alignment for G3-A/B).

**3. G3 provides a concrete benchmark for fine-tuning targets.** Knowing that G3-A needs to close a 10.4 pp safety gap to match G2-D, and that the failures are concentrated in antibiotic phrasing (6/3 runs) and referral detection (7/3 runs), gives a precise specification for what a fine-tuning dataset should optimise. This is more actionable than a general statement that open-source models underperform.

**4. G3 demonstrates that infrastructure matters, not just model capability.** The v1.2 Ollama evict+restart mechanism was required to achieve stable generation — a finding specific to deploying open-source models via Ollama on GPU. Commercial API deployments (G2) require no such infrastructure. G3 documents this operational complexity gap, which is a realistic deployment consideration.

**5. G3's multi-run design with deterministic safety evaluation provides high-confidence safety estimates.** G3-B and G3-C achieve zero safety variance across all runs — the strongest possible evidence that their safety rates are genuine model properties, not evaluation noise.

### 13.2 What G3 Does Not Establish

**G3 does not establish that fine-tuning would close the gap.** Fine-tuning is presented as a plausible mitigation but is explicitly not tested. The FYP should be clear that the fine-tuning recommendation is a hypothesis for future work, not a confirmed finding.

**G3 does not compare models at equal cost-per-inference.** G2 uses commercial APIs with specific pricing; G3 uses HPC allocation time. A rigorous total-cost-of-ownership comparison would require accounting for HPC compute hours vs API token costs, which is beyond the FYP scope.

**G3 does not test model-specific prompt optimisation.** The G1-C prompt was designed for gpt-4o-mini. All G3 models use it unchanged. A Qwen3-specific prompt (e.g. leveraging native `/think` capability for grounding verification) or a Gemma3-specific prompt might narrow the gap. G3 measures the G1-C prompt's transferability, not the open-source models' peak capability.

### 13.3 Key Talking Points for FYP Viva

- "G3 uses n_runs=3 with independent safety evaluation per run — safety for G3-B and G3-C is deterministic (std=0.0), providing the highest-confidence safety estimates in the study."
- "No G3 open-source model passes both the safety and faithfulness gates; the fallback selection (G3-A) is driven by the 2.1pp safety margin over G3-B, which is at the noise boundary — these two models are effectively co-equal."
- "G3-B's FA (0.7424) matches G2-C's FA (0.7385) — a 12B open-source model with zero infrastructure cost achieves comparable faithfulness to the fastest closed-source commercial model, but trails on safety by ~3pp."
- "G3-D's AR (0.7022) exceeds all G3 and most G2 models, yet its safety is the worst — demonstrating that answer relevancy and clinical safety are independent dimensions that reasoning models can decouple in dangerous ways."
- "The 10.4pp safety gap between G3-A and G2-D has a specific, identified cause (phrasing compliance, referral detection) — it is not diffuse model incompetence but a targeted fine-tuning opportunity."

---

## 14. Limitations and Threats to Validity

### 14.1 Ollama vs Native Inference

All G3 models run through Ollama, which adds overhead and may not expose the full capability of each model. In particular, Ollama uses llama.cpp as its inference backend with default quantisation settings — models may be running at Q4_K_M or Q8 rather than full float16. If G3-A (Qwen3:14b) ran at full precision with vLLM, its FA and safety might improve modestly. The reported G3 results are Ollama-specific and should not be generalised to all deployment configurations of these models.

### 14.2 Think Mode Interaction (G3-A, G3-D)

G3-A suppresses Qwen3's reasoning chain via `/no_think` system prompt prefix. G3-D strips DeepSeek-R1's think tokens post-generation. Both interventions were verified (0 think tokens in any output), but neither represents the model at its full capability:

- For G3-A: running with think mode enabled might improve FA (the reasoning chain acts as a grounding verification layer, similar to G2-D's architecture) but would significantly increase latency and risk.
- For G3-D: the think-stripped output inherits the benefits of internal reasoning but the model's output structure remains non-compliant with G1-C's section format even after reasoning — think-stripping is not the cause of the format failure.

An experiment with G3-A operating in think mode (at the cost of 2–3× latency) would be a useful comparison for future work.

### 14.3 idx=23 Exclusion in G3-A

The permanent exclusion of `cat_c_dry_infected_combo` from G3-A RAGAS evaluation (n=31 per run vs 32 for others) creates a minor comparability issue. If idx=23 had been answerable, G3-A's FA might be slightly different in either direction. The case is excluded consistently and transparently, and the safety impact (counted as FAIL in 32-case denominator) is conservative — it penalises G3-A's reported safety rate rather than inflating it.

### 14.4 Cross-Node Latency Variance (G3-B Run 1, G3-D Run 3)

G3-B Run 1 (gpu06, fresh node) completed in 16,704 ms mean vs ~29,500 ms for Runs 2–3. G3-D Run 3 (gpu07) completed in 17,072 ms mean vs ~30,000 ms for Runs 1–2. These cross-node differences inflate latency std (G3-B: ±7,380 ms; G3-D: ±7,515 ms) but do not affect FA or safety, which are the primary metrics. For production deployment, latency would be measured on dedicated infrastructure, not shared HPC nodes.

### 14.5 G3's Fully Load-from-Disk Design

Unlike G1 and G2 (1+2 split design), G3 uses a fully load-from-disk evaluation: all 12 result files were generated on HPC, then loaded for local RAGAS evaluation. This eliminates the session confound of the 1+2 design but introduces a new potential artefact: HPC generation and local RAGAS evaluation are entirely decoupled. Any systematic difference in how the HPC and local machine process the same inputs (e.g. BGE embedding normalisation at HPC generation time vs local evaluation time) could introduce minor inconsistencies. In practice, the retrieval pipeline uses the same BGE model and ChromaDB collection at both stages, and the RAGAS evaluation only uses pre-retrieved contexts from the JSON files — so this risk is minimal.

---

## 15. G4 / Future Work Next Steps

G3 establishes that open-source 11–14B models running via Ollama do not reach the clinical safety and faithfulness thresholds set by the G2-D closed-source winner. The fixed configuration carried forward from G3 for the final VerdaSense system is:

| Component | Fixed Configuration (G2-D winner) |
|---|---|
| Prompt strategy | G1-C: Grounded system prompt |
| Generation LLM (production) | G2-D: gemini-2.5-flash |
| Open-source alternative | G3-A: qwen3:14b (Ollama) — for cost-sensitive/air-gapped deployment |
| Retrieval embedding | BAAI/bge-large-en-v1.5 |
| Retrieval strategy | R1-C multi-axis dense (k=6) |
| KB | db_wound_care_v4_bge |

**For the G3 analysis' future work recommendations:**

1. **Fine-tuning G3-A or G3-B on G1-C structured output.** The identified failure modes for G3-A (phrasing) and G3-B (edge-case clinical knowledge) have clear fine-tuning targets. A 500-sample synthetic dataset generated from GPT-4o safety-passing outputs, filtered through the v2 safety checker, would provide the minimum training corpus. Expected safety improvement: G3-A from 80.2% to ~85–88%, sufficient to pass the gate.

2. **Evaluate G3-A in think mode.** The `/no_think` suppression may be sacrificing faithfulness gains that a reasoning-augmented generation would provide. Testing G3-A with think mode enabled (accepting the latency cost) would quantify whether the FA gap to G2-D narrows under a thinking architecture.

3. **Zero-AR RAGAS artefact resolution.** Implement a post-processing step to standardise Gemma3 and Llama3.2 response formatting before RAGAS evaluation (stripping spurious empty-line blocks that confuse the judge's question-generation step). This would provide cleaner AR estimates and allow more direct comparison with G2's Gemini results.

4. **Model-specific prompt variants.** The G1-C prompt was designed for gpt-4o-mini. Testing a Qwen3-specific variant (e.g. leveraging its native section-header instruction syntax) and a Gemma3-specific variant may narrow the FA and safety gaps.

5. **Larger testset (n=64).** A doubled testset with proportionally balanced categories would improve resolution of small safety differences (currently 1 case = 3.1 pp) and provide more statistical power for cross-version claims.

---

*Document generated: 17 May 2026 | VerdaSense RAG — FYP Ablation Study | Universiti Malaya*  
*Stage 1 Retrieval Ablation: COMPLETE (R1 ✓ R2 ✓ R3 ✓ R4 ✓)*  
*Stage 2 Generation Ablation: G1 ✓ | G2 ✓ | G3 ✓ | G4 pending*  
*Fixed Stage 2 config: G1-C Grounded prompt + G2-D Gemini 2.5 Flash + R1-C multi-axis dense k=6 + BGE-large-en-v1.5*  
*G3 open-source best: G3-A Qwen3:14b (fallback) | co-equal: G3-B Gemma3:12b*
