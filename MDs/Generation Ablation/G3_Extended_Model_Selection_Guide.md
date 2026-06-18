# VerdaSense RAG — G3 Extended: 2026 Open-Source LLM Model Selection Guide
## Adding G3-D through G3-G via OpenRouter API

**Context:** G3-OR ran A/B/C with FA scores of 0.766 / 0.701 / 0.682 and safety of 82.3% / 81.2% / 64.6%. G3-D (DeepSeek-R1-Distill-Qwen-14b) failed with a 404. This guide selects four additional open-source models (G3-D through G3-G) from the 2026 OpenRouter catalogue that are faster, more capable, and better suited for near-real-time mobile deployment.

**Date:** May 2026  
**Criteria:** Fast latency (mobile), high instruction-following, open-weights/open-license, confirmed available on OpenRouter, comparable or better than G3-A/B/C results

---

## Table of Contents

1. [Why the Current G3-A/B/C Are Not Enough](#1-why-the-current-g3-abc-are-not-enough)
2. [2026 OpenRouter Model Landscape Overview](#2-2026-openrouter-model-landscape-overview)
3. [Candidate Shortlist: Research & Evaluation](#3-candidate-shortlist-research--evaluation)
4. [Selected Models: G3-D through G3-G](#4-selected-models-g3-d-through-g3-g)
5. [Full Pricing Comparison: All 11 Models](#5-full-pricing-comparison-all-11-models)
6. [Per-Query Cost for VerdaSense (3,500 input + 1,800 output tokens)](#6-per-query-cost-for-verdasense-3500-input--1800-output-tokens)
7. [Mobile Latency Expectation](#7-mobile-latency-expectation)
8. [How to Add G3-D through G3-G to the Existing Notebook](#8-how-to-add-g3-d-through-g3-g-to-the-existing-notebook)
9. [New Cell Code: VERSION_CONFIG Extension](#9-new-cell-code-version_config-extension)
10. [New Run Cells: G3-D through G3-G](#10-new-run-cells-g3-d-through-g3-g)
11. [Updated Summary Cell](#11-updated-summary-cell)
12. [Model Availability Verification Script](#12-model-availability-verification-script)
13. [Thinking Mode Handling per New Model](#13-thinking-mode-handling-per-new-model)
14. [Final Recommendation and Priority Order](#14-final-recommendation-and-priority-order)

---

## 1. Why the Current G3-A/B/C Are Not Enough

| Issue | G3-A (Qwen3:14b) | G3-B (Gemma3:12b) | G3-C (Llama3.2:11b) |
|---|---|---|---|
| **FA vs G2-D gap** | −4.9 pp (0.766 vs 0.815) | −11.4 pp | −13.3 pp |
| **Safety vs G2-D gap** | −8.3 pp | −9.4 pp | −26 pp |
| **Latency (mobile)** | 27.8 s (too slow, high variance) | 10.9 s ✅ | 9.6 s ✅ |
| **Latency std** | 9.98 s (unstable) | 917 ms ✅ | 923 ms ✅ |
| **Key problem** | Slow & variable | FA below 0.75 threshold | Safety failure |

The 2026 model frontier has advanced significantly since the original G3 model selection (April–May 2025 models). The latest open-source MoE models offer dramatically better quality at equal or lower active parameter counts, meaning both better reasoning and faster inference. Adding G3-D through G3-G tests whether these newer models close the quality and latency gaps simultaneously.

---

## 2. 2026 OpenRouter Model Landscape Overview

### Key 2026 Developments Relevant to VerdaSense

Between April and May 2026, major labs shipped Kimi K2.6, GLM-5.1, DeepSeek V4 Pro and V4 Flash, Gemma 4, Qwen 3.6, and MiniMax M2.7 — representing a step-change in open-source quality.

In 2026, models from Google (Gemma 4), Meta (Llama 4), Alibaba (Qwen3), and Microsoft (Phi 4) now match or exceed proprietary models for most practical tasks, and there are now a dozen platforms providing access via OpenAI-compatible APIs without managing any server infrastructure.

**The most important development for VerdaSense is the rise of MoE (Mixture-of-Experts) models.** MoE architectures have total parameter counts of 26B–300B+, but only activate 3–15B parameters per forward pass. This means:
- **Quality** comparable to much larger dense models (better knowledge and reasoning)
- **Speed** comparable to much smaller dense models (few active parameters = fast inference)
- **Cost** closer to small models because billing on compute actually used

### Architecture Classes Available on OpenRouter (May 2026)

| Class | Examples | Active Params | Quality | Latency | Mobile? |
|---|---|---|---|---|---|
| Dense 11–14B | Llama3.2, Gemma3:12b, Qwen3:14b | 11–14B | Medium | Medium | ⚠️ |
| MoE ~3B active | Qwen3.5/3.6-35B-A3B, Gemma4-26B-A4B | 3–4B | High | Fast | ✅ |
| MoE ~13B active | DeepSeek V4 Flash (284B/13B active) | 13B | Very High | Medium | ✅ |
| Thinking (hybrid) | DeepSeek V4 Pro/Flash non-think mode | varies | Very High | Medium–Slow | ⚠️ |

**The MoE ~3B active class is the sweet spot for mobile RAG:** same or better quality than the dense 12–14B models you already tested, at 3× fewer active parameters → faster inference → lower latency → better mobile UX.

---

## 3. Candidate Shortlist: Research & Evaluation

### Models Researched and Evaluated

| Model | Family | Total/Active Params | License | OpenRouter ID | Available? | Mobile? | Verdict |
|---|---|---|---|---|---|---|---|
| **Gemma 4 26B A4B** | Google Gemma4 | 26B / 4B active (MoE) | Apache 2.0 | `google/gemma-4-26b-a4b-it` | ✅ | ✅ | **SELECTED G3-D** |
| **Qwen3.6-35B-A3B** | Alibaba Qwen3.6 | 35B / 3B active (MoE) | Apache 2.0 | `qwen/qwen3.6-35b-a3b` | ✅ | ✅ | **SELECTED G3-E** |
| **DeepSeek V4 Flash** | DeepSeek V4 | 284B / 13B active (MoE) | MIT | `deepseek/deepseek-v4-flash` | ✅ | ✅ | **SELECTED G3-F** |
| **Qwen3.5-35B-A3B** | Alibaba Qwen3.5 | 35B / 3B active (MoE) | Apache 2.0 | `qwen/qwen3.5-35b-a3b` | ✅ | ✅ | **SELECTED G3-G** |
| Gemma 4 31B Dense | Google Gemma4 | 31B / 31B (dense) | Apache 2.0 | `google/gemma-4-31b-it` | ✅ | ⚠️ | Slower than A4B variant — skip |
| Qwen3.6-Plus | Alibaba Qwen3.6 | Proprietary (closed-weight) | Proprietary | `qwen/qwen3.6-plus` | ✅ | ✅ | Closed-weight — excluded (G3 is open-source ablation) |
| Kimi K2.6 | Moonshot AI | 1T / 32B active (MoE) | MIT | Not on OpenRouter | ❌ | — | Not available on OpenRouter |
| MiniMax M2.7 | MiniMax | Large MoE | Apache 2.0 | Not confirmed on OR | ❌ | — | Not confirmed on OpenRouter |
| GLM-5.1 | Zhipu AI | 754B MoE | MIT | Not confirmed on OR | ❌ | — | Not on OpenRouter at time of writing |
| MiMo-V2-Pro | Xiaomi | 1T MoE | Apache 2.0 | Not on OR (self-hosted only) | ❌ | — | No API access |
| DeepSeek V4 Pro | DeepSeek V4 | 865B / 37B active (MoE) | MIT | `deepseek/deepseek-v4-pro` | ✅ | ⚠️ | Heavy model — latency too high for mobile |
| GPT-OSS | OpenAI | 120B (rumoured) | Closed | On OpenRouter | ✅ | — | Not open-source — excluded from G3 |
| Llama 4 | Meta | Various MoE sizes | Llama 4 License | On OpenRouter | ✅ | ⚠️ | License has 700M MAU cap — excluded |
| Mistral Small 3.5 | Mistral | 24B dense | Apache 2.0 | `mistralai/mistral-small-3.5` | ✅ | ⚠️ | Larger than MoE alternatives; skip for now |

### Why Kimi, MiniMax, GLM are Excluded

While Kimi K2.6, MiniMax M2.7, and GLM-5.1 are available via Vercel AI Gateway and OpenRouter/Helicone for observability, direct API access varies by region, and endpoint stability for these newer Chinese models on OpenRouter is less reliable than the established Google, Alibaba, and DeepSeek endpoints. For a research project requiring reproducible 3-run experiments, model endpoint stability is critical. These models can be revisited once their OpenRouter endpoints stabilise.

### Why Llama 4 is Excluded

Meta's Llama 4 license includes a 700 million monthly active users threshold above which commercial use requires a separate agreement. For an FYP research project this is not a practical concern, but for a *mobile health application* that could scale, this is a genuine deployment risk. The open-source G3 ablation should use models with clean commercial licenses (Apache 2.0 or MIT). Llama 3.x (used in G3-C) has a similar cap at 700M MAU but is more established — Llama 4's newer and less widely reviewed license makes it a lower-priority choice.

---

## 4. Selected Models: G3-D through G3-G

### G3-D: Gemma 4 26B A4B — `google/gemma-4-26b-a4b-it`

| Property | Value |
|---|---|
| **OpenRouter ID** | `google/gemma-4-26b-a4b-it` |
| **Released** | April 2, 2026 |
| **Architecture** | MoE — 26B total / ~4B active per token |
| **License** | Apache 2.0 ✅ |
| **Context** | 262K tokens |
| **Thinking mode** | No native thinking mode |
| **Input price (OpenRouter)** | $0.060 / M tokens |
| **Output price (OpenRouter)** | $0.330 / M tokens |
| **Latency expectation** | ~3–7 s (4B active params = very fast) |

**Why G3-D:** Gemma 4 is Google's first fully open-source (Apache 2.0) model family, released April 2, 2026, and is the direct successor to Gemma 3 which was used in your G3-B experiment. The key upgrade is the MoE architecture: where Gemma3:12b activated all 12B parameters, Gemma4-26B-A4B activates only ~4B per forward pass. This means faster inference, lower cost, and — critically — the full 26B parameters available as a knowledge base, making the model significantly more capable on instruction-following and grounding tasks than its Gemma 3 predecessor. This is a direct upgrade path from G3-B and should substantially improve on G3-B's FA of 0.7009.

**Why not Gemma4 31B Dense:** The 31B dense model loads all 31 billion parameters into VRAM on every forward pass, making it noticeably slower than the MoE variant. For mobile latency, the 26B-A4B MoE is the correct choice.

---

### G3-E: Qwen3.6-35B-A3B — `qwen/qwen3.6-35b-a3b`

| Property | Value |
|---|---|
| **OpenRouter ID** | `qwen/qwen3.6-35b-a3b` |
| **Released** | April 14–27, 2026 |
| **Architecture** | Hybrid MoE (Gated DeltaNet + Gated Attention) — 35B total / 3B active |
| **License** | Apache 2.0 ✅ |
| **Context** | 262K native (1M via YaRN) |
| **Thinking mode** | Integrated thinking mode (controllable — use `/no_think`) |
| **Input price (OpenRouter)** | $0.150 / M tokens |
| **Output price (OpenRouter)** | $1.000 / M tokens |
| **Latency expectation** | ~3–8 s (3B active params = extremely fast) |

**Why G3-E:** Qwen3.6-35B-A3B is an open-weight multimodal model with 35 billion total parameters but only 3 billion active parameters per token, using a hybrid sparse mixture-of-experts architecture combining Gated DeltaNet linear attention with standard gated attention layers, enabling efficient inference at a fraction of the compute cost. With only 3B active parameters, this model is faster than Gemma4-A4B (4B active) while having a larger knowledge base (35B total). It is the direct successor to Qwen3.5-35B-A3B (G3-G), representing the state-of-the-art in the Qwen open-source line as of May 2026.

Qwen 3.6 delivers major gains in agentic coding, front-end development, and overall reasoning compared to the 3.5 series. For wound care RAG, the structured output discipline (function calling, structured output support) and improved reasoning are directly relevant to following the G1-C prompt's 8-section template.

**Thinking mode:** Disable via `/no_think` prefix in system prompt (same as G3-A Qwen3:14b). This is the same Qwen family — the `/no_think` directive works identically.

---

### G3-F: DeepSeek V4 Flash — `deepseek/deepseek-v4-flash`

| Property | Value |
|---|---|
| **OpenRouter ID** | `deepseek/deepseek-v4-flash` |
| **Released** | April 24, 2026 |
| **Architecture** | Hybrid Attention MoE — 284B total / ~13B active |
| **License** | MIT ✅ |
| **Context** | 1M tokens |
| **Thinking mode** | Both Thinking and Non-Thinking modes supported |
| **Input price (OpenRouter)** | $0.100 / M tokens |
| **Output price (OpenRouter)** | $0.200 / M tokens |
| **Latency expectation** | ~5–12 s (13B active params — faster than dense 14B) |

**Why G3-F:** DeepSeek-V4-Flash has 284B total parameters with only 13B active parameters, supporting a 1M-token context window, and supports both Thinking and Non-Thinking modes. This model replaces the originally planned G3-D (DeepSeek-R1-Distill-Qwen-14b) as the DeepSeek representative in G3. V4 Flash is the production-ready successor — it is DeepSeek's current flagship fast model, not a distill of an older generation.

DeepSeek V4 Flash is priced at $0.14/M input and $0.28/M output (via DeepSeek API directly); OpenRouter lists $0.10/$0.20 per 1M tokens, making it one of the most cost-efficient models in its class. This makes V4 Flash the cheapest non-free open-source model available on OpenRouter for this quality tier.

**Thinking mode:** Run in **Non-Thinking mode** for this experiment. The thinking mode is disabled by not including a `think` flag in the API call. Unlike Qwen3, DeepSeek V4 Flash defaults to non-thinking mode without any `/no_think` injection needed. Verify this by checking whether `<think>` blocks appear in outputs — the `strip_thinking_tokens()` function provides a safety net regardless.

**Important migration note:** DeepSeek has announced that the legacy `deepseek-chat` and `deepseek-reasoner` endpoints will be fully retired after July 24, 2026. Always use `deepseek-v4-flash` or `deepseek-v4-pro` as the model IDs going forward.

---

### G3-G: Qwen3.5-35B-A3B — `qwen/qwen3.5-35b-a3b`

| Property | Value |
|---|---|
| **OpenRouter ID** | `qwen/qwen3.5-35b-a3b` |
| **Released** | February 16, 2026 |
| **Architecture** | MoE — 35B total / 3B active |
| **License** | Apache 2.0 ✅ |
| **Context** | 262K tokens |
| **Thinking mode** | Hybrid (thinking + non-thinking, controllable) |
| **Input price (OpenRouter)** | $0.140 / M tokens |
| **Output price (OpenRouter)** | $1.000 / M tokens |
| **Latency expectation** | ~3–7 s (3B active params = extremely fast) |

**Why G3-G:** Qwen3.5-35B-A3B is the immediate predecessor to G3-E (Qwen3.6-35B-A3B), both using a near-identical MoE architecture. Including both in the ablation creates a **within-family version comparison**: G3-E vs G3-G isolates the effect of the Qwen3.5 → Qwen3.6 upgrade under identical experimental conditions. This is a clean intra-family ablation that directly demonstrates whether the newer model generation is worth the slight price premium.

Qwen3.5 was released as open-weights on February 16, 2026. It is already a well-established model with known performance characteristics, reducing the risk of endpoint unavailability or API instability compared to the newer G3-E.

**Thinking mode:** Disable via `/no_think` prefix, same as G3-E and G3-A.

---

## 5. Full Pricing Comparison: All 11 Models

### G2 Closed-Source Models

| Version | Model | Input $/M | Output $/M | License |
|---|---|---|---|---|
| G2-A | gpt-4o-mini | $0.150 | $0.600 | Proprietary |
| G2-B | gpt-4o | $2.500 | $10.000 | Proprietary |
| G2-C | gemini-2.5-flash-lite | $0.100 | $0.400 | Proprietary |
| G2-D | gemini-2.5-flash | $0.300 | $2.500 | Proprietary |

### G3 Open-Source Models (Original A/B/C + New D/E/F/G)

| Version | Model | Input $/M | Output $/M | License | Active Params |
|---|---|---|---|---|---|
| G3-A | qwen/qwen3-14b | $0.100 | $0.240 | Apache 2.0 | 14B (dense) |
| G3-B | google/gemma-3-12b-it | $0.040 | $0.130 | Apache 2.0 | 12B (dense) |
| G3-C | meta-llama/llama-3.2-11b-vision-instruct | $0.245 | $0.245 | Llama 3.2 | 11B (dense) |
| G3-D *(new)* | google/gemma-4-26b-a4b-it | $0.060 | $0.330 | Apache 2.0 | ~4B (MoE) |
| G3-E *(new)* | qwen/qwen3.6-35b-a3b | $0.150 | $1.000 | Apache 2.0 | ~3B (MoE) |
| G3-F *(new)* | deepseek/deepseek-v4-flash | $0.100 | $0.200 | MIT | ~13B (MoE) |
| G3-G *(new)* | qwen/qwen3.5-35b-a3b | $0.140 | $1.000 | Apache 2.0 | ~3B (MoE) |

---

## 6. Per-Query Cost for VerdaSense (3,500 input + 1,800 output tokens)

| Version | Model | Per Query | Per Patient (10 queries) | vs G2-D ($0.0056) |
|---|---|---|---|---|
| G3-B | gemma-3-12b-it | **$0.000374** | $0.004 | 15× cheaper |
| **G3-D** | **gemma-4-26b-a4b-it** | **$0.000810** | **$0.008** | **7× cheaper** |
| G3-F | deepseek-v4-flash | $0.000710 | $0.007 | 8× cheaper |
| G3-A | qwen3-14b | $0.000782 | $0.008 | 7× cheaper |
| G3-G | qwen3.5-35b-a3b | $0.002290 | $0.023 | 2.4× cheaper |
| G3-E | qwen3.6-35b-a3b | $0.002325 | $0.023 | 2.4× cheaper |
| G3-C | llama-3.2-11b | $0.001299 | $0.013 | 4.3× cheaper |
| G2-C | gemini-2.5-flash-lite | $0.001070 | $0.011 | 5.2× cheaper |
| G2-A | gpt-4o-mini | $0.001605 | $0.016 | 3.5× cheaper |
| G2-D | gemini-2.5-flash | $0.005550 | $0.056 | baseline |
| G2-B | gpt-4o | $0.026750 | $0.268 | 4.8× more expensive |

**G3-F (DeepSeek V4 Flash) is the most cost-efficient** at $0.000710/query — only marginally more expensive than the cheapest model (G3-B), but with dramatically higher expected quality due to the larger knowledge base. G3-D (Gemma4 MoE) is similarly cheap at $0.000810/query.

**G3-E/G (Qwen3.6/3.5 A3B)** are more expensive on output ($1.00/M) but the output token count (1,800 tokens) means the total per-query cost remains manageable at $0.0023.

---

## 7. Mobile Latency Expectation

Expected API generation latencies via OpenRouter (based on active parameter count and 2026 infrastructure benchmarks):

| Version | Model | Active Params | Expected API Latency | Total (+ 300ms retrieval) | Mobile UX |
|---|---|---|---|---|---|
| G3-B | gemma-3-12b-it | 12B dense | ~10–12 s | ~10–13 s | ✅ Good |
| G3-C | llama-3.2-11b | 11B dense | ~9–11 s | ~9–12 s | ✅ Good |
| **G3-D** | gemma-4-26b-a4b-it | **~4B MoE** | **~3–6 s** | **~3–7 s** | **✅ Excellent** |
| **G3-E** | qwen3.6-35b-a3b | **~3B MoE** | **~3–6 s** | **~3–7 s** | **✅ Excellent** |
| **G3-F** | deepseek-v4-flash | **~13B MoE** | **~5–10 s** | **~5–11 s** | **✅ Good** |
| **G3-G** | qwen3.5-35b-a3b | **~3B MoE** | **~3–6 s** | **~3–7 s** | **✅ Excellent** |
| G3-A | qwen3-14b | 14B dense | ~20–40 s | ~21–40 s | ⚠️ Marginal |
| G2-D | gemini-2.5-flash | — | ~18–22 s | ~18–23 s | ⚠️ Marginal |

**G3-D, G3-E, and G3-G are projected to be 3–5× faster than G3-A** due to the MoE active parameter reduction. If G3-D/E/G achieve both better FA (>0.75) and safety (>86.7%) with 3–6 s latency, they would represent the ideal production configuration for VerdaSense: better quality than G3-A, 4–8× faster, at equal or lower cost.

---

## 8. How to Add G3-D through G3-G to the Existing Notebook

### Strategy: Extend the Existing Notebook, Load A/B/C from Disk

The cleanest approach is to:
1. **Keep** the existing `ragas_ablation_G3_opensource_llm_openrouter.ipynb` as-is (do not re-run G3-A/B/C)
2. **Add** new cells at the end of the notebook (or in a new continuation notebook) for G3-D through G3-G
3. **Load** G3-A/B/C results from disk for the summary table using `load_agg_from_ragas_json()`

This is exactly the same pattern as loading the G3-D results alongside G3-A/B/C in the summary — extended to 7 versions.

### Files to Update

```
RAGAS_EVAL/
└── G3_OpenSource_LLM/
    └── evaluation/
        ├── ragas_ablation_G3_opensource_llm_openrouter.ipynb    ← original (A/B/C done)
        ├── ragas_ablation_G3_extended_DEFG.ipynb                ← NEW notebook for D/E/F/G
        └── results/
            ├── G3_G3A_*.json / *.csv                            ← already written
            ├── G3_G3B_*.json / *.csv                            ← already written
            ├── G3_G3C_*.json / *.csv                            ← already written
            ├── G3_G3D_*.json / *.csv                            ← will be written (Gemma4)
            ├── G3_G3E_*.json / *.csv                            ← will be written (Qwen3.6)
            ├── G3_G3F_*.json / *.csv                            ← will be written (DeepSeek V4)
            ├── G3_G3G_*.json / *.csv                            ← will be written (Qwen3.5)
            └── G3_summary.json                                  ← will be updated (7 versions)
```

---

## 9. New Cell Code: VERSION_CONFIG Extension

Add this at **Cell 1b** (after the original Cell 1 config, before Cell 2):

```python
# ── Cell 1b: Extended VERSION_CONFIG for G3-D through G3-G ─────────────────
# (Add to existing notebook or put in continuation notebook)

VERSIONS_EXTENDED = ["G3-D", "G3-E", "G3-F", "G3-G"]
VERSIONS_ALL      = ["G3-A", "G3-B", "G3-C", "G3-D", "G3-E", "G3-F", "G3-G"]

VERSION_CONFIG_EXTENDED = {
    "G3-D": {
        "label":       "Gemma 4 26B-A4B (Google MoE) via OpenRouter",
        "description": (
            "G1-C grounded system prompt with google/gemma-4-26b-a4b-it via OpenRouter. "
            "26B total / ~4B active parameters MoE — successor to G3-B (Gemma3:12b). "
            "Apache 2.0 license. No thinking mode. Tests whether the 2026 Gemma4 MoE "
            "fixes the FA < 0.75 issue seen in G3-B at a fraction of the latency."
        ),
        "provider":        "openrouter",
        "model":           "google/gemma-4-26b-a4b-it",
        "thinking_model":  False,
        "no_think_prefix": False,
    },
    "G3-E": {
        "label":       "Qwen3.6-35B-A3B (Alibaba MoE) via OpenRouter",
        "description": (
            "G1-C grounded system prompt with qwen/qwen3.6-35b-a3b via OpenRouter. "
            "35B total / ~3B active parameters MoE — latest open-weight Qwen generation (Apr 2026). "
            "Apache 2.0 license. Thinking disabled via /no_think prefix. "
            "Successor to G3-A (Qwen3:14b dense) — tests whether the MoE upgrade improves FA and latency."
        ),
        "provider":        "openrouter",
        "model":           "qwen/qwen3.6-35b-a3b",
        "thinking_model":  True,    # has thinking mode — disable via /no_think
        "no_think_prefix": True,    # /no_think prefix injected in system prompt
    },
    "G3-F": {
        "label":       "DeepSeek V4 Flash (DeepSeek MoE) via OpenRouter",
        "description": (
            "G1-C grounded system prompt with deepseek/deepseek-v4-flash via OpenRouter. "
            "284B total / ~13B active parameters MoE — DeepSeek's 2026 fast model (Apr 24, 2026). "
            "MIT license. Defaults to non-thinking mode (no /no_think needed). "
            "Replaces the failed G3-D (deepseek-r1-distill-qwen-14b 404). "
            "1M token context window. Tests the cost-performance frontier of 2026 open MoE."
        ),
        "provider":        "openrouter",
        "model":           "deepseek/deepseek-v4-flash",
        "thinking_model":  True,    # has thinking mode but defaults to non-thinking
        "no_think_prefix": False,   # does NOT use /no_think — non-think is the default
        # Note: strip_thinking_tokens() still applied as safety net
    },
    "G3-G": {
        "label":       "Qwen3.5-35B-A3B (Alibaba MoE) via OpenRouter",
        "description": (
            "G1-C grounded system prompt with qwen/qwen3.5-35b-a3b via OpenRouter. "
            "35B total / ~3B active parameters MoE — February 2026 open-weight Qwen release. "
            "Apache 2.0 license. Thinking disabled via /no_think prefix. "
            "Direct predecessor to G3-E — enables intra-family Qwen3.5 vs Qwen3.6 comparison."
        ),
        "provider":        "openrouter",
        "model":           "qwen/qwen3.5-35b-a3b",
        "thinking_model":  True,
        "no_think_prefix": True,
    },
}

# Merge into the master VERSION_CONFIG
VERSION_CONFIG.update(VERSION_CONFIG_EXTENDED)

print("Extended VERSION_CONFIG loaded — G3-D through G3-G added.")
for v in VERSIONS_EXTENDED:
    cfg = VERSION_CONFIG[v]
    print(f"  {v}: {cfg['model']}  | thinking={cfg['thinking_model']} | no_think={cfg['no_think_prefix']}")
```

---

## 10. New Run Cells: G3-D through G3-G

### Cell 17b — Verify All New Models Are Live Before Running

```python
# ── Cell 17b: Verify new models are available on OpenRouter ─────────────────
import requests

def verify_openrouter_model(model_id: str) -> bool:
    """Check if a model ID is available on OpenRouter."""
    try:
        resp = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
            timeout=10
        )
        models = {m["id"] for m in resp.json().get("data", [])}
        return model_id in models
    except Exception as e:
        print(f"  Verification error for {model_id}: {e}")
        return False

print("Verifying G3-D through G3-G model availability on OpenRouter...")
for v in VERSIONS_EXTENDED:
    model_id = VERSION_CONFIG[v]["model"]
    available = verify_openrouter_model(model_id)
    status = "✅ Available" if available else "❌ NOT FOUND — do not run"
    print(f"  {v} ({model_id}): {status}")

print("\nNote: If any model shows ❌, update its model ID in VERSION_CONFIG_EXTENDED before proceeding.")
print("Alternative IDs to try if a model is missing:")
print("  G3-D fallback: google/gemma-4-31b-it (dense variant)")
print("  G3-E fallback: qwen/qwen3.5-35b-a3b (same as G3-G)")
print("  G3-F fallback: deepseek/deepseek-v4-pro (heavier, slower)")
print("  G3-G fallback: qwen/qwen3-32b (Qwen3 dense 32B)")
```

### Cell 18b — Run G3-D: Gemma 4 26B-A4B

```python
# ── Cell 18b: Run G3-D (Gemma 4 26B-A4B) ───────────────────────────────────
print("▶" * 65)
print("  Running G3-D: Gemma 4 26B-A4B (Google MoE) via OpenRouter  [3 runs]")
print("▶" * 65)

run_results_G3D = []
llm_D = build_generation_llm("G3-D")

for run_num in [1, 2, 3]:
    records_r, ragas_r = run_single_run("G3-D", run_num, llm_D)
    run_results_G3D.append((records_r, ragas_r))

agg_G3D = aggregate_runs(run_results_G3D)
write_multirun_files("G3-D", run_results_G3D, agg_G3D)

print(f"\n── G3-D Aggregated Summary (n_runs=3) ──")
print(f"   FA            : {agg_G3D['faithfulness']:.4f} ± {agg_G3D['std_faithfulness']:.4f}")
print(f"   AR            : {agg_G3D['answer_relevancy']:.4f} ± {agg_G3D['std_answer_relevancy']:.4f}")
print(f"   Safety Pass   : {agg_G3D['mean_safety_pct']:.1f}% ± {agg_G3D['std_safety_pct']:.1f}%")
print(f"   Gen Lat       : {agg_G3D['mean_gen_latency_ms']:.1f} ± {agg_G3D['std_gen_latency_ms']:.1f} ms")
print(f"   Per-run FA    : {agg_G3D['per_run_fa']}")
print(f"   Per-run Safety: {agg_G3D['per_run_safety']}")
```

### Cell 19b — Run G3-E: Qwen3.6-35B-A3B

```python
# ── Cell 19b: Run G3-E (Qwen3.6-35B-A3B) ──────────────────────────────────
print("▶" * 65)
print("  Running G3-E: Qwen3.6-35B-A3B (Alibaba MoE) via OpenRouter  [3 runs]")
print("▶" * 65)

run_results_G3E = []
llm_E = build_generation_llm("G3-E")

for run_num in [1, 2, 3]:
    records_r, ragas_r = run_single_run("G3-E", run_num, llm_E)
    run_results_G3E.append((records_r, ragas_r))

agg_G3E = aggregate_runs(run_results_G3E)
write_multirun_files("G3-E", run_results_G3E, agg_G3E)

print(f"\n── G3-E Aggregated Summary (n_runs=3) ──")
print(f"   FA            : {agg_G3E['faithfulness']:.4f} ± {agg_G3E['std_faithfulness']:.4f}")
print(f"   AR            : {agg_G3E['answer_relevancy']:.4f} ± {agg_G3E['std_answer_relevancy']:.4f}")
print(f"   Safety Pass   : {agg_G3E['mean_safety_pct']:.1f}% ± {agg_G3E['std_safety_pct']:.1f}%")
print(f"   Gen Lat       : {agg_G3E['mean_gen_latency_ms']:.1f} ± {agg_G3E['std_gen_latency_ms']:.1f} ms")
print(f"   Per-run FA    : {agg_G3E['per_run_fa']}")
print(f"   Per-run Safety: {agg_G3E['per_run_safety']}")
```

### Cell 20b — Run G3-F: DeepSeek V4 Flash

```python
# ── Cell 20b: Run G3-F (DeepSeek V4 Flash) ─────────────────────────────────
print("▶" * 65)
print("  Running G3-F: DeepSeek V4 Flash (DeepSeek MoE) via OpenRouter  [3 runs]")
print("▶" * 65)

run_results_G3F = []
llm_F = build_generation_llm("G3-F")

for run_num in [1, 2, 3]:
    records_r, ragas_r = run_single_run("G3-F", run_num, llm_F)
    run_results_G3F.append((records_r, ragas_r))

agg_G3F = aggregate_runs(run_results_G3F)
write_multirun_files("G3-F", run_results_G3F, agg_G3F)

print(f"\n── G3-F Aggregated Summary (n_runs=3) ──")
print(f"   FA            : {agg_G3F['faithfulness']:.4f} ± {agg_G3F['std_faithfulness']:.4f}")
print(f"   AR            : {agg_G3F['answer_relevancy']:.4f} ± {agg_G3F['std_answer_relevancy']:.4f}")
print(f"   Safety Pass   : {agg_G3F['mean_safety_pct']:.1f}% ± {agg_G3F['std_safety_pct']:.1f}%")
print(f"   Gen Lat       : {agg_G3F['mean_gen_latency_ms']:.1f} ± {agg_G3F['std_gen_latency_ms']:.1f} ms")
print(f"   Per-run FA    : {agg_G3F['per_run_fa']}")
print(f"   Per-run Safety: {agg_G3F['per_run_safety']}")
```

### Cell 21b — Run G3-G: Qwen3.5-35B-A3B

```python
# ── Cell 21b: Run G3-G (Qwen3.5-35B-A3B) ──────────────────────────────────
print("▶" * 65)
print("  Running G3-G: Qwen3.5-35B-A3B (Alibaba MoE) via OpenRouter  [3 runs]")
print("▶" * 65)

run_results_G3G = []
llm_G = build_generation_llm("G3-G")

for run_num in [1, 2, 3]:
    records_r, ragas_r = run_single_run("G3-G", run_num, llm_G)
    run_results_G3G.append((records_r, ragas_r))

agg_G3G = aggregate_runs(run_results_G3G)
write_multirun_files("G3-G", run_results_G3G, agg_G3G)

print(f"\n── G3-G Aggregated Summary (n_runs=3) ──")
print(f"   FA            : {agg_G3G['faithfulness']:.4f} ± {agg_G3G['std_faithfulness']:.4f}")
print(f"   AR            : {agg_G3G['answer_relevancy']:.4f} ± {agg_G3G['std_answer_relevancy']:.4f}")
print(f"   Safety Pass   : {agg_G3G['mean_safety_pct']:.1f}% ± {agg_G3G['std_safety_pct']:.1f}%")
print(f"   Gen Lat       : {agg_G3G['mean_gen_latency_ms']:.1f} ± {agg_G3G['std_gen_latency_ms']:.1f} ms")
print(f"   Per-run FA    : {agg_G3G['per_run_fa']}")
print(f"   Per-run Safety: {agg_G3G['per_run_safety']}")
```

---

## 11. Updated Summary Cell

Add this as Cell 22b — it loads A/B/C from disk and combines with fresh D/E/F/G:

```python
# ── Cell 22b: Updated G3 Full Summary (all 7 versions) ─────────────────────

def load_agg_from_ragas_json(version: str) -> dict:
    """Reconstruct aggregate dict from saved ragas JSON (for A/B/C disk load)."""
    v_short    = version.replace("-", "")
    ragas_path = RESULTS_DIR / f"G3_{v_short}_ragas.json"
    with open(ragas_path, encoding="utf-8") as f:
        rj = json.load(f)
    return {
        "faithfulness":        rj["faithfulness"],
        "std_faithfulness":    rj["std_faithfulness"],
        "answer_relevancy":    rj["answer_relevancy"],
        "std_answer_relevancy":rj["std_answer_relevancy"],
        "mean_safety_pct":     rj["mean_safety_pct"],
        "std_safety_pct":      rj["std_safety_pct"],
        "mean_gen_latency_ms": rj["mean_gen_latency_ms"],
        "std_gen_latency_ms":  rj["std_gen_latency_ms"],
        "mean_tot_latency_ms": rj["mean_tot_latency_ms"],
        "std_tot_latency_ms":  rj["std_tot_latency_ms"],
        "mean_ret_latency_ms": rj["mean_ret_latency_ms"],
        "per_run_fa":          rj["all_fa_vals"],
        "per_run_ar":          rj["all_ar_vals"],
        "per_run_safety":      rj["all_safety_vals"],
        "per_run_gen_lat":     [p["gen_latency_ms"] for p in rj["per_run"]],
        "per_run_tot_lat":     [p["tot_latency_ms"] for p in rj["per_run"]],
        "per_sample_fa":       rj["per_sample_fa"],
        "per_sample_ar":       rj["per_sample_ar"],
    }

# Load original A/B/C from disk — do NOT re-run them
agg_G3A = load_agg_from_ragas_json("G3-A")
agg_G3B = load_agg_from_ragas_json("G3-B")
agg_G3C = load_agg_from_ragas_json("G3-C")
# agg_G3D/E/F/G freshly computed from Cells 18b–21b above

all_agg_full = {
    "G3-A": agg_G3A, "G3-B": agg_G3B, "G3-C": agg_G3C,
    "G3-D": agg_G3D, "G3-E": agg_G3E, "G3-F": agg_G3F, "G3-G": agg_G3G,
}

# G2-D reference baseline
G2D_REF = {"fa": 0.8147, "ar": 0.677, "safety": 90.6, "gen_lat": 17961}

print("=" * 140)
print("  G3 OPEN-SOURCE LLM COMPARISON — FULL RESULTS (7 versions, via OpenRouter API)")
print(f"  (n_runs=3 each, 32 cases/run, G1-C prompt fixed, BGE embedding, mean ± std)")
print(f"  [G2-D reference: FA={G2D_REF['fa']:.4f} | Safety={G2D_REF['safety']:.1f}% | GenLat={G2D_REF['gen_lat']:.0f}ms]")
print("=" * 140)

print(f"  {'Ver':<6} {'Model':<42} {'FA ± std':<20} {'AR ± std':<20} {'Safety%±std':<16} {'GenLat ± std':<18} {'Qualified?'}")
print("-" * 140)

for v in VERSIONS_ALL:
    a   = all_agg_full[v]
    cfg = VERSION_CONFIG[v]
    fa_ok   = "✅" if a["faithfulness"] >= 0.75 else "❌"
    safe_ok = "✅" if a["mean_safety_pct"] >= 86.7 else "❌"
    lat_ok  = "✅" if a["mean_gen_latency_ms"] < 15000 else "⚠️"
    qualifies = "✅ Yes" if a["faithfulness"] >= 0.75 and a["mean_safety_pct"] >= 86.7 else "❌ No"
    model_short = cfg["model"].split("/")[-1][:40]
    print(
        f"  {v:<6} {model_short:<42} "
        f"{a['faithfulness']:.4f}±{a['std_faithfulness']:.4f}     "
        f"{a['answer_relevancy']:.4f}±{a['std_answer_relevancy']:.4f}     "
        f"{a['mean_safety_pct']:.1f}%±{a['std_safety_pct']:.1f}pp  "
        f"{a['mean_gen_latency_ms']:.0f}±{a['std_gen_latency_ms']:.0f}ms   "
        f"{fa_ok}FA {safe_ok}Safe {lat_ok}Lat"
    )

print("=" * 140)

# Winner selection across all 7
FA_FLOOR   = 0.75
SAFE_FLOOR = max(all_agg_full[v]["mean_safety_pct"] for v in VERSIONS_ALL) - 5.0

candidates = [
    v for v in VERSIONS_ALL
    if all_agg_full[v]["mean_safety_pct"] >= SAFE_FLOOR
    and all_agg_full[v]["faithfulness"]   >= FA_FLOOR
]

if candidates:
    best_v7 = max(candidates, key=lambda v: all_agg_full[v]["faithfulness"])
    print(f"\n  G3 WINNER (all 7 versions): {best_v7} — {VERSION_CONFIG[best_v7]['model']}")
    a = all_agg_full[best_v7]
    print(f"  FA={a['faithfulness']:.4f}±{a['std_faithfulness']:.4f}  AR={a['answer_relevancy']:.4f}  Safety={a['mean_safety_pct']:.1f}%  GenLat={a['mean_gen_latency_ms']:.0f}ms")
else:
    best_v7 = max(VERSIONS_ALL, key=lambda v: all_agg_full[v]["mean_safety_pct"])
    print(f"\n  No version passes all gates — best safety fallback: {best_v7}")

# Save updated summary
summary_extended = {
    "experiment": "G3-extended",
    "experiment_label": "G3 Open-Source LLM Comparison — Full 7-Version (A–G via OpenRouter API)",
    "timestamp": datetime.datetime.now().isoformat(),
    "n_versions": 7,
    "fixed_config": {
        "prompt_strategy": "G1-C (Grounded system prompt)",
        "retrieval_embedding": "BAAI/bge-large-en-v1.5",
        "retrieval_db": "db_wound_care_v4_bge",
        "retrieval_strategy": "R1-C multi-axis dense (k=6)",
        "ragas_llm_judge": "gpt-4o-mini",
        "ragas_embed_judge": "text-embedding-3-small",
        "generation_backend": "OpenRouter API (https://openrouter.ai/api/v1)",
    },
    "g2d_reference": G2D_REF,
    "best_version": best_v7,
    "best_model":   VERSION_CONFIG[best_v7]["model"],
    "versions": {
        v: {
            "faithfulness":         all_agg_full[v]["faithfulness"],
            "std_faithfulness":     all_agg_full[v]["std_faithfulness"],
            "answer_relevancy":     all_agg_full[v]["answer_relevancy"],
            "std_answer_relevancy": all_agg_full[v]["std_answer_relevancy"],
            "safety_pass_rate_pct": all_agg_full[v]["mean_safety_pct"],
            "std_safety_pass_rate_pct": all_agg_full[v]["std_safety_pct"],
            "generation_latency_ms":    all_agg_full[v]["mean_gen_latency_ms"],
            "std_generation_latency_ms":all_agg_full[v]["std_gen_latency_ms"],
            "model": VERSION_CONFIG[v]["model"],
            "label": VERSION_CONFIG[v]["label"],
            "per_run_fa":     all_agg_full[v]["per_run_fa"],
            "per_run_ar":     all_agg_full[v]["per_run_ar"],
            "per_run_safety": all_agg_full[v]["per_run_safety"],
        }
        for v in VERSIONS_ALL
    },
}

summary_ext_path = RESULTS_DIR / "G3_summary_extended.json"
with open(summary_ext_path, "w", encoding="utf-8") as f:
    json.dump(summary_extended, f, indent=2, ensure_ascii=False)
print(f"\n  Extended summary saved → {summary_ext_path.name}")
```

---

## 12. Model Availability Verification Script

Run this standalone script before starting the notebook to confirm all 4 new models are live:

```python
# verify_g3_models.py — run before starting the extended notebook
import os, requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["OPENROUTER_API_KEY"]

NEW_MODELS = {
    "G3-D": "google/gemma-4-26b-a4b-it",
    "G3-E": "qwen/qwen3.6-35b-a3b",
    "G3-F": "deepseek/deepseek-v4-flash",
    "G3-G": "qwen/qwen3.5-35b-a3b",
}

resp = requests.get(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {API_KEY}"},
    timeout=15
)
available_ids = {m["id"] for m in resp.json().get("data", [])}

print("G3 Extended Model Availability Check")
print("=" * 60)
all_ok = True
for version, model_id in NEW_MODELS.items():
    ok = model_id in available_ids
    if not ok:
        all_ok = False
    print(f"  {version}: {model_id}")
    print(f"         → {'✅ AVAILABLE' if ok else '❌ NOT FOUND — update model ID'}")

print("=" * 60)
if all_ok:
    print("✅ All 4 new models confirmed available. Safe to run extended notebook.")
else:
    print("⚠️  Some models missing — update VERSION_CONFIG_EXTENDED before running.")
    print("   Check https://openrouter.ai/models for current availability.")
```

---

## 13. Thinking Mode Handling per New Model

| Version | Model | Thinking? | Strategy |
|---|---|---|---|
| G3-D | gemma-4-26b-a4b-it | ❌ No | No action needed |
| G3-E | qwen3.6-35b-a3b | ✅ Yes | `/no_think` prefix in system prompt (same as G3-A, G3-G) |
| G3-F | deepseek-v4-flash | ✅ Optional | Non-think is the default — no prefix needed; `strip_thinking_tokens()` as safety net |
| G3-G | qwen3.5-35b-a3b | ✅ Yes | `/no_think` prefix in system prompt (same as G3-A, G3-E) |

**DeepSeek V4 Flash (G3-F) thinking mode details:** DeepSeek V4 Flash supports both Thinking and Non-Thinking modes. Non-thinking mode is the default — you do not need to inject any system prompt directive to disable it. The `thinking_model: True` flag in VERSION_CONFIG ensures `strip_thinking_tokens()` is applied as a safety net, but in practice no `<think>` blocks should appear for V4 Flash in non-thinking mode.

All Qwen variants (G3-A, G3-E, G3-G) use the identical `/no_think` injection: it is prepended to the system prompt as `/no_think\n\n` before the G1-C grounding rules. This is already implemented in the existing `generate_answer()` function — no code changes needed.

---

## 14. Final Recommendation and Priority Order

### Recommended Run Order (Prioritise by Expected Impact)

| Priority | Version | Model | Why Run First |
|---|---|---|---|
| 1st | **G3-E** | qwen3.6-35b-a3b | MoE successor to G3-A; 3B active params = fastest Qwen; best expected quality |
| 2nd | **G3-D** | gemma-4-26b-a4b-it | MoE successor to G3-B (known near-miss at FA=0.701); highest impact if it clears 0.75 |
| 3rd | **G3-F** | deepseek-v4-flash | Fills the DeepSeek gap; cheap, MIT license, reasonable latency |
| 4th | **G3-G** | qwen3.5-35b-a3b | Comparison baseline for G3-E; run after G3-E to judge the 3.5→3.6 upgrade effect |

### Predicted Outcomes (Hypotheses to Test)

| Version | FA Hypothesis | Safety Hypothesis | Reasoning |
|---|---|---|---|
| G3-D | **0.75–0.82** (up from G3-B 0.701) | ~84–88% | Gemma4 is a major generation upgrade; MoE architecture should improve faithfulness |
| G3-E | **0.78–0.85** (up from G3-A 0.766) | ~83–90% | Qwen3.6 with 3B active = faster and stronger than Qwen3:14b dense |
| G3-F | **0.77–0.84** | ~85–91% | DeepSeek V4 Flash has 1M context; excellent instruction following; closest to G2-D territory |
| G3-G | **0.75–0.82** | ~82–88% | Qwen3.5 is slightly weaker than 3.6 but still a MoE upgrade over G3-A |

### Success Criteria for G3 Extended Experiment

An extended G3 version is considered **successful** if it achieves:
- **FA ≥ 0.80** (approaching G2-D's 0.815)
- **Safety ≥ 86.7%** (G1/G2 deployment gate)
- **Generation latency ≤ 12 s** (mobile-acceptable)

If G3-E or G3-F achieves all three, VerdaSense has a viable open-source production model that matches the G2-D closed-source winner at 7–8× lower cost per query and with full commercial/self-hosting rights under MIT or Apache 2.0.

---

*Document prepared: 27 May 2026 | VerdaSense RAG — FYP Ablation Study | Universiti Malaya*
