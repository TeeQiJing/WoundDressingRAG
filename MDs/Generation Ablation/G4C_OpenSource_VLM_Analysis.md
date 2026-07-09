# VerdaSense RAG — Experiment G4-C: Open-Source VLM Comparison (OpenRouter)

## Comprehensive Analysis & Discussion

**Experiment:** G4-C — Among **open-source** vision models, which reads wounds best, and do open models avoid the clinical-image refusals that disqualified Gemini in G4-B?
**Stage:** FYP2 — Multimodal Generation Ablation (Pillar 1) · companion to **G4-B** (closed-source)
**Date:** 5 July 2026 — 34-case curated testset · all VLMs via **OpenRouter**
**Arms:** **C1** Qwen2.5-VL-72B (dense) · **C2** Qwen3-VL-235B-A22B (MoE) · **C3** Gemma-3-27B (dense) · **C4** Gemma-4-26B-A4B (MoE)
**Fixed:** VLM prompt = **blind** (G4-P winner) · Gen LLM = `gpt-4o-mini` · retrieval R1-C k=6 BGE-v5 · G1-F patient schema · RAGAS judge = `gpt-4o-mini` + `text-embedding-3-small` · **reasoning/thinking DISABLED** (OpenRouter `reasoning:effort=none` + `/no_think` for Qwen, as in G3) · 3 runs
**Notebook:** `RAGAS_EVAL/G4C_OpenSource_VLM/ragas_ablation_G4C_opensource_vlm.ipynb`

---

## 0. Insights Gained (executive summary)

1. **Open-source VLMs eliminate the refusal problem.** All 4 open models had a ~**0% error rate** (1 transient OpenRouter error each on C1/C3; 0 on C2/C4) — versus **Gemini's 41% content-blocked refusals** (G4-B). None refused the graphic/infected/necrotic wounds. A self-hostable open VLM removes the vendor content-policy risk entirely.
2. **VLM-DISC alone is gameable — it must be read *with* non-adversarial accuracy.** **Gemma-3-27B scores 100% VLM-DISC but is the *worst* reader** (infection accuracy 49%, tissue-bucket 35%). It achieves "perfect" discrepancy detection by **over-calling "Infected" on 95% (40/42) of clean wounds** — a broken alarm that always rings. Its 100% is an artifact, not perception. This is a clean methodological lesson: a model that flags everything catches every discrepancy while being clinically useless.
3. **The best open model — Qwen2.5-VL-72B — matches GPT-4o-mini on accuracy.** Infection **76%** (vs GPT's 73%), tissue-bucket **85%** (vs 86%), best calibration (21% over-call), 0 refusals — at **~6× lower cost** ($0.008 vs $0.047/run) and self-hostable. Its VLM-DISC (71%) is lower than GPT-4o-mini's 100%, the price of being *calibrated* rather than trigger-happy.
4. **Bigger ≠ better.** The largest model, Qwen3-VL-235B, *under*-performed the smaller Qwen2.5-VL-72B on every caption metric (infection 68% vs 76%, DISC 57% vs 71%). Scale did not predict wound-reading quality.
5. **Deployment implication (clinically important):** for a wound tool handling patient photos, a **self-hostable open VLM (Qwen2.5-VL-72B)** offers near-GPT-4o-mini accuracy with **data sovereignty** (images never leave your infrastructure) and no content-policy refusals — a genuinely viable alternative, with GPT-4o-mini remaining the best single choice on the accuracy+sensitivity sweet spot.

---

## 1. Results — the 4 open models (3 runs, 34 cases) + closed-source reference

| Arm | Model | Refusal/err | **Infection acc** | **Tissue-bucket** | **VLM-DISC** | Over-call\* | FA | Safety | $/run | Latency |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **C1** | Qwen2.5-VL-72B | 1% | **0.763** | **0.851** | 0.714 | **21%** | 0.604 | 91.2% | **$0.008** | 6036 ms |
| **C2** | Qwen3-VL-235B | 0% | 0.684 | 0.794 | 0.571 | 38% | 0.578 | 89.2% | $0.012 | 6143 ms |
| **C3** | Gemma-3-27B | 1% | 0.487 | 0.347 | **1.000** | **95%** | 0.603 | 90.2% | $0.003 | 6888 ms |
| **C4** | Gemma-4-26B | 0% | 0.600 | 0.784 | 0.857 | 62% | 0.581 | 90.2% | $0.004 | 5075 ms |
| *B1* | *GPT-4o-mini-V (G4-B)* | *0%* | *0.731* | *0.863* | *1.000* | *mod.* | *0.586* | *89.2%* | *$0.047* | *3390 ms* |
| *B2* | *Gemini-2.5-Flash (G4-B)* | ***41%*** | *0.674\** | *0.667\** | *0.476\** | — | *0.583* | *91.2%* | *$0.063* | *5567 ms* |

\* **Over-call** = fraction of *non-adversarial, not-infected* cases the VLM wrongly read as "Infected" (calibration; lower is better). Gemini's caption metrics (\*) are on the ~59% of images it did not refuse.

---

## 2. Finding 1 — Open models do not refuse clinical wound images

The single most important deployment result. In G4-B, **Gemini-2.5-Flash refused 41%** of the wound images (empty, `BlockedReason.OTHER` responses concentrated on infected/necrotic/adversarial wounds), and the block was **non-configurable** (`safety_settings=BLOCK_NONE` recovered 0/5). G4-C shows the open field has **no such barrier**: every open model returned a real assessment for the full severity range, including the graphic wounds Gemini blocked. Error rates were 0–1% and the non-zero ones were **transient OpenRouter timeouts, not content blocks**.

**Why this matters:** a cross-validation safety-net is worthless if it refuses to look at the dangerous wounds. Open models (self-hostable, no usage-policy filter) structurally avoid the failure mode that disqualified the pricier proprietary Google model.

---

## 3. Finding 2 — VLM-DISC is gameable; read it *with* accuracy (the Gemma-3 trap)

Taken naïvely, **Gemma-3-27B "wins": 100% VLM-DISC** — matching GPT-4o-mini. But the caption-accuracy columns expose this as a mirage:

- Infection accuracy **48.7%** (worse than a coin flip on a binary label)
- Tissue-bucket accuracy **34.7%**
- **Over-call rate 95%** — it read "Infected" on **40 of 42** clean, not-infected wounds

Gemma-3 achieves perfect discrepancy detection not by *perceiving* discrepancies but by reading **"abnormal" on everything** — so it disagrees with whatever label is planted (its adversarial detections were all via the *infection+tissue* or *tissue* axes, i.e. it over-reads both). It is a fire alarm wired to the "on" position.

The over-call rate cleanly rank-orders the field and inversely tracks accuracy:

| Model | Over-call | Infection acc | VLM-DISC | Reading |
|---|:--:|:--:|:--:|---|
| Qwen2.5-VL-72B | **21%** | **0.76** | 0.71 | calibrated → accurate, misses some subtle discrepancies |
| Qwen3-VL-235B | 38% | 0.68 | 0.57 | moderate |
| Gemma-4-26B | 62% | 0.60 | 0.86 | over-caller → higher DISC, lower accuracy |
| Gemma-3-27B | **95%** | **0.49** | **1.00** | pure over-caller → fake 100% DISC |

**The lesson (methodological, thesis-worthy):** discrepancy-detection rate must **never** be reported without the non-adversarial accuracy beside it. GPT-4o-mini's 100% VLM-DISC is *genuine* (backed by 73% infection / 86% tissue accuracy and only moderate over-call); Gemma-3's identical 100% is *manufactured* by a 95% over-call bias. Same headline number, opposite clinical value. (This mirrors the G4-P finding that label-shown prompts had illusory 100% accuracy by echoing labels — here a model manufactures illusory 100% *detection* by flagging everything.)

---

## 4. Finding 3 — The best open model matches GPT-4o-mini on accuracy

Reading Finding 2 correctly, the strongest open model is **Qwen2.5-VL-72B (C1)**:

- **Infection accuracy 0.763 — actually higher than GPT-4o-mini's 0.731**
- **Tissue-bucket 0.851 ≈ GPT-4o-mini's 0.863**
- Best calibration of the field (21% over-call)
- 0 refusals, $0.008/run (~6× cheaper than GPT-4o-mini), self-hostable

Its one deficit is **VLM-DISC 71% vs GPT-4o-mini's 100%** — but this is the *flip side of being calibrated*: because it does not over-call, it misses ~6/21 of the (sometimes subtle) planted discrepancies. GPT-4o-mini is the rare model that is **both** highly sensitive (100% DISC) **and** accurate (73%/86%) with only moderate over-call — the genuine sweet spot, which is why it remains the single best choice. Qwen2.5-VL is the closest, and the trade (a little less discrepancy-sensitivity for self-hosting + 6× cost + data sovereignty) is very reasonable for a clinical deployment.

---

## 5. Finding 4 — Bigger is not better

Model scale did **not** predict wound-reading quality:

| | Qwen2.5-VL-**72B** dense | Qwen3-VL-**235B** MoE |
|---|:--:|:--:|
| Infection acc | **0.76** | 0.68 |
| Tissue-bucket | **0.85** | 0.79 |
| VLM-DISC | **0.71** | 0.57 |

The newer, ~3× larger Qwen3-VL-235B lost to the older, smaller Qwen2.5-VL-72B on every caption metric. Likewise Gemma-4 (MoE) did not beat a well-calibrated dense model on accuracy. For this niche visual task (wound tissue/infection reading), the specific model's calibration matters far more than parameter count — a useful, non-obvious result for anyone selecting a VLM for medical imaging.

---

## 6. Cost, latency & self-hosting

- **Cost:** open models are **4–15× cheaper** per run than GPT-4o-mini ($0.003–0.012 vs $0.047), and if **self-hosted** the marginal per-image cost approaches zero.
- **Latency:** open-via-OpenRouter is *slower* (5–7 s/caption) than GPT-4o-mini (3.4 s); self-hosting on adequate GPUs could close this, and it is non-blocking for a non-real-time wound tool.
- **Data sovereignty (the clinical clincher):** a self-hosted open VLM means **patient wound photos never leave your infrastructure** — no third-party API sees them. For a clinical tool this is a significant privacy/governance advantage that neither GPT-4o-mini nor Gemini can offer.

---

## 7. Downstream FA / AR / Safety — not a differentiator (as expected)

FA (0.578–0.604), AR (0.31–0.37), and Safety (89–91%) are essentially flat across all four open models and match the G4-A/G4-B range. This reconfirms the G4-A finding: **the caption is FA-/safety-neutral downstream regardless of which VLM produces it** — the VLM choice shows up in *caption quality* (accuracy + DISC + refusals), not in RAGAS FA. FA remains the wrong lens for the VLM contribution.

---

## 8. Limitations & Threats to Validity

1. **VLM-DISC over 7 adversarial cases (21 samples).** Firm enough to rank, but the exact rates carry sampling noise; the over-call rate (42 clean samples) is the more stable calibration signal.
2. **OpenRouter pricing is approximate** — cost figures are indicative; the ranking (open ≪ proprietary) is robust regardless.
3. **Latency reflects OpenRouter shared endpoints**, not self-hosted deployment; the self-hosting latency claim is an inference, not measured here.
4. **Schema discipline varies** — C1 occasionally reported tissue %s not summing to 100, C4 occasionally emitted odd infection strings ("0"). Handled gracefully (excluded/normalised) but a minor reliability note.
5. **Downstream metrics for the over-caller (C3)** are partly meaningless as a VLM signal — its captions are dominated by a fixed "infected" bias.

---

## 9. Recommendation & Next Steps

1. **Production default stays GPT-4o-mini** — genuine 100% VLM-DISC + high accuracy + fastest + 0 refusals; the accuracy-and-sensitivity sweet spot.
2. **Recommended open / self-hostable alternative: Qwen2.5-VL-72B** — near-identical accuracy (infection *higher* than GPT-4o-mini), 0 refusals, 6× cheaper, and it keeps patient images in-house. Use it where data sovereignty or vendor-lock avoidance outweighs the ~29 pp lower discrepancy-sensitivity.
3. **Avoid Gemma-3-27B for this task** — its 100% VLM-DISC is an over-calling artifact (49% accuracy). A cautionary example, not a candidate.
4. **Report DISC only alongside accuracy + over-call rate** in the thesis — G4-C is the evidence that DISC in isolation is gameable.
5. Remaining Pillar 1: **VLM-ACC** (single-arm caption accuracy) and **H1** (Ms Saw) — the decisive clinical validation.

---

## 10. One-Paragraph Viva Narrative

*"G4-C asks whether an open-source vision model can replace the proprietary one. The answer has three layers. First, open models solve the problem that disqualified Gemini: all four had ~0% refusals versus Gemini's 41% non-configurable content blocks — a self-hostable open VLM will actually look at graphic wounds. Second, G4-C exposes a methodological trap: Gemma-3-27B scores a perfect 100% discrepancy-detection rate, but it earns it by reading 'Infected' on 95% of clean wounds — its infection accuracy is 49%, worse than chance, so the 100% is a manufactured artifact of over-calling, not perception. Discrepancy detection must therefore always be read together with non-adversarial accuracy — GPT-4o-mini's 100% is genuine because its accuracy backs it; Gemma-3's is not. Third, once read correctly, the best open model — Qwen2.5-VL-72B — matches GPT-4o-mini on accuracy (infection 76% vs 73%, tissue 85% vs 86%) at a sixth of the cost, with zero refusals and full data sovereignty, while the largest model tested (Qwen3-VL-235B) was beaten by this smaller one, showing scale doesn't predict wound-reading quality. GPT-4o-mini remains the best single choice on the sensitivity-and-accuracy sweet spot, but Qwen2.5-VL-72B is a genuinely viable, self-hostable, privacy-preserving alternative for a clinical deployment."*

---

*Companion to `G4B_VLM_Comparison_Analysis.md` (closed-source), `G4P_VLM_Prompt_Strategy_Analysis.md`, `G4A_Multimodal_Caption_Analysis.md`. Results: `RAGAS_EVAL/G4C_OpenSource_VLM/results/`.*
