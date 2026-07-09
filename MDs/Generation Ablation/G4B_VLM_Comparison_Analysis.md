# VerdaSense RAG — Experiment G4-B: VLM Comparison (GPT-4o-mini-V vs Gemini-2.5-Flash-V)

## Comprehensive Analysis & Discussion

**Experiment:** G4-B — Does the choice of vision model matter? Holding the **blind prompt** (G4-P winner) and everything else fixed, swap only the VLM.
**Stage:** FYP2 — Multimodal Generation Ablation (Pillar 1)
**Date:** 4 July 2026 — 34-case curated testset
**Arms:** **B1** = `gpt-4o-mini` Vision (OpenAI) · **B2** = `gemini-2.5-flash` Vision (Google)
**Fixed:** VLM prompt = blind/independent (P4) · Gen LLM = `gpt-4o-mini` · retrieval R1-C k=6 BGE-v5 · G1-F patient schema · RAGAS judge = `gpt-4o-mini` + `text-embedding-3-small` · 3 runs
**Testset:** `wound_testset_v5.json` — 34 imaged cases (A:8, B:6, C:4, D:3, E:3, F:3, G:7)
**Notebook:** `RAGAS_EVAL/G4B_VLM_Comparison/ragas_ablation_G4B_vlm_comparison.ipynb`

---

## 0. Insights Gained (executive summary)

1. **The VLM choice matters — decisively, and *against* the pricier model.** GPT-4o-mini-Vision wins on every axis: reliability, discrepancy detection, tissue accuracy, cost, and latency. There is no dimension on which Gemini-2.5-Flash wins.
2. **Gemini refused 41% of the clinical wound images** (42 / 102 caption attempts returned an **empty, content-blocked response**), concentrated on exactly the wounds that matter — infected, necrotic, cavity, and adversarial. Clean wounds (Cat F) had **0** refusals.
3. **The refusal is NOT a tunable safety setting.** A targeted test set all four configurable harm categories to `BLOCK_NONE`; it recovered **0 / 5** blocked images. Gemini reports `block_reason = BlockedReason.OTHER` with `safety_ratings = None` — a **non-configurable** policy filter that `safety_settings` cannot override on the standard Developer API.
4. **Deployment finding:** consumer VLM content filters are a real barrier to clinical wound imaging. The "obvious" choice (adopt the newer, pricier Google model) is *actively wrong* here — the model refuses to look at the severe wounds, which is the worst possible failure for a cross-validation safety-net.

**Verdict: keep `gpt-4o-mini` as the production VLM.** Gemini-2.5-Flash is disqualified for this use case.

---

## 1. Setup — a clean single-variable swap

Both arms use the **identical blind prompt** (the G4-P winner) and the identical downstream pipeline; the **only** difference is the vision model. So any gap is attributable to the model, not the framing. The blind prompt matters here: it is the design that makes the VLM a genuine cross-validator, and G4-B asks whether that design is vendor-robust.

Cost context (from `VLM_REGISTRY`): `gemini-2.5-flash` is ~2× the input / ~4× the output price of `gpt-4o-mini`, so B2 has to *earn* a switch with clearly better accuracy/detection.

---

## 2. Results — B1 vs B2 (3 runs, 34 cases)

| Metric | **B1 · GPT-4o-mini-V** | **B2 · Gemini-2.5-Flash-V** | Winner |
|---|:---:|:---:|:---:|
| **Image refusals** | **0 / 102 (0%)** | **42 / 102 (41%)** | **B1** |
| **VLM-DISC** (discrepancy detection, adv) | **100%** (21/21) | 47.6%\* (~83% on accepted) | **B1** |
| **Infection accuracy** (non-adv) | **73.1%** | 67.4%\* | **B1** |
| **Tissue-bucket accuracy** | **86.3%** | 66.7%\* | **B1** |
| VLM cost / run | **$0.0465** | $0.0634 (+36%) | **B1** |
| VLM latency / caption | **3.4 s** | 5.6 s (+64%) | **B1** |
| FA (downstream) | 0.586 | 0.583 | tie† |
| AR (downstream) | 0.377 | 0.461 | tie† (see §5) |
| Safety Pass | 89.2% | 91.2% | tie† |

\* B2's caption metrics are computed **only on the ~59% of images it did not refuse** — a favourable subsample (the cleaner wounds). VLM-DISC counts a refusal as a miss (a refused image *is* a failure to cross-validate), giving 47.6% overall; even crediting only accepted images it is ~83% (10/12), still below B1's 100%.
† Downstream FA/AR/Safety are **contaminated** for B2 (§5) — the 42 refused cases silently fell back to the no-caption path, so B2's answers were partly unimodal. The fair VLM comparison is the **caption-level** metrics.

---

## 3. The refusal finding — Gemini won't look at the severe wounds

The 42 refusals are not random. They track **how graphic the wound is**:

| Category | Theme | Gemini refusals |
|---|---|:---:|
| **F** | clean superficial | **0 / 9** |
| A | canonical WT1–8 | 12 / 24 (severe WT3/6/7/8 blocked 3/3; clean WT1/2 passed) |
| B | comorbidity (reuse images) | 9 / 18 |
| C | escalation | 3 / 12 |
| **D** | cavity / necrosis | **6 / 9** (deep cavity + extreme necrosis blocked 3/3) |
| E | complex chronic | 3 / 9 |
| **G** | adversarial | **9 / 21** (infected/necrotic traps blocked) |

**Clean wounds pass; infected/necrotic/cavity/adversarial wounds get blocked.** On the 7 adversarial cases — the entire point of VLM-DISC — Gemini refused **9 of 21** reads. A cross-validation safety-net that refuses to examine the dangerous wounds is worse than useless: it fails precisely where an error would be most harmful.

### The BLOCK_NONE test (closes the "did you disable safety filtering?" question)
A focused test (`scratchpad/test_gemini_safety.py`) called 5 of the 3/3-blocked images twice each — default settings vs **all four harm categories set to `BLOCK_NONE`**:

- **Recovered: 0 / 5.** `BLOCK_NONE` made no difference.
- Every block reports `block_reason = BlockedReason.OTHER`, `safety_ratings = None`.

`safety_ratings = None` means the block is **not** from the adjustable harm-category filters (if it were, they would be populated and `BLOCK_NONE` would clear them). `BlockedReason.OTHER` is a **non-configurable** usage-policy filter. Conclusion: the 41% refusal rate is **inherent to `gemini-2.5-flash` on the standard Developer API** and cannot be turned off — not a settings artifact. *(Vertex AI / enterprise medical-access tiers may differ, but were not available and are out of scope.)*

---

## 4. Even on the images it accepted, Gemini is no better

Discounting the refusals entirely and comparing only accepted reads:
- **Infection accuracy:** B1 73% vs B2 67%.
- **Tissue-bucket accuracy:** B1 86% vs B2 67%.
- **VLM-DISC:** B1 100% vs B2 ~83% (accepted).

So there is **no quality argument** for Gemini either — it is not that Gemini reads better but refuses more; it reads *worse* on the images it accepts, at higher cost and latency.

---

## 5. Downstream FA / AR / Safety — why they look like a tie (and why that's misleading)

FA (0.586 vs 0.583), AR (0.377 vs 0.461), Safety (89% vs 91%) look comparable, but for B2 they are **not a clean VLM measurement**: on the 42 refused cases the pipeline received an empty caption and fell back to the *no-caption* branch, so ~41% of B2's answers were effectively **unimodal (A0-like)**. That is exactly why B2's AR (0.461) drifts toward the no-caption behaviour rather than reflecting Gemini's captions. The honest read: **downstream metrics can't distinguish the VLMs here because B2 was partly not using a caption at all.** The decision rests on the caption-level metrics (§2–4), where B1 wins outright.

---

## 6. Interpretation — the deployment lesson

G4-B was framed as "does a stronger/pricier VLM help?" The answer is a clean **no**, for a reason more interesting than raw accuracy: **a general-purpose consumer VLM's content policy can refuse graphic medical imagery outright, and that refusal may be non-negotiable via the public API.** For a clinical decision-support tool, model selection therefore cannot be made on benchmark accuracy alone — **reliability on the actual (graphic) input distribution is a first-class criterion**, and it eliminated the pricier option here. GPT-4o-mini's 0% refusal rate on the full severity range is the decisive property.

This also *reinforces* the G4-P/G4-A story: the blind-cross-validation design works — but its value depends on a VLM that will actually **look at the wound**. B1 satisfies that; B2 does not.

---

## 7. Limitations & Threats to Validity

1. **Single Developer-API tier.** Gemini via Vertex AI or an enterprise/medical-access tier might refuse less; we tested the standard API + `safety_settings=BLOCK_NONE` (the levers available to a typical developer), and it did not help.
2. **`gemini-2.5-flash` only.** A larger Gemini (Pro) or other vendors (Claude Vision, etc.) are not tested — G4-B answers "does *this* pricier alternative beat the incumbent?", which is the practical question.
3. **B2 caption metrics on a subsample.** Accepted-image accuracy (67–83%) is on the easier ~59%; the true full-distribution accuracy would likely be *lower*, strengthening the verdict.
4. **Downstream metrics contaminated for B2** (§5) — reported for completeness, not used for the decision.

---

## 8. Recommendation & Next Steps

1. **Keep `gpt-4o-mini` as the production/eval VLM** (already the default). No change needed.
2. **Do NOT re-run B2** — `BLOCK_NONE` is proven ineffective; a re-run would reproduce the 41% refusals.
3. Record the deployment finding in the thesis: *VLM content filters are a real clinical-imaging barrier; model choice must weigh input-distribution reliability, not just accuracy.*
4. Remaining Pillar 1: **VLM-ACC** (single-arm caption accuracy on all imaged cases — even cheaper) and **H1** (Ms Saw) — the decisive clinical validation.

---

## 9. One-Paragraph Viva Narrative

*"G4-B asks whether swapping the vision model helps. Holding the blind prompt fixed and changing only the VLM, GPT-4o-mini wins on every axis. The reason is striking: Gemini-2.5-Flash refused 41% of the clinical wound images — returning empty, content-blocked responses concentrated on the infected, necrotic, and adversarial wounds, while passing the clean ones. Crucially, this is not a tunable safety setting: disabling all four configurable harm categories with `BLOCK_NONE` recovered none of the blocked images, and Gemini reports the block as `BlockedReason.OTHER` with no safety ratings — a non-configurable policy filter. So the pricier, newer model is disqualified not on accuracy but on reliability: a cross-validation safety-net that refuses to look at the dangerous wounds is worse than useless. GPT-4o-mini, with a 0% refusal rate across the full severity range, 100% discrepancy detection, higher tissue accuracy, lower cost, and lower latency, is the correct production choice — and the finding itself, that consumer VLM content filters can silently block graphic medical imagery, is a genuine deployment lesson for clinical multimodal systems."*

---

*Companion to `G4P_VLM_Prompt_Strategy_Analysis.md` and `G4A_Multimodal_Caption_Analysis.md`. Results: `RAGAS_EVAL/G4B_VLM_Comparison/results/`. Safety-settings test: `scratchpad/test_gemini_safety.py`.*
