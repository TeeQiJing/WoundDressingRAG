# VerdaSense RAG — Experiment G4-P: VLM Caption Prompt Strategy Ablation
## Comprehensive Analysis & Discussion

**Experiment:** G4-P — VLM Caption Prompt Strategy (how to prompt the vision model)
**Stage:** FYP2 — Multimodal Generation Ablation (Pillar 1: VLM contribution)
**Date:** 1 July 2026 · **re-run 3 July 2026 on the fully curated + Gemini-cross-validated testset** (numbers below are the curated run)
**Configuration:** VLM = `gpt-4o-mini`-Vision · Generation LLM = `gpt-4o-mini` · Retrieval R1-C multi-axis dense (k=6, fixed) · Embedding `BAAI/bge-large-en-v1.5` · `db_wound_care_v5_bge` · Prompt G1-F patient schema · 3 runs each
**Testset:** `wound_testset_v5.json` — **21 imaged cases** (Cat A:8, B:2, C:1, D:1, E:1, F:1, **G:7 adversarial**), every image three-way validated (Claude read ↔ gold label ↔ Gemini-Pro blind read)
**RAGAS Judge:** `gpt-4o-mini` + `text-embedding-3-small` (fixed — never changed)
**Notebook:** `RAGAS_EVAL/G4P_VLM_Prompt_Strategy/ragas_ablation_G4P_vlm_prompt_strategy.ipynb`

---

## 0. Insights Gained (executive summary)

G4-P produced four clean, defensible findings:

1. **Blind prompting is decisively better at cross-validation.** The blind prompt (P4) caught the label↔image discrepancy on **100% (21/21)** of adversarial samples; the three label-shown prompts caught only **14–19%** (P1 3/21, P2 4/21, P3 3/21). Robust across 7 cases × 3 runs — and after curation, **perfectly stable**: every P4 adversarial read is identical across all three runs.
2. **Label anchoring is *axis-specific* and near-total on infection.** Pooling the label-shown prompts, they caught only **2% (1/45)** of *infection*-axis discrepancies but **50% (9/18)** of *tissue*-axis ones. The VLM will almost never contradict an "infected/not-infected" label it is shown, but *sometimes* reports visible necrosis despite a "clean" label. Anchoring is worst on the most safety-critical axis — missed infection.
3. **Blind costs essentially nothing downstream (new — the pilot's FA penalty disappeared on clean labels).** On the curated testset, blind Faithfulness is **0.629**, statistically tied with the best label-shown prompt (P3 0.630) and *above* the shipped P2 (0.622). The earlier "~5 pp FA cost of blind" was an artifact of the mis-labelled pilot cases; with correct labels, **blind buys 100% error-detection at no Faithfulness cost**. Safety is identical (90.5%) across all four.
4. **Blind's only "cost" is honest, and clinically safe.** Its non-adversarial infection accuracy is **78.6%** (vs the label-shown 100%), but that 100% is *label-echoing, not perception*. P4's 3 canonical disagreements are all explainable: it over-calls infection on **necrotic/sloughy** wounds (wt5, wt6 — which need review anyway) and under-calls the one **peri-wound-only** infection (wt4, where infection is genuinely not bed-visible).

---

## Table of Contents
1. [Experiment Overview](#1-experiment-overview)
2. [Why This Experiment Exists (the G4-A puzzle)](#2-why-this-experiment-exists)
3. [Variants Tested](#3-variants-tested)
4. [Metric Reference](#4-metric-reference)
5. [Results Summary Table](#5-results-summary-table)
6. [Finding 1 — Blind Detection vs Label Anchoring](#6-finding-1--blind-detection-vs-label-anchoring)
7. [Finding 2 — Anchoring is Axis-Specific](#7-finding-2--anchoring-is-axis-specific)
8. [Finding 3 — The Cost of Blind Assessment](#8-finding-3--the-cost-of-blind-assessment)
9. [Downstream FA / AR and Safety](#9-downstream-fa--ar-and-safety)
10. [Interpretation — Why the VLM Anchors](#10-interpretation--why-the-vlm-anchors)
11. [Recommended Design](#11-recommended-design)
12. [Limitations and Threats to Validity](#12-limitations-and-threats-to-validity)
13. [Connection to G4-A and Next Steps](#13-connection-to-g4-a-and-next-steps)

---

## 1. Experiment Overview

G4-P isolates one question: **how should the VLM be prompted to caption the wound?** All retrieval, generation, and the vision model itself are fixed; only the **VLM prompt** changes across four strategies. It measures both *caption quality* (does the VLM read the wound correctly, and does it catch CV-label errors?) and *downstream* recommendation quality (FA/AR). The four strategies separate **framing** (appearance vs cross-validation vs terse) from **whether the VLM is shown the CV labels** (P1–P3 shown; P4 blind).

---

## 2. Why This Experiment Exists

Experiment **G4-A** (caption vs no-caption) showed the VLM caption raised Answer-Relevance but slightly lowered Faithfulness, and — critically — **did not catch the CV-label error** on the adversarial case: the production caption echoed the wrong "Not infected" label. G4-P tests whether this is a *framing* problem (soft "does it agree?" wording) or an *anchoring* problem (merely *showing* the labels), with P4 (blind) as the decisive test.

---

## 3. Variants Tested

| ID | Label | CV labels shown to VLM? | Framing |
|----|-------|:---:|---|
| **P1** | Appearance-only | ✅ | Describe what you see; no cross-validation (R5-style) |
| **P2** | Current production | ✅ | Cross-validate against CV labels + dressing implications (shipped prompt) |
| **P3** | Minimal / terse | ✅ | 1–2 sentence caption + structured fields |
| **P4** | **Blind / independent** | ❌ | Assess tissue / infection / moisture / depth **from the image alone** |

All four return an identical JSON schema so the metrics parse uniformly.

---

## 4. Metric Reference

| Metric | Type | What it measures |
|---|---|---|
| **Caption Infection-Accuracy** | deterministic | On the 14 non-adversarial cases (×3 runs = 42): does the VLM's *visual* infection read match the (correct) CV label? *(See §8 for the circularity caveat.)* |
| **Tissue-bucket accuracy** | deterministic | Non-viable ≥25% vs <25% agreement with the label |
| **Discrepancy-Detection Rate** | deterministic | On the 7 adversarial cases (×3 = 21): does the VLM *disagree* with the wrong label on the axis that is actually wrong (**infection OR tissue**)? |
| **FA / AR** | RAGAS LLM-judge | Downstream recommendation faithfulness / relevance |
| **Safety Pass** | deterministic | Contraindication / antibiotic / referral correctness |

The adversarial set spans three directions: **missed infection** (2), **missed necrosis** (2), **CV over-call** (3). The multi-axis metric credits necrosis catches via the tissue axis, not infection.

---

## 5. Results Summary Table

**Mean over 3 runs · 21 cases · non-adversarial n=14 (×3=42) · adversarial n=7 (×3=21). Curated-testset run, 2026-07-03.**

| Variant | Infection Acc (non-adv) | Tissue-bucket | **Discrepancy Detection (adv)** | FA (mean ± SD) | AR (mean ± SD) | Safety |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **P1** Appearance | 1.00 | 0.952 | **0.143** (3/21) | 0.6092 ± 0.0055 | 0.3577 ± 0.044 | 90.5% |
| **P2** Current prod | 1.00 | 0.937 | **0.190** (4/21) | 0.6215 ± 0.0197 | 0.3562 ± 0.079 | 90.5% |
| **P3** Terse | 1.00 | 0.952 | **0.143** (3/21) | **0.6301** ± 0.0129 | **0.3665** ± 0.030 | 90.5% |
| **P4** **Blind** | **0.786** | **0.794** | **1.00** (21/21) | 0.6293 ± 0.0270 | 0.3238 ± 0.085 | 90.5% |

Two headlines: (1) only the blind prompt catches the discrepancies (100% vs 14–19%); (2) **FA is flat across all four (0.609–0.630)** — blind is *not* penalised on Faithfulness (P4 0.629 ≈ P3 0.630 > P2 0.622 > P1 0.609). AR is uniformly low/noisy (0.32–0.37), P4 marginally lowest but within SD.

---

## 6. Finding 1 — Blind Detection vs Label Anchoring

Across 21 adversarial samples (7 cases × 3 runs):

- **P4 (blind): 21/21 = 100%** — caught every discrepancy, in every direction (missed infection, missed necrosis, over-call), on **every run identically**.
- **P1 / P3 (label-shown): 3/21 = 14.3%; P2: 4/21 = 19.0%** — near-total failure to cross-validate; the extra P2 catch is a single necrosis case.

The blind prompt's per-case reads confirm it is *reading the image*, not guessing — and after curation, every read is stable across all 3 runs:

| Adversarial case | CV label | P4 blind read (×3 runs) | Caught via |
|---|---|---|---|
| miss_infection_wt1 | Not infected | **Infected**, NV 0% | infection |
| miss_infection_wt2 | Not infected | **Infected**, NV 30% | infection + tissue |
| miss_necrosis_wt1 | clean (NV 0%) | Infected, **NV 60%** | tissue + infection |
| miss_necrosis_wt2 | clean (NV 5%) | Infected, **NV 30%** | tissue + infection |
| overcall_wt3 | Infected | **Not infected**, NV 0% | infection |
| overcall_wt4 | Infected | **Not infected**, NV 0% | infection |
| overcall_clean | Infected | **Not infected**, NV 10% | infection |

Note the **over-call resistance**: on all three over-call cases (`wt3`, `wt4`, `overcall_clean`) the blind VLM correctly said "Not infected" on a clean wound the CV pipeline wrongly called infected — so blind assessment resists false positives *and* catches false negatives. (In the pilot, `overcall_fusc` flip-flopped between runs because its image was actually sloughy/infected; the curation replaced it with a genuinely clean wound — `overcall_clean` — which now reads cleanly 3/3.)

> This is the core FYP2 result: the VLM performs genuine visual cross-validation **only when it is not told the upstream CV answer** — a robust, now-stable 100% vs 14–19%.

---

## 7. Finding 2 — Anchoring is Axis-Specific

Pooling the three label-shown prompts (P1+P2+P3) over 3 runs and splitting the adversarial catches by which axis was actually wrong:

- **Infection axis** (miss_infection ×2 + over-call ×3 = 5 cases → 45 samples): caught **1/45 = 2%** → the VLM *almost never* contradicts an infection label it is shown (**~98% anchored**).
- **Tissue axis** (miss_necrosis ×2 = 2 cases → 18 samples): caught **9/18 = 50%** → it *sometimes* reports the necrosis it visually sees despite a "clean" label (**~50% anchored**).

**Interpretation:** infection is a *judgment* the VLM defers to the label on; visible dead tissue is a *concrete observation* it finds harder to suppress. So **anchoring is strongest exactly on the most safety-critical axis — missed infection** — the single strongest argument for the blind prompt: the axis where a CV-pipeline error is most dangerous is the axis where the label-shown VLM is most useless as a checker. (Blind, by contrast, caught infection 15/15 and tissue 6/6 — 100% on both.)

---

## 8. Finding 3 — The Cost of Blind Assessment

Blind's detection gain is paid for in a small number of non-adversarial disagreements — now fully characterised (all 3/3 runs, i.e. systematic and explainable, not noise):

| P4 (blind) canonical disagreement | Case | Direction | Why |
|---|---|---|---|
| **Infection over-call on necrotic eschar** | cat_a_wt5 | false-positive | black eschar visually mimics infection |
| **Infection over-call on heavy slough** | cat_a_wt6 | false-positive | sloughy bed visually mimics infection |
| **Infection under-call, peri-wound only** | cat_a_wt4 | false-negative | infection is peri-wound erythema/induration, *not bed-visible* |

Net: non-adversarial **infection accuracy 78.6%** (11/14) and **tissue-bucket 79.4%** (vs the label-shown 100% / 95%). Three crucial reframings:

1. **The label-shown "100%" is circular.** Those prompts score perfectly on non-adversarial cases by *echoing* the correct label, not by perceiving — the same behaviour that makes them ~2% on infection-axis adversarial cases. P4's 78.6% is the **first honest measure** of the VLM's genuine visual infection-reading ability.
2. **Two of the three errors are the *safe* direction.** wt5/wt6 over-call infection on necrotic/sloughy wounds — both *warrant review anyway* (they need debridement), and because VLM reads are **advisory and never override the rule engine**, a false "possible infection — get reviewed" is clinically benign.
3. **The one dangerous-direction error (wt4) is a documented data-modality limit, not a model failure.** WT3/WT4 ("infected + low non-viable") is intrinsically hard to photograph — infection there is peri-wound/clinical, not bed-visible (this is exactly why the testset curation could not find a bed-visible infected-WT4 image among 5 candidates). The pipeline stays safe because the **CV/clinical infection label still drives the antibiotic decision** — the blind VLM is a *checker*, not the source of truth.

---

## 9. Downstream FA / AR and Safety

- **FA is flat and blind is not penalised (revised from the pilot).** Ordering: **P3 (0.6301) ≈ P4 (0.6293) > P2 (0.6215) > P1 (0.6092)** — a ~2 pp spread, within run-to-run SD. The pilot's "blind costs ~5 pp FA" was an artifact of mis-labelled cases forcing the blind caption to diverge from the KB; with curated labels the blind caption is accurate enough that its recommendation stays as guideline-faithful as any other. **Blind therefore buys 100% error-detection at no Faithfulness cost** — a materially stronger result than the pilot.
- **AR is low and noisy across all variants (0.32–0.37)** with P4 marginally lowest (0.324) but well within SD — not a decisive signal on this 21-case mix. (Plausibly the blind caption's added caveats trim answer relevance slightly.)
- **Safety is identical (90.5%) across every variant** — the caption prompt is safety-neutral; the deterministic referral/antibiotic checker is caption-independent (driven by `classify_wound`, not the VLM).

---

## 10. Interpretation — Why the VLM Anchors

`gpt-4o-mini`-Vision, given both an image and a textual assertion about it, treats the assertion as a high-confidence prior and reproduces it — classic prompt anchoring / confirmation bias. Finding 2 sharpens this: the effect is **strongest on subjective judgments (infection, ~98% anchored)** and **weaker on concrete visual facts (dead-tissue area, ~50%)**. Critically, instruction does not fix it — P2's explicit "flag any discrepancy" wording anchored just as hard as P1's appearance-only wording (19% vs 14%, both near-floor on infection). The only reliable intervention is **information removal**: to use a VLM as an independent verifier of an upstream model, withhold the upstream model's output from it.

---

## 11. Recommended Design

Adopt a **two-pass / dual-role caption** (the evidenced default, already live in the app):

1. **Blind verification pass (P4):** the VLM assesses the image with CV labels withheld → an *independent* read. Disagreement with the CV pipeline becomes an **advisory discrepancy flag** (implemented in `wound_app_multimodal.py`, which computes VLM-read vs CV-label and surfaces the flag). It never overrides the rule engine.
2. **Recommendation pass:** the dressing plan continues to use the CV labels + retrieved guidelines, enriched by the caption's descriptive content.

Robustness upgrades, motivated by Finding 3:
- **Confidence-gate / de-prioritise the flag on necrotic / heavy-slough wounds** (where blind over-calls infection) to trim benign false positives.
- **Treat a blind "not infected" on a CV-"infected" WT3/4 as low-confidence** (peri-wound infection is not bed-visible) — do not let it soften the antibiotic advice.

**Concrete action:** the blind prompt is already live; keep it, and re-run **G4-A** under it to show the caption now *catches* CV errors end-to-end at no FA cost.

---

## 12. Limitations and Threats to Validity

1. **7 adversarial cases (21 samples) — better, still modest.** The 100% vs 14–19% split is consistent across all 7 cases and 3 runs (and now perfectly stable), a strong signal; a larger VLM-DISC set (15–20 cases) would tighten the CI. The axis-specific split (Finding 2) rests on 45 infection + 18 tissue samples — reasonably firm but still one testset.
2. **Circular non-adversarial accuracy for label-shown arms** (§8) — their 100% is label-echoing, not perception. P4's 78.6% is the only trustworthy accuracy figure.
3. **Adversarial images now unambiguous (pilot limitation resolved).** The pilot's flip-flopping `overcall_fusc` was replaced during curation with a genuinely clean wound; all 7 adversarial reads are now stable 3/3. No remaining borderline adversarial images.
4. **VLM = `gpt-4o-mini`-Vision only.** Whether a stronger VLM anchors as hard, or over-calls less on necrotic wounds, is **G4-B** (run under the blind prompt).
5. **FA rebased vs FYP1** (~0.62 vs 0.81) — different testset + patient-friendly schema; only within-G4-P comparisons are valid.

---

## 13. Connection to G4-A and Next Steps

G4-P **explains and repairs** the G4-A puzzle, now with robust, curated numbers:

- G4-A: *"the caption cost Faithfulness and did not catch the CV error."*
- G4-P: *"because the production prompt showed the VLM the CV labels, it anchored — catching ~2% of infection discrepancies. The blind prompt catches 100%, and — on correct labels — at **no** Faithfulness cost and a conservative, clinically-safe false-positive profile."*

**Viva narrative:** the VLM adds genuine cross-validation value — but only when prompted **blind**, a non-obvious, empirically-demonstrated design decision; the failure it fixes (missed infection) is precisely the most dangerous one; and it does so without degrading recommendation faithfulness or safety.

**Next steps:**
1. **Re-run G4-A under the blind prompt** — expect the caption to shift from "costs FA" to "catches errors at no FA cost" end-to-end.
2. **Grow VLM-DISC to 15–20 cases** to firm up the rate and the axis-specific split.
3. **G4-B** — GPT-4o-Vision vs Gemini-Vision under the blind prompt (does a stronger VLM keep 100% detection with fewer necrotic-wound false positives?).
4. **H1** — Ms Saw spot-checks the blind captions (are the flagged discrepancies real, and the necrotic-wound flags acceptable as advisories?).

---

*Companion to `VerdaSense_FYP2_Ablation_Map_v5.md` (Pillar 1) and the `G4A` analysis. Results: `RAGAS_EVAL/G4P_VLM_Prompt_Strategy/results/`.*
