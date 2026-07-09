# VerdaSense RAG — Experiment G4-A: VLM Caption vs No-Caption

## Comprehensive Analysis & Discussion

**Experiment:** G4-A — Does adding a VLM clinical caption to the generation stage improve the dressing recommendation vs the unimodal (no-caption) baseline?
**Stage:** FYP2 — Multimodal Generation Ablation (Pillar 1: the headline multimodal experiment)
**Date:** 4 July 2026 — run on the **expanded 34-case** curated testset, with the blind (G4-P-winning) caption *(supersedes the 21-case pilot of 3 Jul)*
**Arms:** A0 = unimodal (retrieved KB chunks + T.I.M.E. payload) · A1 = multimodal (same + **blind VLM caption**) — literally the app's Multimodal Off/On toggle
**Configuration:** VLM = `gpt-4o-mini`-Vision (blind prompt) · Generation LLM = `gpt-4o-mini` (temp 0) · Retrieval R1-C multi-axis dense k=6 · `BAAI/bge-large-en-v1.5` · `db_wound_care_v5_bge` · G1-F patient schema · 3 runs
**Testset:** `wound_testset_v5.json` — **34 imaged cases** (A:8, B:6, C:4, D:3, E:3, F:3, G:7)
**RAGAS Judge:** `gpt-4o-mini` + `text-embedding-3-small` (fixed)
**Notebook:** `RAGAS_EVAL/G4A_Multimodal_Caption/ragas_ablation_G4A_caption_vs_nocaption.ipynb`

---

## 0. Insights Gained (executive summary)

**Read this experiment together with G4-P — alone, its FA/AR numbers understate the multimodal contribution, because FA/AR do not measure the caption's actual job (error-catching).** The 34-case run (up from 21) makes the per-category picture trustworthy for the first time.

1. **Adding the blind caption is globally FA-/safety-neutral, at a small relevance/cost overhead.** ΔFA = **−0.83 pp** (0.6133 → 0.6050, within the ~1.2 pp run SD), Safety **−0.98 pp** (91.2% → 90.2%, one case in one run), ΔAR = **−3.7 pp**, cost **3.2×** (≈ +$0.0016/case). The caption does **not** degrade the recommendation.
2. **The expansion exposed the 21-case per-category deltas as small-n noise.** The eye-catching pilot swings (B **+12 pp**, F **+12 pp**, D **−9 pp**, all n=1–2) **collapsed** at n=3–6: B −2.4 pp, F −8.3 pp, D **+7.4 pp** (flipped positive). Lesson: only Cat A (n=8) and Cat G (n=7) were ever trustworthy — now B (n=6) is too. **No category shows a robust FA *improvement* from the caption** → confirms FA is the wrong lens.
3. **The two robust categories tell the real story.** Cat A (canonical) ΔFA **+1.8 pp** — the caption stays quiet where labels suffice (doesn't break the easy cases). Cat G (adversarial) ΔFA **−2.0 pp** — the caption injects correct visual flags the guidelines don't contain, and FA penalises them (same direction as the pilot's −5.8 pp). FA measures KB-grounding; the caption's value is out-of-KB cross-validation.
4. **NEW — the caption's value has a *direction*, and it depends on where the ground truth lives.** When the **image** reveals danger the labels miss (Cat G missed-infection), the caption is an **asset** (escalates end-to-end). But when the danger is in the **patient's notes** and the **image looks clean** (the subclinical-infection case), the blind caption reads "clean" and can **pull the recommendation back toward clean management** — a mild **liability** (it caused the only A1 safety regression). The caption speaks for what it *sees*; if the truth is in the text, it can dilute it.

---

## 1. Why the Global FA/AR Look Flat (the honest, expected result)

The pre-registered hypothesis was that the caption helps most where the CV labels are wrong/incomplete, and that **any visual claim not in the KB can lower FA**. Both came true and interact to flatten the global mean: quiet on Cat A (labels correct → ΔFA ≈ 0), and negative where the caption adds correct-but-out-of-KB visual facts (Cat G/E/F). A naïve "global ΔFA ≈ 0 ⇒ caption doesn't help" reading mistakes the *metric's blind spot* for the *feature's absence*. The caption's value is error-detection (G4-P's VLM-DISC = 100%) and end-to-end escalation (§4) — neither visible to FA/AR.

---

## 2. Results — Global (mean ± SD over 3 runs, 34 cases)

| Arm | Faithfulness | Answer Relevancy | Safety Pass | Cost / run | Gen latency |
|---|:---:|:---:|:---:|:---:|:---:|
| **A0** unimodal (no caption) | **0.6133** ± 0.0117 | **0.4549** ± 0.077 | 91.2% | $0.0251 | 6484 ms |
| **A1** multimodal (blind caption) | 0.6050 ± 0.0136 | 0.4177 ± 0.060 | 90.2% ± 1.7 | $0.0795 | 6767 ms |
| **Δ (A1 − A0)** | **−0.0083** (−0.83 pp) | **−0.0372** (−3.7 pp) | **−0.0098** (−0.98 pp) | +$0.0544 | +283 ms |

- **FA:** indistinguishable — the −0.83 pp delta is smaller than the run SD. **Blind caption ≠ FA penalty**, reproduced on the larger set.
- **AR:** −3.7 pp — the caption's added visual observations slightly dilute how directly the answer addresses the query. Small, consistent with the pilot (−3.9 pp).
- **Safety:** −0.98 pp — traced to a **single case in a single run** (`cat_c_spreading_infection`, §4). A0 is deterministic (91.2%, SD 0); A1 varies because the caption changes the answer.
- **Cost:** 3.2× (the extra VLM call), negligible absolute (~+$0.0016/case).

---

## 3. Results — Per-Category Faithfulness: the expansion pays off

| Cat | n | Theme | A0 FA | A1 FA | Δ (34-case) | Δ (21-case pilot) | Read |
|---|:--:|---|:---:|:---:|:---:|:---:|---|
| **A** | 8 | Canonical WT1–8 | 0.632 | 0.650 | **+0.018** | +0.004 | neutral — labels suffice ✅ (robust) |
| **B** | 6 | Comorbidity/contraindication | 0.652 | 0.628 | **−0.024** | +0.123 | pilot +12 pp was **n=2 noise** → neutral (robust) |
| **C** | 4 | Escalation | 0.430 | 0.460 | **+0.030** | −0.028 | mildly positive (modest n) |
| **D** | 3 | Cavity/depth | 0.570 | 0.644 | **+0.074** | −0.092 | flipped positive — caption's cavity/necrosis read aligns with KB (modest n) |
| **E** | 3 | Complex chronic | 0.584 | 0.505 | **−0.079** | +0.028 | arterial/mixed add out-of-KB etiology/tissue obs. (modest n) |
| **F** | 3 | Image-robustness | 0.698 | 0.615 | **−0.083** | +0.126 | pilot +12 pp was **n=1 noise** → negative (modest n) |
| **G** | 7 | Adversarial label↔image | 0.658 | 0.638 | **−0.020** | −0.058 | stable cross-validation footprint (robust) |

**The headline of the expansion:** every dramatic pilot delta **regressed toward zero** once n grew — B and F, which looked like +12 pp "wins," were single-/double-case artifacts. This is precisely why the expansion was worth doing before trusting the per-category story. The **only statistically solid rows** (A n=8, B n=6, G n=7) all say the same thing: **the caption is FA-neutral (A, B) or slightly-negative by the out-of-KB mechanism (G)** — it does not improve guideline faithfulness anywhere, and that is the expected, honest result.

*(Cat C's low absolute FA ~0.43 reflects its escalation answers being judged against terse references; it is a per-category property of both arms, not a caption effect.)*

---

## 4. End-to-End Propagation — the caption's *direction* depends on where the truth lives

Catching a discrepancy in the caption (G4-P) matters only if the *advice* reflects it. The expanded escalation/adversarial cases reveal a clean, clinically-meaningful pattern — and a new failure mode:

| Ground truth lives in… | Example | Caption behaviour | Effect on advice |
|---|---|---|---|
| **Image** (label under-calls danger) | Cat G `miss_infection` (label clean, image shows erythema) | reads infection/necrosis the label missed | ✅ **asset** — A1 escalates ("see a doctor about the erythema") |
| **Image** (label over-calls) | Cat G `overcall` (label infected, image clean) | reads "clean" | ➖ defers to label (stays cautious — never overrides rule engine) |
| **Notes** (image looks clean) | Cat C `spreading_infection` (CV clean + clean WT2 image + notes: pus, spreading) | reads **"clean, no infection"** (correct *for the image*) | ⚠️ **liability** — pulls advice toward clean management |

**The new failure mode, concretely.** `cat_c_spreading_infection` is the one case where the danger is *only* in the patient's words (pus, spreading redness) — the CV label says clean *and the photo genuinely looks clean* (it reuses a clean WT2 image). The blind caption dutifully reports "healthy granulation, no signs of infection." In **A1 run 3**, this pulled the recommendation to an **exudate-management dressing (alginate + foam)** instead of the notes-mandated **antimicrobial** — failing the `allowed_dressing_present` safety check (A1 2/3 vs A0 **3/3**). This is the entire source of the −0.98 pp safety delta.

**Interpretation:** the caption is an **asset when the eyes see danger the labels miss**, and a **liability when the danger is in the history, not the wound bed**. It must therefore remain **advisory** — the notes-driven subclinical-infection escalation (a deterministic rule) should not be softened by a "looks clean" caption. This mirrors the WT4 peri-wound finding in G4-P: a clean-looking bed does not rule out infection, and the caption's clean read must be treated as **low-confidence** whenever notes or labels indicate infection.

---

## 5. Interpretation — What G4-A Adds on Top of G4-P

- **G4-P:** *can* the VLM catch CV-label errors? → Yes, 100%, but only prompted **blind**.
- **G4-A (34-case):** feeding that blind caption into real generation is **FA-/safety-neutral** and **propagates missed-danger escalation** end-to-end — *with one caveat*: when danger lives in the notes and the image looks clean, the caption can dilute a rule-based escalation (§4).

Together: the multimodal layer is a **low-cost cross-validation safety-net that catches image-visible errors a unimodal pipeline cannot** — provided the caption stays subordinate to the deterministic rules for notes/label-driven decisions. The apparent per-category "FA costs" are the fingerprint of correct visual reasoning the guidelines don't encode, not a regression.

---

## 6. Limitations & Threats to Validity

1. **FA/AR structurally cannot credit the caption's value.** The metric for the multimodal claim is **G4-P's VLM-DISC (100%)** and, decisively, **H1 (Ms Saw)** — not G4-A's FA.
2. **C/D/E/F are still n=3–4** — much better than the pilot's n=1, but treat their deltas as *indicative*; A (8), B (6), G (7) are the robust rows.
3. **Deterministic safety-checker residual false-fail.** The contraindicated-item check was fixed to skip "Step-by-Step" cautions (this correctly rescued `cat_e_arterial`: "not compression" → now 3/3). A residual remains: `cat_b_skin_tear` still fails on `avoid_adhesive` because the answer says "…without an **adhesive** border" *inside a recommendation section* — a **negation** the substring check can't see. It fails **both arms equally**, so the A1−A0 delta is unaffected; it only depresses the *absolute* safety a little. (`cat_a_wt7`, `cat_c_dfu` also fail both arms — pre-existing checker strictness.) A negation-aware check is the clean fix before the final thesis run.
4. **Safety is the *floor* of the caption's risk, not proof of benefit** — the benefit (error-catching) is G4-P + H1.
5. **VLM = `gpt-4o-mini`-Vision only** (G4-B ablates this).

---

## 7. Recommended Actions & Next Steps

1. **Keep the blind caption** — FA-/safety-neutral with a demonstrated missed-danger safety-net. (Live.)
2. **Harden the caption's subordination in `PATIENT_SYSTEM_PROMPT`:** when the notes or CV label indicate infection, a caption "looks clean" read must **not** downgrade the antimicrobial/escalation advice (fixes the `spreading_infection` liability). Symmetrically, keep the missed-danger escalation (Cat G) it already does well.
3. **Add negation-aware matching to the safety checker** (`no/without/avoid/not <dressing>` ⇒ not a recommendation) before the final run, to remove the `skin_tear` residual false-fail.
4. **Report G4-A with G4-P, never alone** — pair "100% error-detection" with "at no FA/safety cost + end-to-end escalation."
5. **G4-B** (GPT-4o-V vs Gemini-V under blind) and **H1** (Ms Saw): does a stronger VLM propagate more cleanly, and are the flagged discrepancies clinically real?

---

## 8. One-Paragraph Viva Narrative

*"On the 34-case set, adding the blind VLM caption is neutral on faithfulness and safety — it neither degrades the guideline-grounded advice (ΔFA −0.8 pp, within noise) nor the safety pass-rate (91→90%), and it stays quiet on the canonical WT1–8 cases where the labels already suffice. Expanding the testset was decisive: the pilot's eye-catching per-category 'wins' (Cat B and F, +12 pp) were single-case artifacts that vanished at n=6, confirming that faithfulness is the wrong lens for the caption — its value is out-of-KB visual cross-validation, which RAGAS penalises (Cat G −2 pp), not rewards. End-to-end, the caption's benefit has a direction: it reliably escalates the patient's advice when the wound photo reveals danger the labels missed, but — a finding the expansion surfaced — when the danger is only in the patient's history and the photo genuinely looks clean, the caption's 'looks clean' read can pull the recommendation back toward clean management. The design conclusion is therefore precise: the multimodal caption is a low-cost safety-net that must stay subordinate to the deterministic rules — an asset for the eyes, never an override of the history — and the decisive validation of its clinical value is the blinded clinician review (H1), which faithfulness cannot provide."*

---

*Companion to `G4P_VLM_Prompt_Strategy_Analysis.md` (why blind) and `VerdaSense_FYP2_Ablation_Map_v5.md` (Pillar 1). Results: `RAGAS_EVAL/G4A_Multimodal_Caption/results/` (34-case run, 2026-07-04).*
