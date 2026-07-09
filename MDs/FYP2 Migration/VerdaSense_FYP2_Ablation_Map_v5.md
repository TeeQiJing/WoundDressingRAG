# VerdaSense FYP2 — Revised Multimodal Ablation Map (v5 KB + v5 Testset)

**Author:** Tee Qi Jing (23004894) · Universiti Malaya
**Date:** June 2026 · post supervisor meeting
**Supersedes (for FYP2):** the ablation framing in Master Plan Part 5. FYP1 (R1–R5, G1–G3 on v3 testset + v4 KB) stays as the historical baseline.

---

## 0. What the supervisor locked in

- **Multimodal RAG is the right FYP2 move — but the deliverable is the EVALUATION, not more development.** Do not stack features without measuring them.
- **In scope:** VLM ↔ CV cross-validation · patient-friendly output · H1 human clinical evaluation + UAT with Ms Saw.
- **Deferred (do NOT evaluate yet):** etiology classification, wound depth. (Code stays in the app but is excluded from the FYP2 ablation + testset for now.)
- **Dev mode = the evaluation mode.** All ablation runs use Dev output (cited, full evidence). The product gallery is a Prod-only UX surface and is **out of scope for ablation**.
- **Everything rebaselines on v5 KB + v5 testset.** The golden v5 KB is done (160 chunks, 9 sources, BGE). The golden v5 testset is **not finalised** (Cat A done; B–G pending) and needs Ms Saw's validation.

**Three evaluation pillars** (the supervisor's framing):
1. **VLM contribution + how to prompt the VLM** (the multimodal input).
2. **Retrieval** (does visual info help/hurt retrieval — re-run R5 on v5).
3. **Generation** (with/without VLM caption · prompt strategy).

---

## 1. Direct answers to your open questions

### Q1. Does the golden testset need a wound image per case, or keep it unimodal?
**It needs ONE curated wound image per case — but it stays a single unified testset.** Reasoning by use:

| Ablation type | Needs the image? | Why |
|---|---|---|
| Retrieval (R-series, R5) | image only for the *caption-injection* arms | retrieval query is text (T.I.M.E.→BGE); the caption that gets injected is generated from the image |
| Generation **without** caption / prompt-strategy (G1-E/F) | **No** | text-only (T.I.M.E. payload + retrieved chunks) |
| Generation **with** caption (G4) + VLM caption accuracy | **Yes** | the VLM must read a real photo for that case |
| H1 human eval | **Yes** | Ms Saw rates outputs/captions tied to a photo |

So: **add one `image_ref` per case** (the v5 schema already has the field — currently `null` for 13/15 cases). Text-only ablations simply ignore it. Do **not** maintain two testsets.

**Critical rigor point — "single image per case" must mean the image's visual T.I.M.E. matches that case's payload.** FYP1's R5 reused one image *per wound type* (8 images). For G4 + caption-accuracy to be meaningful, each case's image must visually correspond to its labels — otherwise "caption accuracy vs ground truth" is undefined. The exception is the **adversarial** cases, where the mismatch is deliberate (that's what `cat_g_label_says_clean_image_infected` already is).

### Q2. Regenerate the VLM captions with gpt-4o-mini instead of Gemini 2.5 Flash?
**Yes — but the deeper fix is: captions are NOT stored as fixed gold in the testset.** A caption is a *system output*, generated at run time by the pipeline under test, using `multimodal.py`'s `VLM_SYSTEM_PROMPT`. That's why your current `G4_Clinical_Review_Form.docx` captions feel "misaligned with the app" — they were made ad-hoc with Gemini and a different prompt, so they don't reflect what the system actually produces.

Concretely:
- For **G4-B** (GPT-4o-V vs Gemini-V) you generate captions with **both** models *through the pipeline* and compare — so you don't pre-pick one.
- For **Ms Saw's caption spot-check** and the review form, regenerate using the **production default (gpt-4o-mini) through `generate_vlm_caption()`** on the finalised per-case images. Then the form shows exactly what the app shows.
- The "caption is a bit misaligned with its T.I.M.E." observation is **not a bug to hide — it's the thing G4 measures** (VLM Caption Accuracy Rate). If it's badly off, first check it's a prompt issue (the VLM prompt already asks for `time_crossvalidation`), else it's a model-quality finding (→ G4-B).

### Q3. What to do with `VerdaSense_G4_Clinical_Review_Form.docx`?
Repopulate it **after** the testset images + captions are finalised: pipeline-generated captions (gpt-4o-mini) on the per-case images. Keep its structure — it becomes **H1 Part B (caption quality)**. But note: with Ms Saw now free, her highest-value task is **validating the golden testset reference answers**, which is more foundational than caption spot-checks.

---

## 2. Golden evaluation assets

### 2.1 Golden KB v5 — DONE
`db_wound_care_v5_bge` / `wound_care_v5_bge` — 160 chunks, 9 sources (AJGP, SFP, WCM, GP, DyaMed, ANZBA, ISTAP, RCH, EWMA). MedEmbed twin exists (`db_wound_care_v5_medembed`) but **BGE is the fixed embedder** (R4-B winner) — embedding choice is NOT re-ablated in FYP2.

### 2.2 Golden testset v5 — TO FINALISE (current: 15 cases)
Current state (verified): Cat A ×8 (WT1–8, cross-verified ✅), Cat B ×2, C ×1, D ×1 (cavity — **deferred**), E ×1, F ×1 (has image), G ×1 (adversarial, has image). Only 2/15 have an `image_ref`.

Each case carries (schema already supports this): `time_payload`, `user_input`, `demographics`, `reference` (patient-friendly gold answer with `[S#]`), `reference_contexts` (+ `_meta` with chunk_id/grade/role), `allowed_dressings`, `contraindicated_dressings`, `antibiotic_required`, `referral_required`, `expected_change_frequency`, `escalation_flags_expected`, **`image_ref`**.

**Finalisation actions:**
1. **Add `image_ref` to every case** (one image, visually matched to the payload). Cat A → the WT01–WT08 archetypes already in `ragas_testset/wound_images/`. Other cats → source 1 representative image each (AZH / WSNet / Medetec, as in R5).
2. **Cross-verify Cat B–G** (Claude+Gemini pre-pass like Cat A) → then **Ms Saw validates** the clinical content (dressing class, referral, antibiotic, change frequency) for **all** cases.
3. **Park the depth case** (`cat_d_cavity_wt2`) and any etiology-specific intent — deferred scope. Keep the case but don't use it for G4-C/R6.
4. **Build the adversarial discrepancy set → 8–10 cases** (currently only `cat_g`). Each = a case whose image visibly contradicts a T.I.M.E. label (e.g. visibly infected image + "Not infected" label). Reference answer: the system should *flag* the discrepancy. Powers **VLM-DISC**.

### 2.3 RAGAS judge (fixed, never changed)
`gpt-4o-mini` + `text-embedding-3-small`, as in FYP1. Every experiment = **3 independent runs, report mean ± SD.**

---

## 3. The revised ablation map

> Notation: B = baseline, R = retrieval, G4 = multimodal generation, VLM = caption-quality, H1 = human. Etiology (G4-D) and depth (G4-C, R6) are **deferred** and omitted.

### Phase 0 — v5 rebaseline (do first)

| ID | Question | Arms | Stage | Primary metric | Image? |
|---|---|---|---|---|---|
| **B0** | What are the v5 numbers for the FYP1-best config? | R1-C + Dense + k=6 + BGE + grounded prompt, on **v5 KB + v5 testset** | retrieval + gen | CR, CP, FA, AR (the FYP2 reference point) | no |

> Note: G-stage scores are **not comparable to FYP1** because the output schema changed (9-section → patient-friendly G1-F). State this explicitly; B0 is the new internal baseline.

### Pillar 2 — Retrieval (re-confirm R5 on v5)

| ID | Question | Arms | Primary metric | Image? | Priority |
|---|---|---|---|---|---|
| **R5-v5** | Does injecting the VLM caption into *retrieval* help or hurt on v5? | A: text-only retrieval (R1-C) · B: caption replaces Sub-query C · C: caption appended | CR, CP, **Hit-Rate@6**, MRR, NDCG (vs `reference_contexts`) | yes (caption arms) | **must** |
| **R-KB** *(optional)* | Does pruning measured-noise chunks improve precision without losing Cat B recall? | v5-full vs v5-pruned (Master Plan Part 16) | CP, CR | no | low |

> Expectation: R5-v5 should reproduce the FYP1 finding (caption-in-retrieval hurts) on the v5 KB — confirming the generation-only (Paradigm B) design with current data. If it doesn't, that's a publishable surprise.

### Pillar 1 — VLM contribution + how to prompt it (core FYP2 novelty)

| ID | Question | Arms | Primary metric | Image? | Priority |
|---|---|---|---|---|---|
| **G4-A** ✅ | Does the VLM caption improve generation vs no caption? *(the headline result + the live On/Off demo)* | A0: no caption (unimodal) · A1: + VLM caption | **FA, AR**, dressing-class correctness, Safety Pass Rate | yes (A1) | **must** |
| **G4-B** ✅ | Does the choice of VLM matter? | B1: GPT-4o-mini-V · B2: Gemini-2.5-Flash-V (blind prompt fixed) | Caption Accuracy, **VLM-DISC**, refusal rate, cost | yes | **must** |
| **G4-P** ✅ | **How should the VLM be prompted?** | P1: appearance-only · P2: cross-validation (old prod) · P3: terse · **P4: blind** (added) | FA + **Caption Accuracy** + Discrepancy Detection | yes | **high** |
| **VLM-ACC** ✅ | How often does the VLM's T.I.M.E. read match ground truth? | *satisfied by G4-B/G4-C accuracy columns* | **VLM Caption Accuracy** — GPT-4o-mini: infection 73% · tissue 86% (moisture/depth excluded as low-confidence, §18.4) | yes | **must** |
| **VLM-DISC** ✅ | Can the VLM catch deliberate CV-label errors? | adversarial set (7, Cat G) | **Discrepancy Detection Rate** | yes (mismatched) | **high** |

> G4-P is the experiment that directly answers your supervisor's "how should the VLM be prompted (multimodal input)?" — it's a genuinely novel sub-ablation and a strong viva point.

#### Pillar 1 — completed runs (2026-07-03, curated 21-case v5 testset · gpt-4o-mini VLM+gen · 3 runs)

**G4-P — VLM prompt strategy (✅ done).** Added a 4th variant **P4 = blind** (CV labels withheld from the VLM). Result: only the blind prompt cross-validates — **P4 caught 100% (21/21)** of adversarial discrepancies vs **P1/P3 14%, P2 19%**; the label-shown prompts *parrot* the labels (infection-axis anchoring ~98% — caught only 1/45). Blind's nominal infection "accuracy" is 78.6% (the label-shown 100% is label-echoing, not perception). **Blind costs nothing downstream: FA 0.629 ≈ best (P3 0.630) > P2 0.622; Safety identical 90.5%.** Winner = **P4 blind** (now live in `wound_app_multimodal.py`). Doubles as **VLM-DISC = 100%**. Full write-up: `MDs/Generation Ablation/G4P_VLM_Prompt_Strategy_Analysis.md`.

**G4-A — caption vs no-caption (✅ done, under the blind caption; re-run on 34 cases 2026-07-04).** Global: **ΔFA −0.8 pp (within noise), ΔAR −3.7 pp, ΔSafety −1.0 pp, cost 3.2×** → blind caption is FA-/safety-neutral. **Expansion was decisive:** the 21-case per-category "wins" (Cat B/F **+12 pp**) were small-n artifacts (n=1–2) that collapsed at n=6 (B −2.4, F −8.3); robust rows A(8)+1.8, B(6)−2.4, G(7)−2.0 agree the caption improves FA **nowhere** → **FA is the wrong lens** (penalises out-of-KB visual cross-validation). **NEW directionality finding:** caption is an **asset** when the *image* reveals missed danger (Cat G → escalates), a **liability** when danger is in the *notes* + image looks clean (`spreading_infection` → "clean" read pulled advice off the mandated antimicrobial = the −1 pp safety drop) → caption must stay **advisory**, never override notes/label escalation. **Read with G4-P:** blind caption = 100% error-detection (G4-P) at no FA/safety cost (G4-A) = a clinical safety-net, bounded by the advisory rule. Full write-up: `MDs/Generation Ablation/G4A_Multimodal_Caption_Analysis.md`.

**G4-B — VLM comparison (✅ done, blind prompt fixed, 2026-07-04).** GPT-4o-mini-V (B1) vs Gemini-2.5-Flash-V (B2): **B1 wins on every axis** (0 refusals · 100% VLM-DISC · 73% inf / 86% tissue acc · $0.047/run · 3.4 s). **B2 refused 41% of clinical images** (empty/content-blocked, concentrated on infected/necrotic/adversarial; clean Cat F = 0 refusals) → VLM-DISC 47.6%, worse accuracy, +36% cost, +64% latency. **`safety_settings=BLOCK_NONE` recovered 0/5** — block is `BlockedReason.OTHER` (non-configurable on the Developer API). **Verdict: keep gpt-4o-mini; Gemini disqualified.** Deployment lesson: VLM content filters block graphic medical imagery — reliability on the input distribution is a first-class selection criterion. Full write-up: `MDs/Generation Ablation/G4B_VLM_Comparison_Analysis.md`.

**G4-C — open-source VLM comparison (✅ done, OpenRouter, blind prompt + reasoning-off, 2026-07-05).** 4 arms: Qwen2.5-VL-72B, Qwen3-VL-235B, Gemma-3-27B, Gemma-4-26B. **All ~0% refusals** (vs Gemini 41%) → open models solve the content-filter problem. **Methodological finding: VLM-DISC is gameable** — Gemma-3 = 100% DISC but by over-calling "Infected" on 95% of clean wounds (49% acc) → DISC must be read *with* accuracy. **Best open = Qwen2.5-VL-72B** (infection 76% > GPT's 73%, tissue 85%≈86%, 6× cheaper, self-hostable/data-sovereign; DISC 71% — calibrated not trigger-happy). Bigger≠better (Qwen3-VL-235B < Qwen2.5-VL-72B). GPT-4o-mini stays best single choice; Qwen2.5-VL-72B the recommended self-hostable alternative. Full write-up: `MDs/Generation Ablation/G4C_OpenSource_VLM_Analysis.md`.

**Still open in Pillar 1:** **H1** (Ms Saw clinical review — the decisive external check FA/AR can't provide). *(VLM-ACC ✅ satisfied by the G4-B/G4-C accuracy columns; moisture/depth excluded as low-confidence per §18.4.)* H1 package prepped: `ragas_testset/h1_review.html` + `MDs/FYP2 Migration/H1_Review_Session_Guide.md`.

### Pillar 3 — Generation prompt strategy (VLM held constant)

| ID | Question | Arms | Primary metric | Image? | Priority |
|---|---|---|---|---|---|
| **G1-F** | Patient-friendly schema vs old 9-section? | old structured output vs new patient-friendly schema (both cited internally) | FA + conciseness/readability + H1 ratings | no | **must** |
| **G1-E** | Do the clinical prompt fixes help? | G1-C baseline vs G1-C + debridement (WT5–8) + the Part 17 guardrails (contraindication-consistency, exudate-tier) + sepsis gate | FA + **Safety Pass Rate** on debridement/contraindication cases | no | **high** |

### Human evaluation

| ID | Question | What Ms Saw does | Metric | Priority |
|---|---|---|---|---|
| **H1** | Is the output clinically correct, and is multimodal better than unimodal? | (A) **validate the golden testset** reference answers; (B) blinded Likert on multimodal vs unimodal outputs; (C) caption spot-check; (D) debridement completeness; (E) **KB-conflict rulings** (Master Plan 17.3); + **UAT** usability | Clinical Concordance Rate · multimodal-vs-unimodal preference · caption quality | **highest** |

---

## 4. Metrics catalogue

**Retrieval:** Context Recall (CR), Context Precision (CP), Hit-Rate@6, MRR, NDCG — all vs each case's `reference_contexts`.
**Generation:** Faithfulness (FA), Answer Relevance (AR); **VerdaSense-specific:** Safety Pass Rate, referral-correctness, antibiotic-correctness, **dressing-class correctness** (recommended classes ⊆ `allowed_dressings` and ∩ `contraindicated_dressings` = ∅), change-frequency faithfulness; for G1-F a **conciseness/readability** measure.
**VLM-specific:** VLM Caption Accuracy Rate (caption's T.I.M.E. axes vs ground-truth labels, within ±1 category); VLM-T.I.M.E. Discrepancy Detection Rate (adversarial).
**Human (H1):** Clinical Concordance Rate (% cases ≥4/5 clinical accuracy), multimodal-vs-unimodal preference delta, caption quality (Yes/Mostly/No), UAT usability.

> The dressing-class correctness metric is computable deterministically from the testset's `allowed_dressings` / `contraindicated_dressings` — it directly scores the Part 17 guardrails (e.g. it would have caught the Zorflex-on-WT1 contradiction).

---

## 5. Execution order (evaluation-first)

1. **Finalise the golden testset** (Section 2.2) — add per-case images, cross-verify Cat B–G, build the 8–10 adversarial set. *Blocks everything.*
2. **Ms Saw validates** the testset (H1-A) while you run the text-only ablations.
3. **B0** v5 rebaseline → **R5-v5** (retrieval) → **G1-F / G1-E** (text-only generation; no images needed).
4. **G4-A / G4-B / G4-P + VLM-ACC / VLM-DISC** (multimodal; needs finalised images).
5. **Regenerate the G4 review form** (gpt-4o-mini captions on final images) → **H1 B–E + UAT** with Ms Saw.
6. Write up: each experiment = 3 runs, mean ± SD, RAGAS judge fixed.

---

## 6. Decisions still to confirm

- **Adversarial set size:** 8 or 10 cases? (more = a stronger Discrepancy Detection Rate, but more images to source/validate).
- **Image sourcing for non-Cat-A cases:** reuse the 8 archetypes (simpler, replicates R5's limitation) vs source per-case images (stronger, more work). Recommendation: per-case where feasible, archetypes as fallback.
- **G4-P scope:** 3 VLM-prompt variants (recommended) vs 2, to keep the run count manageable.
- **R-KB:** include the pruning ablation this semester or list as future work.
- **Conciseness/readability metric:** which one (word count + a readability index vs an LLM-judge rubric) — pick one and fix it like the RAGAS judge.

---

*Companion to `VerdaSense_FYP2_Master_Plan.md` (see Part 5 ablation map and Part 17 prototype/guardrails/KB-conflict findings) and `VerdaSense_Paper_Analysis_and_FYP2_Meeting_Brief.md` (scope checklist §10).*
