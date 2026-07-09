# VerdaSense Testset v5 — Construction Plan

**Student:** Tee Qi Jing (23004894) · Universiti Malaya
**Companion to:** `VerdaSense_FYP2_Master_Plan.md` (esp. Part 12 referral/antibiotic/debridement, Part 13 output design, Part 14 dressing-class bridge, Part 15 retrieval chain).
**Supersedes:** `ragas_testset/wound_testset_v3.json` (32 cases). Target v5: **~45–50 cases**.

---

## 0. Why v5 (what changed since v3)

| Driver | Consequence for the testset |
|---|---|
| KB expanded with DyaMed (9th source, 160 chunks) | `reference_contexts` must include DyaMed protocol/product chunks for type/product/step cases |
| Output schema redesigned (Part 13) | `reference` answers rewritten to the patient-friendly, cited schema |
| Dressing-class bridge (Part 14) | `allowed_dressings` now carries **type + example product** |
| Multimodal (VLM caption, G4) | new case categories with `image_ref`; adversarial discrepancy cases |
| IR metrics (MRR, NDCG) in retrieval ablation | `reference_contexts` must be **ranked + graded**, not a flat set |
| Referral logic corrected (Part 12) | `referral_required` ground truth re-aligned to the MOH algorithm |

**Build philosophy:** v5 is **manually curated** (like v3), not RAGAS-auto-generated. Auto-generation cannot produce reliable *ranked* gold contexts or clinically-correct referral/contraindication labels.

---

## 1. Case object schema (v5)

```jsonc
{
  "case_id": "A-WT4-01",
  "category": "A",                       // A–G (see §8)
  "wound_type": 4,                        // 1–8 (or null for pure-escalation cases)

  "user_input": {
    "time_payload": {
      "tissue":   {"granulation": 70, "slough": 12, "necrotic": 0, "epithelial": 18},
      "infection": "Locally infected",   // CV/IME-Net label (clinical string)
      "moisture":  "High exudate",
      "edge":      "Non-advancing"
    },
    "patient_notes": "Allergic to silver. Wound on shin, 3 weeks.",
    "demographics": {"diabetic": false, "age_group": "adult"},
    "wound_depth": "superficial",         // superficial | cavity  (Part: G4-C)
    "image_ref": "testset_images/A-WT4-01.jpg"   // only for multimodal cases (F/G); else null
  },

  "reference": { /* §3 — the gold patient-schema answer, cited */ },

  "reference_contexts": [ /* §4 — RANKED + GRADED gold chunks */
    {"rank": 1, "chunk_id": "d622ee9f4c9c", "grade": 3, "role": "algorithm_anchor", "why": "WT4 MOH treatment chunk (Sub-query A pin)"},
    {"rank": 2, "chunk_id": "<dyamed_wt4>",  "grade": 3, "role": "primary_protocol", "why": "DyaMed WT4 application protocol"},
    {"rank": 3, "chunk_id": "<flaminal_forte>", "grade": 2, "role": "primary_product", "why": "alginogel monograph (primary dressing)"},
    {"rank": 4, "chunk_id": "<drawtex>",     "grade": 2, "role": "secondary_product", "why": "hydroconductive absorbent (secondary)"},
    {"rank": 5, "chunk_id": "<silver_contra>", "grade": 1, "role": "contraindication", "why": "triggered by 'silver allergy' note"}
  ],

  "allowed_dressings":      [ /* §5 — type + example product */ ],
  "contraindicated_dressings": [ /* §5 */ ],

  "antibiotic_required": true,            // §6 (from Part 12 matrix)
  "referral_required":   false,           // §6
  "expected_change_frequency": {"Flaminal Forte": "EOD up to 4 days", "Drawtex": "3-4 days"},
  "escalation_flags_expected": ["watch for spreading infection"]   // red-flag section
}
```

---

## 2. `user_input` — how to construct

- **`time_payload`** mirrors the upstream CV pipeline output verbatim (percentages + the four normalised T.I.M.E. labels). This is the **only** input that drives retrieval (Part 15 Stage 3).
- **`patient_notes`** is where you encode the things rules cannot see: allergies, comorbidity, chronicity, anatomical site, dressing-adherence problems. These are the Sub-query C triggers — design them to exercise contraindication retrieval.
- **`demographics.diabetic`** → drives the DFU flag (G4-D).
- **`wound_depth`** (superficial/cavity) → drives cavity-dressing logic (G4-C).
- **`image_ref`** → only for multimodal categories F/G. For retrieval-only evaluation it is irrelevant (caption is generation-stage; see §6).

---

## 3. `reference` (big update — aligned to Part 13 + 14)

The gold answer follows the **Part 13 patient schema**, written **with `[Source N]` citations** (the patient view hides them; the eval view keeps them — see Part 13.1). Encode every section so generation metrics can score completeness:

```
1. Your wound: "Mostly healing tissue with some soft dead tissue (slough), moderate fluid, and early signs of infection." [S1]
2. Dressing type: Primary = alginate/alginogel (antimicrobial, high-absorbency); Secondary = absorbent foam/hydroconductive. [S1][S2]
3. Example products: Flaminal Forte (alginogel) [S3]; Drawtex (hydroconductive) [S4].
4. Avoid: silver dressings (you reported a silver allergy). [S5]
5. Change frequency: Flaminal Forte every other day, extend to 4 days as fluid settles [S3]; Drawtex every 3–4 days [S4].
6. Antibiotics: your wound shows infection signs — see a clinician for assessment (culture-guided). Do not self-medicate. [S1]
7. See a doctor? Not urgently required for this wound type, but review if it worsens. [S1]
8. Step-by-step: clean with saline → apply alginogel 0.5 cm → cover with absorbent secondary → secure. [S2]
9. ⚠️ Red flags: spreading redness, fever, increasing pain or smell → seek same-day care.
```

**Rules for writing `reference`:**
- **Dressing TYPE is rule-derived** (must match the WT algorithm) — never let the gold answer's type depend on a retrieved brand.
- **Products are quoted from the gold contexts** (the DyaMed chunks you list in `reference_contexts`). If a case has no product chunk in context, the gold answer shows **type only**.
- **Change frequency per product** is copied from the DyaMed monograph (Part 14).
- Keep citations `[S1..N]` indexed to the `reference_contexts` ranks.

---

## 4. `reference_contexts` WITH RANKING — the hard one (MRR / NDCG-ready)

This is the methodological crux. Flat presence (v3-style) supports only Hit-Rate/Context-Recall; **MRR and NDCG need a ranked, graded gold list.**

### 4.1 Graded-relevance rubric (for NDCG)

| Grade | Meaning | Typical chunk |
|---|---|---|
| **3 — Binding** | the answer is *wrong* without it | WT algorithm anchor (Sub-query A pin); the DyaMed WT protocol; the **primary** dressing chunk |
| **2 — Highly relevant** | needed for a *complete* plan | secondary dressing; mechanism/"why" chunk; the recommended product monograph; change-frequency source |
| **1 — Supporting** | improves quality / patient-specific | contraindication chunk (**only if** the case notes trigger it); debridement guidance (WT5–8); general T.I.M.E./assessment context |
| **0 — Not relevant** | everything else (implicit negatives) | — |

### 4.2 Ideal rank order (the gold ranking)

Sort by grade desc, then by **clinical priority tied to the retrieval architecture** (Part 15):

```
rank 1  WT algorithm anchor            (grade 3, Sub-query A — must be top; the system pins it)
rank 2  DyaMed WT protocol / primary   (grade 3)
rank 3  primary dressing product       (grade 2/3)
rank 4  secondary dressing             (grade 2)
rank 5  mechanism / change-frequency   (grade 2)
rank 6  contraindication (if triggered)(grade 1)
```

This ordered list **is** `reference_contexts`. Keep it to **3–6 chunks** (matches k=6).

### 4.3 How each metric consumes it (define thresholds once, document them)

| Metric | Uses | Definition for v5 |
|---|---|---|
| **Context Recall** (RAGAS) | the **set** of grade≥2 chunks | fraction of grade≥2 gold chunks present in retrieved top-k |
| **Context Precision** (RAGAS) | grades | fraction of retrieved chunks that are grade≥1, position-weighted |
| **Hit-Rate@k** | binary (grade≥2 = relevant) | 1 if any grade≥2 gold chunk in top-k |
| **MRR** | rank of first relevant | 1 / rank of the first retrieved grade≥3 chunk |
| **NDCG@k** | full grades + position | DCG of retrieved order vs IDCG of the gold order above |

> **Binary threshold:** grade ≥ 2 = "relevant" for Hit-Rate/MRR; full 0–3 grades for NDCG. State this in the eval script header so it's reproducible.

### 4.4 Honest caveat on ranking validity

Ranking ground truth is **partly subjective** — the single biggest validity threat in v5. Mitigations:
1. **Derive rank from the §4.1 rubric**, not gut feel — this makes it reproducible and defensible even solo.
2. **Ms Saw spot-check:** have her validate the **top-3** gold contexts for ~8 representative cases (cheap, ~15 min) → report this as light inter-annotator agreement. This is high-value credibility for the viva.
3. **Report the rubric** in the thesis appendix so examiners can reproduce the grading.

---

## 5. `allowed_dressings` & `contraindicated_dressings` (updated from DyaMed)

Express each as **generic type + example product** (Part 14 bridge). Ground `allowed` in the MOH WT algorithm; ground `contraindicated` in SFP/WCM contraindication chunks + clinical rules.

**Per-WT allowed (type → example product):**

| WT | Allowed dressing types (example DyaMed product) |
|---|---|
| 1 | Foam / film / hydrocolloid (RenoCare Thin) — *no* silver/charcoal |
| 2 | Foam, alginate/alginogel (Flaminal Hydro), hydrofibre, polymeric membrane |
| 3 | Tulle, hydrogel (Dermacyn Hydrogel), hydrocolloid, silver, iodine; + secondary |
| 4 | Alginate/alginogel (Flaminal Forte), foam, silver, hydrofibre, iodine; absorbent secondary (Drawtex) |
| 5 | Hydrogel (Dermacyn Hydrogel), hydrocolloid (RenoCare), polymeric membrane, honey |
| 6 | Alginate/alginogel (Flaminal Forte), foam (RenoFoam), polymeric membrane, hydrofibre |
| 7 | Silver, hydrogel, hydrocolloid, iodine, activated carbon (Zorflex LA) |
| 8 | Alginate, silver, hydrofibre, foam, charcoal/activated carbon (Zorflex), iodine |

**Contraindication set (trigger → avoid):** silver/iodine on a clean granulating wound (WT1); silver allergy → avoid silver; iodine in pregnancy/thyroid disease; alginate/alginogel on a dry wound; film on infected/high-exudate; honey + bee allergy; activated carbon dry-adherence (use Zorflex **LA** on dry wounds). Add the specific contraindication to each case where the notes/T.I.M.E. trigger it.

---

## 6. `antibiotic_required` & `referral_required` (from Part 12 matrix)

Set the **deterministic** ground truth from the MOH algorithm (matches `classify_wound()`):

| WT | antibiotic_required | referral_required |
|---|---|---|
| 1, 2, 5 | false* | false |
| 3, 4 | **true** (C&S) | false |
| 6 | false* | **true** |
| 7, 8 | **true** (C&S) | **true** |

\* WT2/WT6 = "may or may not" → default `false`; flip to `true` only when notes carry an infection trigger (subclinical). Escalation overrides: diabetic / sepsis-keyword / spreading-infection notes → `referral_required = true` (Part 12.4). Encode a few such **Cat C escalation** cases.

> The VLM "borderline infection → consider review" signal is an **advisory** in the output (§9), **not** a flip of the deterministic `referral_required` label. Keep the label algorithm-true; measure the advisory separately.

---

## 7. VLM caption — needed in the testset or not? (resolved)

**Short answer: NOT for retrieval; YES (as the image) for the generation/G4 cases.**

- **Retrieval metrics (CR, CP, HR, MRR, NDCG)** are driven by `time_payload` only — the caption is a *generation-stage* input (R5 proved it hurts retrieval). So `reference_contexts` and all IR metrics are **caption-independent**. You do **not** author gold captions for retrieval.
- **Generation metrics (FA, AR) + G4 ablation** need the caption. But for **G4-A** (caption vs no-caption) you compare the **same** case with and without — so you need the **image** (`image_ref`), not a hand-written "gold caption". The caption is generated at eval time.
- **Caption quality** is validated separately (VLM Caption Accuracy Rate + Ms Saw H1 Part B), not via a per-case gold caption.

**Therefore:** add `image_ref` only to categories **F/G**; do **not** add a `vlm_caption` field to the gold schema. This removes the hardest, most subjective part of the burden.

> Honest note: a frozen `reference_caption` is optional and only useful if you later want a reproducible (image-free) generation eval. Skip it for now — it is high-effort, low-return, and subjective.

---

## 8. Case categories (v5) + targets

| Cat | Description | Source | ~Count |
|---|---|---|---|
| **A** | WT1–8 canonical | v3 (re-pointed to DyaMed contexts) | 8–12 |
| **B** | Special etiologies (DFU, VLU, burn, skin tear) | v3 + 2–3 new | 6–8 |
| **C** | Escalation logic (referral/antibiotic/sepsis) | v3 + new | 5–6 |
| **D** | Data edge cases (ambiguous %, missing field) | v3 | 4 |
| **E** | Complex chronic wounds | v3 | 4 |
| **F** | **Multimodal** (image + T.I.M.E.) for G4-A/B | NEW (needs images) | 6–8 |
| **G** | **Adversarial T.I.M.E.–image discrepancy** (CV label wrong; VLM should flag) | NEW | 6–8 |
| (sub) | **Product/brand retrieval** + **cavity** (G4-C) + **DFU** (G4-D) | woven into A/B/F | — |

Keep the 32 v3 cases (re-pointed), add ~15–18 new → **~47–50 cases**.

---

## 9. Construction workflow

1. **Re-point v3 (Day 1):** for each of the 32, re-derive `referral_required` from Part 12; add DyaMed chunk(s) to `reference_contexts` with ranks/grades; rewrite `reference` to the Part 13 schema.
2. **Author the ranking (Day 2–3):** apply the §4.1 rubric to every case; this is the slow part — do it source-by-source per WT.
3. **Add FYP2 cases (Day 3–4):** F (multimodal — source 1–3 images per WT from public datasets AZH/WSNet/Medetec or your R5 archetypes), G (craft `time_payload` that contradicts the image), cavity, DFU.
4. **Validate (Day 4):** script-check every `chunk_id` exists in the v5 store; assert `reference_contexts` grades/ranks well-formed; assert antibiotic/referral match the Part 12 matrix.
5. **Ms Saw spot-check (async):** top-3 contexts for ~8 cases + 4 reference answers (folds into H1).

---

## 10. Metric → field map (what each field powers)

| Field | Powers |
|---|---|
| `time_payload` | retrieval query (R-series), `classify_wound` inputs |
| `reference_contexts` (ranked+graded) | CR, CP, HR@k, **MRR, NDCG** (R-series) |
| `reference` (cited) | Faithfulness, Answer-Relevance (G-series) |
| `allowed/contraindicated_dressings` | dressing-correctness + contraindication-avoidance (Safety) |
| `antibiotic_required`, `referral_required` | referral/antibiotic accuracy (Safety) |
| `image_ref` (F/G) | G4-A/B caption ablation; G adversarial discrepancy detection |
| `wound_depth`, `demographics.diabetic` | G4-C (cavity), G4-D (DFU) |

---

## 11. Honest interpretation (read before you build)

- **The testset is the linchpin.** Garbage gold → meaningless metrics. The **ranked `reference_contexts`** is both the hardest and the most decision-relevant artifact (it gates MRR/NDCG, your retrieval-ablation headline metrics). Invest there; lean on the rubric for reproducibility; get Ms Saw's top-3 spot-check for credibility.
- **Caption is the highest-risk, highest-novelty contribution — and it must be *measured*, not assumed.** The honest framing is *cross-validation*: the VLM earns its place on **Cat G** (catching CV-label errors) and **cavity/periwound** observation, not on canonical Cat A cases. If G4-A shows no FA gain on Cat A but a real gain on Cat G + cavity → that is a clean, publishable, *honest* result ("the caption helps where the structured labels are wrong or incomplete").
- **KB cleanup is an ablation, not a prerequisite** (Master Plan Part 16). Build the testset on **v5-full**; let the retrieved-but-never-relevant analysis tell you what (if anything) to prune. Do not delete sources before you can measure the effect — and do not let pruning silently drop your Cat B gold contexts.
- **Don't over-scope v5.** ~50 well-curated cases with accurate rankings beat 150 noisy ones. The marginal viva value is in *correct rankings + a few sharp adversarial cases*, not volume.

---

---

## 12. Build Progress (status · last updated Jun 2026)

> Built by **`ragas_testset/wound_testset_builder_v5.py`** → `wound_testset_v5.json` (+ `.csv`). 15 curated cases so far. **Target revised down** to ~24–30 (from ~45–50): post-supervisor, the brief is *evaluation quality over volume*, and **etiology + wound-depth cases are deferred** (G4-C/G4-D/R6 dropped from the active map). The depth case (`cat_d_cavity_wt2`) and VLU case stay in the file but aren't scored for depth/etiology.

### 12.1 Builder corrections applied
- **WT2 → Flaminal Forte** (high-exudate tier; was Flaminal Hydro — the gold now matches the app's exudate-tier guardrail so they don't disagree).
- **New `conditional_contraindications` field** in `make_case()` — "iodine (if thyroid disorder)" moved out of *hard* `contraindicated_dressings` for WT3/4/7/8 (keeps the dressing-class metric clean: hard contraindications only). Hard ones (WT5/WT7 alginate-on-dry) stay; `cat_b_iodine_thyroid` keeps iodine as a *hard* contraindication (thyroid note present).

### 12.2 Cat A multimodal images — ALL LOCKED ✅ (VLM-spot-checked per case)
Each was pasted through `wound_app_multimodal.py` (Dev mode) and the VLM caption checked against the wound type before wiring `image_ref`. Images in `ragas_testset/wound_images/`; source = `wound_images_dataset/` (Kaggle wound-segmentation: **fusc** = diabetic foot, **medetec** = textbook clinical, **wsnet** = mixed/skin-tone-diverse).

| Case | image_ref | Case | image_ref |
|---|---|---|---|
| cat_a_wt1 | WT01_medetec_0021 | cat_a_wt5 | WT05_medetec_0065 |
| cat_a_wt2 | WT02_medetec_0116 | cat_a_wt6 | WT06_medetec_0298 |
| cat_a_wt3 | WT03_wsnet_0096 | cat_a_wt7 | WT07_wsnet_0539 |
| cat_a_wt4 | WT04_wsnet_0816 | cat_a_wt8 | WT08_medetec_0175 |

(The old 4 KB WT01/WT03 placeholders were deleted. `WT05_wsnet_0384` is now an unused alternate.)

### 12.3 Image-selection rules learned (apply to the rest)
- **A still photo can't convey exudate level or reliably confirm not-infected/depth** → set moisture via the label; pick on the *visually reliable* axes (tissue, infection signs).
- **Selection flips by wound type:** WT1–4 (low non-viable) → want a *granulating bed + peri-wound erythema halo* (infection without a sloughy crater, which would push NV>25%). WT5–8 (high non-viable) → want the *sloughy/necrotic bed*; for WT5/WT6 (not infected) reject peri-wound erythema; for WT7/WT8 (infected) erythema is wanted.
- **Avoid multi-wound images** (confound YOLO/SAM + the single-case archetype).
- **VLM mis-reads necrotic beds as "cavity"** → expected; depth is deferred, keep `wound_depth` as designed.
- **Score VLM Caption Accuracy on tissue + infection primarily**; moisture/depth = low-confidence (don't penalise).

### 12.4 Still to do
1. **Special-case images** (paste-VLM-check → lock): `cat_c` DFU (fusc_0086), `cat_d` cavity (medetec_0095), `cat_e` VLU (medetec_0141 / 0142), `cat_g` adversarial (medetec_0066 repurposed).
2. **Expand Cat B–G** to ~24–30 (§8 categories; note F/G adversarial set → 8–10 for VLM-DISC).
3. **Apply Ms Saw's KB-conflict rulings** (C1–C5 + Q8, Master Plan Part 18.5) to `allowed_dressings` / examples once she replies.
4. **One-pass Ms Saw review** per `VerdaSense_FYP2_Testset_Construction_and_Review_Plan.md` → freeze `wound_testset_v5_GOLD.json`.

---

*VerdaSense Testset v5 Plan · companion to the FYP2 Master Plan · 2026.*
