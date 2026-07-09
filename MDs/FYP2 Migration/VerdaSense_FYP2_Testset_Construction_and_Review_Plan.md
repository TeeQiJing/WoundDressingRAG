# VerdaSense FYP2 — Golden Testset v5: Construction & One-Pass Clinical Review Plan

**Author:** Tee Qi Jing (23004894) · Universiti Malaya · June 2026
**Companion to:** `VerdaSense_FYP2_Ablation_Map_v5.md` (what the testset feeds) · `VerdaSense_G4_Clinical_Review_Form.docx` (the review artifact this redesigns).

---

## 0. The one idea that makes this tractable

**Ms Saw VALIDATES pre-filled answers — she does NOT author them.**

The current `G4_Clinical_Review_Form.docx` asks her to *write* her own clinical description, primary dressing, secondary dressing, etc. for every case. For 24+ cases that's hours of writing — which is exactly why she returned photos instead of the completed form last time. Flip it: you + AI draft the gold answer; she only **ticks Agree / Minor-fix / Disagree** and comments when she disagrees. This:
- cuts her effort ~5×,
- yields clean inter-rater agreement data (= your **Clinical Concordance Rate** metric directly), and
- makes it a realistic **single 30–45 min session**.

The second idea: **the testset has many fields, but Ms Saw's VIEW is tiny.** A big JSON is fine for the machine — she only ever sees the 5 clinically-decisive things per case.

---

## 1. Three layers — who owns / validates what

Split every field by **who is the authority for it**. This is what stops the testset from being "a massive thing Ms Saw must review."

| Layer | Fields | Built by | Ms Saw reviews? |
|---|---|---|---|
| **L1 — Case design (inputs)** | `case_id`, `category`, `wound_type_expected`, `time_payload` (T.I.M.E.), `user_input`, `demographics`, optional patient `notes`, `image_ref` | **You** (author the scenarios) | only implicitly (is the case realistic) |
| **L2 — Clinical ground truth** | `reference` (gold recommendation) → its **clinical core**: primary dressing, secondary dressing, antibiotic, referral, debridement, change-frequency; `allowed_dressings`, `contraindicated_dressings`, `antibiotic_required`, `referral_required`, `escalation_flags_expected` | **You + AI draft** (Claude ⨉ Gemini cross-check) | **YES — this is the ONLY layer she validates** |
| **L3 — Retrieval / technical gold** | `reference_contexts` + `reference_contexts_meta` (ranked chunk IDs, grade, role, why), `[S#]` citations, `example_products` brand mapping | **You + AI** (no clinician) | **No** — she never sees chunk rankings or citations |

> **Where does the VLM caption sit?** It is the **output of the VLM stage** and an **input to the generation stage** — so it is *neither* a gold input you author *nor* a gold answer. In the testset it is a **reproducible system artifact**: store a frozen `vlm_caption_snapshot` (with `vlm_model` + `vlm_prompt_version`) for (a) Ms Saw to rate accuracy, (b) the Caption-Accuracy metric, (c) reproducibility. The live ablation **regenerates** it from the image each run. Mark it clearly `non_gold`.

---

## 2. Your 11 fields, resolved (+ what's missing)

| # | Field | Layer | Notes |
|---|---|---|---|
| 1 | wound image (`image_ref`) | L1 | one per case, visually matched to the payload; record provenance (dataset, license) for the appendix |
| 2 | VLM caption | artifact | `vlm_caption_snapshot` — non-gold, regenerated live (see above) |
| 3 | T.I.M.E. input (`time_payload`) | L1 | you author |
| 4 | patient note (`notes`) | L1 | only for Cat B/C/E cases; Cat A has none |
| 5 | `reference` (gold answer) | L2 | the patient-friendly cited answer; Ms Saw validates its **clinical core**, not the prose/citations |
| 6 | `reference_contexts` (+ ranked `_meta`) | L3 | for CR/CP/MRR/NDCG; **you + AI rank**, Ms Saw never touches |
| 7 | `allowed_dressings` | L2 | algorithm-derived; confirmed by Ms Saw **once per wound type**, not per case |
| 8 | `contraindicated_dressings` | L2 | same — once per wound type + per-case note-driven additions |
| 9 | `antibiotic_required` | L2 | per case; Ms Saw ticks |
| 10 | `referral_required` | L2 | per case; Ms Saw ticks |
| 11 | change frequency / example products | L2/L3 | Ms Saw sanity-checks ranges once (from her own DyaMed material) |
| **+** | `escalation_flags_expected` | L2 | for discrepancy/escalation cases — what the system should flag |
| **+** | `expected_discrepancy` | L2 | adversarial cases only — what the VLM should catch |
| **+** | image provenance + `vlm_model`/`prompt_version` | meta | reproducibility / dissertation appendix |

**Nothing critical is missing.** The trap isn't a missing field — it's exposing L3 fields to Ms Saw. Don't.

---

## 3. Construction pipeline (you + AI)

```
STEP 1  Case matrix (you)            → ~24–27 scenarios across categories (Section 4)
STEP 2  Image assignment (you)       → 1 image/case, visually matched; provenance logged
STEP 3  AI ground-truth draft        → Claude AND Gemini independently draft, per case:
        (L2)                            reference (clinical core) + allowed/contra + abx/referral
                                        + change-freq. Cross-check → reconcile → FLAG residual
                                        conflicts for Ms Saw (e.g. Zorflex/charcoal, Drawtex/Gamgee)
STEP 4  Retrieval gold (you + AI)     → run v5 retrieval per case; hand-rank gold chunks with
        (L3)                            grade/role/why → reference_contexts_meta
STEP 5  Caption snapshot (pipeline)  → generate_vlm_caption(gpt-4o-mini) on each image → freeze
STEP 6  Assemble JSON + AUTO-GEN form → render the Ms Saw review form straight from the JSON
STEP 7  One-pass review (Section 5)  → fold her edits back → FREEZE golden testset v5
```

**Cross-validation rule (Step 3):** a case's L2 ground truth is "AI-provisional" only when **Claude and Gemini agree** on primary/secondary/antibiotic/referral. Where they disagree, that case is auto-escalated to a Ms Saw decision point (don't silently pick one). This makes her review *targeted* — she spends time where the AI was uncertain, not re-confirming the obvious.

---

## 4. Proposed case matrix (~24–27 cases, >20 target)

| Cat | Theme | n | Notes? | Images |
|---|---|---|---|---|
| **A** | Canonical WT1–8 (no notes) | 8 | no | matched archetype/case image · **done, cross-verified** |
| **B** | Note-driven contraindication / comorbidity | 6 | yes | silver allergy · iodine+thyroid · pregnancy (iodine/silver) · honey+bee allergy · warfarin/bleeding · fragile/skin-tear |
| **C** | Escalation logic | 4 | yes | diabetic foot → referral · spreading infection → abx · sepsis red-flag (bypass) · chronic non-healing |
| **D** | Data edge cases | 2 | mixed | conflicting inputs · extreme non-viable load |
| **E** | Complex chronic | 2 | yes | VLU (+compression caveat) · mixed-tissue (depth **noted but not scored** — deferred) |
| **G** | **Adversarial image↔label discrepancy** | 5–6 | maybe | image visibly infected + "Not infected" label · necrotic image + "granulating" label · etc. → powers **VLM-DISC** |

Total ≈ 27. The **Cat G** block doubles as the discrepancy testset for `VLM-DISC` and is cheap to make (reuse images, flip a label). Cat A is locked; everything else is Step 3 onward.

> Etiology and depth are **deferred** (supervisor): keep the cavity/VLU cases but **don't** build G4-C/R6 ground truth or ask Ms Saw to score depth.

### 4.1 Build status — image curation COMPLETE (2026-07-02) · EXPANDED to 34 (2026-07-03)

**34 cases built** (`wound_testset_v5.json`): **A=8** (WT1–8), **B=6** (comorbidity/contraindication), **C=4** (escalation), **D=3** (depth/cavity), **E=3** (complex-chronic), **F=3** (image-robustness), **G=7** (adversarial). Expanded from the original 21 to give per-category statistical power (G4-A showed B/C/D/E/F were n=1–2, "directional only"). B/C are **note-driven** (reuse curated images); D/E/F use **new Gemini-validated images**. New B: skin-tear/fragile, honey+bee-allergy, anticoagulant/warfarin, silver/sulfa-allergy. New C: spreading-infection (subclinical-abx via notes), sepsis red-flag (emergency bypass), chronic-nonhealing. New D: deep-cavity→NPWT, extreme-necrosis→debride. New E: arterial-ulcer→**no-compression**, mixed-tissue-chronic→TIME wound-bed-prep. New F: 2 clean superficial (varied skin tone/anatomy). Every new case's live `classify_wound` referral/antibiotic matches gold; all 34 pass the end-to-end sanity run. *(Two honest grounding notes: honey+bee-allergy and silver-allergy treat the allergy as a patient-reported fact — KB grounds the alternative antimicrobial, not the allergy contraindication itself; the arterial "no compression" corollary is the standard clinical inference from the KB's arterial-ulcer→revascularisation chunk.)*

**Original 21-case core (2026-07-02):** A=8, B=2, C=1, D=1 (cavity), E=1 (VLU), F=1, G=7 (adversarial). Every case underwent **three-way image validation**: Claude read ↔ gold label ↔ **Gemini-Pro blind read** (user pastes each `wound_images/` image into Gemini with a fixed blind-read prompt; results reconciled). All 21 images resolve; Cat A classifies cleanly WT1→WT8; live-classifier output matches each case's intended wound type.

Image swaps this pass: a_wt2→wsnet_0494, a_wt4→wsnet_0466 (infected+granulating), c_dfu→fusc_0902 (infected sloughy plantar), d_cavity→medetec_0373 (true sinus; old medetec_0095 was hypergranulation, not a cavity), f→medetec_0283 (was a duplicate of a_wt2), overcall_wt4→wsnet_0494, overcall_fusc renamed **overcall_clean**→medetec_0158. Tissue realism: a_wt3 0/20/80, a_wt6 0/65/35 (necrosis→slough). cat_d notes reworded to drop "deep" (a `_REFERRAL_TRIGGERS` word) so it stays ref=False.

**Curation finding (writeup-worthy):** WT3/WT4 — "infected + low non-viable%" — is intrinsically hard to photograph. Across 5 candidates, clean granulating beds read as *not infected* and visibly infected beds are *slough-heavy* (NV out of range). Infection at these types is **peri-wound/clinical, not bed-visible** — which supports the design choice to keep the CV/clinical infection label rather than override it from the VLM caption.

**Gemini caveat:** it is erythema-trigger-happy (over-calls "spreading infection" on peri-wound redness it itself flags as possibly stasis dermatitis) — discount its infection calls lacking purulence/swelling.

---

## 5. The one-pass review (redesigned)

### 5.1 What Ms Saw sees per case — exactly 5 decisions
Pre-fill everything; she taps a box and only writes when she disagrees:

1. **Image suitable for this wound type?**  ☐ Yes ☐ No (why)
2. **AI caption accurate?**  ☐ Accurate ☐ Minor errors (note) ☐ Misleading
3. **Dressing (primary + secondary) — pre-filled gold shown:**  ☐ Agree ☐ Minor fix ☐ Disagree (what instead)
4. **Antibiotic — pre-filled (Yes/No):**  ☐ Agree ☐ Disagree
5. **Referral — pre-filled (Yes/No):**  ☐ Agree ☐ Disagree
   + one optional free comment box.

That's it. **No "write your own description", no "write your own primary dressing".** Debridement appropriateness folds into the comment for WT5–8 only.

### 5.2 Things reviewed ONCE (not per case) — a 1-page "invariants" sheet
- The **8 wound-type → allowed/contraindicated dressing tables** (confirm the algorithm mapping once).
- The **DyaMed product ↔ class ↔ change-frequency** table (confirm once, from her own material).
- The **2 KB-conflict rulings** (Master Plan 17.3): Zorflex-LA-on-WT1 (charcoal?) · Drawtex vs Gauze&Gamgee for high-exudate WT2.

This removes the same dressing-list confirmation from repeating across 27 cases.

### 5.3 Format + logistics — make it genuinely one-pass
- **Primary format: Google Form** with each case's image embedded + pre-filled gold in the question text + checkbox answers. Mobile, tap-to-answer, auto-collected. (She didn't complete the docx last time — minimise friction.)
- **Run it as a live 30–45 min call** where you screen-share and tick together. Surgeons talk faster than they type; you capture her verbal calls into the form. This *guarantees completion* and surfaces nuance.
- **Batch by wound type** so her clinical "mode" stays fixed (all WT3/4 infected cases together, etc.).
- Send the **invariants sheet (5.2) 1–2 days before** so the live call is just per-case ticks.
- One session, hard-capped. If she's energetic, great; if not, you still got every clinically-decisive field.

### 5.4 What Ms Saw verifies vs does NOT

| She verifies | She does NOT touch |
|---|---|
| Image suitability per case | `reference_contexts` / chunk rankings |
| Caption accuracy per case | `[S#]` citations / faithfulness |
| Primary + secondary dressing (agree/fix) | the patient-friendly prose wording |
| Antibiotic indication (agree/disagree) | exact change-freq numbers (ranges only, once) |
| Referral indication (agree/disagree) | retrieval / NDCG / any metric |
| Debridement appropriateness (WT5–8) | example-product brand strings (confirm once) |
| Note-driven contraindication (Cat B) — did the note correctly change the rec? | |
| Discrepancy cases (Cat G) — does the image truly contradict the label, and is flagging the right action? | |
| The 2 KB-conflict rulings (once) | |

---

## 6. After the review → freeze
1. Fold her ticks/edits into the JSON: where she disagreed, update L2 gold; where she confirmed, mark `clinician_validated: true` with her note.
2. Record **agreement stats** (her Agree-rate per field) — that *is* the Clinical Concordance Rate baseline for H1.
3. Tag the file `wound_testset_v5_GOLD.json` and freeze. All ablations (Ablation Map B0…H1) run against this frozen file.

---

## 7. What I can build for you next (pick any)
- **(a) Case-matrix + ground-truth drafter** — generate the L2 gold for Cat B/C/D/E/G with Claude, structured to drop into the JSON, each case flagged where it needs the Gemini cross-check.
- **(b) Auto form generator** — a script that renders the Ms Saw review form (Google-Form CSV import *or* docx) **directly from the testset JSON**, so the form always matches the data (no more hand-built docx).
- **(c) Caption snapshot script** — run `generate_vlm_caption(gpt-4o-mini)` over every `image_ref` and write `vlm_caption_snapshot` back into the JSON.
- **(d) reference_contexts ranker helper** — for each case, pull top-k from v5 and pre-fill a ranking sheet for you to grade.

Recommended order: **(c) → (a) → (b)** (snapshots first so the drafts and form are consistent), with **(d)** alongside.

---

*The golden testset is the measuring stick for every FYP2 number — so the time spent here is the highest-leverage work in the project. Keep Ms Saw's surface tiny, pre-fill aggressively, and run it once.*
