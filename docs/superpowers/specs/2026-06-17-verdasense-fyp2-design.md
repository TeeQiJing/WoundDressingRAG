# VerdaSense FYP2 Design Spec

**Date:** 2026-06-17  
**Author:** Tee Qi Jing (23004894), Universiti Malaya  
**Status:** Validated — ready for implementation planning  
**Scope:** FYP2 migration from unimodal text-only RAG to multimodal RAG with product gallery and patient-friendly UI  

---

## 1. Context & Baseline

### 1.1 FYP1 System (Current State)

`wound_app_unimodal.py` is the sole active production file. FastAPI app, `uvicorn wound_app_unimodal:app --reload`.

**Pipeline (per request):**
1. `interpret_tissue_percentages()` + `normalize_*()` — convert raw CV outputs to T.I.M.E. labels
2. `classify_wound()` — rule-based, outputs wound type 1–8, etiology, `referral_required`, `antibiotic_required`
3. `retrieve_chunks_multiaxis()` — three dense sub-queries against ChromaDB; returns top-6 deduplicated chunks
4. `generate_recommendation()` — G1-C grounded prompt; 9-section structured output with source citations
5. Token count + cost via `MODEL_REGISTRY`

**Ablation winners (frozen — do not re-run):**

| Component | Winner |
|---|---|
| Query strategy | R1-C — multi-axis sub-queries A+B+C |
| Retrieval | R2-A — dense only |
| Top-K | R3-C — k=6 |
| Embedding | R4-B — `BAAI/bge-large-en-v1.5` → `db_wound_care_v4_bge/` |
| Prompt | G1-C — grounded system prompt |
| LLM | G2-D — Gemini 2.5 Flash (FA=0.8147, Safety=90.6%) |

**Testset:** `ragas_testset/wound_testset_v3.json` — 32 cases, 5 categories (Cat A–E).  
**RAGAS judge:** `gpt-4o-mini` + `text-embedding-3-small` — never change across experiments.

---

## 2. FYP2 Scope

### 2.1 In Scope

| Priority | Item | Type |
|---|---|---|
| P0 | G4 Multimodal RAG ablation | Primary research contribution |
| P0 | UI/UX patient-friendly redesign | Required (Ms Saw feedback) |
| P0 | Product dressing gallery | Required (Ms Saw requirement) |
| P1 | RCH metadata fix | Bug fix |
| P1 | Debridement prompt injection (WT5–8) | Bug fix |
| P2 | Wound category classification | Conditional — drop if no dataset by Week 3 |
| P3 | TTS/STT | Optional — only if time permits |

### 2.2 Dropped

- **Conversational multi-turn RAG** — requires new 75–150 case conversational testset, new per-turn RAGAS metrics, and 8–10 weeks of solo work. Moved to Future Work chapter.
- **R7 (conversation history retrieval)** — depends on conversational RAG.
- **G5 (OOD abstention) + Category F testset** — deferred to future work.

### 2.3 Unchanged from FYP1

- `classify_wound()` referral logic — `referral_required=False` for Types 3 & 4 is **correct**. Ms Saw confirmed referral depends on clinical photo, not automatic. Do not change.
- 32-case testset — valid as-is. `reference_contexts` are actual KB chunks. Do not reconstruct.
- ChromaDB path, collection name, embedding model — frozen.
- RAGAS judge configuration — frozen.

---

## 3. Primary Contribution: G4 Multimodal RAG

### 3.1 Research Question

> Does injecting VLLM-generated visual wound descriptions into the existing text-only RAG pipeline improve evidence-grounded dressing recommendation quality, and at which injection point(s) is the improvement greatest?

### 3.2 Architecture Change

**FYP1 input:** structured T.I.M.E. payload only  
**FYP2 input:** structured T.I.M.E. payload + optional wound image

```
[FYP1 — Unimodal]

T.I.M.E. payload
    → classify_wound()          [rule-based, unchanged]
    → retrieve_chunks_multiaxis()
        Sub-A: wound-type algorithm chunk (metadata filter)
        Sub-B: dressing mechanism query
        Sub-C: patient notes query
    → generate_recommendation() [G1-C grounded prompt]
    → 9-section output


[FYP2 — Multimodal, G4-D config]

T.I.M.E. payload + wound image
    → classify_wound()          [unchanged]
    → vllm_caption(image)       [NEW — Gemini 2.5 Flash Vision]
    → retrieve_chunks_multiaxis()
        Sub-A: wound-type algorithm chunk [unchanged]
        Sub-B: dressing mechanism query   [unchanged]
        Sub-C: patient notes + VLLM caption [ENRICHED]
    → generate_recommendation()
        assessment_text += VLLM visual description [ENRICHED]
    → output (patient-friendly language) [REDESIGNED]
```

### 3.3 VLLM Caption Generation

**Model:** Gemini 2.5 Flash Vision via Google API (`gemini-2.5-flash` with image input).  
**Prompt:** structured 4-axis prompt requesting output in the format:

```
TISSUE: [description]
INFECTION: [description]
MOISTURE: [description]
EDGE: [description]
ADDITIONAL: [size, periwound, depth]
```

**Latency:** ~8–13 s per image (acceptable; optional feature, user-triggered).  
**Caption storage:** generated per-request, not cached (images differ per patient).

### 3.4 Caption Injection Points

**Sub-query C enrichment (retrieval):**
```python
# FYP1
sub_c_query = patient_notes or build_narrative_query(wound_data)

# FYP2 (G4-B and G4-D configs)
sub_c_query = f"{patient_notes}\n\nVISUAL WOUND ANALYSIS:\n{vllm_caption}"
```

**assessment_text enrichment (generation):**
```python
# FYP1
assessment_text = format_time_labels(wound_data)

# FYP2 (G4-C and G4-D configs)
assessment_text = format_time_labels(wound_data) + \
    f"\n\nVISUAL WOUND ANALYSIS (AI-generated from photograph):\n{vllm_caption}"
```

### 3.5 Caption-T.I.M.E. Conflict Resolution

**Core design principle:** T.I.M.E. labels are predictions from companion CV models (tissue decomposition kMeans, IMEnet) that may be inaccurate. The primary motivation for multimodal RAG is to reduce sole reliance on these model predictions by grounding assessment in the original image. VLLM caption represents direct visual evidence; T.I.M.E. labels represent indirect model inference.

**Conflict handling strategy — dual-source injection with explicit disagreement flagging:**

Instead of silently preferring one source, inject both into the generation prompt and surface the disagreement explicitly:

```python
# In assessment_text (G4-C, G4-D):
assessment_text = (
    f"CV MODEL ASSESSMENT (T.I.M.E. labels from companion models — may contain prediction errors):\n"
    f"{format_time_labels(wound_data)}\n\n"
    f"VISUAL WOUND ANALYSIS (AI caption from actual photograph):\n"
    f"{vllm_caption}\n\n"
    f"{conflict_note(wound_data, vllm_caption)}"   # injected only when discrepancy detected
)

def conflict_note(wound_data, caption):
    # Heuristic: if T.I.M.E. says infected but caption says "no infection signs"
    if wound_data["infection"] in ("locally_infected", "systemic") and \
       "no visible infection" in caption.lower():
        return (
            "NOTE: The CV model predicts infection but the wound photograph does not show "
            "clear visible infection signs. Please weigh both sources and recommend conservatively."
        )
    return ""
```

The generation LLM (Gemini 2.5 Flash) then reasons over both sources with the G4 grounded prompt instructing it to:
1. Prefer VLLM visual evidence for tissue composition, moisture level, wound edge characterisation (these are directly observable in an image)
2. Prefer T.I.M.E. labels for clinical decisions that require context beyond the photograph (e.g. patient history, odour, pain — inputs the CV pipeline captures but the image does not)
3. When sources conflict on infection axis: **recommend conservatively** — treat as potentially infected and note the discrepancy in the output

**Safety-critical decisions (ABx, referral):** If *either* source suggests infection or referral trigger, `classify_wound()` output is preserved. These are not overridden by caption.

**Research angle:** G4 also tests whether VLLM visual assessment is a more reliable wound characteriser than companion CV model predictions, surfacing cases where they agree vs disagree as a finding.

### 3.6 G4 Ablation Design

5 configurations, 3 independent runs each, same RAGAS judge as FYP1:

| Config | Retrieval | Generation | What it isolates |
|---|---|---|---|
| G4-A | T.I.M.E. only (Sub-C = notes) | T.I.M.E. only | Baseline cross-check vs G2-D |
| G4-B | Sub-C = notes + caption | T.I.M.E. only | Retrieval-only benefit |
| G4-C | T.I.M.E. only | assessment_text + caption | Generation-only benefit |
| G4-D | Sub-C = notes + caption | assessment_text + caption | Combined — main result |
| G4-E *(optional)* | G4-D with GPT-4o as VLLM | G4-D with GPT-4o as VLLM | VLLM model quality |

### 3.7 G4 Wound Image Dataset

8 representative images, one per wound type, stored in `ragas_testset/wound_images/`:

| WT | Image file | Source | Use for |
|---|---|---|---|
| WT01 | `WT01_wound.jpeg` | — | Type 1: clean granulating, dry |
| WT02 | `WT02_medetec_0116.png` | Medetec | Type 2: granulating, high exudate |
| WT03 | `WT03_wound.jpeg` | — | Type 3: locally infected, low exudate |
| WT04 | `WT04_wsnet_0816.png` | WSNET | Type 4: locally infected, high exudate |
| WT05 | `WT05_wsnet_0384.png` | WSNET | Type 5: necrotic, dry |
| WT06 | `WT06_medetec_0298.png` | Medetec | Type 6: necrotic, high exudate |
| WT07 | `WT07_wsnet_0539.png` | WSNET | Type 7: infected + necrotic, dry |
| WT08 | `WT08_medetec_0175.png` | Medetec | Type 8: infected + necrotic, wet |

**Note:** `WT01_wound.jpeg` is only 4 KB. If VLLM produces a poor caption, swap to `WT01_wsnet_0053.png` (256 KB, also in the directory) before running G4 ablation.

**Image-to-testcase mapping:** each WT image is reused across all testset cases of that wound type. This is a known static caption limitation — document in thesis.

**Expert validation:** `MDs/FYP2 Migration/VerdaSense_G4_Clinical_Review_Form.docx` sent to Ms Saw for image suitability + caption accuracy review. Awaiting response.

### 3.8 G4 Evaluation Metrics

| Metric | Method |
|---|---|
| Faithfulness (FA) | RAGAS, same judge (gpt-4o-mini + text-embedding-3-small) |
| Answer Relevancy (AR) | RAGAS |
| Context Recall (CR) | RAGAS — does caption improve chunk retrieval? |
| Safety Pass Rate | Deterministic rule checker (unchanged from FYP1) |
| Caption Accuracy Rate | % of 8 captions Ms Saw rates ≥ "partially accurate" |
| Infection Detection Rate | % of WT03/04/07/08 captions that correctly flag infection |
| Clinical Concordance | % of G4-D outputs matching Ms Saw's dressing recommendations |

Report: mean ± SD across 3 runs for RAGAS metrics. Single-run for Safety, Caption, Concordance.

---

## 4. Product Dressing Gallery

### 4.1 Data Source

`dressing_products/wound_products_bigpharmacy.json` — 157 products scraped from Big Pharmacy Malaysia (bigpharmacy.com.my) via Shopify public JSON API. Fields per product:

```
product_id, title, vendor, dressing_types[], price_min_myr, 
primary_image_url, product_url, available
```

Dressing type distribution: silver (62), gauze (36), general_wound (30), crepe_bandage (20), hydrocolloid (8), film (8), hydrogel (7), low_adherent (7), alginate (3), iodine (3), foam (2), hydrofibre (1), charcoal (1).

### 4.2 Mapping Layer

New module `product_gallery.py` (or inline in `wound_app_unimodal.py`):

```python
# Deterministic mapping: wound type → recommended dressing types → products
WOUND_TYPE_DRESSING_MAP = {
    1: ["film", "hydrocolloid"],           # clean granulating, dry
    2: ["alginate", "hydrofibre"],          # granulating, high exudate
    3: ["silver"],                          # locally infected, low exudate
    4: ["silver", "alginate"],             # locally infected, high exudate
    5: ["hydrogel"],                        # necrotic, dry
    6: ["alginate", "hydrofibre"],          # necrotic, high exudate
    7: ["silver", "hydrogel"],             # infected + necrotic, dry
    8: ["silver", "alginate", "charcoal"], # infected + necrotic, wet
}

def get_products_for_wound_type(wound_type: int, max_per_type: int = MAX_PER_TYPE, max_total: int = MAX_GALLERY_TOTAL) -> list[dict]:
    dressing_types = WOUND_TYPE_DRESSING_MAP.get(wound_type, [])
    results = []
    for dtype in dressing_types:
        matches = [p for p in PRODUCTS if dtype in p["dressing_types"] and p["available"]]
        results.extend(matches[:max_per_type])
        if len(results) >= max_total:
            break
    return results[:max_total]
```

### 4.3 API Change

`POST /get_recommendation` response adds:
```json
{
  "recommendation": "...",
  "sources": [...],
  "product_gallery": [
    {
      "dressing_type": "silver",
      "product_name": "Biatain Ag Non-Adhesive Foam Dressing",
      "vendor": "Coloplast",
      "price_myr": 45.90,
      "image_url": "https://...",
      "product_url": "https://..."
    }
  ],
  "tokens": {...}
}
```

### 4.4 Frontend Change

New card below the recommendation result in `wound_index_unimodal.html`:

- Section title: "Where to Buy"
- Grid of product cards: image thumbnail, product name, vendor, price in MYR, "View on Big Pharmacy" link
- Grouped by dressing type (primary dressings first, secondary second)
- Only show `available: true` products

**Configurable constants (set in `wound_app_unimodal.py`):**
```python
MAX_PER_TYPE    = 2   # max products shown per dressing type
MAX_GALLERY_TOTAL = 6  # hard cap on total products shown
```
Both values are adjustable after Ms Saw reviews the UI without re-running any ablation.

**FYP evidence panel:** source citations in the text output are retained unchanged. The product gallery is additive.

---

## 5. UI/UX Redesign

### 5.1 Ms Saw's Requirements

| Change | Action |
|---|---|
| Remove "Rationale by T.I.M.E. Factor" section | Delete this section from the generation prompt and frontend renderer |
| Output must be short, concise, patient-friendly | Rewrite G1-C prompt for patient-facing language; no clinical jargon |
| Add dressing type description + image | Add brief plain-English description per dressing type alongside product gallery |
| Show dressing product images | Product gallery card includes `primary_image_url` |

### 5.2 Revised Output Structure

Reduce from 9 sections to 5–6 patient-facing sections:

```
1. What to Use         → primary dressing name + plain-English description
2. What to Use Next    → secondary dressing (if needed) 
3. How to Apply        → simple step-by-step (max 4 steps)
4. When to Change      → frequency in plain days (e.g. "every 2–3 days")
5. See a Doctor If...  → escalation signs in plain language
6. Important           → antibiotic note if applicable (plain language)
[Evidence Sources]     → retained for clinical transparency / FYP citation
[Where to Buy]         → product gallery card
```

### 5.3 Prompt Rewrite Constraints

The new G1-C patient-facing prompt must:
- Use ≤ Grade 8 reading level language
- Avoid terms: "exudate", "granulation tissue", "debridement", "contraindicated", "antimicrobial", "peri-wound"
- Use plain equivalents: "wound fluid", "healing tissue", "wound cleaning", "do not use", "antibacterial", "skin around the wound"
- Keep output ≤ 350 words (current outputs average ~600 words)
- Retain source citations as numbered references at the bottom (not inline)

This prompt change is a **new config (G1-D)**. Run a mini-evaluation on 3 testset cases (one clean: `cat_a_type1_dry`, one infected: `cat_a_type3_dry_infected`, one complex: `cat_a_type8_wet_infected_necrotic`) to verify Safety Pass Rate does not drop before deploying.

---

## 6. Bug Fixes

### 6.1 RCH Metadata Fix

**Problem:** RCH (Royal Children's Hospital) chunks are paediatric-only but retrieved for adult patients.  
**Fix:**
1. Re-ingest RCH chunks into `db_wound_care_v4_bge/` with metadata `"population": "paediatric"`
2. In `_dense_search()`, add metadata filter for adult patients: `where={"population": {"$ne": "paediatric"}}`
3. No testset change needed — existing cases all adult patients.

### 6.2 Debridement Prompt Injection (WT5–8)

**Problem:** VerdaSense does not mention debridement for wound types with necrotic burden > 30% (WT5–8), which the three clinical panels flagged.  
**Fix:** In `generate_recommendation()`, detect wound type ≥ 5 and inject a clinical note into the system prompt:

```python
if wound_type >= 5:
    system_addendum = (
        "\nIMPORTANT: This wound has significant non-viable tissue (necrotic/slough burden > 30%). "
        "Your recommendation MUST address wound bed preparation: include a recommendation to assess "
        "the need for debridement (autolytic via dressing, or refer for surgical/bedside debridement). "
        "This is required before or alongside primary dressing selection."
    )
```

---

## 7. Conditional: Wound Category Classification (R6)

**Trigger:** Ms Saw confirms labelled wound image dataset by Week 3 of FYP2.  
**Drop condition:** If no dataset confirmed by Week 3, remove from scope entirely.

**If proceeding:**
- Fine-tune EfficientNet-B0 or ViT-Small on labelled wound images (DFU, VLU, Pressure Ulcer, Burn, Skin Tear, Surgical, Abrasion)
- Output: `wound_category` label fed as additional metadata filter in Sub-query A
- New ablation R6: does `wound_category` metadata filter improve CR/FA for category-specific cases?
- Minimum dataset: 200–500 labelled images across 5–6 categories

---

## 8. Testset Additions

No new testset categories in FYP2. The 32-case `wound_testset_v3.json` is used as-is for G4 ablation.

Category F (OOD / abstention cases) and G5 abstention rate evaluation are **deferred to future work**.

---

## 9. Implementation Order

```
Phase 1 — Fixes & Foundations (Week 1–2)
  [ ] RCH metadata re-ingestion + retrieval filter
  [ ] Debridement prompt injection (WT5–8)
  [ ] G1-D patient-friendly prompt + mini safety check (3 cases)
  [ ] Decision gate: wound category dataset (confirm or drop)

Phase 2 — Product Gallery (Week 2–4)
  [ ] WOUND_TYPE_DRESSING_MAP + get_products_for_wound_type()
  [ ] /get_recommendation response extended with product_gallery
  [ ] Frontend "Where to Buy" card
  [ ] Dressing type description text (plain English, per type)

Phase 3 — G4 Multimodal Infrastructure (Week 3–5)
  [ ] vllm_caption() function (Gemini 2.5 Flash Vision)
  [ ] Sub-query C enrichment (G4-B)
  [ ] assessment_text enrichment (G4-C)
  [ ] Frontend: optional image upload field → caption display
  [ ] Finalise WT01–WT08 image set (replace WT01_wound.jpeg if needed)
  [ ] G4-A baseline notebook

Phase 4 — G4 Ablation (Week 5–8)
  [ ] G4-B notebook (retrieval-only)
  [ ] G4-C notebook (generation-only)
  [ ] G4-D notebook (combined — main result)
  [ ] G4-E notebook (GPT-4o VLLM, optional)
  [ ] Caption accuracy analysis (T.I.M.E. alignment, automated)

Phase 5 — Human Evaluation (Week 8–12)
  [ ] Ms Saw review form follow-up (image suitability + concordance)
  [ ] Clinical concordance rate computation (8 WT cases vs G4-D output)
  [ ] Infection detection rate analysis (WT03/04/07/08 captions vs T.I.M.E. infection labels)

Phase 6 — Report & Wrap-Up (Week 12–15)
  [ ] FYP2 report writing
  [ ] TTS/STT integration (optional, if time)
  [ ] Final demo prep
```

---

## 10. Key Constraints & Risks

| Risk | Mitigation |
|---|---|
| VLLM infection detection fails (WT03/04) | T.I.M.E. infection label takes priority. Document as finding: "early local infection not visually detectable by VLLM." |
| G4 shows no FA/Safety improvement over G2-D | Null result is acceptable for FYP. Framing: "VLLM caption enrichment did not significantly improve grounded metrics for structured T.I.M.E. inputs — suggests T.I.M.E. labels are already sufficient for retrieval." |
| Wound category dataset not secured | Hard drop at Week 3. G4 alone is sufficient FYP2 scope. |
| Ms Saw unavailable for G4 review | Review form designed for async WhatsApp completion. Even image suitability ratings alone are usable. |
| WT01_wound.jpeg too low-res for VLLM | Swap to WT01_wsnet_0053.png before G4 ablation. |

---

## 11. Files Created / Modified

| File | Status | Notes |
|---|---|---|
| `wound_app_unimodal.py` | Modify | G4 caption injection, product gallery endpoint, G1-D prompt, RCH filter, debridement addendum |
| `templates/wound_index_unimodal.html` | Modify | Remove T.I.M.E. rationale section, add image upload field, product gallery card |
| `dressing_products/wound_products_bigpharmacy.json` | Read-only | 157 products, already scraped |
| `ragas_testset/wound_testset_v3.json` | Read-only | Used as-is for G4 ablation; no new categories in FYP2 |
| `ragas_testset/wound_images/` | Read-only | 8 WT images for G4 |
| `RAGAS_EVAL/G4_Multimodal/` | Create | G4-A through G4-D/E notebooks |
| `MDs/FYP2 Migration/VerdaSense_G4_Clinical_Review_Form.docx` | Created | Sent to Ms Saw |
| `docs/VerdaSense_FYP2_Supervisor_Proposal.docx` | Created | For supervisor meeting |

---

*Spec written 2026-06-17. Supersedes FYP2_Migration_Rationale.md entries for conversational RAG as primary contribution.*
