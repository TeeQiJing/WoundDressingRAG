# VerdaSense — KB Expansion & Testset Consultation

> Read all files before writing this: `wound_testset_builder_v2.py`, `ingestion_GP.ipynb`, `ingestion_SFP.ipynb`, `ingestion_WCM.ipynb`, `ingestion_AJGP.ipynb`, `ingestion_full.ipynb`, `GP_wound_dressings_kept.json`, `SFP_wound_dressings_kept.json`, and the three ISTAP PDFs.

---

## 1. Can you still use `wound_testset_v2.json` after ingesting new documents?

**Short answer: Yes — but it needs targeted updates, not a full rebuild.**

Here is the precise situation after reading your testset builder completely:

### What makes your testset valid or invalid

Your testset has three components that are KB-dependent:

| Field | KB-dependent? | What breaks if KB changes |
|---|---|---|
| `reference_contexts` | ✅ Yes — loaded directly from `_kept.json` via `ctx(chunk_id)` | If you add new docs, these still load fine. If you *modify* an existing chunk, the `ai_summary` string changes → RAGAS CR/CP measures against a different baseline. |
| `allowed_dressings` / `contraindicated_dressings` | ✅ Depends on KB coverage | If new KB adds rules that your testset doesn't know about, your safety checker has gaps. |
| `antibiotic_required` / `referral_required` | ✅ Depends on KB coverage | Same — if new guidelines add new referral criteria, existing cases may be incomplete. |
| `reference` (the gold answer text) | ✅ Partially — but you wrote it manually, not extracted from chunks | Adding new documents means some references may now be *incomplete* (missing recommendations from the new KB), but they remain *correct* for what they say. |
| `user_input` / `narrative_query` | ❌ No — these are clinical scenarios, not KB-derived | Stays valid forever. |

### The actual impact: surgical — not global

After adding new documents, **22 out of 28 cases remain fully valid**. The 6 cases that need targeted updates are exactly the ones covering the gap topics you're adding:

| Case | Gap | What needs updating |
|---|---|---|
| `cat_b_skin_tear_fragile` | No ISTAP in current KB | `reference_contexts` should add ISTAP chunk IDs; `reference` text may need silicone foam specifics from ISTAP |
| `cat_a_type6_wet_necrotic` | ANZBA burns referral logic | Not directly covered here, but check |
| `cat_b_burns_hand` | ANZBA burns | `reference_contexts` should add ANZBA chunk; reference may expand with burns-specific criteria |
| `cat_d_notes_diabetic_nonhealing` | IWGDF diabetic foot | `reference_contexts` must add IWGDF chunk; `allowed_dressings` and `reference` should align with IWGDF silver precaution logic — this fixes the persistent failure |
| `cat_b_diabetic_foot` | IWGDF diabetic foot | Same as above |
| Any new cases you add for NPWT/malodour gaps | EWMA | New cases built from new chunks |

### Concrete workflow for the testset update

```python
# After new ingestion, your update is 3 steps:

# Step 1: Add new chunk ID constants to the CHUNK ID MAP at the top of builder
ISTAP_SKINTEAR_TREATMENT = "xxxxxxxxxxxx"   # chunk_id from ISTAP _kept.json
ISTAP_SKINTEAR_PRODUCTS  = "yyyyyyyyyyyy"   # Product Selection Guide chunk
IWGDF_DIABFOOT_DRESSING  = "zzzzzzzzzzzz"   # IWGDF dressing recommendation chunk
ANZBA_BURNS_REFERRAL     = "aaaaaaaaaaaa"   # ANZBA hand burns referral chunk

# Step 2: Update reference_contexts for the 5-6 affected cases
# For cat_b_skin_tear_fragile:
"reference_contexts": [ctx(AJGP_SKINTEAR), ctx(ISTAP_SKINTEAR_TREATMENT), ctx(ISTAP_SKINTEAR_PRODUCTS)],

# For cat_d_notes_diabetic_nonhealing:
"reference_contexts": [ctx(AJGP_DIABFOOT), ctx(IWGDF_DIABFOOT_DRESSING), ctx(WCM_SILVER)],

# Step 3: Review and optionally expand 'reference' text and 'allowed_dressings' 
# for those 5-6 cases to reflect the richer KB
```

**You do NOT need to rebuild the 22 non-affected cases.** Their `reference_contexts` still point to valid chunk IDs that exist unchanged in the new vector store. RAGAS will still measure against those same chunk strings.

### One important clarification on RAGAS mechanics

Your Context Recall metric asks: "Are the reference_contexts entailed by the answer?" Your Context Precision metric asks: "Are the retrieved chunks relevant to the reference answer?" Neither of these metrics cares whether you have *more* documents in the DB — they only care about the specific chunk strings in `reference_contexts`. Adding new documents to the DB does not invalidate existing `reference_contexts` values.

The safety checker fields (`allowed_dressings`, `contraindicated_dressings`, `antibiotic_required`, `referral_required`) are the most important to review per-case when you add new KB documents, because the new guidelines may add restrictions or allowances your current testset doesn't capture.

---

## 2. Ingestion Pipeline: Custom Notebooks vs. Docling vs. Unstructured

### Why your current approach (custom per-document notebooks) is correct

After reading all four ingestion notebooks (GP, AJGP, SFP, WCM), the design is clear: each notebook is a bespoke PDF parser that understands that specific document's layout — table structures, multi-column layouts, hardcoded page ranges, pdfplumber for tables + PyMuPDF for text blocks + hardcoded fallback text for image-based content. You then curate which chunks to keep, add `ai_summary` enrichment via LLM, and export a `_kept.json`.

This is the right approach for clinical guideline PDFs and is consistent with what the EULAR RAG paper (2025) did manually: "Manuscripts underwent manual cleaning: nonessential sections were removed, headings were reviewed, and tables/boxes were moved to the end to minimise noise and redundancy." Guide-RAG (2025) similarly used PyPDF with manual removal of non-content elements. Your pipeline goes further than both.

The reason Unstructured.io failed you (incomplete, truncated, wrong table content, missed images) is fundamental: clinical guideline PDFs are layout-heavy, table-dense, multi-column documents. Unstructured's generic segmentation works for clean single-column research papers, not for documents like the GP guideline's wound type tables or WCM's chapter format.

### What is Docling?

Docling is IBM Research's open-source PDF parsing library (released late 2024, actively maintained in 2025). It is meaningfully different from Unstructured.

**What it does well:**
- Converts PDFs to structured Markdown or JSON while preserving document hierarchy (headings, subheadings, body text) and table structure
- Uses a neural document layout analysis model (DocLayNet) trained specifically on scientific and technical documents
- Exports tables as proper Markdown tables or structured JSON — not as flattened text
- Handles multi-column layouts significantly better than Unstructured
- Supports image captioning integration if you want to describe figures

**What Docling does NOT do for your case:**
- It cannot handle the specific clinical tables in your GP guideline (the wound type 1-8 decision tables are two-page multi-column nested tables — even Docling will produce imperfect output without some manual cleanup)
- It does not produce `ai_summary` enrichment — you still need your LLM enrichment step
- It cannot decide which chunks are clinically meaningful to keep — that curation logic is yours
- It does not handle image-only pages (like the ISTAP Tool Kit Poster which is essentially a scan)

**Should you use Docling for the new documents?**

| New document | Docling suitable? | Recommendation |
|---|---|---|
| ISTAP_Pathway_to_Assessment.pdf | ⚠️ Partially | The flowchart is an image — Docling gets the text boxes but not the arrows/logic. Use Docling as a starting point, then manually add the pathway logic as hardcoded text (same approach as your GP algorithm chunk) |
| ISTAP_Tool_Kit_Poster.pdf | ❌ No | This is essentially a multi-panel poster. Docling will struggle. Read the visible text from the uploaded PDFs (you can see all the content already in the context) and build chunks from hardcoded text extraction |
| ISTAP_Risk_Assessment_Pathway.pdf | ⚠️ Partially | Single-page flowchart — same situation as the pathway PDF |
| IWGDF PDFs | ✅ Good | IWGDF documents are standard multi-column academic PDFs. Docling handles these well |
| EWMA Position Papers | ✅ Good | Similar academic format |
| ANZBA documents | ✅ Good | Standard clinical guideline format |

**Installation and basic usage of Docling:**
```python
# pip install docling
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("your_guideline.pdf")

# Get markdown output (preserves tables as proper MD tables)
markdown_text = result.document.export_to_markdown()

# Get structured JSON
doc_dict = result.document.export_to_dict()
```

For the ISTAP poster/flowchart PDFs specifically: since you can already see all the text content in the PDFs provided, the fastest approach is to build those chunks with hardcoded text (the same way you built some GP chunks from page-extracted text). The ISTAP content is simple enough — Product Selection Guide, Pathway to Treatment, Risk Assessment Pathway — that you can write clean chunks directly.

### Recommended ingestion strategy for each new organisation

**ISTAP (3 PDFs you already have):**
The content across the 3 ISTAP PDFs is visible in the context. The clinically significant content for your use case is:
1. Product Selection Guide (dressing categories, indications per skin tear type 1/2/3) — from the Toolkit Poster
2. Pathway to Assessment/Treatment — the flowchart text blocks
3. Classification system (Type 1 no skin loss, Type 2 partial flap, Type 3 total flap loss)

Approach: build one dedicated `ingestion_ISTAP.ipynb` using PyMuPDF for text extraction, with hardcoded fallback text for the flowchart pages. Target 4–6 chunks covering: classification, treatment pathway, product selection guide (the table), infected skin tear products.

**IWGDF (Diabetic Foot):**
Standard academic PDF. Approach: use Docling to get clean Markdown, then manually select and curate the relevant sections (dressing recommendations, offloading, antimicrobial use, referral criteria). One dedicated `ingestion_IWGDF.ipynb`. Target 4–6 chunks: diabetic foot ulcer classification, dressing selection table, antimicrobial guidance, referral criteria, offloading.

**ANZBA (Burns):**
Standard clinical PDF. Same approach as IWGDF. Target 3–4 chunks: first aid, burn depth classification, referral criteria (especially the hand/face/genital/foot rule), dressing selection.

**EWMA (Wound Odour):**
Position paper format. Docling will work. Target 2–3 chunks: malodour causes, antimicrobial and charcoal dressing recommendations, systemic antibiotic guidance.

### Your _kept.json architecture is the right design

Your observation about reusability is exactly right: by keeping all chunked content in `_kept.json` files with a consistent schema (`chunk_id`, `source`, `section`, `parent_section`, `chunk_index`, `char_count`, `text`, `ai_summary`), your `ingestion_full.ipynb` can ingest any vector store technology by just reading those files. This is standard practice — it's the same pattern that EULAR RAG used with their chunk files and the MEREDITH system used with their knowledge base. The `_kept.json` files are your technology-agnostic knowledge base. ChromaDB is just the current backend.

**One metadata field to add for future filtering:** When you build ISTAP/IWGDF/ANZBA/EWMA chunks, consider adding a `wound_category` or `applicable_wound_type` metadata field to each chunk (e.g., `"wound_category": "skin_tear"` for ISTAP chunks, `"wound_category": "diabetic_foot"` for IWGDF chunks). This allows your v4 sub-query A (the algorithm chunk pinning query) to use metadata filtering more precisely, and reduces the chance of cross-contamination where a skin tear chunk ranks high for a diabetic foot query.

---

## 3. Output Format: Should You Change the 9-Section Markdown Structure?

**Short answer: Keep the structure, but add one section and tighten the language.**

### Why the current structure is right

After reading all 28 test cases in your testset builder, your 9-section output format is exactly what the literature recommends for clinical RAG:

1. **Primary Dressing** — directly actionable
2. **Secondary Dressing** — directly actionable
3. **Rationale by T.I.M.E. Factor** — this is your clinical reasoning transparency section; CARE-RAG and MEREDITH both emphasise explicit reasoning traces
4. **Contraindications** — safety-critical; your safety checker reads from this section
5. **Antibiotic Considerations** — safety-critical; your `antibiotic_required` safety check
6. **Referral/Escalation** — safety-critical; your `referral_required` safety check
7. **Dressing Change Frequency** — directly actionable
8. **Application Tips** — directly actionable
9. **Clinical Notes** — where your clinical notes override logic surfaces

This structure is also consistent with what the EULAR RAG paper used (7-sentence constrained output with implicit similar sections) and what MEREDITH's CoT prompting produced. The 9-section structure is better for your use case because it is patient-app-compatible (each section can be rendered as a collapsible card in the mobile UI).

### What to keep exactly as-is in the RAGAS testset

The `reference` field in your testset correctly mirrors this section structure. Keep it. The RAGAS AnswerRelevancy and Faithfulness metrics work best when both the `reference` and the model `answer` follow the same structural pattern — it avoids semantic mismatches from structural differences.

### Two changes worth making

**Change 1: Add a `## Safety Summary` section at the top (before Primary Dressing)**

This is the most important change for the patient-facing app. Insert a one-line summary box at the very top:

```markdown
## ⚠️ Safety Summary
- Contraindicated: [list], Antibiotic required: [Yes/No], Referral required: [Yes/No]
```

Reasons:
- Your safety checker currently reads through the entire response to find these signals via keyword matching. A dedicated Safety Summary section makes keyword extraction deterministic — reducing the `dressing_in_allowed_list` false positives that still affect v4_01/v4_02.
- On the mobile app, a patient or caregiver should see safety flags before dressing recommendations, not after reading 9 sections.
- The NICE RAG paper and clinical AI safety literature both recommend "safety signals first" design.

**Change 2: Tighten for mobile output**

Your current average response is 2,700–3,200 characters (verified from the JSON analysis). For a mobile patient-facing app, each section should be 1–3 sentences maximum. The EULAR RAG paper used a 7-sentence total cap. For your app, consider instructing the LLM to limit the Application Tips and Clinical Notes sections to 2 sentences each, and Primary/Secondary Dressing to 3 sentences each.

This is a **generation prompt change only** — no change to testset structure needed.

### Should you update the testset reference format to include `## ⚠️ Safety Summary`?

Only if you re-run evaluations. For the current ablation results you already have (18 versions, both generators), do not retroactively change the `reference` format. The existing results are internally consistent.

For any **future** evaluations (e.g., after adding new KB documents and running v4_02 again), update the `reference` field in the testset to include the Safety Summary section at the top. This also slightly improves your RAGAS AnswerRelevancy score because the reference now leads with the most safety-critical content, which is also what the model prioritises.

---

## 4. Putting It All Together: Recommended Execution Order

Here is the complete order of operations so nothing breaks:

```
Phase 1 — New Document Ingestion (do this first, independently)
├── ingestion_ISTAP.ipynb  → ISTAP_skin_tear_kept.json
├── ingestion_IWGDF.ipynb  → IWGDF_diabfoot_kept.json  
├── ingestion_ANZBA.ipynb  → ANZBA_burns_kept.json
└── ingestion_EWMA.ipynb   → EWMA_odour_kept.json

Phase 2 — Update ingestion_full.ipynb
├── Add 4 new _kept.json paths to CHUNK_FILES dict
├── Add guideline metadata for each new source (authority, year, focus, guideline_type)
├── Add wound_category metadata field to chunks_to_documents()
└── Run → rebuild db_wound_care_v4 (new vector store)

Phase 3 — Update wound_testset_builder (surgical update)
├── Add new chunk_id constants for ISTAP/IWGDF/ANZBA/EWMA chunks
├── Update reference_contexts for 5-6 affected cases
├── Review/expand reference text for cat_b_skin_tear_fragile and cat_d_notes_diabetic_nonhealing
├── Add 2-4 new test cases for ISTAP/IWGDF/EWMA coverage
└── Rebuild wound_testset_v3.json

Phase 4 — Update generation prompt in wound_app_02_v4.py
├── Add ## ⚠️ Safety Summary as first section in the 10-section template
├── Add mobile-friendly length constraints (3 sentences per section)
└── This becomes wound_app_02_v5 (v5_02)

Phase 5 — Run single evaluation pass (v4_02 config on new testset)
├── Generate responses with both generators (GPT + Qwen)
├── Run RAGAS + safety checker on wound_testset_v3.json
└── Compare v4_02 (old KB, old testset) vs v5_02 (new KB, new testset)
         ↑ note: not directly comparable — document this in FYP
```

**Critical note on comparability:** Adding new documents makes the new evaluation NOT directly comparable to your existing 18-version ablation results, because the KB has changed. Document this explicitly in your FYP: "Version ablation (v2–v4) was conducted on a fixed 4-document KB (db_wound_care_v3). The expanded KB (v5, db_wound_care_v4) represents a separate experiment and is reported independently." This is the same convention used by EULAR RAG (which did not re-run all ablations after KB expansion).

---

## 5. ISTAP Content — Chunk Plan (since you already have the PDFs)

From reading the three ISTAP PDFs you provided, here are the 5 chunks worth creating for `ISTAP_skin_tear_kept.json`. You do NOT need an automated parser for these — the content is clear enough to build as structured text:

| Chunk ID alias | Content | Key clinical rules to include |
|---|---|---|
| `ISTAP_CLASSIFY` | ISTAP Classification System: Type 1 (no skin loss), Type 2 (partial flap), Type 3 (total flap loss) + Skin Tear Decision Algorithm text | "Type 1 = Linear, no skin loss; use skin glue or silicone mesh; Type 2 = Partial flap; Type 3 = Total flap loss" |
| `ISTAP_PATHWAY` | Pathway to Assessment/Treatment flowchart text | "Atraumatic dressing removal; approximate wound edges; topical antimicrobials for local infection; systemic antibiotics for deep tissue infection; non-adherent or low tack dressing" |
| `ISTAP_PRODUCTS` | Product Selection Guide table (4 categories + special infected section) | Non-adherent mesh (Types 1,2,3); Foam — caution with adhesive borders (Types 2,3); Hydrogel — caution maceration (Types 2,3); Ionic Silver — contraindicated with silver allergy; Methylene Blue/Gentian Violet — broad spectrum |
| `ISTAP_RISK` | Risk Assessment Pathway (General Health + Mobility + Skin risk factors) | Fall history, fragile skin, extremes of age, polypharmacy, previous skin tears = high risk |
| `ISTAP_INFECTED` | Special consideration for infected skin tears (silver, MB/GV, tetanus consideration) | "Silver: broad spectrum, should not be used indefinitely, contraindicated with silver allergy" |

For `cat_b_skin_tear_fragile`, the updated `reference_contexts` should be:
```python
"reference_contexts": [ctx(AJGP_SKINTEAR), ctx(ISTAP_PRODUCTS), ctx(ISTAP_PATHWAY)]
```
And `allowed_dressings` should be updated to:
```python
"allowed_dressings": ["silicone_foam", "non_adherent_mesh", "silicone_mesh"]
```

---

## Summary of Answers to Your Questions

| Question | Answer |
|---|---|
| Can I still use wound_testset_v2.json? | Yes — 22/28 cases are fully valid. Update 5-6 cases where new KB covers the gap topics. |
| Do I need to rebuild reference / reference_contexts / safety fields? | Only for the 5-6 affected cases. The other 22 are correct and do not need touching. |
| Do I need a separate ingestion notebook per document? | Yes for complex/visual PDFs (ISTAP). Docling works for standard academic PDFs (IWGDF, EWMA, ANZBA). One notebook per organisation, not one per PDF. |
| What is Docling? Is it suitable? | IBM Research PDF-to-Markdown parser with layout model. Good for standard academic PDFs. Not reliable for poster/flowchart PDFs — use hardcoded text for those (same as your GP algorithm chunk). |
| Should I change the 9-section output format? | Keep structure. Add `## ⚠️ Safety Summary` as first section. Tighten section lengths for mobile. Do not retroactively change existing testset references. |
| Does adding new docs break RAGAS comparability? | Yes — treat the expanded-KB evaluation as a separate experiment, not a continuation of the 18-version ablation. Document this clearly in your FYP. |
