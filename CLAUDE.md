# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: VerdaSense

AI-degree FYP at Universiti Malaya. A RAG-based clinical decision support system for wound dressing recommendation grounded in the T.I.M.E. assessment framework (Tissue, Infection, Moisture, Edge). FYP1 (proposal + ablation) is complete; FYP2 (multimodal RAG extension) is in **active development — a working multimodal prototype exists** (`wound_app_multimodal.py`). Authoritative plans: `MDs/FYP2 Migration/VerdaSense_FYP2_Master_Plan.md` (living master plan) + `VerdaSense_FYP2_Ablation_Map_v5.md` (eval plan) + `VerdaSense_FYP2_Testset_Construction_and_Review_Plan.md` (testset + Ms Saw review). See **FYP2 Current Status** at the bottom of this file.

## Running the App

```bash
uvicorn wound_app_unimodal:app --reload
```

The app loads the BGE embedding model and ChromaDB on startup (~15–30 s). Requires a CUDA-capable GPU (falls back to CPU but is slow).

**FYP2 multimodal app** (runs beside the unimodal one, port 8001):

```bash
uvicorn wound_app_multimodal:app --reload --port 8001
```

Same CV pipeline + manual I/M/E input; adds a **VLM caption** (GPT-4o-mini-Vision / Gemini-2.5-Flash-Vision) feeding generation, VLM-derived etiology + depth (shown in UI), a **patient-friendly output** with a **Dev/Prod toggle** (Dev = citations + evidence + caption internals; Prod = product gallery), **token-by-token SSE streaming** (`/get_recommendation_stream`), and a static **DyaMed product gallery**. Serves `templates/wound_index_multimodal.html`. Uses the **v5 KB** (below). Etiology + depth are **deferred from FYP2 scope** (kept in code, excluded from the ablation per supervisor).

## Environment Variables

Create `.env` with these keys (all required):

```
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=...
HUGGINGFACEHUB_API_TOKEN=...
WOUND_DB_PATH=...   # optional override for ChromaDB path
```

## Architecture

### Active System

**`wound_app_unimodal.py`** — sole production file. FastAPI app serving `templates/wound_index_unimodal.html` via Jinja2.

Pipeline per request (`POST /get_recommendation`):

1. **Normalise inputs** — `interpret_tissue_percentages()`, `normalize_infection/moisture/edge()` convert raw CV pipeline outputs to clinical labels.
2. **Clinical pre-classifier** — `classify_wound()` → wound type 1–8, etiology (burn/diabetic\_foot/skin\_tear/generic), `referral_required`, `antibiotic_required`. Rule-based, not ML.
3. **Multi-axis retrieval (R1-C × R2-A × R3-C)** — `retrieve_chunks_multiaxis()` fires three dense sub-queries:
   - Sub-query A: pinned wound-type algorithm chunk (ChromaDB metadata filter on `wound_type`)
   - Sub-query B: dressing mechanism query built from exudate/infection/tissue profile
   - Sub-query C: patient free-text notes (or narrative fill)
   - Returns top\_n=6 deduplicated chunks.
4. **Generation (G1-C)** — `generate_recommendation()` builds a structured human prompt that injects the binding algorithm chunk as Source 1, then remaining evidence. LLM follows the G1-C grounded system prompt with strict citation rules. Output has 9 fixed sections (Primary Dressing → Clinical Notes).
5. **Token counting & cost** — `_extract_tokens()` + `_compute_cost()` using `MODEL_REGISTRY` pricing.

### Embedding & Vector Store

- **Embedding model (R4-B winner):** `BAAI/bge-large-en-v1.5` via `HuggingFaceEmbeddings`
- **ChromaDB path:** `db_wound_care_v4_bge/` · collection `wound_care_v4_bge`
- **KB:** 138 manually curated chunks from 8 clinical guidelines (GP, WCM, AJGP, SFP, EWMA, ISTAP, ANZBA, RCH). Raw chunk JSONs are in `ingestion_output_ai/` (`*_kept.json`).
- **Chunk metadata fields:** `wound_type`, `wound_category`, `authority`, `year`, `guideline_type`, `raw_text`, `ai_summary`.

### LLM Selection (per request)

Three models in `MODEL_REGISTRY`: `gpt-4o-mini` (OpenAI), `gemini-2.5-flash` (Google), `qwen/qwen3.5-35b-a3b` (OpenRouter). Qwen gets `/no_think` prefix injected and `<think>…</think>` stripped post-generation.

## KB Ingestion

`ingestion_full_8KB.ipynb` — ingests all 8 clinical PDFs from `clinical_pdfs_v2/` using `unstructured`, 8 KB chunk target, MedEmbed embedding → writes `*_kept.json` to `ingestion_output_ai/` → loads into `db_wound_care_v3/` and `db_wound_care_v4/`.

`ingestion_R4_additional_models.ipynb` — same chunks re-embedded with BGE (`db_wound_care_v4_bge/`) and E5 (`db_wound_care_v4_e5/`) for the R4 ablation.

`ingestion_DYAMED_surgeon_images.ipynb` (FYP2) — transcribes the KK Sultan Ismail / DyaMed Biotech clinical noticeboard photos in `surgeon_images/` into a **9th KB source** → `ingestion_output_ai/DYAMED_clinical_protocol_kept.json` (22 chunks: 8 per-wound-type application protocols tagged `wound_type`, 9 product monographs, 3 dressing-selection trees, T.I.M.E.→product map, S&N rationale). Fills the application-protocol + product-name gaps. **Loaded into the v5 stores** (`db_wound_care_v5_bge` / `db_wound_care_v5_medembed`, 160 chunks / 9 sources) — the active FYP2 KB; product monographs carry the Part 14 `dressing_class` + `moh_category` bridge.

**Note on RCH source:** RCH chunks are paediatric-only. The FYP2 fix adds `"population": "paediatric"` metadata and a retrieval filter excluding RCH for adult patients.

## Evaluation

**Testset:** `ragas_testset/wound_testset_v3.json` — 32 curated cases, 5 categories (Cat A: WT1–8 canonical, Cat B: special etiologies, Cat C: escalation logic, Cat D: data edge cases, Cat E: complex chronic wounds). Each case has `time_payload`, `reference_contexts` (3 chunks), ground truth fields.

**RAGAS ablation notebooks** are in `RAGAS_EVAL/` subdirectories:
- `R1_Query_Strategy/`, `R2_Retrieval_Strategy/`, `R3_TopK_Strategy/`, `R4_EmbeddingModel_Strategy/`, `R5_Multimodal_Caption_Retrieval/`
- `G1_Prompt_Strategy/`, `G2_LLM_Comparison/`, `G2_LLM_Comparison_G1D/`, `G3_LLM_Comparison/`

Analysis write-ups for each experiment are in `MDs/Retrieval Ablation/` and `MDs/Generation Ablation/`.

RAGAS judge is always `gpt-4o-mini` + `text-embedding-3-small` (never changed). Each experiment runs 3 independent runs; mean ± SD reported.

## Ablation-Best Configuration (current system)

| Component | Winner | Setting |
|---|---|---|
| Query strategy | R1-C | Multi-axis sub-queries (A+B+C) |
| Retrieval method | R2-A | Dense only (`similarity_search`) |
| Top-K | R3-C | k = 6 |
| Embedding | R4-B | `BAAI/bge-large-en-v1.5` |
| Prompt | G1-C | Grounded system prompt |
| LLM | G2-D | User-selectable (gpt-4o-mini default) |

## What Is Legacy / Ignored

- `wound_app.py`, `wound_app_v2.py`, `wound_app_v3.py`, `wound_app_v4.py` — superseded
- `ragas_eval_00*/`, `ragas_eval_01*/`, `ragas_eval_02*/` — old evaluation runs, replaced by `RAGAS_EVAL/`
- `templates/wound_index.html`, `wound_index_v2.html`, `wound_index_v4.html`, `wound_index_v5.html` — superseded by `wound_index_unimodal.html`
- `db_wound_care_v2/`, `db_wound_care_v3/`, `db_wound_care_v4/`, `db_wound_care_v4_e5/` — older/R4-ablation-only stores

## FYP2 Plans

Authoritative source: `MDs/FYP2 Migration/VerdaSense_FYP2_Master_Plan.md` (consult before any FYP2 work). The older `VerdaSense_FYP2_Migration_Rationale.md` / `..._Comprehensive_Plan.md` are superseded background.

**Direction shift:** FYP2 is **multimodal RAG**, *not* conversational RAG (conversational was dropped). A VLM (GPT-4o-Vision / Gemini Vision; **no fine-tuning** — the task is captioning, not classification) directly observes the wound photo and produces a multi-aspect clinical caption that feeds the **generation stage only**.

**Why generation-only (R5 result):** Injecting captions into *retrieval* hurt it (R5-B: −6.6 pp CR, −18.75 pp HR@6) — guideline text and visual-appearance language are different semantic registers BGE-large can't bridge. So retrieval stays exactly as FYP1; the caption is a third input to the LLM alongside retrieved chunks + the T.I.M.E. payload, letting it cross-validate (and flag) CV-label errors.

**Two-layer hybrid (the "why RAG vs rules" answer):** Layer 1 (rules) — `classify_wound()` + Sub-query A pinned retrieval — decides the *dressing category* deterministically. Layer 2 (RAG evidence) — Sub-query B/C + VLM caption — supplies the *why / how / when / patient-specific* (mechanism, application steps, change frequency, allergies, comorbidities) that no rule table or zero-shot LLM provides. FA: 0.69 zero-shot → 0.81 grounded.

**Pain-point resolutions:** (1) multimodal as above; (2) hybrid framing above; (3) add `wound_depth` field (superficial/cavity, from VLM + optional patient self-report) → cavity-filling dressing forms; full etiology classification dropped (Ms Saw: dressings are T.I.M.E.-driven, not etiology-driven, except vascular) — reduced to a single DFU flag; (4) product gallery showing brands/images after a recommendation; (5) conversational RAG dropped.

**Concrete changes:**
- New 9th KB source (DyaMed/KKSI) — see `ingestion_DYAMED_surgeon_images.ipynb`; load into `db_wound_care_v4_bge/`, mapping `wound_type`/`wound_category`/`authority`/`guideline_type` into ChromaDB metadata.
- `classify_wound()` referral/antibiotic logic already matches the MOH algorithm (referral = WT6/7/8; antibiotic = WT3/4/7/8) — **no change needed**; the earlier "all locally infected → referral" fix was cancelled as clinically incorrect (over-refers). Borderline infection→referral is an image-dependent *advisory* judgment handled by the multimodal generation layer, not a hard rule. See master plan Part 12.
- RCH paediatric metadata fix (above).
- Add debridement guidance to the prompt for wound types 5–8.

**FYP2 ablations (FYP1 R1–R5/G1–G3 are fixed):** G1-E (clinical prompt fixes), G4-A (caption vs none), G4-B (GPT-4o-V vs Gemini-V), G4-C (wound-depth field), G4-D (DFU flag), R6 (depth metadata filter), **H1 (blinded clinical eval by Ms Saw — highest-priority deliverable)**. *(Post-supervisor: G4-C/G4-D/R6 — depth + etiology — are **deferred**; the revised eval plan is `VerdaSense_FYP2_Ablation_Map_v5.md`.)*

**Ablation status (2026-07-03):** **G4-P ✅** (added P4=blind; blind caught 100% of adversarial discrepancies vs 14–19% label-shown; winner P4-blind is live; = VLM-DISC 100%) and **G4-A ✅** (34-case: blind caption FA-/safety-neutral, ΔFA −0.8 pp, ΔSafety −1 pp; the pilot's per-category "wins" (B/F +12 pp) were small-n noise, collapsed at n=6 → FA can't credit the caption; **directionality finding**: caption is an asset when the image reveals danger the labels miss (Cat G), a liability when danger is in the notes + image looks clean (`spreading_infection` → caption "clean" pulled advice off the antimicrobial) → caption must stay advisory, never override notes/label escalation). Write-ups: `MDs/Generation Ablation/G4P_VLM_Prompt_Strategy_Analysis.md` + `G4A_Multimodal_Caption_Analysis.md`. **Read G4-A only with G4-P.** **G4-B ✅** (VLM comparison under blind prompt): **`gpt-4o-mini`-V wins decisively** — 0 refusals, 100% VLM-DISC, 86% tissue acc; **`gemini-2.5-flash` refused 41% of clinical images** (empty/`BlockedReason.OTHER`, concentrated on infected/necrotic/adversarial; clean Cat F = 0 refusals). A `safety_settings=BLOCK_NONE` test recovered 0/5 — the block is **non-configurable** on the Developer API → Gemini disqualified; keep gpt-4o-mini. `MDs/Generation Ablation/G4B_VLM_Comparison_Analysis.md`. **G4-C ✅** (open-source VLMs via OpenRouter, blind prompt, reasoning-off like G3): 4 arms — **Qwen2.5-VL-72B, Qwen3-VL-235B, Gemma-3-27B, Gemma-4-26B**. **All ~0% refusals** (vs Gemini 41%) → open models solve the refusal problem. **Key methodological finding: VLM-DISC is gameable** — Gemma-3 scores 100% VLM-DISC but by over-calling "Infected" on 95% of clean wounds (49% infection acc) → DISC must be read *with* non-adversarial accuracy. **Best open = Qwen2.5-VL-72B** (infection 76% > GPT's 73%, tissue 85%≈86%, 6× cheaper, self-hostable = data sovereignty), but lower DISC (71% vs 100%). Bigger≠better (Qwen3-VL-235B lost to Qwen2.5-VL-72B). GPT-4o-mini stays best single choice. `MDs/Generation Ablation/G4C_OpenSource_VLM_Analysis.md`. Still open: VLM-ACC, H1.

## FYP2 Current Status (20 Jul 2026)

**Supervisor decisions:** multimodal is the right move but **evaluation is the deliverable** (don't stack features); **etiology + wound-depth deferred**; patient-friendly output ✅; Dev mode is the evaluation mode (ablation ignores the product gallery); H1 + UAT with Ms Saw confirmed.

**Built (prototype):** `wound_app_multimodal.py` + `templates/wound_index_multimodal.html` — v5 BGE KB, VLM caption/etiology/depth, patient-friendly G1-F output, Dev/Prod toggle, multimodal On/Off A/B (= live G4-A), SSE streaming, DyaMed product gallery. Two generation guardrails baked into `PATIENT_SYSTEM_PROMPT`: **contraindication-consistency** (MOH algorithm overrides local protocol; no dressing in both a recommendation and "avoid") and **exudate-tier matching** (Flaminal Hydro↔Forte). See Master Plan **Part 17**.

**Active FYP2 stores:** `db_wound_care_v5_bge` (R4-B winner, used by the app) + `db_wound_care_v5_medembed` (twin). FYP1 used `db_wound_care_v4_bge`.

**Testset v5:** built by `ragas_testset/wound_testset_builder_v5.py` → `ragas_testset/wound_testset_v5.json` (**34 curated cases** — A:8 WT1–8 · B:6 comorbidity/contraindication · C:4 escalation · D:3 depth/cavity · E:3 complex-chronic · F:3 image-robustness · G:7 adversarial TIME↔image). Expanded from 21 (2026-07-03): B/C are note-driven (reuse curated images), D/E/F use new Gemini-validated images. Every new case's live-classifier referral/antibiotic matches gold; all 34 pass the end-to-end sanity run. Patient-friendly `reference` with `[S#]` cites, **ranked+graded `reference_contexts`** (MRR/NDCG), `conditional_contraindications` field. **Full image curation + Gemini-Pro cross-validation complete (2026-07-02)** — every case validated three ways (Claude read ↔ gold label ↔ Gemini blind read); all 21 images resolve, Cat A classifies cleanly WT1→WT8. Images in `ragas_testset/wound_images/` (14 distinct), sourced from the Kaggle wound-segmentation dataset `wound_images_dataset/` (fusc/medetec/wsnet). Rebuild view via `ragas_testset/build_testset_viewer.py` → `testset_viewer.html`. **Curation finding:** WT3/WT4 ("infected + low non-viable%") is intrinsically hard to photograph — clean granulating beds read as not-infected, visibly infected beds are slough-heavy (NV out of range); infection at these types is peri-wound/clinical, not bed-visible (supports keeping the CV/clinical infection label rather than overriding from the image). See memory `testset_v5_curation.md`.

**H1 package — BUILT, NOT YET SENT (blocker).** `ragas_testset/build_h1_review.py` → `ragas_testset/h1_review.html` (~1 MB, self-contained, images base64-embedded). Offline-capable clinician review: localStorage auto-save across sittings (key `verdasense_h1_review_v1`), sticky toolbar (reviewer name · X/34 progress · **Download my answers** JSON/CSV · **Clear all**), per-case and per-question ✕ clear buttons, radios + real comment fields with unique keys for machine fold-back. **Part 1** = the 8 MOH↔DyaMed conflict questions (merged in from the WhatsApp message); **Part 2** = the 34 cases (image + T.I.M.E. + AI caption + pre-filled gold, 5 decisions each: image suitable / caption accurate / dressing / antibiotic / referral, + debridement for WT5–8). **Delivery decision: email the single HTML file — do NOT deploy to HuggingFace/public hosting** (no backend to receive answers; hosting clinical images publicly is a needless ethics flag). She reviews → clicks Download → sends back one JSON. Session guide + WhatsApp text: `MDs/FYP2 Migration/H1_Review_Session_Guide.md`.
**Not yet built:** the H1 **fold-back script** (parse her JSON → concordance %, Cohen's κ per decision, diff of every case where she disagreed with gold). Build only once her answers arrive.

**Pending Ms Saw (sent via WhatsApp, no reply yet):** 5 inter-guideline KB conflicts (C1 carbon/Zorflex across WT1–7 vs MOH WT8-only; C2 Drawtex hydroconductive; C3 Drawtex vs Gauze&Gamgee secondary; C4 alginogel on dry WT7; C5 foam-secondary on dry WT3/7) + Q8 brand-scope (DyaMed-only vs include Aquacel Ag/Activon honey). See Master Plan **Part 17.3** / **Part 18**. *(These 8 are now also embedded as Part 1 of `h1_review.html`, so sending that file supersedes chasing the WhatsApp thread.)*

## Open Threads / Next Actions (as of 20 Jul 2026)

1. **Send `h1_review.html` to Ms Saw** — the single highest-priority action; H1 is the FYP2 deliverable and everything else is done without it.
2. **Build the H1 fold-back script** once her JSON returns (concordance, Cohen's κ — κ not raw %, per master-plan standards work).
3. **Deferred — Fix 1 Option B:** bump `classify_wound()` so a subclinical-infection note escalates the *wound type* (not just the antibiotic). Rule **3d** in `PATIENT_SYSTEM_PROMPT` (VLM advisory never de-escalates) already fixed the antibiotic path, but the dressing still resolves to WT2 on the `spreading_infection` adversarial case — a deeper grounding conflict. Option B would require **re-running G4-A**; user leaned Option A (prompt-only). Not done.
4. **VLM-ACC** — optional cheap single-arm caption-accuracy measurement on all imaged cases. Not run.
5. `git push origin main` — user pushes manually. **Git convention: commits must NOT carry a `Co-Authored-By: Claude` trailer** (user does not want Claude shown as a GitHub collaborator); author is `TeeQiJing <qijingtee1227@gmail.com>`.
6. Exploratory only (not to implement): `MDs/VerdaSense_Alternative_AI_Approaches.md` — agentic RAG, GraphRAG, LangGraph, DSPy etc. Verdict recorded there: **no shift needed**; current hybrid design is defensible.
