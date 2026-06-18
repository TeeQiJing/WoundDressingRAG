# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: VerdaSense

AI-degree FYP at Universiti Malaya. A RAG-based clinical decision support system for wound dressing recommendation grounded in the T.I.M.E. assessment framework (Tissue, Infection, Moisture, Edge). FYP1 (proposal + ablation) is complete; FYP2 (conversational RAG extension) is in planning — see `MDs/FYP2 Migration/`.

## Running the App

```bash
uvicorn wound_app_unimodal:app --reload
```

The app loads the BGE embedding model and ChromaDB on startup (~15–30 s). Requires a CUDA-capable GPU (falls back to CPU but is slow).

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

See `MDs/FYP2 Migration/VerdaSense_FYP2_Migration_Rationale.md` and `VerdaSense_FYP2_Comprehensive_Plan.md`. Key planned changes:

- Add conversational multi-turn RAG tab (Proposal B — highest priority)
- Fix `classify_wound()`: all locally infected wounds should set `referral_required=True`
- RCH metadata fix: add `population: paediatric` filter to retrieval
- Add debridement guidance to prompt for wound types 5–8 (necrotic burden > 30%)
- New ablation experiments: R6 (wound category metadata filter), R7 (conversation history retrieval), G4 (multi-turn), G5 (OOD abstention rate)
