# VerdaSense — How to Evaluate Your RAG Pipeline with RAGAS
### Complete Step-by-Step Guide for FYP

---

## Overview — What Are You Actually Evaluating?

Your RAG pipeline has **three stages** that can each fail in different ways:

```
Patient inputs T.I.M.E. + notes
        ↓
[RETRIEVAL]  → wound_app_v4 queries ChromaDB + BM25, reranks with CrossEncoder
        ↓
[GENERATION] → GPT-4o-mini reads retrieved chunks, writes clinical recommendation
        ↓
Patient receives dressing recommendation
```

RAGAS gives you **4 metric scores** that diagnose each stage:

| RAGAS Metric | What It Diagnoses | Stage |
|---|---|---|
| `context_precision` | Are the retrieved chunks actually about the right topic? | Retrieval quality |
| `context_recall` | Did retrieval find all the relevant guideline content? | Retrieval coverage |
| `faithfulness` | Is the answer grounded in evidence (not hallucinated)? | Generation safety |
| `answer_relevancy` | Does the answer actually address the wound question? | Generation quality |

---

## Your File Structure

Before starting, your project folder should look like this:

```
rag-for-beginners/
├── wound_app_v4.py                    ← your FastAPI RAG server
├── templates/
│   └── wound_index_v4.html
├── db_wound_care_v2/                  ← ChromaDB vector store
├── chunks_selection/
│   ├── AJGP_kept.json
│   ├── GarisPanduan_kept.json
│   ├── SFP_kept.json
│   └── WoundCare_kept.json
├── RAGAS/
│   └── verdasense_ragas_testset_generation.ipynb   ← generates test questions
├── wounds_ragas_evaluation.ipynb      ← THIS new notebook (evaluation)
└── ragas_eval/                        ← auto-created, stores outputs
    ├── testset_curated.json           ← produced by generation notebook
    ├── eval_results_raw.json          ← produced by evaluation notebook
    ├── eval_results_summary.csv
    └── eval_report.md
```

---

## PHASE 1 — Generate the Test Questions (if not done yet)

> Skip this phase if you already have `ragas_eval/testset_curated.json`

### Step 1.1 — Run the testset generation notebook

Open `RAGAS/verdasense_ragas_testset_generation.ipynb` and run:

- **Cell 2** — Imports
- **Cell 3** — Wrap LLM and Embedding model
- **Cell 4** — Load curated chunks from your `*_kept.json` files
- **Cell 6** — Skip (Cell 4b is the PDF fallback, already commented out)
- **Cell 5** — Define wound-care personas
- **Cell 6** — Define query distribution
- **Cell 7** — Build TestsetGenerator
- **Cell 8** — Generate testset (takes ~5–10 min, calls OpenAI)
- **Cell 9** — Inspect raw output
- **Cell 10** — Browse the generated questions
- **Cell 11** — Save to JSON and CSV

### Step 1.2 — Curate the questions

After Cell 10 prints all questions, read through them carefully.

**Remove questions that are:**
- Too generic: *"What is wound care?"*
- Not about dressings: *"Who published this guideline?"*
- Trivially obvious: *"Should infected wounds be treated?"*

Edit `SKIP_INDICES` in **Cell 12** (0-based index), then run it:
```python
SKIP_INDICES = [2, 5, 11]   # example — fill in after reviewing
```

This saves `testset_curated.json` and `testset_curated.csv` to `../ragas_eval/`.

**Target: keep 15–20 high-quality clinical questions.**

---

## PHASE 2 — Patch wound_app_v4.py (Recommended)

> This patch makes RAGAS context metrics accurate. Without it, `context_recall` will be underestimated.

### Step 2.1 — Add chunk_texts to the API response

Open `wound_app_v4.py` and find the Step 7 section in `/get_recommendation`:

```python
# ── Step 7: Build response ────────────────────────────────────────────────
sources = list(dict.fromkeys(
    chunk.metadata.get("source", "Unknown") for chunk in top_chunks
))

return JSONResponse({
    "result":           result,
    "sources":          sources,
    ...
```

Add `chunk_texts` before the `return`:

```python
# ── Step 7: Build response ────────────────────────────────────────────────
sources = list(dict.fromkeys(
    chunk.metadata.get("source", "Unknown") for chunk in top_chunks
))

chunk_texts = [chunk.page_content for chunk in top_chunks]   # ← ADD THIS

return JSONResponse({
    "result":           result,
    "sources":          sources,
    "chunk_texts":      chunk_texts,                           # ← ADD THIS
    ...
```

### Step 2.2 — Restart the server after patching

```bash
# Stop the running server (Ctrl+C), then restart:
uvicorn wound_app_v4:app --host 0.0.0.0 --port 8000
```

---

## PHASE 3 — Run the Evaluation Notebook

Open `wounds_ragas_evaluation.ipynb` and follow these steps:

### Step 3.1 — Cell 1 (Install)
Only needed once. Uncomment and run if you get import errors in Cell 2:
```python
!pip install ragas langchain_openai httpx pandas python-dotenv --break-system-packages
```

### Step 3.2 — Cell 2 (Imports)
Run this first in every new session. Verify the output shows:
```
✅ Imports OK
   OpenAI key : ✅ set
```
If the key is not set, add it to your `.env` file:
```
OPENAI_API_KEY=sk-...
```

### Step 3.3 — Cell 3 (Config)
Check `OUTPUT_DIR` and `RAG_SERVER_URL`. Defaults should work if your server is on port 8000.

### Step 3.4 — Cell 4 (Judge LLM)
This wraps GPT-4o-mini as the **judge** that scores your answers. It does **not** call your RAG server — it only evaluates the answers later.

### Step 3.5 — Cell 5 (Load testset)
Loads `testset_curated.json`. Should print:
```
✅ Loaded testset: 17 test cases
```

### Step 3.6 — Cell 6 (Define RAG caller)
This defines `call_rag_server()` which calls your `/get_recommendation` endpoint.

**Important — how it works:** Since your endpoint takes T.I.M.E. form fields (sliders), not raw text, this function:
1. Parses the question text to infer likely tissue %, infection, moisture, edge values
2. Passes the **full question as the `notes` field** — your `ClinicalSignalExtractor` then processes it exactly as it would a real clinical note
3. Returns the full JSON response including chunk_texts (if patched)

### Step 3.7 — Cell 7 (Sanity check) ⚠️ IMPORTANT
**Always run this before Cell 8.** It pings your server with one test question to confirm it's running and responding correctly. If you see:
```
❌ Cannot connect to RAG server at http://localhost:8000/get_recommendation
```
Start the server first:
```bash
uvicorn wound_app_v4:app --host 0.0.0.0 --port 8000
```

### Step 3.8 — Cell 8 (Main evaluation loop) ⏱ ~10–20 min
This is the main cell. It calls your RAG server once for every test question.

**Key features:**
- **Saves incrementally** — if it crashes at question 12, re-running resumes from question 13
- **Progress bar** shows confidence label and elapsed time per call
- Output goes to `eval_results_raw.json`

Expected output per question:
```
[ 1/17] SingleHopSpecific | Q: What dressing is recommended for a wound with...
        ✅ 18.3s | confidence=HIGH | contexts=6 [chunk_texts (enhanced mode)]
```

### Step 3.9 — Cell 9 (Patch reminder)
Informational — shows you the exact patch to apply to `wound_app_v4.py` if you haven't already.

### Step 3.10 — Cell 10 (Build RAGAS dataset)
Converts your raw evaluation records into the RAGAS `EvaluationDataset` format.

### Step 3.11 — Cell 11 (Run RAGAS) ⏱ ~3–5 min
Runs the judge LLM to score each answer. Costs approximately **$0.05–0.15 total** at gpt-4o-mini rates for 20 questions.

### Step 3.12 — Cell 12 (Save results)
Saves `eval_results_summary.csv` — open this in Excel to review per-question scores.

### Step 3.13 — Cell 13 (Generate report)
Saves `eval_report.md` — this is your polished report for your FYP supervisor. It includes:
- Aggregate scores with interpretation
- Per-query-type breakdown
- Dynamic recommendations based on your actual scores
- Limitations section

### Step 3.14 — Cell 14 (Plots)
Saves 3 charts to `ragas_eval/plots/`:
- `aggregate_scores.png` — bar chart of all 4 metrics
- `per_question_heatmap.png` — colour-coded grid of every question × every metric
- `by_synthesizer.png` — grouped bars comparing SingleHop vs MultiHop performance

### Step 3.15 — Cell 15 (Summary)
Final checklist confirming all output files exist, plus a scored interpretation table.

---

## Understanding Your Scores

### What scores to aim for (FYP context)

| Metric | ❌ Needs work | ⚠️ Acceptable | ✅ Good |
|---|---|---|---|
| context_precision | < 0.60 | 0.60–0.79 | ≥ 0.80 |
| context_recall | < 0.60 | 0.60–0.79 | ≥ 0.80 |
| faithfulness | < 0.70 | 0.70–0.84 | ≥ 0.85 |
| answer_relevancy | < 0.70 | 0.70–0.84 | ≥ 0.85 |

### If context_precision is low
Your retriever is returning irrelevant chunks. In `wound_app_v4.py`:
- Reduce BM25 weight from 0.4 to 0.3 in `EnsembleRetriever`
- Increase the reranker threshold in `rerank_with_moisture_boost`
- Increase `top_n` in `rerank_with_moisture_boost` then raise the threshold

### If context_recall is low
Your retrieval misses important guideline content. Try:
- Increase `k` in `build_hybrid_retriever` from 10 to 15
- Add more query variants to `expand_queries_v3`
- Check if your `*_kept.json` files have good coverage — you may have skipped too many chunks in `chunk_visualiser`

### If faithfulness is low
Your LLM is making things up (hallucinating). Try:
- Add "Only use information from the provided guideline sources. Do not add any information not present in the evidence." to the prompt in `generate_recommendation_v3`
- Reduce evidence to top 3–4 chunks (fewer chunks = less confusion)

### If answer_relevancy is low
The answers wander off-topic. Try:
- Start the generation prompt with: "First, directly answer this clinical question: {question}"
- Check that `user_input` is being passed into the assessment text in Step 5 of the endpoint

---

## What to Write in Your FYP

In your methodology section, write something like:

> "The RAG pipeline was evaluated using RAGAS (Retrieval-Augmented Generation Assessment), a framework for automated evaluation of RAG systems. A synthetic testset of 17 wound-dressing clinical questions was generated from the four clinical guideline documents using RAGAS's TestsetGenerator with wound-care-specific personas. The pipeline was evaluated on four metrics: context precision (retrieved chunk relevance), context recall (guideline coverage), faithfulness (answer grounding), and answer relevancy (clinical question adherence). The judge model was GPT-4o-mini, consistent with established RAGAS evaluation protocols."

In your results section, present the aggregate table from `eval_report.md` and include the heatmap from `plots/per_question_heatmap.png`.

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Cannot connect to RAG server` | Server not running | `uvicorn wound_app_v4:app --port 8000` |
| `UnicodeDecodeError` reading CSV | Windows encoding | The notebook already handles this with cp1252/latin-1 fallback |
| `OpenAI key not set` | Missing .env | Add `OPENAI_API_KEY=sk-...` to .env |
| `FileNotFoundError testset_curated.json` | Generation not done | Run testset generation notebook first |
| RAGAS returns NaN scores | Empty answer or context | Check Cell 8 output — look for ERROR entries in eval_results_raw.json |
| Server timeout on Cell 8 | Slow GPU/CPU | Increase `RAG_TIMEOUT_SEC` in Cell 3 to 180 |

---

*VerdaSense — FYP RAGAS Evaluation Guide*
