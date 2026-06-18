# VerdaSense Wound RAG — Comprehensive Pipeline Analysis
**Ingestion · Retrieval · Generation · Evaluation · KB Expansion**

> Compiled from: 8 `_kept.json` files · `wound_app_02_v4.py` · `wound_ragas_ablation_v4.ipynb` · `wound_testset_builder_v2.py` · `ingestion_full.ipynb` · `ingestion_GP.ipynb` · `ingestion_WCM.ipynb` · Three context documents from prior consultation

---

## Table of Contents

1. [KB Expansion: v3 → v4 — Should You Do It?](#1-kb-expansion-v3--v4--should-you-do-it)
2. [Question 1: Does Manual Chunk Curation Hurt Retrieval?](#2-question-1-does-manual-chunk-curation-hurt-retrieval)
3. [Question 2: Does Markdown Formatting in ai_summary Hurt Similarity?](#3-question-2-does-markdown-formatting-in-ai_summary-hurt-similarity)
4. [Proposed Standard Ingestion Pipeline (Real-World Reference)](#4-proposed-standard-ingestion-pipeline-real-world-reference)
5. [Strong Parts of Your Current Pipeline](#5-strong-parts-of-your-current-pipeline)
6. [Weak Parts and Concrete Improvements](#6-weak-parts-and-concrete-improvements)
7. [Is Your RAGAS Evaluation Pipeline Correct?](#7-is-your-ragas-evaluation-pipeline-correct)
8. [Additional Notes: The Additional Notes Field](#8-additional-notes-the-additional-notes-field)
9. [Overall Verdict and Priority Action List](#9-overall-verdict-and-priority-action-list)

---

## 1. KB Expansion: v3 → v4 — Should You Do It?

### Current state

| DB version | Sources | Chunks |
|---|---|---|
| `db_wound_care_v3` (current) | GP, SFP, WCM, AJGP | 108 chunks |
| `db_wound_care_v4` (proposed) | + EWMA, ISTAP, ANZBA, RCH | **138 chunks** (+28%) |

### Short answer: Yes — and the timing is right

The ablation analysis you already have (`VerdaSense_RAG_Ablation_Analysis.md`) identifies KB coverage as **the single largest unresolved weakness in v4_02**. Specifically, three persistent failure categories map directly to the four new sources you now have:

| Failure category | Root cause | New source that covers it |
|---|---|---|
| `cat_d_notes_diabetic_nonhealing` (fails all 18 versions) | KB has no diabetic foot wound guideline; AJGP has a partial section | EWMA has dedicated DFU chapters (T, I, M, E per DFU) — 4 DFU chunks |
| `cat_b_skin_tear_fragile` | No skin tear classification or product guide in KB | ISTAP (3 chunks: classification, pathway, product selection) |
| `cat_b_burns_hand` | No burns referral criteria in KB | ANZBA (4 chunks: burn depth, referral criteria, first aid, dressing selection) |
| TIME framework consistency | GP is only source with structured TIME-aligned algorithm; retrieval competes between sources | EWMA adds a second authoritative TIME framework source with DFU + VLU application |

The Guide-RAG finding (NeurIPS 2025) is directly relevant here: a **small, well-curated corpus of 4 guidelines outperforms large unfiltered retrieval**. Going from 4 to 8 sources is not "going large and unfiltered" — it is adding the four clinically-targeted documents that your failure patterns tell you are missing.

### What you gain by source

| Source | Chunks | Clinical gap it closes | Unique value vs. existing KB |
|---|---|---|---|
| **EWMA** | 12 | DFU, VLU, TIME framework depth | Only source with TIME-applied-to-chronic-wound detail; DFU inflammation/debridement/edge advancement that WCM lacks |
| **ISTAP** | 3 | Skin tear classification and dressing protocol | Only source with Type 1/2/3 skin tear classification and non-adherent mesh / silicone foam selection logic |
| **ANZBA** | 4 | Burns referral, depth classification, first aid dressing | Only source with burns-specific referral criteria (hand/face/genital/foot rule) |
| **RCH** | 11 | Paediatric wound care; TIME framework from a nursing guideline perspective | The only paediatric-specific source; adds practical application tips and product-level guidance that other sources lack |

### What you must NOT do when building v4

Because you already have an 18-version ablation on `db_wound_care_v3 + wound_testset_v2.json`, the expanded KB makes the new evaluation **not directly comparable** to your existing results. This is not a problem — it is expected and documented in the KB Expansion Consultation doc. Handle it cleanly:

- Keep your existing 9-version ablation as "Experiment 1: Architecture Ablation (fixed KB)."
- Call the new KB evaluation "Experiment 2: KB Expansion Impact (fixed architecture = v4_02)."
- Run only v4_02 with the new KB and the updated testset (wound_testset_v3.json with ~6 updated cases + 2–4 new cases for ISTAP/ANZBA coverage).
- Do not re-run all 9 versions — that is unnecessary and wastes cost.

---

## 2. Question 1: Does Manual Chunk Curation Hurt Retrieval?

### Your concern

You manually reconstructed chunks from PDFs rather than using automated tools (Unstructured.io, LangChain text splitter). You worry this creates unnatural text that could harm retrieval.

### Answer: No — manual curation is a strength, not a weakness

Your concern conflates two different problems: *parsing quality* and *chunk naturalness*. Here is what the evidence from both your own ablation data and the 2024–2025 literature says:

**Why automated PDF parsing fails for clinical guidelines:**

Clinical guideline PDFs are layout-heavy: multi-column tables, page-spanning decision matrices, nested bullet hierarchies, header/footer noise, image-embedded text. When you tried Unstructured.io, it produced incomplete, truncated, and incorrectly-ordered content — especially for the GP Wound Type 1–8 decision tables, which span two pages in a multi-column nested format. This is not a bug in Unstructured.io; it is a fundamental limitation of generic layout parsers applied to domain-specific document formats.

The EULAR RAG paper (Madrid-García et al., 2025) — the closest published parallel to your work — explicitly describes: *"Manuscripts underwent manual cleaning: nonessential sections were removed, headings were reviewed, and tables/boxes were moved to the end to minimise noise and redundancy."* Your approach goes further: you reconstructed tables as structured plain text (the GP Wound Type 1–8 algorithm blocks), which preserves the decision logic that automated parsers would either flatten or misorder.

**What "manual curation" actually means for retrieval quality:**

Your chunks have three properties that automated splits would not guarantee:

1. **Semantic coherence** — each chunk covers one clinical concept (e.g., "Wound Type 3: dry infected wound — algorithm, dressing list, antibiotic guidance"). A LangChain `RecursiveCharacterTextSplitter` at 500 chars would cut mid-table, mixing dressing selection for two different wound types into one chunk. That mixed chunk would retrieve for both but be semantically diluted for either.

2. **Clinical decision completeness** — your GP chunks contain the full wound type decision: tissue profile → wound type → recommended dressings → referral flag → antibiotic flag. A split chunk might contain the tissue profile but not the recommended dressings. Your v4 algorithm chunk pinning (Sub-query A) only works because the algorithm chunks are coherent and complete.

3. **No fictional content** — the ai_summary audit in the prior analysis confirmed that all 11 RCH chunks (the most recently audited set) have zero hallucinated clinical facts. Automated splitting + automated summarisation compounds error risk.

**The quantitative evidence from your own ablation:**

v4_00/v4_01 achieved Context Recall of 83–85%, with the primary driver being the algorithm chunk pinning — which only works because your GP Wound Type chunks are clean, complete, and retrievable. If automated splitting had fragmented those chunks, the metadata filter in Sub-query A would retrieve partial chunks that don't contain the binding algorithm text, and the G1 binding block would fail to inject the dressing allowed list.

**Honest limitation to acknowledge:**

Manual curation does not scale. For a system with 8 sources and 138 chunks, it is entirely feasible. At 300+ guidelines (the NICE RAG scale), you would need a hybrid approach: Docling or a neural parser for initial extraction, with manual review of high-stakes chunks (decision tables, algorithm blocks). For your FYP scope, manual curation is the correct choice.

---

## 3. Question 2: Does Markdown Formatting in ai_summary Hurt Similarity?

### The question in detail

Your `ai_summary` fields are formatted in Markdown: `**bold headers**`, `## section headings`, `- bullet points`, numbered lists. Your `text` fields have a mix — some sources (ANZBA, ISTAP) have Markdown in the raw text too; others (EWMA plain sections) have no Markdown in text but Markdown in the ai_summary. You embed the `ai_summary` as `page_content` (what gets embedded into ChromaDB).

**Does the Markdown formatting hurt cosine similarity retrieval?**

### Answer: Minimally in practice — but there are specific cases where it matters

**How sentence-transformer embeddings handle Markdown:**

`abhinand/MedEmbed-large-v0.1` is based on a sentence-transformer architecture (likely `e5-large` or similar finetuned on biomedical text). These models tokenise text using a WordPiece or SentencePiece tokeniser. Markdown tokens like `**`, `##`, `- `, `1.` are:

- Split by the tokeniser into their character sequences (`*`, `*`, `-`)
- Given token embeddings from the model's vocabulary
- Because the model was likely pre-trained on general text corpora (including Markdown-heavy GitHub/StackOverflow/Reddit text), these tokens are not out-of-vocabulary — they are just low-semantic-weight tokens

The key insight is that **cosine similarity is dominated by content tokens** (clinical terms like `"alginate"`, `"infected"`, `"exudate"`, `"debridement"`), not by structural tokens. A retrieval query like `"What dressing for sloughy wound high exudate?"` has zero Markdown tokens. The cosine similarity is computed against the semantic content of the ai_summary, which is the same regardless of whether clinical terms appear as `alginate` or `**alginate**`. The `**` tokens contribute near-zero weight to the final embedding direction.

**Where Markdown CAN hurt — two specific cases:**

**Case 1: BM25 retrieval (your hybrid retrieval Sub-queries B and C)**

BM25 is a term frequency model. It is literally counting token occurrences. If your ai_summary contains `**alginate**`, the BM25 index sees three tokens: `**`, `alginate`, `**`. The `**` tokens get low IDF (they appear in almost every chunk → low inverse document frequency) and thus near-zero BM25 weight. The `alginate` token gets normal BM25 weight. Net effect: BM25 retrieval is unaffected by Markdown, with one exception.

The exception: **list prefixes that look like terms**. If your ai_summary has:
```
1. Silver dressings
2. Alginate
3. Foam
```
BM25 sees `1.`, `2.`, `3.` as tokens. These are filtered by most tokenisers as stopwords or numerics. No real impact.

**Case 2: RAGAS AnswerRelevancy scoring**

AnswerRelevancy computes cosine similarity between the generated answer embedding and the user input (narrative query) embedding. If your generated answers are heavily Markdown-formatted (which they are, given your 9-section template with `## Primary Dressing`, `## Contraindicated Dressings`, etc.), and the narrative query is plain text, the structural Markdown in the answer could slightly dilute the content-semantic similarity. In practice, the AnswerRelevancy scores in your v3/v4 evaluations (74–77%) are consistent with what the NICE RAG paper reports (not far from their ceiling on structured clinical QA), suggesting this is not a significant factor.

**Concrete recommendations for the new 4 sources:**

| Source | Current state | Action needed |
|---|---|---|
| EWMA | Plain text in `text`; Markdown in `ai_summary` | No change — this is the correct pattern. The raw text is the evidence; the summary is the retrieval surface. |
| ISTAP | Markdown in both `text` and `ai_summary` | Minor issue: raw `text` has Markdown bullets. When stored as `raw_text` in metadata and injected into the generation prompt as `Guideline Source N`, the LLM sees Markdown bullets. This is fine — it is readable and clinically accurate. No action needed. |
| ANZBA | Markdown in both `text` and `ai_summary` | Same as ISTAP — fine. |
| RCH | Mixed: plain text in `text`; Markdown in `ai_summary` | Correct pattern already, no action needed. |

**One real improvement to make:**

When you call the generation LLM and inject `raw_text` as `Guideline Source N`, the Markdown formatting is visible to the LLM. The LLM handles Markdown well — it does not confuse `**Silver dressings**` with a clinical instruction about asterisks. However, for very long chunk injections, stripping Markdown from `raw_text` before injection (keeping it for `ai_summary` only) reduces the token cost by 3–8% without losing clinical information. This is a micro-optimisation, not a correctness issue.

**Summary verdict on Markdown:**

Markdown in `ai_summary` does not measurably hurt cosine similarity retrieval with MedEmbed-large, does not meaningfully affect BM25 term matching, and does not create hallucination risk. It does not need to be fixed before ingestion. Your ingestion design is sound.

---

## 4. Proposed Standard Ingestion Pipeline (Real-World Reference)

Below is the standard ingestion pipeline used in published 2024–2025 medical RAG systems (EULAR RAG, NICE RAG, Guide-RAG, MEREDITH), annotated against your current implementation to show where you match, where you diverge, and whether each divergence matters.

```
STANDARD MEDICAL RAG INGESTION PIPELINE (2024–2025 Consensus)

Stage 1 — Document Acquisition & Pre-screening
  1a. Source selection (clinical guidelines, systematic reviews, position papers)
  1b. Quality gate: peer-reviewed, authority-issued, within 5 years
  1c. Scope filter: remove appendices, references, author bios, legal notices

Stage 2 — Raw Text Extraction
  2a. Layout-aware parser (Docling for academic PDFs; custom for clinical tables)
  2b. Table structure preservation (keep as Markdown table or structured text)
  2c. Image handling (caption if meaningful; discard if decorative)
  2d. OCR if needed for scanned pages

Stage 3 — Semantic Chunking
  3a. Chunk by logical clinical unit (not by character count)
  3b. Each chunk = one clinical decision context (algorithm, dressing type, etc.)
  3c. Chunk overlap strategy: 0–10% overlap for clinical text (high specificity needed)
  3d. Minimum chunk length: 60–100 chars (filter one-liners)

Stage 4 — Metadata Enrichment
  4a. Per-chunk: chunk_id, source, section, parent_section, chunk_index, char_count
  4b. Per-document: authority, year, guideline_type, focus, population
  4c. Domain-specific: wound_category, wound_type (if applicable)

Stage 5 — AI Summary Generation
  5a. LLM rewrites raw text as a clinical summary preserving all facts
  5b. Faithfulness audit: human check that ai_summary does not hallucinate
  5c. Store both text (evidence) and ai_summary (retrieval surface) separately

Stage 6 — Embedding & Indexing
  6a. Embed ai_summary as page_content (retrieval surface)
  6b. Store raw text as metadata (evidence injection surface, not embedded)
  6c. Domain-appropriate embedding model (medical-finetuned > general)
  6d. Cosine similarity space (normalised embeddings, HNSW index)
  6e. Persist to vector store with full metadata schema

Stage 7 — Smoke Test
  7a. At least 3 representative test queries covering the clinical domain
  7b. Verify expected chunks surface in top-5 results
  7c. Check metadata fields are populated and filterable
```

**Where your pipeline matches this standard:**

- Stage 1 (source selection): your sources are all peer-reviewed, authority-issued, recent (2014–2024). RCH is 2023, ISTAP 2024, ANZBA current. 
- Stage 2 (extraction): custom notebook per document type is superior to generic parser for your PDF types, as confirmed by the Unstructured.io failure and the EULAR RAG manual cleaning choice.
- Stage 3 (semantic chunking): your chunks are exactly clinical-unit-bounded. The GP wound type algorithm chunks are the best example — each chunk is one complete decision context.
- Stage 4 (metadata enrichment): chunk-level metadata is complete. You have `authority`, `year`, `guideline_type`, `focus`. The one missing field (see below) is `wound_category`.
- Stage 5 (ai_summary): faithfully audited for RCH (prior session), structurally sound across all sources.
- Stage 6 (embedding): `MedEmbed-large-v0.1` is a domain-appropriate medical embedding model, not a general one. `normalize_embeddings=True` with cosine space is correct.
- Stage 7 (smoke test): your `ingestion_full.ipynb` Cell 5 implements this.

**Where your pipeline diverges from the standard (and whether it matters):**

| Divergence | Impact | Action |
|---|---|---|
| No `wound_category` metadata field per chunk | Sub-query A's metadata filter tries `{"wound_type": {"$eq": str(wt)}}` — this only works for GP chunks that have wound_type embedded in their content, not as a metadata field. The filter falls back to unfiltered dense for 3 out of 4 filter attempts in the current code. | Add `wound_category` to every chunk during ingestion. Map: GP chunks with "Wound Type N" → `wound_type: N`; EWMA DFU chunks → `wound_category: "diabetic_foot_ulcer"`; ISTAP → `wound_category: "skin_tear"`; ANZBA → `wound_category: "burn"`. This makes Sub-query A's metadata filtering actually work. |
| No `population` field on most sources | Only RCH has `"population": "paediatric"`. All other sources have empty `""`. | Add `"population": "adult_general"` as default for all non-RCH sources in `ingestion_full.ipynb`. This enables future filtering to exclude RCH paediatric-specific recommendations from adult patient queries, or to boost them for paediatric-flagged notes. |
| `ai_summary` used as `page_content` but `raw_text` used for generation injection | This is actually the correct design (embed the summary, inject the raw evidence). But the gap between summary quality and raw text detail matters. | No change needed — this is the right design. The audit confirms summaries are faithful. |
| No chunk overlap | Your chunks are semantically bounded, so overlap is less important than in character-split pipelines. But cross-chunk concepts (e.g., an infection principle in Chunk 2 that relates to a dressing in Chunk 6) will not be bridged. | For your current scale (138 chunks), this is acceptable. For a future 300+ chunk KB, consider adding a "parent chunk" strategy where each chunk stores the preceding and following chunk IDs for re-ranking context window expansion. |
| Single-document ingestion notebooks (one per source) | Correct design for different PDF layouts. The `ingestion_full.ipynb` correctly aggregates them. | No change needed — this is the right architecture for clinical guidelines with heterogeneous layouts. |

---

## 5. Strong Parts of Your Current Pipeline

### 5.1 Architecture design: independently converges on 2025 best practices

Your v4_02 pipeline implements every component that the 2024–2025 medical RAG literature identifies as essential for high-performing clinical RAG:

- **Hybrid dense + BM25 retrieval** — confirmed by EULAR RAG (2025), NICE RAG (2025), MEREDITH enhanced (2024)
- **Narrative query construction** — your `build_narrative_query()` converts structured T.I.M.E. inputs into natural language; this is the same insight that drove v3's Answer Relevancy jump from 58% to 77%
- **Grounded system prompt with mandatory citations** — the v3 grounded prompt instruction ("every claim MUST be supported by a numbered source") mirrors CARE-RAG's finding that explicit grounding instructions are essential for faithfulness
- **Binding algorithm block (G1)** — MEREDITH calls this "expert-guided generation"; your implementation constrains the LLM to only recommend dressings from the retrieved algorithm's allowed list, which is what drove the safety pass rate from 78% (v3) to 96% (v4_02)
- **Mandatory injection (G2) without expensive verifier (G3)** — the verifier fired in 1/56 cases across v4_00+v4_01 and did not resolve the safety failure when it did fire. Removing it was the correct engineering decision. G2 deterministic injection gives you the safety guarantee at zero marginal cost.
- **Pre-classifier with expanded infection keyword detection** — the FIX 1 expansion to include subclinical infection signals (warmth, redness, increased exudate, malodour, non-healing) is clinically grounded. These are the exact TIMES "I" signs that clinicians use to infer early infection before it is formally diagnosed.

### 5.2 The ai_summary / raw_text dual-field architecture

Embedding the `ai_summary` for retrieval while injecting `raw_text` for generation is the correct design for clinical RAG. It separates the retrieval concern (semantic similarity) from the generation concern (clinical accuracy and completeness). The `ai_summary` is a compact, semantically dense representation of the chunk; the `raw_text` is the complete evidence from which the LLM must generate its grounded response. This two-field design is used implicitly by EULAR RAG (where they clean the passages before indexing but inject the original guideline text into the prompt).

### 5.3 The 9-section structured output

Your generation prompt enforces a fixed 9-section clinical output structure (Primary Dressing, Secondary Dressing, Rationale by T.I.M.E., Contraindicated Dressings, Antibiotic Considerations, Referral/Escalation, Dressing Change Frequency, Application Tips, Clinical Notes). This structure:

- Makes your rule-based safety checker deterministic — it can locate the Antibiotic and Referral sections by heading rather than scanning the whole response
- Is aligned with clinical practice (GP consultation structure)
- Enables your `_POSITIVE_SECTION_HEADERS` regex to correctly exclude avoidance language in recommendation sections
- The MEREDITH and EULAR RAG papers both use structured output sections for the same reason

### 5.4 Dual-generator ablation with RAGAS + rule-based safety

Running 9 architectures × 2 generators (GPT-4o-mini + Qwen3:14b) with both RAGAS metrics and a domain safety checker is methodologically stronger than most published papers. CARE-RAG (NeurIPS 2025 Workshop) tests 20 LLMs but uses a single architecture. Guide-RAG tests 6 KB configurations with a single LLM. EULAR RAG tests a single architecture with LLM-judge + partial human validation but no safety checker. Your approach of combining automated RAGAS evaluation with a rule-based clinical safety checker is a genuine methodological contribution not present in most reviewed papers.

### 5.5 The `_kept.json` technology-agnostic schema

Your `_kept.json` schema (chunk_id, source, section, parent_section, chunk_index, char_count, text, ai_summary) is technology-agnostic. It is not tied to ChromaDB or LangChain — it is a clean intermediate representation. If you wanted to switch to Qdrant, Pinecone, Weaviate, or PostgreSQL+pgvector, you would only need to update `ingestion_full.ipynb`, not your 8 ingestion notebooks. This is exactly the design that the EULAR RAG paper used with their knowledge base files.

### 5.6 The FIX 2 constrained diabetic escalation

The v4_01 bug (diabetic keyword alone → escalate wound_type to 7) was clinically incorrect. A clean granulating wound in a diabetic patient is NOT a Wound Type 7 — it is a Type 1 or 2 with a referral flag. Your fix correctly separates the referral decision from the wound type classification, adding the `etiology` field to carry diabetic context into the generation prompt via the etiology note injection. This is a clinically sound design: the wound type determines the dressing algorithm; the etiology adds contraindications (no adhesive bordered foam on diabetic feet, no hydrocolloid) via a soft note rather than overriding the algorithm.

---

## 6. Weak Parts and Concrete Improvements

### 6.1 CRITICAL: `wound_type` metadata filter in Sub-query A does not reliably work

**The problem:**

In `retrieve_chunks_multiaxis()`, Sub-query A attempts three filter strategies:
```python
{"wound_type": {"$eq": str(wt)}},     # string
{"wound_type": {"$eq": wt}},           # int
{"source": {"$contains": f"Wound Type {wt}"}},  # content
```

None of these filters will hit correctly on the existing KB because:
- Your chunk metadata does not contain a `wound_type` key — check `ingestion_full.ipynb` Cell 3: the metadata dict includes `chunk_id`, `source`, `section`, `parent_section`, `chunk_index`, `char_count`, `raw_text`, `guideline_type`, `authority`, `year`, `focus` — but **not `wound_type`**.
- The third filter tries `$contains` on `source` (the PDF filename), which will not contain "Wound Type N".
- So all three filters fail every time, and the code falls back to unfiltered dense search with the wound-type-specific query phrase.

The unfiltered dense fallback often retrieves the right chunk by semantic similarity (since the GP algorithm chunks have "Wound Type N" in their text), but it is not guaranteed, especially when multiple wound types have similar tissue descriptions.

**The fix** (add to `ingestion_full.ipynb` before building ChromaDB):
```python
import re

def extract_wound_type(chunk: dict) -> int | None:
    """Extract wound_type integer from section name or text."""
    text = chunk.get("section", "") + " " + chunk.get("text", "")
    match = re.search(r"[Ww]ound\s+[Tt]ype\s+(\d)", text)
    return int(match.group(1)) if match else None

# In chunks_to_documents(), add to metadata dict:
metadata = {
    ...existing fields...,
    "wound_type"     : extract_wound_type(chunk),          # int or None
    "wound_category" : infer_wound_category(chunk),        # see below
    "population"     : chunk.get("population", "adult_general"),
}
```

And add `infer_wound_category()`:
```python
def infer_wound_category(chunk: dict) -> str:
    text = (chunk.get("section", "") + " " + chunk.get("source", "")).lower()
    if "dfu" in text or "diabetic foot" in text or "diabetic_foot" in text:
        return "diabetic_foot_ulcer"
    if "skin tear" in text or "istap" in text.lower():
        return "skin_tear"
    if "burn" in text or "anzba" in text.lower():
        return "burn"
    if "vlu" in text or "venous leg ulcer" in text:
        return "venous_leg_ulcer"
    if "pressure" in text:
        return "pressure_injury"
    return "general"
```

Then in `wound_app_02_v4.py`, update Sub-query A:
```python
# Filter attempt 1: exact wound_type match (now works with metadata field)
{"wound_type": {"$eq": wt}},
# Filter attempt 2: wound_category filter for etiology-specific chunks
{"wound_category": {"$eq": etiology_to_category.get(classifier["etiology"], "general")}},
```

This change converts Sub-query A from a probabilistic fallback into a reliable metadata-pinned retrieval step — the design intent from the beginning.

### 6.2 IMPORTANT: The `confidence_score` is hardcoded at 0.5

In the `/get_recommendation` endpoint response:
```python
"confidence_score": 0.5,
"confidence_label": "MEDIUM",
```

This is hardcoded. Every query returns `confidence_score: 0.5, confidence_label: MEDIUM` regardless of retrieval quality. For a patient-facing application, this is misleading — a query retrieving chunks with cosine similarity 0.92 and a query retrieving chunks with cosine similarity 0.41 both report "MEDIUM" confidence.

**The fix:** Compute a real confidence score from retrieval quality:
```python
# After similarity_search_with_score, compute confidence
results_with_scores = db.similarity_search_with_score(narrative_query, k=top_n)
top_scores = [score for _, score in results_with_scores[:3]]
avg_top_score = sum(top_scores) / len(top_scores) if top_scores else 0.0

# Cosine similarity: 0.0 = identical, 2.0 = opposite (ChromaDB cosine space)
# Convert to [0,1] confidence: higher similarity = lower distance = higher confidence
confidence_score = max(0.0, 1.0 - (avg_top_score / 2.0))

if confidence_score >= 0.75:
    confidence_label = "HIGH"
elif confidence_score >= 0.50:
    confidence_label = "MEDIUM"
else:
    confidence_label = "LOW"
```

Note: ChromaDB with `hnsw:space: cosine` returns cosine distance (0 = most similar), not cosine similarity (1 = most similar). Adjust accordingly.

### 6.3 MODERATE: The additional notes field is used but not fully leveraged

Sub-query C appends the first 300 chars of notes to a generic retrieval query. This is better than nothing but misses two clinical signals that notes commonly carry:

**Signal 1 — Notes as TIME override**: Your notes say things like "patient has not been improving for 6 weeks despite hydrogel." This implies the current dressing is wrong — but the retrieval does not search for "alternative to hydrogel non-healing wound." Sub-query C retrieves for the notes content, but the query is `notes[:300] + " wound dressing recommendation"` — not shaped around the override implication.

**Signal 2 — Notes-to-classifier gap remains**: Despite FIX 1's expanded keyword list, notes like "wound has been there for 8 weeks without improvement, no obvious signs of infection" would NOT trigger the antibiotic flag (no keywords match), yet persistent non-healing in a clean wound is a biofilm risk signal. Consider adding a dedicated LLM call to classify the notes: `"Given these clinical notes, does this wound show signs of: biofilm (yes/no), inadequate dressing (yes/no), patient compliance issue (yes/no)? Answer as JSON."` This is a lightweight pre-processing step (<0.001 USD per query) that extracts structured signal from notes and injects it into the classifier before wound_type assignment.

### 6.4 MODERATE: EWMA has very long chunks — retrieval surface may dilute

EWMA chunks average 4,400 chars for raw text and 2,900 chars for ai_summary. The SFP and WCM chunks average 1,200–2,000 chars for ai_summary. When you embed the EWMA ai_summary (2,900 chars), the embedding is averaging over many more tokens than a focused 800-char summary. Longer embeddings are more semantically diffuse — they represent the *average* of many clinical concepts, not the dominant concept. This can reduce retrieval precision for EWMA chunks.

**Recommendation**: Consider splitting the four longest EWMA chunks (DFU-T/I/M/E and VLU-T/I/M/E) each into two sub-chunks: one covering the clinical assessment component (what to look for) and one covering the management/dressing component (what to do). This gives you ~8 more focused EWMA chunks with better embedding specificity, at the cost of 8 additional documents in the vector store.

### 6.5 LOW: The testset is not yet updated for the 4 new sources

The `wound_testset_v2.json` (28 cases) was built entirely from the 4 v3 sources (GP, SFP, WCM, AJGP). After building `db_wound_care_v4`, the following cases have misaligned `reference_contexts`:

| Case | Issue | Fix |
|---|---|---|
| `cat_b_skin_tear_fragile` | No ISTAP context in reference_contexts | Add ISTAP_PRODUCTS and ISTAP_PATHWAY chunk IDs |
| `cat_d_notes_diabetic_nonhealing` | No EWMA DFU context in reference_contexts | Add EWMA DFU chunk IDs; update `allowed_dressings` |
| `cat_b_burns_hand` | No ANZBA context in reference_contexts | Add ANZBA_REFERRAL chunk ID |
| `cat_b_diabetic_foot` | Partial — AJGP has some diabetic section | Add EWMA DFU chunks as secondary reference_contexts |

You need to add ~4 new test cases for ISTAP (skin tear types), ANZBA (burns referral boundary cases), and EWMA (VLU-specific). Without these, your evaluation on the expanded KB will undercount the benefit of the new sources, because the RAGAS metrics will not be measuring recall of the new chunks.

### 6.6 LOW: RCH paediatric source has no guard in retrieval

The RCH source has `"population": "paediatric"` but there is no retrieval filter to prevent RCH chunks from surfacing for adult patient queries. If a 65-year-old patient with a diabetic foot ulcer submits their T.I.M.E. inputs, Sub-query B might retrieve RCH's foam dressing guidance (which is paediatric-specific — e.g., "consider LESS frequent dressing changes in the paediatric population").

**Fix**: After adding `population` metadata to all chunks, add a population filter to Sub-query B:
```python
# For adult patients (no "paediatric" or "child" in notes):
population_filter = {"population": {"$ne": "paediatric"}}
mech_docs = db.similarity_search(mech_query, k=3, filter=population_filter)
```

Or softer: store the `population` field in the retrieval_notes output and let the generation prompt be aware — "Note: Source N is a paediatric guideline; apply age-appropriate adjustments."

---

## 7. Is Your RAGAS Evaluation Pipeline Correct?

**Short answer: Yes — with two specific caveats that you must disclose in your FYP report.**

### What is correct

The evaluation pipeline is methodologically sound and internally consistent, as confirmed in the ablation analysis document. The key validating facts:

- Retrieved contexts are byte-for-byte identical between generators for every version — meaning the retrieval architecture is correctly held fixed when comparing generators
- The RAGAS judge (gpt-4o-mini, temperature=0) and embeddings (text-embedding-3-small) are consistent across all 18 versions
- The rule-based safety checker applies the same logic from the same testset fields for both generators
- The four RAGAS metrics (Context Precision, Context Recall, Faithfulness, Answer Relevancy) together cover the three RAG failure modes: retrieval quality (CP/CR), grounding (FA), and response quality (AR)
- The 28-testset ablation is the right design for an FYP: it enables controlled comparison of architectural components while keeping evaluation cost manageable

### Caveat 1: Answer Relevancy in v2 is not comparable to v3/v4

In v2, `user_input` for RAGAS was the raw structured T.I.M.E. string (e.g., `"Necrotic: 0%, Slough: 0%, Granulation: 100%, Infection: Not infected, Moisture: Low, Edge: Advancing"`). In v3+, `user_input` is the narrative query. RAGAS AnswerRelevancy measures cosine similarity between the generated answer embedding and the user_input embedding. Comparing a 3,000-char clinical recommendation to a 60-char structured T.I.M.E. string gives systematically lower scores than comparing it to a 200-char narrative question — not because the answer is worse, but because the measurement baseline changed. **You must label v2 AR scores with a caveat in your FYP report.** The ideal fix is to retroactively generate narrative queries for all 28 v2 cases using `build_narrative_query()` and re-run RAGAS AR only.

### Caveat 2: Construct circularity — testset derived from KB

Your `wound_testset_v2.json` was built by extracting `reference_contexts` directly from the same 4 KB sources that the RAG retrieves from. This means Context Precision and Context Recall are measuring retrieval quality *within the KB's own content*, not against an independent clinical standard. This is best practice for a controlled ablation study (EULAR RAG and Guide-RAG do the same), but it means you cannot claim generalisation beyond your KB coverage. State this explicitly: *"Evaluation was conducted on a closed-loop testset derived from the same 4 guideline sources. Generalisation to unseen clinical scenarios was not evaluated."*

### The safety checker's special value — and its limitation

Your rule-based safety checker adds something no published RAGAS evaluation includes: a domain-specific pass/fail on clinical safety-critical phrases. The jump from 28% (v2_00 GPT) to 96% (v4_02) is the most clinically meaningful result in your entire evaluation. However, the checker has one structural limitation: it uses keyword matching for phrases like `"antibiotic therapy is recommended"` which your system prompt explicitly instructs the LLM to produce. The v3→v4 safety improvement partly reflects the LLM learning to produce the exact trigger phrase, not necessarily becoming clinically safer in an absolute sense. A human clinical reviewer reading 10 responses would disentangle these — which is why the lightweight human evaluation (1 wound nurse, 10 v4_02 cases) in your next steps list is important.

### On testset size

28 test cases provides confidence intervals of approximately ±12% on a 96% pass rate (using Wilson interval). Your results are statistically reliable for the conclusion that v4_02 is better than v3, but not for claiming a specific numeric pass rate for a deployment context. EULAR RAG (740 questions) and CARE-RAG (clinician-validated questions) demonstrate the scale required for clinical validation. For an FYP, 28 is adequate for ablation; for any deployment claim, 50–100 cases minimum is the threshold.

---

## 8. Additional Notes: The Additional Notes Field

This section addresses your observation that "additional notes are true and can override those TIME inputs if other students' model gives my RAG wrong TIME inputs."

This is architecturally correct and one of the most clinically thoughtful aspects of your system. Here is the full picture:

### Why notes-as-override is clinically justified

In real clinical practice, a GP's free-text observations always take precedence over a structured checklist filled out before the consultation. Your system mirrors this: the classifier reads structured T.I.M.E. inputs (which may be wrong, from imperfect classification models) AND the free-text notes (which the patient or GP has typed directly). FIX 1's expanded keyword list is specifically designed to let notes override the structured "Not infected" flag when clinical signs suggest otherwise.

### How the override currently works (and its limits)

Currently, notes influence the system through four pathways:
1. `classify_wound()` reads notes for antibiotic triggers (FIX 1) and referral triggers
2. `build_narrative_query()` appends `notes[:200]` to the semantic retrieval query
3. Sub-query C retrieves chunks specifically for the notes context
4. The assessment text injected into the generation prompt includes the full notes under "ADDITIONAL CLINICAL NOTES"

The limit: the override is **keyword-triggered**, not **semantically-interpreted**. Notes saying "patient reports the wound feels warmer than usual and has a slight smell" WILL trigger `warmth` and `malodour` keywords. But notes saying "patient has been applying honey at home but it doesn't seem to help" will NOT trigger any special pathway — the RAG will recommend honey if the KB supports it for this wound type, even though the notes suggest treatment failure.

### Proposed enhancement: a notes pre-classifier

Add a lightweight LLM call before `classify_wound()` to extract structured signals from notes:

```python
async def preprocess_notes(notes: str) -> dict:
    """
    Use LLM to extract structured clinical signals from free-text notes.
    Returns JSON with override flags.
    """
    if not notes.strip() or len(notes.strip()) < 20:
        return {}
    
    prompt = f"""You are a clinical triage assistant. 
    Read these clinical notes about a wound and extract key signals.
    Respond ONLY with valid JSON, no other text.
    
    Notes: {notes[:500]}
    
    {{
      "treatment_failure": true/false,  // current dressing not working
      "failed_dressings": ["list if mentioned"],  // dressings that have failed
      "duration_weeks": null or integer,  // how long wound has been present
      "biofilm_risk": true/false,  // non-healing + no infection signs
      "patient_concern": "brief string or null"
    }}"""
    
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    try:
        return json.loads(response.content)
    except:
        return {}
```

Then inject this into the generation prompt as additional context:
```
📋 NOTES ANALYSIS:
- Treatment failure detected: current/previous dressings not resolving wound
- Failed dressings mentioned: hydrogel
- Duration: 8 weeks (chronic wound threshold exceeded)
INSTRUCTION: Do NOT recommend hydrogel (noted as treatment failure). 
Consider alternative dressings from the allowed list.
```

This is the "notes-as-override" mechanism fully realised. It converts free-text clinical context into actionable constraints on the generation — which is the highest clinical value your notes field can provide.

---

## 9. Overall Verdict and Priority Action List

### Verdict

Your pipeline is methodologically sound and clinically thoughtful. It independently converges on the design decisions that the best 2024–2025 medical RAG papers also arrived at. The v4_02 architecture at 96% safety pass rate is a strong result. The KB expansion from 4 to 8 sources is the right next step — the new sources directly address the four failure categories that persist even in your best architecture.

### KB expansion impact estimate

Based on the failure pattern analysis, adding EWMA+ISTAP+ANZBA+RCH to `db_wound_care_v4` is expected to:
- Resolve the `cat_d_notes_diabetic_nonhealing` persistent failure (EWMA DFU coverage)
- Resolve `cat_b_skin_tear_fragile` (ISTAP coverage)
- Resolve `cat_b_burns_hand` (ANZBA coverage)
- Improve Context Recall on those cases from ~50–60% to ~80%+ (estimate; actual depends on testset update)

### Priority action list

**Priority 1 — Must do before v4 evaluation (these affect correctness):**

1. Add `wound_type`, `wound_category`, and `population` metadata fields to `ingestion_full.ipynb` — fixes the Sub-query A metadata filter that currently always falls back to unfiltered dense.
2. Add `GUIDELINE_METADATA` entries for all 4 new sources (EWMA, ISTAP, ANZBA, RCH) in `ingestion_full.ipynb`.
3. Update testset: add new chunk ID constants for new sources; update `reference_contexts` for 5–6 affected cases; add 4 new test cases for ISTAP/ANZBA coverage. Rebuild as `wound_testset_v3.json`.
4. Run RCH ai_summary audit fixes: Chunk 4 (add paediatric less-frequent dressing change note) and Chunk 11 (add Aquacel Rope foreign body reasoning) — both flagged in the prior RCH audit.

**Priority 2 — High value (do if time allows before submission):**

5. Fix the hardcoded `confidence_score: 0.5` — replace with real cosine similarity from retrieval.
6. Add `population` filter to Sub-query B to prevent RCH paediatric chunks from surfacing for adult queries.
7. Consider splitting the 4 longest EWMA chunks (>4,000 chars text) into assessment + management sub-chunks for better embedding specificity.
8. Add the notes pre-classifier (lightweight LLM call to extract treatment_failure and failed_dressings from notes) — enables true notes-as-override for treatment failure cases.
9. Retroactively generate narrative queries for v2 RAGAS cases and re-run AR only — makes your cross-version AR comparison valid.

**Priority 3 — Future work / Post-FYP:**

10. Add a medical domain cross-encoder reranker (`abhinand/MedEmbed-reranker-v0.1`) as v4_03 variant — your ablation shows `ms-marco-MiniLM` hurts recall on clinical text; a medical reranker may reverse this.
11. Add VLM image pre-processing as optional RAG input enrichment — use GPT-4o Vision to describe wound image in T.I.M.E. terms → append as supplementary structured input.
12. Lightweight human clinical evaluation: 1 wound care nurse, 10 v4_02 outputs, 5-criterion Likert scale. This is the single highest-ROI improvement to your FYP score.
13. Variance estimation: run v4_02 RAGAS 3× and report mean ± SD on all 4 metrics.

---

### Execution order for db_wound_care_v4

```
Step 1: Update ingestion_full.ipynb
  - Add EWMA, ISTAP, ANZBA, RCH to CHUNK_JSON_FILES
  - Add GUIDELINE_METADATA entries for all 4 new sources
  - Add wound_type, wound_category, population to chunks_to_documents()
  - Run → produces db_wound_care_v4/

Step 2: Update wound_testset_builder
  - Add new chunk_id constants from new _kept.json files
  - Update reference_contexts for cat_b_skin_tear_fragile, cat_d_notes_diabetic_nonhealing,
    cat_b_burns_hand, cat_b_diabetic_foot
  - Add 4 new test cases: skin_tear_type3, burns_minor_hand, vlu_chronic, dfu_infected
  - Rebuild as wound_testset_v3.json

Step 3: Update wound_app_02_v4.py → wound_app_02_v5.py
  - Fix Sub-query A metadata filter (use wound_type and wound_category fields)
  - Fix confidence_score (compute from cosine similarity)
  - Add population filter to Sub-query B
  - Update DB path to db_wound_care_v4

Step 4: Run single evaluation pass (v4_02 config only)
  - Both generators (GPT + Qwen)
  - wound_testset_v3.json (32 cases)
  - Report as Experiment 2 (KB expansion) — separately from Experiment 1 ablation

Step 5: Compare and document
  - Label: "Experiment 1: Architecture Ablation (db_wound_care_v3, 28 cases, 9 versions)"
  - Label: "Experiment 2: KB Expansion Impact (db_wound_care_v4, 32 cases, v4_02 config)"
  - Do not present as a continuation of Experiment 1 — they are separate experiments
```

---

*Analysis compiled from full reading of all 16 uploaded files plus three prior consultation documents. All recommendations grounded in the literature cited in VerdaSense_FYP_LitReview_NextSteps.md and the ablation results in VerdaSense_RAG_Ablation_Analysis.md.*
