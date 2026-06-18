# Consistency Cross-Check Report: `wound_app_unimodal.py` vs Ablation Studies

Based on a detailed code review of `wound_app_unimodal.py` and the RAGAS ablation study implementations (`R1_Query_Strategy`, `R2_Retrieval_Strategy`, `G1_Prompt_Strategy`, etc.), here is the cross-check analysis.

Overall, **`wound_app_unimodal.py` is highly consistent** with the ablation winners, and in a few places, it implements robust corrections and enhancements over the raw ablation code.

## 1. Architecture Alignment

| Component | Ablation Winner | `wound_app_unimodal.py` Implementation | Consistency |
| :--- | :--- | :--- | :--- |
| **Retrieval Strategy (R1)** | **R1-C (Multi-axis sub-queries)** | Implemented via `retrieve_chunks_multiaxis()`. Divides queries into Algorithm, Mechanism, and Notes. Adds a robust fallback to fill remaining chunks with the narrative query if notes are missing. | ✅ Consistent & Enhanced |
| **Dense Search (R2)** | **R2-A (Dense only)** | Implemented via `_dense_search()`. Pure Chroma `similarity_search` without BM25/hybrid/RRF logic. | ✅ Consistent |
| **Top-K (R3)** | **Top-K = 6** | `retrieve_chunks_multiaxis()` explicitly deduplicates and caps the final pool at `top_n=6`. | ✅ Consistent |
| **Prompt Strategy (G1)** | **G1-C (Grounded System Prompt)** | Uses the exact G1-C system prompt text. Furthermore, it incorporates the **G1-D (Full Clinical Scaffolding)** logic (binding algorithm block, pre-classifier injections) which builds upon G1-C. | ✅ Consistent |

---

## 2. Component-Level Cross-Check

### `interpret_tissue_percentages()`
- **Ablation (R1/G1):** Thresholds set at `n >= 50%` (necrotic), `s >= 50%` (sloughy), `g >= 70%` (granulating), `n>=25 & s>=25` (mixed).
- **Unimodal App:** Identical logic and thresholds.

### `WOUND_TYPE_QUERY_PHRASES`
- **Ablation (G1):** Maps 1-8 wound types to specific semantic queries.
- **Unimodal App:** Identical mapping.

### `_ANTIBIOTIC_TRIGGERS`, `_DIABETIC_TRIGGERS`, `_REFERRAL_TRIGGERS`
- **Ablation (G1):** Included expanded triggers (e.g., `_ANTIBIOTIC_TRIGGERS` with 36 keywords for subclinical infection).
- **Unimodal App:** Exact matching lists for all three trigger sets.

### `classify_wound()`
- **Ablation:** 
  - In `R1`: Contained a subtle logical gap where `nv` between 25-50% was misrouted if not infected.
  - In `G1`: Corrected to use a uniform `nv_high = nv >= 25` threshold across all branches, alongside etiology detection and subclinical infection escalation.
- **Unimodal App:** **Matches the corrected `G1` version.** It uniformly applies the `nv >= 25` threshold, safely routes diabetic escalations, and detects subclinical infections.

### `build_narrative_query()` & `retrieve_chunks_multiaxis()`
- **Ablation:** R1-C performs SubQ-A (Algorithm), SubQ-B (Mechanism), and SubQ-C (Notes). If notes are empty, SubQ-C is skipped.
- **Unimodal App:** Enhances the ablation logic.
  - SubQ-A adds **ChromaDB metadata filtering** (`where={"wound_type": {"$eq": str(wt)}}`) to guarantee precision for the binding algorithm, falling back to dense search if needed.
  - If notes are empty, it uses `build_narrative_query()` to generate a full NL string to fill the remaining chunks (SubQ-Fill) up to `k=6`. This is a more robust implementation of R1-C.

### `_dense_search()`
- **Unimodal App:** Implements pure dense cosine similarity search, perfectly matching R2-A's requirement (no BM25 / EnsembleRetriever).

### `_build_dressing_mechanism_query()`
- **Unimodal App:** Logic mapping moisture/infection/tissue to dressing mechanisms (e.g., `nv >= 50` $\rightarrow$ autolytic/enzymatic) is identical to the ablation studies.

### `SYSTEM_PROMPT`
- **Unimodal App:** The 6-point strict grounding rules match the G1-C ablation text character-for-character. 

### `_build_etiology_note()`
- **Ablation (G1-D):** Injects etiology notes inline with verbose instructions (e.g., "The patient is diabetic. Per AJGP... Incorporate these...").
- **Unimodal App:** Extracts this into a cleaner helper function. It trims the verbose LLM instructions but retains the exact clinical constraints (e.g., "Adhesive bordered foam dressings are CONTRAINDICATED on feet").

### `generate_recommendation()`
- **Unimodal App:** Integrates all G1 scaffolding. It correctly identifies the algorithm chunk to create the `binding_block`, appends `mandatory_injections` for referral/antibiotics, and adds `etiology_note`. 
- It also uses `_maybe_no_think` which perfectly aligns with the **G3 Ablation** findings for OpenRouter Qwen models, ensuring `<think>` tokens are bypassed/stripped to save cost.

---

## Conclusion
The `wound_app_unimodal.py` file is a highly faithful production implementation of the RAGAS ablation study winners. It successfully merges the **R1-C/R2-A/R3-C** retrieval pipeline with the **G1-D** prompt scaffolding, while adding necessary production safeguards (like deterministic metadata filtering and narrative query fallbacks).
