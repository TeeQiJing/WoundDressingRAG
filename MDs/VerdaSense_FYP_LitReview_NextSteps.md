# VerdaSense Wound RAG — FYP Comprehensive Literature Review & Progress Assessment

> **Student:** AI FYP · **System:** VerdaSense Wound Dressing Recommendation RAG
> **Target Users:** Patients with wounds seeking evidence-based dressing guidance via mobile app
> **Date compiled:** April 2026

---

## Table of Contents

1. [Should You Accept Raw Wound Images as RAG Input? (Multimodal Question)](#1-should-you-accept-raw-wound-images-as-rag-input)
2. [Literature Review: Standard Medical RAG Pipelines](#2-literature-review-standard-medical-rag-pipelines)
3. [Most Relevant Parallel Work to Your FYP](#3-most-relevant-parallel-work-to-your-fyp)
4. [Where Your Architecture Sits in the Literature](#4-where-your-architecture-sits-in-the-literature)
5. [Limitations of Your Current System](#5-limitations-of-your-current-system)
6. [Next Steps (Prioritised)](#6-next-steps-prioritised)
7. [FYP Progress Rating: 72 / 100](#7-fyp-progress-rating-72--100)

---

## 1. Should You Accept Raw Wound Images as RAG Input?

### Short answer: Not as a direct RAG query input — but yes as a structured metadata source feeding INTO your existing RAG pipeline.

Here is the distinction:

**What multimodal RAG means in 2025:** Systems like MIRA (ACM Multimedia 2025), MED-RWR, and the MEDIQA-WV 2025 winning systems use wound images *alongside* text queries to retrieve visually similar exemplars or to derive wound attributes that then ground generation. The image is not shoved directly into a ChromaDB cosine similarity retrieval as a query — it is *described* or *classified* first, and those outputs feed the text-based retrieval or the generation prompt.

**The MEDIQA-WV 2025 shared task** (ClinicalNLP 2025, ACL) is directly parallel to your FYP. Teams built systems that accept wound images + patient text queries and generate free-text wound care responses. The top-performing approach (MasonNLP, LLaMA-4 Scout 17B with RAG) achieved the best results by: (1) using CLIP-like embeddings to retrieve visually similar wound exemplars, then (2) feeding both the retrieved exemplar text and the wound image into a vision-language model (VLM) for generation. The EXL Health AI Lab approach used metadata-guided generation: classifiers predicted 4 key wound attributes from the image, and those structured predictions were injected into the generation prompt — which is essentially what your v4 classifier already does for T.I.M.E.

**For your app's current architecture (v4_02), the practically sensible answer is:**

| Option | What it means for you | Verdict |
|---|---|---|
| Pass raw image directly as query into ChromaDB | ChromaDB is a text vector store. Images cannot be queried against text chunks via cosine similarity without a CLIP-style multimodal encoder. Your KB chunks are clinical text, not images. This would require re-embedding your entire KB with a vision-language encoder. | ❌ Not practical for FYP scope |
| Use a VLM (e.g., GPT-4o Vision, LLaVA, Qwen-VL) to *describe* the wound image → feed description into your existing text RAG pipeline | The image becomes a narrative: "The wound shows 70% granulation tissue with pink edges, no signs of purulence, moderate exudate." This narrative can then optionally supplement or replace the structured T.I.M.E. inputs. | ✅ Achievable, high value for FYP |
| Use the image as a quality signal: validate that classified T.I.M.E. labels are consistent with the visual appearance (a "verifier" for Student B's model output) | Cross-check: if Student B classifies "not infected" but VLM sees clear purulence, flag a discrepancy. | ✅ Strong FYP contribution |
| Full multimodal RAG: visual exemplar retrieval + VLM generation | Requires wound image dataset for exemplar retrieval, CLIP embeddings, significantly larger system. State-of-the-art but exceeds FYP scope. | ⚠️ Future work only |

**Concrete recommendation for your FYP:** Add a single lightweight step before your existing v4_02 RAG — call GPT-4o Vision (or locally, Qwen2-VL or LLaVA-Med) with the wound image and a structured prompt: *"Describe this wound in T.I.M.E. framework terms: tissue composition %, infection signs, exudate level, edge status. Output JSON."* Feed the JSON as additional context into your v4_02 generation prompt alongside the structured T.I.M.E. inputs from Student A/B. This is called **structured VLM pre-processing as a RAG input enrichment step** and it is exactly what the MEDIQA-WV EXL team did to achieve top-5 performance.

---

## 2. Literature Review: Standard Medical RAG Pipelines

### 2.1 The Canonical 5-Stage Clinical RAG Pipeline (2024–2025 consensus)

Across all reviewed papers — CARE-RAG (NeurIPS 2025), EULAR RAG (Elsevier 2025), Guide-RAG (NeurIPS 2025), MEREDITH (JCO 2024), RAGMed (MDPI 2025) — the standard architecture converges on five stages:

```
[Clinical Query / Patient Input]
        ↓
[1. Pre-processing & Structured Signal Extraction]
        ↓
[2. Hybrid Retrieval  (dense + sparse BM25)]
        ↓
[3. Reranking / Context Selection]
        ↓
[4. Grounded Generation  (system prompt + evidence injection)]
        ↓
[5. Post-generation Safety / Verification]
        ↓
[Clinical Recommendation Output]
```

Your v4_02 pipeline implements all five stages. What varies across the literature is *how* each stage is realised.

---

### 2.2 Key Papers Reviewed and Their Relevance to Your Work

#### A. CARE-RAG (Potluri et al., NeurIPS 2025 Workshop)
**What it does:** Evaluates 20 LLMs on Written Exposure Therapy (PTSD) guidelines using three context conditions (correct, noisy, misleading) and three reasoning levels (no/light/heavy). Uses RAGAS-style metrics + inference fidelity score.

**How it maps to your work:** You face the same core problem — *"models may retrieve the right content but still misinterpret clinical instructions."* CARE-RAG finds that no model achieves perfect reasoning fidelity even with correct context, and that yes/no (binary) questions are particularly unstable — exactly your `antibiotic_recommended` and `referral_recommended` safety checks. Their finding that Llama-3.1-8B-Instruct, Gemini-2.5-Flash, and BioMistral-7B show stronger context sensitivity is worth noting since your Qwen3:14b has reasoning capability but inconsistent grounding.

**Relevance rating:** ★★★★☆ — directly cite this as motivation for your safety checker.

---

#### B. EULAR/ACR Rheumatology RAG (Madrid-García et al., Elsevier 2025)
**What it does:** Integrates 74 EULAR + ACR clinical guidelines into a RAG system using hybrid retrieval (dense voyage-3 + BM25), Qdrant vector store, ChatGPT o3-mini as generator, Gemini 2.0 Flash as judge. Evaluated on 740 specialist-level questions across 5 Likert criteria; human rheumatologist validation on 15%.

**Key findings:**
- RAG outperformed baseline on factual accuracy, safety, completeness (P<.001) in both LLM-judge and human evaluation.
- RAG preferred in 92.8% of pairwise comparisons (LLM judge) and 71–75% by human rheumatologists.
- Their limitation: omitted reranking for speed — mirrors your finding that cross-encoder reranker hurt recall.
- **Completeness given retrieval vs. completeness overall gap:** RAG answers sometimes miss content not in the top retrieved chunks — this is your Context Recall problem showing up identically.

**How it maps to you:** This is your closest published peer. Both use hybrid retrieval, guideline KB, structured output, LLM-as-judge. Their evaluation pipeline (LLM judge + 15% human validation) is what yours needs. The 74-guideline scale vs. your 4-document KB is the principal gap.

**Relevance rating:** ★★★★★ — directly cite as the closest published parallel; your FYP extends this paradigm to the wound care + mobile app context.

---

#### C. Guide-RAG (DiGiacomo et al., NeurIPS 2025 Workshop)
**What it does:** Evaluates 6 corpus configurations for Long COVID clinical QA: guideline-only, guideline+systematic reviews (GS-4), references-only, PubMed, web search, no retrieval. Metrics: faithfulness, relevance, comprehensiveness (LLM-as-judge pairwise).

**Key finding:** GS-4 (1 guideline + 3 systematic reviews = 4 documents) outperformed everything including PubMed (39M articles). Larger unfiltered corpora produced authoritative-sounding but clinically misleading answers (e.g., endorsing exercise for Long COVID patients where contraindicated).

**Direct lesson for you:** Your 4-document KB is NOT a weakness — it is the right design choice. Guide-RAG proves that a small, curated corpus of high-quality clinical guidelines consistently beats large unfiltered literature retrieval for clinical recommendation. When you expand your KB (next steps), prioritise curated guidelines over any idea of connecting to PubMed.

**Relevance rating:** ★★★★★ — use this to defend your KB design decision in your FYP report.

---

#### D. MEREDITH (Lammert et al., JCO Precision Oncology 2024)
**What it does:** Expert-guided iterative RAG for molecular tumor board treatment recommendations. Used chain-of-thought prompting, multi-source KB (PubMed, clinical trials, guidelines, drug availability), Gemini Pro. Evaluated qualitatively by MTB experts + cosine similarity quantitatively.

**Key finding:** Concordance jumped from 77.1% (draft, PubMed-only) to 94.7% (enhanced, multi-source) when domain-specific KB was added. Mirroring the expert's reasoning process (how an MTB expert thinks step-by-step) in the prompt was the biggest single improvement — which maps directly to your v3 → v4 transition (structured system prompt + binding algorithm block).

**How it maps to you:** MEREDITH's "draft → enhanced" iteration mirrors your v2 → v4 journey. Their key move — adding a binding algorithm block that forces the LLM to follow the clinical decision framework — is exactly your v4 G1 binding algorithm block. You arrived at the same design independently.

**Relevance rating:** ★★★★☆ — cite as independent validation of your v4 design choice.

---

#### E. RAGMed (Patil et al., MDPI AI 2025)
**What it does:** EHR-based RAG assistant for FAQ answering, appointment scheduling, clinical note summarisation. Compares GTE-Large (1024-dim) vs. all-MiniLM-L6-v2 (384-dim) embeddings using RAG-Triad (context relevance, answer relevance, groundedness). GTE-Large: avg 0.72/0.70/0.47 vs. MiniLM: 0.53/0.61/0.31.

**Key finding:** Higher embedding dimensionality → better retrieval → better downstream generation quality. GTE-Large 35% better on groundedness.

**How it maps to you:** You use `abhinand/MedEmbed-large-v0.1` (a medical embedding model, also large-scale). This validates your embedding choice. Their RAG-Triad vs. your RAGAS — both measure the same three dimensions (just labelled differently). The concern: your groundedness/faithfulness scores (0.62–0.74) are higher than RAGMed's GTE-Large (0.47), which is partly explained by your explicit grounding system prompt (v3+).

**Relevance rating:** ★★★☆☆ — useful comparison; your system is architecturally more advanced.

---

#### F. MasonNLP at MEDIQA-WV 2025 (Wound-care VQA)
**What it does:** Multimodal RAG for wound care — takes wound images + patient queries, retrieves textual and visual exemplars (few-shot), generates free-text wound care responses using LLaMA-4 Scout 17B. Ranked top-5 in a formal shared task benchmark.

**Key finding:** Exemplar retrieval (RAG over similar wound cases) improves both clinical accuracy and schema adherence in structured wound attribute extraction. Text-only retrieval baseline underperformed text+image retrieval.

**Direct lesson for you:** The wound image adds genuine signal beyond what text alone provides. Implementing even a lightweight form of image-based input (VLM pre-processing as described in Section 1) is justified by this benchmark.

**Relevance rating:** ★★★★★ — most directly relevant to your multimodal question; cite as the state-of-the-art in wound-care VQA RAG.

---

#### G. NICE Clinical Guidelines RAG (arXiv 2510.02967, 2025)
**What it does:** RAG over 300 NICE guidelines (10,195 chunks). Hybrid embedding retrieval. Faithfulness with RAG: 99.5% (vs. baseline LLM: ~35%). Uses O4-Mini as generator.

**Key finding:** Hybrid retrieval (dense + sparse) is the dominant factor in faithfulness improvement. RAG adds 64.7 percentage points in faithfulness on clinical guideline QA.

**How it maps to you:** Confirms your v4's faithfulness improvement trajectory is realistic and that 72–74% faithfulness is an intermediate milestone, not the ceiling.

**Relevance rating:** ★★★★☆ — directly validates your architecture direction; quantifies the ceiling you should aim for.

---

#### H. DM-WAT (Saadati Fard et al., IEEE JTEHM 2025)
**What it does:** Multimodal wound referral decision tool. DeiT-Base-Distilled (ViT) for wound image features + DeBERTa-base for clinical notes → intermediate fusion → referral classification (refer/monitor/continue). Tested on 205 wound images.

**Direct lesson for you:** This is the multimodal alternative to your approach. Instead of multimodal RAG, they fused image + text features for classification (not generation). The limitation of their approach: it only does referral classification, not dressing recommendation. Your RAG approach is more informative and actionable for your use case.

**Relevance rating:** ★★★☆☆ — useful to cite as a comparison approach; your RAG-first approach produces richer, more actionable outputs.

---

#### I. SCARWID (arXiv 2502.20277, 2025)
**What it does:** Wound-BLIP (VLM fine-tuned on GPT-4o-generated wound descriptions) + cross-attention fusion → wound infection classification. Sensitivity 0.85, specificity 0.78, accuracy 0.81.

**Key finding:** GPT-4o-generated wound image captions are a viable synthetic data source for training domain-specific wound VLMs. The VLM description enriches downstream classification significantly over image-only or text-only approaches.

**Direct lesson for you:** You can use GPT-4o to generate wound image descriptions from your (or Student A/B's) wound images, and use these descriptions as the "image pre-processing" step to enrich your RAG query. No need to fine-tune a VLM from scratch.

**Relevance rating:** ★★★★☆ — practical pathway for your multimodal extension.

---

### 2.3 Convergent Design Principles Across the Literature

After surveying all reviewed papers, the following design principles appear consistently in high-performing medical RAG systems:

**On Knowledge Base Construction:**
- Curated clinical guidelines (4–74 documents) outperform unfiltered large literature corpora (Guide-RAG, MEREDITH, EULAR RAG, NICE RAG)
- Manual noise removal (non-essential sections, references, affiliations) before ingestion is standard practice (EULAR RAG, MEREDITH)
- Metadata-enriched chunks (guideline name, authority, year, wound type) improve retrieval precision (NICE RAG, your v4 metadata filter)

**On Retrieval:**
- Hybrid dense + BM25 is the dominant retrieval strategy in 2025 (EULAR RAG, NICE RAG, MEREDITH enhanced)
- General-domain cross-encoders hurt domain-specific retrieval; medical rerankers or no reranker is preferable (your finding confirmed by EULAR RAG omitting reranking)
- Multi-axis sub-query retrieval (your v4 R2) is independently endorsed by MEREDITH's multi-source parallel retrieval

**On Generation:**
- Explicit grounded system prompts ("cite source X," "use only retrieved evidence") are essential (CARE-RAG finding, your v3 fix)
- Mandatory injection for safety-critical phrases (your v4 G2) appears across MEREDITH, Guide-RAG — not stated explicitly but present in prompt design
- Chain-of-thought + structured output sections (T.I.M.E. sections, MEREDITH's CoT) consistently improve clinical alignment
- 7-sentence response caps (EULAR RAG) vs. your uncapped responses — consider a length constraint for the mobile app UI

**On Evaluation:**
- RAGAS + human expert review on 15% is the gold standard (EULAR RAG used this exact split; CARE-RAG uses clinician-validated questions)
- Rule-based safety checkers on top of RAGAS are underused in literature but are your most clinically meaningful metric — this is a methodological contribution of your FYP
- LLM-as-judge (gpt-4o-mini) is now standard, but cross-generator reproducibility (your dual-generator ablation) is rare in the literature and strengthens validity

---

## 3. Most Relevant Parallel Work to Your FYP

Your system's closest published analogues, ranked by similarity:

| Rank | Paper | Similarity to your FYP | Key gap |
|---|---|---|---|
| 1 | MEDIQA-WV 2025 (MasonNLP + EXL Health) | Wound care, image + text input, RAG, clinical recommendation | They have a wound image dataset; you route through T.I.M.E. classification |
| 2 | EULAR/ACR RAG (Madrid-García et al. 2025) | Clinical guideline RAG, hybrid retrieval, dual evaluation, mobile point-of-care target | Rheumatology domain; no safety rule checker; no multimodal |
| 3 | Guide-RAG (DiGiacomo et al. 2025) | Small curated guideline corpus, clinical recommendation, faithfulness focus | Long COVID domain; no structured clinical input; no multi-architecture ablation |
| 4 | MEREDITH (Lammert et al. 2024) | Multi-source KB, binding algorithm block, expert validation, iterative refinement | Oncology; no mobile app; no RAGAS evaluation |
| 5 | NICE RAG (arXiv 2025) | Clinical guideline grounding, faithfulness measurement | Static QA, no structured input, no safety checker |

**Your unique contribution vs. all of them:** None of the reviewed papers combine (a) structured clinical input (T.I.M.E.) → (b) wound-type classifier → (c) metadata-filtered multi-axis retrieval → (d) binding algorithm block + mandatory injection → (e) rule-based domain safety checker → (f) dual-generator ablation study → (g) mobile app deployment. This combination is novel at FYP level and competitive at publication level.

---

## 4. Where Your Architecture Sits in the Literature

```
ARCHITECTURE MATURITY SPECTRUM (2024-2025)

  NAIVE RAG              ADVANCED RAG             MODULAR / AGENTIC RAG
  (dense only,           (hybrid, reranker,        (classifier, multi-axis,
   basic prompt)          grounded prompt)          binding rules, verifier)
       |                       |                           |
  [v2 series]            [v3 series]                 [v4 series]
  
  You are here: v4_02 → sits at the Advanced-to-Modular boundary.
  Compared to literature: more engineered than EULAR RAG (no reranker, no classifier)
                          less complex than full agentic RAG (no iterative retrieval)
                          closest to MEREDITH enhanced + MEDIQA-WV metadata-guided
```

Your v4_02 is not naive. It implements the key features that the literature identifies as separating good from great clinical RAG: domain classifier, metadata filtering, binding algorithm block, mandatory injection. What puts it in the "advanced" rather than "agentic" category is the absence of iterative query refinement and the absence of multimodal input — both are reasonable future work extensions.

---

## 5. Limitations of Your Current System

These are the limitations you must disclose in your FYP report, now grounded in the literature:

### 5.1 Knowledge Base Scope (most critical)
4 clinical documents is the most important constraint. Guide-RAG proves 4 can be enough *if they are the right 4* — but your persistent `cat_d_notes_diabetic_nonhealing` failure and `dressing_in_allowed_list` failures across multiple cases suggest specific coverage gaps. The EULAR RAG paper (74 guidelines) and the NICE RAG paper (300 guidelines) show the direction. For your mobile app's intended clinical scope, you need at minimum: a diabetic foot wound guideline, a burns referral guideline, a skin tear management guideline, and an NPWT indications guideline (the 4 gaps identified in your previous analysis).

### 5.2 Text-Only Retrieval Against a Visual Clinical Task
As MEDIQA-WV 2025 confirms, wound assessment inherently relies on visual cues. Your pipeline routes around this via T.I.M.E. classification (Student A/B's models), but those models have known accuracy limitations (Student A's k-means tissue segmentation is noted as "not so good"). If the upstream T.I.M.E. classification is wrong, your RAG recommendation will be wrong — this is a propagation-of-error risk that the literature (DM-WAT, SCARWID) addresses via visual grounding.

### 5.3 Testset Size and Construct Circularity
28 cases, all derived from the same 4 KB documents, evaluated against references also derived from those documents. As noted in the CARE-RAG paper and the EULAR RAG limitations, AI-generated test questions can introduce bias and may not represent real clinical practice. Your FYP should acknowledge that 28 cases provides adequate ablation comparison but not clinical deployment validation.

### 5.4 Single-Run Variance (no confidence intervals)
The literature (EULAR RAG, RAGMed) does not estimate RAGAS variance either, but the RAG-X paper (arXiv 2603.03541) and the CARE-RAG discussion flag that LLM-as-judge scoring has non-trivial variance. Your scores are point estimates, not distributions.

### 5.5 Patient-Facing Safety Framing
Your app's target users are *patients*, not clinicians. CARE-RAG, Guide-RAG, and all reviewed papers target clinician-facing tools. For patient-facing deployment, the safety standard is higher — you need clear disclaimers, escalation pathways, and simpler language. The app UI must communicate uncertainty and always include "seek medical attention if X."

### 5.6 No Verified Calibration Against Human Clinical Judgment
None of your 18 ablation versions have been evaluated by a wound care nurse or GP. The EULAR RAG paper (P<.001 human validation), MEREDITH (MTB expert review), and MEDIQA-WV (clinical advisory) all include some human validation. For your FYP, even 1–2 wound nurses reviewing 10 cases of v4_02's output would significantly strengthen the work.

---

## 6. Next Steps (Prioritised)

### Priority 1 — Immediate (before FYP submission)

**1a. Add VLM image pre-processing as optional RAG input enrichment**
- Use GPT-4o Vision API (or locally Qwen2-VL if GPU available) with the wound image
- Prompt: *"You are a wound care assistant. Describe this wound image in clinical T.I.M.E. terms as JSON: {tissue_necrotic_pct, tissue_slough_pct, tissue_granulation_pct, infection_signs, exudate_level, edge_status, additional_observations}."*
- Append this JSON as `visual_description` to your v4_02 assessment text
- This gives you the "multimodal" capability without rebuilding your architecture
- Expected benefit: improves cases where Student A/B classification errors propagate into wrong RAG queries

**1b. Fix the `cat_d_notes_diabetic_nonhealing` testset ambiguity**
- Add the IWGDF Diabetic Foot Guidelines 2023 (free PDF) to your KB
- Re-ingest with same metadata schema
- Re-run v4_02 only (no need to re-run all 9 versions)
- Expected benefit: this single case that fails in both generators for all versions likely resolves

**1c. Write the human clinical evaluation (lightweight)**
- Have 1 wound care nurse or GP review 10 v4_02 outputs against the reference answers
- Use the same 5-criterion Likert scale as EULAR RAG (relevance, factual accuracy, safety, completeness, conciseness)
- This costs ~2 hours of a clinician's time and dramatically strengthens your FYP

---

### Priority 2 — High Value (if time allows)

**2a. Expand KB with 4 targeted guidelines**
- IWGDF Diabetic Foot 2023
- ISTAP Skin Tear Classification (free)
- ANZBA Burns First Aid (free)
- EWMA Wound Odour Position Document (free)
- Expected benefit: closes the 4 recurring failure categories in your safety checker

**2b. Retroactively fix Answer Relevancy for v2 series**
- Generate narrative queries for the 6 v2 JSONs using `build_narrative_query()` from v3
- Re-run RAGAS AR only (cheapest metric, just needs embeddings)
- This makes your cross-version AR comparison valid for your FYP report

**2c. Add a response length constraint**
- Cap generated responses at 7 sentences for the mobile app output (following EULAR RAG precedent)
- Your current avg response is 2,700–3,200 chars — far too long for a mobile patient-facing screen
- This should be a separate app-layer formatting step, not a change to the RAG pipeline

---

### Priority 3 — Future Work / Post-FYP

**3a. Full multimodal RAG (visual exemplar retrieval)**
- Build a wound image exemplar database annotated with T.I.M.E. labels and dressing outcomes
- Use CLIP or BioViL-T embeddings to retrieve visually similar wound cases
- Feed retrieved image exemplars + clinical text into generation
- This is the MEDIQA-WV 2025 state-of-the-art architecture — excellent Master's thesis topic

**3b. Medical domain cross-encoder reranker**
- Replace `cross-encoder/ms-marco-MiniLM-L-6-v2` with a biomedical reranker
- Options: `abhinand/MedEmbed-reranker-v0.1`, `ncats/pmc_llama_13b_reranker`
- Test as v4_03

**3c. Larger KB with systematic review integration**
- Following Guide-RAG's GS-4 approach: add 3 high-quality systematic reviews on wound care
- Options: Cochrane wound care reviews, Wounds International consensus documents, EWMA position papers
- Expected benefit: covers emerging wound types not in your current 4 guidelines

**3d. Agentic RAG with iterative query refinement**
- If the classifier misidentifies a wound type, allow the system to issue a follow-up retrieval query
- Based on i-MedRAG (Xiong et al., 2024) and MED-RWR (arXiv 2510.18303)

**3e. Variance estimation**
- Run v4_02 RAGAS 3× with fixed random seed
- Report mean ± SD for all 4 metrics
- This is a single overnight run, publishable-level rigour

---

## 7. FYP Progress Rating: 72 / 100

### Breakdown

| Dimension | Score | Reasoning |
|---|---|---|
| **RAG Architecture Design** | 18/20 | v4_02 implements all standard components identified in the 2024–2025 literature: hybrid retrieval, narrative query, grounded system prompt, domain classifier, metadata filtering, binding algorithm block, mandatory injection. The only missing piece for top marks is multimodal input. |
| **Evaluation Methodology** | 17/20 | Dual-generator ablation across 9 versions with RAGAS + rule-based safety checker is methodologically stronger than most published papers (MEREDITH, Guide-RAG). The AR-in-v2 measurement artefact, absence of variance estimation, and no human clinical validation prevent full marks. |
| **Knowledge Base** | 10/20 | 4 documents is defensible (Guide-RAG proves 4 can work) but the identified coverage gaps (diabetic foot, burns, skin tear, NPWT) still cause failures in the best-performing version. The KB is the single largest unresolved weakness. |
| **Clinical Safety Contribution** | 14/15 | Your rule-based domain safety checker on top of RAGAS is a genuine methodological contribution not present in most reviewed papers. The persistent `cat_d_notes_diabetic_nonhealing` edge case and the AR measurement issue prevent full marks. |
| **App Integration Readiness** | 8/15 | The RAG backend (v4_02) is deployment-ready. The app integration — mobile UI, response formatting for patients, VLM image pre-processing, connection to Student A/B models — is still ahead. No human clinical validation yet. |
| **Literature Positioning** | 5/10 | The work is clearly of publishable quality with the right framing, but you have not yet explicitly positioned it against the closest papers (EULAR RAG, MEDIQA-WV). Adding that comparison in your report would strengthen the academic contribution significantly. |

**Total: 72 / 100**

### What would take you to 80+

- Add 2–4 KB documents (Priority 1b) → +4 pts
- Add VLM image pre-processing step (Priority 1a) → +3 pts
- Conduct lightweight human clinical evaluation on 10 cases (Priority 1c) → +4 pts
- Fix v2 Answer Relevancy comparison (Priority 2b) → +2 pts

All four of those are achievable in the remaining FYP timeline. They would bring you to ~85/100 — comfortably above the distinction threshold for most Malaysian university FYP rubrics.

### What you should be proud of

- Your v4_02 architecture independently converges on design decisions that the best 2025 papers (MEREDITH, EULAR RAG, MEDIQA-WV EXL) also arrived at — the binding algorithm block, mandatory injection, and hybrid retrieval. This is not luck; it reflects sound iterative engineering.
- A 9-version ablation study with dual-generator evaluation is rare in the literature. Published papers rarely ablate this systematically. CARE-RAG (NeurIPS 2025 Workshop) tests 20 LLMs but uses a single architecture; you test 9 architectures across 2 generators. This is a genuine contribution.
- Going from 28% safety pass rate (v2_00 GPT) to 96% (v4_02, both generators) is a compelling headline result that maps directly to clinical relevance — you reduced unsafe recommendations by more than 3×.
- You understand the *reasons* behind every metric movement. That analytical depth is what distinguishes strong FYPs from average ones.

---

## References (Literature Reviewed)

1. Potluri, D. et al. (2025). CARE-RAG: Clinical Assessment and Reasoning Evaluation for RAG. NeurIPS 2025 Workshop on GenAI for Health.
2. Madrid-García, A. et al. (2025). Optimising the clinical application of rheumatology guidelines using LLMs: a RAG framework integrating EULAR and ACR recommendations. EULAR Rheumatology Open.
3. DiGiacomo, P. et al. (2025). Guide-RAG: Evidence-Driven Corpus Curation for RAG in Long COVID. NeurIPS 2025 Workshop on GenAI for Health.
4. Lammert, J. et al. (2024). Expert-Guided Large Language Models for Clinical Decision Support in Precision Oncology (MEREDITH). JCO Precision Oncology.
5. Patil, R. et al. (2025). RAGMed: A RAG-Based Medical AI Assistant for Improving Healthcare Delivery. AI (MDPI).
6. Yim, W. et al. (2025). Overview of the MEDIQA-WV 2025 Shared Task on Woundcare Visual Question Answering. ClinicalNLP 2025, ACL.
7. Durgapraveen, B. et al. (2025). EXL Health AI Lab at MEDIQA-WV 2025: Mined Prompting and Metadata-Guided Generation for Wound Care VQA. ACL Anthology.
8. Saadati Fard, R. et al. (2025). Multimodal AI for Home Wound Patient Referral Decisions From Images With Specialist Annotations. IEEE JTEHM.
9. Xu, Y. et al. (2025). SCARWID: Explainable Multi-modal Wound Infection Classification. arXiv 2502.20277.
10. Grounding LLMs in NICE Clinical Guidelines (2025). arXiv 2510.02967.
11. Wang, J. et al. (2025). MIRA: A Novel Framework for Fusing Modalities in Medical RAG. ACM Multimedia 2025.
12. Bunnell, D.J. et al. (2025). Bridging AI and Healthcare: A Scoping Review of RAG. medRxiv.
13. Xiong, G. et al. (2024). MIRAGE + MEDRAG: Benchmarking RAG for Medicine. ACL 2024 Findings.
