# VerdaSense — FYP2 Migration Rationale Analysis

**Student:** Tee Qi Jing (23004894) · Universiti Malaya  
**Date:** June 2026 · Post-FYP1 Viva  
**Purpose:** Honest feasibility analysis of the proposed FYP2 migration — what to carry forward, what to add, what to be realistic about, and how to structure the ablation for a defensible, standout FYP2.

---

## Part 0 — Why the Desire to Migrate Is Justified

Your FYP1 Viva exposed a real tension: your T.I.M.E.-only structured RAG is technically well-executed but *experientially narrow*. The panel's questions ("why not rule-based?", "how do you know the output is correct?") were partly probing technical gaps — but they were also partly reacting to the fact that a system answering only "which dressing for this T.I.M.E. wound type" does not feel like a clinically useful tool to a non-technical examiner. They expected something closer to an AI doctor and got a structured dressing lookup with evidence grounding.

This is not a failure of your FYP1. It is a scope misalignment between what "RAG for wound care" sounds like and what your narrowly scoped system delivers. FYP2 is the right time to close that gap — but closing it requires distinguishing between what is genuinely achievable in one academic year, what is technically feasible but academically risky, and what is aspirational but out of scope.

This document gives you that honest analysis.

---

## Part 1 — What You Are Proposing (Structured Summary)

You have described three connected proposals for FYP2:

**Proposal A — Wound Category Classification (New CV Module)**  
Train or fine-tune a classification model to assign a wound category label (DFU, VLU, Pressure Ulcer, Burn, Abrasion, Cut, Surgical, Laceration, Vascular) from the wound image, in addition to the existing T.I.M.E. labels from IME-Net and K-Means. This would feed into the RAG as an additional metadata filter axis — selecting KB chunks appropriate for the specific wound category, not just the wound type 1–8.

**Proposal B — Conversational AI Doctor Tab (Multi-turn RAG)**  
A second interface tab where the patient can have a free-form, multi-turn conversation with an AI about any aspect of their wound — questions about their dressing, wound progression, when to seek help, why their wound is not healing. The conversation is grounded in the wound care KB (same 8 sources, possibly expanded). The T.I.M.E. assessment from the CV pipeline is injected as context at session start. Memory persists across turns within a session.

**Proposal C — UX Upgrades (STT/TTS, dressing product gallery)**  
Voice input (Whisper STT) and voice output (edge-TTS) for elderly patients who struggle to type. A dressing product gallery showing Malaysian-available products for each recommended dressing type. Multi-language support (BM/English). These are app layer concerns handled by the SE student collaborator, not directly your RAG research scope.

---

## Part 2 — Honest Feasibility Analysis Per Proposal

### Proposal A: Wound Category Classification

**What is needed:**
- A labelled dataset of wound images with category-level ground truth labels (DFU, VLU, Burn, etc.)
- Fine-tuning or training a classification model (EfficientNet, ResNet, or ViT-based) on this dataset
- Integration of the category label as a ChromaDB metadata filter axis
- KB expansion: your current 8 sources do not adequately cover DFU, VLU, and Vascular Ulcer wound categories — you would need at minimum 2–3 additional specialist guideline sources for these

**The dataset problem (your biggest realistic risk):**  
This is where Proposal A could collapse. Public wound image datasets with category-level labels exist but are small, imbalanced, and often not representative of Southeast Asian patient demographics. AZH Wound Care dataset, Medetec, and MICCAI wound challenge datasets are the main options. If Ms Saw cannot provide de-identified images, you are building and evaluating on a dataset that does not represent your actual patient population — which creates a clinical validity problem more severe than your FYP1 testset construction issue.

**Feasibility verdict:** Technically feasible if dataset is secured early (first 4 weeks of FYP2). If dataset is not secured by week 6, drop Proposal A from FYP2 scope and redesign as a "proposed future work" section. Do not let dataset acquisition block your RAG development progress.

**If Proposal A succeeds, what it adds to your FYP:**  
- The RAG metadata filter now has two axes: wound type (WT1–8 from T.I.M.E.) AND wound category (DFU/VLU/etc.). This is a genuinely novel RAG contribution — hybrid classification-driven metadata filtering for clinical guideline retrieval.
- Ms Saw's explicit request for vascular wound classification is satisfied with clinical endorsement.
- Your R1 ablation extends: a new sub-query axis (Sub-query D: wound category axis) can be tested and ablated.

**What to ask Ms Saw before committing:**  
"Can you provide or point us to a labelled dataset of wound category images — even 50–100 images across 5–6 wound types — for model validation? Without clinical image validation, the classifier's accuracy cannot be verified for Malaysian patients."

---

### Proposal B: Conversational AI Doctor Tab (Multi-turn RAG)

**This is your most important and most defensible FYP2 contribution.** Here is why.

**What multi-turn conversational RAG adds that your FYP1 system cannot do:**

Your FYP1 system is single-turn: one T.I.M.E. payload in, one structured recommendation out. A patient cannot ask "why is silver dressing better than normal gauze for my wound?" or "my wound smells worse after 3 days, what should I do?" or "I can't find alginate at the pharmacy, what can I use instead?" These are the questions real self-care patients actually have. Zero-shot GPT-4o can answer these questions (with inconsistent clinical grounding). Your conversational RAG answers them with retrieval-grounded evidence from the same 8 clinical KB sources — which is exactly the architectural advantage you failed to articulate in your FYP1 Viva.

**Technical architecture for Proposal B (what you actually need to build):**

```
Session initialisation:
  1. CV pipeline runs on wound image → T.I.M.E. payload
  2. System prompt injected with: T.I.M.E. assessment + wound type + session context
  3. Initial dressing recommendation generated (your existing FYP1 RAG — no change)

Multi-turn conversation:
  4. Patient asks follow-up question (text or voice via Whisper)
  5. Each turn: question → multi-axis retrieval from KB → top-K chunks injected as context
  6. Conversation history maintained in session (LangChain ConversationBufferMemory or equivalent)
  7. LLM generates response grounded in KB + conversation history + initial T.I.M.E. context
  8. Safety checker runs on each generated turn (not just the first recommendation)

Session end:
  9. Conversation summary can be exported as patient care log
```

**What this is technically:**  
This is a RAG-augmented conversational agent with session memory and a wound-image-seeded initial context. It is NOT a general-purpose medical AI. It is wound-care-specific, KB-grounded, and initiated from a T.I.M.E. assessment. This scope is defensible and clinically appropriate.

**Why this is more academically rigorous than Cyber-Doctor (赛博华佗):**  
Cyber-Doctor is an impressive general-purpose medical AI assistant. It uses Neo4j knowledge graph + RAG + TTS/STT — a broad architecture for broad medical questions. Your FYP2 Proposal B is *domain-specific*: wound care only, KB-grounded on 8 clinically validated sources, initialised from a structured CV assessment. This narrower scope is actually *harder to evaluate rigorously* — because you can measure whether each conversational turn is grounded in the wound care KB, something Cyber-Doctor's general-purpose design does not attempt to measure. Your ablation study will be more rigorous than theirs precisely because your domain is narrow.

**On Knowledge Graph (Neo4j) — should you adopt it?**  
Honest answer: No, not for FYP2. Here is why.

Neo4j knowledge graphs are valuable when your data has rich relational structure that vector similarity cannot capture — for example, "which dressings are contraindicated for patients who have both iodine allergy AND diabetes AND are on anticoagulants?" This multi-hop relational query genuinely benefits from a graph traversal that vector search cannot replicate. However, building a wound care knowledge graph requires:
- Manual entity and relationship extraction from your 8 KB sources (minimum 40–60 hours of curation work)
- Neo4j schema design for wound care entities (wound types, dressing classes, contraindications, patient conditions, clinical outcomes)
- Cypher query generation (either manual or LLM-based Text-to-Cypher — both require validation)
- A new ablation experiment comparing ChromaDB RAG vs GraphRAG vs Hybrid

This is a full FYP in itself. The risk-to-benefit ratio for FYP2 is unfavourable: you would spend 60% of your time on the knowledge graph and have less time to build and evaluate the conversational RAG that is your actual clinical contribution. Cyber-Doctor's knowledge graph is impressive as a feature — but your FYP will be judged on research rigour, not feature count.

**What to do instead of Neo4j:** Use structured metadata filtering (your existing ChromaDB metadata schema: `wound_category`, `wound_type`, `guideline_type`, `authority`) as a lightweight substitute for graph relationships. Add one new metadata field: `contraindication_type` (flags chunks that specifically address contraindications). This gives you most of the relational benefit without the Neo4j overhead.

**Feasibility verdict for Proposal B:** High. Your existing ChromaDB retrieval pipeline, LangChain stack, and LLM generation pipeline can be extended to multi-turn with session memory in approximately 1–2 weeks of development. The hard work is the new ablation study design and the expanded testset — which is your actual academic contribution.

---

### Proposal C: STT/TTS, Product Gallery, Multi-Language

**These are all UX concerns for the SE student collaborator.** Your RAG API contract does not change — the SE student builds a frontend that calls your existing generation endpoint. From your FYP research perspective:

- **STT (Whisper):** Transcribed voice input becomes a free-text patient notes field — no change to your retrieval pipeline.
- **TTS (edge-TTS):** Post-generation audio rendering — no change to your generation pipeline.  
- **Dressing product gallery:** Static data (dressing type → Malaysian products mapping confirmed by Ms Saw) — frontend concern.
- **Multi-language (BM):** If you want BM-language output, you add a `language` parameter to your generation prompt and test whether the LLM correctly generates BM clinical language. This is a 1-experiment ablation (G4: Language) — simple and fast.

**Do not spend your FYP2 research effort on Proposal C.** Your contribution is the RAG architecture, the conversational extension, the wound category integration, and the ablation study. The SE student owns the frontend.

---

## Part 3 — The Recommended FYP2 Scope

Based on the feasibility analysis, here is the recommended scope that is ambitious, defensible, achievable in one year, and genuinely advances the state of the art for wound care RAG in Malaysia:

### Core Contribution (Must-deliver)

**C1 — Conversational Wound Care RAG (Proposal B)**  
Multi-turn KB-grounded conversational agent for patient wound care Q&A, initialised from T.I.M.E. assessment context. This is your primary FYP2 research contribution.

**C2 — Referral and Safety Logic Update**  
Update `classify_wound()` and testset `referral_required` fields to reflect Ms Saw's confirmed clinical guidance: all locally infected wounds require referral. Re-run G2-D and G3-G safety evaluation. Report corrected Safety Pass Rate.

**C3 — Out-of-Distribution (OOD) Handling + Category F Testset**  
Add 6–8 Category F test cases (out-of-KB-distribution) with reference answer = abstention/refusal. Add Appropriate Abstention Rate as a new evaluation metric. Add retrieval confidence threshold (cosine < 0.45 → low-confidence flag).

**C4 — Clinician Human Evaluation**  
3-part evaluation with Ms Saw: blinded Likert rating (8 Cat A cases), RAG vs zero-shot comparison (4 cases), root-cause failure analysis. This closes the FYP1 Viva gap on "how do you know it's clinically accurate."

### Secondary Contribution (Should-deliver if dataset secured)

**S1 — Wound Category Classification (Proposal A)**  
Train a wound category classifier (EfficientNet or ViT-based) on a secured labelled dataset. Integrate category label as an additional ChromaDB metadata filter axis. Add KB sources for DFU, VLU, Vascular Ulcer. Ablate the new classification-driven retrieval axis (R6: Wound Category Metadata Filter).

### Future Work (Document but do not implement in FYP2)

**F1 — Knowledge Graph (Neo4j GraphRAG)**  
Documented architecture design for a wound care knowledge graph based on the 8 KB sources. Entity schema, relationship types, and proposed Cypher query patterns. Estimated 3-month implementation effort for a future researcher.

**F2 — Network Retrieval Enhancement**  
Automated crawling of MOH Malaysia and clinical guideline update feeds to maintain KB currency. Not applicable to FYP2 scope.

---

## Part 4 — The New Ablation Study Map for FYP2

Your FYP1 ablation covered R1–R5 (retrieval) and G1–G3 (generation). FYP2 should extend this with:

### New Retrieval Ablation

| Experiment | Research Question | What You Test |
|---|---|---|
| **R6** | Does wound category metadata filtering improve retrieval for category-specific cases? | Dense retrieval with wound_category filter vs without, on Category B–E cases (burn, DFU, VLU) |
| **R7** | Does conversation history injection improve retrieval in multi-turn sessions? | Retrieval with only current turn query vs retrieval with [history + current turn] as query |

### New Generation Ablation

| Experiment | Research Question | What You Test |
|---|---|---|
| **G4** | Does multi-turn session context improve recommendation quality for follow-up questions? | Single-turn vs multi-turn generation on a new conversational testset |
| **G5** | What is the Appropriate Abstention Rate for OOD cases? | Category F testset (6–8 cases) — measure refusal vs hallucination rate |
| **G6 (optional)** | Does BM language output degrade clinical accuracy? | English output vs BM output on 8 Cat A cases — FA and Safety Pass Rate comparison |

### New Evaluation Metrics

| Metric | What It Measures | Applies To |
|---|---|---|
| **Appropriate Abstention Rate** | % of OOD cases where system correctly refuses to recommend | Category F testset |
| **Conversational Faithfulness** | FA measured across multi-turn sessions (each turn individually) | G4 conversational testset |
| **Session Coherence** | Does the LLM maintain consistent clinical advice across turns? (LLM-as-judge) | G4 conversational testset |
| **Clinical Concordance Rate** | % of Cat A recommendations rated ≥ 4/5 by Ms Saw | Human evaluation |
| **Turn-level Safety Pass Rate** | Safety checker pass rate applied to each conversational turn, not just first recommendation | G4 conversational testset |

### New Testset Requirements

You will need to construct:

1. **Conversational testset (20–30 sessions, 3–5 turns each):** Each session starts with a T.I.M.E. assessment (from your existing 32 cases), followed by patient-like follow-up questions ("can I shower with this dressing?", "the wound looks redder today", "I'm out of silver dressing, what can I use?"). Reference answers grounded in KB. This is your most significant testset construction work.

2. **Category F testset (6–8 cases):** Out-of-distribution wound presentations. Reference answer for each = the specific refusal phrase your prompt produces. Measures Appropriate Abstention Rate.

3. **Wound category testset (if S1 implemented):** Cases where wound category label is available and tested as retrieval filter input.

---

## Part 5 — The Two-Tab Architecture: How It Fits Together

Your proposed two-tab design is architecturally sound and creates a clean research narrative:

```
Tab 1 — Wound Assessment & Dressing Recommendation
┌─────────────────────────────────────────────────────────┐
│  Patient uploads wound image                             │
│  → CV pipeline: YOLO + T-SegNet + K-Means + IME-Net     │
│  → T.I.M.E. payload displayed to patient                 │
│  [Optional: Wound Category label from Proposal A CV]    │
│  → [Generate Dressing Recommendation] button            │
│  → VerdaSenseRAG: single-turn, structured output        │
│     (P.Dressing, S.Dressing, Contraindications,         │
│      Referral, ABx, Frequency, Application tips)        │
│  → Dressing product gallery (SE student builds)         │
└─────────────────────────────────────────────────────────┘

Tab 2 — AI Wound Care Assistant (Conversational RAG)
┌─────────────────────────────────────────────────────────┐
│  Session initialised with T.I.M.E. context from Tab 1   │
│  Patient asks: "Why can't I use normal gauze?"          │
│  → Retrieval: KB query on current question + history    │
│  → Generation: answer grounded in KB + session memory   │
│  → Safety checker: runs on every generated turn         │
│  Patient asks: "What if I can't find alginate?"         │
│  → [continues, multi-turn, memory-aware]                │
│  [Voice input/output via Whisper + edge-TTS — SE]       │
└─────────────────────────────────────────────────────────┘
```

**The critical research insight this architecture demonstrates:**  
Tab 1 proves your FYP1 contribution (T.I.M.E.-structured RAG with ablation-optimised configuration). Tab 2 extends it to show that the same KB-grounded retrieval system can power a broader conversational experience — *but remains constrained to KB-grounded evidence*, unlike zero-shot ChatGPT. The transition from Tab 1 to Tab 2 is a natural UX flow where the patient gets a recommendation first, then can ask follow-up questions grounded in the same clinical evidence that generated the recommendation.

---

## Part 6 — On the "AI Doctor" Aspiration: Honest Boundaries

You wrote that you want to make your FYP into an "AI Doctor / GPT in wound care." Here is the honest response to that aspiration.

**What your system will be after FYP2 (if you execute the plan above):**  
A KB-grounded, multi-turn, wound-image-aware conversational clinical decision support tool for self-care wound management in Malaysia, with a structured single-turn dressing recommendation capability, an optional wound category classification module, and a 3-part human clinical evaluation confirming its output quality. This is a serious, novel, clinically relevant research contribution that no existing academic wound care RAG system has fully implemented.

**What it will not be:**  
A general-purpose medical AI. It will not diagnose diseases, interpret lab results, prescribe medications, or answer questions outside wound care. It will not work reliably for wound categories with insufficient KB coverage. It will not replace clinical judgment — it will augment self-care decisions in the specific context of wound dressing selection and wound care management.

**Why this boundary is a strength, not a weakness:**  
Clinically appropriate AI systems are *narrow by design*. A system that says "I am a wound care assistant specialising in dressing selection according to Malaysian clinical guidelines" is more trustworthy than one that says "I am a medical AI that can answer anything about your health." Ms Saw confirmed this — she scoped the system to self-care wound management with mandatory referral triggers for complex cases. This is the clinically endorsed boundary. Own it.

**The sentence for your FYP2 Viva opening:**
> "VerdaSense is not designed to be a general medical AI. It is a wound-care-specific, KB-grounded clinical decision support system that provides structured dressing recommendations from T.I.M.E. assessment inputs and supports follow-up clinical Q&A through a conversational RAG interface — all grounded in 8 Malaysian and international clinical guidelines, with every claim traceable to a specific guideline source. The goal is not to replace clinical judgment, but to make evidence-based wound care guidance accessible to self-care patients in Malaysia who do not have immediate access to a clinician."

That is a defensible, clinically honest, and technically accurate description of what your FYP2 system is. No panel can question it, because it makes no overclaims.

---

## Part 7 — Priority Action Items Before FYP2 Begins

| Priority | Action | Deadline | Blocks |
|---|---|---|---|
| 🔴 **P1** | Update `classify_wound()` — add referral_required=True for all locally infected wounds | Week 1 | Corrected Safety Pass Rate for FYP1 submission |
| 🔴 **P1** | Update testset v3 referral_required fields accordingly and re-run G2-D + G3-G safety evaluation | Week 1 | Final FYP1 reported results |
| 🔴 **P1** | Confirm wound category image dataset source with Ms Saw | Week 2 | Proposal A feasibility decision |
| 🟡 **P2** | Design conversational testset schema (session structure, turn types, reference answer format) | Week 3 | G4 ablation can begin |
| 🟡 **P2** | Build Category F testset (6–8 OOD cases) | Week 3 | G5 Abstention Rate evaluation |
| 🟡 **P2** | Implement LangChain session memory + multi-turn retrieval pipeline | Week 4–5 | Tab 2 development |
| 🟢 **P3** | Expand KB with DFU/VLU/Vascular Ulcer sources (if Proposal A confirmed) | Week 4 | R6 ablation |
| 🟢 **P3** | Design clinician evaluation form (Google Form, Likert 1–5, 3 dimensions) | Week 4 | Ms Saw evaluation session |
| 🟢 **P3** | BM language output ablation (G6) — 1 afternoon experiment | Week 6 | Optional, fast |

---

## Summary Table: FYP1 vs FYP2 Comparison

| Dimension | FYP1 (Completed) | FYP2 (Proposed) |
|---|---|---|
| **Interaction model** | Single-turn (T.I.M.E. → recommendation) | Single-turn + Multi-turn conversational RAG |
| **Input** | Structured T.I.M.E. payload + optional notes | Same + wound category label (if S1) + voice (SE) |
| **KB** | 8 sources, 138 chunks | 8–11 sources, ~180–210 chunks |
| **Retrieval** | Multi-axis (3 sub-queries), dense semantic | Same + conversation history axis + category filter |
| **Generation** | Single structured recommendation | First turn recommendation + free-form conversational turns |
| **Safety** | Post-gen rule checker (first turn only) | Post-gen rule checker (every turn) + confidence threshold |
| **Evaluation** | RAGAS + IR metrics + Safety Pass Rate (automated) | All FYP1 metrics + Conversational FA + Session Coherence + Abstention Rate + Human Clinical Evaluation |
| **Testset** | 32 cases (5 categories) | 32 cases + 20–30 conversational sessions + 6–8 OOD cases |
| **Ablation** | R1–R4, G1–G3 (8 experiments) | + R6, R7, G4, G5, G6 (5 new experiments) |
| **Clinical validation** | Proposed (Ms Saw review pending) | Completed (blinded Likert + comparative + root-cause) |
| **Panel Q1 answer** | "Hybrid by design" (improved framing) | Demonstrated by two-tab architecture — rules + RAG + conversation |
| **Panel Q2 answer** | FA = 0.81, G1-A→G2-D delta framing | + Human Clinical Concordance Rate |
| **Panel Q3 answer** | Prompt-level graceful degradation | + Category F testset + Abstention Rate metric |
| **Panel Q4 answer** | FYP2 planned | Completed — 3-part clinician evaluation |

---

*VerdaSense FYP2 Migration Rationale · Tee Qi Jing (23004894) · Universiti Malaya · June 2026*
