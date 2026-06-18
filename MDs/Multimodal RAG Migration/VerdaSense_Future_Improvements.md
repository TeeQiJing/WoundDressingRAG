# VerdaSense RAG — Future Work & Production-Level Improvements
## Evidence-Grounded Clinical Decision Support for Wound Dressing Recommendation

**Document Type:** Future Work Ideation & Research-Backed Improvement Roadmap  
**Author:** Tee Qi Jing | Universiti Malaya | FYP2  
**Date:** May 2026  
**Status:** Research synthesis + original proposals — for FYP2 Future Work chapter and beyond

---

## Preface: Why This Document Exists

The current VerdaSense RAG achieves strong ablation results — BGE Large embeddings, R1-C multi-axis dense retrieval, G1-C grounded prompt, and Gemini 2.5 Flash as the optimal closed-source LLM (FA=0.8147, Safety=90.6%). However, the system has a fundamental structural dependency: its input quality is bounded by the accuracy of upstream companion computer vision models. If the companion CV pipeline produces an inaccurate T.I.M.E classification — e.g., misclassifying a moderate-exudate wound as low, or missing early infection signs — VerdaSense will generate a syntactically perfect but clinically wrong recommendation, with no awareness of the upstream error.

This document proposes major improvements to make VerdaSense production-level ready and clinically accurate, organized into six strategic themes. Each section begins with a problem statement, provides a literature anchor from recent (2024–2026) work, and ends with a concrete implementation path relevant to the existing VerdaSense architecture.

---

## Table of Contents

1. [Improvement I: Vision-Language Model (VLLM) as a Direct Wound Captioning Layer](#improvement-i-vision-language-model-vllm-as-a-direct-wound-captioning-layer)
2. [Improvement II: Multimodal RAG — Image-Grounded Retrieval Beyond Text Queries](#improvement-ii-multimodal-rag--image-grounded-retrieval-beyond-text-queries)
3. [Improvement III: Longitudinal Wound Tracking — Multi-Visit Context for RAG](#improvement-iii-longitudinal-wound-tracking--multi-visit-context-for-rag)
4. [Improvement IV: Confidence-Aware Uncertainty Output and Escalation Logic](#improvement-iv-confidence-aware-uncertainty-output-and-escalation-logic)
5. [Improvement V: Fine-Tuning the Generation LLM on Wound Care Recommendations](#improvement-v-fine-tuning-the-generation-llm-on-wound-care-recommendations)
6. [Improvement VI: Agentic Multi-Step Clinical Reasoning Pipeline](#improvement-vi-agentic-multi-step-clinical-reasoning-pipeline)
7. [Synthesis: Recommended Priority Roadmap](#synthesis-recommended-priority-roadmap)
8. [Architecture Diagram: VerdaSense Next-Gen](#architecture-diagram-verdaSense-next-gen)

---

## Improvement I: Vision-Language Model (VLLM) as a Direct Wound Captioning Layer

### 1.1 The Problem: Structured T.I.M.E. Inputs Are Information-Lossy

The current VerdaSense input pipeline is:

```
Wound Image → [YOLO Detection] → [MobileSAM Segmentation] → [K-Means Tissue Classification] → [IME-Net Classification] → Structured T.I.M.E. payload → VerdaSense RAG
```

Each stage of this pipeline compresses the original wound image into progressively more abstract labels. By the time VerdaSense receives the input, an enormous amount of potentially clinically meaningful visual information has been discarded — wound bed texture, periwound skin condition, exudate colour, wound depth cues, odour (inferable from appearance patterns), and maceration. The structured T.I.M.E. payload — e.g., `{necrotic: 5%, slough: 30%, granulation: 65%, infection: "Not infected", moisture: "Moderate", edge: "Non-advancing"}` — captures the abstracted classification outcome but loses all the richness of the original image. This means VerdaSense is essentially a text-only RAG system operating on coarse labels, despite being part of a visual pipeline.

**The key risk:** Companion CV models are themselves imperfect. IME-Net's classification performance on the I.M.E. axes may vary by wound type, lighting conditions, and patient skin tone. An incorrect companion model output — e.g., classifying a moderately infected wound as "Not infected" — propagates invisibly through the pipeline and produces a recommendation that VerdaSense cannot identify as clinically unsafe, because it has no access to the original image to cross-check the classification. The current rule-based safety checker catches structural safety violations (contraindicated dressings, incorrect antibiotic language) but cannot detect classification errors in the upstream T.I.M.E. payload.

---

### 1.2 Literature Support

**SCARWID (Busaranuvong, 2025; ACM THRI)**  
The SCARWID framework is the closest published analogue to your proposal. It used GPT-4o to generate natural language captions of diabetic foot ulcer (DFU) images, then fine-tuned a BLIP model (Wound-BLIP) on those captions to produce consistent textual descriptions at inference without requiring label information. SCARWID's captions were used as multimodal metadata, enriching infection classification by combining image features with the generated textual description. The result was an explainable, multimodal wound classification model. Critically, the caption served as an intermediate representation that linked visual evidence to language-space classification, exactly the role you propose for a VLLM captioning layer in VerdaSense.

**MEDIQA-WV 2025 (MasonNLP, Karim & Uzuner, 2025; ACL ClinicalNLP)**  
The MEDIQA-WV shared task required systems to generate free-text wound care responses and structured wound attributes directly from wound images and patient queries. The MasonNLP winning system used LLaMA-4 Scout 17B in a multimodal RAG framework, incorporating visual and textual exemplars from in-domain wound care data. This demonstrated that instruction-tuned VLLMs — without domain-specific fine-tuning — can generate clinically relevant wound descriptions suitable for downstream RAG pipelines.

**AI vs. MD — ChatGPT/Gemini for Wound Management (Forte et al., 2025; JMIR)**  
GPT-4o and Gemini were benchmarked on 20 complex wound images, scored against a panel of expert plastic surgeons. Both models produced accurate visual descriptions and initial management proposals from image input alone, suggesting that state-of-the-art VLLMs have sufficient wound perception capability to serve as a zero-shot captioning layer without fine-tuning.

**Microsoft Wound Care with Foundation Models (Microsoft Blog, March 2026)**  
Microsoft's wound care research used GPT-4o and GPT-4.1 for wound assessment tasks in three settings: zero-shot, few-shot, and RAG. The key finding relevant to VerdaSense is that VLLMs used in the RAG setting — where retrieved wound care exemplars were injected alongside the image query — produced more precise structured attribute predictions than either zero-shot or few-shot settings alone.

---

### 1.3 The Proposed Improvement: VLLM Captioning as a Pre-RAG Enrichment Layer

The proposed improvement inserts a VLLM captioning step into the VerdaSense pipeline between the original wound image and the RAG input construction step:

```
Wound Image ─────────────────────────────────────────────────────────────────┐
    │                                                                          │
    ▼                                                                          ▼
[Companion CV Models: YOLO + SAM + K-Means + IME-Net]              [VLLM Captioning Layer]
    │                                                                          │
    ▼                                                                          ▼
Structured T.I.M.E. payload                                     Narrative wound description
    │                                                                          │
    └──────────────────────────┬───────────────────────────────────────────────┘
                                ▼
                    Combined Input to VerdaSense RAG
              (T.I.M.E. labels + VLLM description + patient notes)
                                ▼
                    [Multi-Axis Query Construction]
                                ▼
                      Retrieval → Generation
```

**VLLM Captioning Prompt Design**

The VLLM receives the original wound image and a structured captioning prompt specifically designed to elicit clinically relevant wound descriptions across T.I.M.E. axes:

```
"You are a wound care specialist. Describe the wound in this image using clinical terminology.

Structure your response around the following dimensions:
1. TISSUE COMPOSITION: Describe the visible tissue types (necrotic tissue, slough/fibrin, granulation, epithelialization) and estimate their approximate proportions.
2. INFECTION/INFLAMMATION SIGNS: Describe any visible signs of infection or inflammation (erythema, oedema, pus, malodour indicators, periwound changes).
3. MOISTURE/EXUDATE: Describe the apparent exudate level (dry, moist, wet, macerated) and any visible exudate characteristics.
4. WOUND EDGE: Describe the wound edge characteristics (rolled, undermined, epithelialising, non-advancing, callused).
5. ADDITIONAL OBSERVATIONS: Note wound size cues, depth, periwound condition, any concerning features not captured above.

Be specific. Use clinical descriptors. Do not recommend treatments."
```

This caption is then injected alongside the structured T.I.M.E. payload into the multi-axis query construction step, enriching Sub-query B (Clinical Condition) and Sub-query C (Patient Context) with the visual evidence directly.

**VLLM Model Options for the Captioning Layer:**

| Model | Notes |
|---|---|
| **GPT-4o / GPT-4.1** | Best-in-class vision + clinical knowledge; zero-shot wound captioning without fine-tuning. Preferred for production. |
| **Gemini 2.5 Flash** | Already your Stage 2 winner for generation; can serve dual role as captioner + generator in a single API call, reducing latency overhead. |
| **LLaMA-4 Scout 17B** | MEDIQA-WV 2025 winner architecture; open-source, deployable on UM HPC, no API cost. |
| **Wound-BLIP (SCARWID approach)** | Fine-tuned BLIP model; lightweight (~400M params), fast, deployable on-device. Requires wound caption dataset for fine-tuning (can be bootstrapped from GPT-4o generated captions). |

**Fusing VLLM Description with T.I.M.E. Labels — The Arbitration Problem**

A critical design decision is what to do when the VLLM description disagrees with the companion CV model's classification. For example:
- CV model: `infection: "Not infected"`
- VLLM caption: `"...visible periwound erythema extending 2cm, green-tinged exudate visible at wound base, consistent with early infection..."`

Three strategies exist, each with different clinical risk profiles:

| Strategy | Description | Clinical Risk |
|---|---|---|
| **VLLM Override** | If VLLM detects infection signs not captured by CV model, flag as "potentially infected" | Safest for missed infections; may over-escalate |
| **CV Priority with Disclosure** | Trust CV model labels; include VLLM description in context as additional notes | Risk of passing incorrect labels to RAG unchanged |
| **Dual-Channel with Safety Flag** | Run both independently; if conflict detected, append safety warning and recommend clinical review | Best for deployment; most transparent |

For a clinical deployment where under-detection of infection is more dangerous than over-escalation, the **Dual-Channel with Safety Flag** strategy is recommended. This means VerdaSense would flag any case where the VLLM description contains infection indicators absent from the CV model's output, and append a mandatory note to the recommendation: *"Note: Visual analysis suggests potential infection signs not detected by automated classifier. Clinical review is recommended before applying dressing."*

---

### 1.4 Sub-Idea: CLIP for Image-to-Guideline Chunk Retrieval

Beyond captioning, CLIP (or its medical variants like **BiomedCLIP** or **MedCLIP**) can be used to directly retrieve wound care guideline images — wound photographs with annotated management notes — from a multimodal knowledge base, bypassing the text-only retrieval limitation entirely.

**Architecture:**
```
Wound Image → CLIP Image Encoder → Image Embedding
                                          │
                                          ▼
                              Cosine similarity against:
                              - Text embeddings of guideline chunks (current)
                              - Image embeddings of reference wound photos (new)
                                          │
                                          ▼
                              Fused retrieval results → Generation
```

**Literature anchor:** BiomedCLIP (Zhang et al., 2023) pre-trained on 15M biomedical image-text pairs from PubMed Central achieved strong performance on medical image-text retrieval. For wound care, a CLIP model fine-tuned on wound image + dressing recommendation pairs would create a semantically aligned image-text retrieval space — enabling queries like "retrieve dressing guidance for wounds visually similar to this image."

**Practical constraint for FYP:** Building a wound-image CLIP retrieval index requires a dataset of labelled wound photographs paired with guideline text, which does not currently exist at the scale needed. This is therefore best positioned as a longer-term future work item, with the VLLM captioning approach (Section 1.3) as the near-term priority.

---

## Improvement II: Multimodal RAG — Image-Grounded Retrieval Beyond Text Queries

### 2.1 The Problem: VerdaSense Retrieves by Text, Not by Visual Evidence

The current retrieval pipeline constructs text queries from T.I.M.E. labels and retrieves text chunks from the knowledge base. This is fundamentally a text-to-text retrieval paradigm, even though the original clinical information is visual. The MMed-RAG framework (Xia et al., ICLR 2025) demonstrated that for medical VLMs, cross-modal misalignment — where RAG retrieves textually correct but visually misaligned content — is a significant source of generation errors. Specifically, when the retrieved text does not correspond to the visual characteristics of the actual image, the generator hallucinates to bridge the gap.

### 2.2 Proposed Improvement: MMed-RAG-Inspired Dual-Path Retrieval

Based on MMed-RAG (ICLR 2025), the improvement introduces a second retrieval path — image-grounded retrieval — that runs in parallel with the existing text retrieval path:

```
                    ┌──── Text Path ────────────────────────────────┐
                    │  T.I.M.E. labels + VLLM caption                │
                    │  → Multi-axis text queries                      │
                    │  → BM25 + Dense retrieval from text KB          │
Wound Image ────────┤                                                  ├──→ RRF Fusion → Generation
                    │                                                  │
                    └──── Visual Path ───────────────────────────────┘
                       Wound image → BiomedCLIP / CLIP image encoder
                       → cosine similarity against image-indexed
                         reference wound case database
                       → Retrieved: similar wound cases + their
                         documented dressing outcomes
```

The visual retrieval path requires a **wound case image database** — a collection of wound photographs paired with documented dressing selections and outcomes. This does not need to be massive; even 200–500 annotated cases creates a meaningful visual retrieval index that improves recommendation specificity for visually unusual wound presentations.

**Sources for wound image case database:**
- Public: AZH dataset (250+ DFU images), MICCAI Foot Ulcer dataset, DFU2021 competition dataset
- Clinical partner: Images collected during the VerdaSense deployment period can be used to expand the retrieval database over time (with patient consent)

**MMed-RAG's key innovation** was a modal-aware retrieval strategy that selected the retrieval modality (text vs image) based on which modality was more informative for the query type. For VerdaSense, a simpler heuristic is sufficient: always retrieve from both paths, then use RRF or a confidence-weighted fusion to merge the ranked results.

---

## Improvement III: Longitudinal Wound Tracking — Multi-Visit Context for RAG

### 3.1 The Problem: VerdaSense Treats Every Wound Assessment in Isolation

The current VerdaSense system generates a dressing recommendation from a single wound assessment snapshot. In clinical practice, wound management is inherently longitudinal — the dressing selection for a wound at Week 3 should be informed by the wound's trajectory since Week 1. A wound that was 40% slough at Week 1 and is still 40% slough at Week 3 (no improvement) requires a different intervention than a wound that improved from 40% to 25% slough (responding to treatment). The current system cannot distinguish these cases and will generate the same recommendation for both.

### 3.2 Literature Support

**Mobile AI-enhanced Platform for Wound Assessment (medRxiv, Jan 2026)**  
A mobile platform study demonstrated that longitudinal wound tracking — measuring wound area, tissue composition changes, and healing rate across multiple visits — provided superior clinical decision support compared to single-visit snapshots. The platform's longitudinal healing analytics computed healing rate as a percentage area reduction per week, enabling trajectory-based treatment escalation decisions. The paper identified that longitudinal tracking reduced unnecessary dressing changes by approximately 23% by confirming continued healing progress.

**LILAC — Learning-based Inference of Longitudinal Image Changes (PNAS, Feb 2025)**  
LILAC demonstrated that pairwise comparison of longitudinal wound images using a Siamese convolutional architecture could extract meaningful temporal difference signals — i.e., capturing what changed between Visit 1 and Visit 2 beyond what a single-visit assessment reveals. This is directly applicable to wound bed progression assessment.

### 3.3 Proposed Improvement: Longitudinal Context Injection into the RAG Input

The improvement adds a "wound history" field to the VerdaSense input payload, populated by storing and retrieving previous wound assessments from the patient's visit history:

**New Input Payload Structure:**
```json
{
  "current_visit": {
    "time_payload": { ... },
    "vllm_description": "...",
    "patient_notes": "..."
  },
  "wound_history": [
    {
      "visit_number": 1,
      "date": "2026-04-01",
      "time_payload": { "necrotic": 40, "slough": 35, "granulation": 25, ... },
      "dressing_applied": "Hydrocolloid + Alginate",
      "response": "No improvement at 2-week review"
    },
    {
      "visit_number": 2,
      "date": "2026-04-15",
      "time_payload": { "necrotic": 35, "slough": 30, "granulation": 35, ... },
      "dressing_applied": "Hydrocolloid + Alginate (continued)",
      "response": "Slow improvement"
    }
  ],
  "healing_trajectory": {
    "healing_rate_pct_per_week": 4.2,
    "trend": "slow_improvement",
    "weeks_since_last_dressing_change": 3
  }
}
```

**How the longitudinal context enriches the RAG pipeline:**

1. **Trajectory-aware Sub-query D (new):** A fourth sub-query is constructed from the wound history to retrieve guideline content specifically about treatment escalation, dressing changes for non-responding wounds, and chronic wound management protocols.

2. **Mandatory injection enrichment:** The pre-classifier receives the healing trajectory as additional input, enabling classification of cases like "stalled wound requiring debridement escalation" that cannot be detected from a single-visit T.I.M.E. snapshot alone.

3. **Contraindication extension:** Dressings that have already been tried and failed should be flagged in the recommendation, preventing repetition of ineffective treatments. This requires extending the safety checker to cross-reference the wound history against the recommended dressings.

**Implementation Path:**
- Patient wound visit data is already logged in the mobile app backend for tracking purposes.
- The wound history payload is assembled at query time from the patient's visit log stored in the app database.
- No changes to the knowledge base or vector store are needed — only the query construction and prompt template require updating.
- **New ablation experiment (G5):** Compare single-visit RAG vs. multi-visit context RAG on a longitudinal test set — a meaningful academic contribution extending beyond the current 32-case snapshot testset.

---

## Improvement IV: Confidence-Aware Uncertainty Output and Escalation Logic

### 4.1 The Problem: VerdaSense Outputs Without Expressing Clinical Uncertainty

The current VerdaSense system generates recommendations at a fixed level of confidence regardless of input quality. A wound with borderline T.I.M.E. inputs — e.g., 25% necrotic tissue (right at the debridement threshold), or a VLLM description that shows mild infection signs while CV model says "Not infected" — receives a recommendation with the same presentational authority as a clear-cut case. This is clinically dangerous: patients interpreting a confident-sounding recommendation for an ambiguous wound may forgo seeking clinical consultation.

### 4.2 Literature Support

**FRANQ — Faithfulness-based RAG Uncertainty Quantification (arXiv, May 2025)**  
FRANQ introduced a method that distinguishes between two types of uncertainty in RAG outputs: (1) faithfulness uncertainty — whether the generated claim is supported by retrieved context, and (2) factuality uncertainty — whether the claim is factually correct regardless of retrieval support. FRANQ achieved significantly better hallucination detection in RAG outputs than existing uncertainty quantification methods by treating these two uncertainty sources separately.

**Agentic AI and LLMs in Radiology — Hallucination and Uncertainty (MDPI, Nov 2025)**  
A comprehensive 2024–2025 review demonstrated that uncertainty quantification approaches in clinical AI systems consistently improved clinician trust and appropriate use of AI recommendations. The key finding: systems that explicitly communicated confidence levels and identified low-confidence cases for human review had higher clinical acceptance rates than systems that presented all outputs uniformly.

**QuCo-RAG — Uncertainty-Triggered Dynamic Retrieval (arXiv, Dec 2025)**  
QuCo-RAG showed that dynamically adjusting retrieval depth based on query uncertainty — retrieving more chunks for low-confidence queries — improved factuality without increasing average latency.

### 4.3 Proposed Improvement: Three-Tier Confidence Classification and Escalation

VerdaSense should output a confidence tier alongside every recommendation, computed from multiple signals:

**Confidence Tier Computation:**

```
Confidence Score = f(
  classification_confidence,    # confidence scores from companion CV models
  vllm_cv_agreement,            # do VLLM description and CV labels agree?
  retrieval_coverage,           # context recall of retrieved chunks for this case
  edge_case_flag,               # is this case flagged as Category D or E in testset-equivalent logic?
  healing_trajectory,           # for multi-visit: is wound stalled or deteriorating?
)
```

**Three Tiers:**

| Tier | Criteria | RAG Behaviour | Output Suffix |
|---|---|---|---|
| **HIGH** | CV labels + VLLM agree; retrieval coverage high; clear wound category | Standard generation, current pipeline | None |
| **MEDIUM** | Minor disagreement between CV + VLLM; borderline classification; edge values | Add Sub-query D for escalation guidance; increase k from 6 to 8 | "*Note: Some wound parameters fall near clinical thresholds. Consider clinical review if wound does not respond to recommended dressing within [X] days.*" |
| **LOW** | CV + VLLM significant disagreement; wound category unclear; deteriorating trajectory | Trigger mandatory escalation language in mandatory injection; recommend specialist referral | "*Important: Automated assessment found conflicting signals for this wound. This recommendation is provisional. Clinical assessment by a wound care specialist is strongly recommended before applying any dressing.*" |

**Implementation:**
- Confidence scoring is computed in the pre-classification step (before retrieval), using the companion CV models' softmax probabilities if available, and VLLM agreement score (cosine similarity between VLLM description embedding and CV label embedding).
- The tier determines the prompt variant selected (standard G1-C vs. escalation-enriched variant).
- A new ablation experiment (G6) evaluates whether confidence-aware prompting improves safety pass rate on edge cases (Category D and E testset cases) without degrading performance on straightforward Category A cases.

---

## Improvement V: Fine-Tuning the Generation LLM on Wound Care Recommendations

### 5.1 The Problem: General-Purpose LLMs Have No Wound Care Specialization

The current best generation LLM (Gemini 2.5 Flash, FA=0.8147) is a general-purpose model with no wound care specialization. Its high faithfulness score reflects how well it follows the grounding prompt's instructions — not that it has internalized wound care clinical knowledge. This means its recommendations are only as good as the retrieved context it receives. When retrieval quality degrades (e.g., for rare wound types not well covered in the current 8-source KB), the general-purpose LLM has no fallback clinical knowledge to draw on safely.

### 5.2 Literature Support

**MedGemma (Google, 2025)**  
Google released MedGemma, a medical-domain fine-tuned version of Gemma 3, specifically for clinical tasks including report generation, medical VQA, and clinical decision support. In the German nursing paper (Powering & Rothgang, 2026) — VerdaSense's closest comparison work — MedGemma was evaluated and showed competitive performance with larger general-purpose models specifically because its fine-tuning provided implicit clinical safety guardrails.

**MMedPO — Clinical-Aware Preference Optimization for Med-VLMs (ICLR 2025)**  
MMedPO (Multimodal Medical Preference Optimization) demonstrated that preference-learning fine-tuning on clinical-quality outputs — where expert-preferred outputs are rewarded and unsafe outputs are penalized — significantly improved both factuality and safety alignment for medical VLMs, beyond what RAG grounding alone achieved.

**NICE Guideline RAG with SME Validation (Lewis et al., 2025)**  
The NICE Guideline RAG achieving 99.5% faithfulness used a combination of highly curated retrieval AND an instruction-tuned LLM. The study implied that for clinical RAG, the LLM's fine-tuning on clinical reasoning tasks was a necessary complement to retrieval grounding — neither component alone achieved the same result.

### 5.3 Proposed Improvement: Wound Care SFT (Supervised Fine-Tuning) Dataset Construction

The most ambitious but highest-impact improvement is to fine-tune a smaller open-source LLM (Gemma 3 12B or Qwen3 14B) on a wound care recommendation dataset, creating a domain-specialized wound care generation model.

**Dataset construction strategy:**

The key challenge is that no wound care recommendation dataset exists. A construction pipeline using VerdaSense itself as a data generation tool is proposed:

```
Step 1: Generate synthetic wound T.I.M.E. inputs
   → Systematically sample the T.I.M.E. parameter space
     (all combinations of tissue types × infection × moisture × edge)
   → ~200–500 synthetic cases covering the full wound type space

Step 2: Generate reference recommendations
   → Pass all synthetic cases through VerdaSense (best configuration)
     using the full 8-source KB and G1-D full scaffolding prompt
   → Filter to keep only cases where Safety Pass = TRUE and FA ≥ 0.90

Step 3: Clinical expert validation
   → Review generated recommendations with clinical collaborator
   → Annotate preferred responses, flag unsafe responses

Step 4: SFT dataset format
   (input: T.I.M.E. payload + retrieved context)
   (output: expert-validated recommendation)

Step 5: Fine-tune Gemma 3 12B or Qwen3 14B using LoRA
   → 4-bit quantized training on UM HPC A100
   → Evaluate against current best G2/G3 configuration on testset
```

**Expected benefit:** A fine-tuned model has clinical safety guardrails baked into its parameters — even when retrieval quality degrades, it will not recommend contraindicated dressings for common wound types, because those constraints are part of its learned distribution. This creates a defense-in-depth safety architecture: RAG grounding as the primary safety layer, fine-tuning as the secondary fallback layer.

**Practical note for FYP:** Full fine-tuning is a significant undertaking beyond current FYP2 scope. The SFT dataset construction (Steps 1–3) is achievable within FYP2 and constitutes a standalone academic contribution — it would be the first published wound care recommendation SFT dataset.

---

## Improvement VI: Agentic Multi-Step Clinical Reasoning Pipeline

### 6.1 The Problem: Single-Pass Generation Has No Clinical Reasoning Step

The current VerdaSense generates a recommendation in a single LLM call: retrieve → generate. This single-pass design does not allow the model to reason through the clinical decision process that an expert wound care nurse would follow: identify wound type → consider complicating factors → check contraindications → formulate primary recommendation → verify safety. All of this happens implicitly within one generation pass, constrained by the prompt.

### 6.2 Literature Support

**Agentic AI for Clinical Decision Support (MDPI, Nov 2025)**  
The comprehensive 2024–2025 review of agentic AI in radiology demonstrated that multi-agent frameworks using role-based specialization — where different agents handle different aspects of clinical reasoning — produced more reliable outputs than single-pass generation, particularly for complex cases. The review noted that "agents that explicitly communicated uncertainty to one another" achieved the lowest hallucination rates.

**Multi-agent Clinical RAG frameworks (EmergentMind survey, 2025)**  
The emerging MMed-RAG literature identified modular, decoupled optimization as a key architectural benefit — separate retriever, generator, and verifier modules enable component-level improvement without full system retraining.

### 6.3 Proposed Improvement: VerdaSense Agentic Pipeline

A three-agent architecture that decomposes the current single-pass recommendation into three specialized reasoning steps:

```
Stage 1: WOUND ANALYST AGENT
├── Input: T.I.M.E. payload + VLLM description
├── Task: Classify wound type, severity, and key clinical factors
│         Identify any conflicting signals between CV and VLLM
│         Determine confidence tier
└── Output: Structured wound analysis summary

                    ↓

Stage 2: EVIDENCE RETRIEVER AGENT
├── Input: Wound analysis summary from Stage 1
├── Task: Execute multi-axis sub-queries (R1-C) using the
│         analysis summary as enriched query basis
│         Select top-k most relevant chunks
│         Identify any evidence gaps
└── Output: Retrieved guideline context + evidence quality score

                    ↓

Stage 3: RECOMMENDATION GENERATOR AGENT
├── Input: Wound analysis (Stage 1) + Retrieved context (Stage 2)
├── Task: Generate structured recommendation (9 output fields)
│         Explicitly cite evidence for each field
│         Apply safety rules
│         Append confidence-appropriate disclaimer
└── Output: Final evidence-grounded dressing recommendation

                    ↓

Stage 4 (Optional): SAFETY VERIFIER AGENT
├── Input: Generated recommendation
├── Task: Cross-check recommendation against rule-based safety checker
│         Verify all contraindicated dressings are absent
│         Verify antibiotic/referral language appropriateness
└── Output: PASS/FAIL + specific failure annotations for regeneration
```

**Cost vs. benefit:** The agentic pipeline involves 3–4 LLM calls per recommendation, increasing latency significantly (potentially 3× the current ~18s for Gemini 2.5 Flash). This makes it unsuitable for real-time patient-facing use but appropriate for a "deep review mode" where the clinician or caregiver has time for a more thorough assessment — for example, a weekly dressing planning review rather than a bedside quick-check. VerdaSense could offer both modes: standard single-pass for rapid bedside use, and agentic review mode for complex or stalled wounds.

---

## Synthesis: Recommended Priority Roadmap

The six improvements are organized by implementation effort vs. clinical impact:

| Priority | Improvement | Effort | Clinical Impact | Suggested Timeline |
|---|---|---|---|---|
| **P1 🔥** | VLLM Captioning Layer (Improvement I) | Medium | Very High | FYP2 presentation / immediate next step |
| **P2** | Confidence-Aware Uncertainty Output (Improvement IV) | Low-Medium | High | FYP2 Future Work chapter |
| **P3** | Longitudinal Context Injection (Improvement III) | Medium | High | Post-FYP2, production v2 |
| **P4** | Multimodal RAG — Dual-Path Retrieval (Improvement II) | High | Medium-High | Post-FYP2, research paper |
| **P5** | LLM Fine-Tuning on Wound Care Data (Improvement V) | Very High | Highest (long-term) | Post-graduation / MSc/PhD level |
| **P6** | Agentic Multi-Step Pipeline (Improvement VI) | High | High for complex cases | Post-FYP2, production v2 |

---

### P1 Justification: Why VLLM Captioning is the Single Most Impactful Near-Term Improvement

The VLLM captioning layer (Improvement I) has the best effort-to-impact ratio for the following reasons:

1. **It directly addresses the core structural dependency problem** — the system becomes partially independent of companion CV model accuracy, because it can cross-check CV outputs against visual evidence.

2. **It requires no changes to the retrieval or generation architecture** — the VLLM caption is simply added as a new input field to the existing multi-axis query construction step, and injected into the prompt alongside the T.I.M.E. payload.

3. **The technology is mature and available today** — GPT-4o, Gemini 2.5 Flash, and LLaMA-4 Scout are all production-quality VLLMs that can perform wound captioning zero-shot without fine-tuning, as demonstrated by SCARWID, MEDIQA-WV 2025, and the Microsoft wound care research.

4. **It creates a new ablation dimension (G4-VLLM)** — comparing T.I.M.E.-only input vs. T.I.M.E. + VLLM caption input on the existing 32-case testset would be a clean, publishable ablation contribution.

5. **It directly addresses the patient safety gap** — by surfacing visual infection or deterioration signals that the CV models may have missed, the VLLM caption layer adds a qualitative safety net that the current rule-based checker cannot provide.

**Concrete Implementation for VLLM Captioning:**

```python
# Step 1: Add wound_image_path to the VerdaSense API input
api_input = {
    "wound_image_path": "/path/to/wound_image.jpg",  # NEW
    "time_payload": { ... },                           # existing
    "patient_notes": "...",                            # existing
}

# Step 2: VLLM captioning using Gemini 2.5 Flash (already your best LLM)
import google.generativeai as genai
import PIL.Image

def generate_wound_caption(image_path: str) -> str:
    model = genai.GenerativeModel("gemini-2.5-flash")
    wound_img = PIL.Image.open(image_path)
    caption_prompt = """
    You are a wound care specialist. Describe the wound in this image using clinical terminology.
    Structure your response around: tissue composition, infection/inflammation signs, 
    moisture/exudate level, wound edge characteristics, and additional observations.
    Be concise (max 150 words). Do not recommend any treatments.
    """
    response = model.generate_content([caption_prompt, wound_img])
    return response.text

# Step 3: Enrich the VerdaSense input with the VLLM caption
caption = generate_wound_caption(api_input["wound_image_path"])
enriched_input = {
    **api_input,
    "vllm_description": caption
}

# Step 4: Pass enriched_input to existing multi-axis query construction
# Sub-query C now uses: patient_notes + vllm_description (instead of patient_notes only)
sub_query_c = construct_context_query(
    patient_notes=api_input["patient_notes"],
    vllm_description=caption  # NEW enrichment
)
```

**New ablation experiment to add to Stage 2:**

| Exp | Version | Input Configuration | VLLM | 
|---|---|---|---|
| G4 | G4-A | T.I.M.E. only (current baseline) | None |
| G4 | G4-B | T.I.M.E. + Gemini 2.5 Flash caption | Gemini 2.5 Flash |
| G4 | G4-C | T.I.M.E. + GPT-4o caption | GPT-4o |
| G4 | G4-D | T.I.M.E. + patient notes (baseline with notes) | None |
| G4 | G4-E | T.I.M.E. + patient notes + Gemini caption | Gemini 2.5 Flash |

This experiment isolates the contribution of VLLM visual enrichment independent of patient notes, answering: *"How much does adding a VLLM wound caption improve recommendation quality and safety beyond what structured T.I.M.E. labels alone can achieve?"*

---

## Architecture Diagram: VerdaSense Next-Gen

The following describes the proposed full VerdaSense next-generation architecture combining all high-priority improvements:

```
                        ┌─────────────────────────────────────────┐
                        │         PATIENT MOBILE APP              │
                        │  Captures wound image + optional notes  │
                        └─────────────────┬───────────────────────┘
                                          │
                                          ▼
                        ┌─────────────────────────────────────────┐
                        │         COMPANION CV PIPELINE           │
                        │  YOLO → SAM → K-Means → IME-Net         │
                        │  Output: T.I.M.E. structured payload    │
                        └─────────────┬───────────────────────────┘
                                      │
                    ┌─────────────────┼──────────────────────────┐
                    │                 │                           │
                    ▼                 ▼                           ▼
           T.I.M.E. payload    VLLM Captioner              Patient Notes
           (structured labels)  (Gemini 2.5 Flash         (optional free text)
                    │            or GPT-4o)                      │
                    │                 │                           │
                    └────────────┬────┘                           │
                                 ▼                                │
                    ┌─────────────────────────────────────────┐   │
                    │      CONFLICT DETECTION MODULE          │   │
                    │  Compare CV labels vs VLLM description  │   │
                    │  Compute confidence tier (H/M/L)        │   │
                    └─────────────────┬───────────────────────┘   │
                                      │                           │
                                      ▼                           │
                    ┌─────────────────────────────────────────┐   │
                    │       WOUND HISTORY RETRIEVER           │   │
                    │  Fetch patient's previous assessments   │   │
                    │  Compute healing trajectory             │   │
                    └─────────────────┬───────────────────────┘   │
                                      │                            │
                                      └────────────┬───────────────┘
                                                   │
                                                   ▼
                    ┌──────────────────────────────────────────────────┐
                    │       VERDASENSE RAG CORE (CURRENT + ENHANCED)   │
                    │                                                    │
                    │  ┌────────────────────────────────────────────┐   │
                    │  │  INPUT NORMALIZATION + PRE-CLASSIFIER       │   │
                    │  │  (T.I.M.E. + VLLM + history + confidence)  │   │
                    │  └──────────────────┬─────────────────────────┘   │
                    │                     │                              │
                    │                     ▼                              │
                    │  ┌────────────────────────────────────────────┐   │
                    │  │  MULTI-AXIS QUERY CONSTRUCTION              │   │
                    │  │  Sub-A: Wound type algorithm query          │   │
                    │  │  Sub-B: Dressing mechanism query            │   │
                    │  │  Sub-C: Patient context + VLLM desc         │   │
                    │  │  Sub-D: Escalation/trajectory query (NEW)   │   │
                    │  └──────────────────┬─────────────────────────┘   │
                    │                     │                              │
                    │                     ▼                              │
                    │  ┌────────────────────────────────────────────┐   │
                    │  │  HYBRID RETRIEVAL                           │   │
                    │  │  Dense (BGE Large) + Sparse (BM25) + RRF   │   │
                    │  │  k=6, db_wound_care_v4                      │   │
                    │  └──────────────────┬─────────────────────────┘   │
                    │                     │                              │
                    │                     ▼                              │
                    │  ┌────────────────────────────────────────────┐   │
                    │  │  GENERATION (G1-C GROUNDED PROMPT)          │   │
                    │  │  Gemini 2.5 Flash / GPT-4o                  │   │
                    │  │  Confidence-tier prompt variant selection   │   │
                    │  └──────────────────┬─────────────────────────┘   │
                    │                     │                              │
                    │                     ▼                              │
                    │  ┌────────────────────────────────────────────┐   │
                    │  │  SAFETY CHECKER v3                          │   │
                    │  │  Rule-based: contraindications, antibiotics │   │
                    │  │  + History check: previously failed dress.  │   │
                    │  │  + Confidence tier disclaimer injection      │   │
                    │  └──────────────────┬─────────────────────────┘   │
                    │                     │                              │
                    └─────────────────────┼──────────────────────────────┘
                                          │
                                          ▼
                    ┌─────────────────────────────────────────────────┐
                    │         STRUCTURED RECOMMENDATION OUTPUT         │
                    │  Primary Dressing | Secondary Dressing           │
                    │  T.I.M.E. Rationale | Antibiotic Guidance        │
                    │  Referral Indicator | Dressing Change Frequency  │
                    │  Contraindicated Dressings | Application Tips    │
                    │  Clinical Notes | [Confidence Tier Badge]        │
                    │  [VLLM-CV Conflict Warning if applicable]        │
                    └─────────────────────────────────────────────────┘
```

---

## Key Academic Contributions of the Proposed Improvements

| Contribution | Type | Novelty |
|---|---|---|
| VLLM captioning as pre-RAG visual enrichment for wound care | Architectural | First application of VLLM captioning layer in a wound dressing RAG pipeline |
| VLLM-CV agreement scoring as a clinical confidence signal | Methodological | Novel confidence signal combining structured classification and visual language model output |
| Longitudinal wound history injection into multi-axis RAG queries | Architectural | First RAG framework incorporating multi-visit temporal wound context |
| Wound care SFT dataset construction via VerdaSense bootstrap | Dataset | First publicly documented wound care recommendation SFT dataset pipeline |
| Confidence-tier escalation with automatic clinical disclaimer injection | Safety | Novel patient safety mechanism combining uncertainty quantification with mandatory prompt variant selection |
| Dual-path text + visual retrieval for wound care guidelines | Retrieval | Application of MMed-RAG cross-modal retrieval to wound dressing recommendation |

---

## References (Selected — Supporting Literature for This Document)

- Busaranuvong, P. et al. (2025). *Explainable, Multimodal Wound Infection Classification from Images Augmented with Generated Captions (SCARWID).* ACM Transactions on Computing for Healthcare.
- Karim, A.H.M.R. & Uzuner, Ö. (2025). *MasonNLP at MEDIQA-WV 2025: Multimodal RAG with LLMs for Medical VQA.* ACL ClinicalNLP 2025. arXiv:2510.13856.
- Xia, Y. et al. (2025). *MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models.* ICLR 2025.
- Wang et al. (2025). *Promoting wound healing through AI-powered dressing development.* Wounds International.
- Forte, A.J. et al. (2025). *AI vs. MD: Benchmarking ChatGPT and Gemini for Complex Wound Management.* JMIR, 14(24):e8825.
- Microsoft (2026, March). *Advancing Wound Care with Foundation Models and Context-Aware Retrieval.* Microsoft Health + Life Sciences Blog.
- Curti, N. et al. (2024). *Automated Prediction of Photographic Wound Assessment Tool in Chronic Wound Images.* JMIR Medical Informatics.
- Abbas et al. (2026). *A Mobile AI-enhanced Platform for Standardized Wound Assessment and Clinical Decision Support.* medRxiv.
- Powering, L. & Rothgang, E. (2026). *RAG-based Clinical Decision Support for Nursing Guidelines.* [VerdaSense comparison paper]
- Zhang, H. et al. (2023). *BiomedCLIP: A multimodal biomedical foundation model pre-trained from fifteen million scientific image-text pairs.* arXiv:2303.00915.
- Radford, A. et al. (2021). *Learning Transferable Visual Models from Natural Language Supervision (CLIP).* ICML 2021.
- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020.
- He, Y. et al. (2024). *MedDr: Diagnosis-Guided Bootstrapping for Large-Scale Medical Vision-Language Learning.*
- Kalpelbe, K. et al. (2025). *FRANQ: Faithfulness-based Retrieval Augmented UNcertainty Quantification.* arXiv:2505.21072.

---

*VerdaSense RAG Future Improvements Document | Tee Qi Jing | Universiti Malaya FYP2 | May 2026*
