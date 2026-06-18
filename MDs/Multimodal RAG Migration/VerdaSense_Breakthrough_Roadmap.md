# VerdaSense RAG — Breakthrough Roadmap
## Breaking Current Limitations for Clinically Reliable Wound Dressing Recommendations

**For:** Tee Qi Jing | Universiti Malaya | AI Degree FYP  
**Status:** Practical roadmap synthesising current system analysis + latest literature (2024–2026)  
**Date:** May 2026  

---

## A Quick Note on the Douyin Comments

The Mandarin comments from Douyin are **clinically and technically correct** and map precisely onto your system's limitations. The commenter argues that complex clinical reasoning — like multi-pathway pharmacology — cannot be reliably handled by a flat vector RAG or a single LLM alone. They advocate for **compound architectures**: fine-tuned intent recognition, domain-specialised embeddings, graph-based structured knowledge, multi-path routing, and verification stages. In their words: *"it's definitely not something a single LLM can replace; it must be a composite architecture."*

For wound dressing recommendations, the equivalent complexity is: one wound can have infection + high exudate + non-advancing edges + diabetic aetiology + iodine contraindication simultaneously. Each axis requires different retrieved evidence from different guideline sections. If any one axis is wrong upstream (K-Means tissue estimate, IME-Net infection classification), the flat RAG cannot self-correct — it produces a confident hallucination dressed in clinical language. The commenter's prescription — multi-type model calls, graph-ready preparation, accurate positioning, accuracy verification — is exactly what the rest of this document addresses, mapped to your specific system.

---

## Part 0 — Honest Diagnosis: Where VerdaSense Actually Stands

Before proposing improvements, it helps to be precise about what is and is not a limitation of your RAG versus the companion pipeline.

| Layer | Owner | Core Weakness | Consequence for RAG |
|---|---|---|---|
| Wound boundary segmentation | Senior student (MobileSAM LoRA) | Generally accurate — dataset-fine-tuned | Low risk to RAG input |
| **Tissue decomposition (T)** | **K-Means unsupervised** | **No ground truth — completely unsupervised; lighting, skin tone, wound type all affect colour clusters** | **Direct: wrong necrotic/slough/granulation % → wrong wound type classification (Types 5–8) → wrong retrieval axis** |
| **Infection / Moisture / Edge (IME)** | **IME-Net** | **Unknown accuracy on out-of-distribution wounds; no TIME-labelled validation set** | **Direct: misclassified infection status → wrong dressing family entirely (e.g., silver vs film)** |
| Retrieval (RAG) | You | 8 sources, manually chunked, no cross-guideline linking | Gaps in evidence; older guidelines may be superseded |
| Generation (RAG) | You | Single-pass, no self-verification | Cannot catch upstream classification errors |
| Evaluation | You | 32 synthetic test cases, no real wound images | Cannot measure real-world degradation from CV errors |

**The critical insight:** Your RAG pipeline is architecturally sound (ablation confirms this — FA=0.81, Safety=90.6% on the synthetic testset). The clinical reliability gap is not primarily a RAG problem — it is a **modality gap**: your RAG is blind to the original wound image that the patient uploaded. If K-Means says 5% necrotic on what is actually a 40% necrotic wound, VerdaSense generates a perfect recommendation for the wrong wound. It cannot know it is wrong.

This is also precisely why "the patient just uploads to ChatGPT-4o and gets a better recommendation" — GPT-4o with vision *sees* the wound and generates from visual evidence directly, bypassing K-Means and IME-Net entirely.

**Your FYP's differentiation must therefore be:** close the modality gap without abandoning your structured RAG evidence grounding, which is something raw GPT-4o cannot offer.

---

## Part 1 — Limitation 3 (Biggest Priority): Close the Modality Gap with a VLLM Visual Enrichment Layer

### Why this is the single most impactful change

Adding a VLLM captioning step makes VerdaSense see the wound, not just receive coarse labels derived from it. This:
- Adds a cross-check against K-Means and IME-Net outputs
- Captures visual features that have no structured output channel (periwound maceration, exudate colour, wound depth cues)
- Brings your system's capability closer to GPT-4o while retaining the evidence-grounding architecture GPT-4o cannot replicate
- Creates a clean new ablation experiment (G4) publishable as the key contribution

### 1.1 What the literature tells you is possible today

**SCARWID (Busaranuvong, 2025 — ACM THRI)**  
Used GPT-4o to generate narrative wound captions from DFU images, then fine-tuned BLIP (Wound-BLIP) on those captions for lightweight inference. The caption served as a cross-modality bridge — turning visual evidence into language-space representation that downstream classifiers and retrievers could use. Directly analogous to what you need.

**MEDIQA-WV 2025 / WoundCareVQA (Yim et al., 2025 — ACL ClinicalNLP)**  
The shared task required systems to generate free-text wound care responses *directly from wound images + patient queries*. The MasonNLP winning system (Karim & Uzuner) used LLaMA-4 Scout 17B in a multimodal RAG framework — textual + visual exemplar retrieval, grounded generation. Crucially: the dataset (WoundCareVQA) is bilingual (English/Chinese), contains wound images with patient queries, wound type metadata, and expert-written answers. **This dataset is your evaluation solution for Limitation 2 — discussed in Part 2.**

**EXL Health at MEDIQA-WV 2025 (Durgapraveen et al., 2025)**  
Showed that metadata-guided generation — where structured wound attributes (predicted by classifiers) are injected into the generation prompt alongside image evidence — outperformed image-only or text-only approaches. This is your G1-C grounded prompt + VLLM caption combined.

**AI vs. MD — GPT-4o/Gemini for Wound Management (Forte et al., 2025 — JMIR)**  
GPT-4o and Gemini benchmarked against plastic surgeon panels on 20 complex wound images, zero-shot. Both produced accurate visual descriptions and initial management proposals directly from image input. Implication: for straightforward wounds, VLLM zero-shot captioning is clinically reliable enough to serve as a cross-check signal — no fine-tuning required at this stage.

**Microsoft Wound Care with Foundation Models (2026)**  
RAG setting with VLLMs outperformed zero-shot and few-shot settings for wound attribute prediction. The key mechanism: retrieved wound care exemplars injected alongside the image guided the VLLM toward clinically structured output rather than generic descriptions.

### 1.2 Concrete Implementation — Step by Step

**Step 1: Add `wound_image` as optional input field in `wound_app_v5.py`**

Your `/get_recommendation` endpoint currently accepts:
```python
necrotic_pct, slough_pct, granulation_pct, infection, moisture, edge, notes, tissue_confidence
```

Add one optional field:
```python
wound_image_base64: str = Form("")   # base64-encoded JPEG from mobile app
```

The mobile app already has the wound image — it just needs to forward it to the RAG backend alongside the existing T.I.M.E. payload.

**Step 2: VLLM Captioning Function**

Insert before `Step 2: Classify` in `get_recommendation`:

```python
# Step 1b: VLLM Visual Captioning (if image provided)
vllm_caption = ""
vllm_cv_conflict = False

if wound_image_base64.strip():
    vllm_caption, vllm_cv_conflict = generate_vllm_caption(
        image_b64=wound_image_base64,
        cv_infection=infection,
        cv_moisture=moisture,
        cv_necrotic=necrotic_pct,
        cv_slough=slough_pct,
        cv_granulation=granulation_pct,
    )
    print(f"[VLLM] Caption generated. CV conflict detected: {vllm_cv_conflict}")
```

**Step 3: The Captioning Function with Conflict Detection**

```python
import google.generativeai as genai
import base64
from PIL import Image
import io

VLLM_CAPTION_PROMPT = """You are a wound care clinical specialist reviewing a wound photograph.

Describe the wound using clinical wound care terminology. 
Structure your response in exactly these five sections:

TISSUE: Describe visible tissue types and estimate proportions 
(e.g., "Approximately 60% granulation tissue with healthy red/pink appearance, 
30% yellow fibrinous slough at wound base, 10% dark eschar at periphery").

INFECTION: Describe any visible signs of infection or inflammation 
(perilesional erythema, oedema, green/cloudy exudate, visible pus, 
malodour-consistent appearance). State clearly: "No visible infection signs" 
or "Visible infection indicators: [list]".

MOISTURE: Describe exudate level and characteristics 
(dry/moist/wet/macerated periwound skin, exudate colour if visible).

EDGE: Describe wound edge characteristics 
(rolled/epibole, undermined, epithelialising, callused, macerated, well-defined).

ADDITIONAL: Note wound size estimation, depth cues, periwound skin condition, 
or any clinically significant features not captured above.

Be specific and clinically precise. Maximum 180 words total. 
Do NOT recommend treatments."""

def generate_vllm_caption(image_b64, cv_infection, cv_moisture, 
                           cv_necrotic, cv_slough, cv_granulation):
    """
    Generate VLLM wound caption and detect conflicts with CV model outputs.
    Returns: (caption_str, conflict_detected_bool)
    """
    try:
        # Decode image
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes))
        
        # Use Gemini 2.5 Flash (already your generation LLM — dual use)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content([VLLM_CAPTION_PROMPT, img])
        caption = response.text.strip()
        
        # Conflict detection — check if VLLM mentions infection 
        # signals that CV model missed
        conflict = False
        infection_keywords = [
            "erythema", "redness", "purulent", "pus", "green", 
            "cloudy exudate", "oedema", "swelling", "infection indicator"
        ]
        caption_lower = caption.lower()
        cv_says_not_infected = "not" in cv_infection.lower()
        
        if cv_says_not_infected:
            vllm_infection_signals = [k for k in infection_keywords 
                                       if k in caption_lower]
            if vllm_infection_signals:
                conflict = True
                print(f"[CONFLICT] CV: Not infected | "
                      f"VLLM signals: {vllm_infection_signals}")
        
        # Tissue conflict: CV says >70% granulation but VLLM sees necrosis/slough
        if cv_granulation >= 70:
            if any(w in caption_lower for w in ["necrotic", "eschar", 
                                                  "slough", "non-viable"]):
                conflict = True
                print("[CONFLICT] CV: Predominantly granulation | "
                      "VLLM: Non-viable tissue visible")
        
        return caption, conflict
        
    except Exception as e:
        print(f"[VLLM] Captioning failed: {e}")
        return "", False
```

**Step 4: Inject VLLM Caption into Assessment Text and Retrieval**

In `get_recommendation`, after generating the caption, enrich the assessment text:
```python
# Step 7 (existing): Build assessment text — add VLLM section
if vllm_caption:
    assessment_text += (
        f"\n\nVISUAL WOUND ANALYSIS (AI Image Assessment):\n"
        f"{vllm_caption}"
    )
    if vllm_cv_conflict:
        assessment_text += (
            "\n\n⚠️ CLINICAL ALERT: Visual image analysis detected potential "
            "discrepancies with automated tissue/infection classification. "
            "Clinical review is strongly recommended before applying any dressing."
        )
```

Also update `build_narrative_query` to include VLLM description in Sub-query C (patient context sub-query). This means the dense retrieval for the context axis now uses visual evidence, not just the patient's text notes.

**Step 5: New Prompt Section for Multimodal Grounding**

Add to `SYSTEM_PROMPT`:
```
7. If a Visual Wound Analysis section is present in the assessment, you MUST 
   cross-reference the visual description with the T.I.M.E. structured inputs. 
   If any discrepancy exists between visual findings and structured classification, 
   note it in Clinical Notes and recommend clinical review.
```

### 1.3 What this gives you academically

You now have a new ablation dimension — **G4: Visual Enrichment Ablation**:

| Config | Input | Image |
|---|---|---|
| G4-A (baseline) | T.I.M.E. labels only | ✗ |
| G4-B | T.I.M.E. + patient notes | ✗ |
| G4-C | T.I.M.E. + Gemini caption | ✓ Gemini 2.5 Flash |
| G4-D | T.I.M.E. + notes + Gemini caption | ✓ Gemini 2.5 Flash |
| G4-E | T.I.M.E. + notes + GPT-4o caption | ✓ GPT-4o |

Evaluate on the new multimodal testset (Part 2). Measure: FA, Safety Pass, and a new metric — **VLLM-CV Agreement Rate** (how often does the VLLM caption confirm the CV model's classification?).

---

## Part 2 — Limitation 2 (Evaluation Crisis): Build a Multimodal Testset from Public Datasets

### The core problem

Your 32 test cases are purely synthetic text inputs — there are no real wound images, no ground truth T.I.M.E. labels, and no expert-validated dressing recommendations. This means you cannot measure how your system degrades when real CV model errors propagate through the pipeline. You also cannot evaluate the VLLM captioning layer without real wound images to caption.

### 2.1 Publicly Available Datasets to Build From

**WoundCareVQA (Yim et al., 2025) — HIGHEST PRIORITY**  
Published alongside MEDIQA-WV 2025. Contains wound photographs paired with patient-generated text queries and expert-written clinical responses. The dataset is bilingual (English/Chinese), includes wound type metadata, anatomic site labels, and structured wound attributes. The expert responses cover wound care guidance including dressing-related information.  
**Get it:** Contact authors via ACL Anthology (yim-etal-2025-overview) or the shared task page. It is a research dataset released for the shared task — email request is standard.  
**How to use:** WoundCareVQA images + queries → feed wound image to your VLLM captioner → generate T.I.M.E.-equivalent structured inputs → run through VerdaSense → compare against expert responses using your existing RAGAS FA metrics.

**WoundTissue Dataset (Kabir et al., arXiv 2502.10652, Feb 2025)**  
147 wound images with pixel-level labels for 6 tissue types: granulation, slough, maceration, necrosis, bone, tendon. Ground truth tissue proportions can be computed directly from the pixel masks.  
**Get it:** Available at `https://github.com/akabircs/WoundTissue` (stated in paper).  
**How to use:** Ground truth tissue masks → compute accurate T.I.M.E. Tissue component → use as reference to test how accurately K-Means and IME-Net perform. Also use wound images as visual inputs to your VLLM captioner — the pixel-level masks let you verify whether the VLLM description matches ground truth tissue types. This directly validates your captioning layer's accuracy.

**DFUTissue Dataset (Kabir et al., arXiv 2406.16012, June 2024)**  
110 DFU images with tissue labels (granulation, fibrin/slough, callus) by wound experts + 600 unlabelled images. Available from same research group.  
**How to use:** Same as WoundTissue — provides ground truth tissue composition for cross-checking K-Means accuracy and VLLM captioning quality.

**AZH Wound Dataset (UWM Big Data Lab)**  
730 ROI + 538 whole-wound images from AZH Wound Care Center. Wound type labels (venous, diabetic, pressure, surgical). No tissue segmentation masks but wound type labels are useful.  
**Get it:** Request form at `https://sites.uwm.edu/bigdata/datasets/`  
**How to use:** Wound type diversity — add these to multimodal testset for wound aetiology coverage beyond DFU.

**Medetec Wound Database**  
Free, no annotation, ~100+ wound photos across diverse types. Useful for qualitative VLLM captioning evaluation.  
**Get it:** `https://www.medetec.co.uk/files/medetec-image-databases.html`

### 2.2 Testset Construction Strategy — No Expert Annotation Required

The insight is: you do not need expert-annotated dressing recommendations for every wound image. You need wound images where you can **verify your system's input pipeline**. Here is a pragmatic three-tier approach:

**Tier 1 — Tissue Ground Truth Verification (WoundTissue + DFUTissue)**

```
Wound image → compute GT tissue % from pixel mask
                                ↓
              Compare vs K-Means tissue estimate
              Compare vs VLLM caption tissue description
                                ↓
              Measure: K-Means MAE vs GT tissue %
              Measure: VLLM caption tissue mention accuracy
```

This produces a quantitative answer to: *"How much does K-Means tissue classification deviate from ground truth, and does VLLM captioning reduce that error?"* — a clean, publishable result.

**Tier 2 — VQA Response Quality (WoundCareVQA)**

```
Wound image + patient query → VerdaSense multimodal RAG
                                     ↓
                    Compare response against expert answer
                    using: FA (your existing RAGAS metric),
                    ROUGE-L, BERTScore
```

This measures end-to-end multimodal RAG quality against real expert clinical responses — a much stronger evaluation than your 32 synthetic cases.

**Tier 3 — Safety Extension of Current Testset (Your existing 32 cases)**

Augment each of your 32 synthetic test cases with a real wound image sourced from public datasets, matched by wound type and tissue profile. This creates a **32-case image-augmented testset** that bridges your existing structured ablation with the new multimodal evaluation, maintaining continuity across FYP1 ablation and FYP2 multimodal results.

**Practical: How to match images to synthetic cases**

```python
# Example: synthetic case cat_a_type1_dry
# wound_type_expected = 1, granulation=100%, not infected, dry
# → Search WoundTissue/DFUTissue for images where GT shows >80% granulation
# → Verify with VLLM caption that image appears clean and granulating
# → Add image_path field to testset case

# Updated testset schema:
{
  "case_id": "cat_a_type1_dry",
  "category": "A",
  "wound_type_expected": 1,
  "time_payload": { ... },            # existing
  "wound_image_path": "woundtissue/img_047.jpg",  # NEW
  "gt_tissue_pct": { "granulation": 0.89, ... },  # from pixel mask
  "reference": "...",                 # existing
  ...
}
```

---

## Part 3 — Limitation 1 (Knowledge Base): Scale and Enrich the KB Without Manual Effort

### 3.1 The structural problem with your current KB

Your 8 manually chunked sources have several compounding issues:
- **Static and outdated-prone:** Clinical guidelines update every 3–5 years. Manual re-chunking every update is unsustainable.
- **No cross-source linking:** Hydrocolloid dressings appear in WCM, GP, and EWMA. Currently these are three independent chunks with no structural link. When Sub-query A retrieves the WCM hydrocolloid entry and Sub-query B retrieves the EWMA moisture management entry, the generator must infer the relationship between them.
- **Coverage gaps:** 8 sources cannot cover all wound edge cases (e.g., palliative wound care, fungating wounds, radiation wounds).
- **No recency signal:** Your retrieval cannot distinguish whether the WCM (2014, MoH Malaysia) or EWMA (2019) guidance supersedes the other on a contested point.

### 3.2 Targeted Improvements (Practical for FYP Scope)

**Add 3–5 new high-value sources (low effort, high impact)**

These are freely available and directly relevant:
| Source | Content Value | Effort |
|---|---|---|
| **WUWHS Wound Infection consensus 2022** | International infection management — fills your infection axis gap | PDF available, ~20–30 chunks |
| **NICE NG19 Pressure Ulcers 2014/2023** | Evidence-based pressure injury guidelines — adds pressure wound type coverage | Free download |
| **IWGDF DFU Guidelines 2023** | International diabetic foot guidelines — your system has diabetic_foot etiology path but limited KB support | Free download |
| **Malaysian CPG for DFU 2018** | Local guideline — directly relevant for UM clinical context, aligns with GP source language | MOH Malaysia |
| **WOUNDPEDIA dressing product database** | Structured dressing-to-indication mapping | Web-scrape or PDF |

Each new source uses your existing chunking pipeline — the manual work is a one-time 2–4 hour investment per source, and you already have the workflow from v4/v5 development.

**Add a `year` metadata field to all chunks and use it in retrieval**

Currently your Chroma metadata does not have a `year` field. Add it:
```python
# In chunk metadata:
"year": "2022",
"guideline_tier": "international",  # or "national" or "local"
```

Then in `retrieve_chunks_multiaxis`, when two chunks conflict, prefer the newer one. This is a minimal code change with meaningful clinical impact — it prevents your system from recommending a 2014-era approach when a 2023 guideline explicitly supersedes it.

**Cross-source linking via shared dressing category tags**

Add a `dressing_categories` metadata field to each chunk listing the dressing types it covers. This enables a new Sub-query D that retrieves specifically across dressing categories rather than wound types:
```python
# Example metadata enrichment:
"dressing_categories": ["silver", "foam", "antimicrobial"],
"time_axes": ["I"],   # this chunk is primarily relevant to Infection axis
```

This moves your retrieval from pure semantic similarity toward structured clinical knowledge routing — the "multi-path multi-axis" approach the Douyin commenter describes.

---

## Part 4 — Limitation 4 (Evaluation Framework): Beyond RAGAS — Clinical Metrics That Matter

### 4.1 Why RAGAS is necessary but not sufficient

Your current metrics — Faithfulness (FA), Answer Relevance (AR), Context Recall (CR) — are standard RAG quality metrics. They measure whether the generation is grounded in retrieved context and whether the context is relevant. They do NOT measure:
- Whether the recommended dressing is actually appropriate for the wound type
- Whether the recommended dressing would cause harm in the specific patient context
- Whether the recommendation is complete (all 9 output fields populated with valid content)
- Whether the VLLM caption contributed meaningfully to retrieval and generation quality

For an FYP with a clinical deployment aim, you need at minimum two additional metric categories:

### 4.2 Clinical Safety Metrics (Already Partially Implemented — Extend Them)

Your existing safety checker already verifies contraindicated dressings and antibiotic language. Extend it into a formal `ClinicalSafetyScore` with sub-dimensions:

```
ClinicalSafetyScore (0–100):
├── S1: Contraindicated dressing absent         (30 pts) — binary per case
├── S2: Antibiotic recommendation correct       (25 pts) — compare vs reference flag
├── S3: Referral recommendation correct         (20 pts) — compare vs reference flag
├── S4: No fabricated dressing names            (15 pts) — verified against allowed list
└── S5: Confidence-appropriate language         (10 pts) — no false certainty on edge cases
```

Your current testset already has `antibiotic_required`, `referral_required`, `allowed_dressings`, and `contraindicated_dressings` fields per case — you have all the ground truth needed to compute S1–S4 without new annotation.

### 4.3 Completeness Score

A recommendation that correctly identifies the primary dressing but fails to provide T.I.M.E. rationale, frequency, or application notes is clinically incomplete. Add a `CompletenessScore`:

```python
REQUIRED_SECTIONS = [
    "## Primary Dressing",
    "## Secondary Dressing", 
    "## Rationale by T.I.M.E.",
    "## Antibiotic Considerations",
    "## Referral / Escalation",
    "## Contraindicated Dressings",
    "## Dressing Change Frequency",
    "## Application Notes",
    "## Clinical Notes"
]

def compute_completeness(result_text: str) -> float:
    present = sum(1 for s in REQUIRED_SECTIONS if s in result_text)
    return present / len(REQUIRED_SECTIONS)
```

### 4.4 VLLM Caption Quality Metrics (for G4 ablation)

When using the WoundTissue dataset (which has pixel-level GT tissue labels), add:

**Tissue Mention Accuracy (TMA):** Does the VLLM caption mention the correct dominant tissue type?
```
TMA = 1 if GT dominant tissue mentioned in caption, else 0
```

**VLLM-CV Agreement Rate (VAR):** Across all test images, what % of cases does the VLLM caption agree with K-Means tissue classification?
```
VAR = count(VLLM_tissue ≈ KMeans_tissue) / total_cases
```

**Conflict Precision:** Of cases where VLLM flags conflict with CV model, what % are genuine errors (verifiable from GT)?
```
Conflict_Precision = true_CV_errors_caught / total_conflicts_flagged
```

These three metrics, reported in your G4 ablation, give concrete, quantitative evidence that the VLLM captioning layer adds clinical value — which is the academic contribution of the multimodal RAG migration.

---

## Part 5 — The Big Picture: How to Frame This as "Better Than Uploading to ChatGPT-4o"

This is your most important positioning challenge. A clinically unsophisticated user can upload their wound image to ChatGPT-4o and get a reasonable-looking recommendation in 5 seconds. How does VerdaSense justify its existence?

**The answer has four parts:**

### 5.1 Evidence Grounding with Citations (Your Core Differentiator)

ChatGPT-4o does not tell you *which clinical guideline* supports its recommendation. It cannot cite WCM 2014 or EWMA 2019 for a specific claim. VerdaSense's Source 1, Source 2, Source 3 citation system means every recommendation is traceable to a specific document and section. For clinical practice — where accountability matters — this is the difference between a tool a clinician can trust and defend, and a black-box suggestion.

**Make this explicit in your FYP:** The GP Guideline (your Malaysian MOH source) specifically recommends certain dressings that may differ from international guidelines. A local clinical tool using local guidelines is clinically appropriate for Malaysian patients in a way GPT-4o is not.

### 5.2 Contraindication Safety (Your Safety Layer)

ChatGPT-4o cannot reliably detect that a patient is on levothyroxine and therefore cannot use iodine-containing dressings — because it has no structured contraindication checking. Your rule-based safety checker + notes parser does this. Your testset case `cat_b_iodine_thyroid` demonstrates this is a real clinical safety gap.

**Extend this argument:** Document a set of "GPT-4o failure cases" — wounds where GPT-4o zero-shot produces unsafe recommendations (you can test this manually on your 32 test cases as a comparative study). Even 3–4 demonstrated failure cases on safety-critical scenarios (Categories D and E) make a compelling FYP2 argument.

### 5.3 Structured T.I.M.E. Alignment (Your Clinical Framework Differentiation)

VerdaSense structures every recommendation around the T.I.M.E. framework that Malaysian clinical practice uses. GPT-4o does not structure its output this way by default. For a Malaysian wound care nurse using the GP Guideline, VerdaSense's output format is directly compatible with their clinical workflow.

### 5.4 The Hybrid Architecture: Multimodal + Evidence-Grounded (Your Research Contribution)

After implementing the VLLM captioning layer, VerdaSense occupies a unique position:
- GPT-4o: Sees the image, no evidence grounding, no structured safety checks
- Your old RAG: Evidence grounded, structured safety, blind to the image
- **VerdaSense Multimodal RAG: Sees the image, evidence grounded, structured safety checks**

This combination — the SCARWID/MasonNLP approach of visual description feeding into a RAG-grounded generator — is the research contribution. Frame it as: *"VerdaSense combines the visual comprehension of foundation VLLMs with the evidence accountability of clinical RAG, addressing a gap that neither approach alone can fill."*

---

## Part 6 — Practical FYP2 Execution Plan

### What you have already (FYP1 deliverables, which remain valid)

- ✅ Working RAG pipeline (wound_app_v5.py) with v4 DB, 8-source KB
- ✅ 32-case structured ablation testset (v3)
- ✅ Stage 1 Retrieval Ablation (R1–R4) completed
- ✅ Stage 2 Generation Ablation (G1–G3) completed
- ✅ FA=0.8147, Safety=90.6% on best configuration (Gemini 2.5 Flash + G1-C + MedEmbed)

### FYP2 Development Phases

**Phase 1 (Months 1–2): Multimodal Testset Construction**

Priority: Get WoundCareVQA and WoundTissue datasets.
- Email Wen-wai Yim (UW) for WoundCareVQA access
- Download WoundTissue from GitHub
- Match 15–20 images to your existing 32 synthetic cases (Tier 3 approach)
- Compute GT tissue proportions from WoundTissue pixel masks
- Add `wound_image_path` and `gt_tissue_pct` to testset schema

Expected output: `wound_testset_v4_multimodal.json` (32 existing + 15–20 image-matched cases)

**Phase 2 (Months 2–3): VLLM Captioning Layer Integration**

- Implement `generate_vllm_caption()` function (code provided in Part 1)
- Add `wound_image_base64` input field to `/get_recommendation`
- Update `assessment_text` construction to include caption
- Update Sub-query C to use VLLM description in narrative query
- Update `SYSTEM_PROMPT` with multimodal grounding rule

Expected output: `wound_app_v6_multimodal.py`

**Phase 3 (Month 3): G4 Ablation — Multimodal Enrichment Experiment**

Run G4-A through G4-E configurations on the new multimodal testset.
Evaluate: FA, Safety Score, Completeness, TMA, VAR, Conflict Precision.

Expected output: G4 ablation results table — the primary new academic contribution of FYP2.

**Phase 4 (Months 3–4): KB Expansion + New Sources**

Add 2–3 new sources (IWGDF DFU 2023 and WUWHS Wound Infection 2022 as priority).
Re-run Stage 1 retrieval ablation on expanded KB to verify no degradation.

Expected output: `db_wound_care_v5` (expanded KB), Stage 1 re-validation.

**Phase 5 (Month 4): Clinical Metrics + Comparative Study**

- Implement `ClinicalSafetyScore` and `CompletenessScore`
- Run GPT-4o zero-shot (no RAG) on your 32 test cases as a comparative baseline
- Document GPT-4o failure cases on safety-critical scenarios (D and E category)
- Write the "VerdaSense vs ChatGPT-4o" comparison section

Expected output: Comparative results table for FYP2 report Chapter 4.

**Phase 6 (Month 4–5): Response Time and O3 Evaluation**

Measure end-to-end latency for the multimodal pipeline (VLLM caption + retrieval + generation). Target: total < 30 seconds on mobile network. If VLLM captioning adds > 8s, implement an async caption-while-form-submitting design (caption starts generating when image is uploaded, not when form is submitted).

---

## Summary: The FYP Narrative Arc

Your FYP can be structured as a coherent research story:

> *"Current wound care RAG systems are either evidence-grounded but visually blind (accepting only structured T.I.M.E. labels from CV models), or visually capable but evidence-ungrounded (raw VLLM/ChatGPT approaches). VerdaSense addresses this gap by introducing a three-layer architecture: (1) VLLM visual enrichment that generates clinically structured wound descriptions from the raw wound image, cross-checking and enriching the CV model's structured outputs; (2) multi-axis evidence retrieval from curated clinical guidelines aligned to T.I.M.E. assessment axes; and (3) evidence-grounded generation with source citation and rule-based safety checking. Ablation studies across retrieval, generation, and visual enrichment configurations demonstrate that the combined multimodal architecture achieves higher clinical safety and recommendation faithfulness than either unimodal text RAG or VLLM-only approaches."*

This narrative addresses all three of your objectives:
- **O1 + RQ1:** Architecture described — multimodal RAG with VLLM captioning + evidence retrieval
- **O2 + RQ2:** Evaluation with FA, Safety Score, Completeness, TMA/VAR on structured + image testset
- **O3 + RQ3:** Mobile deployment with latency measurement; response time under 30s target

---

## References

- Karim, A.H.M.R. & Uzuner, Ö. (2025). MasonNLP at MEDIQA-WV 2025: Multimodal RAG with LLMs for Medical VQA. *ACL ClinicalNLP 2025*. arXiv:2510.13856.
- Yim, W. et al. (2025). Overview of MEDIQA-WV 2025 Shared Task on Woundcare VQA. *ACL ClinicalNLP 2025*. https://aclanthology.org/2025.clinicalnlp-1.3/
- Durgapraveen, B. et al. (2025). EXL Health AI Lab at MEDIQA-WV 2025: Mined Prompting and Metadata-Guided Generation. *ACL ClinicalNLP 2025*. arXiv:2511.10591.
- Busaranuvong, P. et al. (2025). SCARWID: Explainable Multimodal Wound Infection Classification. *ACM THRI*.
- Forte, A.J. et al. (2025). AI vs. MD: Benchmarking ChatGPT and Gemini for Complex Wound Management. *JMIR*, 14(24):e8825.
- Kabir, M.A. et al. (2025). Deep Learning for Wound Tissue Segmentation: A Comprehensive Evaluation using A Novel Dataset. arXiv:2502.10652. GitHub: akabircs/WoundTissue.
- Kabir, M.A. et al. (2024). Wound Tissue Segmentation in DFU Images Using Deep Learning: A Pilot Study. arXiv:2406.16012.
- Xia, Y. et al. (2025). MMed-RAG: Versatile Multimodal RAG System for Medical VLMs. *ICLR 2025*.
- Microsoft (2026). Advancing Wound Care with Foundation Models. *Microsoft Health + Life Sciences Blog*.
- Lewis, P. et al. (2020). RAG for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.

