# VerdaSense — Three Critical Architecture Decisions
## Testset Alignment, Dataset Reality Check, and Multimodal Architecture Choice

**For:** Tee Qi Jing | FYP2 | Universiti Malaya  
**Date:** May 2026  
**Purpose:** Direct answers to three sharp architectural questions raised after the previous roadmap

---

## Question 1: Testset–KB Alignment Problem
### "If I use WoundCareVQA as my evaluation testset, its ground truth answers come from different clinical sources — not from my KB. Doesn't this break my evaluation?"

**Yes — you are completely correct. This is a real evaluation validity problem.**

Here is exactly why it breaks:

Your RAGAS **Faithfulness (FA)** score measures:  
> *"Is every claim in the generated answer supported by the retrieved context?"*

Your RAGAS **Context Recall (CR)** measures:  
> *"What proportion of the ground truth answer's claims are present in the retrieved chunks?"*

If the reference answer in your testset is grounded in WoundCareVQA's clinical sources (e.g., a US wound clinic's internal protocols), and your KB contains WCM 2014 (Malaysian MOH), EWMA, GP Guideline Malaysia — the two will recommend similar dressings conceptually but with different specific phrasing, different cited sources, and sometimes different thresholds. Context Recall would be artificially low, and FA would be penalised for claims that are clinically correct but absent from your specific KB chunks. Your evaluation would be invalid.

### What actually makes an evaluation testset valid for YOUR system

A testcase is only valid for VerdaSense if **all three of these hold**:

```
1. The reference answer is derivable from YOUR KB sources
2. The reference_contexts are actual chunks from YOUR KB
3. The input (TIME payload or wound image) produces retrievable 
   evidence from YOUR KB
```

Your current 32 cases in wound_testset_v3.json satisfy all three. Look at what you confirmed:
- `reference_contexts[0]` → "Ministry of Health Malaysia, 2019; Wound Care Manual 2014" — **that is your WCM chunk**
- `reference_contexts[1]` → "WCM Modern/Advanced Dressing - Film ... MOH Malaysia, Chapter 14" — **that is your WCM chunk**
- `reference` answer is generated FROM those KB chunks, not from an external source

This is exactly right. Your testset is properly KB-aligned.

### How to build a MULTIMODAL testset that stays KB-aligned

The solution is to separate the evaluation into two independent but linked layers:

```
Layer A — Existing text RAG evaluation (UNCHANGED):
  Input:  Synthetic TIME payload
  Eval:   FA, CR against your existing reference_contexts from KB
  Cases:  Your existing 32 cases
  Status: Already done — valid

Layer B — NEW Visual Enrichment evaluation:
  Input:  Real wound image → VLLM caption → feeds into existing pipeline
  Eval:   Does the VLLM caption improve or degrade FA / Safety / Completeness?
  Cases:  Same 32 TIME payloads + matched wound images from public datasets
  Key:    Reference answer and reference_contexts REMAIN YOUR KB CHUNKS
          The wound image only affects HOW the input is enriched,
          not what the correct answer is
```

The wound image is NOT the source of the ground truth answer. The KB is. The image is the source of the VLLM caption, which is an additional input signal. So evaluation validity is preserved: you are measuring whether adding a visual description enriches retrieval and generation quality relative to your KB, not whether your KB matches some external dataset.

### Concrete testset augmentation plan

For each of your 32 existing cases, find a real wound image that visually matches the TIME profile:

```python
# Example matching logic
case_image_map = {
    "cat_a_type1_dry": {
        # 100% granulation, not infected, dry
        # Look for: clean granulating wound, red/pink, minimal exudate
        "image_source": "Medetec or AZH dataset",
        "image_path": "wounds/granulating_dry_001.jpg",
        "match_rationale": "Predominantly granulating, no infection signs visible"
    },
    "cat_b_iodine_thyroid": {
        # 77% granulation, infected, dry
        "image_source": "WoundCareVQA or Medetec",
        "image_path": "wounds/infected_dry_granulating.jpg",
        "match_rationale": "Mild erythema visible, moderate granulation, dry wound bed"
    }
}
```

You can match images using the VLLM itself as a labelling assistant:
```python
MATCHING_PROMPT = """Given this wound image, answer these questions:
1. Dominant tissue type: Granulation / Slough / Necrotic / Mixed
2. Infection signs visible: Yes / No  
3. Exudate level: Dry / Moderate / High
4. Edge characteristics: Advancing / Non-advancing / Unclear
Answer in JSON only."""
```

Run this over your available public wound images, collect the structured responses, and match them to your existing 32 synthetic TIME profiles. You do NOT need perfect pixel-level accuracy — approximate visual matching is sufficient because you are evaluating the VLLM caption layer's contribution to retrieval, not the accuracy of wound classification.

**Result:** A `wound_testset_v4_multimodal.json` with 32 existing cases + a new `wound_image_path` field per case. Your reference answers and reference_contexts remain unchanged from v3 — full KB alignment preserved.

---

## Question 2: WoundTissue Dataset is Incomplete on GitHub
### Confirmed — The Dataset Availability Problem

You are right that `github.com/akabircs/WoundTissue` appears incomplete. This is a common issue with academic dataset releases — the paper promises availability but the repository is either under embargo, partially uploaded, or pending IRB clearance.

### Alternative public datasets with confirmed availability

Based on the research, here are datasets with verified or highly likely accessibility:

**Tier 1 — Confirmed public, no access request needed:**

| Dataset | Content | Access |
|---|---|---|
| **Medetec Wound Database** | ~100 wound photos, diverse types, NO tissue masks, no annotation | `medetec.co.uk/files/medetec-image-databases.html` — free download |
| **DFU Challenge 2021 (DFUC2021)** | 15,683 DFU image patches, wound boundary labels | `github.com/uwm-bigdata/wound-image-segmentation` |
| **AZH Wound Dataset** | 730 ROI + 538 whole wound images, wound type labels (venous/diabetic/pressure/surgical), no tissue masks | UWM Big Data Lab — `sites.uwm.edu/bigdata/datasets/` request form |
| **FUSeg (Foot Ulcer Segmentation)** | 1,210 foot ulcer images with wound boundary masks, NO tissue-type masks | IEEE DataPort, MDPI open access |

**Tier 2 — Request-based but typically granted within days:**

| Dataset | Content | Access |
|---|---|---|
| **WoundCareVQA (MEDIQA-WV 2025)** | Wound images + patient queries + expert clinical responses. Structured wound attributes. This is what you want for qualitative evaluation. | Email Wen-wai Yim `wyim@uw.edu` — standard academic data request, usually fast |
| **DFUTissue (Kabir et al., 2024)** | 110 DFU images with tissue labels (granulation/fibrin/callus) + 600 unlabelled. Limited tissue types but annotated. | Email `akabir@csu.edu.au` — same lab as WoundTissue, paper is published and code/data promised |

**Tier 3 — Synthetic data generation (no dataset needed):**

For images, you can use a VLLM to generate wound IMAGE DESCRIPTIONS for synthetic test inputs you already have, rather than requiring actual wound photographs. More on this in the architecture section below.

### Realistic recommendation for FYP2

Given dataset access uncertainty, **do not make your entire FYP2 evaluation dependent on a single external dataset**. The safest strategy is:

```
Primary evaluation (high confidence, no external dependency):
  → Your 32 synthetic cases (testset v3) — already done
  → Extend with manually sourced 10–15 Medetec images (free, no request)
  → Use these for VLLM captioning quality assessment only

Secondary evaluation (if WoundCareVQA access granted):
  → Use WoundCareVQA images as visual inputs ONLY
  → Keep your KB as the reference for answers (Layer B approach above)
  → Do not use their expert responses as your reference answers

Fallback (if no suitable images obtainable):
  → VLLM-generated synthetic wound descriptions (see below)
  → Sufficient to demonstrate G4 ablation contribution
```

---

## Question 3: Architecture Decision — Replace CV Pipeline, Run Parallel, or Build Standalone?
### "Do I need to replace companion CV models with VLLM entirely? Or run parallel? Or build VerdaSense as an independent CV + dressing 2-in-1 multimodal RAG?"

This is the most important architectural decision for your FYP. There are three distinct options, and you need to understand what each one means for your project scope, evaluatability, and academic positioning.

---

### Option A: VLLM Replaces CV Pipeline (Full Replacement)

```
Wound Image
     │
     ▼
 VLLM API (GPT-4o / Gemini 2.5 Flash)
 "Describe tissue, infection, moisture, edge"
     │
     ▼
 Structured TIME output (extracted from VLLM response)
     │
     ▼
 VerdaSense RAG → Dressing Recommendation
```

**What this means:**  
K-Means tissue classification, IME-Net infection/moisture/edge classifier, and MobileSAM tissue decomposition are all bypassed. VerdaSense becomes completely self-contained. The companion CV models from your seniors are no longer used by your module.

**Pros:**
- Full independence — you own the entire pipeline end-to-end
- Evaluatable without any companion model accuracy uncertainty
- Simpler integration — one API call replaces three CV model inferences
- VLLM (GPT-4o / Gemini) produces richer TIME-equivalent descriptions than binary class outputs
- Clean FYP boundary: your FYP does CV-equivalent perception AND RAG reasoning AND safety checking

**Cons:**
- VLLM tissue percentage estimation is inherently less precise than pixel-level segmentation. GPT-4o will say "approximately 60% granulation" not "62.3% by pixel area". For wound area measurement, this matters.
- Loses your seniors' LoRA-fine-tuned MobileSAM boundary segmentation, which is actually the most accurate component of the current pipeline.
- Cost: each recommendation query requires a VLLM vision API call (Gemini 2.5 Flash ~$0.001/image at current pricing — acceptable for research, needs consideration for deployment scale).
- May weaken the collaborative FYP narrative that the app combines multiple students' work.

**Verdict: DO NOT CHOOSE THIS AS YOUR PRIMARY ARCHITECTURE** for the mobile app. However, **implement it as an evaluation mode** — it lets you evaluate VerdaSense completely independently of companion model accuracy, which is critical for your ablation study.

---

### Option B: VLLM + CV Pipeline in Parallel (Enrichment Layer)

```
Wound Image ─────────────────────────────────┐
     │                                        │
     ▼                                        ▼
Companion CV Pipeline                    VLLM Captioner
(K-Means + IME-Net + MobileSAM)         (Gemini 2.5 Flash)
     │                                        │
     ▼                                        ▼
Structured TIME labels              Narrative wound description
(necrotic%, slough%, infection...)   ("60% granulation, mild
     │                                periwound erythema...")
     │                                        │
     └─────────────────┬──────────────────────┘
                        ▼
              Conflict Detection Module
              (Does VLLM agree with CV?)
                        │
              ┌─────────┴──────────┐
              │                    │
           AGREE               CONFLICT
              │                    │
         Normal RAG          Append warning +
         processing          escalate to clinical review
                        │
                        ▼
             Multi-axis RAG retrieval
             (TIME + VLLM description + patient notes)
                        │
                        ▼
             Grounded generation + safety check
```

**What this means:**  
Both the companion CV pipeline AND the VLLM captioner run independently. Their outputs are compared (conflict detection) and fused. The CV labels provide structured precision; the VLLM caption provides visual richness and a cross-check signal.

**Pros:**
- Best clinical reliability — multiple signal sources, cross-validation
- Preserves your seniors' work and the collaborative pipeline
- VLLM description enriches retrieval sub-queries with visual context
- Conflict detection flags cases where CV models may be wrong → safety net
- Gradual migration path — you can add the VLLM layer without breaking anything

**Cons:**
- VerdaSense's quality is still partially dependent on companion model accuracy (for the structured labels)
- Harder to evaluate VerdaSense's contribution independently from CV model quality
- More complex system overall
- For your FYP evaluation, you still cannot fully isolate "RAG + VLLM" from "CV pipeline errors"

**Verdict: This is the right DEPLOYMENT architecture** for the mobile app. It makes clinical sense — multiple independent signals are always safer than one. But it is harder to evaluate academically.

---

### Option C: VerdaSense as a Standalone Multimodal RAG (2-in-1 Vision + Dressing)

```
Wound Image + Optional Patient Notes
              │
              ▼
┌─────────────────────────────────────────────────────┐
│           VerdaSense Multimodal RAG                 │
│                                                     │
│  Step 1: VLLM TIME Extraction                       │
│    Image → GPT-4o/Gemini → Structured TIME report   │
│    (Tissue%, Infection, Moisture, Edge)             │
│                                                     │
│  Step 2: Multi-axis Retrieval                       │
│    TIME report + VLLM description → Sub-queries     │
│    → Dense + Sparse retrieval from KB               │
│                                                     │
│  Step 3: Evidence-grounded Generation               │
│    G1-C prompt + retrieved context → Recommendation │
│                                                     │
│  Step 4: Safety Check + Confidence Tier             │
│    Rule-based check + VLLM uncertainty signal       │
└─────────────────────────────────────────────────────┘
              │
              ▼
     Structured Dressing Recommendation
     (with evidence citations from KB)
```

**What this means:**  
VerdaSense accepts raw wound image + notes and handles everything: visual interpretation, TIME assessment, retrieval, generation, safety. It is completely independent of companion models. Companion models can optionally feed their structured outputs in as an additional channel, but they are not required.

**Pros:**
- Fully evaluatable — you control all inputs and can measure everything
- Clean academic contribution — a complete multimodal clinical RAG system
- Can be deployed standalone (not dependent on seniors' pipeline being ready, deployed, or accurate)
- Directly comparable to GPT-4o zero-shot: "VerdaSense multimodal vs raw VLLM" is a clean comparison because both start from the same input (wound image)
- Strong FYP narrative: your system does MORE than just RAG — it does visual clinical assessment + evidence-grounded recommendation in one unified pipeline
- Addresses your concern about K-Means inaccuracy at the root — the VLLM's visual assessment is more nuanced than K-Means colour clustering

**Cons:**
- VLLM TIME extraction introduces imprecision in tissue percentages (inherently approximate)
- Needs careful prompt engineering to get consistent structured TIME output from VLLM
- Still needs a structured TIME extraction step (not free-form text) to feed your existing classifier and retrieval logic
- Companion CV models become optional rather than primary — may affect the broader app integration story

**Verdict: This is the right RESEARCH/EVALUATION architecture** for your FYP ablation study. It makes your contribution independently verifiable, directly comparable to baselines, and fully under your control.

---

### Recommended Approach: Dual-Mode Architecture

The answer is not "choose one" — it is **build Option C, design it to also accept Option B inputs**:

```python
@app.post("/get_recommendation")
async def get_recommendation(
    # MODE 1: Structured CV inputs (Option B — companion pipeline feeds in)
    necrotic_pct:      float = Form(0.0),
    slough_pct:        float = Form(0.0),
    granulation_pct:   float = Form(0.0),
    infection:         str   = Form(""),
    moisture:          str   = Form(""),
    edge:              str   = Form(""),
    
    # MODE 2: Visual input (Option C — standalone, image-first)
    wound_image_base64: str  = Form(""),   # raw wound image
    
    # Both modes accept these
    notes:             str   = Form(""),
    tissue_confidence: float = Form(0.0),
):
    # Determine input mode
    has_cv_inputs = any([necrotic_pct, slough_pct, granulation_pct, 
                         infection.strip(), moisture.strip()])
    has_image    = bool(wound_image_base64.strip())
    
    if has_image:
        # VLLM extracts TIME — Option C path (for evaluation)
        # OR enriches CV inputs — Option B path (for deployment)
        vllm_time, vllm_caption = extract_time_from_image(wound_image_base64)
        
        if has_cv_inputs:
            # Option B: Parallel — use CV labels, VLLM adds enrichment + conflict check
            time_to_use = cv_inputs  # trust CV labels
            caption_for_rag = vllm_caption  # enrich retrieval
            conflict = detect_conflict(cv_inputs, vllm_time)
        else:
            # Option C: Standalone — VLLM is the only source of TIME
            time_to_use = vllm_time   # VLLM labels become the structured input
            caption_for_rag = vllm_caption
            conflict = False
    else:
        # Option A (text-only, no image): existing pipeline unchanged
        time_to_use = cv_inputs
        caption_for_rag = ""
        conflict = False
    
    # Rest of pipeline unchanged from v5...
```

This dual-mode design means:
- For your **FYP ablation evaluation** → use Option C mode (image-only input), evaluate independently of companion models
- For the **mobile app deployment** → use Option B mode (CV labels + image enrichment), maximise clinical reliability
- Your existing 32-case testset → remains valid for text-only baseline comparison
- New multimodal testset → uses Option C mode with wound images from public sources

---

## Pulling It Together: What to Implement for FYP2

Here is the clearest possible summary of what this means for your work:

### For Architecture
Build **Option C as primary evaluation mode, Option B as deployment mode**, sharing the same codebase. The input router above handles both.

### For Testset
Your **existing 32 cases remain your primary evaluation dataset**. You extend them with wound images matched visually to each case's TIME profile (sourced from Medetec free images + any accessible public dataset). The reference answers and reference_contexts do NOT change — KB alignment is preserved.

### For VLLM TIME Extraction (the key new function)

```python
VLLM_TIME_EXTRACTION_PROMPT = """You are a wound care clinical specialist.
Analyse this wound image and provide a structured assessment.

Respond ONLY in this exact JSON format, nothing else:
{
  "tissue": {
    "granulation_pct": <0-100, integer>,
    "slough_pct": <0-100, integer>,
    "necrotic_pct": <0-100, integer>,
    "confidence": "<low|medium|high>",
    "description": "<1-2 sentence clinical description of tissue>"
  },
  "infection": {
    "status": "<Not infected|Locally infected|Systemic infection suspected>",
    "confidence": "<low|medium|high>",
    "visible_signs": "<list any visible signs or 'none visible'>"
  },
  "moisture": {
    "level": "<Dry|Moderate exudate|High exudate>",
    "confidence": "<low|medium|high>",
    "description": "<brief description>"
  },
  "edge": {
    "status": "<Advancing wound edge|Non-advancing wound edge>",
    "confidence": "<low|medium|high>",
    "description": "<brief description>"
  },
  "additional_observations": "<any clinically relevant features not above, max 2 sentences>"
}

Important: tissue percentages must sum to 100. 
Be conservative — if uncertain, lower your confidence field rather than guessing."""

def extract_time_from_image(image_b64: str) -> tuple[dict, str]:
    """
    Extract structured TIME assessment from wound image using VLLM.
    Returns: (time_dict, narrative_caption)
    """
    import google.generativeai as genai
    import base64, json, io
    from PIL import Image
    
    img_bytes = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(img_bytes))
    
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content([VLLM_TIME_EXTRACTION_PROMPT, img])
    
    raw = response.text.strip()
    # Strip markdown code fences if present
    raw = raw.replace("```json", "").replace("```", "").strip()
    
    try:
        vllm_time = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: return empty time dict, use caption only
        return {}, raw
    
    # Build narrative caption from the structured output
    caption = (
        f"TISSUE: {vllm_time['tissue']['description']} "
        f"(Granulation {vllm_time['tissue']['granulation_pct']}%, "
        f"Slough {vllm_time['tissue']['slough_pct']}%, "
        f"Necrotic {vllm_time['tissue']['necrotic_pct']}%)\n"
        f"INFECTION: {vllm_time['infection']['status']}. "
        f"{vllm_time['infection']['visible_signs']}\n"
        f"MOISTURE: {vllm_time['moisture']['description']}\n"
        f"EDGE: {vllm_time['edge']['description']}\n"
        f"ADDITIONAL: {vllm_time['additional_observations']}"
    )
    
    return vllm_time, caption
```

### For Ablation — The New G4 Experiment

| Config | Image Input | CV Labels | VLLM TIME | VLLM Caption |
|---|---|---|---|---|
| G4-A | ✗ | ✓ (from synthetic testset) | ✗ | ✗ |
| G4-B | ✓ | ✓ (ground truth from testset) | ✗ | ✓ Gemini |
| G4-C | ✓ | ✗ | ✓ Gemini | ✓ Gemini |
| G4-D | ✓ | ✓ (ground truth) | ✓ Gemini | ✓ (conflict detection active) |

- G4-A vs G4-B: Does VLLM caption alone (without replacing CV labels) improve FA/Safety?
- G4-A vs G4-C: Does VLLM TIME replacement match the quality of ground truth TIME labels?
- G4-B vs G4-D: Does conflict detection when both sources present add safety value?

### Academic Framing

Your FYP contribution is no longer just "a RAG system for wound dressings." It becomes:

> *"VerdaSense is a dual-mode multimodal clinical RAG that can operate either as an enrichment layer over structured computer vision outputs, or as a standalone vision-language system for evidence-grounded wound dressing recommendation. Ablation studies demonstrate that VLLM-based visual assessment combined with evidence-grounded RAG generation achieves [X]% on ClinicalSafetyScore and FA=[Y], outperforming both raw VLLM (no evidence grounding) and unimodal text RAG (no visual input) on [Z] evaluation cases."*

---

## Summary: Direct Answers to Your Three Questions

| Question | Direct Answer |
|---|---|
| Must testset align with KB? | **Yes, strictly.** Use WoundCareVQA wound images as visual inputs only — keep your KB as the reference source for answers. Your existing 32-case reference answers and reference_contexts do NOT change. |
| WoundTissue GitHub is incomplete — now what? | Use Medetec (free, immediate) + email DFUTissue author + email WoundCareVQA author. For evaluation, you only need images as visual inputs — you do NOT need pixel-level tissue masks. |
| Replace CV pipeline or run parallel or standalone? | **Build standalone (Option C) for evaluation, parallel (Option B) for deployment, sharing one codebase via an input router.** Do NOT fully replace or fully depend on companion models — a dual-mode architecture gives you both independent evaluatability and deployment reliability. |
