# VerdaSense FYP2 — Master Plan

**Student:** Tee Qi Jing (23004894) · Universiti Malaya
**Date:** June 2026 · Post-FYP1 Viva + Collaborator Discussion
**Purpose:** Comprehensive, honest plan for FYP2 — architecture decisions, ablation map, realistic timeline, and the full argument for why VerdaSense is meaningfully better than both zero-shot GPT and a simple rule table.

---

## Preamble: What FYP1 Proved and What FYP2 Must Prove

FYP1 proved that a T.I.M.E.-structured multi-axis RAG system can retrieve clinically relevant wound care guideline evidence and generate a structured dressing recommendation with **CR = 0.897, FA = 0.814, Safety Pass Rate = 90.6%** — outperforming single-query retrieval (R1-A CR = 0.824) and ungrounded prompting (G1-A). The ablation-best configuration is fixed: **R1-C + Dense + k=6 + BGE-large-en-v1.5 + G1-C grounded prompt**.

FYP1's gap, identified by both your viva panel and yourself, is that the system is **unimodal and single-turn**: it trusts upstream CV model labels blindly without ever seeing the wound image, and it produces one structured response with no subsequent clinical reasoning. It also doesn't yet have clinical validation by a human expert, and it doesn't yet include a debridement pathway for complex wounds (Types 5–8), which Ms Saw's three-panel review unanimously flagged.

**FYP2 must prove:** A multimodal RAG system — where a VLM directly observes the wound photograph and generates a multi-aspect clinical caption that enriches the generation stage — produces better dressing recommendations than the unimodal system, and that the hybrid rule-RAG architecture outperforms both pure zero-shot GPT and a pure rule table. This is measurable, novel, and clinically grounded.

---

## Part 1 — Honest Assessment of the Five Pain Points

### Pain Point 1: Unimodal RAG Blindly Trusts CV Labels → Multimodal RAG

**The real problem:**
The upstream CV pipeline (K-Means + IME-Net) gives you percentage tissue labels and binary I/M/E classifications. These are predictions, not ground truth — IME-Net has classification error, K-Means cluster thresholds are approximate, and neither model sees the whole clinical picture: the wound's depth, its periwound skin condition, anatomical location, size, or the qualitative texture of the wound bed. A clinician examining the same photograph would note at least five dimensions the T.I.M.E. payload does not encode.

**The multimodal RAG answer:**
A VLM (Vision Language Model) looks at the actual wound photograph and generates a structured clinical caption covering dimensions the upstream CV pipeline does not produce. This caption is then passed as additional context to the generation LLM alongside the retrieved KB chunks. The recommendation is now grounded in both the structured T.I.M.E. payload (from the CV pipeline) and the VLM's direct visual observation of the wound.

**What R5 already tells you about architecture:**
R5 tested four strategies for injecting VLM captions into the retrieval layer and found that all three caption injection strategies *hurt* retrieval — with R5-B (caption replacing Sub-query C) causing −6.6 pp CR and −18.75 pp HR@6 degradation. The mechanistic explanation: guideline KB chunks are written in functional clinical language ("high absorbency for moderate-to-heavy exudate"), while VLM captions describe appearance ("greenish slough with perilesional erythema"). These are different semantic registers that BGE-large's embedding space cannot bridge at the retrieval stage.

**The correct architecture — already validated by R5:**

```
Wound image
    ├── Upstream CV pipeline → T.I.M.E. payload
    │       (YOLO + T-SegNet + K-Means + IME-Net)
    │
    ├── VLM captioner → multi-aspect clinical caption   ← FYP2 addition
    │       (per-patient, generation-layer only)
    │
    └── Retrieval (UNCHANGED from FYP1)
            T.I.M.E. payload → R1-C sub-queries → ChromaDB BGE-large
            → top-6 chunks
  
    All three feed into → Generation LLM → Structured recommendation
        inputs: [retrieved chunks] + [T.I.M.E. payload] + [VLM caption]
```

**Why this is better than unimodal:**
The generation LLM now has three information streams: (1) guideline evidence from the KB, (2) structured T.I.M.E. assessment from CV, and (3) the VLM's direct visual observation. If IME-Net says "Not infected" but the VLM caption observes "perilesional erythema, foul odour, and purulent discharge consistent with spreading infection", the generation LLM can reconcile this discrepancy and flag it clinically — something the unimodal system cannot do. This cross-validation between CV labels and VLM observation is the single most important clinical contribution of FYP2.

**Honest limitation:**
VLM captions can hallucinate. GPT-4o-Vision and Gemini 2.5 Flash Vision both occasionally describe wound features that aren't there, particularly for images with poor lighting or unusual angles (common in patient self-photography). FYP2 must measure this — include a caption quality validation step in the evaluation. One approach: for the 32 test cases with known wound type ground truth, measure how often the VLM caption's T.I.M.E. axis descriptions agree with the ground truth labels.

---

### Pain Point 2: Why RAG Instead of Rule-Based?

This is the most important conceptual question of your entire FYP, and the viva panel's challenge deserves a complete, empirically-grounded answer. Let's build it properly.

**What the rule-based system can do:**
The wound care algorithm from Ms Saw (rawatan_yang_disarankan_woundtype_1234.JPG + 5678.JPG) is genuinely comprehensive: given wound type 1–8, it maps directly to dressing materials, antibiotic need, and surgical procedure. The WCM Wound Care Manual flowchart and the Garis Panduan algorithm are already implemented as the `classify_wound()` function in your system. This rule-based pre-classifier is **correct, appropriate, and should stay**. Rules are ideal for deterministic decisions where the inputs are structured and the logic is clinical protocol.

**What the rule-based system cannot do:**

| Patient scenario                                                                                  | Rule table handles it?                       | What handles it instead?                                  |
| ------------------------------------------------------------------------------------------------- | -------------------------------------------- | --------------------------------------------------------- |
| WT3 wound + patient notes: "I have a silver allergy"                                              | ❌ No allergy field in rule table            | RAG retrieves allergy contraindication chunk              |
| WT2 wound + notes: "wound is 6 months old, I have diabetes"                                       | ❌ No comorbidity or chronicity in rules     | RAG retrieves DFU + chronic wound chunks                  |
| "Why is alginate better than normal gauze for my wet wound?"                                      | ❌ Rules don't explain                       | RAG retrieves mechanism chunk, LLM explains               |
| WT5 wound + VLM observes "tunnelling wound, very deep"                                            | ❌ Rules don't have depth                    | RAG retrieves cavity wound management chunk               |
| Patient notes: "I'm 8 months pregnant"                                                            | ❌ No pregnancy field in rules               | RAG retrieves iodine/silver pregnancy contraindication    |
| "My dressing keeps falling off — what can I do?"                                                 | ❌ Not in any rule table                     | RAG retrieves application technique chunk                 |
| VLM caption contradicts CV label: visual shows spreading erythema but IME-Net says "not infected" | ❌ Rules can't reconcile conflicting signals | LLM observes both and flags discrepancy in Clinical Notes |

**The correct framing for your viva:**

> "VerdaSense is a hybrid system. The rule-based classifier (`classify_wound()`) determines wound type 1–8 from the T.I.M.E. assessment — this is appropriate because wound type classification follows deterministic clinical protocol. RAG then handles everything the rules cannot: patient-specific comorbidities in free text, allergy flags, clinical explanation, complex multi-factor cases, and the synthesis of multiple evidence sources. For FYP2, a VLM caption provides a third input stream — the system's own visual observation of the wound — allowing the generation model to cross-validate the CV pipeline's labels and flag discrepancies. None of these three capabilities (free-text patient notes, evidence-grounded explanation, visual cross-validation) can be implemented with a rule table."

**Why this is not just "zero-shot GPT":**
Zero-shot GPT-4o with a wound image has no access to the Malaysian MOH wound care algorithm, the WCM First Edition, ANZBA guidelines, or your collaborator's locally-validated clinical protocols. It will generate a plausible-sounding dressing recommendation with no traceability to any specific clinical guideline. It cannot say "Source 1: Wound Care Manual, Chapter 7, page 54" — it can only say "based on general medical knowledge". For a patient self-care tool that may influence wound management decisions, the difference between a traceable, guideline-grounded recommendation and an untraceable GPT response is not academic — it is the entire clinical validity argument.

---

### Pain Point 3: Wound Depth + Wound Etiology Classification

#### Wound Depth (Deep vs Superficial)

**Clinical value:** High. Wound depth directly determines whether you need a surface dressing (sheet) or a cavity filler (rope/ribbon alginate, hydrofibre, foam cavity). The current WT1–8 system is completely blind to depth — a WT2 wound (clean, wet) that is 1 mm deep needs a foam sheet; the same WT2 wound that is a 3 cm tunnelling cavity needs alginate rope + secondary foam. The dressings are different products.

**Implementation approach:**Pure image-based depth classification from a phone camera is unreliable — camera angles, lighting, and lack of scale markers make quantitative depth estimation impossible. The pragmatic approach is a **two-input depth estimate**:

1. **VLM depth observation**: ask the VLM captioner to specifically assess wound depth/volume from the image (shallow/surface vs evident depth/cavity). Include specific language in the caption prompt: "Estimate wound depth: is this wound superficial (skin surface only) or does it appear to have depth/volume that would require cavity filling?"
2. **Patient self-report field (optional UI addition)**: a simple question in the app UI: "How deep does your wound look? (a) Surface scratch/shallow, (b) Deeper wound with visible tissue/fat, (c) Deep hole or cavity." This is one button click for the patient and eliminates most ambiguity.

Combine VLM + patient report into a binary `wound_depth: "superficial" | "cavity"` field passed to the generation prompt. Then add a depth-handling instruction to the system prompt: "For cavity wounds, recommend cavity-filling dressing forms (rope alginate, ribbon hydrofibre, cavity foam) rather than sheet dressings. Always specify the dressing form in your recommendation."

**KB impact:** The current WCM and SFP sources have chunks on cavity wound management. Adding `wound_depth` as a ChromaDB metadata field and including it in Sub-query B ("dressing mechanism for [wound_depth] [exudate_level] wound") would pull cavity-specific chunks when appropriate. This is a clean, low-risk KB and retrieval enhancement with direct clinical impact.

**Realistic verdict:** Add wound depth. It is the highest-value, lowest-complexity classification enhancement for FYP2. Cost: one VLM caption field, one optional patient UI question, one metadata field, one prompt instruction.

#### Wound Etiology Classification (DFU, VLU, Pressure Ulcer, Burn, Traumatic, Surgical)

**Ms Saw's clinical clarification (post-viva discussion):**

> "Basically all wound etiology can use similar sets of wound dressings, except for vascular wounds which might need different."

This single statement from your clinical collaborator substantially changes the value calculation for etiology classification. Here is what it means precisely.

**Why dressings are mostly etiology-agnostic:**For DFU, VLU, Pressure Ulcer, Traumatic, and Surgical wounds — the wound contact dressings (alginate, foam, silver, hydrogel, hydrocolloid) are all selected based on T.I.M.E., not etiology. The etiology-specific interventions for these wound types are *not dressings*:

- DFU → offloading (pressure relief from the foot) + blood glucose control + referral
- VLU → compression bandaging (multi-layer bandage system) + referral for vascular assessment
- PrU → repositioning schedule + pressure-redistributing mattress + referral for Stage 3–4

Your WT1–8 system already drives the correct dressing selection for all of these. The T.I.M.E. assessment is sufficient for the dressing decision.

**The vascular wound exception — real but already handled:**
Arterial ulcers with dry ischemic eschar are the one case where standard moist wound healing is contraindicated. Applying hydrogel to dry necrotic tissue on an ischemic limb can worsen outcomes (moist environment promotes bacterial growth when there is insufficient blood supply for immune response). However, this presentation will already produce WT5, 7, or 8 in your classification (dry + necrotic burden > 25%), which triggers `referral_required = True`. The appropriate clinical action is urgent referral — not a different dressing. Your system already handles this correctly. It does not need an etiology classifier to produce the right output.

**What etiology classification actually adds to VerdaSense — and what it doesn't:**

| Value                                                    | Does etiology classification add this?                                              |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Change the primary dressing recommendation               | ❌ No — dressings are T.I.M.E.-driven (Ms Saw confirmed)                           |
| Change the secondary dressing                            | ❌ No — same logic                                                                 |
| Add management caveats in generated text                 | ✅ Yes — "for DFU patients, offloading is essential alongside dressing management" |
| Strengthen the referral trigger for high-risk etiologies | ✅ Yes — DFU, VLU, vascular wounds should always be referred                       |
| Improve KB retrieval precision                           | ⚠️ Marginal — KB already has ANZBA for burns, ISTAP for skin tears               |
| Justify training a new CV classifier                     | ❌ No — the dressing output doesn't change enough to warrant the dataset cost      |

**The correct FYP2 approach — minimal, targeted, no new classifier:**Add two lightweight etiology-context inputs to the existing UI:

1. **"Do you have diabetes?"** (Yes / No / Not sure) — if Yes + wound on foot/ankle area → inject "DFU management context" flag into generation prompt
2. **VLM anatomical location** from the caption — already being added for wound depth purposes

These two signals, fed as flags into the generation prompt, allow the LLM to append the appropriate management caveat in the Clinical Notes section ("As this wound is on the foot of a diabetic patient, specialist referral is required regardless of wound severity. Offloading and glycaemic control are essential.") without any new KB chunks, retrieval changes, or trained classifiers.

**Realistic verdict:** Do NOT build an etiology CV classifier for FYP2. Ms Saw's clinical statement removes the primary justification — the dressings are the same. Add a single "diabetes?" demographic question to the UI and let the VLM caption note anatomical location. That is sufficient for the management caveat text and the referral trigger. Etiology classification as a full pipeline component is out of scope.

---

### Pain Point 4: Dressing Product Gallery

**Ms Saw's request:** After the recommendation, patients should see what the recommended dressing looks like, where to get it, and what brands are available in Malaysia.

**The right approach — surgeon images are your asset:**
You already have the highest-value product reference material in `surgeon_images/`: the DyaMed Biotech product charts (dyamed_biotech_products_and_its_recommendation_of_usage.JPG, pemilihan_material_dressing.JPG, dyamed_biotech_TIME_dressings.JPG) and the treatment recommendation table (treatment_recommendation_based_on_the_flow_chart_of_wound_care_algorithm.JPG) showing specific product names (Dermacyn, Flaminal Hydro, Flaminal Forte, Zorflex, Drawtex, RenoFoam, RenoCare) mapped to wound types 1–8 with step-by-step application instructions.

This is more actionable than anything from the clinical guideline PDFs — it's locally validated by Ms Saw and maps directly to products available through her clinic's distributor. The product gallery doesn't need web crawling for FYP2 scope. It needs:

1. Transcription of the surgeon images into structured KB chunks (add as new source `authority: "DyaMed_Biotech_Clinical_Protocol"`)
2. A dressing type → product name mapping (static JSON, maintained manually for now)
3. A simple UI component in the app that shows product name + category after the recommendation

**For FYP2 scope:** Transcribe and ingest surgeon image content into KB (Part 3 below). Build a minimal static product gallery as a UI component — dressing type → product name + availability note (OTC / clinic-only / hospital). Web crawling for pharmacy links is future work.

**Realistic verdict:** Low-priority but low-effort if done the right way. Don't crawl e-commerce. Transcribe the surgeon images, add to KB, build a static mapping JSON. One afternoon of work for the product gallery mapping; the ingestion is higher value and has research impact.

---

### Pain Point 5: Drop Conversational RAG, Keep Multimodal RAG

**This is a defensible scope decision.** Conversational RAG and multimodal RAG are both valid FYP2 extensions. You've chosen multimodal because it addresses the core limitation (unimodal blindness) and aligns with the full-system vision (patient uploads wound image → AI observes it directly). This is architecturally cleaner for a 6-month scope.

**What you lose by dropping conversational RAG:**
The strongest single answer to "why not rule-based?" was the conversational scenario — rules can't answer "my wound looks worse after 3 days, what should I do?" However, multimodal RAG provides a different equally strong answer: "the VLM directly observes visual features the rules don't capture, and cross-validates the CV pipeline labels." Both are valid; multimodal is the right choice if you must choose one.

**What you gain by dropping it:**
No need to build a conversational testset (20–30 multi-turn sessions × 3–5 turns each = ~100 annotated reference answers). No session memory management complexity. The ablation is cleaner and faster. G4 (multimodal generation ablation) is more novel than G4 (multi-turn retrieval) because conversational RAG has more prior work.

**One thing to be careful about:**
The viva panel will ask "can the patient ask follow-up questions?" You should have a prepared answer: "In-session follow-up questions were descoped to focus on the more foundational multimodal contribution. The architecture supports adding a conversational layer as a direct extension — the session memory and multi-turn retrieval design is documented as FYP3/future work. The current FYP2 scope validates whether the VLM visual input genuinely improves recommendation quality, which is the prerequisite for a robust conversational system."

---

# Part 2 — The FYP2 Architecture in Full

```
PATIENT INTERACTION FLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1: Patient uploads wound image + optional notes
         + 1 demographic question (diabetes? Yes/No/Not sure)

Step 2: Upstream CV Pipeline (UNCHANGED — senior's models)
    YOLO → wound detection + bbox
    T-SegNet/MobileSAM → wound region segmentation
    K-Means → tissue % (granulation, slough, necrotic)
    IME-Net → I/M/E classification (binary labels)
    Output: T.I.M.E. payload

Step 3: Rule-based Pre-classifier (UPDATED from FYP1)
    classify_wound(T.I.M.E.) → wound_type (1–8)
                              + referral_required
                              + antibiotic_required
                              + dfu_flag (from diabetes question + VLM location)
                              + wound_depth (from VLM + patient input)

Step 4: VLM Captioner [NEW — FYP2 CORE CONTRIBUTION]
    Input: wound image + T.I.M.E. payload + patient demographics
    Model: GPT-4o-Vision or Gemini 2.5 Flash Vision (to ablate in G4)
    Caption covers:
        - Visual T.I.M.E. validation (does VLM agree with CV labels?)
        - Wound depth / cavity estimate
        - Periwound skin condition (maceration, erythema, fragility)
        - Anatomical location
        - Wound dimensions estimate (if ruler visible, or comparative)
        - Any visual features suggesting urgency (spreading redness, crepitus signs, exposed tissue)
    Output: structured clinical caption (~300–500 words)

Step 5: Retrieval (UNCHANGED from FYP1 — R5-A confirmed optimal)
    T.I.M.E. payload → R1-C multi-axis sub-queries (A+B+C)
    Sub-query A: wound type algorithm chunk (metadata filter)
    Sub-query B: dressing mechanism (exudate/infection/tissue profile)
    Sub-query C: patient notes + demographics
    → ChromaDB BGE-large-en-v1.5, Dense, k=6

Step 6: Generation [EXTENDED — FYP2]
    Inputs:
        [retrieved chunks × 6]
        [T.I.M.E. payload]
        [VLM caption]                   ← NEW
        [wound_depth]                   ← NEW
        [dfu_flag if diabetes + foot location]  ← NEW (lightweight)
    System prompt: G1-E (G1-C + debridement + depth + etiology instructions)
    Output: 9-section structured recommendation
        + debridement pathway for WT5-8 (NEW)
        + cavity filler guidance for deep wounds (NEW)
        + VLM-T.I.M.E. discrepancy alert if detected (NEW)

Step 7: Safety Checker (UPDATED)
    Rule-based post-generation checks (existing)
    + sepsis bypass gate (NEW — pre-RAG keyword check)
    + VLM discrepancy flagging (NEW)

Step 8: Product Gallery (MINIMAL, STATIC)
    Dressing type → product name → availability (OTC / clinic-only)
    Source: transcribed surgeon images (DyaMed Biotech protocol)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Part 3 — KB Expansion Strategy

The current KB (138 chunks, 8 sources) is adequate for FYP1's unimodal scope. FYP2's multimodal generation needs broader context because the VLM caption will introduce new clinical dimensions (depth, periwound skin, etiology) that current chunks may not fully address. Priority is strictly ordered.

### Priority 1 (Do First — Highest Clinical Value, Zero Dataset Risk)

**Transcribe and ingest surgeon images as a new KB source**This is the single highest-impact action for FYP2. The surgeon images contain:

- Product-specific treatment protocols for WT1–8 (Dermacyn, Flaminal, Zorflex, Drawtex, RenoFoam, RenoCare) with application instructions
- The Malaysian Wound Care Algorithm flowchart in condensed tabular form
- DyaMed Biotech T.I.M.E.-to-product mapping
- `pemilihan_material_dressing.JPG`: the visual decision tree (Luka Tanpa Jangkitan / Luka Dengan Jangkitan × exudate level → primary + secondary dressing)

Action: manually transcribe text from all 5 clinical surgeon images into structured JSON chunks. Assign metadata: `authority: "DyaMed_Biotech_Clinical_Protocol"`, `guideline_type: "local_clinical_protocol"`, `source_origin: "KK_Sultan_Ismail"`. Estimated 15–25 new chunks. This directly justifies why the AI recommends Flaminal or Zorflex over generic dressing categories — locally validated, by your collaborator's distributor.

This also provides the product names needed for the product gallery and makes the "why RAG vs rule-based" answer concrete: the RAG now retrieves locally-validated clinical protocols that include specific Malaysian product recommendations, which no static rule table encodes.

### Priority 2 (Do If Etiology Classification Implemented)

**MOH Malaysia DFU Clinical Practice Guideline (2020)** — 15–20 chunks. Essential for DFU etiology patients. Available from MOH Malaysia website.

**IWGDF DFU Guidelines 2023** — 10–15 chunks. International authority supporting the MOH CPG. Available from iwgdf.org.

**Malaysian CPG Pressure Ulcer (MOH 2019)** — 10–15 chunks. For bedridden patients.

### Priority 3 (High Value for Generation Quality)

**Patient education FAQ source** — 10–15 chunks. Cover practical questions: "Can I shower?", "How do I remove the dressing?", "My dressing smells — is that normal?", "How do I know if my wound is healing?". Source from NHS wound care patient information leaflets (publicly available) or MOH patient education materials. These cannot be answered by clinical guideline chunks (which are clinician-facing), but they are exactly what self-care patients will need.

### RCH Metadata Fix (Do in Week 1)

Add `"population": "paediatric"` to all 11 RCH chunks. Update retrieval to exclude RCH unless patient notes contain paediatric keywords. This is a correctness and patient safety fix.

---

## Part 4 — VLM Caption Design (G4 Experiment)

### 4.1 What the Caption Must Cover

R5 established that generic T.I.M.E.-mirroring captions (tissue composition %, colour descriptors) do not help retrieval. For FYP2, the caption is generation-layer context only, and it should provide **clinical information the T.I.M.E. payload doesn't contain**:

```
Caption dimensions (what to prompt the VLM to assess):
1. T.I.M.E. cross-validation: does the visual appearance agree with the CV labels?
   e.g., "Tissue: visual confirms ~80% granulation consistent with CV report. 
   However, note perilesional erythema and slight warmth not captured by IME-Net."

2. Wound depth/volume: superficial vs cavity estimate
   e.g., "Wound appears to have significant depth — estimated cavity of 1–2 cm depth
   requiring cavity-filling dressing form rather than a sheet dressing."

3. Periwound skin condition: maceration, erythema, fragility, dryness
   e.g., "Periwound skin shows mild maceration (softening and whitening) 
   within 2 cm of wound edge, consistent with High Moisture classification."

4. Anatomical location: from image context
   e.g., "Wound is located on the dorsum of the right foot — consistent 
   with diabetic foot ulcer location."

5. Wound dimensions estimate: if scale is visible, or comparative
   e.g., "Wound appears approximately 3–4 cm × 2–3 cm based on visible context."

6. Urgency flags: any visual features requiring immediate attention
   e.g., "No spreading erythema, no visible necrotic tissue extending beyond 
   wound margins, no signs of crepitus. No visual urgency indicators."

7. Dressing-relevant inferences: what dressing properties does this wound visually require?
   e.g., "The high exudate visible at wound margins and the surrounding maceration 
   suggest this wound requires a highly absorbent dressing with moisture-managing 
   secondary layer and periwound skin protection."
```

Dimension 7 is the most important change from R5's caption design. R5 captions described appearance; FYP2 captions must include **dressing-mechanism inference**. This directly addresses R5's L3 limitation (caption-to-query alignment mismatch) and makes the caption useful to the generation LLM rather than just descriptive.

### 4.2 Caption Prompt Template

```
You are a clinical wound assessment assistant. Examine this wound photograph and provide 
a structured clinical assessment covering these dimensions. Write in the language of 
clinical wound care documentation, focusing on dressing selection implications.

Patient T.I.M.E. Assessment from CV pipeline (for cross-validation only):
- Tissue: {tissue_breakdown}
- Infection: {infection_label}  
- Moisture: {moisture_label}
- Edge: {edge_label}
- Wound Type: {wound_type}
Patient context: {demographics_summary}  (e.g., "Patient has diabetes" if applicable)

Assess the wound across these dimensions:
1. T.I.M.E. cross-validation: Does the visual appearance agree with the above CV labels? 
   Note any discrepancies, especially for infection and tissue composition.
2. Wound depth/volume: Estimate whether this is a superficial/surface wound or 
   a cavity wound requiring filling. Describe any tunnelling or undermining if visible.
3. Periwound skin: Describe condition (maceration, erythema, fragility, healthy).
4. Anatomical location: State the apparent anatomical location from the image.
5. Wound dimensions: Estimate wound size if scale context is visible.
6. Urgency flags: Note any visual features requiring immediate clinical attention 
   (spreading erythema, exposed deep tissue, crepitus signs, obvious necrosis at margins).
7. Dressing implications: Based purely on visual observation, what dressing properties 
   does this wound appear to require (absorption level, antimicrobial, moisture donation, 
   cavity filling, periwound protection)? Note: dressing type is determined by wound 
   characteristics, not wound cause — focus on what you see, not what caused the wound.

Be concise and clinically precise. Flag any limitations due to image quality or angle.
```

### 4.3 Which VLM to Use — and How to Ablate It

For the G4 ablation, test at minimum two VLMs:

- **GPT-4o-Vision** (OpenAI): strongest general visual reasoning, best for nuanced clinical description
- **Gemini 2.5 Flash Vision** (Google): lower cost, comparable quality, same provider as your best G2-D LLM

Optional third: **LLaVA-Med** (open-source medical VLM) — if you want an open-source multimodal option, though quality may be lower for wound-specific images.

The G4 ablation should not just test VLM caption presence/absence — it should test the interaction between caption quality (which VLM) and recommendation quality (which generation LLM). The most important ablation question: **does the VLM caption improve Faithfulness (FA) and Answer Relevance (AR) scores versus the unimodal baseline?**

---

## Part 5 — FYP2 Ablation Map

All FYP1 experiments (R1–R5, G1–G3) are fixed. FYP2 adds:

### Immediate Fix Experiments (Before Main Ablation)

| Exp            | Description                                       | What to Test                                                                                                        | Expected Outcome                                          |
| -------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| **G1-E** | Prompt improvements from Ms Saw's clinical review | G1-C (baseline) vs G1-C + debridement pathway + high exudate frequency caveat + sepsis gate + time-based escalation | FA ↑ 0.01–0.03, Safety Pass Rate ↑ (debridement cases) |

Run G1-E on the existing 32-case testset first thing in FYP2. It's one afternoon, uses the same evaluation pipeline, and documents the clinical improvements from Ms Saw's review.

### New FYP2 Experiments

| Exp            | Research Question                                                                         | Versions to Test                                                                     | Primary Metric                             |
| -------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------ |
| **G1-F** | Does the patient-friendly dual-mode schema (Part 13) hold up vs the old 9-section output? | old structured output vs new patient-friendly schema (both cited internally)         | FA + conciseness/readability + H1 ratings  |
| **G4-A** | Does VLM caption improve generation vs no caption?                                        | G4-A0: no caption (unimodal baseline) vs G4-A1: with VLM caption                     | FA, AR (RAGAS)                             |
| **G4-B** | Does caption VLM choice matter?                                                           | G4-B1: GPT-4o-Vision caption vs G4-B2: Gemini Vision caption                         | FA, AR                                     |
| **G4-C** | Does wound depth field improve cavity wound cases?                                        | With vs without wound_depth in generation prompt                                     | Safety Pass Rate for deep wound test cases |
| **G4-D** | Does DFU flag add value for diabetic foot cases?                                          | Without DFU flag vs with DFU management caveat injected                              | FA + Safety on Cat B DFU/diabetic cases    |
| **R6**   | Does wound depth metadata filter improve retrieval for cavity wound cases?                | Dense without depth filter vs Dense with depth filter on cavity wound subset         | CR on cavity wound cases                   |
| **H1**   | Human clinical evaluation (Ms Saw)                                                        | Blinded Likert rating of 8 Cat A recommendations + multimodal vs unimodal comparison | Clinical Concordance Rate                  |

### New Evaluation Metrics

| Metric                                            | Definition                                                                                        | Applies To             |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ---------------------- |
| **VLM Caption Accuracy Rate**               | % of captions where VLM T.I.M.E. assessment agrees with ground truth labels (within ±1 category) | G4 test cases          |
| **VLM-T.I.M.E. Discrepancy Detection Rate** | % of cases with known label errors where VLM caption correctly flags the discrepancy              | Adversarial test cases |
| **Cavity Wound Coverage Rate**              | % of cavity wound cases receiving cavity-filling dressing recommendation (not sheet dressings)    | G4-C test cases        |
| **Etiology-Specific Recall**                | % of Cat B cases (DFU, burns, skin tear) retrieving ≥1 etiology-specific chunk                   | R6 test cases          |
| **Clinical Concordance Rate**               | % of Cat A cases rated ≥4/5 on Clinical Accuracy by Ms Saw (H1)                                  | Human evaluation       |

### New Testset Requirements

**Multimodal testset (G4):**
The existing 32-case testset uses structured T.I.M.E. payloads with wound type images pre-assigned. For G4, each test case needs an actual wound image (already partially done in R5 — WT01–WT08 images). The gap: cases need per-case images, not one image per wound type. For FYP2, you can use: (a) the 8 archetype images with 4 cases each (simpler, replicates R5's limitation but is defensible), or (b) source at least 2–3 additional images per wound type from public wound datasets (AZH, WSNet, Medetec) for more diverse visual testing. Option (b) is stronger; do it if time allows.

**Adversarial T.I.M.E. discrepancy cases (8–10 cases):**
Create test cases where the CV pipeline labels are intentionally misaligned with the wound image (e.g., image is clearly infected but IME-Net label says "Not infected"). Reference answer: VLM should flag the discrepancy. Measures VLM cross-validation capability.

**Cavity wound cases (6–8 cases):**
Create test cases with deep/cavity wounds. Reference answer must specify cavity-filling dressing form. Measures G4-C depth integration.

**Cat B extension for etiology:**
Add 4–6 Cat B cases with explicit etiology context (2 DFU, 2 VLU/PrU, 2 burns) to measure G4-D etiology injection benefit.

---

## Part 6 — Human Clinical Evaluation (H1 — Highest Priority Deliverable)

The viva panel's hardest unanswered question was "how do you know the output is clinically correct?" FYP2 must answer this with direct evidence.

**Design (send to Ms Saw, target 25–30 minutes):**

**Part A — Blinded recommendation quality (8 cases, Likert 1–5):**Show wound description + generated recommendation (both unimodal FYP1 output AND multimodal FYP2 output, in randomised order, labelled System A and System B). For each, ask:

1. Clinical Accuracy: "Is the primary dressing recommendation clinically appropriate?" (1–5)
2. Safety: "Does this recommendation contain anything potentially harmful?" (1–5)
3. Completeness: "Does this recommendation include all clinically important information (dressing, frequency, debridement if needed, referral if needed)?" (1–5)

This simultaneously produces Clinical Concordance Rate AND a multimodal vs unimodal comparative signal.

**Part B — Caption quality spot check (4 cases):**
Show Ms Saw the VLM-generated caption for 4 wound images. Ask: "Does this clinical description of the wound image appear accurate based on what you can see in the photograph?" (Yes / Mostly Yes / Mostly No / No). This validates VLM caption quality with a clinical eye.

**Part C — Debridement completeness check (4 WT5–8 cases):**
Show the FYP2 recommendation for Types 5–8. Ask: "Does this recommendation appropriately address wound debridement?" (Yes / Partially / No). This directly validates the G1-E prompt improvement.

**Logistics:** Google Form with embedded images. Send no more than 2 weeks after you have FYP2 generation working. Give Ms Saw 2 weeks to respond. This is your highest-stakes deliverable — treat it as a project deadline.

---

## Part 7 — Six-Month Realistic Timeline

Half a year (26 weeks) from July 2026 to December 2026.

### Phase 0 — Corrections and Setup (Weeks 1–2)

| Task                                                                                                                                                                                                 | Deadline | Why                                                        |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ---------------------------------------------------------- |
| ~~Fix `classify_wound()` referral logic (all locally infected → referral)~~ **CANCELLED — clinically incorrect; see Part 12.** Verify the rule still matches the MOH algorithm (it does). | Week 1   | Avoids over-referral; keeps rules guideline-aligned        |
| Verify testset_v3 referral ground-truth matches the algorithm (WT6/7/8 = referral; WT3/4 = antibiotic only)                                                                                          | Week 1   | Confirms FYP1 metrics were computed against correct labels |
| Re-run G2-D + G3-G safety evaluation on corrected testset                                                                                                                                            | Week 1   | Definitive FYP2 baseline                                   |
| Add RCH population metadata + retrieval filter                                                                                                                                                       | Week 1   | Patient safety fix                                         |
| Transcribe surgeon images → JSON chunks → ingest into new KB source                                                                                                                                | Week 2   | Foundation for product gallery + KB expansion              |

### Phase 1 — Prompt Improvements + G1-E Ablation (Weeks 2–4)

| Task                                                                                             | Deadline |
| ------------------------------------------------------------------------------------------------ | -------- |
| Add debridement pathway to G1-C system prompt → G1-E (use the per-wound-type matrix in Part 12) | Week 2   |
| Add high exudate dressing frequency caveat                                                       | Week 2   |
| Add time-based escalation statement for infected wounds                                          | Week 2   |
| Implement sepsis bypass gate (pre-RAG keyword check)                                             | Week 3   |
| Run G1-E ablation (3 runs, full 32-case testset)                                                 | Week 3   |
| Document G1-E results                                                                            | Week 4   |

### Phase 2 — VLM Caption Infrastructure (Weeks 4–8)

| Task                                                                                            | Deadline |
| ----------------------------------------------------------------------------------------------- | -------- |
| Design multi-aspect VLM caption prompt (Section 4.2)                                            | Week 4   |
| Implement VLM captioner endpoint (GPT-4o-Vision via OpenAI, Gemini Vision via Google)           | Week 5   |
| Run caption generation on all 8 wound type images (R5 test images)                              | Week 5   |
| Qualitative review of captions (compare to ground truth labels)                                 | Week 5   |
| Integrate caption into generation pipeline (passed to LLM as additional context, not retrieval) | Week 6   |
| Add wound depth field to caption prompt + generation prompt                                     | Week 6   |
| Add etiology inference from demographics + VLM                                                  | Week 7   |
| Update UI for 3 demographic questions + depth input                                             | Week 7   |
| Internal smoke test: run 8 Cat A cases through multimodal pipeline                              | Week 8   |

### Phase 3 — G4 Ablation (Weeks 8–14)

| Task                                                             | Deadline |
| ---------------------------------------------------------------- | -------- |
| Build adversarial T.I.M.E. discrepancy testset (8–10 cases)     | Week 9   |
| Build cavity wound testset (6–8 cases)                          | Week 9   |
| Extend Cat B with etiology test cases (6 cases)                  | Week 9   |
| Run G4-A: no caption vs with caption (3 runs each, full testset) | Week 10  |
| Run G4-B: GPT-4o-Vision vs Gemini Vision caption (3 runs each)   | Week 11  |
| Run G4-C: with vs without wound_depth (cavity wound subset)      | Week 12  |
| Run G4-D: with vs without etiology flag (Cat B subset)           | Week 12  |
| Run R6: wound depth metadata filter (cavity wound subset)        | Week 13  |
| Compute VLM Caption Accuracy Rate on all G4 test cases           | Week 13  |
| Document all G4 + R6 results                                     | Week 14  |

### Phase 4 — Human Clinical Evaluation (Weeks 12–18)

| Task                                                                                      | Deadline |
| ----------------------------------------------------------------------------------------- | -------- |
| Design H1 evaluation form (Google Form, 3-part)                                           | Week 12  |
| Generate FYP2 multimodal recommendations for 8 Cat A + 4 Cap B cases                      | Week 13  |
| Send H1 form to Ms Saw                                                                    | Week 14  |
| Chase response + collate results by                                                       | Week 18  |
| Compute Clinical Concordance Rate, Multimodal vs Unimodal delta, Debridement completeness | Week 18  |

### Phase 5 — Product Gallery + UI Polish (Weeks 14–18)

| Task                                                                                         | Deadline |
| -------------------------------------------------------------------------------------------- | -------- |
| Build static product gallery JSON (dressing type → brand name → availability → image URL) | Week 15  |
| Add product gallery UI component to wound_app_unimodal.py                                    | Week 16  |
| UI polish: show demographic questions, wound depth input, display VLM caption summary        | Week 17  |

### Phase 6 — Writing and Viva Preparation (Weeks 18–26)

| Task                                                                                                    | Deadline |
| ------------------------------------------------------------------------------------------------------- | -------- |
| Write Chapter 3 (Methodology) — multimodal architecture, KB expansion, G4 design                       | Week 20  |
| Write Chapter 4 (Results) — G1-E, G4-A through G4-D, R6, H1                                            | Week 22  |
| Write Chapter 5 (Discussion) — hybrid architecture argument, multimodal vs unimodal, clinical validity | Week 23  |
| Prepare viva slides                                                                                     | Week 25  |
| Mock viva with supervisor                                                                               | Week 26  |

---

## Part 8 — The Viva Defence: Pre-Loaded Answers

### "Why not just use a rule table?"

> "VerdaSense uses both. The rule-based pre-classifier determines wound type 1–8 from the T.I.M.E. assessment — this is appropriate for deterministic protocol logic. RAG then handles everything rules cannot: patient comorbidities and allergies from free-text notes, evidence-grounded clinical explanation, and complex multi-factor edge cases. FYP2 adds a third capability that neither rules nor text RAG provide: a VLM directly observes the wound photograph and cross-validates the CV pipeline's predictions. In [X%] of our test cases, the VLM caption identified a clinically relevant discrepancy between the CV labels and the actual visual wound presentation — for example, detecting spreading erythema consistent with infection that IME-Net had classified as 'not infected'. Rules and CV models cannot do this. That is why RAG augmented with multimodal observation outperforms a rule table."

### "Isn't this just GPT-4o with a wound image?"

> "No. Zero-shot GPT-4o has no access to the Malaysian MOH Wound Care Manual, the Garis Panduan, DyaMed Biotech's locally-validated clinical protocols, or the 8 clinical guideline sources our KB is built on. It will generate a plausible response grounded in general training data with no traceability to any specific guideline. Our system returns a recommendation where every clinical claim is traceable to a specific source and page. In our human evaluation, [X] of 8 cases were rated ≥4/5 for clinical accuracy by our clinical collaborator — a standard that zero-shot GPT has never been formally evaluated against for Malaysian wound care practice."

### "How do you know the VLM caption is accurate?"

> "We measured VLM Caption Accuracy Rate across our 32-case testset — the fraction of cases where the VLM's T.I.M.E. axis assessments agreed with the ground truth labels. We found [X%] agreement. We also designed 8–10 adversarial test cases where the CV labels are intentionally misaligned with the image, and measured whether the VLM correctly flags these discrepancies. Our clinical collaborator additionally reviewed 4 captions as part of the H1 human evaluation. Caption quality limitations are documented honestly in the Discussion chapter."

### "Why not add conversational multi-turn?"

> "Conversational RAG was descoped to focus on the more foundational multimodal contribution. The hybrid rule-RAG-multimodal architecture is the prerequisite for a robust conversational system — you need to first validate that the system's single-turn recommendations are clinically grounded and visually aware before extending to multi-turn sessions. The conversational extension is fully designed and documented as future work."

---

## Part 9 — What Makes VerdaSense FYP2 Genuinely Novel

Three things will make this system stand out against the "zero-shot GPT can do this" objection:

**1. The first wound care RAG system with VLM cross-validation of CV pipeline labels**
No existing academic wound care RAG system validates the upstream computer vision pipeline's predictions by directly observing the wound image. VerdaSense FYP2 adds a VLM that independently assesses the wound and can flag disagreements with the structured labels. This is clinically meaningful — the most dangerous clinical failure mode (missed infection) can be partially caught by visual cross-validation.

**2. A complete FYP1→FYP2 ablation chain covering both retrieval and multimodal generation**
R1–R5 (retrieval) + G1–G3 (generation unimodal) + G1-E + G4-A through G4-D (generation multimodal) + R6 (depth-conditioned retrieval) + H1 (human evaluation) = 15+ ablation experiments with a coherent research narrative. This is the depth of empirical rigour that separates a serious FYP from a demo project. Most clinical AI papers do not have this degree of systematic ablation.

**3. KB grounded in locally-validated Malaysian clinical protocols**
By ingesting the DyaMed Biotech protocols from the surgeon images, VerdaSense's KB now includes material that Ms Saw has directly validated for her clinic's practice. The recommendations can cite a locally-endorsed protocol alongside international guidelines. Zero-shot GPT does not know about Flaminal Forte, Drawtex, or Zorflex — your KB does, and it knows which wound type each product is appropriate for.

---

## Summary: What FYP2 Delivers

| Dimension                           | FYP1                           | FYP2                                                                     |
| ----------------------------------- | ------------------------------ | ------------------------------------------------------------------------ |
| **Modality**                  | Unimodal (T.I.M.E. text only)  | Multimodal (T.I.M.E. + wound image VLM caption)                          |
| **CV validation**             | None — trusts labels blindly  | VLM cross-validates CV labels, flags discrepancies                       |
| **Wound depth**               | Not captured                   | Estimated from VLM + patient input; cavity vs superficial                |
| **Etiology**                  | Generic (WT1–8 only)          | Inferred from demographics + VLM + notes NER                             |
| **KB**                        | 138 chunks, 8 sources          | ~170–200 chunks, 10–12 sources incl. locally-validated DyaMed protocol |
| **Debridement**               | Not addressed                  | Debridement pathway for WT5–8 (G1-E fix)                                |
| **Sepsis gate**               | None                           | Pre-RAG keyword bypass for emergency presentation                        |
| **Product info**              | None                           | Static product gallery (DyaMed products mapped to wound types)           |
| **Human eval**                | Planned but not done           | H1 completed: blinded Likert + multimodal vs unimodal + caption quality  |
| **Ablation**                  | R1–R5, G1–G3 (8 experiments) | +G1-E, G4-A/B/C/D, R6, H1 (7 new experiments)                            |
| **Viva answer on "why RAG?"** | Architectural argument only    | Empirical: VLM discrepancy detection + H1 Clinical Concordance Rate      |

---

## Part 10 — The Core Architectural Question: Is RAG Actually Needed Here?

*This section was added post-viva to directly address the concern: "wound type → dressing is already fixed as rules — is RAG limited in my case? Should I reshape the FYP?"*

### 10.1 The Honest Diagnosis: What Your System Actually Does vs What You Think It Does

After reading the actual KB chunks in `ingestion_output_ai/`, the architecture is clear:

**The GP WT1-8 algorithm chunks (13 chunks) ARE the rule table, just encoded as KB chunks:**

- `chunk_id: 52ef696853c7` — "WT1: all dressings except silver, charcoal, special"
- `chunk_id: 4643f10b8894` — "WT2: Foam / Alginate / Hydrofibre / Polymeric membrane"
- `chunk_id: c0a350e36ecf` — "WT3: Tulle / Hydrogel / Hydrocolloid / Silver / Iodine"
- ... (WT4 through WT8 follow the same pattern)

These chunks are always retrieved by Sub-query A via metadata filter `wound_type=N`. They are deterministic. Sub-query A is effectively a **rule lookup disguised as retrieval**. This is correct and intentional — pinning the algorithm chunk ensures the LLM always sees the authoritative Malaysian MOH dressing categories for the patient's wound type.

**The concern "wound type → dressing is fixed as rules" is accurate for this part of the system.** The GP algorithm chunks confirm exactly this: given T.I.M.E. input, the dressing category list is deterministic.

**But this is not what RAG is for in VerdaSense.** Sub-query A answers "which dressing categories are clinically indicated?" Sub-queries B and C, and the LLM generation step, answer the four questions that a rule table fundamentally cannot:

---

### 10.2 The Four Things a Rule Table Cannot Do

A rule table encodes: `IF wound_type = N THEN dressing_categories = [X, Y, Z]`.

That is all it can encode. Everything else requires retrieval and generation:

**1. WHY — Clinical rationale and explanation**
A patient sees "Silver dressing" in their recommendation. They ask why. The rule table has no answer. The KB has SFP chunks (36 chunks) and WCM chunks that explain silver dressing mechanism, when it is preferred over tulle, why it is contraindicated in clean wounds. Sub-query B retrieves these. The LLM synthesises them into "Silver dressing is recommended because it releases silver ions into the wound environment, which have broad-spectrum antimicrobial activity without inducing antibiotic resistance. It is specifically indicated for locally infected wounds (your wound type 3) where topical antimicrobial control is the priority."

**2. HOW — Application instructions and product specifics**
The rule table gives a category: "Hydrogel". It does not say: apply Flaminal Hydro in a 0.5 cm layer using a spatula, cover with tulle dressing, change EOD or when saturated. This is the DyaMed protocol — currently NOT in the KB. It is the single biggest gap between "dressing category selected" and "patient can actually apply this dressing correctly at home." Without this, a self-care patient reading "Hydrogel" does not know what to buy, how to apply it, or how often to change it. Adding the DyaMed surgeon image chunks fills this gap directly.

**3. WHEN — Dressing change frequency with clinical nuance**
The rule table has no frequency information. Current KB chunks contain some frequency guidance but it is scattered and generic ("varies depending on wound and dressing type"). The DyaMed protocol chunks contain specific, wound-type-matched frequency guidance: "Change Flaminal Forte every 2–4 days starting from EOD, once exudate control is optimum"; "Zorflex can remain 3–7 days". This is the actionable self-care information patients need.

**4. PATIENT-SPECIFIC MODIFICATION — Free text notes processing**
This is the clearest case for RAG over rules. The rule table has no mechanism to process "I have a silver allergy", "I'm 8 months pregnant", "wound has been there for 6 months, I have diabetes", or "I'm on warfarin". These require semantic retrieval from the KB (find contraindication chunks matching the patient's notes) and LLM synthesis (apply the contraindication to modify the dressing recommendation). Sub-query C handles this. No rule table can.

---

### 10.3 The Two-Layer Architecture Stated Clearly

```
LAYER 1 — DECISION LAYER (Rules + Pinned Retrieval)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
classify_wound(T.I.M.E.) → wound_type 1–8
                         → referral_required
                         → antibiotic_required
Sub-query A (pinned) → GP WT1-8 algorithm chunk
                    → ALWAYS retrieves the authoritative dressing 
                      category list for this wound type

This layer answers: WHICH dressing categories are indicated?
This IS rule-based. It is correct, fast, and deterministic.

LAYER 2 — EVIDENCE LAYER (RAG)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sub-query B → dressing mechanism retrieval
           → why this category, contraindications, properties
Sub-query C → patient notes processing
           → free text allergies, comorbidities, duration
[FYP2] VLM caption → visual context for generation

This layer answers: WHY, HOW, WHEN, and WHAT TO WATCH FOR?
This REQUIRES RAG — not encodable as rules.

GENERATION (LLM)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Takes Layer 1 output (dressing category) + Layer 2 evidence
→ Synthesises the COMPLETE dressing plan:
   - Primary dressing (specific product recommendation)
   - Secondary dressing
   - Application instructions
   - Change frequency
   - Contraindications for this patient
   - Debridement pathway (if WT5-8)
   - Referral/antibiotic guidance
   - Monitoring: when to escalate
   - Clinical rationale (why this dressing, with source citation)
```

**This is why the FYP title is correct and the RAG approach is right.** The title is "Evidence-Grounded Clinical Decision Support." Evidence-grounded means every recommendation is traceable to a specific clinical guideline source — something a rule table or zero-shot GPT cannot provide. The evidence layer is what makes VerdaSense different from a lookup table.

---

### 10.4 The Current KB Gap: Why the System Cannot Yet Produce a Complete Dressing Plan

After reading all 8 `_kept.json` files, here is the honest diagnosis of what the current KB is strong and weak at:

**What the current 138-chunk KB does well:**

| Source           | What it provides                                                                | Use in pipeline                                |
| ---------------- | ------------------------------------------------------------------------------- | ---------------------------------------------- |
| GP (13 chunks)   | WT1-8 algorithm — dressing categories per wound type                           | Sub-query A anchor (pinned)                    |
| WCM (40 chunks)  | Wound biology, bacteriology, acute/chronic wound management, DFU/VLU assessment | Sub-query B (clinical context)                 |
| SFP (36 chunks)  | Dressing categories: mechanism, indications, contraindications                  | Sub-query B (dressing selection rationale)     |
| EWMA (12 chunks) | T.I.M.E. theory, wound bed preparation, DFU/VLU wound bed principles            | Sub-query B (chronic wound complexity)         |
| ISTAP (3 chunks) | Skin tear classification and management                                         | Sub-query A/B (Cat B skin tear cases)          |
| ANZBA (4 chunks) | Minor burn management, depth assessment, referral criteria                      | Sub-query A/B (burn cases)                     |
| RCH (11 chunks)  | Paediatric wound care                                                           | Sub-query C (age filter — after metadata fix) |
| AJGP (19 chunks) | General wound dressing principles for family physicians                         | Sub-query B (general principles)               |

**What the current KB CANNOT produce (critical gaps):**

| Missing capability                             | What is missing from KB                                                                                              | Impact on output quality                                                                     |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Application instructions**             | Step-by-step how to apply each dressing: thickness, method, coverage area, secondary dressing choice                 | Patient cannot apply dressing correctly without this                                         |
| **Dressing change frequency (specific)** | Current chunks say "varies" or give ranges; no wound-type-specific + exudate-level + dressing-type-matched frequency | Patient doesn't know when to change                                                          |
| **Malaysian product names**              | No chunks naming Flaminal, Zorflex, Drawtex, Dermacyn, RenoFoam, RenoCare — only generic category names             | Recommendation says "Hydrogel" not "Flaminal Hydro or Dermacyn Hydrogel available at clinic" |
| **Monitoring guidance (patient-facing)** | What does normal healing look like day by day? What signs should prompt same-day clinic visit?                       | Self-care patients have no reference point                                                   |
| **Wound cleansing instructions**         | How to clean the wound before applying dressing (Dermacyn spray sequence, irrigation technique)                      | Patient starts at Step 2, skips Step 1                                                       |
| **Periwound skin protection**            | When to use barrier cream, zinc oxide, Cavilon — the pemilihan_material_dressing.JPG shows this but it's not in KB  | Maceration and periwound breakdown not addressed                                             |

**All six missing capabilities are present in the surgeon images (DyaMed Biotech protocol).** The `treatment_recommendation_based_on_the_flow_chart_of_wound_care_algorithm.JPG` shows the complete 4-step sequence per wound type (cleanse → primary dressing → secondary dressing → change frequency), with specific product names and application details. The `products_and_its_recommendation_of_usage.JPG` and `dyamed_biotech_products_and_its_recommendation_of_usage.JPG` provide per-product application instructions.

**Transcribing these images into structured KB chunks is the single most impactful KB improvement possible for FYP2.** It directly fills every critical gap above.

---

### 10.5 What the KB Should Contain (Target State)

For VerdaSense to produce a complete, self-care-ready dressing plan, the KB needs five chunk types. Current status per type:

**Chunk Type 1 — Algorithm Chunks (WT1-8 decision anchors)**
*Purpose: Sub-query A target. Always retrieved. Gives dressing category list.*
*Current status: ✅ Complete — 13 GP chunks, one per wound type + decision tree + referral criteria*
*Action: No change needed. These are well-curated.*

**Chunk Type 2 — Dressing Category Description Chunks**
*Purpose: Sub-query B target. Explains mechanism, indications, contraindications per dressing type.*
*Current status: ✅ Good — SFP 36 chunks cover most dressing types with mechanism and contraindications*
*Action: Verify SFP coverage for all dressing types in the WT1-8 list. Ensure chunks for: film, hydrocolloid, alginate, hydrofibre, silver, hydrogel, foam, charcoal, iodine, honey, NPWT, polymeric membrane.*

**Chunk Type 3 — Application Protocol Chunks (CRITICAL GAP)**
*Purpose: Sub-query B target. Gives step-by-step application instructions, change frequency, product names.*
*Current status: ❌ Missing entirely from current KB*
*Source: DyaMed Biotech surgeon images (treatment_recommendation_based_on_the_flow_chart_of_wound_care_algorithm.JPG + products_and_its_recommendation_of_usage.JPG + dyamed_biotech_products_and_its_recommendation_of_usage.JPG)*
*Action: Transcribe and ingest as new source. Target ~20–25 chunks. This is Phase 0 Week 2 work.*

**Chunk Type 4 — Contraindication / Safety Chunks**
*Purpose: Sub-query C target. Retrieved when patient notes contain risk factors.*
*Current status: ⚠️ Partial — SFP has some contraindication information embedded in dressing category chunks, but not as standalone retrievable chunks for specific patient conditions*
*Action: Review SFP chunks to confirm these patient conditions have retrievable contraindication text: (1) iodine allergy/thyroid disease, (2) silver allergy, (3) pregnancy/breastfeeding, (4) alginate on dry wound, (5) film on infected/high-exudate wound, (6) honey and bee allergy, (7) silver on clean granulating wound. If any are missing, add targeted chunks from WCM/SFP.*

**Chunk Type 5 — Patient Self-Care Guidance Chunks (MISSING)**
*Purpose: Sub-query C target. Answers practical patient questions from free-text notes.*
*Current status: ❌ Missing — all current sources are clinician-facing, not patient-facing*
*Examples needed: "can I shower with the dressing on?", "how do I remove the dressing without pain?", "what does normal healing smell like vs infection smell?", "when should I go to A&E today (not the clinic tomorrow)?"*
*Source: NHS wound care patient information leaflets (public domain), MOH patient education materials*
*Action: Curate 10–15 patient FAQ chunks as a new KB source. Low effort, high impact for self-care patients.*

---

### 10.6 Why the FYP Title Is Correct — and How to Defend It

**The title:** *"Evidence-Grounded Clinical Decision Support for Wound Dressing Recommendation: A RAG Framework Based on the T.I.M.E. Assessment Framework"*

Every word is defensible:

- **"Evidence-Grounded"** — Every clinical claim in the output is traceable to a specific KB source and chunk. Zero-shot GPT is not evidence-grounded; a rule table lookup is not evidence-grounded (rules have no citations). RAG with a curated clinical KB IS evidence-grounded. This is the core differentiator.
- **"Clinical Decision Support"** — The system does not make the clinical decision. It provides structured, evidence-backed information to support the decision. A patient uses VerdaSense as a first step before clinician review — especially for referred cases (WT6/7/8). This framing is appropriate and clinically honest.
- **"Wound Dressing Recommendation"** — This is precisely the output. Not wound diagnosis, not treatment prescription, not surgical planning — just dressing recommendation based on T.I.M.E. assessment. The scope is correctly narrow.
- **"RAG Framework"** — RAG is the method. It is appropriate because the output requires (a) retrieval of patient-specific evidence from a clinical KB and (b) generation of a synthesised, coherent recommendation. Neither a rule table nor a lookup function can produce the output format required.
- **"T.I.M.E. Assessment Framework"** — This is the clinical input structure. It grounds the system in established clinical practice rather than ad hoc AI.

**The viva answer to "why not just rule-based?" with empirical evidence:**

> "The wound type classification is rule-based — `classify_wound()` implements the MOH T.I.M.E. algorithm deterministically, producing wound types 1–8 with referral and antibiotic flags. This part does not need RAG and should not use RAG. RAG operates at the evidence layer: it retrieves why a dressing is indicated (clinical rationale), how to apply it (application protocol), what contraindications apply for this specific patient (from their free-text notes), and what to watch for (monitoring guidance). These four outputs cannot be encoded in a rule table. Our ablation proves this: zero-shot generation without KB retrieval achieved FA = 0.69, while KB-grounded generation achieved FA = 0.81 — a 12 percentage point improvement in faithfulness to clinical guidelines. The rule table tells a patient 'use silver dressing'. VerdaSense tells them why silver is appropriate for their infection status, how to apply Zorflex or Aquacel Ag to their wound bed, when to change it, and when the wound is deteriorating enough to go to A&E today — all traceable to the specific Malaysian MOH guideline or DyaMed clinical protocol that says so."

---

### 10.7 Immediate Action: KB Gap Closure (Highest Priority for FYP2)

The three additions that will produce the largest improvement in output completeness, in priority order:

**Priority 1 — DyaMed Biotech Protocol Chunks (surgeon_images transcription)**
Fills: application instructions, change frequency specifics, product names
Effort: 1–2 days of careful transcription
Impact: directly enables complete dressing plans with product names and application steps
Source chunks to produce: ~20–25 chunks
Metadata: `authority: "DyaMed_Biotech_Clinical_Protocol"`, `guideline_type: "local_clinical_protocol"`, `source_origin: "KK_Sultan_Ismail"`

**Priority 2 — Patient Self-Care FAQ Chunks**
Fills: practical patient questions from free-text notes
Effort: 1 day of curation from NHS/MOH patient education materials
Impact: Sub-query C can now retrieve answers to "can I shower?", "how do I remove it?", "what does infection smell like?"
Source chunks to produce: ~12–15 chunks
Metadata: `guideline_type: "patient_education"`, `audience: "patient"`

Once these two sources are added, the KB will have all five chunk types covered and VerdaSense will be capable of producing a genuinely complete, self-care-ready dressing plan for the first time.

---

## Part 11 — Multimodal RAG: CNN Embeddings, Image KB, and What VerdaSense Actually Needs

*Addresses two questions raised June 2026: (1) whether CNN feature vectors from T-SegNet/IME-Net can query ChromaDB, and (2) whether the KB needs image-based embeddings for multimodal RAG.*

---

### 11.1 Question 1: Can CNN Feature Embeddings Query ChromaDB?

**The suggestion (from ChatGPT):** Extract embedding vectors from a specific layer of T-SegNet, IME-Net, or a fine-tuned ResNet/EfficientNet, and use that vector directly as the query input to ChromaDB retrieval instead of a BGE-encoded text query.

**Your instinct is correct — this is not feasible. Here is the precise technical reason.**

ChromaDB stores each KB chunk as a **BGE-large-en-v1.5 text embedding** — a 1024-dimensional vector in the semantic space of a text encoder trained on English clinical language with contrastive text similarity loss. Retrieval computes cosine similarity between the query vector and these stored vectors to rank chunks by relevance.

For this to work with a CNN image feature vector, the query and the stored embeddings must live in the **same vector space** — same dimensionality, same training objective, same semantic correspondence. A CNN trained for image segmentation (T-SegNet) or binary classification (IME-Net) produces vectors in a completely incompatible space:

| Property                 | BGE-large-en-v1.5 (KB embeddings)                      | T-SegNet / IME-Net feature vector                           |
| ------------------------ | ------------------------------------------------------ | ----------------------------------------------------------- |
| Input modality           | Text (English sentences)                               | Image (pixel arrays)                                        |
| Training objective       | Contrastive text similarity                            | Segmentation / classification loss                          |
| Semantic content encoded | "How similar is this text to clinical guideline text?" | "Which pixels are granulation?" / "Is this wound infected?" |
| Shared space with BGE?   | ✅ Same space                                          | ❌ Incompatible space                                       |

A cosine similarity between a T-SegNet feature vector and a BGE text embedding **produces a number, but that number has no clinical meaning**. The retrieval results would be essentially random from a clinical relevance perspective.

**The deeper problem with T-SegNet and IME-Net specifically:**

Even if you had a shared multimodal embedding space (e.g. CLIP, which aligns images and text in one space), T-SegNet and IME-Net features would still be wrong:

- **T-SegNet** is a segmentation model. Its intermediate features encode *spatial pixel membership* — which region belongs to granulation vs slough vs necrotic tissue. This geometric segmentation information has no natural correspondence to the semantic content of clinical guideline text about dressing protocols.
- **IME-Net** is a classification model. Its features encode *discriminative visual cues for binary classification* — the visual patterns that separate infected from non-infected. These features are optimised to answer a yes/no classification question, not to semantically match against text about wound care management.

The CNN models' feature vectors encode *what those models need to perform their specific classification tasks*. That is fundamentally different from *what clinical guideline text says about wound management*. There is no training signal that would align these two spaces.

**The correct way to use T-SegNet and IME-Net — which VerdaSense already does correctly:**

```
T-SegNet output → pixel mask → K-Means → tissue percentages
                                               ↓
                              convert to text: "80% granulation, 12% slough, 8% necrotic"

IME-Net output → class probabilities → labels: "Locally Infected | High | Non-Advancing"
                                               ↓
Combined: T.I.M.E. structured text payload → BGE-large encodes as text → ChromaDB query
```

The CNN outputs (labels and percentages) are already correctly used as text inputs to the RAG pipeline. There is no reason to change this design. The T.I.M.E. structured text is the bridge between the image modality and the text KB — and it works.

---

### 11.2 Question 2: Does Multimodal RAG Require Image Embeddings in the KB?

**No. Two different definitions of "multimodal RAG" are being conflated. VerdaSense uses the correct one for this problem.**

Reading all 8 ingestion notebooks confirms that every KB chunk is pure text. The ingestion code explicitly skips image blocks from PDFs (`if b[6] != 0: continue  # skip image/non-text blocks`). Diagrams and tables were manually reconstructed as structured text. The tissue illustration page (GP page 18) kept the text descriptions; the photographs themselves were skipped. **This is correct and should not change.**

**The two paradigms of multimodal RAG:**

**Paradigm A — Retrieval-side multimodal RAG (image queries + image-text KB)**

```
Patient wound image → CLIP image encoder → image query vector
                                                  ↓
KB contains: [wound image + clinical report] pairs, all embedded with CLIP
                                                  ↓
Retrieve most visually similar case reports from KB
```

This paradigm is used in radiology report retrieval (retrieve similar chest X-ray reports given a new X-ray). It requires: a KB of image-report pairs, a shared multimodal encoder like CLIP or BioViL-T, and re-embedding the entire KB in that new space.

**VerdaSense cannot and should not use this paradigm because:**

1. The KB is clinical guidelines (WCM, GP, SFP, EWMA, ANZBA...) — text documents, not image-report pairs. Clinical guidelines have no associated wound photographs.
2. Building a wound image + outcome case library requires patient data, de-identification, and clinical annotation — out of FYP2 scope.
3. R5 already proved empirically that even TEXT captions of wound images (the closest possible bridge) hurt retrieval performance (-6.6pp CR, -18.75pp HR@6). Raw image embeddings for retrieval would perform worse, not better.

**Paradigm B — Generation-side multimodal RAG (text retrieval + multimodal generation)**

```
Patient wound image → VLM captioner (GPT-4o-Vision / Gemini Flash) → clinical caption (text)
Patient T.I.M.E. text payload → BGE-large → text query → ChromaDB → top-6 text chunks
                                                                           ↓
Generation LLM receives: retrieved KB chunks + T.I.M.E. payload + VLM caption
                                                                           ↓
Multimodal output: evidence-grounded dressing plan enriched with visual context
```

The KB is entirely text. Retrieval is entirely text-based. "Multimodal" refers to the GENERATION stage — the LLM receives both retrieved text evidence and visual context from the VLM caption. The image modality enters the system once, via the VLM captioner, and remains as text from that point forward.

**VerdaSense FYP2 implements Paradigm B. The KB does not need image embeddings. This is the right design.**

---

### 11.3 Why Paradigm B Is Architecturally Correct, Not a Compromise

**From R5 (your own FYP1 ablation):**
Even the most favourable form of image-to-text bridging at the retrieval layer — structured clinical captions generated by a VLM — hurt retrieval by -6.6pp CR and -18.75pp HR@6. This is direct empirical evidence that visual information does not improve text-based retrieval for this KB. Raw image vectors would be worse still. R5 settles this question with your own experimental data.

**From the KB content itself:**
Clinical guideline chunks answer: "given a wound with these T.I.M.E. characteristics, what dressing is indicated and why?" This is a text-to-text semantic matching problem. BGE-large is trained exactly for this. The KB content does not benefit from visual queries because the clinical guidelines do not pair their text with wound photographs — the text is self-contained clinical knowledge.

**From the contribution structure:**
Paradigm B keeps the RAG contribution and the multimodal contribution independently ablatable:

| Ablation             | Comparison                    | What it proves                                         |
| -------------------- | ----------------------------- | ------------------------------------------------------ |
| G1 (FYP1, done)      | No KB vs KB-grounded RAG      | RAG adds FA +12pp — RAG is justified                  |
| G4-A (FYP2, planned) | RAG only vs RAG + VLM caption | VLM caption adds X pp to FA — multimodal is justified |

The two contributions are cleanly separated. If VerdaSense had used CNN embeddings to query ChromaDB, the retrieval mechanism and the visual input would be conflated — the examiner could ask "is it the image or the retrieval driving the result?" and you could not answer. The current design avoids this.

---

### 11.4 How to Frame "Multimodal RAG" in the FYP Write-up

The term "multimodal RAG" in the literature covers both paradigms. Be explicit about which one VerdaSense implements and why it is appropriate:

> "VerdaSense implements generation-side multimodal RAG, where visual information from the patient's wound photograph is introduced at the generation stage via a VLM captioner, while retrieval remains text-only using BGE-large-en-v1.5 embeddings against a text clinical guideline KB. This design is motivated by two findings: (1) R5 ablation demonstrated that visual caption injection at the retrieval layer degraded performance (CR -6.6pp, HR@6 -18.75pp), consistent with the semantic gap between wound photograph descriptions and clinical guideline text; (2) the clinical KB encodes protocol knowledge in natural language, not image-report pairs, making text-to-text retrieval the appropriate paradigm. The VLM caption enriches the generation stage with visual observations — wound depth, periwound skin condition, urgency flags — that the upstream T.I.M.E. labels alone do not capture."

This framing is honest, cites your own ablation evidence, and pre-empts the examiners' most likely challenge.

---

### 11.5 Summary: Two Questions, Definitive Answers

| Question                                                            | Answer                                                                                                                                                                                                                                                        |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Can CNN feature vectors (T-SegNet, IME-Net, ResNet) query ChromaDB? | **No.** No shared embedding space between CNN image features and BGE-large text embeddings. Cosine similarity between incompatible spaces produces meaningless retrieval. CNN features encode classification signals, not clinical guideline semantics. |
| Does multimodal RAG require image embeddings in the KB?             | **No.** VerdaSense uses generation-side multimodal RAG (Paradigm B). The KB remains text-only. R5 empirically proved that visual information in the retrieval layer hurts performance. The VLM caption enriches GENERATION, not retrieval.              |
| Is the text-only KB design correct?                                 | **Yes.** Confirmed by reading all 8 ingestion notebooks. All chunks are extracted from PDF text layers; image blocks are explicitly discarded. Clinical knowledge in guidelines is textual, not visual.                                                 |
| Does Paradigm B weaken the multimodal contribution claim?           | **No.** G4 ablation isolates the VLM caption's contribution independently of the RAG contribution (already proved by G1). Two independently ablated contributions is academically stronger than a conflated design.                                     |

---

## Part 12 — Referral, Antibiotic & Debridement Logic (Grounded in MOH Sources)

*This part corrects an earlier assumption in the Phase 0 plan ("all locally infected wounds → referral_required = True") and settles how `classify_wound()` and the generation prompt should handle referral, antibiotics, and debridement. It is grounded in the Garis Panduan (GP, 2019) and Wound Care Manual (WCM) wound-type treatment tables, plus the Stakeholder Wound Cases Review panel comments.*

### 12.1 The Question and Ms Saw's Answer

> **Asked:** "Are all locally infected wounds (currently WT3 & WT4 = no referral) required to be referred?"
> **Ms Saw:** "That's why need photo 😅 — if there is a sign of infection, may need extra management like antibiotic or surgery."

Read precisely, this does **not** say "infected ⇒ refer." It says the *severity* of an infected wound — localised vs spreading (cellulitis/sepsis), superficial vs involving deep structures — **cannot be decided from the T.I.M.E. labels alone; it needs the image.** That is an argument *for* the multimodal layer, not for hardcoding referral.

### 12.2 What the MOH Sources Actually Say

The GP Garis Panduan WT1–8 treatment table (and the WCM table it derives from) specifies **antibiotic** and **surgical/debridement** per type, but the treatment table itself has **no referral column**. Referral is defined separately, in the GP's referral-criteria section, framed as cases needing *extensive care (surgical debridement, NPWT)* or with *systemic complications (sepsis, severe cellulitis)*.

| WT | T.I.M.E.                     | Antibiotic (GP)                   | Surgical/Debridement (GP)                                         | Referral (GP) |
| -- | ---------------------------- | --------------------------------- | ----------------------------------------------------------------- | ------------- |
| 1  | <25%, no inf, dry            | No                                | Secondary closure; dressing till healed                           | No            |
| 2  | <25%, no inf, wet            | May or may not (underlying cause) | Find & treat underlying cause                                     | No            |
| 3  | <25%,**infected**, dry | **Yes (C&S)**               | Debridement**may** be needed                                | **No**  |
| 4  | <25%,**infected**, wet | **Yes (C&S)**               | Debridement**may** be needed                                | **No**  |
| 5  | >25%, no inf, dry            | No                                | Debridement**is** needed                                    | No            |
| 6  | >25%, no inf, wet            | May or may not                    | **Surgical/mechanical** debridement recommended; may repeat | **Yes** |
| 7  | >25%,**infected**, dry | **Yes (C&S)**               | **Surgical** debridement strongly recommended               | **Yes** |
| 8  | >25%,**infected**, wet | **Yes (C&S)**               | **Surgical** debridement strongly recommended               | **Yes** |

**The key pattern:** referral tracks the need for **surgical debridement** (WT6/7/8, the >25%-necrosis types), *not* local infection. Local infection (WT3/4) mandates **antibiotics (C&S-guided)**, not referral.

### 12.3 Honest Verdict: `classify_wound()` Needs NO Change

The current rule engine (`wound_app_unimodal.py`) already encodes exactly this:

```
WT1/2  → referral=False, antibiotic=False
WT3/4  → referral=False, antibiotic=True      (infected, <25% → Abx, no referral)
WT5    → referral=False, antibiotic=False
WT6    → referral=True,  antibiotic=False
WT7/8  → referral=True,  antibiotic=True
```

This matches the MOH algorithm precisely. **Forcing WT3/4 → referral would *deviate* from the guideline and cause systematic over-referral — it would make the system clinically worse, not safer.** The earlier "fix" is cancelled. The notes-driven escalation already in `classify_wound()` (diabetic → referral, `_REFERRAL_TRIGGERS`, subclinical-infection antibiotic triggers) is the correct mechanism for the genuine exceptions.

### 12.4 Where the "Need a Photo" Judgment Lives: the Multimodal Layer

The severity judgments that the rules *cannot* make are exactly what the VLM caption + generation layer should surface as **advisory escalation flags** (never overriding the hard rule, only adding a caution):

- **Spreading infection:** VLM observes perilesional erythema extending >2 cm, streaking, or the patient notes fever/malaise → flag *"signs may indicate spreading infection/cellulitis — seek same-day review."* (This is the WT3/4 "might need referral" case Ms Saw meant.)
- **Deep structures / arterial compromise (esp. WT5):** Panel 1 flagged WT5 as *"need expert opinion, assess vascular supply — sometimes arterial compromise."* VLM observes exposed tendon/bone, dusky/black tissue, or a foot/lower-limb location → flag vascular assessment.
- **This is the strongest single argument for FYP2's multimodal thesis:** the referral *rule* is deterministic and guideline-bound; the referral *judgment* for borderline infected wounds is image-dependent and is precisely the value the VLM adds.

### 12.5 Debridement Pathway (for the G1-E Prompt) — and Debridement ≠ Referral

Debridement is a spectrum. The safety-critical distinction for a self-care tool:

- **Conservative / autolytic debridement** (hydrogel, Dermacyn soaking, gentle bedside deslough of small slough) — **within patient/primary-care scope.**
- **Surgical / sharp / OT debridement** — **requires referral.**

Per-wound-type pathway, reconciling the GP WT1–8 table + DyaMed surgeon-image protocol + the 3-panel Stakeholder Cases Review:

| WT | Debridement pathway (prompt guidance)                                                                       | Scope                                    |
| -- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| 3  | Debride slough at edges; autolytic via hydrogel; if not progressing in 2–3 days → hydrofibre + antibiotic | Conservative — self/clinic              |
| 4  | Daily dressing (high exudate) + bedside deslough; debride + antibiotic                                      | Conservative — clinic                   |
| 5  | Debride + daily hydrogel;**assess vascular supply / expert opinion** (arterial compromise risk)       | Conservative + escalation flag           |
| 6  | Debride + daily/bd Dermacyn 1–2 days until debridement no longer needed → switch to Ag alginate; deslough | **Referral** (surgical/mechanical) |
| 7  | Debride + daily/bd Dermacyn until clean → Ag alginate + antibiotic;**may need OT debridement**       | **Referral** (surgical)            |
| 8  | Debride + daily/bd Dermacyn until clean → Ag alginate + antibiotic;**may need OT debridement**       | **Referral** (surgical)            |

**Prompt rule for G1-E:** for WT6/7/8, the recommendation must direct the patient to clinician/hospital review for debridement rather than instruct them to debride; for WT3/4/5, conservative autolytic debridement guidance is in-scope, paired with the appropriate antibiotic/escalation note. The DyaMed protocol chunks (Dermacyn soaking = autolytic + mechanical debridement aid) are the grounded KB source for this guidance — i.e. Ms Saw's hospital wound-team materials.

### 12.6 On Ms Saw's Engagement (H1 logistics)

Ms Saw did not complete `VerdaSense_G4_Clinical_Review_Form.docx`; instead she sent the 13 clinical noticeboard photos (now `surgeon_images/`) with: *"Something used in my hospital wound team… ask AI to read and analyse it… please include in your study… I think this answers your question."* Interpretation: her substantive expert contribution **is** the hospital wound-team protocol (the DyaMed/KKSI material, now ingested as the 9th KB source). For H1, treat those images as her primary expert input and keep the formal evaluation lightweight and in-person/short rather than relying on a returned form.

---

## Part 13 — Recommendation Output Design (patient-facing + evidence-grounded)

*The generation-stage output contract for FYP2. Resolves the tension between Ms Saw's "no evidence block in the patient app" and the FYP's need to demonstrate evidence-grounding.*

### 13.1 Core principle — one generation, two render modes

Do **not** choose between "show sources (FYP)" and "hide sources (patient)". The LLM **always** generates the full structured answer **with inline `[Source N]` citations**; the application decides what to *render*:

```
                         ┌── Clinician / FYP / Evaluation view  → shows Evidence block + citations
LLM generation (cited) ──┤        (viva demo, RAGAS Faithfulness/Context metrics, Ms Saw H1)
                         └── Patient app view                   → hides citations, friendly sections only
```

**Why this is correct, not a compromise:** RAGAS **Faithfulness / Context-Precision are only measurable if the citations exist in the generated answer**. Stripping them at generation time would destroy the metric that proves the RAG contribution. So citations live in the generation; the patient view simply does not paint them. This satisfies Ms Saw (patient simplicity) *and* the FYP (evidence-grounding) from a single model call — and leaves the door open for an optional future "why this dressing?" tap-to-expand without re-architecting.

### 13.2 Patient-view section schema

Maps cleanly onto the two-layer architecture (Part 1/10): **rule-layer = the clinically critical, deterministic fields; RAG-layer = the why/how/products**.

| #  | Section                                                     | Source layer                | Notes                                                                                                                         |
| -- | ----------------------------------------------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| 1  | **Your wound, in plain words**                        | rules (WT + T.I.M.E.)       | e.g. "Mostly clean and healing, moderate fluid, no signs of infection." Define jargon inline:*"slough (soft dead tissue)"*. |
| 2  | **Dressing type you need** (primary + secondary)      | **rules**             | Generic category FIRST — deterministic and safe.                                                                             |
| 3  | **Example products** (tappable → gallery / purchase) | **RAG (DyaMed only)** | Lowest priority. Brand names quoted from evidence only (see 13.3).                                                            |
| 4  | **Dressings to avoid**                                | RAG (contraindications)     | Safety.                                                                                                                       |
| 5  | **How often to change** (per dressing)                | RAG                         | Differs per dressing — state each.                                                                                           |
| 6  | **Does it need antibiotics?**                         | rules                       | Phrase as*"may need antibiotics — see a clinician"* (C&S-guided);**never** "take antibiotics".                         |
| 7  | **Do you need to see a doctor?** (referral)           | rules + VLM flag            | WT6/7/8 → yes. Spreading-infection VLM flag (Part 12.4) surfaces here.                                                       |
| 8  | **Step-by-step care guide**                           | RAG (DyaMed protocols)      | Friendly, numbered, mobile-short.                                                                                             |
| 9  | **⚠️ Red flags — get help now**                    | rules + VLM                 | Highest-stakes.**Do not soften** the tone here even though the rest is gentle.                                          |
| — | **Evidence**                                          | RAG                         | Clinician/eval view**only**.                                                                                            |

### 13.3 The two hallucination guardrails (clinical-safety critical)

1. **Type before product; type from rules.** The clinically critical decision — the dressing *category* — is rule-derived and deterministic (`classify_wound()` + algorithm). Products are illustrative. So even if the model names a wrong brand, the *category* (what matters) is correct. This is the single most important safety property of the design.
2. **Products are quoted, never invented.** Prompt rule: *"Product names must be quoted verbatim from the provided evidence. If no product evidence is retrieved, give the dressing type only and omit examples."* This is exactly why the **`dressing_class` metadata bridge (Part 14)** matters — it makes the correct product chunk *retrievable* so the model never has to guess a brand.

### 13.4 Tone & length

Gemini-style: warm, plain-language, professional, jargon defined inline. **Enforce brevity in the prompt for mobile** (e.g. each section ≤ 2 short sentences; total ≤ ~250 words; collapsible sections). Long outputs hurt both UX and RAGAS Answer-Relevance.

**Purchase / pharmacy links** (Section 3, lowest priority): gate behind the referral/red-flag logic and attach a disclaimer — a patient buying a product off a possibly-wrong CV self-assessment is a real risk surface.

### 13.5 Testset v5 implications + new ablation

Rewriting `wound_testset_v3.json` → **v5** to the new schema/tone is correct, with four deliberate points:

1. **It rebases the generation metrics** — v5 reference answers are a different target than the FYP1 9-section references. FYP2-G scores are **not** directly comparable to FYP1-G. State this explicitly.
2. **Keep `[Source N]` citations in the reference answers** (even though the patient view hides them) so Faithfulness / Context-Precision stay measurable.
3. **Add DyaMed chunks to `reference_contexts`** for cases that should surface a type/product/step — otherwise Context-Recall can't credit the new KB. (This is the testset half of Part 14: the class bridge makes the chunk retrievable; the testset marks it as expected context.)
4. **New ablation G1-F — patient-friendly schema:** old structured 9-section output vs the new patient-friendly dual-mode schema, scored on Faithfulness + a conciseness/readability measure + Ms Saw's H1 ratings. Add to the Part 5 ablation map.

---

## Part 14 — KB Upgrade: Dressing-Class Bridge (brand → generic category)

*Implemented in `ingestion_DYAMED_surgeon_images.ipynb` + propagated by `ingestion_v5_BGE_MedEmbed.ipynb`. Directly enables the Part 13.3 product guardrail.*

**Problem.** The MOH/GP algorithm speaks in **generic categories** ("use hydrocolloid / foam / alginate"); the DyaMed chunks speak in **brand names**. The embedding model cannot bridge "RenoCare Thin" ↔ "hydrocolloid" or "Flaminal Forte" ↔ "alginogel" on its own, so (a) Sub-query B retrieval misses the right product chunk and (b) the LLM may miscategorise a product → clinical-risk hallucination.

**Fix.** Each DyaMed product monograph now carries an explicit, clearly-delineated **generic dressing class** — both as embedded text (`Dressing class (generic category): …` + `MOH/GP category bridge: …`) and as filterable metadata (`dressing_class`, `moh_category`). WT-protocol dressing lists and the T.I.M.E.-map carry a short parenthetical / legend too. The classes are an **editorial cross-reference verified against manufacturer datasheets** (Flen Health / DyaMed / S&N IFUs) — *not* on Ms Saw's posters — and are kept separate from the verbatim transcription (preserving the no-hallucination integrity of the source).

| Product                          | `dressing_class`                              | `moh_category` (algorithm bridge)      |
| -------------------------------- | ----------------------------------------------- | ---------------------------------------- |
| Dermacyn WoundCare Solution      | Super-oxidised HOCl antimicrobial cleanser/soak | Antimicrobial wound cleanser             |
| Dermacyn WoundCare Hydrogel      | HOCl antimicrobial hydrogel                     | Hydrogel                                 |
| Flaminal Hydro / Forte           | Enzyme alginogel (3.5% / 5.5% alginate)         | Alginate/Alginogel                       |
| Zorflex / Zorflex LA             | 100% activated carbon cloth (LA = low-adherent) | Charcoal/activated carbon                |
| Drawtex                          | Hydroconductive (LevaFiber)                     | High-exudate absorbent (hydroconductive) |
| RenoCare Thin / B / Hydrocolloid | Hydrocolloid (B = foam-backed)                  | Hydrocolloid                             |
| RenoFoam                         | Polyurethane foam                               | Foam                                     |

**Impact:** Sub-query B can now match the algorithm's generic category to the specific Malaysian product, and the generation layer can state "Hydrocolloid — e.g. RenoCare Thin" with the type rule-anchored and the brand evidence-quoted. Re-run v5 ingestion after regenerating `DYAMED_clinical_protocol_kept.json` to load the new metadata.

---

## Part 15 — End-to-End Multimodal RAG Chain-of-Thought (TIME → dressing plan)

*The full sequential pipeline for one recommendation, showing where each input enters, which retrieval axis fires, and which output section / metric it feeds. This is the reasoning spine the generation prompt should follow.*

### 15.1 The pipeline (stages)

```
STAGE 0 · INPUTS
   CV pipeline  → tissue % (K-Means) + IME-Net I/M/E labels   → T.I.M.E. payload
   Wound image  → (held for Stage 2)
   Patient      → free-text notes + demographics (diabetes?) + wound_depth (self-report)

STAGE 1 · RULE LAYER (deterministic — Layer 1)
   normalise() → classify_wound()
   → wound_type (1–8), referral_required, antibiotic_required, etiology(DFU flag), subclinical_infection
   → decides the DRESSING CATEGORY pin + escalation flags  (no RAG here)

STAGE 2 · VLM CAPTION (FYP2 — generation input ONLY, never retrieval; R5)
   image → VLM → caption: TIME cross-validation, depth, periwound, location, urgency flags
   → may RAISE a soft escalation advisory (e.g. spreading erythema) but never overrides the rule

STAGE 3 · MULTI-AXIS RETRIEVAL (Layer 2 evidence — R1-C, Dense, k=6, population-filtered)
   Sub-query A (PIN)  : filter wound_type == classified  → GP algorithm anchor + DyaMed WT protocol     → "WHICH category"
   Sub-query B (MECH) : query from exudate/infection/tissue → dressing-class descriptions (SFP) +
                        DyaMed product monographs (dressing_class bridge) + pemilihan tree              → "WHICH product + WHY + HOW"
   Sub-query C (NOTES): patient free-text → contraindication / patient-specific chunks                  → "patient-specific cautions"
   merge + dedup → top-6 ranked evidence (Source 1 = pinned anchor, binding)

STAGE 4 · ASSEMBLY
   change-frequency per product ← DyaMed monographs + WT-protocol "Change Frequency" step
   contraindications ← Sub-query C hits ∩ allowed/contraindicated rules

STAGE 5 · GENERATION (grounded prompt G1-C → G1-E/G1-F)
   LLM inputs = [retrieved chunks] + [T.I.M.E. payload] + [VLM caption] + [classifier flags]
   → Part 13 structured output, cited. Guardrails: TYPE from rules; PRODUCTS quoted from evidence only.

STAGE 6 · RENDER
   patient view (no citations)  |  clinician/eval view (citations + evidence block)
```

### 15.2 Worked reasoning trace (one case)

> **Case:** tissue 12% slough, **infected**, **high exudate**, non-advancing; notes "allergic to silver".
>
> 1. **Rule:** <25% slough + infected + wet → **Wound Type 4**; antibiotic_required=True; referral=False.
> 2. **VLM:** confirms purulent + 1 cm perilesional erythema (agrees with IME-Net); depth superficial; periwound macerated. No spreading → no referral escalation.
> 3. **Sub-query A:** pins WT4 algorithm + DyaMed WT4 protocol → category = **alginate/alginogel + foam + (silver) + antimicrobial absorbent**.
> 4. **Sub-query B** (infected, high exudate): retrieves alginogel + silver descriptions, **Flaminal Forte (alginogel)** + **Drawtex (hydroconductive)** monographs → WHY = antimicrobial + high absorbency.
> 5. **Sub-query C** ("silver allergy"): contraindication chunk → **avoid silver dressing**.
> 6. **Assembly:** change frequency — Flaminal Forte EOD→up to 4 days; Drawtex 3–4 days (from monographs).
> 7. **Output:** Primary = alginogel (e.g. Flaminal Forte); Secondary = absorbent (e.g. Drawtex/foam); **avoid silver**; antibiotic — see clinician (C&S); change EOD; red-flag watch for spreading infection.

### 15.3 Axis → output → metric map

| Pipeline element                 | Feeds output section (Part 13)         | Measured by                                  |
| -------------------------------- | -------------------------------------- | -------------------------------------------- |
| Stage 1 rule (wound_type, flags) | §2 type, §6 antibiotic, §7 referral | referral/antibiotic correctness, Safety      |
| Sub-query A pin                  | §2 dressing type                      | Hit-Rate@k, MRR (rank-1 anchor), NDCG        |
| Sub-query B (mech/product)       | §2 products, §3 examples, §8 steps  | CR, CP, NDCG; Faithfulness                   |
| Sub-query C (notes)              | §4 avoid, §7 escalation              | CR on contraindication chunks; Safety        |
| DyaMed monograph change-freq     | §5 change frequency                   | Faithfulness                                 |
| VLM caption                      | §1 plain summary, §7/§9 flags       | G4-A (FA Δ), VLM-discrepancy detection rate |

---

## Part 16 — KB v5 Cleanup / Source-Pruning Decision (honest analysis)

*Question raised: should low-value sources/chunks (AJGP, ANZBA, EWMA, ISTAP, SFP advanced-therapy, RCH, WCM surgical) be removed from v5 as retrieval noise?*

### 16.1 The principle — distinguish dead weight from active noise

- **Dead weight** = chunks that are (almost) never retrieved → harmless to retrieval quality; cost is only index size. **No urgent reason to delete.**
- **Active noise** = chunks retrieved *into* top-k *instead of* a better chunk, displacing it → genuinely lowers CR/precision. **This is the only thing worth pruning.**

You cannot tell which is which by intuition — **measure it.** Run retrieval over the testset and flag chunks that appear in top-k but are in **no** case's `reference_contexts`. A source whose chunks are frequently retrieved-but-never-relevant is a pruning candidate; one that's simply never retrieved is harmless.

### 16.2 Honest source-by-source take

| Source                                                       | Role                                                                                  | Verdict                                                                                                                                                                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **RCH** (paediatric)                                   | paediatric coverage                                                                   | The adult**population filter already neutralises it** — not noise for adult cases. Keep (cheap); only drop if you commit to adult-only scope.                                       |
| **WCM surgical** (surgical debridement, NPWT)          | not self-care,**but** grounds the *referral/escalation rationale* for WT6/7/8 | **Do not delete** — it justifies "needs surgical debridement → refer". Better: tag `scope="specialist"` so generation cites it only for escalation, not as patient instructions. |
| **AJGP, EWMA, ISTAP, ANZBA**                           | etiology/coverage (skin tear, burns, wound-bed-prep)                                  | These are the**gold contexts for Cat B** cases. Small (3–12 chunks each), low noise risk, authoritative. Keep unless measured as noise.                                             |
| **SFP advanced-therapy** (NPWT, HBOT, low-power laser) | out of self-care scope, niche                                                         | **Strongest pruning candidate** — most likely to be spuriously retrieved and never patient-relevant. Tag `scope="out_of_scope"` or remove after measuring.                        |

### 16.3 Recommendation — don't blind-prune; make it a measured ablation

1. **Keep v5-full as the baseline** (already built). Don't delete sources ad hoc — it changes chunk-ids/counts and invalidates the store.
2. **Add a `scope` metadata tag** (`self_care` / `specialist` / `out_of_scope`) — non-destructive; lets you filter at retrieval time and ablate cleanly.
3. **Measure noise** on the testset (retrieved-but-never-relevant analysis).
4. **Run a KB-pruning ablation (R-KB):** v5-full vs v5-pruned (drop/`out_of_scope`-filter the measured-noisy chunks) on CR / CP / HR@k. Prune only if it improves precision **without** hurting Cat B recall.
5. **Lean conservative:** with ~160 chunks and k=6, noise is probably modest; the bigger risk is losing Cat B (etiology) recall and the referral rationale. Prune only what is *both* measured-noisy *and* clearly out-of-scope (e.g. HBOT/laser).

> Net: "should I clean up the KB?" becomes a **defensible measured contribution (R-KB ablation)** rather than a guess — and it gives you another result to report at viva.

---

## Part 17 — Multimodal Prototype Build, Generation Guardrails & KB Source-Conflict Findings (June 2026)

*This part records what was actually BUILT and what testing the prototype revealed. It supersedes the "planned" framing of Parts 2/4/13 for the items below — those are now implemented and demo-ready.*

### 17.1 The working multimodal prototype (status: BUILT)

Two new files sit beside the FYP1 unimodal pair (which is untouched):

- **`wound_app_multimodal.py`** — FastAPI app, port 8001.
- **`templates/wound_index_multimodal.html`** — extended UI.

What it implements end-to-end (all verified working):

| FYP2 element | Implementation | Notes |
|---|---|---|
| KB v4 → **v5 BGE** | `db_wound_care_v5_bge` / `wound_care_v5_bge` (160 chunks, 9 sources) | R4-B winner; **v5 is the active FYP2 store** (v4_bge was FYP1) |
| **VLM caption** (generation-stage only) | `generate_vlm_caption()` → structured JSON; passed to generation, **never to retrieval** (R5/Paradigm B confirmed) | GPT-4o-mini-V / Gemini-2.5-Flash-V selectable |
| **Etiology via VLM** | zero-shot, no CNN trained (per Pain-Point 3) | surfaced in UI; e.g. WT04 image → "venous_leg_ulcer" |
| **Wound depth via VLM** | VLM estimate + patient self-report → `resolve_wound_depth()` | no depth CNN, no labelling project |
| **Patient-friendly output (G1-F)** | new `PATIENT_SYSTEM_PROMPT`, the v5 9-section schema with `[S#]` citations | one generation, two render modes |
| **Dev/Prod dual render** | front-end toggle: Dev = citations + evidence sidebar + VLM internals + analytics; Prod = citations hidden + product gallery | Part 13 contract, realised |
| **Static product gallery** | `build_product_gallery()` (DyaMed catalogue, placeholder tiles) | now **exudate-aware** (see 17.2) |
| **Multimodal On/Off A-B** | front-end switch → live unimodal-vs-multimodal on the same case | demonstrates the G4-A contrast in the demo |
| **Token-by-token streaming** | SSE endpoint `/get_recommendation_stream` (`astream`): `meta` → `delta`×N → `done` | ChatGPT-style; perceived-latency fix (see 17.4) |

The senior CV pipeline (YOLO + MobileSAM + K-Means via the HF segmenter Space) and manual I/M/E + notes input are **unchanged** from FYP1.

### 17.2 Generation guardrails added (refinements to G1-E / G1-F)

Prototype testing surfaced three generation-stage rules now baked into `PATIENT_SYSTEM_PROMPT`. These are **G1-E/G1-F prompt refinements** and should be reported as part of that ablation:

1. **Brand → class inline labelling** — every product names its generic class, e.g. *"Flaminal Forte (alginogel)"*. Activates the Part 14 dressing-class bridge at generation time and matches the v5 reference style.
2. **Contraindication consistency guard** — before finalising, every recommended product's class is cross-checked against the binding algorithm's (Source [S1]) exclusion list. If excluded, it is dropped — **the MOH/GP algorithm overrides a local DyaMed protocol** — and no dressing may appear in both a recommendation and the "Dressings to Avoid" section. (Fixes the Zorflex/charcoal self-contradiction; see 17.3.)
3. **Exudate-tier matching** — high exudate → prefer a high-absorbency primary (alginate / hydrofibre / alginogel e.g. Flaminal Forte) over plain foam (moderate); dry/low → moisture-donating primary; always within S1-allowed classes. The product gallery mirrors this (Flaminal **Hydro ↔ Forte** swap by moisture level).

### 17.3 KB source-conflict findings → route to Ms Saw (H1 Part D — KB reconciliation)

Testing found two places where two *grounded* sources in the v5 KB genuinely disagree. These are **not bugs and not hallucinations** — the system faithfully surfaced a real guideline-vs-local-protocol tension, which is itself evidence the human-in-the-loop validation layer is the correct design. Each is a concrete H1 question for Ms Saw:

| # | Wound type | The conflict | Question for Ms Saw |
|---|---|---|---|
| C1 | **WT1** (clean, dry) | DyaMed/KKSI protocol lists **Zorflex LA (activated carbon)** as a valid WT1 dressing, but the **MOH/GP algorithm excludes charcoal** for WT1 | Is the MOH "charcoal" exclusion meant to cover low-adherent carbon contact layers (Zorflex LA), or only odour-control charcoal dressings for infected wounds? |
| C2 | **WT2, high exudate** | The **WT2 protocol chunk (S2)** offers **Drawtex** as the exudate secondary; the **exudate-selection tree chunk (S4)** says **Gauze & Gamgee** for high exudate | Which secondary should the system prefer for high-exudate non-infected wounds — Drawtex or Gauze & Gamgee? |

Current default behaviour: the binding MOH algorithm wins (C1 → Zorflex dropped for WT1); for C2 the more exudate-specific source (S4) is followed. Both are documented as defaults pending Ms Saw's confirmation. **Do not edit her transcribed protocol chunks to "resolve" these** — preserve source fidelity (Part 14 principle) and reconcile at the generation layer / via her ruling.

### 17.4 Performance & perceived latency

Measured per request: VLM caption ≈ 6–8 s (image dominates input tokens), generation ≈ 5 s, retrieval ≈ 0.3 s. **SSE streaming** masks the generation wait (first token in ~1 s instead of a 5 s blank). The deterministic, rule-derived parts (dressing type, product gallery, referral/antibiotic flags) render instantly in the `meta` event. **Open speed lever (not yet implemented):** downscale the uploaded image (~768 px) before the VLM call to cut its ~7 s by 2–4 s — low-risk, recommended before any clinical pilot.

### 17.5 Implications for the ablation map (Part 5)

- **G1-E / G1-F** now include the three 17.2 guardrails — report them as prompt refinements with before/after consistency examples (the Zorflex case is a clean qualitative illustration).
- **H1** gains a **Part D — KB reconciliation** (the two 17.3 conflicts) alongside the existing blinded-rating, caption-quality, and debridement-completeness parts.
- **G4-A** (caption vs none) is directly demonstrable live via the Multimodal On/Off switch.

---

## Part 18 — FYP2 Progress Log (living status · last updated Jun 2026)

*Consolidated status after the supervisor meeting + the v5 testset image-curation work. New companion docs sit beside this plan; this log points to them.*

### 18.1 Supervisor meeting — scope locked
- Multimodal RAG = correct FYP2 move, **but the deliverable is the EVALUATION** (don't stack development without measurement).
- **Etiology + wound-depth: DEFERRED** (kept in the prototype, excluded from the ablation + testset scoring for now). → drops G4-C, G4-D, R6 from the active map.
- Patient-friendly output ✅ · VLM↔CV cross-validation ✅ · H1 human eval + UAT with Ms Saw ✅.
- **Dev mode is the evaluation mode**; ablation ignores the product gallery (Prod-only UX).

### 18.2 New companion documents
- **`VerdaSense_FYP2_Ablation_Map_v5.md`** — the revised, eval-first multimodal ablation map (3 pillars: VLM contribution + how-to-prompt-VLM, retrieval R5-v5, generation G1-E/F), on v5 KB + v5 testset, etiology/depth excluded. Supersedes Part 5 for FYP2 execution.
- **`VerdaSense_FYP2_Testset_Construction_and_Review_Plan.md`** — the 3-layer construction model + the **one-pass Ms Saw review** redesign (she *validates pre-filled* answers, not authors them; per-case = 5 tick decisions; invariants reviewed once).

### 18.3 Testset v5 — image curation COMPLETE (2026-07-02) · EXPANDED to 34 cases (2026-07-03)
Built by `ragas_testset/wound_testset_builder_v5.py`. **34 cases**: A:8, B:6, C:4, D:3, E:3, F:3, G:7 (expanded from 21 for per-category statistical power — G4-A showed B/C/D/E/F were n=1–2). B/C note-driven (reuse curated images); D/E/F new Gemini-validated images (deep-cavity→NPWT, extreme-necrosis, arterial→no-compression, mixed-tissue→TIME, 2 clean-F). Every new case's live-classifier referral/abx matches gold; all 34 pass end-to-end sanity. See testset-construction §4.1 + memory `testset_v5_curation.md`. *(Original 21-case core details below.)* Builder fixes carried: **WT2 → Flaminal Forte** (high-exudate tier); **`conditional_contraindications`** field (iodine-if-thyroid out of hard contraindications for WT3/4/7/8).

**Full image curation done — all 21 cases three-way validated** (Claude read ↔ gold label ↔ **Gemini-Pro blind read**; user pastes each `wound_images/` image into Gemini with a fixed blind prompt, results reconciled). All images resolve; Cat A classifies cleanly WT1→WT8; live-classifier output matches each intended wound type. 14 distinct images in `ragas_testset/wound_images/` (source `wound_images_dataset/` = Kaggle wound-seg, fusc/medetec/wsnet).

| WT | image | WT | image |
|---|---|---|---|
| 1 | WT01_medetec_0021 | 5 | WT05_medetec_0065 |
| 2 | **wsnet_0494** (was medetec_0116) | 6 | WT06_medetec_0298 |
| 3 | WT03_wsnet_0096 | 7 | WT07_wsnet_0539 |
| 4 | **wsnet_0466** (was wsnet_0816) | 8 | WT08_medetec_0175 |

Special/adversarial images locked: cat_c DFU=**fusc_0902** (infected sloughy plantar), cat_d cavity=**medetec_0373** (true sinus; old medetec_0095 was hypergranulation), cat_e VLU=medetec_0142, cat_f=**medetec_0283** (was a duplicate), cat_g miss_infection=wsnet_0096/medetec_0066, miss_necrosis=medetec_0065/0298, overcall=medetec_0021/wsnet_0494/**medetec_0158** (overcall_fusc renamed **overcall_clean**). Tissue realism: a_wt3 0/20/80, a_wt6 0/65/35. **Next:** end-to-end sanity run → re-run G4-A/G4-P on corrected labels → one-pass Ms Saw review.

### 18.4 Curation findings (documentable — VLM adds value at the data-curation stage)
The per-case VLM-caption suitability check rejected several images for concrete reasons — evidence the multimodal layer earns its place beyond inference:
- **Tissue mismatch:** medetec_0066 (too necrotic for WT3) → rejected.
- **Clinical grounds:** medetec_0064 (dry gangrene toe → wrong management for WT5) → rejected.
- **"Not infected" violation:** medetec_0026 / 0058 (peri-wound erythema vs WT6's no-infection) → rejected.
- **Face validity:** medetec_0047 / 0070 / 0051 (multiple wounds confound YOLO/SAM) → rejected.
- **VLM weak axis confirmed:** necrotic beds repeatedly mis-read as **"cavity"** (WT5, WT7) — supports deferring depth scoring.

**Image-suitability scoring rule (lock into the G4 metrics):** a still photo cannot convey **exudate level** (dry vs wet) or reliably confirm **not-infected** / depth. So score **VLM Caption Accuracy on tissue / infection-signs primarily**; treat **moisture and depth as low-confidence axes** (don't penalise). The selection logic also flipped by type: WT1–4 (low non-viable) want a *granulating bed + peri-wound erythema halo*; WT5–8 (high non-viable) want the *sloughy/necrotic bed*.

### 18.5 Pending Ms Saw — inter-guideline KB conflicts (sent via WhatsApp; supersedes/expands 17.3)
Systematic DyaMed-vs-MOH (and internal-DyaMed) sweep across all 8 WTs found **5 conflicts + 1 scope question**:

| # | Conflict | Default pending her ruling |
|---|---|---|
| **C1** | Carbon (Zorflex/Zorflex LA) — DyaMed WT1–7 vs MOH charcoal only WT8 | follow MOH (carbon = infected/odour only) |
| **C2** | Drawtex (hydroconductive) — not a named MOH category | allow as high-exudate absorbent secondary |
| **C3** | Secondary for high-exudate: Drawtex (WT protocol) vs Gauze&Gamgee (selection tree) | follow the exudate-selection tree |
| **C4** | Alginogel (Flaminal Hydro) on a **dry** wound (WT7) | allow (alginogel ≠ alginate fibre — it donates moisture) |
| **C5** | Foam as secondary on a dry wound (WT3/WT7) | low priority; advisory |
| **Q8** | Product scope: DyaMed-only vs include non-DyaMed brands (Aquacel Ag, Activon honey, Kaltostat, Winner Foam) | speak in classes; DyaMed products as named examples; honey gated by bee allergy |

These become **H1 Part D (KB reconciliation)**. Do **not** edit Ms Saw's transcribed protocol chunks to "resolve" them — reconcile at generation / via her ruling.

### 18.6 Pillar 1 ablations — G4-P + G4-A COMPLETE (2026-07-03)

Both run on the curated 21-case v5 testset (`gpt-4o-mini` VLM + generation, 3 runs, RAGAS judge `gpt-4o-mini`+`text-embedding-3-small`).

**G4-P (VLM prompt strategy) ✅** — added a 4th variant **P4 = blind** (CV labels withheld from the VLM). *Only blind cross-validates:* **P4 caught 100% (21/21)** of adversarial label↔image discrepancies vs **P1/P3 14%, P2 19%**. Label-shown prompts *parrot* the labels — infection-axis anchoring ~98% (caught 1/45); tissue-axis 50%. Blind's nominal infection accuracy is 78.6% (the label-shown 100% is label-echoing, not perception — its 3 disagreements are explainable: over-calls infection on necrotic wt5/slough wt6, under-calls the peri-wound-only wt4). **Blind is free downstream: FA 0.629 ≈ best (P3 0.630) > old-prod P2 0.622; Safety identical 90.5%** — the pilot's "~5 pp FA cost of blind" was a mis-labelled-data artifact, gone on clean labels. **Winner = P4 blind, now live in the app.** Also serves as **VLM-DISC = 100%**. → `MDs/Generation Ablation/G4P_VLM_Prompt_Strategy_Analysis.md`.

**G4-A (caption vs no-caption, under the blind caption) ✅ — re-run on 34 cases (2026-07-04):** Global **ΔFA −0.8 pp (within run SD), ΔAR −3.7 pp, ΔSafety −1.0 pp (one case), cost 3.2×** → blind caption is FA-/safety-neutral. **The expansion was decisive:** the 21-case pilot's per-category "wins" (Cat B **+12 pp**, F **+12 pp**) were **small-n artifacts (n=1–2) that collapsed at n=6** (B −2.4, F −8.3). The robust rows — A (n=8) +1.8, B (n=6) −2.4, G (n=7) −2.0 — agree the caption improves FA **nowhere** → **FA is structurally the wrong lens** (it penalises the caption's out-of-KB visual cross-validation). **NEW directionality finding:** the caption is an **asset** when the *image* reveals danger the labels miss (Cat G missed-infection → escalates), but a **liability** when danger is in the *notes* and the image looks clean (`cat_c_spreading_infection`: clean WT2 photo + notes say pus/spreading → blind caption reads "clean" → pulled the advice to alginate+foam off the mandated antimicrobial in 1/3 runs = the −1 pp safety drop). **Design rule:** the caption must stay **advisory** — never soften a notes/label-driven escalation. (Safety-checker fixed to skip Step-by-Step cautions → `cat_e_arterial` now 3/3; residual negation false-fail on `skin_tear` hits both arms equally.) → `MDs/Generation Ablation/G4A_Multimodal_Caption_Analysis.md`.

**The combined multimodal claim (thesis-ready):** the blind VLM caption catches **100%** of CV-label errors (G4-P) and, pushed into generation, does so at **no faithfulness/safety cost** while surfacing missed danger to the patient (G4-A) = a **low-cost clinical safety-net**. G4-A read alone understates this (flat FA); it *must* be reported with G4-P, and the decisive external validation is **H1 (Ms Saw)** — FA/AR cannot credit cross-validation.

**G4-B (VLM comparison under the fixed blind prompt) ✅ (2026-07-04):** GPT-4o-mini-V (B1) vs Gemini-2.5-Flash-V (B2). **B1 wins on every axis** — 0 refusals, **100% VLM-DISC**, 73% infection / 86% tissue acc, $0.047/run, 3.4 s. **B2 refused 41% (42/102) of clinical images** (empty, content-blocked responses) concentrated on the infected/necrotic/cavity/adversarial wounds (clean Cat F = 0 refusals); its VLM-DISC collapses to 47.6%, and even on accepted images it is worse (67% tissue) + pricier (+36%) + slower (+64%). **The refusal is non-configurable:** a `safety_settings=BLOCK_NONE` test on all four harm categories recovered **0/5** blocked images — Gemini reports `BlockedReason.OTHER` / `safety_ratings=None`, so the standard Developer API cannot disable it. **Verdict: keep `gpt-4o-mini`; Gemini disqualified.** Deployment finding: consumer VLM content filters are a real barrier to clinical wound imaging — reliability on the graphic input distribution is a first-class model-selection criterion. → `MDs/Generation Ablation/G4B_VLM_Comparison_Analysis.md`.

**G4-C (open-source VLM comparison via OpenRouter, blind prompt, reasoning-off like G3) ✅ (2026-07-05):** 4 arms — Qwen2.5-VL-72B, Qwen3-VL-235B, Gemma-3-27B, Gemma-4-26B. **(1) Open models eliminate the refusal problem:** all ~0% errors (transient only) vs Gemini's 41% content blocks — a self-hostable open VLM structurally avoids the vendor content-filter. **(2) Methodological finding — VLM-DISC is gameable:** Gemma-3-27B scores 100% VLM-DISC but earns it by over-calling "Infected" on **95% (40/42) of clean wounds** (infection acc 49%, worse than chance) → DISC must always be read *with* non-adversarial accuracy + over-call rate; GPT-4o-mini's 100% is genuine (backed by 73% acc), Gemma-3's is a manufactured artifact. **(3) Best open = Qwen2.5-VL-72B:** infection **76%** (> GPT-4o-mini's 73%), tissue 85%≈86%, best calibration (21% over-call), 0 refusals, ~6× cheaper, **self-hostable = patient images stay in-house (data sovereignty)** — its only deficit is lower discrepancy-sensitivity (DISC 71% vs 100%, the price of being calibrated not trigger-happy). **(4) Bigger≠better:** Qwen3-VL-235B lost to the smaller Qwen2.5-VL-72B on every caption metric. **Verdict:** GPT-4o-mini stays the best single choice (sensitivity+accuracy sweet spot); Qwen2.5-VL-72B is the recommended self-hostable/privacy-preserving alternative. → `MDs/Generation Ablation/G4C_OpenSource_VLM_Analysis.md`.

**Pillar 1 still open:** VLM-ACC (single-arm caption accuracy on all imaged cases — cheap), H1 (Ms Saw — the decisive clinical validation).

---
