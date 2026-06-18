# VerdaSense — G4 Ablation Design & Dressing Product Card Feature
## Two Practical Implementation Guides

---

## Part 1 — Your G4 Ablation Understanding: Mostly Correct, One Important Clarification

### What you said:
> "Manually select best-matched wound images for each 32 cases, pass raw images to VLLM to generate captions, use captions as additional retrieval query, keep evaluation pipeline consistent, measure whether raw images as VLLM input help generation performance."

### What is exactly right:
- ✅ Manually select matched wound images for each of the 32 cases
- ✅ Pass raw wound images to VLLM (Gemini 2.5 Flash) to generate captions
- ✅ Keep the evaluation pipeline consistent — same 32 reference answers, same reference_contexts, same RAGAS FA/Safety/Completeness metrics
- ✅ Measure whether raw image → VLLM caption improves generation performance

### The one clarification — where exactly the caption enters the pipeline:

The caption is not just "an additional retrieval query" as a standalone fourth sub-query. It is injected at **two points** in the existing pipeline:

```
Point 1 — Retrieval (Sub-query C enrichment):
  Current Sub-query C = patient notes-based context query
  New Sub-query C     = patient notes + VLLM caption combined
  
  This means the dense retrieval for the patient context axis 
  now uses visual evidence, making it richer than notes alone.

Point 2 — Generation (assessment_text injection):
  Current assessment_text:
    "T.I.M.E. WOUND ASSESSMENT: T (Tissue): ... I (Infection): ..."
  
  New assessment_text adds a section:
    "VISUAL WOUND ANALYSIS (AI Image Assessment):
     TISSUE: Approximately 70% healthy red granulation tissue...
     INFECTION: Mild periwound erythema visible on medial aspect...
     MOISTURE: Wound bed appears moderately moist..."
  
  The LLM generator now sees BOTH the structured TIME labels
  AND the visual narrative when generating the recommendation.
```

So the caption simultaneously enriches **what is retrieved** and **what is generated from**. That is the meaningful difference from "just an extra retrieval query."

### The complete G4 ablation design

| Config | Wound Image | CV TIME Labels | VLLM Caption injected | Where injected |
|---|---|---|---|---|
| **G4-A** (baseline) | ✗ | ✓ from testset | ✗ | — |
| **G4-B** | ✓ | ✓ from testset | ✓ Gemini | Sub-query C only |
| **G4-C** | ✓ | ✓ from testset | ✓ Gemini | assessment_text only |
| **G4-D** | ✓ | ✓ from testset | ✓ Gemini | Both Sub-query C + assessment_text |
| **G4-E** | ✓ | ✓ from testset | ✓ GPT-4o | Both (compare VLLM models) |

- **G4-A vs G4-B:** Does visual caption in retrieval improve Context Recall?
- **G4-A vs G4-C:** Does visual caption in generation prompt improve FA?
- **G4-A vs G4-D:** Combined effect — the main result
- **G4-D vs G4-E:** Does VLLM model quality (Gemini vs GPT-4o) matter for captioning?

G4-D is your headline result. Comparing G4-D vs G4-A answers: **"Does adding a VLLM visual captioning layer to a text-only RAG system improve evidence-grounded recommendation quality?"**

### Image selection guide for the 32 cases

For each case you need a wound image that visually matches the TIME profile. You do not need pixel-perfect accuracy — approximate visual match is sufficient because you are evaluating the VLLM caption's contribution to the RAG, not the VLLM's classification accuracy.

**Practical approach:**
1. Download Medetec free wound database (~free, immediate, diverse wound types)
2. Download AZH wound dataset (request form, usually fast)
3. For each case, use Gemini to auto-label each public image against your TIME criteria
4. Manually review and confirm the top match per case

Quick TIME-to-visual matching guide:
```
Wound Type 1 (100% granulation, not infected, dry)
→ Look for: Clean red/pink granulating wound, no pus/erythema, dry appearance

Wound Type 2 (granulating, not infected, high exudate)  
→ Look for: Granulating wound with visible exudate, moist/wet wound bed

Wound Type 3 (infected, dry, <25% NV)
→ Look for: Wound with perilesional erythema, dry bed, mostly granulating

Wound Type 4 (infected, high exudate)
→ Look for: Wound with erythema + wet exudate visible

Wound Type 5 (>25% NV, not infected, dry)
→ Look for: Wound with slough/yellow fibrin or eschar, dry

Wound Type 6 (>25% NV, not infected, wet)
→ Look for: Heavily sloughy wet wound

Wound Type 7/8 (NV + infected)
→ Look for: Necrotic/sloughy wound with visible infection signs
```

For Category B cases (contraindication notes): any appropriate wound image works — the contraindication is triggered by the patient notes, not by the visual appearance.

---

## Part 2 — Dressing Product Visual Cards for Elderly Patients

### Your idea, restated precisely:
> "Output dressing sample images with correct and available brands in Malaysia alongside the text recommendation, to help elderly patients do self-treatment for simple cases (no antibiotics/referral required)."

### Is this practical for your FYP scope? Yes — and here is exactly how.

**The core insight:** You do NOT need web search at runtime, and you do NOT need a large image database. You need a **small static lookup table** (~15 dressing types) that maps dressing type names to product cards. The recommendation output already tells you which dressing type is recommended — you just need to look it up.

---

### Implementation Design

#### Step 1: Build `dressing_products_my.json`

This is a one-time static JSON file you maintain. It maps each dressing type to a product card with Malaysian-specific information.

```json
[
  {
    "type_id": "film",
    "display_name": "Film Dressing",
    "patient_friendly_name": "Transparent Plastic Film Dressing",
    "description_simple": "A thin, clear, waterproof sticker-like dressing. You can see the wound through it without removing it.",
    "appearance_description": "Thin transparent film, like a large clear plaster",
    "when_used": "Clean wounds with little or no fluid. Good for covering small healed or healing wounds.",
    "how_to_apply": "Peel and stick directly over wound. Smooth out air bubbles. Change every 2–5 days or when edges lift.",
    "change_frequency": "Every 2–5 days",
    "image_url": "https://www.3m.com/3M/en_US/p/d/cbgnaw011339/",
    "product_image_filename": "film_dressing.jpg",
    "malaysian_brands": [
      { "brand": "3M Tegaderm Film", "size_guide": "6x7cm, 10x12cm common sizes", "available_at": ["Guardian", "Watson", "hospital pharmacy", "Lazada"] },
      { "brand": "Smith & Nephew OpSite", "size_guide": "6x7cm standard", "available_at": ["hospital pharmacy", "medical supply"] },
      { "brand": "Hartmann Hydrofilm", "size_guide": "6x7cm standard", "available_at": ["hospital pharmacy"] }
    ],
    "price_range_myr": "RM 3–8 per piece",
    "safety_notes": "Do not use on infected wounds or wounds with heavy discharge.",
    "contraindicated_for": ["infected wounds", "high exudate wounds"]
  },
  {
    "type_id": "hydrocolloid",
    "display_name": "Hydrocolloid Dressing",
    "patient_friendly_name": "Waterproof Gel Plaster",
    "description_simple": "A thick, self-adhesive dressing that forms a protective gel over your wound as it absorbs fluid. Waterproof.",
    "appearance_description": "Beige or skin-coloured thick padded plaster",
    "when_used": "Wounds with light to moderate fluid. Promotes healing by keeping wound moist.",
    "how_to_apply": "Peel backing and apply to clean dry skin around wound. Press edges firmly. Change every 3–5 days or when gel leaks from edges.",
    "change_frequency": "Every 3–5 days",
    "image_url": "https://www.convatec.com/wound-skin/duoderm/",
    "product_image_filename": "hydrocolloid_dressing.jpg",
    "malaysian_brands": [
      { "brand": "ConvaTec DuoDERM Extra Thin", "size_guide": "10x10cm common", "available_at": ["Guardian", "Watson", "hospital pharmacy", "Shopee"] },
      { "brand": "3M Tegaderm Hydrocolloid", "size_guide": "10x10cm, oval", "available_at": ["Guardian", "hospital pharmacy"] },
      { "brand": "Molnlycke Comfeel Plus", "size_guide": "10x10cm, various shapes", "available_at": ["hospital pharmacy", "medical supply"] }
    ],
    "price_range_myr": "RM 8–20 per piece",
    "safety_notes": "Do not use on infected wounds. Not suitable for wounds requiring debridement.",
    "contraindicated_for": ["infected wounds", "necrotic wounds"]
  },
  {
    "type_id": "foam",
    "display_name": "Foam Dressing",
    "patient_friendly_name": "Soft Foam Absorbing Dressing",
    "description_simple": "A soft, spongy dressing that absorbs a lot of wound fluid. Very gentle — does not stick to the wound.",
    "appearance_description": "Thick soft foam pad, usually white or cream coloured",
    "when_used": "Wounds with moderate to heavy fluid. Very comfortable for sensitive skin.",
    "how_to_apply": "Place foam pad over wound. Secure with tape or use bordered foam with built-in adhesive. Change every 2–4 days depending on fluid amount.",
    "change_frequency": "Every 2–4 days",
    "image_url": "https://www.molnlycke.com/wound-care-products/mepilex/",
    "product_image_filename": "foam_dressing.jpg",
    "malaysian_brands": [
      { "brand": "Molnlycke Mepilex", "size_guide": "10x10cm, 15x15cm common", "available_at": ["hospital pharmacy", "medical supply"] },
      { "brand": "Smith & Nephew Allevyn", "size_guide": "10x10cm standard", "available_at": ["hospital pharmacy", "Lazada"] },
      { "brand": "ConvaTec Aquacel Foam", "size_guide": "10x10cm, self-adhesive version available", "available_at": ["hospital pharmacy"] }
    ],
    "price_range_myr": "RM 15–40 per piece",
    "safety_notes": "Requires a secondary dressing (tape or bandage) if non-bordered type.",
    "contraindicated_for": []
  },
  {
    "type_id": "alginate",
    "display_name": "Alginate Dressing",
    "patient_friendly_name": "Seaweed Fibre Absorbing Dressing",
    "description_simple": "Made from seaweed fibres. Absorbs a large amount of wound fluid and forms a soft gel. Very good for wet wounds.",
    "appearance_description": "White fibrous pad or rope that looks like cotton wool",
    "when_used": "Wounds producing a lot of fluid. Also helps stop minor bleeding.",
    "how_to_apply": "Lay alginate sheet on wound or pack loosely into wound cavity with alginate rope. Cover with a secondary dressing (foam or bandage). Change every 1–3 days depending on fluid.",
    "change_frequency": "Every 1–3 days",
    "product_image_filename": "alginate_dressing.jpg",
    "malaysian_brands": [
      { "brand": "ConvaTec Kaltostat", "size_guide": "10x10cm sheet, 2g rope", "available_at": ["hospital pharmacy"] },
      { "brand": "Molnlycke Seasorb", "size_guide": "10x10cm sheet", "available_at": ["hospital pharmacy", "medical supply"] },
      { "brand": "Smith & Nephew Algisite M", "size_guide": "10x10cm sheet", "available_at": ["hospital pharmacy"] }
    ],
    "price_range_myr": "RM 12–30 per piece",
    "safety_notes": "Always needs a secondary dressing on top. Do not use on dry wounds.",
    "contraindicated_for": ["dry wounds", "wounds with heavy necrosis requiring debridement"]
  },
  {
    "type_id": "silver",
    "display_name": "Silver Antimicrobial Dressing",
    "patient_friendly_name": "Silver Infection-Fighting Dressing",
    "description_simple": "Contains silver which kills bacteria in the wound. Used when wound shows signs of infection.",
    "appearance_description": "Grey or silver-coloured pad or fibrous dressing",
    "when_used": "Infected wounds or wounds at high risk of infection. Silver side must face wound.",
    "how_to_apply": "Apply with silver side touching the wound bed. May need secondary dressing depending on exudate level. Change every 2–3 days.",
    "change_frequency": "Every 2–3 days, review after 2 weeks",
    "product_image_filename": "silver_dressing.jpg",
    "malaysian_brands": [
      { "brand": "ConvaTec Aquacel Ag", "size_guide": "10x10cm, 15x15cm", "available_at": ["hospital pharmacy"] },
      { "brand": "Smith & Nephew Acticoat 7", "size_guide": "10x10cm, 5x5cm", "available_at": ["hospital pharmacy"] },
      { "brand": "Molnlycke Mepilex Ag", "size_guide": "10x10cm", "available_at": ["hospital pharmacy"] }
    ],
    "price_range_myr": "RM 30–80 per piece",
    "safety_notes": "Do not use for more than 2 consecutive weeks without clinical review. Do NOT use if allergic to silver.",
    "contraindicated_for": ["thyroid conditions (iodine dressings)", "clean non-infected wounds (unnecessary)"]
  },
  {
    "type_id": "hydrogel",
    "display_name": "Hydrogel Dressing",
    "patient_friendly_name": "Water Gel Dressing",
    "description_simple": "A cool, water-based gel that softens dead tissue in the wound and keeps it moist. Good for dry or painful wounds.",
    "appearance_description": "Clear or translucent gel in a tube or pre-loaded on a sheet",
    "when_used": "Dry wounds, wounds with hard or dead tissue (slough/necrosis) that needs softening.",
    "how_to_apply": "Apply gel directly to wound bed (about 5mm thick) or apply gel sheet. Cover with a secondary dressing. Change every 1–3 days.",
    "change_frequency": "Every 1–3 days",
    "product_image_filename": "hydrogel_dressing.jpg",
    "malaysian_brands": [
      { "brand": "Smith & Nephew IntraSite Gel", "size_guide": "8g, 15g tubes", "available_at": ["hospital pharmacy", "Guardian selected"] },
      { "brand": "ConvaTec Aquaflo", "size_guide": "Gel sheet 4x4cm", "available_at": ["hospital pharmacy"] },
      { "brand": "Hartmann Hydrosorb", "size_guide": "Gel sheet 7.5x10cm", "available_at": ["hospital pharmacy"] }
    ],
    "price_range_myr": "RM 15–35 per piece/tube",
    "safety_notes": "Always cover with a secondary dressing. Do not use on heavily infected wounds without clinical advice.",
    "contraindicated_for": ["high exudate wounds", "heavily infected wounds without medical supervision"]
  },
  {
    "type_id": "iodine",
    "display_name": "Iodine Dressing (Cadexomer Iodine)",
    "patient_friendly_name": "Iodine Antibacterial Dressing",
    "description_simple": "Contains iodine which kills bacteria and absorbs wound fluid at the same time.",
    "appearance_description": "Brown/amber coloured pad or paste",
    "when_used": "Infected wounds that also produce moderate fluid. Very effective against biofilm.",
    "how_to_apply": "Apply paste or pad to wound bed. Cover with secondary dressing. Change every 2–3 days.",
    "change_frequency": "Every 2–3 days",
    "product_image_filename": "iodine_dressing.jpg",
    "malaysian_brands": [
      { "brand": "Smith & Nephew Iodosorb Gel/Pad", "size_guide": "10g tube, 5x5cm pad", "available_at": ["hospital pharmacy"] },
      { "brand": "Mundipharma Betadine Wound Dressing", "size_guide": "Various", "available_at": ["hospital pharmacy", "Guardian"] }
    ],
    "price_range_myr": "RM 20–60 per piece",
    "safety_notes": "CONTRAINDICATED in thyroid conditions, pregnancy, breastfeeding, iodine allergy, and renal failure.",
    "contraindicated_for": ["thyroid conditions", "pregnancy", "iodine allergy", "renal failure"]
  },
  {
    "type_id": "non_adherent",
    "display_name": "Non-Adherent / Silicone Contact Layer",
    "patient_friendly_name": "Gentle Non-Stick Wound Contact Layer",
    "description_simple": "A thin layer placed directly on the wound so the outer dressing does not stick to the wound when changed. Reduces pain on removal.",
    "appearance_description": "Thin mesh or silicone sheet, usually translucent",
    "when_used": "Fragile skin, skin tears, wounds where pain-free dressing changes are important.",
    "how_to_apply": "Place directly on wound. Add foam or absorbent pad on top. Change outer dressing as needed, leave contact layer until soiled.",
    "change_frequency": "Outer dressing every 2–3 days; contact layer every 7–14 days",
    "product_image_filename": "non_adherent_dressing.jpg",
    "malaysian_brands": [
      { "brand": "Molnlycke Mepitel One", "size_guide": "6x7cm, 13x15cm", "available_at": ["hospital pharmacy"] },
      { "brand": "Smith & Nephew Adaptic Touch", "size_guide": "7.6x10cm", "available_at": ["hospital pharmacy"] }
    ],
    "price_range_myr": "RM 20–50 per piece",
    "safety_notes": "Suitable for fragile or elderly skin. Not a standalone dressing — always use with absorbent secondary dressing.",
    "contraindicated_for": []
  },
  {
    "type_id": "charcoal",
    "display_name": "Charcoal / Odour-Absorbing Dressing",
    "patient_friendly_name": "Odour Control Dressing",
    "description_simple": "Contains activated charcoal that absorbs wound odour. Often combined with silver for infection control.",
    "appearance_description": "Black or dark grey padded dressing",
    "when_used": "Malodorous wounds, infected wounds with offensive smell.",
    "how_to_apply": "Apply over wound (charcoal layer facing wound or as directed). Do not cut — cutting releases charcoal fibres. Cover with secondary if needed.",
    "change_frequency": "Every 2–3 days",
    "product_image_filename": "charcoal_dressing.jpg",
    "malaysian_brands": [
      { "brand": "Smith & Nephew Actisorb Silver 220", "size_guide": "10.5x10.5cm", "available_at": ["hospital pharmacy"] },
      { "brand": "ConvaTec CarboFlex", "size_guide": "10x10cm", "available_at": ["hospital pharmacy"] }
    ],
    "price_range_myr": "RM 25–55 per piece",
    "safety_notes": "Do NOT cut this dressing. Not for clean non-infected wounds.",
    "contraindicated_for": ["clean non-infected wounds"]
  }
]
```

#### Step 2: Dressing Type Extractor Function

Add this to `wound_app_v5.py` to extract dressing types from the generated recommendation text:

```python
import json
from pathlib import Path

# Load once at startup
DRESSING_DB_PATH = Path("dressing_products_my.json")
with open(DRESSING_DB_PATH) as f:
    DRESSING_PRODUCT_DB = {item["type_id"]: item for item in json.load(f)}

# Keyword → type_id mapping
DRESSING_KEYWORDS = {
    "film":            "film",
    "transparent":     "film",
    "tegaderm":        "film",
    "opsite":          "film",
    "hydrocolloid":    "hydrocolloid",
    "duoderm":         "hydrocolloid",
    "comfeel":         "hydrocolloid",
    "foam":            "foam",
    "mepilex":         "foam",
    "allevyn":         "foam",
    "alginate":        "alginate",
    "kaltostat":       "alginate",
    "algisite":        "alginate",
    "silver":          "silver",
    "aquacel ag":      "silver",
    "acticoat":        "silver",
    "hydrogel":        "hydrogel",
    "intrasite":       "hydrogel",
    "hydrosorb":       "hydrogel",
    "iodine":          "iodine",
    "iodosorb":        "iodine",
    "cadexomer":       "iodine",
    "non-adherent":    "non_adherent",
    "silicone":        "non_adherent",
    "mepitel":         "non_adherent",
    "charcoal":        "charcoal",
    "actisorb":        "charcoal",
    "carboflex":       "charcoal",
}

def extract_dressing_product_cards(recommendation_text: str,
                                    referral_required: bool = False,
                                    antibiotic_required: bool = False) -> list:
    """
    Extract dressing type IDs from recommendation text and return product cards.
    Only returns cards for simple cases (no referral, no antibiotic required).
    For complex cases, returns an empty list with a flag.
    """
    # Do not show product cards for complex cases requiring clinical referral
    # These patients must see a doctor — self-purchase is inappropriate
    if referral_required:
        return []
    
    text_lower = recommendation_text.lower()
    
    # Extract primary and secondary dressing sections only
    primary_text = ""
    secondary_text = ""
    
    lines = recommendation_text.split("\n")
    current_section = None
    for line in lines:
        if "## Primary Dressing" in line:
            current_section = "primary"
        elif "## Secondary Dressing" in line:
            current_section = "secondary"
        elif line.startswith("## "):
            current_section = None
        elif current_section == "primary":
            primary_text += line.lower() + " "
        elif current_section == "secondary":
            secondary_text += line.lower() + " "
    
    found_types = []
    seen = set()
    
    # Primary dressings first (order matters for display)
    for keyword, type_id in DRESSING_KEYWORDS.items():
        if keyword in primary_text and type_id not in seen:
            found_types.append({"role": "primary", "type_id": type_id})
            seen.add(type_id)
    
    # Secondary dressings next
    for keyword, type_id in DRESSING_KEYWORDS.items():
        if keyword in secondary_text and type_id not in seen:
            found_types.append({"role": "secondary", "type_id": type_id})
            seen.add(type_id)
    
    # Build product cards
    cards = []
    for item in found_types:
        type_id = item["type_id"]
        if type_id in DRESSING_PRODUCT_DB:
            card = dict(DRESSING_PRODUCT_DB[type_id])
            card["recommendation_role"] = item["role"]  # "primary" or "secondary"
            cards.append(card)
    
    return cards
```

#### Step 3: Add to API Response

In `get_recommendation`, after Step 8 (generate recommendation), add:

```python
# Step 9: Generate dressing product cards for patient-facing display
referral_flag  = classifier.get("referral", False)
antibiotic_flag = classifier.get("antibiotic", False)

product_cards = extract_dressing_product_cards(
    recommendation_text=result,
    referral_required=referral_flag,
    antibiotic_required=antibiotic_flag,
)

# Add to response
return JSONResponse({
    "result":            result,
    "evidence":          evidence,
    # ... existing fields ...
    "product_cards":     product_cards,   # NEW
    "show_self_care":    not referral_flag,  # NEW — frontend flag
})
```

#### Step 4: Frontend Product Card UI

For each product card in `product_cards`, the frontend displays:

```
┌─────────────────────────────────────────────────────────┐
│  [PRIMARY DRESSING]                                      │
│                                                          │
│  [PRODUCT IMAGE]    FOAM DRESSING                       │
│  (product photo)    "Soft Foam Absorbing Dressing"      │
│                                                          │
│  A soft, spongy dressing that absorbs a lot of wound    │
│  fluid. Very gentle — does not stick to the wound.      │
│                                                          │
│  Change every: 2–4 days                                 │
│  Price range: RM 15–40 per piece                        │
│                                                          │
│  Available brands in Malaysia:                          │
│  • Molnlycke Mepilex — Guardian, Hospital Pharmacy     │
│  • Smith & Nephew Allevyn — Hospital Pharmacy, Lazada  │
│  • ConvaTec Aquacel Foam — Hospital Pharmacy           │
│                                                          │
│  ⚠️ Safety note: Requires tape/bandage if non-bordered  │
└─────────────────────────────────────────────────────────┘
```

---

### Where to get the product images (no web search needed)

You have three options, all practical for FYP:

**Option 1 — Manufacturer websites (free, no license issue for academic use)**  
All major dressing manufacturers have product pages with high-quality photos:
- Molnlycke: `molnlycke.com/wound-care-products/`
- ConvaTec: `convatec.com/wound-skin/`
- Smith & Nephew: `smith-nephew.com/professional/products/advanced-wound-management/`
- 3M: `3m.com/wound-care/`

Download ~1 image per dressing type (9–12 images total). Store as static files in your app's `static/dressing_images/` folder. This is standard for academic prototypes.

**Option 2 — Generic dressing type illustrations**  
Create or use open-source medical illustrations that show the dressing type visually (cross-section or package art). No copyright issue. Tools: BioRender (free for students), or existing Wikipedia medical images (Creative Commons).

**Option 3 — Google Custom Search API (runtime image search — not recommended)**  
Runtime web search for product images is fragile (results vary, wrong products may appear, API costs), slow (adds 1–2s per query), and unnecessary for 12 known dressing types. **Do not use this approach for a clinical tool.**

**Recommendation: Option 1** for packaging/product photos + **Option 2** for anatomical "how it looks on a wound" illustrations. ~12 static images stored locally, mapped via `product_image_filename` field in the JSON.

---

### Why This Feature Matters for Your FYP

This feature directly addresses a real clinical gap. Your ablation study shows 32 test cases — many of them are Type 1 and Type 2 (clean granulating wounds, no infection, no referral required). These are exactly the cases where:
- The patient is self-managing at home
- They need to buy a dressing from Guardian or Watson
- A clinical text recommendation alone is insufficient — they do not know what "hydrocolloid" looks like at the pharmacy shelf

For elderly patients in particular, the picture + Malaysian brand name + "available at Guardian" combination is the most clinically useful output your system can produce. It closes the last mile gap between recommendation and action.

**For your FYP framing:**  
Add this as part of O3 / RQ3: *"Can the RAG system deliver clinically usable recommendations within an acceptable response time?"* — "clinically usable" can now be defined to include patient comprehension, not just clinical accuracy. The product card feature is a direct measure of clinical usability for the patient population (elderly self-care context).

You can add a brief **usability evaluation** (5–10 elderly patient users reviewing the new interface vs old text-only output) as supporting evidence for O3 — very achievable, no IRB needed for non-clinical usability testing of a software interface.

---

## Summary

| Question | Answer |
|---|---|
| G4 ablation — is my understanding correct? | Yes, with one clarification: caption is injected at two points — Sub-query C enrichment (retrieval) AND assessment_text (generation prompt). Both injection points are tested separately in G4-B vs G4-C before combining in G4-D. |
| Product cards practical for FYP? | Yes — static JSON lookup (12–15 dressing types), keyword extraction from recommendation text, 12 static product images from manufacturer websites. No runtime web search needed. Show only for non-referral, non-antibiotic cases (safe self-care scope). |
| Images source for product cards? | Download from manufacturer websites (Molnlycke, ConvaTec, 3M, Smith & Nephew) + store as static files. No web search at runtime. |
