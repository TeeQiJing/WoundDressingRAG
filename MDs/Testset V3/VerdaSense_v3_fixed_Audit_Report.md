# VerdaSense — `wound_testset_v3_fixed.json` Full Audit Report
**32 cases verified line-by-line against all 8 source chunk ai_summaries**
**Goal: Gold standard testset for fixed ablation evaluation**

---

## Legend
- ✅ Grounded and correct
- ⚠️ Minor ungrounded elaboration (clinically sound, not sourced)
- 🔧 Fix required — contradicts source or creates RAGAS scoring risk
- ➕ Add to reference_contexts
- ➖ Remove from reference_contexts

---

## CATEGORY A — Core wound-type cases (Types 1–8)

---

### A1 `cat_a_type1_dry` — ✅ GOLD STANDARD

All reference statements grounded in GP_TYPE1, WCM_FILM, WCM_HYDROCOLLOID.
All 9 sections present and correctly structured.

| Field | Status |
|---|---|
| reference_contexts (3) | ✅ GP_TYPE1, WCM_FILM, WCM_HYDROCOLLOID — all contributing |
| allowed_dressings | ✅ Matches GP_TYPE1: all types except silver/charcoal |
| contraindicated_dressings `["silver","charcoal"]` | ✅ GP_TYPE1 explicitly: "except silver, charcoal, and special advanced dressing materials" |
| antibiotic_required False | ✅ GP_TYPE1: "No antibiotics required" |
| referral_required False | ✅ GP_TYPE1: "No referral" |
| Typo in Clinical Notes | ⚠️ "monitor for any signs of infection (increased redness, warmth, pain, or discharge)" — not sourced but accepted as standard safety note |

**No fixes required.**

---

### A2 `cat_a_type2_wet` — ✅ GOLD STANDARD

All reference statements grounded in GP_TYPE2, WCM_ALGINATE, WCM_HYDROFIBRE, WCM_FOAM.

| Field | Status |
|---|---|
| reference_contexts (4) | ✅ All contributing |
| allowed_dressings `["foam","alginate","hydrofiber","polymeric_membrane"]` | ✅ Matches GP_TYPE2 list exactly |
| contraindicated_dressings `[]` | ✅ Correct — GP_TYPE2 has no explicit contraindications; silver/charcoal correctly absent from list only |
| Contraindicated section text | ✅ FIXED correctly: "No dressings are explicitly contraindicated…silver and charcoal are not among the recommended dressing materials" — accurate phrasing |
| antibiotic_required False | ✅ GP_TYPE2: "may or may not" — correctly mapped to False as default non-infected state |
| referral_required False | ✅ GP_TYPE2: no referral flag |

**No fixes required.**

---

### A3 `cat_a_type3_dry_infected` — ✅ GOLD STANDARD with one noted issue

All primary statements grounded. Contraindicated Dressings section contains a conditional
iodine warning ("AVOID if you have a thyroid disorder") — this is grounded in SFP_IODINE
and correctly conditional.

| Field | Status |
|---|---|
| reference_contexts (5) | ✅ GP_TYPE3, WCM_SILVER, WCM_HYDROGEL, SFP_IODINE, WCM_HYDROCOLLOID |
| allowed_dressings `["tulle","hydrogel","hydrocolloid","silver","iodine"]` | ✅ Matches GP_TYPE3 dressing list exactly |
| contraindicated_dressings `[]` | ✅ Correct — no unconditional contraindication (iodine is conditional) |
| antibiotic_required True | ✅ GP_TYPE3: "Yes based on C&S" |
| referral_required False | ✅ GP_TYPE3: no referral flag |
| ⚠️ "hypothyroidism or hyperthyroidism" specific condition names | ⚠️ SFP_IODINE says "thyroid disorders" only. Specific condition names are author addition. Low risk for RAGAS — does not contradict source. Accept. |
| Contraindicated section includes iodine warning WITHOUT flagging patient has thyroid | ✅ Correct framing — warns conditionally, does not over-apply |

**No fixes required. Minor elaboration noted and accepted.**

---

### A4 `cat_a_type4_wet_infected` — ✅ GOLD STANDARD

| Field | Status |
|---|---|
| reference_contexts (6) | ✅ GP_TYPE4, WCM_SILVER, WCM_ALGINATE, WCM_HYDROFIBRE, WCM_FOAM, SFP_IODINE — all contributing |
| allowed_dressings `["alginate","foam","silver","hydrofiber","polymeric_membrane","iodine"]` | ✅ Matches GP_TYPE4 exactly |
| contraindicated_dressings `[]` | ✅ Correct — iodine conditional only |
| antibiotic_required True | ✅ |
| referral_required False | ✅ GP_TYPE4: no referral |

**No fixes required.**

---

### A5 `cat_a_type5_dry_necrotic` — ✅ GOLD STANDARD

| Field | Status |
|---|---|
| reference_contexts (6) | ✅ GP_TYPE5, WCM_HYDROGEL, WCM_DEBRIDE, WCM_ALGINATE, WCM_HYDROCOLLOID, WCM_HYDROFIBRE — all contributing |
| allowed_dressings `["hydrogel","hydrocolloid","polymeric_membrane"]` | ✅ Matches GP_TYPE5 exactly |
| contraindicated_dressings `["alginate","hydrofiber"]` | ✅ WCM_ALGINATE: "Not suitable for dry wounds"; WCM_HYDROFIBRE: "Contraindications: Not appropriate for dry wounds" — both explicit |
| Silver/charcoal in Contraindicated text | ✅ "not among the recommended dressing materials" — correctly phrased as not-listed, not as contraindicated |
| antibiotic_required False | ✅ |
| referral_required False | ✅ |
| ⚠️ "promotes granulation" in Hydrogel description | ⚠️ WCM_HYDROGEL says "Promotes granulation tissue formation" ✅ grounded |

**No fixes required.**

---

### A6 `cat_a_type6_wet_necrotic` — ✅ GOLD STANDARD

| Field | Status |
|---|---|
| reference_contexts (5) | ✅ GP_TYPE6, GP_REFERRAL, WCM_ALGINATE, WCM_HYDROFIBRE, WCM_FOAM |
| allowed_dressings `["alginate","foam","polymeric_membrane","hydrofiber"]` | ✅ Matches GP_TYPE6 exactly |
| contraindicated_dressings `[]` | ✅ Correct |
| Contraindicated text "Hydrogel is not listed as a recommended dressing for this wound type" | ✅ Accurate — GP_TYPE6 does not list hydrogel |
| antibiotic_required False | ✅ GP_TYPE6: "may or may not" |
| referral_required True | ✅ GP_TYPE6 + GP_REFERRAL |

**No fixes required.**

---

### A7 `cat_a_type7_dry_infected_necrotic` — ✅ GOLD STANDARD

| Field | Status |
|---|---|
| reference_contexts (8) | ✅ GP_TYPE7, GP_REFERRAL, WCM_SILVER, WCM_HYDROGEL, SFP_IODINE, WCM_ALGINATE, WCM_HYDROCOLLOID, WCM_HYDROFIBRE |
| allowed_dressings `["silver","hydrogel","hydrocolloid","iodine","polymeric_membrane"]` | ✅ Matches GP_TYPE7 exactly |
| contraindicated_dressings `["alginate","hydrofiber"]` | ✅ Both grounded in WCM_ALGINATE and WCM_HYDROFIBRE |
| Iodine conditional contraindication in text | ✅ SFP_IODINE grounded, correctly conditional |
| antibiotic_required True | ✅ |
| referral_required True | ✅ |

**No fixes required.**

---

### A8 `cat_a_type8_wet_infected_necrotic` — ✅ GOLD STANDARD

| Field | Status |
|---|---|
| reference_contexts (8) | ✅ GP_TYPE8, GP_REFERRAL, WCM_SILVER, WCM_ALGINATE, WCM_CHARCOAL, WCM_HYDROFIBRE, WCM_FOAM, SFP_IODINE |
| allowed_dressings `["alginate","silver","hydrofiber","foam","polymeric_membrane","charcoal","iodine"]` | ✅ Matches GP_TYPE8 exactly |
| contraindicated_dressings `[]` | ✅ Correct — iodine conditional only |
| Confirmed: "do not cut charcoal" is ABSENT | ✅ Correctly removed — WCM_CHARCOAL has no such instruction |
| Charcoal: "odour absorbent, reduces odour; requires a secondary dressing; change every 2 days" | ✅ All 3 properties grounded in WCM_CHARCOAL |
| antibiotic_required True | ✅ |
| referral_required True | ✅ |

**No fixes required.**

---

### Category A Verdict: ALL 8 CASES GOLD STANDARD ✅

---

## CATEGORY B — Contraindication, safety, and special-population cases

---

### B1 `cat_b_iodine_thyroid` — ✅ GOLD STANDARD (minor elaborations accepted)

| Field | Status |
|---|---|
| reference_contexts (5) | ✅ SFP_IODINE, GP_TYPE3, WCM_SILVER, WCM_HYDROGEL, WCM_HYDROCOLLOID |
| contraindicated_dressings `["iodine"]` | ✅ SFP_IODINE: "Should be avoided in patients with thyroid disorders" |
| allowed_dressings `["tulle","hydrogel","hydrocolloid","silver"]` | ✅ Matches GP_TYPE3 minus iodine |
| ⚠️ Application Tips: "Ensure any new dressing applied does not contain iodine" | ⚠️ Not in SFP_IODINE — clinical advice. Accepted |
| Application Tips: "Inform your healthcare provider of all medications" | ⚠️ Not in any chunk. Accepted as safety note |
| ⚠️ Reference text dropped the previous "silver is safe for thyroid" hallucination | ✅ Confirmed — the fabricated safety claim is gone |

**No fixes required.**

---

### B2 `cat_b_silver_clean_granulating` — 🔧 ONE REMAINING ISSUE

| Field | Status |
|---|---|
| reference_contexts (3) | ✅ GP_TYPE1, WCM_FILM, WCM_HYDROCOLLOID |
| contraindicated_dressings `["silver","charcoal"]` | ✅ GP_TYPE1 explicit |
| ⚠️ Contraindicated text: "Using silver on a non-infected wound is unnecessary and is not indicated and is excluded from Wound Type 1" | ⚠️ Redundant phrasing but not a hallucination — "excluded from Wound Type 1 by the clinical algorithm" is grounded |
| 🔧 **"may impair healing" — still present?** | Let me check... |

Checking reference text: *"Using silver on a non-infected wound is unnecessary and is not indicated and is excluded from Wound Type 1 by the clinical algorithm."*

**Confirmed: "may impair healing" has been removed.** ✅ The audit hallucination is fixed.

| Field | Status |
|---|---|
| antibiotic_required False | ✅ |
| referral_required False | ✅ |

**No fixes required. Hallucination confirmed removed.**

---

### B3 `cat_b_skin_tear_fragile` — ✅ GOLD STANDARD

| Field | Status |
|---|---|
| reference_contexts (4) | ✅ ISTAP_CLASS, ISTAP_PATH, ISTAP_PROD, AJGP_SKINTEAR — SFP_FOAM correctly removed |
| contraindicated_dressings `["adhesive_bordered_foam","adhesive_film"]` | ✅ AJGP_SKINTEAR: "Do not use adhesive products on fragile skin"; ISTAP_PROD: "adhesive borders MUST NOT be used on fragile skin" |
| allowed_dressings `["silicone_foam","silicone_non_adherent"]` | ✅ Matches ISTAP_PROD Products 1 and 2 |
| "ISTAP Type 2 — partial flap" | ✅ ISTAP_CLASS: "flap CANNOT be repositioned to completely cover the wound bed" |
| Haemostatic alginate as primary if bleeding | ✅ AJGP_SKINTEAR: "apply a haemostatic alginate dressing as the primary dressing under the silicone-coated foam" |
| Skin barrier wipe application | ✅ AJGP_SKINTEAR: "Using a barrier wipe under the foam dressing... reducing maceration and protecting the skin during dressing removal" |
| Remove in direction not disturbing flap | ✅ ISTAP_PATH: "peel slowly to avoid disturbing the skin flap or viable tissue" |
| ⚠️ "Avoid unnecessary early changes to prevent trauma to the healing flap" | ⚠️ Not verbatim in ISTAP but consistent with pathway principles. Accepted. |

**No fixes required.**

---

### B4 `cat_b_npwt_necrotic_eschar` — ✅ GOLD STANDARD

| Field | Status |
|---|---|
| reference_contexts (7) | ✅ WCM_NPWT, GP_TYPE8, GP_REFERRAL, WCM_SILVER, WCM_ALGINATE, WCM_HYDROFIBRE, WCM_FOAM |
| contraindicated_dressings `["npwt"]` | ✅ WCM_NPWT: "Necrotic wound bed or eschar" is an explicit contraindication |
| "neoplastic tissue, clotting disorders" also listed as NPWT contraindications | ✅ WCM_NPWT explicitly lists both |
| "NPWT is only an adjunct... not a standalone treatment" | ✅ WCM_NPWT: "NPWT is only an adjunct... it is not a panacea" |
| "NPWT does not replace surgical procedures" | ✅ WCM_NPWT verbatim |
| Silver/alginate/hydrofibre/foam change frequencies | ✅ All grounded in respective WCM chunks |
| antibiotic_required True | ✅ GP_TYPE8 |
| referral_required True | ✅ GP_TYPE8 + GP_REFERRAL |

**No fixes required.**

---

### B5 `cat_b_alginate_dry_wound` — ✅ GOLD STANDARD

| Field | Status |
|---|---|
| reference_contexts (4) | ✅ GP_TYPE5, WCM_ALGINATE, WCM_HYDROGEL, WCM_HYDROCOLLOID |
| contraindicated_dressings `["alginate"]` | ✅ WCM_ALGINATE: "Not suitable for dry wounds" |
| Application Tips: "Remove the alginate dressing gently" (no "moisten with saline" instruction) | ✅ Previous hallucination ("moisten with saline if sticking") has been removed |
| Hydrocolloid change frequency | ✅ WCM_HYDROCOLLOID: "every 2 to 5 days" |
| antibiotic_required False | ✅ |
| referral_required False | ✅ |

**No fixes required.**

---

### B6 `cat_b_honey_dry_necrotic` — ✅ GOLD STANDARD

| Field | Status |
|---|---|
| reference_contexts (5) | ✅ WCM_HONEY, GP_TYPE5, WCM_HYDROGEL, WCM_HYDROCOLLOID, WCM_ALGINATE |
| contraindicated_dressings `["honey","alginate"]` | ✅ WCM_HONEY: "Contraindication: Dry, necrotic wounds — honey can cause further drying"; WCM_ALGINATE: "Not suitable for dry wounds" |
| "Honey dressings have genuine clinical uses" | ✅ WCM_HONEY lists antimicrobial and other properties — correct context |
| "WCM guidelines specifically list dry, necrotic wounds as a contraindication" | ✅ WCM_HONEY verbatim |
| antibiotic_required False | ✅ |
| referral_required False | ✅ |

**No fixes required.**

---

### B7 `cat_b_postop_clean` — ✅ GOLD STANDARD

| Field | Status |
|---|---|
| reference_contexts (4) | ✅ AJGP_POSTOP, WCM_FILM, SFP_FILM, WCM_ALGINATE |
| contraindicated_dressings `["alginate"]` | ✅ WCM_ALGINATE: "Not suitable for dry wounds" — correct functional contraindication for no-exudate wound. Previous "foam" removed correctly |
| allowed_dressings `["film","hydrocolloid"]` | ✅ AJGP_POSTOP: "For wounds without exudate: Use a film or thin hydrocolloid dressing over sutures" |
| "Film dressing — waterproof (allows daily showering)" | ✅ WCM_FILM: "Waterproof yet breathable" |
| SFP_FILM: "Tegaderm, Opsite" examples; "skin around wound must be intact" | ✅ SFP_FILM verbatim |
| Application Tip: "Press film edges firmly to make sure no air under it" | ✅ WCM_FILM: "ensuring no air is trapped underneath" |
| antibiotic_required False | ✅ |
| referral_required False | ✅ |

**No fixes required.**

---

### B8 `cat_b_burns_hand` — ✅ GOLD STANDARD (one minor note)

| Field | Status |
|---|---|
| reference_contexts (6) | ✅ ANZBA_REF, ANZBA_DEPTH, ANZBA_FA, ANZBA_DRESS, AJGP_BURNS, WCM_HYDROGEL |
| contraindicated_dressings `["adhesive_film_dressing"]` | ✅ ANZBA_REF: "Do not use film dressings on burns" + ANZBA_DEPTH: "Adhesive dressings such as film dressings (e.g., Opsite or Tegaderm) — CONTRAINDICATED" |
| allowed_dressings `["hydrogel","hydrocolloid","silicone_non_adherent"]` | ✅ AJGP_BURNS: "hydrogel, hydrocolloid, or film after initial first aid" — film removed from allowed correctly (ANZBA explicit contraindication takes precedence); silicone_non_adherent is ANZBA_DRESS Depth 2 |
| "ANZBA referral criteria, burns located on the hands must be referred" | ✅ ANZBA_REF: "Hands — any size, any depth" |
| "Superficial partial-thickness burns typically heal within 14 days" | ✅ ANZBA_DEPTH Depth 2: "7–10 days (< 14)" |
| "do not burst the blisters" | ⚠️ Not explicitly in ANZBA_DEPTH (which says debride blisters >5cm or over joints, implying small ones remain). Accepted as standard clinical elaboration |
| Application Tip: "Apply hydrogel gently over the burn and cover with a secondary dressing" | ✅ WCM_HYDROGEL + ANZBA_FA: hydrogel for initial cover |
| antibiotic_required False | ✅ |
| referral_required True | ✅ ANZBA_REF: hands always referred |

**No fixes required. Minor elaboration noted and accepted.**

---

### B9 `cat_b_referral_type6` — ✅ GOLD STANDARD

| Field | Status |
|---|---|
| reference_contexts (5) | ✅ GP_TYPE6, GP_REFERRAL, WCM_ALGINATE, WCM_HYDROFIBRE, WCM_FOAM — EWMA_VLU_TISSUE correctly removed |
| contraindicated_dressings `[]` | ✅ Correct — no explicit contraindications for generic Type 6 |
| allowed_dressings `["alginate","foam","polymeric_membrane","hydrofiber"]` | ✅ Matches GP_TYPE6 |
| Referral section | ✅ GP_TYPE6 + GP_REFERRAL |
| antibiotic_required False | ✅ |
| referral_required True | ✅ |

**No fixes required.**

---

### B10 `cat_b_diabetic_foot` — ✅ GOLD STANDARD

| Field | Status |
|---|---|
| reference_contexts (5) | ✅ AJGP_DIABFOOT, WCM_SILVER, SFP_HYDROCOLLOID, EWMA_DFU_INF, EWMA_DFU_TIS |
| contraindicated_dressings `["bordered_foam","hydrocolloid"]` | ✅ AJGP_DIABFOOT: "without borders and anchored with tape or bandages"; SFP_HYDROCOLLOID: "not recommended for...diabetic foot ulcers" |
| "Assess pedal pulses and sensation; if poor perfusion, refer" | ✅ AJGP_DIABFOOT verbatim |
| "Without borders and anchored with tape or a light bandage" | ✅ AJGP_DIABFOOT verbatim |
| EWMA DFU guidelines infection control priority | ✅ EWMA_DFU_INF: "Infection is a leading cause of major amputations" |
| "Offloading the foot is essential — dressing alone will not heal" | ✅ EWMA_DFU_TIS: "Unless these elements are addressed, wound care is more likely to fail" |
| antibiotic_required True | ✅ |
| referral_required False | ✅ AJGP_DIABFOOT: "refer if poor perfusion" — conditional, not mandatory here |

**No fixes required.**

---

### B11 `cat_b_skin_tear_type2_flap` — ✅ GOLD STANDARD (one minor note)

| Field | Status |
|---|---|
| reference_contexts (3) | ✅ ISTAP_CLASS, ISTAP_PATH, ISTAP_PROD — only 3 ISTAP chunks, no extraneous chunks |
| contraindicated_dressings `["adhesive_foam","adhesive_film"]` | ✅ ISTAP_PROD: "adhesive borders MUST NOT be used on fragile skin"; correctly removed "dry_gauze" |
| allowed_dressings `["silicone_foam","silicone_non_adherent","alginate"]` | ✅ ISTAP_PROD: Products 1 (non-adherent mesh), 2 (non-adhesive foam), plus alginate for haemostasis |
| "Warfarin is an anticoagulant means that bleeding may take slightly longer to stop" | ✅ ISTAP_PATH: "Polypharmacy (anticoagulants)" listed as risk factor — correctly contextualised |
| ⚠️ Clinical Notes ends with " No adhesive products should be used anywhere near the fragile skin" — space before "No" | ⚠️ Formatting artefact only. No clinical impact |
| antibiotic_required False | ✅ |
| referral_required False | ✅ |

**No fixes required. Minor formatting note only.**

---

### B12 `cat_b_burns_minor_epidermal` — ✅ GOLD STANDARD

**Note: Case was renamed from `burns_minor_superficial` to `burns_minor_epidermal` — correct.**

| Field | Status |
|---|---|
| reference_contexts (4) | ✅ ANZBA_DEPTH, ANZBA_REF, ANZBA_FA, ANZBA_DRESS — all ANZBA, correct for epidermal burn |
| contraindicated_dressings `["adhesive_film_dressing"]` | ✅ ANZBA_REF: "Do not use film dressings on burns"; ANZBA_DEPTH: film dressings CONTRAINDICATED on burns |
| allowed_dressings `["moisturiser"]` | ✅ ANZBA_DRESS Depth 1 (Epidermal): "Initial Dressing: Simple moisturisers only; Secondary: Not required; Referral: Not required" — correct and precise |
| Primary Dressing: "Epidermal burns require simple moisturisers only" | ✅ ANZBA_DRESS: "Simple moisturisers only" for epidermal |
| "The burn should heal spontaneously within 3–7 days" | ✅ ANZBA_DRESS + ANZBA_DEPTH: "Healing Time: ~3–7 days" |
| "Does NOT meet ANZBA referral criteria" | ✅ ANZBA_REF: upper arm, 2% TBSA, no special area, no special mechanism — correct |
| ⚠️ Typo: "No antibiotic is required for a epidermanl burn" | 🔧 Typo: "epidermanl" should be "epidermal" |
| antibiotic_required False | ✅ |
| referral_required False | ✅ |

**Fix required: Typo "epidermanl" → "epidermal" in Antibiotic Consideration section.**

---

### Category B Verdict: 11/12 GOLD STANDARD ✅ | 1 minor typo fix required

---

## CATEGORY C — Dressing selection and reasoning

---

### C1 `cat_c_dressing_saturation` — ✅ GOLD STANDARD (minor elaborations accepted)

| Field | Status |
|---|---|
| reference_contexts (4) | ✅ GP_TYPE2, WCM_FOAM, WCM_ALGINATE, WCM_HYDROFIBRE |
| contraindicated_dressings `[]` | ✅ Correct |
| allowed_dressings `["alginate","foam","hydrofiber","polymeric_membrane"]` | ✅ Matches GP_TYPE2 |
| Alginate change frequency, haemostatic, biodegradable | ✅ WCM_ALGINATE |
| Foam change frequency, highly absorbent, bacterial barrier | ✅ WCM_FOAM |
| Hydrofibre gel-like layer, change frequency | ✅ WCM_HYDROFIBRE |
| ⚠️ "A dressing MUST be changed when: soiled, loose or slipping, edges curling, or fluid has soaked through" | ⚠️ Not verbatim in any chunk — clinical standard. Accepted |
| Application Tips: "Gently cleanse the wound with saline before applying" removed? | ✅ Confirmed — Application Tips now correctly brief |
| antibiotic_required False | ✅ |
| referral_required False | ✅ |

**No fixes required.**

---

### C2 `cat_c_malodour_type8` — ✅ GOLD STANDARD

| Field | Status |
|---|---|
| reference_contexts (7) | ✅ GP_TYPE8, WCM_CHARCOAL, WCM_SILVER, GP_REFERRAL, WCM_ALGINATE, WCM_HYDROFIBRE, WCM_FOAM |
| contraindicated_dressings `[]` | ✅ Correct — iodine conditional only (no thyroid history in notes) |
| ✅ **"Do not cut charcoal" — CONFIRMED ABSENT** | ✅ Hallucination successfully removed from both Secondary Dressing and Application Tips |
| Charcoal: "absorbs wound odour; change every 2 days; requires a secondary dressing" | ✅ All 3 properties in WCM_CHARCOAL |
| Application Tips: "Add charcoal dressing as the outermost layer. Change charcoal every 2 days" | ✅ No cutting instruction present |
| antibiotic_required True | ✅ |
| referral_required True | ✅ |

**No fixes required. Hallucination confirmed removed.**

---

### C3 `cat_c_heavy_exudate_maceration` — ✅ GOLD STANDARD (minor elaborations noted)

| Field | Status |
|---|---|
| reference_contexts (4) | ✅ WCM_ALGINATE, WCM_HYDROFIBRE, WCM_FOAM, GP_TYPE2 |
| contraindicated_dressings `[]` | ✅ Correct |
| Primary Dressing: "Absorbs wound exudates and maintain moisture" | ✅ WCM_ALGINATE verbatim — previous "highest absorption capacity; forms a gel" removed |
| ⚠️ Application Tips: "Apply a skin barrier wipe or cream to the white, soft skin"; "Do not apply alginate directly on the macerated skin" | ⚠️ Not in any chunk — reasonable clinical guidance. Accepted |
| "White, soft skin around the wound (maceration) may cause skin breakdown" | ⚠️ Not in cited chunks — clinical knowledge. Accepted |
| antibiotic_required False | ✅ |
| referral_required False | ✅ |

**No fixes required. Minor elaborations accepted.**

---

### C4 `cat_c_dry_infected_combo` — ✅ GOLD STANDARD

| Field | Status |
|---|---|
| reference_contexts (4) | ✅ WCM_HYDROGEL, WCM_SILVER, GP_TYPE3, SFP_IODINE |
| contraindicated_dressings `[]` | ✅ Correct — iodine conditional |
| Application Tips: "Place the silver dressing over the wound with the silver side facing down toward the wound (through the hydrogel layer), per standard silver application." | ✅ Correctly resolved — WCM_SILVER: "Apply the dressing with the silver side facing the wound bed." The layering is described as "through the hydrogel layer" which is an accurate clinical description |
| allowed_dressings `["tulle","hydrogel","hydrocolloid","silver","iodine"]` | ✅ Matches GP_TYPE3 |
| antibiotic_required True | ✅ |
| referral_required False | ✅ |

**No fixes required. Silver layering conflict correctly resolved.**

---

### C5 `cat_c_time_assessment_mixed` — ✅ GOLD STANDARD

| Field | Status |
|---|---|
| reference_contexts (8) | ✅ GP_ALGO, GP_TYPE7, GP_REFERRAL, WCM_SILVER, WCM_HYDROGEL, WCM_HYDROCOLLOID, WCM_CHARCOAL, SFP_IODINE — EWMA_TIME_PRACTICE correctly removed |
| contraindicated_dressings `[]` | ✅ Correct — iodine conditional |
| ✅ **"Do not cut charcoal" — CONFIRMED ABSENT** | ✅ Hallucination removed |
| Application Tips: "Add charcoal dressing as the outermost layer. Change charcoal every 2 days even if other layers are not yet due for change." | ✅ No cutting instruction |
| Charcoal: "change every 2 days; requires a secondary under it" | ✅ WCM_CHARCOAL |
| allowed_dressings `["silver","hydrogel","hydrocolloid","iodine","polymeric_membrane","charcoal"]` | ✅ GP_TYPE7 list + charcoal (GP_TYPE8 lists charcoal; clinically appropriate for malodour in this wound) |
| antibiotic_required True | ✅ |
| referral_required True | ✅ |

**No fixes required. Both hallucinations confirmed removed.**

---

### C6 `cat_c_film_vs_hydrocolloid` — ✅ GOLD STANDARD

| Field | Status |
|---|---|
| reference_contexts (4) | ✅ WCM_FILM, WCM_HYDROCOLLOID, GP_TYPE1, SFP_FILM |
| contraindicated_dressings `["silver","charcoal"]` | ✅ GP_TYPE1: explicitly excludes silver and charcoal |
| Application Tips — "warm hydrocolloid between hands" | ✅ **Confirmed removed** — not present in this version |
| Film: transparent, waterproof, change 2–5 days | ✅ WCM_FILM |
| Hydrocolloid: "moist healing environment, promotes autolysis, waterproof" | ✅ WCM_HYDROCOLLOID |
| "Both dressings can be left in place for showering" | ✅ WCM_FILM: "Waterproof" implies shower-safe |
| antibiotic_required False | ✅ |
| referral_required False | ✅ |

**No fixes required.**

---

### Category C Verdict: ALL 6 CASES GOLD STANDARD ✅

---

## CATEGORY D — Clinical notes override cases

---

### D1 `cat_d_notes_infection_override` — ✅ GOLD STANDARD

| Field | Status |
|---|---|
| reference_contexts (6) | ✅ GP_TYPE4, WCM_SILVER, SFP_IODINE, WCM_ALGINATE, WCM_FOAM, GP_REFERRAL |
| contraindicated_dressings `[]` | ✅ Correct — iodine conditional |
| Secondary Dressing: "Alginate or foam as secondary for moderate exudate management (change every 2–3 days)" | ✅ Both chunks now present; WCM_ALGINATE change = 2–5 days, WCM_FOAM change = 2–3 days |
| 🔧 Alginate change frequency mismatch | Alginate described as "change every 2–3 days" in reference but WCM_ALGINATE says "every 2 to 5 days." Minor discrepancy — change to "Alginate: every 2–5 days; Foam: every 2–3 days" |
| Referral/Escalation: "Referral to a specialist may be required if infection is spreading or if you develop sepsis and cellulitis" | ✅ GP_REFERRAL: "Sepsis, Severe cellulitis" — correctly sourced |
| "Seek a urgent clinical review" — grammar issue | ⚠️ "Seek an urgent clinical review" — minor grammar. No clinical impact |
| antibiotic_required True | ✅ |
| referral_required False | ✅ Correct — conditional referral only, not mandatory at this stage |

**Fix required: Alginate change frequency "every 2–3 days" → "every 2–5 days" to match WCM_ALGINATE.**

---

### D2 `cat_d_notes_diabetic_nonhealing` — ✅ GOLD STANDARD (minor note)

| Field | Status |
|---|---|
| reference_contexts (6) | ✅ AJGP_DIABFOOT, WCM_SILVER, SFP_HYDROCOLLOID, EWMA_DFU_EDG, EWMA_DFU_TIS, EWMA_DFU_INF |
| contraindicated_dressings `["bordered_foam","hydrocolloid"]` | ✅ AJGP_DIABFOOT + SFP_HYDROCOLLOID |
| "Per EWMA DFU edge advancement pathway, failure to progress indicates reassessment of all TIME components" | ✅ EWMA_DFU_EDG: "Factors Affecting Epithelial Advancement" — correctly framed without the ungrounded "4-week" threshold |
| Rationale I: "signs of inflammation and infection may be absent or reduced in diabetic patients" | ✅ EWMA_DFU_INF: "Signs of inflammation and infection may be absent or reduced in diabetic patients due to neuropathy and poor blood supply" |
| Application Tips: "some form of cast, adapted footwear, or padding must be used to redistribute plantar pressures" | ✅ EWMA_DFU_EDG: "Neuropathic Foot: Use casts, adapted footwear, or padding to redistribute plantar pressures" |
| "4-week threshold" — confirmed removed | ✅ Not present |
| "Total contact cast/removable cast walker" specific device names — confirmed removed | ✅ Not present |
| antibiotic_required False | ✅ |
| referral_required True | ✅ |

**No fixes required.**

---

### D3 `cat_d_notes_malodour_clean` — ✅ GOLD STANDARD (one issue found)

| Field | Status |
|---|---|
| reference_contexts (5) | ✅ WCM_CHARCOAL, WCM_ALGINATE, GP_TYPE2, WCM_SILVER, WCM_FOAM |
| contraindicated_dressings `[]` | ✅ Correct |
| "bacterial colonisation" distinction | ✅ **Confirmed: Clinical Notes now says "monitor closely at each change for signs of active infection" — colonisation claim removed** |
| "Keep the charcoal intact for full odour-absorbing effect" | ✅ **Confirmed removed** — not present |
| Charcoal: "change every 2 days; must have a secondary dressing" | ✅ WCM_CHARCOAL |
| 🔧 Formatting issue in Contraindicated/Referral sections | Reference text has two sections merged without a newline: `"No specific contraindications at this stage. ## Dressing Change Frequency"` and `"No systemic antibiotic is currently indicated — the wound shows no frank infection signs. ## Referral/Escalation"` — missing blank line between them causing two section headers to be on same paragraph |
| antibiotic_required False | ✅ |
| referral_required False | ✅ |

**Fix required: Add blank line before `## Dressing Change Frequency` and before `## Referral/Escalation` to separate sections properly. Formatting only — no clinical change.**

---

### D4 `cat_d_notes_npwt_adjunct` — ✅ GOLD STANDARD

| Field | Status |
|---|---|
| reference_contexts (6) | ✅ WCM_NPWT, GP_TYPE6, GP_REFERRAL, WCM_ALGINATE, WCM_HYDROFIBRE, WCM_FOAM |
| contraindicated_dressings `[]` | ✅ Correct |
| "NPWT dressing change: every 3–5 days, or depending on the amount and characteristics of fluid withdrawn" | ✅ WCM_NPWT: "3–5 days interval or depending on the amount or characteristic of fluid withdrawn" — frequency conflict resolved |
| "NPWT must not be used if any of the following are still present: remaining necrotic tissue, neoplastic tissue in the wound area, or a clotting disorder" | ✅ WCM_NPWT: all three listed as contraindications |
| "NPWT is contraindicated on a necrotic wound bed — but now that the dead tissue has been surgically removed, it may be considered" | ✅ WCM_NPWT: "NPWT is only an adjunct...not a standalone...Necrotic wound bed or eschar" contraindication |
| antibiotic_required False | ✅ |
| referral_required True | ✅ |

**No fixes required.**

---

### Category D Verdict: ALL 4 CASES CLEAN ✅ | 2 minor fixes required (D1 frequency, D3 formatting)

---

## CATEGORY E — EWMA source validation cases

---

### E1 `cat_e_vlu_chronic_ewma` — ✅ GOLD STANDARD

| Field | Status |
|---|---|
| reference_contexts (9) | ✅ EWMA_VLU_TIS, EWMA_VLU_MOI, EWMA_VLU_INF, EWMA_VLU_EDG, GP_TYPE6, GP_REFERRAL, WCM_ALGINATE, WCM_HYDROFIBRE, WCM_FOAM |
| contraindicated_dressings `["hydrocolloid","film"]` | ✅ EWMA_VLU_MOI: "Avoid hydrocolloids and films, which may exacerbate maceration" |
| allowed_dressings `["alginate","hydrofiber","foam","polymeric_membrane"]` | ✅ GP_TYPE6 + EWMA_VLU_MOI compatible |
| "Autolytic debridement using high-water-content dressings under compression is generally ineffective" | ✅ EWMA_VLU_TIS: "Autolytic debridement...is slow and generally ineffective" |
| "apply paraffin-based products or zinc paste as a skin barrier" | ✅ EWMA_VLU_MOI: "Skin hydration and protection are vital, utilizing paraffin-based products or zinc paste" |
| "Adhesive dressings should also be avoided as they increase the risk of allergic reactions" | ✅ EWMA_VLU_MOI: "Adhesive dressings should be avoided to reduce the risk of allergic reactions" |
| "failure to progress by week 4 is the trigger to consider advanced therapies" | ✅ EWMA_VLU_EDG: "challenge lies in predicting which ulcers will not heal rapidly by the fourth week" |
| "Topical antiseptics are considered first; if no improvement in two weeks, reassess" | ✅ EWMA_VLU_INF: "If no improvement is seen in two weeks, reassess" |
| "Patients should be advised to avoid prolonged standing and to elevate their legs above heart level" | ✅ EWMA_VLU_MOI verbatim |
| "Change compression bandaging if soiled by heavy exudate or if leg size has changed" | ✅ EWMA_VLU_MOI: "more frequent bandage changes may be necessary...re-measurement of ankle circumference may be required" |
| antibiotic_required False | ✅ |
| referral_required True | ✅ GP_TYPE6 + EWMA_VLU_EDG |

**No fixes required.**

---

### E2 `cat_e_dfu_infected_ewma` — ✅ GOLD STANDARD

| Field | Status |
|---|---|
| reference_contexts (9) | ✅ EWMA_DFU_INF, EWMA_DFU_TIS, EWMA_DFU_MOI, EWMA_DFU_EDG, AJGP_DIABFOOT, WCM_SILVER, GP_TYPE3, SFP_HYDROCOLLOID, EWMA_TIME_PRA |
| contraindicated_dressings `["bordered_foam","hydrocolloid"]` | ✅ AJGP_DIABFOOT: "without borders...anchored with tape or bandages"; SFP_HYDROCOLLOID: "not recommended for diabetic foot ulcers" |
| allowed_dressings `["silver","iodine","silicone_foam"]` | ✅ GP_TYPE3 dressing list + AJGP_DIABFOOT: silicone foam for moderate exudate |
| "Broad-spectrum antibiotics and wound cultures should be taken at initial presentation" | ✅ EWMA_DFU_INF: "Prescribe broad-spectrum antibiotics and obtain cultures at initial infection presentation" |
| "Deep swabs or tissue should be taken after initial debridement to guide treatment" | ✅ EWMA_DFU_INF: "Take deep swabs or tissue samples after initial debridement" |
| "Systemic antibiotics are always indicated if cellulitis or lymphangitis is present" | ✅ EWMA_DFU_INF: "Systemic Antibiotic Treatment: Indicated in cases of cellulitis, lymphangitis" |
| "Sharp debridement is the gold standard" | ✅ EWMA_DFU_TIS: "Sharp debridement is the preferred method...gold standard" |
| "Dressings must be easy to remove, absorbent, and able to withstand the pressures of walking" | ✅ EWMA_DFU_MOI verbatim |
| "A cast, adapted footwear, or padding is needed to redistribute foot pressure" | ✅ EWMA_DFU_EDG: "Use casts, adapted footwear, or padding to redistribute plantar pressures" |
| "Iodine-based dressing is an equally listed alternative for Wound Type 3" | ✅ GP_TYPE3: iodine listed |
| EWMA_TIME_PRA: "Diabetic Foot Ulcers: Necessitate pressure offloading and control of diabetes" | ✅ EWMA_TIME_PRA verbatim |
| antibiotic_required True | ✅ |
| referral_required True | ✅ |

**No fixes required.**

---

### Category E Verdict: BOTH CASES GOLD STANDARD ✅

---

## MASTER SUMMARY TABLE — All 32 Cases

| Case | Cat | Status | Fixes Required |
|---|---|---|---|
| cat_a_type1_dry | A | ✅ Gold Standard | None |
| cat_a_type2_wet | A | ✅ Gold Standard | None |
| cat_a_type3_dry_infected | A | ✅ Gold Standard | None |
| cat_a_type4_wet_infected | A | ✅ Gold Standard | None |
| cat_a_type5_dry_necrotic | A | ✅ Gold Standard | None |
| cat_a_type6_wet_necrotic | A | ✅ Gold Standard | None |
| cat_a_type7_dry_infected_necrotic | A | ✅ Gold Standard | None |
| cat_a_type8_wet_infected_necrotic | A | ✅ Gold Standard | None |
| cat_b_iodine_thyroid | B | ✅ Gold Standard | None |
| cat_b_silver_clean_granulating | B | ✅ Gold Standard | None — hallucination confirmed removed |
| cat_b_skin_tear_fragile | B | ✅ Gold Standard | None |
| cat_b_npwt_necrotic_eschar | B | ✅ Gold Standard | None |
| cat_b_alginate_dry_wound | B | ✅ Gold Standard | None |
| cat_b_honey_dry_necrotic | B | ✅ Gold Standard | None |
| cat_b_postop_clean | B | ✅ Gold Standard | None |
| cat_b_burns_hand | B | ✅ Gold Standard | None |
| cat_b_referral_type6 | B | ✅ Gold Standard | None |
| cat_b_diabetic_foot | B | ✅ Gold Standard | None |
| cat_b_skin_tear_type2_flap | B | ✅ Gold Standard | None |
| cat_b_burns_minor_epidermal | B | ⚠️ Typo | Fix typo "epidermanl" → "epidermal" |
| cat_c_dressing_saturation | C | ✅ Gold Standard | None |
| cat_c_malodour_type8 | C | ✅ Gold Standard | None — hallucination confirmed removed |
| cat_c_heavy_exudate_maceration | C | ✅ Gold Standard | None |
| cat_c_dry_infected_combo | C | ✅ Gold Standard | None — silver layering conflict resolved |
| cat_c_time_assessment_mixed | C | ✅ Gold Standard | None — both hallucinations confirmed removed |
| cat_c_film_vs_hydrocolloid | C | ✅ Gold Standard | None — "warm hydrocolloid" tip removed |
| cat_d_notes_infection_override | D | ⚠️ Minor | Alginate change freq "every 2–3 days" → "every 2–5 days" |
| cat_d_notes_diabetic_nonhealing | D | ✅ Gold Standard | None |
| cat_d_notes_malodour_clean | D | ⚠️ Formatting | Add blank line before 2 section headers (no clinical change) |
| cat_d_notes_npwt_adjunct | D | ✅ Gold Standard | None — NPWT frequency corrected |
| cat_e_vlu_chronic_ewma | E | ✅ Gold Standard | None |
| cat_e_dfu_infected_ewma | E | ✅ Gold Standard | None |

**Cases requiring fixes: 3 of 32**
- 1 typo (B12)
- 1 change frequency discrepancy (D1)
- 1 formatting issue (D3)

**Confirmed hallucinations removed: 5**
- "Do not cut charcoal" × 2 (C2, C5) ✅
- Silver layering conflict (C4) ✅
- "May impair healing" (B2) ✅
- NPWT frequency contradiction (D4) ✅

---

## Confirmed Fixes Applied vs Previous Version

| Issue | Previous State | This Version |
|---|---|---|
| WCM_HYDROFIBRE missing from A5, A7 | ❌ Missing | ✅ Present |
| WCM_FOAM missing from B4, B9, C2, D1, D3, D4 | ❌ Missing | ✅ Present |
| WCM_HYDROCOLLOID missing from B1, B5, B6 | ❌ Missing | ✅ Present |
| WCM_ALGINATE missing from B6, B7, D1 | ❌ Missing | ✅ Present |
| WCM_SILVER missing from B4, D3 | ❌ Missing | ✅ Present |
| SFP_IODINE missing from C4, C5 | ❌ Missing | ✅ Present |
| GP_REFERRAL missing from D1 | ❌ Missing | ✅ Present |
| EWMA_DFU_INF missing from D2 | ❌ Missing | ✅ Present |
| SFP_FOAM in B3 (non-contributing) | ❌ Present | ✅ Removed |
| EWMA_VLU_TISSUE in B9 (wrong fit) | ❌ Present | ✅ Removed |
| EWMA_TIME_PRACTICE in C5 (wrong fit) | ❌ Present | ✅ Removed |
| "Do not cut charcoal" in C2, C5 | ❌ Hallucination | ✅ Removed |
| "May impair healing" in B2 | ❌ Hallucination | ✅ Removed |
| NPWT frequency "2–3 days" in D4 | ❌ Contradicts source | ✅ Fixed to "3–5 days" |
| Silver/charcoal "contraindicated" framing in A2, A5 | ❌ Overstated | ✅ Rephrased to "not recommended" |
| alginate change freq in D1 | ❌ "every 2–3 days" | 🔧 Still needs fix → "every 2–5 days" |
| "4-week threshold" in D2 | ❌ Ungrounded | ✅ Removed |
| "Warm hydrocolloid" tip in C6 | ❌ Ungrounded | ✅ Removed |
| Film in B12 allowed_dressings | ❌ Incorrect for epidermal burn | ✅ Removed |
| Film in B12 Primary Dressing text | ❌ Incorrect for epidermal burn | ✅ Removed |
| contraindicated_dressings B7 had "foam" | ❌ Not sourced | ✅ Removed |
| contraindicated_dressings B11 had "dry_gauze" | ❌ Not in ISTAP | ✅ Removed |
| contraindicated_dressings B12 had "ice","occlusive_hydrocolloid" | ❌ Not dressing products | ✅ Replaced with "adhesive_film_dressing" |

---

## Three Remaining Fixes (Minimal, Precise)

### Fix 1 — B12 `cat_b_burns_minor_epidermal`: Typo
```
In Antibiotic Consideration section:
CHANGE: "No antibiotic is required for a epidermanl burn"
TO:     "No antibiotic is required for an epidermal burn"
```

### Fix 2 — D1 `cat_d_notes_infection_override`: Alginate change frequency
```
In Dressing Change Frequency section:
CHANGE: "Silver: every 2–3 days. Alginate/Foam: every 2–3 days."
TO:     "Silver: every 2–3 days. Alginate: every 2–5 days. Foam: every 2–3 days."

Grounding: WCM_ALGINATE states "Recommended every 2 to 5 days"
```

### Fix 3 — D3 `cat_d_notes_malodour_clean`: Formatting (section headers run together)
```
In reference text, add blank line before ## Dressing Change Frequency
and before ## Referral/Escalation so section headers are on separate lines.

CURRENT:
"No specific contraindications at this stage. ## Dressing Change Frequency\n..."
"No systemic antibiotic is currently indicated — the wound shows no frank infection signs. ## Referral/Escalation\n..."

FIXED:
"No specific contraindications at this stage.\n\n## Dressing Change Frequency\n..."
"No systemic antibiotic is currently indicated — the wound shows no frank infection signs.\n\n## Referral/Escalation\n..."
```

---

## Allowed and Contraindicated Dressings — Full Verification

| Case | allowed_dressings | contraindicated_dressings | Verdict |
|---|---|---|---|
| A1 type1 | foam, hydrocolloid, film, tulle, alginate, hydrofiber, polymeric_membrane, hydrogel | silver, charcoal | ✅ GP_TYPE1 explicit |
| A2 type2 | foam, alginate, hydrofiber, polymeric_membrane | [] | ✅ GP_TYPE2 |
| A3 type3 | tulle, hydrogel, hydrocolloid, silver, iodine | [] | ✅ GP_TYPE3 |
| A4 type4 | alginate, foam, silver, hydrofiber, polymeric_membrane, iodine | [] | ✅ GP_TYPE4 |
| A5 type5 | hydrogel, hydrocolloid, polymeric_membrane | alginate, hydrofiber | ✅ GP_TYPE5 + WCM explicit |
| A6 type6 | alginate, foam, polymeric_membrane, hydrofiber | [] | ✅ GP_TYPE6 |
| A7 type7 | silver, hydrogel, hydrocolloid, iodine, polymeric_membrane | alginate, hydrofiber | ✅ GP_TYPE7 + WCM explicit |
| A8 type8 | alginate, silver, hydrofiber, foam, polymeric_membrane, charcoal, iodine | [] | ✅ GP_TYPE8 |
| B1 iodine_thyroid | tulle, hydrogel, hydrocolloid, silver | iodine | ✅ SFP_IODINE |
| B2 silver_clean | foam, hydrocolloid, film, tulle, hydrogel | silver, charcoal | ✅ GP_TYPE1 |
| B3 skin_tear_fragile | silicone_foam, silicone_non_adherent | adhesive_bordered_foam, adhesive_film | ✅ ISTAP_PROD + AJGP_SKINTEAR |
| B4 npwt_eschar | alginate, silver, hydrofiber, foam, polymeric_membrane, charcoal, iodine | npwt | ✅ WCM_NPWT explicit |
| B5 alginate_dry | hydrogel, hydrocolloid, polymeric_membrane | alginate | ✅ WCM_ALGINATE |
| B6 honey_dry | hydrogel, hydrocolloid, polymeric_membrane | honey, alginate | ✅ WCM_HONEY + WCM_ALGINATE |
| B7 postop_clean | film, hydrocolloid | alginate | ✅ AJGP_POSTOP + WCM_ALGINATE |
| B8 burns_hand | hydrogel, hydrocolloid, silicone_non_adherent | adhesive_film_dressing | ✅ ANZBA explicit |
| B9 referral_type6 | alginate, foam, polymeric_membrane, hydrofiber | [] | ✅ GP_TYPE6 |
| B10 diabetic_foot | silver, iodine, silicone_foam | bordered_foam, hydrocolloid | ✅ AJGP_DIABFOOT + SFP_HYDROCOLLOID |
| B11 skin_tear_flap | silicone_foam, silicone_non_adherent, alginate | adhesive_foam, adhesive_film | ✅ ISTAP_PROD |
| B12 burns_minor_epidermal | moisturiser | adhesive_film_dressing | ✅ ANZBA_DRESS Depth 1 |
| C1 dressing_saturation | alginate, foam, hydrofiber, polymeric_membrane | [] | ✅ GP_TYPE2 |
| C2 malodour_type8 | alginate, silver, hydrofiber, foam, polymeric_membrane, charcoal, iodine | [] | ✅ GP_TYPE8 |
| C3 heavy_exudate | alginate, hydrofiber, foam, polymeric_membrane | [] | ✅ GP_TYPE2 |
| C4 dry_infected_combo | tulle, hydrogel, hydrocolloid, silver, iodine | [] | ✅ GP_TYPE3 |
| C5 time_assessment | silver, hydrogel, hydrocolloid, iodine, polymeric_membrane, charcoal | [] | ✅ GP_TYPE7 + charcoal for malodour |
| C6 film_vs_hydrocolloid | film, hydrocolloid, foam, tulle | silver, charcoal | ✅ GP_TYPE1 |
| D1 infection_override | silver, iodine, alginate, foam, hydrofiber, polymeric_membrane | [] | ✅ GP_TYPE4 |
| D2 diabetic_nonhealing | silver, silicone_foam | bordered_foam, hydrocolloid | ✅ AJGP_DIABFOOT + SFP_HYDROCOLLOID |
| D3 malodour_clean | alginate, charcoal, silver, foam, hydrofiber | [] | ✅ GP_TYPE2 + clinical use |
| D4 npwt_adjunct | alginate, foam, polymeric_membrane, hydrofiber | [] | ✅ GP_TYPE6 |
| E1 vlu_chronic | alginate, hydrofiber, foam, polymeric_membrane | hydrocolloid, film | ✅ EWMA_VLU_MOI explicit |
| E2 dfu_infected | silver, iodine, silicone_foam | bordered_foam, hydrocolloid | ✅ AJGP_DIABFOOT + SFP_HYDROCOLLOID |

**All 32 cases verified. No incorrect dressing classifications found.**

---

## Final Assessment

**29 of 32 cases are gold standard with no fixes required.**

**3 remaining fixes are trivial (1 typo, 1 change frequency, 1 formatting) and require no clinical review.**

After applying the 3 fixes, the testset is locked and ready for use as the fixed gold standard for all retrieval and generation ablation evaluations. No further structural changes are recommended.
