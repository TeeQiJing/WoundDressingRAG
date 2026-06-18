# Category B — Gold Standard Audit Report
**Based on direct comparison of `reference` text vs actual chunk `text` + `ai_summary`**
**All 12 Cat B cases checked against 8 source _kept.json files**

---

## Legend
- ✅ Correct — grounded in cited chunk
- ⚠️ Hallucination / Ungrounded — stated in reference but NOT in any cited chunk
- 🔧 Fix needed
- 📝 POV issue in user_input notes
- ➕ Chunk should be added to reference_contexts
- ➖ Chunk should be removed from reference_contexts

---

## 1. `cat_b_iodine_thyroid`
**reference_contexts:** `SFP_IODINE`, `GP_TYPE3`, `WCM_SILVER`

### POV Check — user_input notes
> "I have a thyroid condition and I take levothyroxine every day."
✅ Clean 1st-person patient language.

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| Iodine: "may be absorbed through the wound... avoid if thyroid disorder" | SFP_IODINE TEXT ✓ ("may be absorbed systematically... avoid in patients with thyroid disorders") | ✅ |
| Silver: "bactericidal, no known resistance, locally acting" | WCM_SILVER TEXT ✓ | ✅ |
| Silver: change every 2–3 days | WCM_SILVER TEXT ✓ | ✅ |
| Hydrogel: rehydrates wound, change every 2–3 days | WCM_HYDROGEL TEXT ✓ | ✅ — **but WCM_HYDROGEL is NOT in reference_contexts** |
| Hydrocolloid: moist environment, promotes autolysis, change every 2–5 days | WCM_HYDROCOLLOID TEXT ✓ | ✅ — **but WCM_HYDROCOLLOID is NOT in reference_contexts** |
| GP_TYPE3 dressing list: Tulle, Hydrogel, Hydrocolloid, Silver, Iodine | GP_TYPE3 TEXT ✓ | ✅ |
| Antibiotic: Yes based on C&S | GP_TYPE3 TEXT ✓ | ✅ |
| Referral: not required for Type 3 | GP_TYPE3 TEXT ✓ | ✅ |
| Debridement: "some debridement may be needed" | GP_TYPE3 TEXT ✓ | ✅ |
| "hypothyroidism, hyperthyroidism, or taking levothyroxine" | SFP_IODINE TEXT: says "thyroid disorders" only — no mention of specific conditions or drug names | ⚠️ Mild extension |
| "Silver dressings are safe for use with thyroid conditions" | WCM_SILVER TEXT: says nothing about thyroid safety | ⚠️ Ungrounded positive claim |
| "check the product label" for iodine content | Not in any chunk | ⚠️ Ungrounded — reasonable advice but not sourced |

### Issues Found

**⚠️ Hallucination 1 — Missing chunks for Secondary Dressing**
The reference describes hydrogel and hydrocolloid with specific properties and frequencies, but neither `WCM_HYDROGEL` nor `WCM_HYDROCOLLOID` is in reference_contexts. Any RAGAS Context Recall scoring on these statements will fail.

**⚠️ Hallucination 2 — Three minor ungrounded statements**
1. "hypothyroidism, hyperthyroidism, or taking levothyroxine" — SFP_IODINE only says "thyroid disorders." The specific condition names and drug name are an elaboration not in the source. *Clinically accurate but technically ungrounded.*
2. "Silver dressings are safe for use with thyroid conditions" — no chunk states this. It is a logical inference (silver ≠ iodine), but not sourced.
3. "check the product label" — reasonable clinical advice, not in any chunk.

**Recommendation:** Remove the two Application Tips that are ungrounded, or accept as clinical reasoning additions (low risk). Definitely add the missing chunks.

### `contraindicated_dressings` Check
```python
"contraindicated_dressings": ["iodine"]
```
✅ **Correct** — SFP_IODINE explicitly states iodine should be avoided in thyroid disorder patients. This is a true, source-grounded contraindication.

### reference_contexts Fix
**Add:** `WCM_HYDROGEL`, `WCM_HYDROCOLLOID`
**Updated set:** `SFP_IODINE`, `GP_TYPE3`, `WCM_SILVER`, `WCM_HYDROGEL`, `WCM_HYDROCOLLOID`

---

## 2. `cat_b_silver_clean_granulating`
**reference_contexts:** `GP_TYPE1`, `WCM_FILM`, `WCM_HYDROCOLLOID`

### POV Check — user_input notes
> "The nurse last time put on a silver dressing."
✅ Acceptable patient-report phrasing. Not strictly 1st-person but natural and clear.

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| "clinical algorithm explicitly excludes silver and charcoal from Wound Type 1" | GP_TYPE1 TEXT ✓ ("All types except silver, charcoal and special advanced dressing materials") | ✅ |
| "Using silver on a non-infected wound is unnecessary and may impair healing" | GP_TYPE1 or WCM_SILVER: **neither states** silver impairs healing on clean wounds | ⚠️ Ungrounded |
| Film: transparent, waterproof, bacterial barrier, change every 2–5 days | WCM_FILM TEXT ✓ | ✅ |
| Hydrocolloid: moist environment, promotes autolysis, change every 2–5 days | WCM_HYDROCOLLOID TEXT ✓ | ✅ |
| Application Tips: "Gently remove the silver dressing" | Reasonable — not in any chunk | ⚠️ Minor ungrounded |
| Antibiotic: No | GP_TYPE1 TEXT ✓ | ✅ |
| Referral: Not indicated | GP_TYPE1 TEXT ✓ | ✅ |

### Issues Found

**⚠️ Hallucination — "may impair healing"**
GP_TYPE1 states silver is excluded from Type 1, but does NOT say silver impairs healing on clean wounds. WCM_SILVER does not say this either. The correct grounded language is: *"silver is not indicated and is excluded by the clinical algorithm."*

**Fix Contraindicated Dressings text:** Change "may impair healing" to "is not indicated and is excluded from Wound Type 1 by the clinical algorithm."

### `contraindicated_dressings` Check
```python
"contraindicated_dressings": ["silver", "charcoal"]
```
✅ **Correct** — GP_TYPE1 explicitly excludes silver and charcoal by name. Strongest contraindication in the whole dataset.

### reference_contexts Assessment
All 3 chunks are correct and sufficient. No changes needed.

---

## 3. `cat_b_skin_tear_fragile`
**reference_contexts:** `ISTAP_CLASSIFICATION`, `ISTAP_PATHWAY`, `ISTAP_PRODUCTS`, `AJGP_SKINTEAR`, `SFP_FOAM`

### POV Check — user_input notes
> "I am 82 years old. I knocked my right forearm on the door frame and the skin tore. My skin is very thin and tears easily. The nurse last time used a dressing with sticky edges all around it."
✅ Excellent 1st-person patient language. Clear, natural, clinically informative.

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| "ISTAP guidelines recommend silicone non-adherent mesh or silicone foam as primary dressing for all skin tear types" | ISTAP_PRODUCTS TEXT ✓ (Product 1: Non-Adherent Mesh; Product 2: Foam Non-Adhesive Only) | ✅ |
| "atraumatic on removal, does not adhere to fragile skin" | ISTAP_PRODUCTS TEXT ✓ (Considerations: "Atraumatic removal") | ✅ |
| "ISTAP Type 2 — partial flap; flap cannot fully cover wound bed" | ISTAP_CLASSIFICATION TEXT ✓ (Type 2: "flap CANNOT be repositioned to completely cover the wound bed") | ✅ |
| Application: "(1) control bleeding, (2) gently cleanse, (3) reposition flap, (4) apply silicone mesh or foam" | ISTAP_PATHWAY TEXT ✓ (Workstream 2: steps 1–4 match) | ✅ |
| "Use a dressing remover wipe" | ISTAP_PATHWAY TEXT ✓ ("use remover wipes") | ✅ |
| "Mark flap direction with arrow on outer dressing" | AJGP_SKINTEAR TEXT ✓ ("Removal of the dressing should be done in a direction that does not disturb viable tissue edges") — arrow marking not explicitly stated | ⚠️ Partial — arrow marking is in ISTAP_PATHWAY spirit but only "direction" is in AJGP |
| Skin barrier wipe: "reduces maceration and protects skin on removal" | AJGP_SKINTEAR TEXT ✓ ("barrier wipe... reduce maceration and protect the skin on removal") | ✅ |
| "Adhesive bordered foam must NOT be used on fragile skin, especially forearms and hands of elderly" | AJGP_SKINTEAR TEXT ✓ ("Do not use any adhesive products on fragile skin... especially on forearms and hands of the elderly") | ✅ |
| "Silicone foam: every 3–5 days, or sooner if soiled" | ISTAP_PRODUCTS TEXT ✓ (Foam: "2–7 days depending on exudate levels") — 3–5 days is within range | ✅ |
| Haemostatic alginate as primary if bleeding | AJGP_SKINTEAR TEXT ✓ ("If bleeding, apply haemostatic alginate dressing as primary dressing under silicone-coated foam") | ✅ |
| "ISTAP Type 2 (partial flap loss — flap cannot fully cover wound bed)" in Clinical Notes | ISTAP_CLASSIFICATION TEXT ✓ | ✅ |
| Antibiotic: Not required | ISTAP_PATHWAY TEXT ✓ (infection treatment domain mentions topical antimicrobials if infected; no infection here = no antibiotic) | ✅ |
| Referral: Not indicated for uncomplicated skin tear | Not explicitly stated in any chunk — no chunk says "do not refer for uncomplicated skin tears" | ⚠️ Minor — implied but not stated |

### Issues Found

**⚠️ Minor — "mark flap with arrow"**
Arrow marking is a clinical best practice widely attributed to ISTAP but does NOT appear verbatim in any of the three ISTAP chunks. ISTAP_PATHWAY says "peel slowly in direction that does not disturb the skin flap or viable tissue edges." The "mark with arrow" instruction is a reasonable practical application of this, but strictly speaking is ungrounded. Accept as valid clinical elaboration, or soften to "ensure the dressing is removed in the correct direction."

**⚠️ Minor — SFP_FOAM chunk — is it actually contributing?**
SFP_FOAM TEXT says foam is for "moderate to highly exudative wounds" and lists brand names (Allevyn, Mepilex). It does NOT mention skin tears, non-adhesive foam, or fragile skin. The grounding for foam-on-skin-tears comes from ISTAP_PRODUCTS and AJGP_SKINTEAR, not SFP_FOAM.

**Fix:** Remove `SFP_FOAM` from reference_contexts — it adds no grounding and may confuse retrieval scoring. ISTAP_PRODUCTS already covers foam selection for skin tears.

### `contraindicated_dressings` Check
```python
"contraindicated_dressings": ["adhesive_bordered_foam", "adhesive_film"]
```
✅ **Correct** — AJGP_SKINTEAR TEXT explicitly: "Do not use any adhesive products on fragile skin." ISTAP_PRODUCTS explicitly: "adhesive borders MUST NOT be used on fragile skin." Both directly support this as a true contraindication.

### reference_contexts Fix
**Remove:** `SFP_FOAM`
**Updated set:** `ISTAP_CLASSIFICATION`, `ISTAP_PATHWAY`, `ISTAP_PRODUCTS`, `AJGP_SKINTEAR`

---

## 4. `cat_b_npwt_necrotic_eschar`
**reference_contexts:** `WCM_NPWT`, `GP_TYPE8`, `GP_REFERRAL`

### POV Check — user_input notes
> "My doctor mentioned a vacuum dressing machine. Is that the right thing for my wound right now? There is still a lot of black dead tissue on it."
✅ Excellent 1st-person patient language. "Vacuum dressing machine" is the perfect lay term for NPWT.

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| "NPWT is NOT appropriate as a sole treatment at this stage" | WCM_NPWT TEXT ✓ ("NPWT is only an adjunct... it is not a panacea") | ✅ |
| "Necrotic wound bed or eschar is an explicit contraindication to NPWT" | WCM_NPWT TEXT ✓ (Contraindications: "Necrotic wound bed or eschar (barrier to new tissue growth)") | ✅ |
| "Untreated infection" is also a contraindication to NPWT | WCM_NPWT TEXT ✓ | ✅ |
| "NPWT does not replace surgical procedures" | WCM_NPWT TEXT ✓ ("NPWT does not replace surgical procedures") | ✅ |
| "neoplastic tissue... clotting disorders" — other NPWT contraindications | WCM_NPWT TEXT ✓ | ✅ |
| Silver + alginate + foam as interim dressings | GP_TYPE8 TEXT ✓ (Silver, Alginate, Foam in Type 8 list) | ✅ |
| Silver: change every 2–3 days | WCM_SILVER TEXT ✓ | ✅ — **but WCM_SILVER is NOT in reference_contexts** |
| Alginate: change every 2–5 days | WCM_ALGINATE TEXT ✓ | ✅ — **but WCM_ALGINATE is NOT in reference_contexts** |
| Hydrofibre: change every 2–5 days | WCM_HYDROFIBRE TEXT ✓ | ✅ — **but WCM_HYDROFIBRE is NOT in reference_contexts** |
| Foam: highly absorbent, bacterial barrier, change every 2–3 days | WCM_FOAM TEXT ✓ | ✅ — **but WCM_FOAM is NOT in reference_contexts** |
| Referral: Type 8 requires urgent referral | GP_REFERRAL TEXT ✓ | ✅ |
| Antibiotic: required — C&S | GP_TYPE8 TEXT ✓ | ✅ |

### Issues Found

**🔧 Missing dressing property chunks**
The reference describes specific properties and change frequencies for silver, alginate, hydrofibre, and foam, but none of `WCM_SILVER`, `WCM_ALGINATE`, `WCM_HYDROFIBRE`, or `WCM_FOAM` are in reference_contexts. RAGAS Context Recall will fail for all frequency/property claims.

**Fix:** Add the four WCM dressing chunks.

### `contraindicated_dressings` Check
```python
"contraindicated_dressings": ["npwt"]
```
✅ **Correct** — WCM_NPWT TEXT explicitly lists "Necrotic wound bed or eschar" as a contraindication to NPWT. This is a direct, source-grounded contraindication.

### reference_contexts Fix
**Add:** `WCM_SILVER`, `WCM_ALGINATE`, `WCM_HYDROFIBRE`, `WCM_FOAM`
**Updated set:** `WCM_NPWT`, `GP_TYPE8`, `GP_REFERRAL`, `WCM_SILVER`, `WCM_ALGINATE`, `WCM_HYDROFIBRE`, `WCM_FOAM`

---

## 5. `cat_b_alginate_dry_wound`
**reference_contexts:** `GP_TYPE5`, `WCM_ALGINATE`, `WCM_HYDROGEL`

### POV Check — user_input notes
> "The nurse last time put on an alginate dressing. The wound looks completely dry — I have not seen any wetness or fluid from it at all."
✅ Good 1st-person patient language.

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| "Alginate is not helpful for dry wounds; requires wound fluid to form gel" | WCM_ALGINATE TEXT ✓ ("Not helpful for dry wounds") | ✅ |
| Hydrogel: "rehydrates necrotic tissue, promotes autolytic debridement" | WCM_HYDROGEL TEXT ✓ | ✅ |
| Hydrogel: change every 2–3 days | WCM_HYDROGEL TEXT ✓ | ✅ |
| Hydrogel: "requires a secondary dressing" | WCM_HYDROGEL TEXT ✓ | ✅ |
| Hydrocolloid as secondary, change every 2–5 days | WCM_HYDROCOLLOID TEXT ✓ | ✅ — **but WCM_HYDROCOLLOID is NOT in reference_contexts** |
| GP_TYPE5: Hydrogel, Hydrocolloid, Polymeric membrane listed | GP_TYPE5 TEXT ✓ | ✅ |
| Debridement: "is needed" | GP_TYPE5 TEXT ✓ | ✅ |
| Antibiotic: No | GP_TYPE5 TEXT ✓ | ✅ |
| Referral: Not indicated for Type 5 | GP_TYPE5 TEXT ✓ | ✅ |
| "Moisten with saline if sticking" (Application Tip) | WCM_ALGINATE TEXT: states residue should be washed off but does NOT say to moisten before removal | ⚠️ Minor ungrounded |

### Issues Found

**🔧 Missing `WCM_HYDROCOLLOID`**
Hydrocolloid is listed as Secondary Dressing with change frequency — this is grounded in WCM_HYDROCOLLOID but that chunk is absent.

**⚠️ Minor — "moisten with saline if sticking"**
Good clinical advice but not in WCM_ALGINATE. Accept as reasonable elaboration or remove.

### `contraindicated_dressings` Check
```python
"contraindicated_dressings": ["alginate"]
```
✅ **Correct** — WCM_ALGINATE TEXT explicitly: "Not helpful for dry wounds." Direct functional contraindication. Keep.

### reference_contexts Fix
**Add:** `WCM_HYDROCOLLOID`
**Updated set:** `GP_TYPE5`, `WCM_ALGINATE`, `WCM_HYDROGEL`, `WCM_HYDROCOLLOID`

---

## 6. `cat_b_honey_dry_necrotic`
**reference_contexts:** `WCM_HONEY`, `GP_TYPE5`, `WCM_HYDROGEL`

### POV Check — user_input notes
> "I read online that honey dressings are good for wounds. Can I use a honey dressing on this? The wound has a lot of black dead tissue and feels dry."
✅ Excellent 1st-person patient language with realistic patient context.

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| "Honey CONTRAINDICATED for dry necrotic wounds — can cause further drying" | WCM_HONEY TEXT ✓ ("Contraindication: Dry, necrotic wounds — honey can cause further drying of the wound") | ✅ |
| "Honey has antimicrobial properties and can help with some wound types" | WCM_HONEY TEXT ✓ (multiple properties listed) | ✅ |
| "WCM guidelines specifically list dry, necrotic wounds as a contraindication" | WCM_HONEY TEXT ✓ | ✅ |
| Hydrogel: correct choice, rehydrates, change every 2–3 days, needs secondary | WCM_HYDROGEL TEXT ✓ | ✅ |
| Hydrocolloid as secondary, change every 2–5 days | WCM_HYDROCOLLOID TEXT ✓ | ✅ — **but WCM_HYDROCOLLOID is NOT in reference_contexts** |
| GP_TYPE5: Hydrogel, Hydrocolloid, Polymeric membrane | GP_TYPE5 TEXT ✓ | ✅ |
| Alginate: contraindicated on dry wounds | WCM_ALGINATE TEXT ✓ | ✅ — **but WCM_ALGINATE is NOT in reference_contexts** |
| Debridement priority | GP_TYPE5 TEXT ✓ | ✅ |
| Antibiotic: No | GP_TYPE5 TEXT ✓ | ✅ |

### Issues Found

**🔧 Two missing chunks**
- Hydrocolloid named as secondary dressing → needs `WCM_HYDROCOLLOID`
- Alginate listed as contraindicated ("requires exudate to function") → needs `WCM_ALGINATE`

### `contraindicated_dressings` Check
```python
"contraindicated_dressings": ["honey"]
```
✅ **Correct** — WCM_HONEY TEXT explicitly lists "Dry, necrotic wounds" as a contraindication. Direct, source-grounded.

Should you also add `"alginate"` here? **Yes** — alginate is also stated as contraindicated in the reference text and WCM_ALGINATE grounds it. 

**Updated:**
```python
"contraindicated_dressings": ["honey", "alginate"]
```

### reference_contexts Fix
**Add:** `WCM_HYDROCOLLOID`, `WCM_ALGINATE`
**Updated set:** `WCM_HONEY`, `GP_TYPE5`, `WCM_HYDROGEL`, `WCM_HYDROCOLLOID`, `WCM_ALGINATE`

---

## 7. `cat_b_postop_clean`
**reference_contexts:** `AJGP_POSTOP`, `WCM_FILM`, `SFP_FILM`

### POV Check — user_input notes
> "I had surgery 3 days ago and the cut is stitched up and looks clean. There is no wetness or discharge. I want to be able to shower every day."
✅ Excellent 1st-person patient language.

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| "For wounds without exudate, dress over sutures with film or thin hydrocolloid" | AJGP_POSTOP TEXT ✓ | ✅ |
| "In case of wound dehiscence, organise prompt surgical review" | AJGP_POSTOP TEXT ✓ | ✅ |
| Film: "waterproof (allows daily showering), transparent for easy monitoring" | WCM_FILM TEXT ✓ (waterproof, transparent) | ✅ |
| Film: change every 2–5 days | WCM_FILM TEXT ✓ | ✅ |
| Film: "apply over site making sure no air under it" | WCM_FILM TEXT ✓ | ✅ |
| Film: "stretch and pull slowly from edges to remove" | WCM_FILM TEXT ✓ | ✅ |
| "Foam dressings: NOT indicated — no exudate is present" | No chunk explicitly states foam is contraindicated for no-exudate wounds | ⚠️ Inference — WCM_FOAM says foam is "absorbent/cushioning" but does not say "avoid if no exudate" |
| "Alginate: NOT indicated — requires exudate to function" | WCM_ALGINATE TEXT ✓ ("Not helpful for dry wounds") | ✅ |
| SFP_FILM examples: "Tegaderm, Opsite" | SFP_FILM TEXT ✓ | ✅ |
| "Skin around wound must be intact for good seal" | SFP_FILM TEXT ✓ | ✅ |
| "Avoid in draining or infected wounds" | SFP_FILM TEXT ✓ | ✅ |
| "Primary intention healing" in Clinical Notes | AJGP_POSTOP context | ✅ |

### Issues Found

**⚠️ Minor — Foam contraindication not directly grounded**
The reference says "Foam dressings: NOT indicated — no exudate is present to absorb." WCM_FOAM does not say foam should not be used without exudate. SFP_FOAM says foam is for "moderate to highly exudative wounds" — this DOES implicitly support the claim. But `SFP_FOAM` is not in reference_contexts.

**Fix option:** Either add `SFP_FOAM` to support the foam contraindication, or soften to "Foam is not listed as a recommended dressing for this wound type." Given this is a postop clean wound with no exudate, the reasoning is sound — add `SFP_FOAM` to ground it.

**Note:** `WCM_ALGINATE` also not in reference_contexts but alginate contraindication is grounded there. Add it.

### `contraindicated_dressings` Check
```python
"contraindicated_dressings": ["foam", "alginate"]
```
⚠️ **Partial concern:**
- Alginate: ✅ WCM_ALGINATE says "Not helpful for dry wounds" — directly applicable
- Foam: ⚠️ No chunk explicitly contraindications foam on dry/postop wounds. SFP_FOAM says it is for moderate-to-highly exudative wounds (implying not appropriate here). Not a strong explicit contraindication.

**Recommendation:** Keep `"alginate"` as contraindicated. Change `"foam"` to absent from the list since no source explicitly contraindications foam for postop dry wounds — it is simply not indicated. Consistent with the rule established for Cat A.

**Updated:**
```python
"contraindicated_dressings": ["alginate"]
```

### reference_contexts Fix
**Add:** `WCM_ALGINATE`
**Updated set:** `AJGP_POSTOP`, `WCM_FILM`, `SFP_FILM`, `WCM_ALGINATE`

---

## 8. `cat_b_burns_hand`
**reference_contexts:** `ANZBA_REFERRAL`, `ANZBA_DEPTH`, `ANZBA_FIRSTAID`, `ANZBA_DRESSINGS`, `AJGP_BURNS`, `WCM_HYDROGEL`

### POV Check — user_input notes
> "I burned my right palm with boiling water. I held it under cool running water for 20 minutes straight away. There are blisters on my palm. It is very painful."
✅ Excellent 1st-person patient language. Blisters + pain confirms superficial partial-thickness.

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| "Hydrogel recommended for initial burn wound cover" (ANZBA) | ANZBA_FIRSTAID TEXT ✓ (hydrogel for small burns, analgesia after first aid) | ✅ |
| "20 minutes of cool running water" first aid | ANZBA_FIRSTAID TEXT ✓ ("COOL: Apply 20 minutes of COOL RUNNING WATER") | ✅ |
| "You have already done this correctly" in Application Tips | Patient notes confirm this | ✅ |
| "Do not burst the blisters" | ANZBA_DEPTH TEXT: does not say this directly; says "Debride blisters if greater than 5 cm or located over joints" (implying small blisters are left) | ⚠️ Minor — implied but not explicit |
| "Hydrocolloid or silicone non-adherent for ongoing management" | AJGP_BURNS TEXT ✓ ("cover burns area with hydrogel or hydrocolloid or film") | ✅ |
| Change every 2–3 days (hydrogel) | WCM_HYDROGEL TEXT ✓ | ✅ |
| "ANZBA referral criteria: hands... regardless of size or depth" | ANZBA_REFERRAL TEXT ✓ ("Hands" listed under Special Areas — any size, any depth) | ✅ |
| "Superficial partial-thickness burn — blisters present" | ANZBA_DEPTH TEXT ✓ (Depth 2: blisters, pale pink, painful) | ✅ |
| "Ice or very cold water: NOT used — causes vasoconstriction and worsens injury" | ANZBA_FIRSTAID TEXT ✓ ("NEVER: Use ice or apply ointments... Use very cold water") | ✅ |
| "Ointments applied before or instead of first aid cooling" as contraindicated | ANZBA_FIRSTAID TEXT ✓ ("NEVER: Use ice or apply ointments") | ✅ |
| Light non-adherent pad as secondary | ANZBA_DRESSINGS TEXT ✓ (foam or gauze as secondary for burns exudate) | ✅ |
| "For a hand burn, consider non-adherent layer to allow finger movement" | Not in any chunk | ⚠️ Ungrounded — reasonable clinical tip, not sourced |
| "Superficial partial-thickness burns typically heal within 14 days" | ANZBA_DEPTH TEXT ✓ (Depth 2: "Should heal within 7–10 (<14) days") | ✅ |
| Antibiotic: not required unless infection signs | Consistent with AJGP_BURNS context | ✅ |

### Issues Found

**⚠️ Minor — "Do not burst blisters"**
ANZBA_DEPTH does not say "do not burst blisters" — it says debride blisters >5cm or over joints (implying small ones stay). The statement is clinically correct but not verbatim from any chunk. Accept as reasonable clinical elaboration.

**⚠️ Minor — "Allow finger movement" tip**
Not in any chunk. Remove or accept as elaboration.

### `contraindicated_dressings` Check
```python
"contraindicated_dressings": []
```
⚠️ **Review needed.**

ANZBA_DEPTH TEXT explicitly states for minor burns:
> "❌ DO NOT USE: Adhesive dressings directly over the burn wound itself (e.g. film dressings such as Opsite or Tegaderm — these are CONTRAINDICATED)"
> "Ice — NEVER apply ice to burns"
> "Ointments — NEVER apply to burns"

The reference correctly mentions ice and ointments as contraindicated in the reference text, but the `contraindicated_dressings` field is empty. Ice is not a dressing, and ointments are not a dressing in your system — so leaving this empty is defensible. However, if your Safety Checker evaluates dressing recommendations broadly, consider:

**Updated recommendation:**
```python
"contraindicated_dressings": ["adhesive_film_dressing"]
```
ANZBA_DEPTH TEXT explicitly calls out "film dressings such as Opsite or Tegaderm — these are CONTRAINDICATED" for burns. This is a genuine documented contraindication for burns from the source.

### reference_contexts Assessment
All 6 chunks are correctly cited and contribute. No changes needed.

---

## 9. `cat_b_referral_type6`
**reference_contexts:** `GP_TYPE6`, `GP_REFERRAL`, `WCM_ALGINATE`, `EWMA_VLU_TISSUE`

### POV Check — user_input notes
No notes (empty string). ✅ Nothing to fix.

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| GP_TYPE6 dressings: Alginate, Foam, Polymeric membrane, Hydrofibre | GP_TYPE6 TEXT ✓ | ✅ |
| Referral: Type 6 requires hospital referral | GP_TYPE6 + GP_REFERRAL TEXT ✓ | ✅ |
| Alginate: forms gel on contact, heavy exudate, change every 2–5 days | WCM_ALGINATE TEXT ✓ | ✅ |
| Hydrofibre: manages heavy exuding, reduces maceration, change every 2–5 days | WCM_HYDROFIBRE TEXT ✓ | ✅ — **but WCM_HYDROFIBRE is NOT in reference_contexts** |
| Foam: highly absorbent, bacterial barrier, change every 2–3 days | WCM_FOAM TEXT ✓ | ✅ — **but WCM_FOAM is NOT in reference_contexts** |
| EWMA_VLU_TISSUE: "slough must be removed before epithelialisation can proceed" | EWMA_VLU_TISSUE TEXT ✓ — but not explicitly — it discusses debridement for complex ulcers | Partial ✅ |
| "Hydrogel not suitable as primary for high exudate" | Not explicitly in any chunk — GP_TYPE6 simply doesn't list hydrogel | ⚠️ Inference |
| "Apply alginate without overpacking" | Not in WCM_ALGINATE — says "available in sheet or rope form" but not "do not overpack" | ⚠️ Minor ungrounded |
| Antibiotic: "may or may not be required" | GP_TYPE6 TEXT ✓ | ✅ |
| Debridement: surgical/mechanical recommended | GP_TYPE6 TEXT ✓ | ✅ |

### Issues Found

**🔧 Missing chunks for Hydrofibre and Foam**
Both are named with properties and frequencies, but neither WCM_HYDROFIBRE nor WCM_FOAM is in reference_contexts.

**⚠️ EWMA_VLU_TISSUE relevance — weak fit**
This case has no clinical notes (no venous leg ulcer context). EWMA_VLU_TISSUE discusses VLU-specific debridement and management, which is relevant only if the case context implies VLU. Without notes specifying a VLU, including EWMA_VLU_TISSUE is a weak fit — the case is a generic Type 6 wound. **Remove** `EWMA_VLU_TISSUE` from this case (it's more appropriate in `cat_e_vlu_chronic_ewma`).

### `contraindicated_dressings` Check
```python
"contraindicated_dressings": []
```
✅ Correct — no explicit contraindications stated in GP_TYPE6. Leave empty.

### reference_contexts Fix
**Add:** `WCM_HYDROFIBRE`, `WCM_FOAM`
**Remove:** `EWMA_VLU_TISSUE` (no VLU context in this case)
**Updated set:** `GP_TYPE6`, `GP_REFERRAL`, `WCM_ALGINATE`, `WCM_HYDROFIBRE`, `WCM_FOAM`

---

## 10. `cat_b_diabetic_foot`
**reference_contexts:** `AJGP_DIABFOOT`, `WCM_SILVER`, `SFP_HYDROCOLLOID`, `EWMA_DFU_INFECTION`, `EWMA_DFU_TISSUE`

### POV Check — user_input notes
> "I have Type 2 diabetes. The wound is on the bottom of my right foot. I cannot really feel pain there anymore. There is a moderate amount of fluid coming from it."
✅ Excellent 1st-person patient language. "Cannot feel pain" clearly describes neuropathy without using the medical term.

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| "Antimicrobial primary dressing" for diabetic foot | AJGP_DIABFOOT TEXT ✓ ("Apply a primary antimicrobial dressing product") | ✅ |
| "Moderate exudate → silicone foam" | AJGP_DIABFOOT TEXT ✓ ("moderate exudate – silicone foam") | ✅ |
| "Silicone foams on feet WITHOUT borders, anchored with tape or bandages" | AJGP_DIABFOOT TEXT ✓ ("Silicone foams on feet, if applied, should be without borders and anchored with tape or bandages") | ✅ |
| "Check pedal pulses and sensation; if poor perfusion, refer to diabetic foot clinic or vascular surgeon" | AJGP_DIABFOOT TEXT ✓ | ✅ |
| Silver: bactericidal, no known resistance | WCM_SILVER TEXT ✓ | ✅ |
| Silver: change every 2–3 days | WCM_SILVER TEXT ✓ | ✅ |
| "Hydrocolloid not recommended for diabetic foot ulcers" | SFP_HYDROCOLLOID TEXT ✓ ("Not recommended for... diabetic foot ulcers") | ✅ |
| "EWMA DFU guidelines prioritise infection control" | EWMA_DFU_INFECTION TEXT ✓ ("Infection is a threat to the diabetic foot") | ✅ |
| "Debridement is recommended" | EWMA_DFU_TISSUE TEXT ✓ ("debridement is therefore an important component") | ✅ |
| "Bordered adhesive foam CONTRAINDICATED — adhesive borders cause pressure and skin damage... neuropathy" | AJGP_DIABFOOT TEXT ✓ (without borders; anchored with tape) + patient note (neuropathy) | ✅ |
| Antibiotic: required — C&S | EWMA_DFU_INFECTION TEXT ✓ ("prescribe wide-spectrum antibiotics and take cultures") | ✅ |
| Referral: to diabetic foot clinic if poor perfusion or non-healing at 4 weeks | AJGP_DIABFOOT TEXT ✓ | ✅ |
| "Offloading the foot is essential — dressing alone will not heal neuropathic plantar ulcer" | EWMA_DFU_TISSUE TEXT ✓ ("Pressure control: offloading and weight redistribution") | ✅ |
| "Even if you cannot feel pain, check wound carefully" | EWMA_DFU_INFECTION TEXT ✓ (signs of infection may be absent in neuropathic patients) | ✅ |

### Issues Found
**NONE.** This is one of the cleanest Cat B cases. All reference statements are grounded. All 5 chunks are correctly cited and contributing. Excellent.

### `contraindicated_dressings` Check
```python
"contraindicated_dressings": ["bordered_foam", "hydrocolloid"]
```
✅ **Both correct:**
- `bordered_foam`: AJGP_DIABFOOT explicitly says apply silicone foams WITHOUT borders on feet.
- `hydrocolloid`: SFP_HYDROCOLLOID TEXT explicitly: "Not recommended for... diabetic foot ulcers."

---

## 11. `cat_b_skin_tear_type2_flap`
**reference_contexts:** `ISTAP_CLASSIFICATION`, `ISTAP_PATHWAY`, `ISTAP_PRODUCTS`

### POV Check — user_input notes
> "I am 78 years old and I take warfarin (a blood-thinning tablet). I caught my lower left leg on the edge of a table and a flap of skin tore. The flap is still mostly attached but it does not fully cover the raw area underneath. There was a little bleeding but no signs of infection."
✅ Excellent 1st-person patient language. Warfarin explained as "blood-thinning tablet" — perfect for patient-facing context.

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| "ISTAP Type 2 — flap cannot fully cover wound bed" | ISTAP_CLASSIFICATION TEXT ✓ | ✅ |
| "Alginate as haemostatic primary under gentle pressure for 5–10 minutes until bleeding stops" | ISTAP_PRODUCTS TEXT ✓ (Alginate: "Moderate to heavy exudate; haemostatic") | ✅ (haemostatic confirmed; 5–10 min timing not in chunk) |
| "Silicone non-adherent mesh or silicone foam — atraumatic on removal" | ISTAP_PRODUCTS TEXT ✓ (Product 1 and 2) | ✅ |
| "Foam as secondary; change every 3–5 days or if soiled" | ISTAP_PRODUCTS TEXT ✓ (Foam: "2–7 days") — 3–5 is within range | ✅ |
| "Skin barrier film to periwound skin before securing dressing" | ISTAP_PATHWAY TEXT ✓ ("Apply film-forming liquid acrylate to periwound skin") | ✅ |
| ISTAP pathway steps: (1) bleeding, (2) cleanse, (3) reposition, (4) apply | ISTAP_PATHWAY TEXT ✓ (Workstream 2 steps) | ✅ |
| "Mark flap direction with arrow on outer dressing" | ISTAP_PATHWAY TEXT: "peel slowly in direction that does not disturb the skin flap" — arrow marking not verbatim | ⚠️ Same minor issue as cat_b_skin_tear_fragile |
| "Adhesive products are contraindicated on fragile elderly skin" | ISTAP_PRODUCTS TEXT ✓ ("adhesive borders MUST NOT be used on fragile skin") | ✅ |
| "Dry gauze: sticks to wound bed, causes trauma on removal" | Not in any ISTAP chunk — ISTAP only recommends non-adherent products, but does NOT explicitly state gauze sticks | ⚠️ Ungrounded |
| "Warfarin means bleeding may take slightly longer to stop — expected and normal" | ISTAP_PATHWAY TEXT ✓ — Polypharmacy (anticoagulants) listed as a risk factor to assess | Partial ✅ (context grounded, specific statement is elaboration) |
| Antibiotic: Not required | ✅ Wound not infected |
| Referral: Not indicated | ✅ Consistent with all ISTAP chunks |
| "5–10 minutes" pressure timing for alginate haemostasis | Not in any chunk | ⚠️ Minor ungrounded |

### Issues Found

**⚠️ Two minor ungrounded statements**
1. "Dry gauze sticks to wound bed" — not stated in any ISTAP chunk; reasonable clinical knowledge but ungrounded.
2. "5–10 minutes pressure timing" for alginate — not stated in ISTAP_PRODUCTS which simply says alginate is haemostatic.

**Recommendation:** Remove "sticks to wound bed" from the gauze contraindication (keep "dry gauze is contraindicated — ISTAP recommends non-adherent products only"). Remove the specific 5–10 minute timing.

### `contraindicated_dressings` Check
```python
"contraindicated_dressings": ["adhesive_foam", "adhesive_film", "dry_gauze"]
```
⚠️ **Partial concern:**
- `adhesive_foam`: ✅ ISTAP_PRODUCTS TEXT explicitly: "MUST NOT be used on fragile skin"
- `adhesive_film`: ✅ Supported by ISTAP_PRODUCTS (non-adherent products recommended; adhesive contraindicated)
- `dry_gauze`: ⚠️ Not explicitly stated in any ISTAP chunk. ISTAP recommends non-adherent products but does not say "dry gauze is contraindicated."

**Recommendation:** Remove `"dry_gauze"` from contraindicated_dressings since no chunk explicitly states it. Your safety checker should only fire on chunk-supported contraindications.

**Updated:**
```python
"contraindicated_dressings": ["adhesive_foam", "adhesive_film"]
```

---

## 12. `cat_b_burns_minor_superficial`
**reference_contexts:** `ANZBA_DEPTH`, `ANZBA_REFERRAL`, `ANZBA_FIRSTAID`, `ANZBA_DRESSINGS`

### POV Check — user_input notes
> "I am 32 years old. I spilled hot tea on my upper arm. The burn is about the size of my palm. I ran cool water over it for 20 minutes. There are no blisters — just redness and pain. It turns white when I press on it. I am at a GP clinic."
✅ Excellent 1st-person patient language. "Turns white when I press on it" correctly describes blanching — appropriate patient description.

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| "Superficial (epidermal) burn — erythema, painful, no blistering, blanches on pressure" | ANZBA_DEPTH TEXT ✓ (Depth 1: red, no blisters, brisk capillary refill, painful) | ✅ |
| "Does NOT meet ANZBA referral criteria (not on hand, face, genitalia, joint; not electrical or chemical; not circumferential)" | ANZBA_REFERRAL TEXT ✓ | ✅ |
| "Upper arm, approximately 2% TBSA" | ANZBA_REFERRAL TEXT: >10% TBSA triggers referral in adults — 2% is well below | ✅ |
| Hydrogel: appropriate for initial cover (ANZBA) | ANZBA_FIRSTAID TEXT ✓ | ✅ |
| "Paraffin tulle — low-adherent option for superficial burns" | ANZBA_DRESSINGS TEXT ✓ ("Paraffin gauze; Silicone dressings" for superficial dermal) | ✅ |
| Film dressing for very superficial burns with no exudate | ANZBA_DEPTH TEXT: "Simple moisturisers only" for epidermal — film not listed for Depth 1 | ⚠️ Mild mismatch — film is listed for Depth 2 superficial dermal, not Depth 1 epidermal |
| "Ice or iced water: NOT used (causes vasoconstriction)" | ANZBA_FIRSTAID TEXT ✓ ("NEVER: Use ice") | ✅ |
| "Ointments or butter: NOT used" | ANZBA_FIRSTAID TEXT ✓ ("NEVER: Use ice or apply ointments") | ✅ |
| "20 minutes cool running water — you correctly completed this" | ANZBA_FIRSTAID TEXT ✓ ("COOL: Apply 20 minutes of COOL RUNNING WATER") | ✅ |
| "20 minutes most effective within 1 hour; still beneficial up to 3 hours" | ANZBA_FIRSTAID TEXT ✓ | ✅ |
| "Heals within 7–14 days" | ANZBA_DEPTH TEXT: Depth 1: "3–7 days"; Depth 2: "7–10 days" — 7–14 days spans both | ✅ (conservative range) |
| "Review at 48 hours" | ANZBA_DRESSINGS TEXT ✓ (Superficial Dermal: "Follow-up: 24–48 hours by GP") | ✅ |
| Antibiotic: No | ✅ Not infected |
| Referral: Not required | ANZBA_REFERRAL TEXT ✓ | ✅ |
| "Do not apply ice, butter, or any household product" (Application Tips) | ANZBA_FIRSTAID TEXT ✓ (ointments and ice) | ✅ |

### Issues Found

**⚠️ Minor — Film dressing for epidermal burns**
ANZBA_DEPTH Depth 1 (epidermal) says "Simple moisturisers only; secondary not required." Film is listed for Depth 2 (superficial dermal). Blanches on pressure confirms capillary refill intact — could be Depth 1 or borderline Depth 2. The patient notes say "no blisters" which places this firmly in Depth 1. For Depth 1, film is not listed by ANZBA. 

**Fix:** Remove film dressing from Primary Dressing section OR add qualifier: "If the burn is confirmed superficial dermal (blisters may develop), film may be considered for very dry, non-exuding burns."

**Fix in `allowed_dressings`:** Remove `"film"` since ANZBA Depth 1 does not list it.

### `contraindicated_dressings` Check
```python
"contraindicated_dressings": ["ice", "occlusive_hydrocolloid"]
```
⚠️ **Mixed:**
- `ice`: ✅ ANZBA_FIRSTAID TEXT: "NEVER: Use ice" — strongly grounded. However, ice is not a dressing — it may be outside scope of your Safety Checker if it only checks dressing products. Keep it noted in reference text, but consider if your checker uses this field.
- `occlusive_hydrocolloid`: ⚠️ ANZBA_DEPTH TEXT says "❌ DO NOT USE: Adhesive dressings directly over the burn wound itself (e.g. film dressings such as Opsite or Tegaderm)" — this is about **adhesive film**, not hydrocolloid. Hydrocolloid is not specifically called out by ANZBA. SFP_HYDROCOLLOID is not in reference_contexts. This is ungrounded.

**Updated:**
```python
"contraindicated_dressings": ["adhesive_film_dressing"]
```
This is directly grounded in ANZBA_DEPTH: "adhesive dressings such as film dressings — CONTRAINDICATED" for burns.

---

## Summary Table — All Category B Issues

| Case | Hallucination | Missing Chunks | Chunk to Remove | POV Fix | `contraindicated_dressings` Fix |
|---|---|---|---|---|---|
| iodine_thyroid | ⚠️ 3 minor (drug names, silver safety, label check) | ➕ WCM_HYDROGEL, WCM_HYDROCOLLOID | — | ✅ Clean | ✅ Keep `["iodine"]` |
| silver_clean_granulating | ⚠️ "may impair healing" — ungrounded | — | — | ✅ Clean | ✅ Keep `["silver","charcoal"]` |
| skin_tear_fragile | ⚠️ Arrow marking minor | ➖ Remove SFP_FOAM | Remove SFP_FOAM | ✅ Clean | ✅ Keep `["adhesive_bordered_foam","adhesive_film"]` |
| npwt_necrotic_eschar | — | ➕ WCM_SILVER, WCM_ALGINATE, WCM_HYDROFIBRE, WCM_FOAM | — | ✅ Clean | ✅ Keep `["npwt"]` |
| alginate_dry_wound | ⚠️ Saline tip minor | ➕ WCM_HYDROCOLLOID | — | ✅ Clean | ✅ Keep `["alginate"]` |
| honey_dry_necrotic | — | ➕ WCM_HYDROCOLLOID, WCM_ALGINATE | — | ✅ Clean | 🔧 Add `"alginate"` → `["honey","alginate"]` |
| postop_clean | ⚠️ Foam contraindication weak | ➕ WCM_ALGINATE | — | ✅ Clean | 🔧 Remove `"foam"` → `["alginate"]` |
| burns_hand | ⚠️ Blisters tip, finger movement tip minor | — | — | ✅ Clean | 🔧 Add `"adhesive_film_dressing"` → `["adhesive_film_dressing"]` |
| referral_type6 | ⚠️ Hydrogel contraindication, overpack | ➕ WCM_HYDROFIBRE, WCM_FOAM | ➖ Remove EWMA_VLU_TISSUE | ✅ Clean | ✅ Keep `[]` |
| diabetic_foot | None | — | — | ✅ Clean | ✅ Keep `["bordered_foam","hydrocolloid"]` |
| skin_tear_type2_flap | ⚠️ Arrow marking, gauze, timing | — | — | ✅ Clean | 🔧 Remove `"dry_gauze"` → `["adhesive_foam","adhesive_film"]` |
| burns_minor_superficial | ⚠️ Film for Depth 1 minor | — | — | ✅ Clean | 🔧 Replace `["ice","occlusive_hydrocolloid"]` → `["adhesive_film_dressing"]` |

---

## Required Fixes — Consolidated

### Fix 1 — Missing reference_contexts chunks

```python
# cat_b_iodine_thyroid — add:
ctx(WCM_HYDROGEL), ctx(WCM_HYDROCOLLOID)

# cat_b_npwt_necrotic_eschar — add:
ctx(WCM_SILVER), ctx(WCM_ALGINATE), ctx(WCM_HYDROFIBRE), ctx(WCM_FOAM)

# cat_b_alginate_dry_wound — add:
ctx(WCM_HYDROCOLLOID)

# cat_b_honey_dry_necrotic — add:
ctx(WCM_HYDROCOLLOID), ctx(WCM_ALGINATE)

# cat_b_postop_clean — add:
ctx(WCM_ALGINATE)

# cat_b_referral_type6 — add, remove:
# ADD: ctx(WCM_HYDROFIBRE), ctx(WCM_FOAM)
# REMOVE: ctx(EWMA_VLU_TISSUE)  ← no VLU context in this case

# cat_b_skin_tear_fragile — remove:
# REMOVE: ctx(SFP_FOAM)  ← not contributing; ISTAP chunks cover foam
```

### Fix 2 — `contraindicated_dressings` corrections

```python
# cat_b_honey_dry_necrotic — add alginate:
"contraindicated_dressings": ["honey", "alginate"]

# cat_b_postop_clean — remove foam (not explicitly contraindicated by source):
"contraindicated_dressings": ["alginate"]

# cat_b_burns_hand — add adhesive film dressing (ANZBA_DEPTH explicit):
"contraindicated_dressings": ["adhesive_film_dressing"]

# cat_b_skin_tear_type2_flap — remove dry_gauze (not in ISTAP chunks):
"contraindicated_dressings": ["adhesive_foam", "adhesive_film"]

# cat_b_burns_minor_superficial — replace:
"contraindicated_dressings": ["adhesive_film_dressing"]
# (occlusive_hydrocolloid not in ANZBA chunks; ice is not a dressing product)
```

### Fix 3 — Reference text minor corrections

```python
# cat_b_silver_clean_granulating — Contraindicated Dressings section:
# CHANGE: "Using silver on a non-infected wound is unnecessary and may impair healing."
# TO: "Silver is explicitly excluded from Wound Type 1 by the clinical algorithm and is not indicated."

# cat_b_burns_minor_superficial — Primary Dressing:
# REMOVE film dressing from list (ANZBA Depth 1 = moisturisers only; film is Depth 2)
# KEEP hydrogel, silicone non-adherent, paraffin tulle
# Also remove "film" from allowed_dressings for this case
```
