# Category A — Gold Standard Audit Report
**Based on direct comparison of `reference` text vs actual chunk `text` + `ai_summary` content**
**8 source _kept.json files verified. Every finding is chunk-traceable.**

---

## Legend
- ✅ Correct — grounded in cited chunk
- ⚠️ Hallucination / Ungrounded — stated in reference but NOT in any cited chunk
- 🔧 Fix needed — wrong chunk cited, chunk missing, or chunk should be removed
- 📝 POV / style issue
- ➕ Chunk should be added
- ➖ Chunk should be removed

---

## TYPE 1 — `cat_a_type1_dry`
**Current reference_contexts:** `GP_TYPE1`, `WCM_FILM`, `WCM_HYDROCOLLOID`

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| "All types of dressing material except silver, charcoal and special advanced dressing materials" | GP_TYPE1 TEXT ✓ | ✅ |
| Film: "transparent with measurement grid" | WCM_FILM TEXT ✓ | ✅ |
| Film: "waterproof, breathable" | WCM_FILM TEXT ✓ | ✅ |
| Film: apply over site making sure no air under it | WCM_FILM TEXT ✓ | ✅ |
| Film: to remove, stretch and pull slowly from edges | WCM_FILM TEXT ✓ | ✅ |
| Film: change every 2–5 days | WCM_FILM TEXT ✓ | ✅ |
| Hydrocolloid: "provides moist environment, absorbs exudates, bacterial barrier" | WCM_HYDROCOLLOID TEXT ✓ | ✅ |
| Hydrocolloid: "cleans and debrides by autolysis, promotes granulation" | WCM_HYDROCOLLOID TEXT ✓ | ✅ |
| Hydrocolloid: "apply adhesive side without touching wound bed" | WCM_HYDROCOLLOID TEXT ✓ | ✅ |
| Hydrocolloid: "yellow liquid... needs to be cleansed" | WCM_HYDROCOLLOID TEXT ✓ | ✅ |
| Hydrocolloid: change every 2–5 days | WCM_HYDROCOLLOID TEXT ✓ | ✅ |
| Antibiotic: No | GP_TYPE1 TEXT ✓ | ✅ |
| Referral: Not required | GP_TYPE1 TEXT ✓ | ✅ |
| "Ready for secondary wound closure / continue dressing till heals" | GP_TYPE1 TEXT ✓ | ✅ (in Clinical Notes as "continue dressing strategy") |

### reference_contexts Assessment
- `GP_TYPE1` — ✅ Required (dressing list, antibiotic, referral)
- `WCM_FILM` — ✅ Required (film application, frequency, properties)
- `WCM_HYDROCOLLOID` — ✅ Required (hydrocolloid application, frequency)

### Issues Found
**NONE.** Type 1 is fully clean. All reference statements are grounded. All 9 sections are present. No hallucinations detected. reference_contexts set is complete and sufficient.

---

## TYPE 2 — `cat_a_type2_wet`
**Current reference_contexts:** `GP_TYPE2`, `WCM_ALGINATE`, `WCM_HYDROFIBRE`, `WCM_FOAM`

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| GP dressing list for Type 2: Foam, Alginate, Hydrofiber, Polymeric membrane | GP_TYPE2 TEXT ✓ | ✅ |
| Alginate: "absorbs wound exudates and maintains moisture" | WCM_ALGINATE TEXT ✓ | ✅ |
| Alginate: "haemostatic properties" | WCM_ALGINATE TEXT ✓ | ✅ |
| Alginate: "biodegradable" | WCM_ALGINATE TEXT ✓ | ✅ |
| Alginate: "requires secondary dressing" | WCM_ALGINATE TEXT ✓ | ✅ |
| Alginate: "available in sheet or rope form" | WCM_ALGINATE TEXT ✓ | ✅ |
| Alginate: "residue... washed off during the cleansing process" | WCM_ALGINATE TEXT ✓ | ✅ |
| Alginate: change every 2–5 days | WCM_ALGINATE TEXT ✓ | ✅ |
| Hydrofibre: "manages heavy exuding wounds, maintains moist healing environment, reduces maceration risk" | WCM_HYDROFIBRE TEXT ✓ | ✅ |
| Hydrofibre: "can be used on infected wounds" | WCM_HYDROFIBRE TEXT ✓ | ✅ |
| Hydrofibre: "requires secondary dressings" | WCM_HYDROFIBRE TEXT ✓ | ✅ |
| Hydrofibre: "becomes gel-like layer which can be easily removed" | WCM_HYDROFIBRE TEXT ✓ | ✅ |
| Hydrofibre: change every 2–5 days | WCM_HYDROFIBRE TEXT ✓ | ✅ |
| Foam: "highly absorbent, conforms to body contours" | WCM_FOAM TEXT ✓ | ✅ |
| Foam: "bacterial and waterproof barrier" | WCM_FOAM TEXT ✓ | ✅ |
| Foam: "secondary dressing or cavity fillers" | WCM_FOAM TEXT ✓ | ✅ |
| Foam: change every 2–3 days | WCM_FOAM TEXT ✓ | ✅ |
| Antibiotic: "May or may not, based on the underlying cause" | GP_TYPE2 TEXT ✓ | ✅ |
| "Find underlying cause / Treat underlying cause" | GP_TYPE2 TEXT ✓ | ✅ |
| Referral: Not required | GP_TYPE2 TEXT ✓ (no referral flag) | ✅ |

### ⚠️ Hallucination Found — Silver & Charcoal in Contraindicated Dressings

The reference states:
> "Silver dressings: not listed as a recommended dressing for Wound Type 2. Charcoal dressings: not listed as a recommended dressing for Wound Type 2."

**Problem:** GP_TYPE2 only lists *what IS recommended* (Foam, Alginate, Hydrofiber, Polymeric membrane). It does NOT say silver and charcoal are contraindicated — only that they are not listed. Silver is an explicit contraindication for Type 1 only. Stating this for Type 2 is an **inference beyond the source** — clinically defensible but not directly grounded. More accurate would be: "Silver and charcoal are not recommended dressing types listed for Wound Type 2."

**Recommended Fix:** Rephrase Contraindicated Dressings section as:
> "No dressings are explicitly contraindicated for Wound Type 2. Silver and charcoal dressings are not listed among the recommended dressing materials for this wound type by the clinical algorithm."

### reference_contexts Assessment
All 4 chunks are required and correctly cited. No chunks to add or remove.

---

## TYPE 3 — `cat_a_type3_dry_infected`
**Current reference_contexts:** `GP_TYPE3`, `WCM_SILVER`, `WCM_HYDROGEL`, `SFP_IODINE`, `WCM_HYDROCOLLOID`

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| GP dressing list: Tulle, Hydrogel, Hydrocolloid, Silver, Iodine | GP_TYPE3 TEXT ✓ | ✅ |
| Silver: "reduce bacterial bioburden in infected wounds" | WCM_SILVER TEXT ✓ | ✅ |
| Silver: "locally acting, bactericidal, no known resistance" | WCM_SILVER TEXT ✓ | ✅ |
| Silver: "place dressing with silver side facing wound bed" | WCM_SILVER TEXT ✓ | ✅ |
| Silver: change every 2–3 days | WCM_SILVER TEXT ✓ | ✅ |
| Hydrogel: "rehydrates and debrides, promotes moist healing, desloughing agent, promotes granulation, reduces pain" | WCM_HYDROGEL TEXT ✓ | ✅ |
| Hydrogel: "requires secondary dressing" | WCM_HYDROGEL TEXT ✓ | ✅ |
| Hydrogel: change every 2–3 days | WCM_HYDROGEL TEXT ✓ | ✅ |
| Hydrocolloid: "apply adhesive side without touching wound bed" | WCM_HYDROCOLLOID TEXT ✓ | ✅ |
| Hydrocolloid: "cleans and debrides by autolysis, effective for low to moderate exuding wounds" | WCM_HYDROCOLLOID TEXT ✓ | ✅ |
| Hydrocolloid: change every 2–5 days | WCM_HYDROCOLLOID TEXT ✓ | ✅ |
| Iodine: "should be avoided in patients with thyroid disorders" | SFP_IODINE TEXT ✓ ("may be absorbed systematically... avoid in patients with thyroid disorders") | ✅ |
| Antibiotic: "Yes based on C&S report" | GP_TYPE3 TEXT ✓ | ✅ |
| Debridement: "may be needed" | GP_TYPE3 TEXT ✓ | ✅ |
| Referral: Not required | GP_TYPE3 TEXT ✓ | ✅ |
| "yellow liquid may form under dressing... normal... cleansed at next change" | WCM_HYDROCOLLOID TEXT ✓ | ✅ |

### ⚠️ Hallucination Found — "most likely vascular in origin"

GP_TYPE3 TEXT states: "Dry, infected wound with <25% slough/necrotic tissue **(most likely vascular in origin)**"

This note does NOT appear in the reference — which is correct. No issue here; the reference wisely omits it since it's not always true and is a parenthetical in the source.

### 🔧 Issue — Hydrocolloid Listed as Secondary Only

The reference describes hydrocolloid under "Secondary Dressing" as secondary to hydrogel. However, GP_TYPE3 lists hydrocolloid as a **primary dressing option** (item 3 in the dressing list alongside tulle, hydrogel, silver, iodine). In the WCM, hydrocolloid can stand alone for low-to-moderate exuding wounds without needing a primary dressing. This framing is acceptable clinically for a dry wound (silver or hydrogel primary, hydrocolloid as secondary/standalone) but should be clarified.

**Recommended Fix (minor):** Clarify in the reference: "Hydrocolloid may be used as a standalone primary/secondary dressing for low-to-moderate exudate; or as a secondary cover over hydrogel."

### reference_contexts Assessment
All 5 chunks are correctly cited and necessary. The addition of `WCM_HYDROCOLLOID` (vs original 4) is correct — application tips for hydrocolloid are grounded there. No changes needed.

---

## TYPE 4 — `cat_a_type4_wet_infected`
**Current reference_contexts:** `GP_TYPE4`, `WCM_SILVER`, `WCM_ALGINATE`, `WCM_HYDROFIBRE`, `WCM_FOAM`, `SFP_IODINE`

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| GP dressing list: Alginate, Foam, Silver, Hydrofiber, Polymeric membrane, Iodine | GP_TYPE4 TEXT ✓ | ✅ |
| Silver: "reduces bacterial bioburden, locally acting, bactericidal, no known resistance" | WCM_SILVER TEXT ✓ | ✅ |
| Silver: "place silver side facing wound bed" | WCM_SILVER TEXT ✓ | ✅ |
| Silver: change every 2–3 days | WCM_SILVER TEXT ✓ | ✅ |
| Alginate: absorbs exudates, haemostatic, biodegradable, requires secondary | WCM_ALGINATE TEXT ✓ | ✅ |
| Alginate: change every 2–5 days | WCM_ALGINATE TEXT ✓ | ✅ |
| Alginate: "residue... washed off during cleansing" | WCM_ALGINATE TEXT ✓ | ✅ |
| Hydrofibre: manages heavy exuding, maintains moist, reduces maceration, can be used on infected | WCM_HYDROFIBRE TEXT ✓ | ✅ |
| Hydrofibre: requires secondary, gel-like layer | WCM_HYDROFIBRE TEXT ✓ | ✅ |
| Hydrofibre: change every 2–5 days | WCM_HYDROFIBRE TEXT ✓ | ✅ |
| Foam: highly absorbent, conforms, bacterial and waterproof | WCM_FOAM TEXT ✓ | ✅ |
| Foam: change every 2–3 days | WCM_FOAM TEXT ✓ | ✅ |
| Iodine contraindication for thyroid disorders | SFP_IODINE TEXT ✓ | ✅ |
| Antibiotic: "Yes based on C&S" | GP_TYPE4 TEXT ✓ | ✅ |
| Debridement: "may be needed" | GP_TYPE4 TEXT ✓ | ✅ |
| Referral: Not required | GP_TYPE4 TEXT ✓ | ✅ |

### Issues Found
**NONE.** Type 4 is fully clean. The addition of `SFP_IODINE` and `WCM_FOAM` vs the original 4 chunks is correct and necessary. All reference statements are grounded.

---

## TYPE 5 — `cat_a_type5_dry_necrotic`
**Current reference_contexts:** `GP_TYPE5`, `WCM_HYDROGEL`, `WCM_DEBRIDE`, `WCM_ALGINATE`, `WCM_HYDROCOLLOID`

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| GP dressing list: Hydrogel, Hydrocolloid, Polymeric membrane | GP_TYPE5 TEXT ✓ | ✅ |
| Hydrogel: "rehydrate, debride and deslough... promote moist healing... cavity filling" | WCM_HYDROGEL TEXT ✓ | ✅ |
| Hydrogel: "desloughing agent, promotes granulation, reduces pain" | WCM_HYDROGEL TEXT ✓ | ✅ |
| Hydrogel: "requires secondary dressing" | WCM_HYDROGEL TEXT ✓ | ✅ |
| Hydrogel: change every 2–3 days | WCM_HYDROGEL TEXT ✓ | ✅ |
| Hydrocolloid: "moist environment, absorbs exudates, bacterial barrier, autolysis, promotes granulation" | WCM_HYDROCOLLOID TEXT ✓ | ✅ |
| Hydrocolloid: "effective for low to moderate exuding wounds" | WCM_HYDROCOLLOID TEXT ✓ | ✅ |
| Hydrocolloid: change every 2–5 days | WCM_HYDROCOLLOID TEXT ✓ | ✅ |
| Hydrocolloid: "yellow liquid... normal... cleansed at next change" | WCM_HYDROCOLLOID TEXT ✓ | ✅ |
| Alginate: "not helpful for dry wounds" | WCM_ALGINATE TEXT ✓ ("Not helpful for dry wounds" in Disadvantages) | ✅ |
| Hydrofibre: "not helpful for dry wounds" | WCM_HYDROFIBRE TEXT ✓ ("Not helpful for dry wounds" in Disadvantages) | ✅ — **BUT WCM_HYDROFIBRE IS NOT IN reference_contexts!** |
| Antibiotic: No | GP_TYPE5 TEXT ✓ | ✅ |
| Debridement: "is needed" | GP_TYPE5 TEXT ✓ | ✅ |
| Referral: Not required | GP_TYPE5 TEXT ✓ | ✅ |
| Polymeric membrane: listed for Type 5 | GP_TYPE5 TEXT ✓ | ✅ |

### 🔧 Missing Chunk — `WCM_HYDROFIBRE` needed

The reference states "Hydrofibre: not helpful for dry wounds — same reason as alginate." This is grounded in `WCM_HYDROFIBRE` TEXT: *"Not helpful for dry wounds"* — but `WCM_HYDROFIBRE` is NOT in the current reference_contexts. The contraindication claim has no grounding chunk.

**Fix:** Add `WCM_HYDROFIBRE` to reference_contexts.

### ⚠️ Ungrounded — Silver and Charcoal in Contraindicated Dressings

Reference states: "Silver and charcoal: not indicated as wound is not infected."

Neither `GP_TYPE5` TEXT nor any other cited chunk says silver/charcoal are contraindicated for Type 5 — only that they are not in the recommended list. Same issue as Type 2: this is an **inference**, not a stated contraindication.

**Fix:** Rephrase to: "Silver and charcoal dressings are not listed among the recommended dressing materials for Wound Type 5."

### reference_contexts Fix
**Add:** `WCM_HYDROFIBRE`
**Updated set:** `GP_TYPE5`, `WCM_HYDROGEL`, `WCM_DEBRIDE`, `WCM_ALGINATE`, `WCM_HYDROCOLLOID`, `WCM_HYDROFIBRE`

---

## TYPE 6 — `cat_a_type6_wet_necrotic`
**Current reference_contexts:** `GP_TYPE6`, `GP_REFERRAL`, `WCM_ALGINATE`, `WCM_HYDROFIBRE`, `WCM_FOAM`

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| GP dressing list: Alginate, Foam, Polymeric membrane, Hydrofiber | GP_TYPE6 TEXT ✓ | ✅ |
| "Requires hospital referral: YES" | GP_TYPE6 TEXT ✓ | ✅ |
| Surgical/mechanical debridement recommended, may need repeated | GP_TYPE6 TEXT ✓ | ✅ |
| Antibiotic: "May or may not, based on underlying cause" | GP_TYPE6 TEXT ✓ | ✅ |
| Alginate: absorbs exudates, haemostatic, biodegradable, requires secondary | WCM_ALGINATE TEXT ✓ | ✅ |
| Alginate: change every 2–5 days | WCM_ALGINATE TEXT ✓ | ✅ |
| Alginate: "residue washed off" | WCM_ALGINATE TEXT ✓ | ✅ |
| Hydrofibre: manages heavy exuding, moist, reduces maceration | WCM_HYDROFIBRE TEXT ✓ | ✅ |
| Hydrofibre: gel-like layer, easy to remove | WCM_HYDROFIBRE TEXT ✓ | ✅ |
| Hydrofibre: change every 2–5 days | WCM_HYDROFIBRE TEXT ✓ | ✅ |
| Foam: highly absorbent, conforms, bacterial/waterproof | WCM_FOAM TEXT ✓ | ✅ |
| Foam: "secondary dressing or cavity filler" | WCM_FOAM TEXT ✓ | ✅ |
| Foam: change every 2–3 days | WCM_FOAM TEXT ✓ | ✅ |
| GP_REFERRAL: criteria (surgical debridement, NPWT, advanced procedures, acute complications, sepsis, cellulitis) | GP_REFERRAL TEXT ✓ | ✅ |
| "Hydrogel is not listed as recommended dressing for this wound type" | GP_TYPE6 TEXT ✓ (not in list) | ✅ |

### Issues Found
**NONE.** Type 6 is fully clean. All 5 chunks are correctly cited and all reference statements are grounded. The Contraindicated Dressings section appropriately states "No specific dressing contraindications are documented for Wound Type 6. Hydrogel is not listed as a recommended dressing for this wound type." — this is accurate and does not overstate.

---

## TYPE 7 — `cat_a_type7_dry_infected_necrotic`
**Current reference_contexts:** `GP_TYPE7`, `GP_REFERRAL`, `WCM_SILVER`, `WCM_HYDROGEL`, `SFP_IODINE`, `WCM_ALGINATE`, `WCM_HYDROCOLLOID`

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| GP dressing list: Silver, Hydrogel, Hydrocolloid, Iodine, Polymeric membrane | GP_TYPE7 TEXT ✓ | ✅ |
| "Requires hospital referral: YES" | GP_TYPE7 TEXT ✓ | ✅ |
| Surgical/mechanical debridement strongly recommended | GP_TYPE7 TEXT ✓ | ✅ |
| Antibiotic: "Yes based on C&S" | GP_TYPE7 TEXT ✓ | ✅ |
| Silver: "locally acting, bactericidal, no known resistance" | WCM_SILVER TEXT ✓ | ✅ |
| Silver: "place silver side facing wound bed" | WCM_SILVER TEXT ✓ | ✅ |
| Silver: change every 2–3 days | WCM_SILVER TEXT ✓ | ✅ |
| Hydrogel: "rehydrates and debrides, promotes moist healing, reduces pain" | WCM_HYDROGEL TEXT ✓ | ✅ |
| Hydrogel: "requires secondary dressing" | WCM_HYDROGEL TEXT ✓ | ✅ |
| Hydrogel: change every 2–3 days | WCM_HYDROGEL TEXT ✓ | ✅ |
| Hydrocolloid: "moist environment, cleans and debrides by autolysis, promotes granulation, bacterial barrier" | WCM_HYDROCOLLOID TEXT ✓ | ✅ |
| Hydrocolloid: "effective for low to moderate exuding wounds" | WCM_HYDROCOLLOID TEXT ✓ | ✅ |
| Hydrocolloid: "apply adhesive side without touching wound bed" | WCM_HYDROCOLLOID TEXT ✓ | ✅ |
| Hydrocolloid: change every 2–5 days | WCM_HYDROCOLLOID TEXT ✓ | ✅ |
| Iodine: "avoid in thyroid disorders" | SFP_IODINE TEXT ✓ | ✅ |
| Alginate: "not helpful for dry wounds" | WCM_ALGINATE TEXT ✓ | ✅ |
| Hydrofibre: "not helpful for dry wounds" | WCM_HYDROFIBRE TEXT ✓ | ✅ — **BUT WCM_HYDROFIBRE IS NOT IN reference_contexts!** |
| GP_REFERRAL: referral criteria | GP_REFERRAL TEXT ✓ | ✅ |

### 🔧 Missing Chunk — `WCM_HYDROFIBRE` needed

Same issue as Type 5: "Hydrofibre: not helpful for dry wounds" is stated in Contraindicated Dressings, grounded in `WCM_HYDROFIBRE`, but that chunk is absent from reference_contexts.

**Fix:** Add `WCM_HYDROFIBRE` to reference_contexts.

### reference_contexts Fix
**Add:** `WCM_HYDROFIBRE`
**Updated set:** `GP_TYPE7`, `GP_REFERRAL`, `WCM_SILVER`, `WCM_HYDROGEL`, `SFP_IODINE`, `WCM_ALGINATE`, `WCM_HYDROCOLLOID`, `WCM_HYDROFIBRE`

---

## TYPE 8 — `cat_a_type8_wet_infected_necrotic`
**Current reference_contexts:** `GP_TYPE8`, `GP_REFERRAL`, `WCM_SILVER`, `WCM_ALGINATE`, `WCM_CHARCOAL`, `WCM_HYDROFIBRE`, `WCM_FOAM`, `SFP_IODINE`

### Hallucination Check

| Reference Statement | Source | Status |
|---|---|---|
| GP dressing list: Alginate, Silver, Hydrofiber, Foam, Polymeric membrane, Charcoal, Iodine | GP_TYPE8 TEXT ✓ | ✅ |
| "Requires hospital referral: YES" | GP_TYPE8 TEXT ✓ | ✅ |
| Surgical/mechanical debridement strongly recommended, may need repeated | GP_TYPE8 TEXT ✓ | ✅ |
| Antibiotic: "Yes based on C&S" | GP_TYPE8 TEXT ✓ | ✅ |
| Silver: properties and application | WCM_SILVER TEXT ✓ | ✅ |
| Silver: change every 2–3 days | WCM_SILVER TEXT ✓ | ✅ |
| Alginate: properties, haemostatic, biodegradable, requires secondary | WCM_ALGINATE TEXT ✓ | ✅ |
| Alginate: change every 2–5 days | WCM_ALGINATE TEXT ✓ | ✅ |
| Alginate: "residue washed off" | WCM_ALGINATE TEXT ✓ | ✅ |
| Hydrofibre: manages heavy exuding, reduces maceration, can be used on infected | WCM_HYDROFIBRE TEXT ✓ | ✅ |
| Hydrofibre: gel-like layer, requires secondary | WCM_HYDROFIBRE TEXT ✓ | ✅ |
| Hydrofibre: change every 2–5 days | WCM_HYDROFIBRE TEXT ✓ | ✅ |
| Foam: highly absorbent, bacterial/waterproof, secondary dressing | WCM_FOAM TEXT ✓ | ✅ |
| Foam: change every 2–3 days | WCM_FOAM TEXT ✓ | ✅ |
| Charcoal: "odour absorbent, reduces odour" | WCM_CHARCOAL TEXT ✓ | ✅ |
| Charcoal: "requires secondary dressing" | WCM_CHARCOAL TEXT ✓ | ✅ |
| Charcoal: change every 2 days | WCM_CHARCOAL TEXT ✓ | ✅ |
| Iodine: contraindicated in thyroid disorders | SFP_IODINE TEXT ✓ | ✅ |
| GP_REFERRAL: referral criteria | GP_REFERRAL TEXT ✓ | ✅ |

### ✅ Confirmed Correct — "Do not cut charcoal" REMOVED

Good catch in the v3_fixed notes: `WCM_CHARCOAL` TEXT has NO instruction about not cutting the dressing. The prior version hallucinated this. Correctly removed here.

### Issues Found
**NONE.** Type 8 with the 8-chunk reference_contexts set is fully clean. All reference statements are directly grounded.

---

## Summary Table — All Category A Issues

| Case | Hallucination | Missing Chunk | Chunk to Remove | POV Issue |
|---|---|---|---|---|
| Type 1 | None | None | None | None |
| Type 2 | ⚠️ Silver/charcoal framed as "contraindicated" — only "not listed" | None | None | None |
| Type 3 | None | None | None | None |
| Type 4 | None | None | None | None |
| Type 5 | ⚠️ Silver/charcoal framed as "contraindicated" — only "not listed" | ➕ `WCM_HYDROFIBRE` | None | None |
| Type 6 | None | None | None | None |
| Type 7 | None | ➕ `WCM_HYDROFIBRE` | None | None |
| Type 8 | None | None | None | None |

**POV Check (user_input notes):** All 8 Category A cases have NO patient notes field (notes=""), so there is no 1st-person POV issue in any Cat A user_input.

---

## Required Fixes — Category A (Minimal, Precise)

### Fix 1 — Rephrase Contraindicated Dressings for Types 2 and 5

**Type 2 — Change from:**
> "Silver dressings: not listed as a recommended dressing for Wound Type 2. Charcoal dressings: not listed as a recommended dressing for Wound Type 2."

**Change to:**
> "No dressings are explicitly contraindicated for Wound Type 2. Silver and charcoal dressings are not listed among the recommended dressing materials for this wound type by the clinical algorithm."

**Type 5 — Change from:**
> "Silver and charcoal: not indicated as wound is not infected."

**Change to:**
> "Silver and charcoal dressings are not listed among the recommended dressing materials for Wound Type 5 by the clinical algorithm."

*(The alginate/hydrofibre contraindication language for Type 5 is fine — it IS grounded: "Not helpful for dry wounds" is explicit in both WCM_ALGINATE and WCM_HYDROFIBRE.)*

---

### Fix 2 — Add `WCM_HYDROFIBRE` to reference_contexts for Types 5 and 7

**Type 5 — Updated reference_contexts:**
```python
"reference_contexts": [ctx(GP_TYPE5), ctx(WCM_HYDROGEL), ctx(WCM_DEBRIDE),
                       ctx(WCM_ALGINATE), ctx(WCM_HYDROCOLLOID), ctx(WCM_HYDROFIBRE)],
```

**Type 7 — Updated reference_contexts:**
```python
"reference_contexts": [ctx(GP_TYPE7), ctx(GP_REFERRAL), ctx(WCM_SILVER), ctx(WCM_HYDROGEL),
                       ctx(SFP_IODINE), ctx(WCM_ALGINATE), ctx(WCM_HYDROCOLLOID), ctx(WCM_HYDROFIBRE)],
```

---

## No Other Changes Required for Category A

All other reference text across Types 1, 3, 4, 6, 7, 8 is correctly grounded. No brand names introduced. No clinical claims beyond chunk content. No 1st-person POV issues in user_input (no notes fields in Cat A). All 9 required sections are present in all 8 cases.
