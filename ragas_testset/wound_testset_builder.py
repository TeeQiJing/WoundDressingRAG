"""
wound_testset_builder.py
========================
Generates wound_testset_curated.json

ALL 20 test cases are grounded in actual content from:
  - Wound Care Manual, First Edition 2014 (MOH Malaysia)         → WCM
  - Garis Panduan Penjagaan Luka di Fasiliti Kesihatan Primer    → GP
  - Wound Dressings: A Primer for the Family Physician (SFP 2014)→ SFP
  - AJGP Vol.51 Nov 2022 (Sinha, Free, Ladlow)                   → AJGP

The 8 wound types from Figure 17.1 (WCM p.164–167 / GP p.13–14) are:
  1 — Clean healthy granulating, any moisture
  2 — Clean wet wound
  3 — Dry, infected, <25% slough/necrotic  (most likely vascular)
  4 — Wet, infected, <25% slough/necrotic
  5 — Dry, NON-infected, >25% slough/necrotic
  6 — Wet, NON-infected, >25% slough/necrotic
  7 — Dry, infected, >25% slough/necrotic
  8 — Wet, infected, >25% slough/necrotic

IMPORTANT — what the documents DO and DO NOT say:
  ✓ The algorithm uses <25% vs >25% non-viable tissue as the split
  ✓ Dressing lists are explicit per wound type (WCM Table p.165–167)
  ✓ Dressing properties + frequency in WCM p.133–136
  ✓ Silver is bactericidal with no known resistance (WCM p.135)
  ✓ Iodine avoided in thyroid disorders (SFP p.20)
  ✓ Alginates absorb up to 20× weight (SFP p.19)
  ✓ Hydrogel frequency: 2-3 days (WCM p.133)
  ✓ Hydrofibre frequency: 2-5 days (WCM p.135)
  ✓ Foam: 2-3 days or longer for offloading (WCM p.135)
  ✓ Silver: 2-3 days (WCM p.135)
  ✓ Hydrocolloid: 2-5 days (WCM p.134)
  ✓ NPWT: adjunct only, does NOT replace surgery (WCM p.162)
  ✓ NPWT contraindicated in necrotic wound bed/eschar (WCM p.159)
  ✓ Skin tears → silicone foam, NO adhesives on fragile skin (AJGP Table 1)
  ✓ Diabetic foot → antimicrobial primary + secondary by exudate (AJGP Table 1)
  ✓ Check pedal pulses; refer if poor perfusion (AJGP Table 1)
  ✓ Post-op, no exudate → film or thin hydrocolloid (AJGP Table 1)
  ✓ Small burn → hydrogel/hydrocolloid/film; hand burns refer (AJGP Table 1)
  ✓ Silicone foams on feet without borders, anchored with bandage (AJGP Table 1)
  ✓ Honey: daily to every 3 days (WCM p.148)
  ✓ ClinicalSignalExtractor notes override logic (wound_app_v4.py)
  ✗ Compression therapy NOT mentioned in these documents (do not claim)
  ✗ IWGDF not referenced in these documents
  ✗ Specific brand names like "Aquacel", "Kaltostat" only in SFP Table 2
"""

import json, os

OUTPUT_DIR = "./ragas_testset/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

testset = [

  # ──────────────────────────────────────────────────────────────────────────
  # WOUND TYPE 1 — Clean healthy granulating, dry/minimal
  # Documents: WCM p.165, GP p.13
  # ──────────────────────────────────────────────────────────────────────────
  {
    "synthesizer_name": "type1_granulating_dry",
    "time_inputs": {
      "necrotic_pct": 0, "slough_pct": 0, "granulation_pct": 100,
      "infection": "Not infected", "moisture": "Low", "edge": "Advancing"
    },
    "user_input": (
      "Wound bed is 100% healthy pink granulation tissue. No slough or necrosis present. "
      "No signs of infection. Minimal exudate — wound is almost dry. "
      "Wound edges are advancing. Ready for healing by secondary intention."
    ),
    "reference": (
      "Wound Type 1 (Clean, healthy granulating wound — dry/minimal exudate).\n"
      "Dressing: All types of dressing material are suitable EXCEPT silver, charcoal, "
      "and special advanced dressing materials. Silver and charcoal are contraindicated "
      "in clean granulating wounds.\n"
      "Suitable options include: Hydrocolloid (2–5 days), Film dressing (2–5 days), "
      "Foam as secondary, Tulle/non-adherent dressing.\n"
      "Antibiotic: Not indicated.\n"
      "Surgery: If wound is small, continue dressing until healing by secondary intention. "
      "If ready, wound can proceed to secondary closure.\n"
      "Frequency varies depending on dressing type and wound condition."
    ),
    "reference_contexts": [
      "Clean, healthy granulating wound: All types of dressing material except silver, charcoal and special advanced dressing materials. No antibiotic. Ready for secondary wound closure or continue dressing till wound heals by secondary intention.",
      "Silver, charcoal and special advanced materials should not be used on clean granulating wounds.",
      "Frequency of wound dressing varies depending on type of wound and also dressing material used."
    ],
  },

  # ──────────────────────────────────────────────────────────────────────────
  # WOUND TYPE 1 — Clean healthy granulating, moderate/wet
  # Documents: WCM p.165, GP p.13, SFP p.19
  # ──────────────────────────────────────────────────────────────────────────
  {
    "synthesizer_name": "type1_granulating_wet",
    "time_inputs": {
      "necrotic_pct": 0, "slough_pct": 5, "granulation_pct": 95,
      "infection": "Not infected", "moisture": "Moderate", "edge": "Advancing"
    },
    "user_input": (
      "Wound is predominantly granulating — 95% healthy red cobblestone granulation tissue. "
      "Minimal slough at edges. No infection. Moderate exudate. Wound edges advancing. "
      "No signs of dehiscence or deterioration."
    ),
    "reference": (
      "Wound Type 1 (Clean, healthy granulating wound — moderate/wet exudate).\n"
      "Dressing: All types suitable EXCEPT silver, charcoal, and special advanced dressings. "
      "For moderate exudate, foam dressing is appropriate as it is highly absorbent, "
      "conforms to body contours, and is designed for moderately exuding wounds. "
      "Hydrofibre or alginate are also suitable options.\n"
      "Antibiotic: Not indicated.\n"
      "Non-silicone foam types should be avoided in patients with fragile skin. "
      "Foam frequency: 2–3 days or longer.\n"
      "Surgery: Continue dressing until heals by secondary intention or secondary closure if ready."
    ),
    "reference_contexts": [
      "Clean, healthy granulating wound: All types of dressing material except silver, charcoal and special advanced dressing materials.",
      "Foams: highly absorbent, cushioning, conforms to body contours, bacterial and waterproof. Frequency of dressing change: 2 to 3 days or longer.",
      "Foam suitable for moderately exudating wounds, skin tears, skin grafts and donor sites. Nonsilicone types should be avoided in patients with fragile skin."
    ],
  },

  # ──────────────────────────────────────────────────────────────────────────
  # WOUND TYPE 2 — Clean wet wound
  # Documents: WCM p.165, GP p.13, SFP p.19
  # ──────────────────────────────────────────────────────────────────────────
  {
    "synthesizer_name": "type2_clean_wet",
    "time_inputs": {
      "necrotic_pct": 5, "slough_pct": 10, "granulation_pct": 85,
      "infection": "Not infected", "moisture": "High", "edge": "Advancing"
    },
    "user_input": (
      "Wound is clean with mostly granulation tissue. No clinical infection. "
      "High exudate — previous dressing saturated within 24 hours. "
      "Minimal slough but no significant necrosis. Wound is healing."
    ),
    "reference": (
      "Wound Type 2 (Clean and wet wound — high exudate).\n"
      "Recommended dressings:\n"
      "1. Foam — highly absorbent, can be used up to 2–3 days\n"
      "2. Alginate — absorbs up to 20 times its own weight; forms a gel on contact "
      "with wound fluid; suitable for moderate to heavy exudate; frequency 2–5 days\n"
      "3. Hydrofibre — manages heavy exuding wounds, longer wear time, "
      "reduces risk of maceration; frequency 2–5 days\n"
      "4. Polymeric membrane dressing — manages moisture imbalance, "
      "has antiseptic property, frequency 2–5 days\n"
      "Antibiotic: May or may not be needed based on underlying cause.\n"
      "Surgery: Find and treat underlying cause if necessary."
    ),
    "reference_contexts": [
      "Clean and wet wound: 1. Foam  2. Alginate  3. Hydrofiber  4. Polymeric membrane. May or may not need antibiotic based on underlying cause. Find underlying cause and treat if necessary.",
      "Alginates partially dissolve on contact with wound fluid to form a gel able to absorb up to 20 times own weight hence recommended for moderate to heavy exudate. Frequency: 2 to 5 days.",
      "Hydrofibre: Manage heavy exuding wounds. Maintains moist healing environment. Longer wear time. Reduces risk of maceration. Frequency: 2 to 5 days."
    ],
  },

  # ──────────────────────────────────────────────────────────────────────────
  # WOUND TYPE 3 — Dry, infected, <25% slough/necrotic (vascular origin)
  # Documents: WCM p.165, GP p.13-14
  # ──────────────────────────────────────────────────────────────────────────
  {
    "synthesizer_name": "type3_dry_infected_low_necrotic",
    "time_inputs": {
      "necrotic_pct": 10, "slough_pct": 12, "granulation_pct": 78,
      "infection": "Infected", "moisture": "Low", "edge": "Non-advancing"
    },
    "user_input": (
      "Dry wound with minimal exudate. Wound bed shows approximately 22% combined "
      "slough and necrotic tissue (well under 25%). Clinically infected — erythema, "
      "warmth, increased pain. No exudate visible. Most likely vascular in origin. "
      "Wound edges not advancing."
    ),
    "reference": (
      "Wound Type 3 (Dry, infected wound with <25% slough/necrotic tissue — most likely vascular in origin).\n"
      "Recommended dressings:\n"
      "1. Tulle — non-adherent primary dressing for lightly exuding wounds\n"
      "2. Hydrogel — rehydrates, debrids and desloughs; provides moist environment; "
      "promotes granulation; frequency 2–3 days\n"
      "3. Hydrocolloid — provides moist environment, absorbs exudate, promotes "
      "autolysis; frequency 2–5 days\n"
      "4. Silver dressing — reduces bacterial bioburden in infected wounds; "
      "bactericidal with no known resistance; frequency 2–3 days\n"
      "5. Iodine base dressing — broad spectrum bacteriostatic activity; "
      "AVOID in patients with thyroid disorders (iodine absorbed systemically)\n"
      "Antibiotic: Yes, based on culture and sensitivity (C&S) report of infected tissue.\n"
      "Surgery: Debridement may be needed."
    ),
    "reference_contexts": [
      "Dry, infected wound with <25% slough/necrotic tissue (most likely vascular in origin): 1.Tulle  2.Hydrogel  3.Hydrocolloid  4.Silver dressing  5.Iodine base dressing. Yes antibiotic based on C&S report. Debridement may be needed.",
      "Silver: To reduce bacterial bioburden in infected wounds. Bactericidal with no known resistance. Frequency: 2 to 3 days.",
      "Hydrogel: Rehydrate, debride and deslough the wound. Promotes moist healing, cavity filling. Frequency: 2 to 3 days.",
      "Iodine may be absorbed systematically — should be avoided in patients with thyroid disorders."
    ],
  },

  # ──────────────────────────────────────────────────────────────────────────
  # WOUND TYPE 3 — Iodine contraindication scenario
  # Documents: WCM p.135, SFP p.20
  # ──────────────────────────────────────────────────────────────────────────
  {
    "synthesizer_name": "type3_iodine_contraindication",
    "time_inputs": {
      "necrotic_pct": 8, "slough_pct": 15, "granulation_pct": 77,
      "infection": "Infected", "moisture": "Low", "edge": "Non-advancing"
    },
    "user_input": (
      "Dry infected wound, <25% slough. Clinician considering cadexomer iodine. "
      "Patient has hypothyroidism and is on levothyroxine. "
      "Is iodine dressing appropriate for this patient?"
    ),
    "reference": (
      "Wound Type 3 (Dry infected, <25% slough/necrotic).\n"
      "CONTRAINDICATED: Iodine-based dressings should NOT be used in patients with "
      "thyroid disorders because iodine may be absorbed systemically, which is "
      "particularly hazardous for patients on levothyroxine or with hypothyroidism.\n"
      "Alternative from the Type 3 dressing list:\n"
      "1. Silver dressing — bactericidal, no known resistance, no thyroid interaction risk; "
      "frequency 2–3 days\n"
      "2. Hydrogel — rehydrates dry wound, promotes autolytic debridement; frequency 2–3 days\n"
      "3. Hydrocolloid — promotes autolysis, moist environment; frequency 2–5 days\n"
      "Antibiotic: Yes, based on C&S report.\n"
      "Debridement may be needed."
    ),
    "reference_contexts": [
      "Because iodine may be absorbed systematically, it should be avoided in patients with thyroid disorders.",
      "Silver: To reduce bacterial bioburden in infected wounds. Locally acting. Bactericidal with no known resistance. Frequency: 2 to 3 days.",
      "Dry, infected wound with <25% slough/necrotic tissue: 1.Tulle 2.Hydrogel 3.Hydrocolloid 4.Silver dressing 5.Iodine base dressing."
    ],
  },

  # ──────────────────────────────────────────────────────────────────────────
  # WOUND TYPE 4 — Wet, infected, <25% slough/necrotic
  # Documents: WCM p.166, GP p.14
  # ──────────────────────────────────────────────────────────────────────────
  {
    "synthesizer_name": "type4_wet_infected_low_necrotic",
    "time_inputs": {
      "necrotic_pct": 12, "slough_pct": 10, "granulation_pct": 78,
      "infection": "Infected", "moisture": "High", "edge": "Advancing"
    },
    "user_input": (
      "Wet wound with high exudate. Clinically infected — purulent discharge, "
      "erythema around wound margin. About 22% combined slough and necrosis "
      "(less than 25% non-viable tissue). Heavily draining."
    ),
    "reference": (
      "Wound Type 4 (Wet, infected wound with <25% slough/necrotic tissue).\n"
      "Recommended dressings:\n"
      "1. Alginate — highly absorbent, haemostatic; absorbs up to 20× weight; "
      "suitable for moderate to heavy exudate; frequency 2–5 days\n"
      "2. Foam — highly absorbent, cushioning, conforms to body contours; "
      "frequency 2–3 days\n"
      "3. Silver — reduces bacterial bioburden in infected wounds; bactericidal "
      "with no known resistance; frequency 2–3 days\n"
      "4. Hydrofibre — manages heavy exuding wounds; can be used on infected wounds; "
      "frequency 2–5 days\n"
      "5. Polymeric membrane dressing — antiseptic property; frequency 2–5 days\n"
      "6. Iodine base dressing — avoid if patient has thyroid disorder\n"
      "Antibiotic: Yes, based on C&S report of infected tissue.\n"
      "Surgery: Debridement may be needed."
    ),
    "reference_contexts": [
      "Wet, infected wound with <25% slough/necrotic tissue: 1.Alginate 2.Foam 3.Silver 4.Hydrofiber 5.Polymeric membrane 6.Iodine base dressing. Antibiotic: Yes based on C&S report. Debridement may be needed.",
      "Hydrofibre: Manage heavy exuding wounds. Maintains moist healing environment. Can be used on infected wounds. Frequency 2 to 5 days.",
      "Silver: To reduce bacterial bioburden in infected wounds. Bactericidal. No known resistance. Frequency: 2 to 3 days.",
      "Alginates: absorb up to 20 times own weight hence recommended for moderate to heavy level of exudate. Frequency 2-5 days."
    ],
  },

  # ──────────────────────────────────────────────────────────────────────────
  # WOUND TYPE 5 — Dry, NON-infected, >25% slough/necrotic
  # Documents: WCM p.166, GP p.14
  # ──────────────────────────────────────────────────────────────────────────
  {
    "synthesizer_name": "type5_dry_noninfected_high_necrotic",
    "time_inputs": {
      "necrotic_pct": 45, "slough_pct": 25, "granulation_pct": 30,
      "infection": "Not infected", "moisture": "Low", "edge": "Non-advancing"
    },
    "user_input": (
      "Dry wound with no exudate. Wound bed has 45% black necrotic tissue and "
      "25% yellow slough — total 70% non-viable tissue (well over 25%). "
      "No clinical signs of infection. Wound edges stalled."
    ),
    "reference": (
      "Wound Type 5 (Dry, non-infected wound with >25% slough/necrotic tissue).\n"
      "Recommended dressings:\n"
      "1. Hydrogel — primary choice; gently rehydrates dry necrotic tissue; "
      "softens necrotic tissue; promotes autolytic debridement by providing "
      "moist wound healing environment; needs secondary dressing; frequency 2–3 days\n"
      "2. Hydrocolloid — provides moist environment; cleans and debrids by autolysis; "
      "promotes granulation tissue; effective for low to moderate exuding wounds; "
      "frequency 2–5 days\n"
      "3. Polymeric membrane dressing — manages moisture imbalance (dry to moderate); "
      "antiseptic property; frequency 2–5 days\n"
      "Antibiotic: NOT indicated (wound is non-infected).\n"
      "Surgery: Debridement IS needed given >25% non-viable tissue. "
      "Autolytic debridement via hydrogel can be initiated while awaiting "
      "or in addition to mechanical/surgical debridement."
    ),
    "reference_contexts": [
      "Dry, non-infected wound with >25% slough/necrotic tissue: 1.Hydrogel 2.Hydrocolloid 3.Polymeric membrane. No antibiotic. Debridement is needed.",
      "Hydrogel: Rehydrate, debride and deslough the wound. Promote moist healing, cavity filling. Comfortable, provides moist environment and reduces pain. Rehydrate eschar. Desloughing agent. Promotes granulation. Needs secondary dressing. Frequency: 2 to 3 days.",
      "Autolytic debridement: the process by which the wound bed utilizes phagocytes and proteolytic enzymes to remove non-viable tissue. Hydrogel gently rehydrates dry necrotic tissue, provides moist wound healing environment, softens necrotic tissue."
    ],
  },

  # ──────────────────────────────────────────────────────────────────────────
  # WOUND TYPE 6 — Wet, NON-infected, >25% slough/necrotic
  # Documents: WCM p.166, GP p.14
  # ──────────────────────────────────────────────────────────────────────────
  {
    "synthesizer_name": "type6_wet_noninfected_high_necrotic",
    "time_inputs": {
      "necrotic_pct": 10, "slough_pct": 50, "granulation_pct": 40,
      "infection": "Not infected", "moisture": "High", "edge": "Non-advancing"
    },
    "user_input": (
      "Wet wound with heavy exudate. Wound bed has 60% combined slough and necrotic "
      "tissue (over 25% threshold). Predominantly fibrinous slough with some necrosis. "
      "No clinical infection signs. Wound not progressing."
    ),
    "reference": (
      "Wound Type 6 (Wet, non-infected wound with >25% slough/necrotic tissue).\n"
      "Recommended dressings:\n"
      "1. Alginate — highly absorbent, haemostatic; absorbs up to 20× weight; "
      "suitable for moderate to heavy exudate; frequency 2–5 days; "
      "needs secondary dressing\n"
      "2. Foam — highly absorbent, cushioning, bacterial and waterproof barrier; "
      "frequency 2–3 days\n"
      "3. Polymeric membrane dressing — manages moisture imbalance, "
      "has surfactant which helps cleanse; frequency 2–5 days\n"
      "4. Hydrofibre — manages heavy exuding wounds; maintains moist healing "
      "environment; reduces maceration risk; frequency 2–5 days\n"
      "Antibiotic: May or may not be needed based on underlying cause.\n"
      "Surgery: Surgical/mechanical debridement is RECOMMENDED. "
      "May need repeated debridement. Refer as appropriate — wound type 6 is "
      "one of the wound types that should be referred to hospital if extensive."
    ),
    "reference_contexts": [
      "Wet, non-infected wound with >25% slough/necrotic tissue: 1.Alginate 2.Foam 3.Polymeric membrane 4.Hydrofiber. May or may not need antibiotic. Surgical/mechanical debridement is recommended. May need repeated debridement.",
      "Wound types 6, 7 and 8 (wet non-infected >25%, dry infected >25%, wet infected >25%) should be referred to hospital.",
      "Alginates: absorb up to 20 times own weight. Available in sheet or rope form. Effective to stop bleeding. Residue of biodegradable product has to be washed off during cleansing. Frequency: 2 to 5 days."
    ],
  },

  # ──────────────────────────────────────────────────────────────────────────
  # WOUND TYPE 7 — Dry, infected, >25% slough/necrotic
  # Documents: WCM p.167, GP p.14
  # ──────────────────────────────────────────────────────────────────────────
  {
    "synthesizer_name": "type7_dry_infected_high_necrotic",
    "time_inputs": {
      "necrotic_pct": 40, "slough_pct": 30, "granulation_pct": 30,
      "infection": "Infected", "moisture": "Low", "edge": "Non-advancing"
    },
    "user_input": (
      "Dry wound — no exudate visible. 70% non-viable tissue (40% black necrotic, "
      "30% slough), well over 25% threshold. Clinically infected — erythema, "
      "warmth, increased pain, wound not progressing. Wound edges stalled."
    ),
    "reference": (
      "Wound Type 7 (Dry, infected wound with >25% slough/necrotic tissue).\n"
      "REFERRAL INDICATED: Wound type 7 should be referred to hospital.\n"
      "Recommended dressings (interim management):\n"
      "1. Silver dressing — primary antimicrobial choice; reduces bacterial bioburden; "
      "bactericidal with no known resistance; frequency 2–3 days\n"
      "2. Hydrogel — rehydrates dry necrotic tissue; autolytic debridement; "
      "frequency 2–3 days\n"
      "3. Hydrocolloid — moist environment, promotes autolysis; frequency 2–5 days\n"
      "4. Iodine base dressing — AVOID in thyroid disorders\n"
      "5. Polymeric membrane dressing — frequency 2–5 days\n"
      "Antibiotic: Yes, based on C&S report of infected tissue.\n"
      "Surgery: Surgical/mechanical debridement is STRONGLY recommended for "
      ">25% non-viable infected tissue."
    ),
    "reference_contexts": [
      "Dry, infected wound with >25% slough/necrotic tissue: 1.Silver dressing 2.Hydrogel 3.Hydrocolloid 4.Iodine base dressing 5.Polymeric membrane. Antibiotic: Yes based on C&S report. Surgical/mechanical debridement is strongly recommended.",
      "Wound types 6, 7 and 8 require referral to hospital: extensive surgical debridement needed, systemic complications, sepsis, cellulitis.",
      "Silver: To reduce bacterial bioburden in infected wounds. Locally acting. No known resistance. Bactericidal. Frequency: 2 to 3 days."
    ],
  },

  # ──────────────────────────────────────────────────────────────────────────
  # WOUND TYPE 8 — Wet, infected, >25% slough/necrotic
  # Documents: WCM p.167, GP p.14
  # ──────────────────────────────────────────────────────────────────────────
  {
    "synthesizer_name": "type8_wet_infected_high_necrotic",
    "time_inputs": {
      "necrotic_pct": 30, "slough_pct": 35, "granulation_pct": 35,
      "infection": "Infected", "moisture": "High", "edge": "Non-advancing"
    },
    "user_input": (
      "Complex wound — heavy exudate, clinically infected with purulent discharge. "
      "65% non-viable tissue (30% necrotic, 35% slough) — over 25% threshold. "
      "Wound edges stalled. Malodour noted. Wound not progressing."
    ),
    "reference": (
      "Wound Type 8 (Wet, infected wound with >25% slough/necrotic tissue).\n"
      "REFERRAL INDICATED: Wound type 8 should be referred to hospital.\n"
      "Recommended dressings (interim management):\n"
      "1. Alginate — highly absorbent for heavy exudate; frequency 2–5 days\n"
      "2. Silver dressing — reduces bacterial bioburden; bactericidal; "
      "no known resistance; frequency 2–3 days\n"
      "3. Hydrofibre — manages heavy exuding infected wounds; frequency 2–5 days\n"
      "4. Foam — highly absorbent; frequency 2–3 days\n"
      "5. Polymeric membrane dressing — frequency 2–5 days\n"
      "6. Charcoal — odour absorbent; reduces malodour; frequency 2 days; "
      "needs secondary dressing\n"
      "7. Iodine base dressing — AVOID if thyroid disorder\n"
      "Antibiotic: Yes, based on C&S report of infected tissue.\n"
      "Surgery: Surgical/mechanical debridement is STRONGLY recommended. "
      "May need repeated debridement."
    ),
    "reference_contexts": [
      "Wet, infected wound with >25% slough/necrotic tissue: 1.Alginate 2.Silver dressing 3.Hydrofiber 4.Foam 5.Polymeric membrane 6.Charcoal 7.Iodine base dressing. Antibiotic: Yes based on C&S. Surgical/mechanical debridement strongly recommended. May need repeated debridement.",
      "Charcoal: odour absorbent. Reduces odour. Needs secondary dressing. Frequency: 2 days.",
      "Wound types 6, 7 and 8: need referral — extensive wound care such as surgical debridement, vacuum dressing. Systemic complications such as sepsis and severe cellulitis."
    ],
  },

  # ──────────────────────────────────────────────────────────────────────────
  # NOTES OVERRIDE — infection signs not captured in structured labels
  # Documents: wound_app_v4.py ClinicalSignalExtractor logic
  # ──────────────────────────────────────────────────────────────────────────
  {
    "synthesizer_name": "notes_infection_override",
    "time_inputs": {
      "necrotic_pct": 5, "slough_pct": 15, "granulation_pct": 80,
      "infection": "Not infected", "moisture": "Moderate", "edge": "Advancing"
    },
    "user_input": (
      "Structured assessment form marked 'not infected' but clinician notes report: "
      "wound has become more painful over the last 3 days, there is increased warmth "
      "and redness around wound edges, and cloudy exudate was noted today. "
      "Patient also complains of foul odour from the wound."
    ),
    "reference": (
      "CLINICAL NOTES CONTAIN INFECTION INDICATORS — override applies.\n"
      "Despite structured label showing 'not infected', the notes contain multiple "
      "infection red flags:\n"
      "  — Increased wound pain (sign of acute infection)\n"
      "  — Periwaound redness/erythema (sign of local infection)\n"
      "  — Wound warmth (sign of inflammation/infection)\n"
      "  — Cloudy exudate (bacterial infection indicator)\n"
      "  — Foul/offensive wound odour (strong indicator of infection or colonisation)\n\n"
      "Treat as infected wound. This maps to Wound Type 4 (Wet, infected, <25% slough/necrotic).\n"
      "Recommended dressings: Silver dressing (bactericidal, no known resistance, "
      "frequency 2–3 days), or Alginate, Foam, Hydrofibre, or Polymeric membrane dressing.\n"
      "Antimicrobial component is MANDATORY based on clinical signs.\n"
      "Antibiotic: Yes, based on C&S report.\n"
      "Reassess in 48 hours. If spreading erythema, escalate care."
    ),
    "reference_contexts": [
      "Silver: To reduce bacterial bioburden in infected wounds. Bactericidal. No known resistance. Frequency: 2 to 3 days.",
      "Wet, infected wound with <25% slough/necrotic tissue: 1.Alginate 2.Foam 3.Silver 4.Hydrofiber 5.Polymeric membrane 6.Iodine base dressing.",
      "Clinical notes contain infection indicators: foul/offensive wound odour (strong indicator of infection), increased wound pain (sign of acute infection), periWound redness (possible local infection), wound warmth (sign of inflammation/infection), cloudy exudate (bacterial infection indicator)."
    ],
  },

  # ──────────────────────────────────────────────────────────────────────────
  # DIABETIC FOOT — with notes, AJGP Table 1 + WCM Algorithm
  # Documents: AJGP Table 1, WCM p.167-169 Appendix 2
  # ──────────────────────────────────────────────────────────────────────────
  {
    "synthesizer_name": "diabetic_foot_wound",
    "time_inputs": {
      "necrotic_pct": 20, "slough_pct": 30, "granulation_pct": 50,
      "infection": "Infected", "moisture": "Moderate", "edge": "Non-advancing"
    },
    "user_input": (
      "Patient is diabetic, type 2. Wound on plantar surface of right foot. "
      "Clinically infected — erythema, warmth. 50% non-viable tissue (20% necrotic, "
      "30% slough). Moderate exudate. Wound not progressing. "
      "Check pedal pulses requested."
    ),
    "reference": (
      "Wound Type 7 (Dry/moderate, infected, >25% slough/necrotic) — diabetic foot.\n"
      "REFERRAL REQUIRED: Check pedal pulses and sensation. If there is poor perfusion, "
      "referral to a diabetic foot clinic or vascular surgeon is recommended.\n"
      "Dressing strategy per AJGP guidelines for diabetic foot:\n"
      "Apply a primary antimicrobial dressing product with secondary dressing "
      "according to exudate level:\n"
      "  — Moderate exudate: silicone foam as secondary\n"
      "  — Antimicrobial primary: Silver dressing (bactericidal, no known resistance), "
      "or iodine-based (AVOID if thyroid disorder)\n"
      "Silicone foams on feet, if applied, should be WITHOUT borders and anchored "
      "with tape or bandages.\n"
      "Antibiotic: Yes, based on C&S report.\n"
      "Surgery: Surgical/mechanical debridement strongly recommended for "
      ">25% non-viable infected tissue."
    ),
    "reference_contexts": [
      "Diabetic foot: Apply a primary antimicrobial dressing product with secondary dressing according to exudate: 1.low exudate – low-absorbent pad, 2.moderate exudate – silicone foam, 3.high exudate – absorbent pad. Check pedal pulses and sensation; if poor perfusion, referral to diabetic foot clinic or vascular surgeon.",
      "Silicone foams on feet, if applied, should be without borders and anchored with tape or bandages.",
      "Dry, infected wound with >25% slough/necrotic tissue: 1.Silver dressing 2.Hydrogel 3.Hydrocolloid 4.Iodine base dressing 5.Polymeric membrane. Surgical/mechanical debridement strongly recommended."
    ],
  },

  # ──────────────────────────────────────────────────────────────────────────
  # HIGH-RISK PATIENT — diabetic, wound not progressing 6 weeks
  # Documents: AJGP Table 1, WCM Algorithm
  # ──────────────────────────────────────────────────────────────────────────
  {
    "synthesizer_name": "diabetic_high_risk_nonhealing",
    "time_inputs": {
      "necrotic_pct": 10, "slough_pct": 20, "granulation_pct": 70,
      "infection": "Not infected", "moisture": "Low", "edge": "Non-advancing"
    },
    "user_input": (
      "Patient is diabetic with peripheral neuropathy. Plantar foot wound, "
      "not infected currently. Dry wound, minimal exudate. 30% slough and necrosis. "
      "Wound has not progressed despite appropriate dressing changes for 6 weeks. "
      "Patient is ambulatory — no offloading in use."
    ),
    "reference": (
      "Diabetic neuropathic foot ulcer — elevated risk requiring specialist referral.\n"
      "Per AJGP guidelines: Check pedal pulses and sensation. If poor perfusion, "
      "refer to diabetic foot clinic or vascular surgeon.\n"
      "Dressing per wound profile (Wound Type 3/5 — dry, <25–30% slough, no active infection):\n"
      "Antimicrobial primary dressing as precaution given diabetic status, "
      "with secondary dressing based on low exudate level (low-absorbent pad).\n"
      "Silicone foam on feet must be applied WITHOUT borders and anchored with bandages.\n"
      "Wound not progressing for 6 weeks = referral indicated.\n"
      "Non-healing wound definition: wound with no signs of healing process "
      "within 2 to 4 weeks after appropriate intervention.\n"
      "Note: dressing alone will not heal a neuropathic plantar ulcer — "
      "offloading is essential (total contact cast, removable cast walker, or felt foam)."
    ),
    "reference_contexts": [
      "Diabetic foot: Apply a primary antimicrobial dressing product with secondary dressing. Check pedal pulses and sensation; if poor perfusion, referral to diabetic foot clinic or vascular surgeon. Silicone foams on feet without borders and anchored with tape or bandages.",
      "Non-healing wound: Any wound that has no signs of healing process within 2 to 4 weeks after appropriate intervention.",
      "Off-loading: elevated plantar pressure is a causative factor in development of plantar ulcers in diabetic patients. Off-loading methods include total contact casts, removable cast walkers, felt foam."
    ],
  },

  # ──────────────────────────────────────────────────────────────────────────
  # SKIN TEAR — elderly, fragile skin
  # Documents: AJGP Table 1 and Table 2
  # ──────────────────────────────────────────────────────────────────────────
  {
    "synthesizer_name": "skin_tear_elderly",
    "time_inputs": {
      "necrotic_pct": 0, "slough_pct": 0, "granulation_pct": 100,
      "infection": "Not infected", "moisture": "Moderate", "edge": "Advancing"
    },
    "user_input": (
      "Skin tear on forearm of 82-year-old patient. Flap is intact and viable. "
      "Minimal bleeding at edges. Fragile, papery skin around wound. "
      "No signs of infection. Previous nurse applied adhesive bordered foam — "
      "is this correct?"
    ),
    "reference": (
      "Skin tear management (AJGP Table 1):\n"
      "INCORRECT dressing choice — adhesive bordered foam can cause further skin tears "
      "on fragile skin. Correct approach:\n\n"
      "Primary dressing: Apply SILICONE-COATED foam dressing directly over the wound.\n"
      "  — Reposition the skin flap before applying the dressing.\n"
      "  — Do NOT use any ADHESIVE products on fragile skin — they may contribute to "
      "further skin tears, especially on forearms and hands of the elderly.\n"
      "  — Non-silicone foam types should be avoided in patients with fragile skin.\n\n"
      "Barrier wipe: Use a barrier wipe UNDER the foam to secure application, "
      "reduce maceration, and protect the skin on removal.\n\n"
      "Removal: Use remover wipes when removing the dressing from fragile skin. "
      "Remove in a direction that does NOT disturb viable tissue edges and flaps.\n\n"
      "If bleeding: Apply haemostatic alginate dressing as PRIMARY under the silicone foam."
    ),
    "reference_contexts": [
      "Skin tears: Apply silicone-covered foam dressing directly over the wound. Do not use any adhesive products on fragile skin as they may contribute to further skin tears, especially on forearms and hands of the elderly.",
      "Using a barrier wipe under the foam aids to secure application, reduce maceration and protect the skin on removal of the dressing.",
      "Remover wipes should be used when removing a dressing from fragile skin. Removal of the dressing should be done in a direction that does not disturb viable tissue edges and flaps.",
      "If bleeding, apply haemostatic alginate dressing as primary dressing under a silicone-coated foam dressing. Foam: Nonsilicone types should be avoided in patients with fragile skin."
    ],
  },

  # ──────────────────────────────────────────────────────────────────────────
  # POSTOPERATIVE WOUND — no exudate, primary intention
  # Documents: AJGP Table 1 and Table 2
  # ──────────────────────────────────────────────────────────────────────────
  {
    "synthesizer_name": "postoperative_clean",
    "time_inputs": {
      "necrotic_pct": 0, "slough_pct": 0, "granulation_pct": 100,
      "infection": "Not infected", "moisture": "Low", "edge": "Advancing"
    },
    "user_input": (
      "Clean postoperative incision, day 3. Healing by primary intention. "
      "No exudate visible. Sutured. No signs of infection or dehiscence. "
      "Patient wants to shower daily. What dressing is appropriate?"
    ),
    "reference": (
      "Postoperative wound management (AJGP Table 1):\n"
      "For wounds WITHOUT exudate: Dress over sutures with a FILM dressing or "
      "thin hydrocolloid.\n\n"
      "Film dressing characteristics:\n"
      "  — Permeable to gas but impermeable to bacteria and liquid\n"
      "  — Transparent, facilitates easy monitoring of the wound\n"
      "  — Waterproof — suitable for showering\n"
      "  — Most useful for postoperative wounds healing by primary intention\n"
      "  — Wear time: 1–4 days\n"
      "  — Caution if patient has particularly vulnerable skin — may consider "
      "a skin protectant (barrier) product underneath\n\n"
      "If wound dehisces: organise prompt surgical review.\n"
      "Do NOT use foam or alginate — no exudate present."
    ),
    "reference_contexts": [
      "Postoperative wounds: For wounds without exudate, dress over sutures with a film or thin hydrocolloid. In case of wound dehiscence, organise prompt surgical review.",
      "Film dressings: permeable to gas but impermeable to bacteria and liquid. Useful on superficial wounds with minimum exudate. Wear time 1-4 days. Most useful for postoperative wounds healing by primary intention as they facilitate easy monitoring of the wound.",
      "Films: transparent with measurement grid. Bacterial barrier. Waterproof. Breathable. Frequency of dressing change: 2-5 days depending on wound."
    ],
  },

  # ──────────────────────────────────────────────────────────────────────────
  # SMALL SUPERFICIAL BURN — hand location
  # Documents: AJGP Table 1
  # ──────────────────────────────────────────────────────────────────────────
  {
    "synthesizer_name": "burn_hand_referral",
    "time_inputs": {
      "necrotic_pct": 0, "slough_pct": 5, "granulation_pct": 95,
      "infection": "Not infected", "moisture": "Moderate", "edge": "Advancing"
    },
    "user_input": (
      "Small superficial partial-thickness burn from hot water. Wound is on the palm "
      "of the right hand. Pink, moist wound bed, blistered. No signs of infection. "
      "Initial first aid with cool running water was already done."
    ),
    "reference": (
      "Small superficial burn management (AJGP Table 1):\n"
      "After initial first aid treatment: cover burns area with HYDROGEL, "
      "HYDROCOLLOID, or FILM dressing.\n\n"
      "REFERRAL REQUIRED — MANDATORY for hand location:\n"
      "Refer to burns specialist for burns located on hands, feet, face, or genitalia, "
      "regardless of depth or size. Hand burns always require specialist review.\n\n"
      "Dressing options:\n"
      "  — Hydrogel: provides moist environment, reduces pain, rehydrates wound bed; "
      "frequency 2–3 days\n"
      "  — Hydrocolloid: maintains moist environment, promotes autolysis; "
      "frequency 2–5 days\n"
      "  — Film dressing: transparent for monitoring, waterproof; wear time 1–4 days\n\n"
      "Do NOT rupture blisters unless tense and clinically indicated."
    ),
    "reference_contexts": [
      "Small superficial burns: After initial first aid treatment, cover burns area with hydrogel or hydrocolloid or film. Refer to burns specialist for burns that are deep or infected or located on hands, feet, face or genitalia.",
      "Hydrogel: Rehydrate, debride and deslough the wound. Promote moist healing. Comfortable, provides moist environment and reduces pain. Frequency: 2-3 days.",
      "Hydrocolloid: Provide moist environment. Absorb exudates. Bacterial barrier. Cleans and debrids by autolysis. Easy to use. Promotes granulation tissue. Frequency: 2 to 5 days."
    ],
  },

  # ──────────────────────────────────────────────────────────────────────────
  # DRESSING CHANGE FREQUENCY — saturation triggers
  # Documents: SFP p.18, WCM p.135
  # ──────────────────────────────────────────────────────────────────────────
  {
    "synthesizer_name": "dressing_change_saturation",
    "time_inputs": {
      "necrotic_pct": 5, "slough_pct": 10, "granulation_pct": 85,
      "infection": "Not infected", "moisture": "High", "edge": "Advancing"
    },
    "user_input": (
      "Foam dressing applied 3 days ago. On examination today: dressing is soiled, "
      "fluid has struck through the outer layer, and the edges are curling. "
      "When should the dressing be changed?"
    ),
    "reference": (
      "Change the dressing NOW — immediately.\n\n"
      "Per clinical guidelines, a dressing must be changed when ANY of the following "
      "are present:\n"
      "  1. Dressing is soiled\n"
      "  2. Dressing is loose or slipping\n"
      "  3. Edges are curling\n"
      "  4. Fluid has accumulated / dressing is saturated (strikethrough visible)\n\n"
      "All four conditions are present. Change is mandatory regardless of how "
      "recently the dressing was applied or what the scheduled frequency is.\n\n"
      "Important principle: Manufacturer recommendations on frequency are GUIDELINES only — "
      "clinical judgment always takes precedence.\n\n"
      "For this wound profile (granulating, high exudate, no infection — Type 2/clean wet): "
      "Consider alginate or hydrofibre as PRIMARY under foam to increase absorption "
      "capacity and extend wear time. Foam frequency: 2–3 days under normal conditions."
    ),
    "reference_contexts": [
      "If the dressing is soiled, loose, slipping or curling at the edges, it is obvious that it should be changed. If there is accumulation of fluid and/or debris and the dressing is saturated, it needs change. If infection is present, increased frequencies of change need to be considered.",
      "Most dressings come with manufacturer recommendations on the frequency of change or how long each dressing can maintain its efficacy; however these should only be used as guidelines, clinical judgment still rules.",
      "Foam: Frequency of dressing change: 2 to 3 days or longer if for offloading."
    ],
  },

  # ──────────────────────────────────────────────────────────────────────────
  # NPWT — adjunct only, contraindicated when necrotic
  # Documents: WCM p.159, p.162, p.175, SFP p.25
  # ──────────────────────────────────────────────────────────────────────────
  {
    "synthesizer_name": "npwt_contraindication",
    "time_inputs": {
      "necrotic_pct": 50, "slough_pct": 20, "granulation_pct": 30,
      "infection": "Infected", "moisture": "High", "edge": "Non-advancing"
    },
    "user_input": (
      "Large complex wound. 70% non-viable tissue (50% necrotic eschar, 20% slough). "
      "Infected with heavy exudate. Surgeon is considering applying NPWT vacuum "
      "dressing as the sole treatment. Is NPWT appropriate here?"
    ),
    "reference": (
      "NPWT is NOT appropriate as sole treatment in this wound, for two reasons:\n\n"
      "1. NPWT IS CONTRAINDICATED in necrotic wound bed or eschar — the necrotic tissue "
      "acts as a barrier to new tissue growth. This wound has 50% necrotic tissue.\n"
      "2. NPWT is an ADJUNCT treatment only. It does not replace surgical procedures "
      "and is not a panacea. It prepares the wound bed for a greater chance of "
      "successful closure.\n\n"
      "Additional NPWT contraindications:\n"
      "  — Untreated infection (deep extension of infectious focus)\n"
      "  — Clotting disorders (risk of bleeding)\n"
      "  — Neoplastic tissue in wound area\n\n"
      "Correct approach for this wound (Type 8 — wet, infected, >25% necrotic):\n"
      "First: Surgical/mechanical debridement (strongly recommended).\n"
      "Interim dressing: Alginate, Silver dressing, Hydrofibre, Foam, Charcoal, "
      "Polymeric membrane, or Iodine-based dressing.\n"
      "Antibiotic: Yes, based on C&S.\n"
      "NPWT may be considered AFTER adequate debridement."
    ),
    "reference_contexts": [
      "NPWT contraindications: Clotting disorders, Necrotic wound bed or eschar (barrier to new tissue growth), Untreated infection, Neoplastic tissue in the wound area.",
      "NPWT is only an adjunct to the management of chronic, acute and difficult wounds and it is not a panacea. NPWT prepares wound bed for a greater chance of successful closure. NPWT does not replace surgical procedures.",
      "Wet, infected wound with >25% slough/necrotic tissue: 1.Alginate 2.Silver dressing 3.Hydrofiber 4.Foam 5.Polymeric membrane 6.Charcoal 7.Iodine base dressing. Surgical/mechanical debridement strongly recommended."
    ],
  },

  # ──────────────────────────────────────────────────────────────────────────
  # SILVER contraindicated on clean granulating wound
  # Documents: WCM p.165, GP p.13
  # ──────────────────────────────────────────────────────────────────────────
  {
    "synthesizer_name": "silver_contraindicated_granulating",
    "time_inputs": {
      "necrotic_pct": 0, "slough_pct": 0, "granulation_pct": 100,
      "infection": "Not infected", "moisture": "Low", "edge": "Advancing"
    },
    "user_input": (
      "Wound is 100% healthy pink granulation tissue. No slough, no necrosis. "
      "Not infected. Dry/minimal exudate. Wound is healing well and edges advancing. "
      "Previous clinician applied a silver dressing. Is this appropriate?"
    ),
    "reference": (
      "INCORRECT dressing choice — silver dressing is contraindicated in this wound.\n\n"
      "Per the Wound Care Algorithm (Type 1 — Clean healthy granulating wound):\n"
      "All types of dressing material are suitable EXCEPT:\n"
      "  — Silver dressings\n"
      "  — Charcoal dressings\n"
      "  — Special advanced dressing materials\n\n"
      "Silver and charcoal are excluded from clean granulating wounds as they are "
      "not indicated and may be inappropriate for healthy tissue.\n\n"
      "Correct dressing for this wound:\n"
      "  — Film dressing (wear time 2–5 days): protects, allows monitoring, waterproof\n"
      "  — Hydrocolloid (2–5 days): promotes autolysis, moist environment\n"
      "  — Non-adherent silicone or tulle: for lightly exuding or granulating wounds\n"
      "  — Foam (2–3 days): if providing cushioning/protection\n"
      "No antibiotic needed. Wound may be ready for secondary closure."
    ),
    "reference_contexts": [
      "Clean, healthy granulating wound: All types of dressing material except silver, charcoal and special advanced dressing materials. No antibiotic. Ready for secondary wound closure or continue dressing till wound heals by secondary intention.",
      "Non-adherent dressings (porous silicone or tulles): often used as primary dressing for lightly exuding or granulating wounds.",
      "Frequency of wound dressing varies depending on type of wound and also dressing material used."
    ],
  },

  # ──────────────────────────────────────────────────────────────────────────
  # WOUND ASSESSMENT — T.I.M.E. principle application
  # Documents: GP p.12, WCM Algorithm
  # ──────────────────────────────────────────────────────────────────────────
  {
    "synthesizer_name": "time_assessment_mixed_wound",
    "time_inputs": {
      "necrotic_pct": 25, "slough_pct": 25, "granulation_pct": 50,
      "infection": "Infected", "moisture": "Moderate", "edge": "Non-advancing"
    },
    "user_input": (
      "50% non-viable tissue: 25% black necrosis, 25% yellow slough. "
      "50% granulation. Infected — pus, pain, malodour. Moderate exudate. "
      "Wound edges non-advancing. What does T.I.M.E. assessment indicate "
      "and what dressing should be used?"
    ),
    "reference": (
      "T.I.M.E. Wound Assessment (per Garis Panduan / WCM):\n"
      "  T (Tissue): 50% non-viable (25% necrotic + 25% slough) — >25% threshold met. "
      "Debridement is needed.\n"
      "  I (Infection/Inflammation): Signs present — pus, pain, malodour. Treat as infected.\n"
      "  M (Moisture imbalance): Moderate exudate — dressing must manage moisture balance.\n"
      "  E (Epidermal margin): Non-advancing — wound is stalled.\n\n"
      "Algorithm result: Wound Type 7 (>25% non-viable, infected, dry/moderate) — "
      "with moderate exudate trending toward Type 8.\n\n"
      "REFERRAL REQUIRED — wound types 7 and 8 should be referred to hospital.\n"
      "Interim dressings:\n"
      "  Type 7: Silver dressing, Hydrogel, Hydrocolloid, Iodine-based, Polymeric membrane\n"
      "  Type 8 (if heavy exudate): + Alginate, Hydrofibre, Foam, Charcoal\n"
      "Antibiotic: Yes, based on C&S.\n"
      "Surgery: Surgical/mechanical debridement strongly recommended."
    ),
    "reference_contexts": [
      "T.I.M.E. Wound Assessment: T=Tissue (viable/non-viable), I=Infection/Inflammation (signs and symptoms such as pus, pain, malodour), M=Moisture imbalance (exudate level dry/minimal or moderate/wet), E=Epidermal margin (advancing or non-advancing).",
      "Dry, infected wound with >25% slough/necrotic tissue: Silver dressing, Hydrogel, Hydrocolloid, Iodine base dressing, Polymeric membrane. Surgical/mechanical debridement strongly recommended.",
      "Wound types 6, 7 and 8: should be referred to hospital. Require extensive wound care such as surgical debridement, vacuum dressing. Systemic complications such as sepsis and cellulitis."
    ],
  },

  # ──────────────────────────────────────────────────────────────────────────
  # DRESSING SELECTION RATIONALE — foam vs alginate vs hydrofibre
  # Documents: WCM p.134-135, SFP p.19
  # ──────────────────────────────────────────────────────────────────────────
  {
    "synthesizer_name": "dressing_selection_heavy_exudate",
    "time_inputs": {
      "necrotic_pct": 0, "slough_pct": 5, "granulation_pct": 95,
      "infection": "Not infected", "moisture": "High", "edge": "Advancing"
    },
    "user_input": (
      "Heavily exuding granulating wound. Foam dressing changed daily as it saturates "
      "within 24 hours. Periwound skin showing early maceration. "
      "What dressing options better manage very heavy exudate?"
    ),
    "reference": (
      "For very heavy exudate where foam alone saturates within 24 hours, "
      "consider these alternatives or combinations (Wound Type 2 — clean wet wound):\n\n"
      "1. ALGINATE (primary dressing):\n"
      "   — Absorbs up to 20 times its own weight — highest absorption capacity\n"
      "   — Forms a gel on contact with wound fluid that absorbs and contains exudate\n"
      "   — Promotes healing by maintaining physiologically moist environment\n"
      "   — Available in sheets or rope form; can fill cavities\n"
      "   — Always needs a secondary dressing over it\n"
      "   — Frequency: 2–5 days\n"
      "   — Note: residue must be washed off during cleansing\n\n"
      "2. HYDROFIBRE (primary dressing):\n"
      "   — Manages heavy exuding wounds\n"
      "   — Creates soft cohesive gel that conforms to wound surface\n"
      "   — Comfortable and non-traumatic on removal\n"
      "   — Reduces risk of maceration\n"
      "   — Needs secondary dressing\n"
      "   — Frequency: 2–5 days\n\n"
      "3. FOAM (secondary or alone):\n"
      "   — Highly absorbent but less so than alginate for very heavy exudate\n"
      "   — Frequency: 2–3 days\n\n"
      "For macerated periwound skin: apply skin protection/barrier before dressing."
    ),
    "reference_contexts": [
      "Alginates partially dissolve on contact with wound fluid to form a gel able to absorb up to 20 times own weight. Recommended for moderate to heavy level of exudate. Promote healing by maintaining physiologically moist environment. Available in sheet or rope form. Always covered with secondary dressing. Residue has to be washed off. Frequency: 2 to 5 days.",
      "Hydrofibre: Manage heavy exuding wounds. Maintains moist healing environment. Longer wear time. Comfortable and non-traumatic upon removal. Reduces risk of maceration. Needs secondary dressings. Frequency: 2 to 5 days.",
      "Foam: Absorbent. Cushioning. Conforms to body contours. Highly absorbent. Provides protection. Bacterial and waterproof. Frequency: 2 to 3 days or longer."
    ],
  },

]

# Add empty fields that the evaluation loop will fill in
for record in testset:
    record["answer"] = ""
    record["retrieved_contexts"] = []

# Save
json_path = os.path.join(OUTPUT_DIR, "wound_testset_curated.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(testset, f, indent=2, ensure_ascii=False)
print(f"Saved {len(testset)} test cases → {json_path}")

# CSV
import csv
csv_path = os.path.join(OUTPUT_DIR, "wound_testset_curated.csv")
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=[
        "synthesizer_name","time_inputs","user_input","reference",
        "reference_contexts","answer","retrieved_contexts"
    ])
    w.writeheader()
    for r in testset:
        w.writerow({
            "synthesizer_name": r["synthesizer_name"],
            "time_inputs": str(r["time_inputs"]),
            "user_input": r["user_input"],
            "reference": r["reference"],
            "reference_contexts": str(r["reference_contexts"]),
            "answer": r["answer"],
            "retrieved_contexts": str(r["retrieved_contexts"]),
        })
print(f"CSV saved → {csv_path}")

# Print summary
print("\n=== TEST CASE SUMMARY ===")
for t in testset:
    ti = t["time_inputs"]
    nv = ti["necrotic_pct"] + ti["slough_pct"]
    print(f"  {t['synthesizer_name']:<45} N={ti['necrotic_pct']}% S={ti['slough_pct']}% "
          f"G={ti['granulation_pct']}% | NV={nv}% | "
          f"Inf={ti['infection'][:3]} Moist={ti['moisture'][:3]}")
