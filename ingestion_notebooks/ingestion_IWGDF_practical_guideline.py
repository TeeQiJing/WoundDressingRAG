# =============================================================================
# CELL 1 — Imports & Setup
# =============================================================================
# Source: IWGDF 2023 Practical Guidelines on the Prevention and Management of
# Diabetes-Related Foot Disease (IWGDF 2023 update)
# Sections ingested: Pathophysiology, Foot Ulcer Assessment (SINBAD, infection
# grading, ischaemia thresholds, wound depth/area), Local Ulcer Care, Wound
# Dressings, Adjunctive Treatments, Offloading, Person-centred Care,
# Infection Treatment, Perfusion Restoration.
# Sections OMITTED: Cover page, authors/institutions, abstract, intro,
# prevention education, footwear details, Charcot CNO, organisation of care,
# concluding remarks, acknowledgements, conflict of interest, references,
# Appendix 1 (sensory exam technique), Appendix 2 (ABI/TBI measurement
# technique), Appendix 3 (patient education items).
# =============================================================================

import os
import openai
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

VECTOR_STORE_ID = os.environ.get("VECTOR_STORE_ID", "your_vector_store_id_here")

print("Setup complete.")
print(f"Vector Store ID: {VECTOR_STORE_ID}")


# =============================================================================
# CELL 2 — Helper: upload chunks to vector store
# =============================================================================

import json
import time
import tempfile

def upload_chunks_to_vector_store(chunks: list[dict], vector_store_id: str, batch_size: int = 20):
    """
    Upload a list of chunk dicts to an OpenAI vector store.
    Each dict must have 'text' and 'metadata' keys.
    Mirrors the pattern used across all ingestion notebooks in this project.
    """
    uploaded = 0
    file_ids = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_file_ids = []

        for chunk in batch:
            content = chunk["text"]
            metadata = chunk.get("metadata", {})

            # Write chunk to a temp file
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".txt",
                delete=False,
                encoding="utf-8"
            ) as f:
                f.write(content)
                tmp_path = f.name

            # Upload the file
            with open(tmp_path, "rb") as f:
                response = client.files.create(file=f, purpose="assistants")

            file_id = response.id
            batch_file_ids.append(file_id)
            os.remove(tmp_path)

        # Add batch of files to vector store
        client.beta.vector_stores.file_batches.create_and_poll(
            vector_store_id=vector_store_id,
            file_ids=batch_file_ids
        )

        file_ids.extend(batch_file_ids)
        uploaded += len(batch)
        print(f"  Uploaded {uploaded}/{len(chunks)} chunks...")
        time.sleep(0.5)

    print(f"Done. Total chunks uploaded: {uploaded}")
    return file_ids


# =============================================================================
# CELL 3 — Chunk Set A: Pathophysiology of Diabetes-Related Foot Ulcers
# Rationale: Explains the mechanistic basis (neuropathy, PAD, callus, ischaemia)
# that underpins why wounds develop and how wound type influences care decisions.
# =============================================================================

chunks_pathophysiology = [
    {
        "text": (
            "IWGDF 2023 — Diabetes-Related Foot Disease: Pathophysiology\n\n"
            "Diabetes-related foot disease includes one or more of the following in the foot of a person "
            "with current or previously diagnosed diabetes mellitus: peripheral neuropathy, peripheral artery "
            "disease, infection, ulcer(s), neuro-osteoarthropathy, gangrene, or amputation. Foot ulceration "
            "is among the most serious complications of diabetes and is a source of reduced quality of life "
            "as well as financial costs for the person involved.\n\n"
            "Pathways to ulceration: These ulcers usually develop in a person with diabetes simultaneously "
            "having one or more risk factors, such as diabetes-related peripheral neuropathy and/or peripheral "
            "artery disease (PAD), in combination with a precipitating event. The neuropathy leads to an "
            "insensitive and sometimes deformed foot. Loss of protective sensation, foot deformities, and "
            "limited joint mobility can result in abnormal biomechanical loading of the foot. This produces "
            "high mechanical stress in some areas, the response to which is usually thickened skin (callus). "
            "The callus then leads to a further increase in the loading of the foot, often with subcutaneous "
            "haemorrhage and eventually skin ulceration. In addition, in people with neuropathy, minor trauma "
            "(e.g., from ill-fitting shoes, or an acute mechanical or thermal injury) can precipitate ulceration "
            "of the foot. Whatever the primary cause of ulceration, continued walking on the insensitive foot "
            "impairs healing of the ulcer.\n\n"
            "Role of PAD: PAD, generally caused by atherosclerosis, is present in up to 50% of patients with "
            "a diabetes-related foot ulcer and is an important risk factor for impaired wound healing, gangrene "
            "and lower-extremity amputation. A small percentage of foot ulcers in patients with severe PAD are "
            "purely ischaemic; these are usually painful and may follow minor trauma. The majority of foot "
            "ulcers, however, are either purely neuropathic or neuro-ischaemic, i.e., the combination of "
            "neuropathy and ischaemia. In people with diabetes with neuro-ischaemic ulcers, symptoms may be "
            "absent because of the neuropathy, despite severe pedal ischaemia. Although diabetes-related "
            "microangiopathy can be observed in the foot, it does not appear to be the primary cause of either "
            "ulcers or of poor wound healing.\n\n"
            "Ulcer types:\n"
            "- Neuropathic ulcer: Loss of protective sensation (LOPS) present, no PAD.\n"
            "- Neuro-ischaemic ulcer: Both LOPS and PAD present; most common type.\n"
            "- Ischaemic ulcer: PAD present, no LOPS; usually painful.\n\n"
            "To reduce the burden of diabetes-related foot disease, strategies are required that include "
            "elements of prevention, patient and staff education, standardised assessment and classification, "
            "multi-disciplinary treatment, and close monitoring."
        ),
        "metadata": {
            "source": "IWGDF_2023_Practical_Guidelines",
            "section": "Section 2 – Pathophysiology",
            "topic": "diabetes foot ulcer pathophysiology, neuropathy, PAD, wound healing",
            "chunk_id": "IWGDF_PG_001"
        }
    },
]

print(f"Chunk set A (Pathophysiology): {len(chunks_pathophysiology)} chunk(s)")


# =============================================================================
# CELL 4 — Chunk Set B: Foot Ulcer Assessment — SINBAD Classification System
# Rationale: SINBAD directly maps to TIME framework. Site = wound location,
# Ischemia = perfusion (M in TIME context), Neuropathy = healing barrier,
# Bacterial infection = I in TIME, Area = wound extent, Depth = tissue type/T.
# =============================================================================

chunks_sinbad = [
    {
        "text": (
            "IWGDF 2023 — Foot Ulcer Classification: SINBAD System\n\n"
            "When a person with diabetes presents with a foot ulcer, the ulcer should be classified "
            "following the assessment of the six items of the SINBAD system. These items serve as a basic "
            "guide for further treatment, and facilitate communication about the characteristics of an ulcer "
            "between health professionals.\n\n"
            "S — Site: Describe where the ulcer is located on the foot. This includes description of "
            "forefoot, midfoot or hindfoot, and differentiate between plantar, interdigital, medial, lateral "
            "or dorsal.\n\n"
            "I — Ischemia: Assess if pedal blood flow is intact (at least one palpable pulse), or if there "
            "is clinical evidence of reduced blood flow. Further, examine the arterial pedal wave forms (with "
            "a Doppler instrument), measure the ankle and toe pressures, and calculate the ankle-brachial "
            "index (ABI) and toe-brachial index (TBI). PAD is less likely in the presence of triphasic or "
            "biphasic pedal Doppler waveforms, an ABI 0.9–1.3, and a TBI ≥0.70. In selected cases, "
            "transcutaneous pressure of oxygen (TcpO2) can be useful. The level of perfusion deficit can "
            "help estimate the likelihood of healing and amputation, but better risk estimation is obtained "
            "when wound depth and foot infection severity are also taken into account, as in the WIfI scoring "
            "system.\n\n"
            "N — Neuropathy: Assess if protective sensation is intact or lost (using 10g Semmes-Weinstein "
            "monofilament, 128 Hz tuning fork, or Ipswich Touch test).\n\n"
            "B — Bacterial infection: Assess if clinical infection is present. Diagnose infection by the "
            "presence of at least two clinical signs or symptoms of inflammation (redness, warmth, "
            "induration, pain/tenderness) or purulent secretions. These signs may be blunted by neuropathy "
            "or ischaemia, and systemic findings (e.g., pain, fever, leucocytosis) are often absent in mild "
            "and moderate infections.\n\n"
            "Infection grading (IWGDF/IDSA system):\n"
            "- Mild infection: Superficial ulcer with minimal cellulitis (cellulitis ≤2 cm, no deeper "
            "tissue involvement, no systemic signs).\n"
            "- Moderate infection: Ulcer deeper than skin or more extensive cellulitis, with or without "
            "abscess (involves deeper structures — tendon, joint, bone — or cellulitis >2 cm).\n"
            "- Severe infection: Accompanied by systemic signs of sepsis (fever, tachycardia, hypotension, "
            "leucocytosis), with or without osteomyelitis.\n\n"
            "A — Area: Measure ulcer area and express in cm².\n\n"
            "D — Depth: Assess ulcer depth and classify as:\n"
            "- Confined to skin and subcutaneous tissue\n"
            "- Reaching muscle or tendon\n"
            "- Reaching bone\n"
            "Determining depth can be difficult, especially in the presence of overlying callus or necrotic "
            "tissue. Debride any neuropathic or neuro-ischaemic ulcer surrounded by callus or containing "
            "necrotic soft tissue at initial presentation, or as soon as possible. Do not debride a "
            "non-infected ulcer that has signs of severe ischaemia. Neuropathic ulcers can usually be "
            "debrided without the need for local anaesthesia.\n\n"
            "Ulcer type classification:\n"
            "- Neuropathic: LOPS present, no PAD\n"
            "- Neuro-ischaemic: LOPS and PAD both present\n"
            "- Ischaemic: PAD present, no LOPS\n\n"
            "The SINBAD system is simple and quick to use and contains the necessary information to allow "
            "for triage by a specialist team. In addition, infection severity should be classified according "
            "to the IWGDF/IDSA system and ischaemia as part of the WIfI system."
        ),
        "metadata": {
            "source": "IWGDF_2023_Practical_Guidelines",
            "section": "Section 4.1.1 – SINBAD Classification",
            "topic": "SINBAD wound classification, wound assessment, infection grading, IWGDF/IDSA, ischaemia, depth, area",
            "chunk_id": "IWGDF_PG_002"
        }
    },
]

print(f"Chunk set B (SINBAD): {len(chunks_sinbad)} chunk(s)")


# =============================================================================
# CELL 5 — Chunk Set C: Infection Assessment — Microbiology & Bone Involvement
# Rationale: Directly relevant to the "I" (Infection) parameter in TIME.
# Guides wound culture approach, osteomyelitis suspicion, and organism context.
# =============================================================================

chunks_infection_assessment = [
    {
        "text": (
            "IWGDF 2023 — Infection Assessment: Microbiology, Abscess & Osteomyelitis\n\n"
            "For clinically infected wounds, obtain a tissue specimen for culture (and Gram-stained smear, "
            "if available) by curettage or biopsy. Avoid using a swab. Consider bone biopsy in case of "
            "osteomyelitis.\n\n"
            "Microbiology context:\n"
            "- Staphylococcus aureus (alone, or with other organisms) is the predominant pathogen in most "
            "cases of superficial infections.\n"
            "- Chronic and more severe infections are often polymicrobial, with aerobic gram-negative rods "
            "especially in warmer climates and obligate anaerobes accompanying the gram-positive cocci.\n"
            "- Causative pathogens and their antibiotic susceptibilities vary by geographic, demographic and "
            "clinical situation.\n\n"
            "Detecting abscess:\n"
            "- An abscess is more likely in case of fever, high CRP or ESR levels, but normal findings do "
            "not exclude a foot abscess.\n"
            "- When in doubt, perform MRI.\n\n"
            "Detecting osteomyelitis:\n"
            "- Determine if it is possible to visualise or touch bone with a sterile metal probe "
            "(probe-to-bone test).\n"
            "- Obtain plain radiographs in persons with ulcers deeper than skin, tissue gas or foreign body.\n"
            "- Osteomyelitis is likely in case of a positive probe-to-bone test in combination with "
            "abnormalities on plain X-ray.\n"
            "- High levels of ESR, CRP, or procalcitonin further support this diagnosis.\n"
            "- When in doubt, perform an MRI; if MRI is not possible, consider other techniques (e.g., "
            "radionuclide or PET scans).\n\n"
            "Spread risk: If not properly treated, infection can rapidly spread to underlying tissues and "
            "foot compartments, in particular in the presence of PAD. Therefore, explore the depth of the "
            "ulcer carefully at initial assessment."
        ),
        "metadata": {
            "source": "IWGDF_2023_Practical_Guidelines",
            "section": "Section 4.1.1 – Infection Assessment",
            "topic": "wound infection, osteomyelitis, probe-to-bone, abscess, microbiology, Staphylococcus aureus, TIME I parameter",
            "chunk_id": "IWGDF_PG_003"
        }
    },
]

print(f"Chunk set C (Infection assessment): {len(chunks_infection_assessment)} chunk(s)")


# =============================================================================
# CELL 6 — Chunk Set D: Person-Related Factors Affecting Ulcer Healing
# Rationale: Systemic factors directly influence wound healing outcomes and
# dressing choice. Relevant to interpreting TIME findings holistically.
# =============================================================================

chunks_person_factors = [
    {
        "text": (
            "IWGDF 2023 — Person-Related Factors Affecting Ulcer Healing\n\n"
            "Apart from systematic evaluation of the ulcer, the foot and the leg, also consider "
            "person-related factors that can affect ulcer healing and treatment decisions. These factors "
            "include:\n"
            "- Kidney function / end-stage renal disease\n"
            "- Oedema\n"
            "- Malnutrition\n"
            "- Poor metabolic control (hyperglycaemia)\n"
            "- Depression or other psycho-social problems\n"
            "- Frailty\n\n"
            "Determining the cause of the ulcer:\n"
            "Always try to determine the precipitating event that led to ulceration, as this information "
            "is relevant both for treatment plans and for prevention of recurrence. Look for abnormal "
            "walking patterns, deformities, bony prominences and other foot abnormalities (supine and "
            "standing) that could have contributed to ulceration. Wearing ill-fitting shoes and walking "
            "barefoot are practices that frequently lead to foot ulceration, even in patients with "
            "exclusively ischaemic ulcers. Therefore, meticulously examine shoes and footwear behaviour "
            "in every patient with a foot ulcer as part of cause determination.\n\n"
            "Person-centred care — systemic treatment targets (Section 4.2.4):\n"
            "In addition to local wound care, the following systemic factors should also be treated "
            "where possible:\n"
            "- Optimise glycaemic control, if necessary, with insulin.\n"
            "- Treat oedema or malnutrition, if present.\n"
            "- Treat cardiovascular risk factors.\n"
            "- Treat depression or other psycho-social difficulties.\n\n"
            "Clinical implication for dressing RAG: Oedema increases exudate levels and influences "
            "moisture balance (TIME M parameter). Poor glycaemic control impairs granulation and "
            "re-epithelialisation. Malnutrition delays collagen synthesis and wound edge advancement "
            "(TIME E parameter). End-stage renal disease is associated with the highest ulcer recurrence "
            "risk (IWGDF risk category 3)."
        ),
        "metadata": {
            "source": "IWGDF_2023_Practical_Guidelines",
            "section": "Sections 4.1.2, 4.1.3, 4.2.4",
            "topic": "wound healing barriers, systemic factors, glycaemic control, oedema, malnutrition, TIME framework context",
            "chunk_id": "IWGDF_PG_004"
        }
    },
]

print(f"Chunk set D (Person-related factors): {len(chunks_person_factors)} chunk(s)")


# =============================================================================
# CELL 7 — Chunk Set E: Infection Treatment (Mild vs Moderate/Severe)
# Rationale: Core to "I" in TIME. Determines whether wound can progress to
# granulation/healing or requires surgical/antibiotic intervention first.
# Antibiotic guidance directly affects dressing choice (infected vs non-infected).
# =============================================================================

chunks_infection_treatment = [
    {
        "text": (
            "IWGDF 2023 — Treatment of Foot Infection (Section 4.2.1)\n\n"
            "Infection of the foot in a person with diabetes presents an immediate threat to the affected "
            "foot and limb. Prompt treatment is required. Based on the IWGDF/IDSA infection guidelines.\n\n"
            "MODERATE OR SEVERE INFECTION (deep or extensive, potentially limb-threatening):\n"
            "1. Urgently evaluate for need for immediate surgical intervention to remove necrotic tissue, "
            "including infected bone, release compartment pressure and drain abscesses.\n"
            "2. Assess for PAD; if present, consider urgent treatment including revascularisation once "
            "infection is under control.\n"
            "3. Initiate empiric, parenteral, broad-spectrum antibiotic therapy, aimed at common "
            "gram-positive and gram-negative bacteria, including obligate anaerobes.\n"
            "4. Adjust (constrain and target, if possible) the antibiotic regimen based on both the "
            "clinical response to empirical therapy and culture and sensitivity results.\n"
            "5. For soft-tissue infections, antibiotic treatment during 1 to 2 weeks will frequently "
            "suffice; a longer duration may be required in case of a slowly resolving infection or "
            "severe PAD.\n"
            "6. Consider conservative treatment for osteomyelitis with antibiotics when there is no need "
            "for incision and drainage to control infection.\n\n"
            "MILD INFECTION (superficial ulcer with limited soft tissue involvement):\n"
            "1. Cleanse and debride all necrotic tissue and surrounding callus.\n"
            "2. Start empiric oral antibiotic therapy targeted at Staphylococcus aureus and "
            "β-haemolytic streptococci (unless there are reasons to consider other, or additional, "
            "likely pathogens).\n\n"
            "Dressing context: In infected wounds, systemic antibiotic therapy is primary treatment. "
            "Local wound care (cleansing, debridement, exudate management) is adjunctive. Antimicrobial "
            "dressings and topical antiseptics are NOT well-supported as primary treatment of infection "
            "per IWGDF 2023 — see local ulcer care section."
        ),
        "metadata": {
            "source": "IWGDF_2023_Practical_Guidelines",
            "section": "Section 4.2.1 – Infection Treatment",
            "topic": "diabetic foot infection treatment, IWGDF/IDSA, antibiotics, osteomyelitis, surgical debridement, TIME I parameter",
            "chunk_id": "IWGDF_PG_005"
        }
    },
]

print(f"Chunk set E (Infection treatment): {len(chunks_infection_treatment)} chunk(s)")


# =============================================================================
# CELL 8 — Chunk Set F: Ischaemia Assessment Thresholds & Revascularisation
# Rationale: Ischaemia (TIME M context: perfusion) is a critical barrier to
# healing. Thresholds (ABI, TBI, ankle pressure, TcpO2) directly determine
# whether dressing/offloading alone will suffice or urgent vascular referral
# is needed before any wound healing intervention can succeed.
# =============================================================================

chunks_ischaemia = [
    {
        "text": (
            "IWGDF 2023 — Ischaemia Assessment Thresholds & Revascularisation (Section 4.2.2)\n\n"
            "Ischemia in the lower extremity affects the healing potential of a foot ulcer. If ischaemia "
            "has been found during assessment, its treatment should always be considered. Based on the "
            "intersocietal IWGDF/ESVS/SVS guidelines.\n\n"
            "VASCULAR DIAGNOSTIC THRESHOLDS:\n"
            "- ABI <0.4 OR ankle pressure <50 mmHg → consider urgent vascular imaging and "
            "revascularisation.\n"
            "- Toe pressure <30 mmHg OR TcpO2 <25 mmHg → consider urgent assessment for "
            "revascularisation.\n"
            "- PAD less likely if: triphasic or biphasic pedal Doppler waveforms, ABI 0.9–1.3, "
            "TBI ≥0.70.\n"
            "- ABI above 1.3 or below 0.9 is abnormal (indicative of PAD).\n"
            "- TBI below 0.7 is considered abnormal (indicative of PAD).\n"
            "- Clinicians may consider revascularisation at higher pressure levels in patients with "
            "extensive tissue loss or infection (higher WIfI scores).\n\n"
            "REVASCULARISATION INDICATIONS:\n"
            "- When an ulcer fails to show signs of healing within 4–6 weeks despite optimal management, "
            "consider angiography and revascularisation irrespective of vascular diagnostic test results.\n"
            "- If contemplating a major (above the ankle) amputation, first consider the option of "
            "revascularisation.\n\n"
            "REVASCULARISATION GOALS:\n"
            "- Aim to restore in-line flow to at least one of the foot arteries, preferably the artery "
            "that supplies the anatomical region of the wound.\n"
            "- Avoid revascularisation in patients in whom the risk-benefit ratio for probability of "
            "success is unfavourable.\n"
            "- Select revascularisation technique based on individual factors (morphological distribution "
            "of PAD, availability of autogenous vein, patient co-morbidities) and local operator expertise.\n"
            "- After revascularisation, effectiveness should be evaluated with an objective measurement "
            "of perfusion.\n\n"
            "NOT RECOMMENDED:\n"
            "- Pharmacological treatments to improve perfusion have NOT been proven to be beneficial.\n\n"
            "CARDIOVASCULAR RISK REDUCTION in patients with PAD and diabetes:\n"
            "- Cessation of smoking\n"
            "- Control of hypertension and dyslipidaemia\n"
            "- Use of anti-platelet drugs\n"
            "- SGLT2-inhibitor or GLP1-agonist\n\n"
            "Dressing context: Severe ischaemia (ankle pressure <50 mmHg, TBI <0.7, TcpO2 <25 mmHg) "
            "significantly limits wound healing regardless of dressing choice. Non-infected ischaemic "
            "ulcers should NOT be debrided sharply. Moist wound healing environment should still be "
            "maintained. Sucrose octasulfate dressings are an evidence-based adjunct for neuro-ischaemic "
            "ulcers (without severe ischaemia) — see adjunctive treatments section."
        ),
        "metadata": {
            "source": "IWGDF_2023_Practical_Guidelines",
            "section": "Section 4.2.2 – Ischaemia & Revascularisation",
            "topic": "ischaemia, ABI, TBI, TcpO2, ankle pressure, PAD, revascularisation, wound healing barrier, perfusion",
            "chunk_id": "IWGDF_PG_006"
        }
    },
]

print(f"Chunk set F (Ischaemia): {len(chunks_ischaemia)} chunk(s)")


# =============================================================================
# CELL 9 — Chunk Set G: Pressure Offloading for Diabetic Foot Ulcers
# Rationale: Offloading is a prerequisite for healing neuropathic plantar ulcers.
# Without offloading, tissue (T) and edge (E) outcomes in TIME cannot improve
# regardless of dressing. Guides clinical context for dressing recommendations.
# =============================================================================

chunks_offloading = [
    {
        "text": (
            "IWGDF 2023 — Pressure Offloading for Diabetic Foot Ulcers (Section 4.2.3A)\n\n"
            "Offloading is a cornerstone in treatment of foot ulcers caused by increased mechanical "
            "stress. Based on the IWGDF Offloading guidelines.\n\n"
            "PLANTAR NEUROPATHIC ULCERS — preferred hierarchy:\n"
            "1st choice: Non-removable knee-high offloading device — either a total contact cast (TCC) "
            "or a removable walker rendered irremovable (by the provider fitting it).\n"
            "2nd choice: Removable knee-high offloading device, when non-removable device is "
            "contraindicated or not tolerated. Always provide information on the benefits of adherence "
            "to wearing the removable device.\n"
            "3rd choice: Removable ankle-high offloading device — when knee-high is contraindicated or "
            "not tolerated.\n"
            "Fallback: Felted foam, but ONLY in combination with appropriate footwear, when other forms "
            "of biomechanical relief are not available.\n\n"
            "SURGICAL OFFLOADING OPTIONS (when non-surgical offloading fails):\n"
            "- Digital flexor tenotomy for ulcers on digits 2–5 secondary to flexible toe deformity "
            "(if not contraindicated by severe ischaemia or infection).\n"
            "- For metatarsal head ulcer: Achilles tendon lengthening, metatarsal head resection, or "
            "metatarsal osteotomy (all combined with an offloading device).\n"
            "- For hallux ulcer: Joint arthroplasty (combined with offloading device).\n\n"
            "NON-PLANTAR ULCERS:\n"
            "Use a removable offloading device, footwear modifications, toe spacers, orthoses, or "
            "digital flexor tenotomy, depending on the type and location of the foot ulcer.\n\n"
            "CAUTIONS:\n"
            "- When infection or ischaemia are present, offloading is still important, but be more "
            "cautious. The choice of offloading device should be adjusted accordingly.\n\n"
            "Clinical implication: Even the best wound dressing cannot overcome the continued "
            "mechanical stress of an inadequately offloaded plantar ulcer. Offloading status must be "
            "confirmed before attributing wound non-healing to dressing failure."
        ),
        "metadata": {
            "source": "IWGDF_2023_Practical_Guidelines",
            "section": "Section 4.2.3A – Pressure Offloading",
            "topic": "offloading, total contact cast, TCC, neuropathic ulcer, plantar ulcer, wound healing prerequisite",
            "chunk_id": "IWGDF_PG_007"
        }
    },
]

print(f"Chunk set G (Offloading): {len(chunks_offloading)} chunk(s)")


# =============================================================================
# CELL 10 — Chunk Set H: Local Ulcer Care — Wound Bed Preparation & Dressings
# Rationale: CORE to the dressing RAG system. Contains all IWGDF 2023 evidence-
# based local wound care principles, dressing selection guidance, debridement,
# exudate management, and NPWT — directly maps to TIME T, I, M, E parameters.
# =============================================================================

chunks_local_ulcer_care = [
    {
        "text": (
            "IWGDF 2023 — Local Ulcer Care: Wound Bed Preparation & Dressing Selection (Section 4.2.3B)\n\n"
            "Local ulcer care is important to create an environment that increases the likelihood of ulcer "
            "healing. However, even optimum local wound care cannot compensate for inadequately treated "
            "infection or ischaemia, or continuing trauma to the wound bed.\n\n"
            "CORE LOCAL WOUND CARE PRINCIPLES:\n\n"
            "1. Regular inspection:\n"
            "Regular inspection of the ulcer by a trained health care provider is essential. Frequency "
            "depends on the severity of the ulcer and underlying pathology, the presence of infection, "
            "the amount of exudation, and the wound treatment provided.\n\n"
            "2. Debridement:\n"
            "Debride the ulcer and remove surrounding callus (preferably with sharp surgical instruments), "
            "and repeat as needed. Debridement is essential to:\n"
            "- Remove necrotic/sloughy tissue (TIME T — Non-viable tissue)\n"
            "- Allow accurate assessment of wound depth\n"
            "- Stimulate healing response\n"
            "Caution: Do NOT debride a non-infected ulcer that has signs of severe ischaemia.\n"
            "Neuropathic ulcers can usually be debrided without the need for local anaesthesia.\n\n"
            "3. Dressing selection — exudate management:\n"
            "Select dressings to control excess exudation and maintain a moist wound environment.\n"
            "- TIME M (Moisture): Aim for moisture balance — neither too dry nor macerated.\n"
            "- High exudate → absorbent dressings (e.g., foam, alginate, hydrofibre/HCAF)\n"
            "- Low/moderate exudate → moisture-retaining dressings (e.g., hydrogel, film, thin foam)\n\n"
            "4. Washing vs. soaking:\n"
            "Wash but do not soak the feet, as soaking may induce skin maceration.\n\n"
            "5. Negative pressure wound therapy (NPWT):\n"
            "Consider negative pressure wound therapy to help heal post-operative wounds.\n\n"
            "TREATMENTS NOT WELL-SUPPORTED for routine ulcer management (per IWGDF 2023):\n"
            "- Biologically active products (collagen, growth factors, bio-engineered tissue) in "
            "neuropathic ulcers — insufficient evidence for routine use.\n"
            "- Topical antiseptics and antimicrobial dressings or applications — not recommended as "
            "primary treatment for infected wounds; systemic antibiotics are preferred when infection "
            "is present."
        ),
        "metadata": {
            "source": "IWGDF_2023_Practical_Guidelines",
            "section": "Section 4.2.3B – Local Ulcer Care",
            "topic": "wound dressings, debridement, exudate management, moist wound healing, NPWT, TIME framework, local wound care",
            "chunk_id": "IWGDF_PG_008"
        }
    },
    {
        "text": (
            "IWGDF 2023 — Adjunctive Wound Treatments for Non-Healing Ulcers (Section 4.2.3B)\n\n"
            "Consider any of the following adjunctive treatments in NON-INFECTED ulcers that fail to "
            "heal after 4–6 weeks despite optimal clinical care and where resources exist to support "
            "these interventions:\n\n"
            "1. Sucrose octasulfate impregnated dressing (TLC-NOSF / UrgoStart):\n"
            "- Indicated in: Neuro-ischaemic ulcers WITHOUT severe ischaemia.\n"
            "- Evidence: Shown to improve healing rates in neuro-ischaemic diabetic foot ulcers.\n"
            "- Contraindication/caution: Not for use in severely ischaemic wounds.\n\n"
            "2. Multi-layered patch of autologous leucocytes, platelets and fibrin (LPF patch):\n"
            "- Indicated in: Ulcers with or without moderate ischaemia.\n"
            "- Uses patient's own blood components to promote healing.\n\n"
            "3. Placental membrane allografts:\n"
            "- Indicated in: Ulcers with or without moderate ischaemia.\n"
            "- Biological scaffold to support wound healing.\n\n"
            "4. Topical oxygen therapy:\n"
            "- Can be considered for non-healing ulcers failing standard care.\n\n"
            "5. Systemic hyperbaric oxygen therapy (HBOT):\n"
            "- As an adjunctive treatment in ischaemic ulcers.\n"
            "- Indicated when wound fails to heal despite optimal management, particularly with "
            "significant ischaemia.\n\n"
            "NOT RECOMMENDED for routine use:\n"
            "- Biologically active products (collagen, growth factors, bio-engineered tissue) in "
            "neuropathic ulcers — insufficient evidence.\n"
            "- Topical antiseptics and antimicrobial dressings as primary infection treatment.\n\n"
            "Trigger for adjunctive therapy consideration: Ulcer non-healing for ≥4–6 weeks despite:\n"
            "- Adequate offloading\n"
            "- Infection control\n"
            "- Perfusion optimisation\n"
            "- Appropriate local wound care and standard dressings"
        ),
        "metadata": {
            "source": "IWGDF_2023_Practical_Guidelines",
            "section": "Section 4.2.3B – Adjunctive Treatments",
            "topic": "sucrose octasulfate, TLC-NOSF, UrgoStart, placental membrane allograft, hyperbaric oxygen, topical oxygen, LPF patch, neuro-ischaemic ulcer, non-healing wound",
            "chunk_id": "IWGDF_PG_009"
        }
    },
]

print(f"Chunk set H (Local ulcer care + adjunctive): {len(chunks_local_ulcer_care)} chunk(s)")


# =============================================================================
# CELL 11 — Chunk Set I: IWGDF Risk Stratification & Wound Healing Context
# Rationale: Risk category determines healing prognosis context and intensity
# of wound care. Directly relevant to interpreting TIME outputs and calibrating
# expected healing trajectories and monitoring frequency.
# =============================================================================

chunks_risk_stratification = [
    {
        "text": (
            "IWGDF 2023 — Risk Stratification System & Wound Healing Prognosis Context\n\n"
            "The IWGDF 2023 Risk Stratification System categorises persons with diabetes by their risk "
            "of foot ulceration. This stratification is also relevant to wound care planning, healing "
            "prognosis, and monitoring frequency for those who already have an ulcer.\n\n"
            "IWGDF 2023 RISK STRATIFICATION SYSTEM:\n"
            "Category 0 — Very Low Risk:\n"
            "- Characteristics: No Loss of Protective Sensation (LOPS) and no signs of PAD\n"
            "- Screening frequency: Once a year\n\n"
            "Category 1 — Low Risk:\n"
            "- Characteristics: LOPS OR PAD\n"
            "- Screening frequency: Once every 6–12 months\n\n"
            "Category 2 — Moderate Risk:\n"
            "- Characteristics: LOPS + PAD, OR LOPS + foot deformity, OR PAD + foot deformity\n"
            "- Screening frequency: Once every 3–6 months\n\n"
            "Category 3 — High Risk:\n"
            "- Characteristics: LOPS or PAD, AND one or more of:\n"
            "  * History of a foot ulcer\n"
            "  * A lower-extremity amputation (minor or major)\n"
            "  * End-stage renal disease\n"
            "- Screening frequency: Once every 1–3 months\n\n"
            "Note: Screening frequency is based on expert opinion; there is no published evidence to "
            "support these intervals.\n\n"
            "KEY HEALING PROGNOSIS FACTORS from risk categories:\n"
            "- A person with a healed foot ulcer (Category 3) has the highest risk of re-ulceration. "
            "The foot should be considered 'in remission' — requiring lifelong prevention strategies.\n"
            "- End-stage renal disease significantly impairs wound healing and increases amputation risk.\n"
            "- Presence of both LOPS and PAD (Category 2–3) creates the most challenging wound healing "
            "environment — neuro-ischaemic ulcers.\n\n"
            "LOPS = Loss of Protective Sensation; PAD = Peripheral Artery Disease\n\n"
            "Wound care context: Higher IWGDF risk category → more complex wound aetiology → longer "
            "healing trajectory → greater need for adjunctive therapies and specialist involvement."
        ),
        "metadata": {
            "source": "IWGDF_2023_Practical_Guidelines",
            "section": "Section 3 – Risk Stratification (Table 1)",
            "topic": "IWGDF risk stratification, wound healing prognosis, LOPS, PAD, end-stage renal disease, diabetes foot risk",
            "chunk_id": "IWGDF_PG_010"
        }
    },
]

print(f"Chunk set I (Risk stratification): {len(chunks_risk_stratification)} chunk(s)")


# =============================================================================
# CELL 12 — Chunk Set J: Wound Assessment — Comprehensive Summary for RAG
# Rationale: A synthesised chunk mapping IWGDF assessment to TIME parameters
# for direct RAG retrieval. Acts as a bridge/index chunk.
# =============================================================================

chunks_time_mapping = [
    {
        "text": (
            "IWGDF 2023 — Diabetic Foot Ulcer Assessment Mapped to TIME Framework\n\n"
            "The TIME wound assessment framework (Tissue, Infection/Inflammation, Moisture, Edge) can "
            "be mapped to the IWGDF SINBAD system and IWGDF 2023 practical guidelines as follows:\n\n"
            "T — TISSUE (Tissue type: Slough, Granulation, Necrotic):\n"
            "- Maps to SINBAD 'D' (Depth) and 'A' (Area).\n"
            "- Necrotic tissue: Indicates need for debridement. Do NOT debride non-infected ischaemic "
            "ulcers with severe ischaemia.\n"
            "- Slough: Suggests wound is in inflammatory phase; debridement (sharp, enzymatic, or "
            "autolytic) is indicated.\n"
            "- Granulation: Wound in proliferative phase; protect granulation tissue, maintain moist "
            "environment.\n"
            "- IWGDF guidance: Debride neuropathic and neuro-ischaemic ulcers surrounded by callus or "
            "necrotic tissue at initial presentation or as soon as possible. Prefer sharp surgical "
            "debridement.\n\n"
            "I — INFECTION/INFLAMMATION:\n"
            "- Maps to SINBAD 'B' (Bacterial infection) and IWGDF/IDSA grading.\n"
            "- Diagnosis: ≥2 clinical signs of inflammation (redness, warmth, induration, "
            "pain/tenderness) or purulent secretions.\n"
            "- Mild: Superficial, minimal cellulitis → oral antibiotics + debridement.\n"
            "- Moderate: Deeper tissue, extensive cellulitis/abscess → parenteral antibiotics ± surgery.\n"
            "- Severe: Systemic sepsis → urgent surgical + parenteral antibiotics.\n"
            "- Caution: Signs blunted by neuropathy/ischaemia. Culture by tissue biopsy not swab.\n"
            "- Topical antimicrobials/antiseptics NOT recommended as primary infection treatment.\n\n"
            "M — MOISTURE (Exudate Level: Low/Moderate/High):\n"
            "- Maps to wound exudate assessment from SINBAD and clinical inspection.\n"
            "- High exudate: Absorbent dressings (foam, alginate, hydrofibre). Do NOT soak feet.\n"
            "- Low/moderate exudate: Moisture-retaining dressings (hydrogel, film, thin foam).\n"
            "- Oedema increases exudate — treat systemic causes (oedema management).\n"
            "- NPWT: Consider for post-operative wounds.\n\n"
            "E — EDGE (Wound Edge: Advancing/Non-Advancing):\n"
            "- Non-advancing edge after 4–6 weeks despite optimal care → trigger for adjunctive therapy.\n"
            "- Adjunctive options per IWGDF 2023 (non-infected, non-severely-ischaemic ulcers):\n"
            "  * Sucrose octasulfate dressing (neuro-ischaemic ulcers)\n"
            "  * LPF patch (autologous leucocytes/platelets/fibrin)\n"
            "  * Placental membrane allografts\n"
            "  * Topical oxygen therapy\n"
            "  * Systemic hyperbaric oxygen (ischaemic ulcers)\n\n"
            "ADDITIONAL IWGDF HEALING PREREQUISITES (address before optimising dressing):\n"
            "1. Infection control (antibiotics / surgery if needed)\n"
            "2. Perfusion / revascularisation (ABI, TBI, TcpO2 thresholds)\n"
            "3. Pressure offloading (TCC or irremovable walker for plantar neuropathic ulcers)\n"
            "4. Systemic optimisation (glycaemia, nutrition, oedema)"
        ),
        "metadata": {
            "source": "IWGDF_2023_Practical_Guidelines",
            "section": "Sections 4.1.1, 4.2.1–4.2.4 – Synthesised TIME Mapping",
            "topic": "TIME framework, SINBAD, wound assessment, tissue, infection, moisture, edge, diabetic foot ulcer, dressing selection",
            "chunk_id": "IWGDF_PG_011"
        }
    },
]

print(f"Chunk set J (TIME mapping): {len(chunks_time_mapping)} chunk(s)")


# =============================================================================
# CELL 13 — Combine all chunk sets & preview
# =============================================================================

all_chunks = (
    chunks_pathophysiology
    + chunks_sinbad
    + chunks_infection_assessment
    + chunks_person_factors
    + chunks_infection_treatment
    + chunks_ischaemia
    + chunks_offloading
    + chunks_local_ulcer_care
    + chunks_risk_stratification
    + chunks_time_mapping
)

print(f"Total chunks to upload: {len(all_chunks)}")
print("\nChunk IDs and section summaries:")
for c in all_chunks:
    print(f"  [{c['metadata']['chunk_id']}] {c['metadata']['section']}")


# =============================================================================
# CELL 14 — Preview individual chunks (optional, for verification)
# =============================================================================

def preview_chunk(chunk_id: str):
    for c in all_chunks:
        if c["metadata"]["chunk_id"] == chunk_id:
            print(f"=== {chunk_id} ===")
            print(f"Section : {c['metadata']['section']}")
            print(f"Topics  : {c['metadata']['topic']}")
            print(f"Length  : {len(c['text'])} chars")
            print("\n--- TEXT PREVIEW (first 500 chars) ---")
            print(c["text"][:500])
            print("...\n")
            return
    print(f"Chunk {chunk_id} not found.")

# Uncomment to preview specific chunks:
# preview_chunk("IWGDF_PG_008")
# preview_chunk("IWGDF_PG_009")
# preview_chunk("IWGDF_PG_011")


# =============================================================================
# CELL 15 — Upload all chunks to vector store
# =============================================================================

print(f"Starting upload of {len(all_chunks)} chunks to vector store: {VECTOR_STORE_ID}")
print("=" * 60)

uploaded_file_ids = upload_chunks_to_vector_store(
    chunks=all_chunks,
    vector_store_id=VECTOR_STORE_ID,
    batch_size=10
)

print("\nUpload complete.")
print(f"Total file IDs uploaded: {len(uploaded_file_ids)}")


# =============================================================================
# CELL 16 — Verify upload (optional spot-check)
# =============================================================================

def verify_vector_store_files(vector_store_id: str, expected_count: int):
    """List files in the vector store and confirm count."""
    files = client.beta.vector_stores.files.list(vector_store_id=vector_store_id)
    file_list = list(files)
    print(f"Files currently in vector store: {len(file_list)}")
    print(f"Expected new additions: {expected_count}")
    return file_list

# Uncomment to verify:
# verify_vector_store_files(VECTOR_STORE_ID, len(all_chunks))