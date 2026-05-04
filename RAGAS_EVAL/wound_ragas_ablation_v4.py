# ══════════════════════════════════════════════════════════════════════════════
# wound_ragas_ablation_v4.py
# VerdaSense — RAGAS Ablation Evaluation Script (v4)
# ══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE
# ───────
# Evaluate wound_app_00_v4.py and wound_app_01_v4.py against the same
# wound_testset_v2.json used for v2/v3 evaluations, enabling direct
# metric comparison across all 6 previous versions.
#
# USAGE
# ─────
# 1. Start one wound_app_XX_v4.py server (default port 8000):
#      uvicorn wound_app_00_v4:app --port 8000
# 2. Run all cells in order (or python wound_ragas_ablation_v4.py).
# 3. Check ragas_eval_00_v4/ and ragas_eval_01_v4/ for results.
# 4. Cell 11 prints the side-by-side comparison table.
#
# V4 CHANGES vs v3 NOTEBOOK
# ─────────────────────────
# Cell 4  (call_rag)      : reads classifier_output and verifier_output from v4 response
# Cell 5  (check_safety)  : same v3 logic (no change needed — rules are version-agnostic)
# Cell 7  (run_evaluation): stores classifier_output + verifier_output in records;
#                           adds verifier correction rate to safety report
# Cell 11 (comparison)    : updated for v4_00 + v4_01 columns
#
# DIRECTORY LAYOUT (expected)
# ───────────────────────────
#  project_root/
#  ├── ragas_testset/
#  │   └── wound_testset_v2.json
#  ├── ragas_eval_00_v4/
#  │   └── wound_ragas_ablation_results_00_v4.json   ← written by this script
#  ├── ragas_eval_01_v4/
#  │   └── wound_ragas_ablation_results_01_v4.json   ← written by this script
#  └── wound_ragas_ablation_v4.py                    ← this file
#
# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
# CELL 1: Imports
# ──────────────────────────────────────────────────────────────────────────────
# Standard evaluation stack identical to v2/v3.
# Added: ast (not used directly but kept for parity with v3).
# ──────────────────────────────────────────────────────────────────────────────

import os
import json
import time
import ast
import pandas as pd
import httpx
from dotenv import load_dotenv

load_dotenv()

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    Faithfulness,
    AnswerRelevancy,
)
from ragas.llms       import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not installed — charts will be skipped")


# ──────────────────────────────────────────────────────────────────────────────
# CELL 2: RAGAS judge LLM + embeddings setup
# ──────────────────────────────────────────────────────────────────────────────
# Uses gpt-4o-mini for all four RAGAS metrics (context precision, recall,
# faithfulness, answer relevancy).  text-embedding-3-small is used only for
# AnswerRelevancy (cosine similarity between question and answer embeddings).
# ──────────────────────────────────────────────────────────────────────────────

judge_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
judge_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))


# ──────────────────────────────────────────────────────────────────────────────
# CELL 3: Load testset
# ──────────────────────────────────────────────────────────────────────────────
# Uses the same wound_testset_v2.json as v2/v3 evaluations.
# This ensures all 6 versions (v2_00, v2_01, v2_02, v3_00, v3_01, v3_02) and
# both v4 versions are evaluated on an identical test distribution, making
# metric comparisons valid.
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR   = "../ragas_testset/"
TESTSET_JSON = os.path.join(OUTPUT_DIR, "wound_testset_v2.json")

if not os.path.isfile(TESTSET_JSON):
    raise FileNotFoundError(
        f"Testset not found at {TESTSET_JSON}. "
        "Ensure wound_testset_v2.json is in the ragas_testset/ directory."
    )

with open(TESTSET_JSON, "r", encoding="utf-8") as f:
    testset = json.load(f)

print(f"✅ Loaded {len(testset)} test cases from {os.path.basename(TESTSET_JSON)}")

# ── Validate testset format ────────────────────────────────────────────────────
v2_fields = ["time_payload", "reference_contexts", "wound_type_expected",
             "allowed_dressings", "contraindicated_dressings",
             "antibiotic_required", "referral_required"]
has_v2 = all(f in testset[0] for f in v2_fields) if testset else False

if has_v2:
    print("   ✅ v2 testset format confirmed — full evaluation enabled")
    cats = {}
    for tc in testset:
        c = tc.get("category", "?")
        cats[c] = cats.get(c, 0) + 1
    for c, n in sorted(cats.items()):
        print(f"      Category {c}: {n} cases")
else:
    print("   ⚠️  v2 testset fields missing — rule-based safety checks will be skipped")


# ──────────────────────────────────────────────────────────────────────────────
# CELL 4: RAG caller — v4 version
# ──────────────────────────────────────────────────────────────────────────────
# V4 CHANGES vs v3:
#   - Reads classifier_output (wound_type, referral_required, antibiotic_required)
#     from API response — used in the v4 safety report for verifying that the
#     classifier is producing the correct wound type.
#   - Reads verifier_output (failed_checks, correction_applied) — used to report
#     how often the post-gen verifier had to correct the answer.
#   - narrative_query reading identical to v3.
# ──────────────────────────────────────────────────────────────────────────────

RAG_URL = "http://localhost:8000/get_recommendation"

def call_rag(record: dict, timeout: int = 180) -> dict:
    """
    Send T.I.M.E. inputs to the running FastAPI server and return the full
    API response dict.

    v4 new fields in response:
      - classifier_output: {wound_type, referral_required, antibiotic_required, ...}
      - verifier_output:   {failed_checks: [...], correction_applied: bool}

    Timeout increased to 180s for v4 because the verifier adds a second LLM
    call (and potentially a correction call) on top of the generation call.
    Falls back gracefully if the server returns an error or times out.
    """
    tp = record["time_payload"]
    payload = {
        "necrotic_pct":      tp["necrotic_pct"],
        "slough_pct":        tp["slough_pct"],
        "granulation_pct":   tp["granulation_pct"],
        "infection":         tp["infection"],
        "moisture":          tp["moisture"],
        "edge":              tp["edge"],
        "notes":             tp.get("notes", ""),
        "tissue_confidence": 0.0,
    }

    try:
        r = httpx.post(RAG_URL, data=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {
            "result":            f"ERROR: {e}",
            "chunk_texts":       [],
            "confidence_label":  "LOW",
            "narrative_query":   "",
            "classifier_output": {},
            "verifier_output":   {"failed_checks": [], "correction_applied": False},
        }


# ──────────────────────────────────────────────────────────────────────────────
# CELL 5: Rule-based safety checker — v3 logic (unchanged for v4)
# ──────────────────────────────────────────────────────────────────────────────
# The v3 safety checker is version-agnostic: it reads from testset ground truth
# (allowed_dressings, contraindicated_dressings, referral_required,
# antibiotic_required) and checks the generated answer text.
# No changes needed for v4 — the checker is evaluating the output, not the
# pipeline internals.
#
# Checks performed:
#   1. Contraindicated dressings NOT positively recommended in Primary/Secondary
#      Dressing sections (uses positive-section extraction to avoid false positives
#      from "avoid silver" mentions in the Contraindications section).
#   2. Antibiotic recommendation correctly present when antibiotic_required=True.
#   3. Referral recommendation correctly present when referral_required=True.
#   4. At least one allowed dressing positively recommended.
# ──────────────────────────────────────────────────────────────────────────────

import re

_NEGATIVE_PATTERNS = re.compile(
    r"\b(avoid|contraindic|not (to use|recommended|indicated|suitable)|"
    r"should not|must not|do not use|never use|excluded|inappropriate)\b",
    re.IGNORECASE,
)

_POSITIVE_SECTION_HEADERS = re.compile(
    r"^##\s+(primary dressing|secondary dressing)",
    re.IGNORECASE | re.MULTILINE,
)

def _extract_positive_recommendation_text(answer: str) -> str:
    """
    Returns only text from ## Primary Dressing and ## Secondary Dressing
    sections, with avoidance-language sentences stripped.
    """
    lines = answer.split("\n")
    in_positive_section = False
    collected = []
    for line in lines:
        if line.strip().startswith("##"):
            in_positive_section = bool(_POSITIVE_SECTION_HEADERS.match(line.strip()))
            continue
        if in_positive_section:
            if _NEGATIVE_PATTERNS.search(line):
                continue
            collected.append(line.lower())
    return " ".join(collected)


def _is_dressing_recommended(dressing_term: str, answer: str) -> bool:
    positive_text = _extract_positive_recommendation_text(answer)
    return dressing_term.lower().replace("_", " ") in positive_text


DRESSING_ALIASES: dict = {
    "silver":                  ["silver"],
    "charcoal":                ["charcoal"],
    "alginate":                ["alginate"],
    "iodine":                  ["iodine", "povidone", "cadexomer"],
    "npwt":                    ["npwt", "negative pressure", "vacuum"],
    "honey":                   ["honey", "manuka", "medihoney"],
    "foam":                    ["foam"],
    "bordered_foam":           ["bordered foam", "island foam", "adhesive foam"],
    "adhesive_bordered_foam":  ["bordered foam", "adhesive foam", "island foam"],
    "hydrocolloid":            ["hydrocolloid", "duoderm", "comfeel"],
    "silicone_foam":           ["silicone foam", "mepitel", "mepilex"],
    "film":                    ["film", "tegaderm", "opsite"],
    "hydrofiber":              ["hydrofiber", "aquacel"],
    "hydrogel":                ["hydrogel", "intrasite"],
    "tulle":                   ["tulle", "paraffin gauze"],
    "polymeric_membrane":      ["polymeric membrane", "polymem"],
}

def _dressing_surface_forms(token: str) -> list:
    return DRESSING_ALIASES.get(token, [token.replace("_", " ")])


def check_safety(generated_answer: str, test_case: dict) -> dict:
    """
    Rule-based safety evaluation (v3 logic, v4-compatible).

    Returns a dict of check_name → "PASS" | "FAIL", plus "overall".
    """
    if not has_v2:
        return {}

    ans_lower = generated_answer.lower()
    results   = {}

    # ── Check 1: Contraindicated dressings not positively recommended ──────────
    for contra_token in test_case.get("contraindicated_dressings", []):
        surface_forms  = _dressing_surface_forms(contra_token)
        is_recommended = any(
            _is_dressing_recommended(form, generated_answer)
            for form in surface_forms
        )
        results[f"contraindication_absent_{contra_token}"] = (
            "FAIL" if is_recommended else "PASS"
        )

    # ── Check 2: Antibiotic correctly addressed ────────────────────────────────
    if test_case.get("antibiotic_required", False):
        explicit_yes   = "antibiotic therapy is recommended" in ans_lower
        broad_keywords = [
            "antibiotic", "c&s", "culture and sensitivity", "wound swab",
            "systemic antimicrobial", "topical antimicrobial", "oral antibiotic",
        ]
        broad_match    = any(kw in ans_lower for kw in broad_keywords)
        explicit_no    = "antibiotic therapy is not indicated" in ans_lower
        antibiotic_ok  = (explicit_yes or broad_match) and not explicit_no
        results["antibiotic_recommended"] = "PASS" if antibiotic_ok else "FAIL"

    # ── Check 3: Referral correctly addressed ─────────────────────────────────
    if test_case.get("referral_required", False):
        explicit_yes   = "referral is recommended" in ans_lower
        broad_keywords = [
            "refer", "hospital", "specialist", "escalat",
            "wound care team", "wound type 6", "wound type 7", "wound type 8",
            "secondary care",
        ]
        broad_match    = any(kw in ans_lower for kw in broad_keywords)
        explicit_no    = "referral is not required" in ans_lower
        referral_ok    = (explicit_yes or broad_match) and not explicit_no
        results["referral_recommended"] = "PASS" if referral_ok else "FAIL"

    # ── Check 4: At least one allowed dressing positively recommended ──────────
    allowed = test_case.get("allowed_dressings", [])
    if allowed:
        any_allowed = any(
            _is_dressing_recommended(form, generated_answer)
            for token in allowed
            for form in _dressing_surface_forms(token)
        )
        results["dressing_in_allowed_list"] = "PASS" if any_allowed else "FAIL"

    overall = "PASS" if all(v == "PASS" for v in results.values()) else "FAIL"
    results["overall"] = overall
    return results


# ──────────────────────────────────────────────────────────────────────────────
# CELL 6: Metric column definitions
# ──────────────────────────────────────────────────────────────────────────────
# Identical to v3 — same 4 RAGAS metrics, same display labels and chart colours.
# ──────────────────────────────────────────────────────────────────────────────

METRIC_COLS = [
    "llm_context_precision_with_reference",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
]
METRIC_LABELS = [
    "Context precision",
    "Context recall",
    "Faithfulness",
    "Answer relevancy",
]
METRIC_COLORS = ["#185FA5", "#1D9E75", "#BA7517", "#D85A30"]


# ──────────────────────────────────────────────────────────────────────────────
# CELL 7: Main evaluation function — v4 version
# ──────────────────────────────────────────────────────────────────────────────
# V4 CHANGES vs v3:
#   1. Stores classifier_output per record (wound_type predicted, flags).
#   2. Stores verifier_output per record (failed_checks, correction_applied).
#   3. Safety report includes:
#        - classifier_wound_type  (predicted by v4 classifier)
#        - wound_type_expected    (ground truth from testset)
#        - classifier_match       (bool — did classifier predict correct type?)
#        - verifier_correction    (True if verifier correction was applied)
#        - verifier_failed_checks (which checks failed before correction)
#   4. RAGAS user_input uses narrative_query (same as v3 — no change).
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation(experiment_name: str, results_json: str):
    """
    Run RAGAS + rule-based safety evaluation + v4 classifier/verifier audit
    for one experiment.

    Parameters
    ----------
    experiment_name : str
        Human-readable label for this run (e.g. "wound_ragas_eval_00_v4").
    results_json : str
        Path where per-case results will be saved/resumed.

    Returns
    -------
    agg_df : pd.DataFrame
        Aggregated RAGAS scores (metric × score).
    full_df : pd.DataFrame
        Per-sample RAGAS scores from ragas.evaluate().
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*60}")

    # ── Create output directory if needed ─────────────────────────────────────
    out_dir = os.path.dirname(results_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    safety_report_path = results_json.replace(".json", "_safety.csv")

    # ── Resume if partial results exist ───────────────────────────────────────
    if os.path.isfile(results_json):
        with open(results_json, encoding="utf-8") as f:
            records = json.load(f)
        done = {r["index"] for r in records}
        print(f"Resuming: {len(records)}/{len(testset)} done")
    else:
        records = []
        done    = set()

    # ── Iterate over test cases ────────────────────────────────────────────────
    for idx, case in enumerate(testset):
        if idx in done:
            print(f"  [{idx+1:>2}/{len(testset)}] skip (done)")
            continue

        name = case.get("synthesizer_name", case.get("case_id", f"case_{idx}"))
        print(f"  [{idx+1:>2}/{len(testset)}] {name}")
        t0      = time.time()
        resp    = call_rag(case)
        elapsed = time.time() - t0

        answer = resp.get("result", "")
        chunks = resp.get("chunk_texts", [])
        retrieved_contexts = chunks if chunks else [answer]

        # ── v3 field: narrative_query as RAGAS user_input ─────────────────────
        narrative_query = resp.get("narrative_query", "") or case.get("user_input", "")

        # ── [v4 NEW] classifier and verifier outputs ──────────────────────────
        classifier_output = resp.get("classifier_output", {})
        verifier_output   = resp.get("verifier_output", {})

        # Classifier accuracy check
        predicted_type  = classifier_output.get("wound_type", None)
        expected_type   = case.get("wound_type_expected", None)
        classifier_match = (predicted_type == expected_type) if (predicted_type and expected_type) else None

        # Rule-based safety check
        safety = check_safety(answer, case)
        if safety:
            status = safety.get("overall", "N/A")
            match_str = f"classifier={'✓' if classifier_match else '✗' if classifier_match is False else '?'}"
            verifier_str = f"verifier_correction={verifier_output.get('correction_applied', False)}"
            print(f"       Safety: {status} | {match_str} | {verifier_str}")

        records.append({
            "index":               idx,
            "case_id":             case.get("case_id", f"case_{idx}"),
            "category":            case.get("category", "?"),
            "synthesizer_name":    name,
            "wound_type_expected": expected_type,
            # v3 fields
            "user_input":          case.get("user_input", ""),
            "narrative_query":     narrative_query,
            "reference":           case.get("reference", ""),
            "reference_contexts":  case.get("reference_contexts", []),
            "retrieved_contexts":  retrieved_contexts,
            "answer":              answer,
            "confidence_label":    resp.get("confidence_label", "?"),
            "elapsed_sec":         round(elapsed, 1),
            "safety_checks":       safety,
            # [v4 NEW] classifier + verifier fields
            "classifier_wound_type":     predicted_type,
            "classifier_referral":       classifier_output.get("referral_required", None),
            "classifier_antibiotic":     classifier_output.get("antibiotic_required", None),
            "classifier_match":          classifier_match,
            "classifier_notes":          classifier_output.get("classifier_notes", ""),
            "verifier_failed_checks":    verifier_output.get("failed_checks", []),
            "verifier_correction":       verifier_output.get("correction_applied", False),
        })

        with open(results_json, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        # Pause between calls to avoid rate limiting
        time.sleep(1.0)

    # ── Safety + classifier + verifier report ─────────────────────────────────
    if has_v2:
        safety_rows = []
        for r in records:
            row = {
                "case_id":              r.get("case_id", ""),
                "category":             r.get("category", "?"),
                "wound_type_expected":  r.get("wound_type_expected", "?"),
                "classifier_wound_type":r.get("classifier_wound_type", "?"),
                "classifier_match":     r.get("classifier_match", "?"),
                "verifier_correction":  r.get("verifier_correction", False),
                "verifier_failed":      ", ".join(r.get("verifier_failed_checks", [])),
                "overall":              r.get("safety_checks", {}).get("overall", "N/A"),
            }
            for k, v in r.get("safety_checks", {}).items():
                if k != "overall":
                    row[k] = v
            safety_rows.append(row)

        safety_df = pd.DataFrame(safety_rows)
        safety_df.to_csv(safety_report_path, index=False)

        total   = len(safety_df)
        passed  = (safety_df["overall"] == "PASS").sum() if "overall" in safety_df.columns else 0
        correct = safety_df["classifier_match"].sum() if "classifier_match" in safety_df.columns else 0
        corrections = safety_df["verifier_correction"].sum() if "verifier_correction" in safety_df.columns else 0

        print(f"\n Safety Report: {passed}/{total} cases PASS ({100*passed//total if total else 0}%)")
        print(f"   Classifier accuracy: {correct}/{total} wound types correct")
        print(f"   Verifier corrections applied: {corrections}/{total}")
        print(f"   Saved → {safety_report_path}")

        if "overall" in safety_df.columns:
            failed = safety_df[safety_df["overall"] == "FAIL"]
            if not failed.empty:
                print(f"\n   FAILED cases ({len(failed)}):")
                for _, row in failed.iterrows():
                    checks = [
                        k for k, v in row.items()
                        if k not in ("case_id", "category", "wound_type_expected",
                                     "classifier_wound_type", "classifier_match",
                                     "verifier_correction", "verifier_failed", "overall")
                        and v == "FAIL"
                    ]
                    print(f"      {row['case_id']} — {', '.join(checks)}")

    # ── Build RAGAS dataset ────────────────────────────────────────────────────
    samples = []
    for r in records:
        if not r["answer"] or r["answer"].startswith("ERROR"):
            continue

        ref_contexts = r.get("reference_contexts", [])
        if not ref_contexts:
            print(f"   ⚠ No reference_contexts for {r.get('case_id', '?')} — recall will be 0")

        # Use narrative_query as RAGAS user_input (v3 fix, carried into v4)
        ragas_user_input = r.get("narrative_query") or r.get("user_input", "")

        samples.append(SingleTurnSample(
            user_input         = ragas_user_input,
            reference          = r["reference"],
            reference_contexts = [str(c) for c in ref_contexts],
            retrieved_contexts = [str(c) for c in r["retrieved_contexts"]],
            response           = r["answer"],
        ))

    print(f"\nRunning RAGAS scoring on {len(samples)} samples...")
    dataset = EvaluationDataset(samples)
    results = evaluate(dataset, metrics=[
        LLMContextPrecisionWithReference(llm=judge_llm),
        LLMContextRecall(llm=judge_llm),
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_emb),
    ])

    scores_df = results.to_pandas()
    for col in METRIC_COLS:
        if col not in scores_df.columns:
            scores_df[col] = float("nan")

    agg = {col: scores_df[col].mean() for col in METRIC_COLS}

    print(f"\n{'─'*40}")
    print(f"  RAGAS SCORES — {experiment_name}")
    print(f"{'─'*40}")
    for label, col in zip(METRIC_LABELS, METRIC_COLS):
        print(f"  {label:<25} {agg[col]:.4f}  ({agg[col]*100:.1f}%)")
    print(f"{'─'*40}")

    # ── Bar chart ──────────────────────────────────────────────────────────────
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(8, 4))
        vals = [agg[c] for c in METRIC_COLS]
        bars = ax.bar(METRIC_LABELS, vals, color=METRIC_COLORS, alpha=0.85, zorder=3)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Score")
        ax.set_title(f"RAGAS Metrics — {experiment_name}")
        ax.grid(axis="y", alpha=0.3, zorder=0)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        chart_path = results_json.replace(".json", "_chart.png")
        fig.tight_layout()
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        print(f"   Chart saved → {chart_path}")

    agg_df = pd.DataFrame([
        {"metric": lbl, "score": agg[col]}
        for lbl, col in zip(METRIC_LABELS, METRIC_COLS)
    ])
    return agg_df, scores_df


# ──────────────────────────────────────────────────────────────────────────────
# CELL 8: Run evaluation — v4_00 (baseline, dense-only)
# ──────────────────────────────────────────────────────────────────────────────
# Before running:
#   1. Start wound_app_00_v4.py on port 8000:
#        uvicorn wound_app_00_v4:app --port 8000
#   2. Confirm server is up: curl http://localhost:8000/
# ──────────────────────────────────────────────────────────────────────────────

agg_df_00, full_df_00 = run_evaluation(
    experiment_name = "wound_ragas_eval_00_v4",
    results_json    = "../ragas_eval_00_v4/wound_ragas_ablation_results_00_v4.json",
)
agg_df_00


# ──────────────────────────────────────────────────────────────────────────────
# CELL 9: Run evaluation — v4_01 (BM25 hybrid)
# ──────────────────────────────────────────────────────────────────────────────
# Before running:
#   1. Stop wound_app_00_v4.py (Ctrl+C).
#   2. Start wound_app_01_v4.py on port 8000:
#        uvicorn wound_app_01_v4:app --port 8000
#   3. Confirm server is up: curl http://localhost:8000/
# ──────────────────────────────────────────────────────────────────────────────

agg_df_01, full_df_01 = run_evaluation(
    experiment_name = "wound_ragas_eval_01_v4",
    results_json    = "../ragas_eval_01_v4/wound_ragas_ablation_results_01_v4.json",
)
agg_df_01


# ──────────────────────────────────────────────────────────────────────────────
# CELL 10: v4 head-to-head comparison
# ──────────────────────────────────────────────────────────────────────────────
# Side-by-side RAGAS scores for v4_00 vs v4_01.
# ──────────────────────────────────────────────────────────────────────────────

comparison_v4 = pd.DataFrame({
    "Metric":              METRIC_LABELS,
    "Eval_v4_00 (baseline)": [agg_df_00[agg_df_00.metric == lbl].score.values[0] for lbl in METRIC_LABELS],
    "Eval_v4_01 (+BM25)":    [agg_df_01[agg_df_01.metric == lbl].score.values[0] for lbl in METRIC_LABELS],
}).set_index("Metric")

print("\nV4 Head-to-Head Comparison:")
print(comparison_v4.to_string())


# ──────────────────────────────────────────────────────────────────────────────
# CELL 11: Full cross-version comparison table (v2 + v3 + v4)
# ──────────────────────────────────────────────────────────────────────────────
# Loads pre-saved v2 and v3 results to build the complete 6+2 version table.
# Adjust paths to match your local directory layout.
#
# NOTE: If you don't have all v2/v3 results available, comment out the
# v2/v3 load blocks and build a v4-only table.
# ──────────────────────────────────────────────────────────────────────────────

def _load_saved_scores(results_json_path: str) -> dict | None:
    """
    Load a previously saved ablation results JSON and compute mean RAGAS scores.
    Returns {metric_col: mean_score} or None if file not found.

    NOTE: This requires that the JSON file contains rows with RAGAS score columns
    (llm_context_precision_with_reference, context_recall, faithfulness,
    answer_relevancy). If the file was saved before RAGAS scoring, scores will
    not be present — use the original agg_df objects instead.
    """
    if not os.path.isfile(results_json_path):
        print(f"   ⚠ {results_json_path} not found — skipping")
        return None
    with open(results_json_path, encoding="utf-8") as f:
        records = json.load(f)
    scores = {col: [] for col in METRIC_COLS}
    for r in records:
        for col in METRIC_COLS:
            if col in r:
                scores[col].append(r[col])
    if not any(scores[col] for col in METRIC_COLS):
        return None
    return {col: (sum(v)/len(v) if v else float("nan")) for col, v in scores.items()}


def _scores_to_row(scores_dict: dict | None) -> list:
    """Convert a scores dict to a list aligned with METRIC_COLS."""
    if scores_dict is None:
        return [float("nan")] * len(METRIC_COLS)
    return [scores_dict.get(col, float("nan")) for col in METRIC_COLS]


# ── v2 scores — load from agg DataFrames if available, else from JSON ─────────
# Adjust these paths to match your layout:
_V2_RESULTS = {
    "v2_00": "../ragas_eval_00_v2/wound_ragas_ablation_results_00_v2.json",
    "v2_01": "../ragas_eval_01_v2/wound_ragas_ablation_results_01_v2.json",
    "v2_02": "../ragas_eval_02_v2/wound_ragas_ablation_results_02_v2.json",
}
_V3_RESULTS = {
    "v3_00": "../ragas_eval_00_v3/wound_ragas_ablation_results_00_v3.json",
    "v3_01": "../ragas_eval_01_v3/wound_ragas_ablation_results_01_v3.json",
    "v3_02": "../ragas_eval_02_v3/wound_ragas_ablation_results_02_v3.json",
}

# Hard-coded v2 and v3 scores from your existing evaluation runs.
# Replace these with dynamic loading if you re-run v2/v3 evaluations.
_V2_HARDCODED = {
    "v2_00": {"Context precision": 0.8663, "Context recall": 0.6582, "Faithfulness": 0.6126, "Answer relevancy": 0.5732},
    "v2_01": {"Context precision": 0.8632, "Context recall": 0.7196, "Faithfulness": 0.7217, "Answer relevancy": 0.5776},
    "v2_02": {"Context precision": 0.8044, "Context recall": 0.5159, "Faithfulness": 0.6760, "Answer relevancy": 0.5837},
}
_V3_HARDCODED = {
    "v3_00": {"Context precision": 0.9191, "Context recall": 0.6466, "Faithfulness": 0.6880, "Answer relevancy": 0.7387},
    "v3_01": {"Context precision": 0.9310, "Context recall": 0.7277, "Faithfulness": 0.6755, "Answer relevancy": 0.7318},
    "v3_02": {"Context precision": 0.8231, "Context recall": 0.5484, "Faithfulness": 0.7123, "Answer relevancy": 0.7268},
}

# ── Build full comparison DataFrame ───────────────────────────────────────────
comparison_full = pd.DataFrame({
    "Metric": METRIC_LABELS,
    # V2
    "v2_00 (baseline)":    [_V2_HARDCODED["v2_00"][lbl] for lbl in METRIC_LABELS],
    "v2_01 (+BM25)":       [_V2_HARDCODED["v2_01"][lbl] for lbl in METRIC_LABELS],
    "v2_02 (+reranker)":   [_V2_HARDCODED["v2_02"][lbl] for lbl in METRIC_LABELS],
    # V3
    "v3_00 (baseline)":    [_V3_HARDCODED["v3_00"][lbl] for lbl in METRIC_LABELS],
    "v3_01 (+BM25)":       [_V3_HARDCODED["v3_01"][lbl] for lbl in METRIC_LABELS],
    "v3_02 (+reranker)":   [_V3_HARDCODED["v3_02"][lbl] for lbl in METRIC_LABELS],
    # V4 (from this run)
    "v4_00 (baseline)":    [agg_df_00[agg_df_00.metric == lbl].score.values[0] for lbl in METRIC_LABELS],
    "v4_01 (+BM25)":       [agg_df_01[agg_df_01.metric == lbl].score.values[0] for lbl in METRIC_LABELS],
}).set_index("Metric")

print("\n" + "="*80)
print("FULL CROSS-VERSION RAGAS COMPARISON  (v2 → v3 → v4)")
print("="*80)
print(comparison_full.to_string(float_format=lambda x: f"{x:.4f}"))

# ── Delta table: v4_01 vs best v3 (v3_01) ─────────────────────────────────────
print("\nDelta: v4_01 vs v3_01 (best previous baseline):")
delta = comparison_full["v4_01 (+BM25)"] - comparison_full["v3_01 (+BM25)"]
for metric, diff in delta.items():
    arrow = "▲" if diff > 0.005 else ("▼" if diff < -0.005 else "—")
    print(f"  {metric:<25} {arrow} {diff:+.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# CELL 12: Safety pass rate summary — v4 vs v3
# ──────────────────────────────────────────────────────────────────────────────
# Loads saved safety JSONs and computes overall pass rate + per-check breakdown.
# Enables direct comparison of clinical safety improvement from v3 → v4.
# ──────────────────────────────────────────────────────────────────────────────

def _safety_pass_rate(results_json_path: str) -> dict:
    """
    Compute safety pass rates from a results JSON.
    Returns {check_name: pass_count} and total.
    """
    if not os.path.isfile(results_json_path):
        return {}
    with open(results_json_path, encoding="utf-8") as f:
        records = json.load(f)

    total = len(records)
    check_counts = {}
    for r in records:
        sc = r.get("safety_checks", {})
        for k, v in sc.items():
            if k not in check_counts:
                check_counts[k] = {"PASS": 0, "FAIL": 0}
            check_counts[k][v] = check_counts[k].get(v, 0) + 1

    result = {"total": total}
    for k, counts in check_counts.items():
        result[k] = f"{counts.get('PASS', 0)}/{total}"
    return result


# Print safety rate table for all evaluated versions
print("\n" + "="*60)
print("CLINICAL SAFETY PASS RATES")
print("="*60)

_safety_files = {
    "v3_00": "../ragas_eval_00_v3/wound_ragas_ablation_results_00_v3.json",
    "v3_01": "../ragas_eval_01_v3/wound_ragas_ablation_results_01_v3.json",
    "v4_00": "../ragas_eval_00_v4/wound_ragas_ablation_results_00_v4.json",
    "v4_01": "../ragas_eval_01_v4/wound_ragas_ablation_results_01_v4.json",
}

for version, path in _safety_files.items():
    rates = _safety_pass_rate(path)
    if rates:
        overall = rates.get("overall", "?")
        print(f"\n  {version}: overall={overall}")
        for k, v in rates.items():
            if k not in ("total", "overall"):
                print(f"    {k}: {v}")


# ──────────────────────────────────────────────────────────────────────────────
# CELL 13: v4 classifier accuracy audit
# ──────────────────────────────────────────────────────────────────────────────
# Summarises how accurately the v4 pre-classifier predicted the correct wound
# type for each test case.  A wrong wound type prediction means the wrong
# algorithm chunk was targeted for Sub-query A, which is a retrieval failure.
# ──────────────────────────────────────────────────────────────────────────────

def _classifier_accuracy_report(results_json_path: str, version_label: str):
    """Print classifier accuracy breakdown from a v4 results JSON."""
    if not os.path.isfile(results_json_path):
        print(f"  {version_label}: file not found")
        return

    with open(results_json_path, encoding="utf-8") as f:
        records = json.load(f)

    total   = len(records)
    correct = sum(1 for r in records if r.get("classifier_match") is True)
    wrong   = [(r["case_id"], r.get("wound_type_expected"), r.get("classifier_wound_type"))
               for r in records if r.get("classifier_match") is False]
    unknown = sum(1 for r in records if r.get("classifier_match") is None)

    print(f"\n  {version_label} classifier accuracy: {correct}/{total} correct ({100*correct//total if total else 0}%)")
    if wrong:
        print(f"    Misclassified cases ({len(wrong)}):")
        for case_id, expected, predicted in wrong:
            print(f"      {case_id}: expected type={expected}, predicted type={predicted}")
    if unknown:
        print(f"    Unknown (API error or pre-v4 response): {unknown}")

    # Verifier correction rate
    corrections = sum(1 for r in records if r.get("verifier_correction") is True)
    print(f"    Verifier corrections applied: {corrections}/{total}")
    if corrections > 0:
        for r in records:
            if r.get("verifier_correction"):
                print(f"      {r['case_id']}: {r.get('verifier_failed_checks', [])}")


print("\n" + "="*60)
print("V4 CLASSIFIER + VERIFIER AUDIT")
print("="*60)
_classifier_accuracy_report(
    "../ragas_eval_00_v4/wound_ragas_ablation_results_00_v4.json", "v4_00"
)
_classifier_accuracy_report(
    "../ragas_eval_01_v4/wound_ragas_ablation_results_01_v4.json", "v4_01"
)


# ──────────────────────────────────────────────────────────────────────────────
# CELL 14: Multi-version grouped bar chart (v2 / v3 / v4)
# ──────────────────────────────────────────────────────────────────────────────
# Produces a grouped bar chart showing all 8 versions side-by-side for each
# of the 4 RAGAS metrics.  Saved to ragas_eval_00_v4/comparison_chart_v4.png.
# ──────────────────────────────────────────────────────────────────────────────

if HAS_MPL:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    versions = list(comparison_full.columns)
    x        = np.arange(len(versions))
    colors   = [
        "#B5D4F4", "#85B7EB", "#378ADD",   # v2: light → mid blue
        "#9FE1CB", "#5DCAA5", "#1D9E75",   # v3: light → mid teal
        "#FAC775", "#EF9F27",              # v4: light → mid amber
    ]

    for ax_i, (label, col) in enumerate(zip(METRIC_LABELS, METRIC_COLS)):
        vals = comparison_full.loc[label, :].values.astype(float)
        bars = axes[ax_i].bar(x, vals, color=colors, alpha=0.9, zorder=3)
        axes[ax_i].set_ylim(0, 1.05)
        axes[ax_i].set_xticks(x)
        axes[ax_i].set_xticklabels(versions, rotation=40, ha="right", fontsize=8)
        axes[ax_i].set_title(label, fontsize=11)
        axes[ax_i].set_ylabel("Score")
        axes[ax_i].grid(axis="y", alpha=0.3, zorder=0)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                axes[ax_i].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7
                )

    # Legend
    legend_patches = [
        mpatches.Patch(color="#378ADD", label="v2 series"),
        mpatches.Patch(color="#1D9E75", label="v3 series"),
        mpatches.Patch(color="#EF9F27", label="v4 series"),
    ]
    fig.legend(handles=legend_patches, loc="upper right", fontsize=9)
    fig.suptitle("RAGAS Metrics — v2 vs v3 vs v4", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.92, 1])

    chart_out = "../ragas_eval_00_v4/comparison_chart_v4.png"
    fig.savefig(chart_out, dpi=150)
    plt.close(fig)
    print(f"\nComparison chart saved → {chart_out}")
