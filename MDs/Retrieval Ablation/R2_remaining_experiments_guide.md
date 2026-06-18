# R2 Remaining Experiments — Continuation Guide

**Target:** Complete `R2-D-MiniLM-L12` (re-run all 3 rounds clean) and `R2-D-BGE-v2-m3` (run all 3 rounds fresh).

---

## Step 1 — Which original cells to re-run as setup

Open the notebook and run these cells **in order** before pasting the new cells below. **Do NOT run Cell 13 (the main experiment loop).**

| Cell | Label | Why you need it |
|------|-------|-----------------|
| Cell 0 | Environment & Imports | All imports, paths, RESULTS_DIR |
| Cell 1 | Configuration | `EXP_ID`, `K`, `N_RUNS`, `EMBED_MODEL`, `RAGAS_LLM/EMBED`, `BM25_WEIGHT`, `DENSE_WEIGHT`, `RERANK_K_POOL`, `RERANKERS`, `VERSIONS`, `LABELS` |
| Cell 2 | Load Testset | `testset` (32 cases) |
| Cell 3 | Load Embedding Model & Chroma DB | `embedding_model`, `db` |
| Cell 4 | Build BM25 Index | `all_kb_docs`, `bm25_retriever` |
| Cell 5 | RAGAS Judge Setup | `ragas_llm`, `ragas_embed` |
| Cell 6 | Load Cross-Encoder Rerankers | `_reranker_cache`, `get_reranker()` — loads all 3 rerankers including BGE |
| Cell 7 | Input Normalisation & Clinical Pre-Classifier | `interpret_tissue_percentages()`, `normalize_*()`, `parse_time_payload()`, `classify_wound()` |
| Cell 8 | R1-C Multi-Axis Sub-Query Builder | `get_r1c_subqueries()` |
| Cell 9 | Retrieval Implementations | `_dedup_merge()`, `retrieve_R2A/B/C/D()`, `_make_ensemble()`, `_build_rerank_query()` |
| Cell 10 | Retrieval Metrics | `compute_retrieval_metrics()`, `compute_additional_metrics()` |
| Cell 11 | RAGAS Evaluation Helper | `run_ragas_evaluation()` |
| Cell 12 | Core Retrieval Runner | `run_retrieval_for_version()` |

**Then skip Cell 13** (the main loop that broke) and paste the new cells at the bottom.

---

## Step 2 — New cells to paste at the bottom of the notebook

### Cell R-A: Status check — confirm prerequisites are loaded

```python
# ── Cell R-A: Prerequisite check ─────────────────────────────────────────────
# Run this first to confirm all required objects are in scope before proceeding.

required = {
    "testset":              lambda: len(testset) == 32,
    "db":                   lambda: db._collection.count() == 138,
    "bm25_retriever":       lambda: bm25_retriever is not None,
    "ragas_llm":            lambda: ragas_llm is not None,
    "ragas_embed":          lambda: ragas_embed is not None,
    "get_reranker (BGE)":   lambda: get_reranker("BGE-v2-m3") is not None,
    "get_reranker (L12)":   lambda: get_reranker("MiniLM-L12") is not None,
    "run_retrieval_for_version": lambda: callable(run_retrieval_for_version),
    "run_ragas_evaluation": lambda: callable(run_ragas_evaluation),
}

print("─── Prerequisite check ──────────────────────────────────")
all_ok = True
for name, check in required.items():
    try:
        ok = check()
        status = "✓" if ok else "✗ CHECK FAILED"
    except Exception as e:
        ok = False
        status = f"✗ ERROR: {e}"
    print(f"  {name:<35}: {status}")
    if not ok:
        all_ok = False

print("─────────────────────────────────────────────────────────")
print("✓ All prerequisites OK — proceed to Cell R-B." if all_ok else
      "✗ Fix missing prerequisites above before running Cell R-B.")
```

---

### Cell R-B: Run R2-D-MiniLM-L12 — full 3 rounds (replaces incomplete run)

> **Note:** This fully re-runs all 3 rounds and will **overwrite** `R2_R2DMiniLML12_results.json` and `R2_R2DMiniLML12_ragas.json`. The previous 2-round incomplete data is discarded in favour of a clean 3-round result.

```python
# ── Cell R-B: R2-D-MiniLM-L12 — 3 clean runs ────────────────────────────────
import statistics, json, datetime
from pathlib import Path

REDO_VERSION  = "R2-D-MiniLM-L12"
REDO_TAG      = "MiniLM-L12"
N_RUNS        = 3
K             = 6

print(f"\n{'#'*70}")
print(f"  RERUN: {REDO_VERSION}  |  {LABELS[REDO_VERSION]}  |  {N_RUNS} runs (clean)")
print(f"  NOTE: Discards previous incomplete 2-round data.")
print(f"{'#'*70}")

redo_l12_runs = []

for run_idx in range(1, N_RUNS + 1):
    print(f"\n  ── Run {run_idx}/{N_RUNS} ──")

    run_data = run_retrieval_for_version(REDO_VERSION, testset, k=K)

    print(f"\n  Running RAGAS (CR + CP) for {REDO_VERSION} run {run_idx} ...")
    ragas_scores = run_ragas_evaluation(
        questions          = run_data["questions"],
        retrieved_contexts = run_data["all_retrieved_texts"],
        reference_contexts = run_data["all_reference_texts"],
        references         = run_data["references"],
    )
    print(f"  RAGAS CR: {ragas_scores['context_recall']:.4f}  "
          f"CP: {ragas_scores['context_precision']:.4f}")

    run_data["ragas"]   = ragas_scores
    run_data["run_idx"] = run_idx
    redo_l12_runs.append(run_data)

print(f"\n✓ R2-D-MiniLM-L12: {len(redo_l12_runs)}/3 runs complete.")

# ── Aggregate ─────────────────────────────────────────────────────────────────
def _ms(vals):
    m = round(statistics.mean(vals), 4)
    s = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
    return m, s

cr_m,  cr_s  = _ms([r["ragas"]["context_recall"]     for r in redo_l12_runs])
cp_m,  cp_s  = _ms([r["ragas"]["context_precision"]   for r in redo_l12_runs])
hr_m,  hr_s  = _ms([r["ir_metrics"]["hit_rate_at_k"]  for r in redo_l12_runs])
mrr_m, mrr_s = _ms([r["ir_metrics"]["mrr"]            for r in redo_l12_runs])
nd_m,  nd_s  = _ms([r["ir_metrics"]["ndcg_at_k"]      for r in redo_l12_runs])
rc_m,  rc_s  = _ms([r["ir_metrics"]["recall_at_k"]    for r in redo_l12_runs])
pr_m,  pr_s  = _ms([r["ir_metrics"]["precision_at_k"] for r in redo_l12_runs])
lt_m,  lt_s  = _ms([r["latency_ms_mean"]              for r in redo_l12_runs])

agg_l12 = {
    "context_recall_mean":      cr_m,  "context_recall_sd":      cr_s,
    "context_precision_mean":   cp_m,  "context_precision_sd":   cp_s,
    "hit_rate_at_k_mean":       hr_m,  "hit_rate_at_k_sd":       hr_s,
    "mrr_mean":                 mrr_m, "mrr_sd":                 mrr_s,
    "ndcg_at_k_mean":           nd_m,  "ndcg_at_k_sd":           nd_s,
    "recall_at_k_mean":         rc_m,  "recall_at_k_sd":         rc_s,
    "precision_at_k_mean":      pr_m,  "precision_at_k_sd":      pr_s,
    "retrieval_latency_ms_mean": lt_m, "retrieval_latency_ms_sd": lt_s,
    "per_run_cr":       [r["ragas"]["context_recall"]     for r in redo_l12_runs],
    "per_run_cp":       [r["ragas"]["context_precision"]  for r in redo_l12_runs],
    "per_run_hr":       [r["ir_metrics"]["hit_rate_at_k"] for r in redo_l12_runs],
    "per_run_mrr":      [r["ir_metrics"]["mrr"]           for r in redo_l12_runs],
    "per_run_ndcg":     [r["ir_metrics"]["ndcg_at_k"]     for r in redo_l12_runs],
    "per_run_recall":   [r["ir_metrics"]["recall_at_k"]   for r in redo_l12_runs],
    "per_run_precision":[r["ir_metrics"]["precision_at_k"] for r in redo_l12_runs],
    "per_run_latency":  [r["latency_ms_mean"]             for r in redo_l12_runs],
}

print(f"\n  {REDO_VERSION} — FINAL (3 clean runs)")
print(f"    CR:      {cr_m:.4f} ± {cr_s:.4f}")
print(f"    CP:      {cp_m:.4f} ± {cp_s:.4f}")
print(f"    HR@{K}:   {hr_m:.4f} ± {hr_s:.4f}")
print(f"    MRR:     {mrr_m:.4f} ± {mrr_s:.4f}")
print(f"    NDCG@{K}: {nd_m:.4f} ± {nd_s:.4f}")
print(f"    R@{K}:    {rc_m:.4f} ± {rc_s:.4f}")
print(f"    P@{K}:    {pr_m:.4f} ± {pr_s:.4f}")
print(f"    Lat:     {lt_m:.1f} ± {lt_s:.1f} ms")

# ── Save — overwrites the incomplete previous file ────────────────────────────
timestamp_l12 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
version_tag   = REDO_VERSION.replace("-", "")   # R2DMiniLML12

results_payload = {
    "experiment":  EXP_ID,
    "version":     REDO_VERSION,
    "label":       LABELS[REDO_VERSION],
    "timestamp":   timestamp_l12,
    "note":        "Full clean 3-run rerun — replaces incomplete 2-run result from original session.",
    "config": {
        "k": K, "n_runs": N_RUNS,
        "embedding_model": EMBED_MODEL,
        "query_strategy":  "R1-C multi-axis sub-queries (fixed from R1)",
        "bm25_weight":     BM25_WEIGHT,
        "dense_weight":    DENSE_WEIGHT,
        "rerank_k_pool":   RERANK_K_POOL,
        "reranker_model":  RERANKERS[REDO_TAG],
        "ragas_llm":       RAGAS_LLM,
        "ragas_embed":     RAGAS_EMBED,
        "db_path":         str(DB_PATH),
        "testset_path":    str(TESTSET_PATH),
    },
    "aggregated_metrics": agg_l12,
    "runs": [
        {
            "run_idx":           r["run_idx"],
            "ir_metrics":        r["ir_metrics"],
            "additional_metrics":r["additional_metrics"],
            "latency_ms_mean":   r["latency_ms_mean"],
            "latency_ms_sd":     r["latency_ms_sd"],
            "per_case":          r["per_case"],
        }
        for r in redo_l12_runs
    ],
}

ragas_payload = {
    "experiment":  EXP_ID,
    "version":     REDO_VERSION,
    "label":       LABELS[REDO_VERSION],
    "timestamp":   timestamp_l12,
    "note":        "Full clean 3-run rerun — replaces incomplete 2-run result from original session.",
    "ragas_config": {
        "llm":     RAGAS_LLM,
        "embed":   RAGAS_EMBED,
        "metrics": ["context_recall", "context_precision"],
        "note":    "Stage 1 retrieval ablation — FA and AR excluded (no generation step)",
    },
    "aggregated": {
        "context_recall_mean":    cr_m, "context_recall_sd":    cr_s,
        "context_precision_mean": cp_m, "context_precision_sd": cp_s,
        "per_run_cr": agg_l12["per_run_cr"],
        "per_run_cp": agg_l12["per_run_cp"],
    },
    "per_run_ragas": [
        {
            "run_idx":    r["run_idx"],
            "cr":         r["ragas"]["context_recall"],
            "cp":         r["ragas"]["context_precision"],
            "per_case_cr":r["ragas"]["per_case_cr"],
            "per_case_cp":r["ragas"]["per_case_cp"],
        }
        for r in redo_l12_runs
    ],
}

results_path = RESULTS_DIR / f"{EXP_ID}_{version_tag}_results.json"
ragas_path   = RESULTS_DIR / f"{EXP_ID}_{version_tag}_ragas.json"

with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results_payload, f, indent=2, ensure_ascii=False)
with open(ragas_path, "w", encoding="utf-8") as f:
    json.dump(ragas_payload, f, indent=2, ensure_ascii=False)

print(f"\nSaved results : {results_path}")
print(f"Saved RAGAS   : {ragas_path}")
print("✓ R2-D-MiniLM-L12 complete and saved.")
```

---

### Cell R-C: Run R2-D-BGE-v2-m3 — 3 fresh runs

> **Note:** No existing JSON for this version. This is a clean first run.

```python
# ── Cell R-C: R2-D-BGE-v2-m3 — 3 fresh runs ─────────────────────────────────
BGE_VERSION = "R2-D-BGE-v2-m3"
BGE_TAG     = "BGE-v2-m3"
N_RUNS      = 3
K           = 6

print(f"\n{'#'*70}")
print(f"  VERSION: {BGE_VERSION}  |  {LABELS[BGE_VERSION]}  |  {N_RUNS} runs (fresh)")
print(f"{'#'*70}")

bge_runs = []

for run_idx in range(1, N_RUNS + 1):
    print(f"\n  ── Run {run_idx}/{N_RUNS} ──")

    run_data = run_retrieval_for_version(BGE_VERSION, testset, k=K)

    print(f"\n  Running RAGAS (CR + CP) for {BGE_VERSION} run {run_idx} ...")
    ragas_scores = run_ragas_evaluation(
        questions          = run_data["questions"],
        retrieved_contexts = run_data["all_retrieved_texts"],
        reference_contexts = run_data["all_reference_texts"],
        references         = run_data["references"],
    )
    print(f"  RAGAS CR: {ragas_scores['context_recall']:.4f}  "
          f"CP: {ragas_scores['context_precision']:.4f}")

    run_data["ragas"]   = ragas_scores
    run_data["run_idx"] = run_idx
    bge_runs.append(run_data)

print(f"\n✓ R2-D-BGE-v2-m3: {len(bge_runs)}/3 runs complete.")

# ── Aggregate ─────────────────────────────────────────────────────────────────
cr_m,  cr_s  = _ms([r["ragas"]["context_recall"]     for r in bge_runs])
cp_m,  cp_s  = _ms([r["ragas"]["context_precision"]   for r in bge_runs])
hr_m,  hr_s  = _ms([r["ir_metrics"]["hit_rate_at_k"]  for r in bge_runs])
mrr_m, mrr_s = _ms([r["ir_metrics"]["mrr"]            for r in bge_runs])
nd_m,  nd_s  = _ms([r["ir_metrics"]["ndcg_at_k"]      for r in bge_runs])
rc_m,  rc_s  = _ms([r["ir_metrics"]["recall_at_k"]    for r in bge_runs])
pr_m,  pr_s  = _ms([r["ir_metrics"]["precision_at_k"] for r in bge_runs])
lt_m,  lt_s  = _ms([r["latency_ms_mean"]              for r in bge_runs])

agg_bge = {
    "context_recall_mean":      cr_m,  "context_recall_sd":      cr_s,
    "context_precision_mean":   cp_m,  "context_precision_sd":   cp_s,
    "hit_rate_at_k_mean":       hr_m,  "hit_rate_at_k_sd":       hr_s,
    "mrr_mean":                 mrr_m, "mrr_sd":                 mrr_s,
    "ndcg_at_k_mean":           nd_m,  "ndcg_at_k_sd":           nd_s,
    "recall_at_k_mean":         rc_m,  "recall_at_k_sd":         rc_s,
    "precision_at_k_mean":      pr_m,  "precision_at_k_sd":      pr_s,
    "retrieval_latency_ms_mean": lt_m, "retrieval_latency_ms_sd": lt_s,
    "per_run_cr":       [r["ragas"]["context_recall"]     for r in bge_runs],
    "per_run_cp":       [r["ragas"]["context_precision"]  for r in bge_runs],
    "per_run_hr":       [r["ir_metrics"]["hit_rate_at_k"] for r in bge_runs],
    "per_run_mrr":      [r["ir_metrics"]["mrr"]           for r in bge_runs],
    "per_run_ndcg":     [r["ir_metrics"]["ndcg_at_k"]     for r in bge_runs],
    "per_run_recall":   [r["ir_metrics"]["recall_at_k"]   for r in bge_runs],
    "per_run_precision":[r["ir_metrics"]["precision_at_k"] for r in bge_runs],
    "per_run_latency":  [r["latency_ms_mean"]             for r in bge_runs],
}

print(f"\n  {BGE_VERSION} — FINAL (3 runs)")
print(f"    CR:      {cr_m:.4f} ± {cr_s:.4f}")
print(f"    CP:      {cp_m:.4f} ± {cp_s:.4f}")
print(f"    HR@{K}:   {hr_m:.4f} ± {hr_s:.4f}")
print(f"    MRR:     {mrr_m:.4f} ± {mrr_s:.4f}")
print(f"    NDCG@{K}: {nd_m:.4f} ± {nd_s:.4f}")
print(f"    R@{K}:    {rc_m:.4f} ± {rc_s:.4f}")
print(f"    P@{K}:    {pr_m:.4f} ± {pr_s:.4f}")
print(f"    Lat:     {lt_m:.1f} ± {lt_s:.1f} ms")

# ── Save ──────────────────────────────────────────────────────────────────────
timestamp_bge = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
version_tag   = BGE_VERSION.replace("-", "")   # R2DBGEv2m3

results_payload = {
    "experiment":  EXP_ID,
    "version":     BGE_VERSION,
    "label":       LABELS[BGE_VERSION],
    "timestamp":   timestamp_bge,
    "config": {
        "k": K, "n_runs": N_RUNS,
        "embedding_model": EMBED_MODEL,
        "query_strategy":  "R1-C multi-axis sub-queries (fixed from R1)",
        "bm25_weight":     BM25_WEIGHT,
        "dense_weight":    DENSE_WEIGHT,
        "rerank_k_pool":   RERANK_K_POOL,
        "reranker_model":  RERANKERS[BGE_TAG],
        "ragas_llm":       RAGAS_LLM,
        "ragas_embed":     RAGAS_EMBED,
        "db_path":         str(DB_PATH),
        "testset_path":    str(TESTSET_PATH),
    },
    "aggregated_metrics": agg_bge,
    "runs": [
        {
            "run_idx":           r["run_idx"],
            "ir_metrics":        r["ir_metrics"],
            "additional_metrics":r["additional_metrics"],
            "latency_ms_mean":   r["latency_ms_mean"],
            "latency_ms_sd":     r["latency_ms_sd"],
            "per_case":          r["per_case"],
        }
        for r in bge_runs
    ],
}

ragas_payload = {
    "experiment":  EXP_ID,
    "version":     BGE_VERSION,
    "label":       LABELS[BGE_VERSION],
    "timestamp":   timestamp_bge,
    "ragas_config": {
        "llm":     RAGAS_LLM,
        "embed":   RAGAS_EMBED,
        "metrics": ["context_recall", "context_precision"],
        "note":    "Stage 1 retrieval ablation — FA and AR excluded (no generation step)",
    },
    "aggregated": {
        "context_recall_mean":    cr_m, "context_recall_sd":    cr_s,
        "context_precision_mean": cp_m, "context_precision_sd": cp_s,
        "per_run_cr": agg_bge["per_run_cr"],
        "per_run_cp": agg_bge["per_run_cp"],
    },
    "per_run_ragas": [
        {
            "run_idx":    r["run_idx"],
            "cr":         r["ragas"]["context_recall"],
            "cp":         r["ragas"]["context_precision"],
            "per_case_cr":r["ragas"]["per_case_cr"],
            "per_case_cp":r["ragas"]["per_case_cp"],
        }
        for r in bge_runs
    ],
}

results_path = RESULTS_DIR / f"{EXP_ID}_{version_tag}_results.json"
ragas_path   = RESULTS_DIR / f"{EXP_ID}_{version_tag}_ragas.json"

with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results_payload, f, indent=2, ensure_ascii=False)
with open(ragas_path, "w", encoding="utf-8") as f:
    json.dump(ragas_payload, f, indent=2, ensure_ascii=False)

print(f"\nSaved results : {results_path}")
print(f"Saved RAGAS   : {ragas_path}")
print("✓ R2-D-BGE-v2-m3 complete and saved.")
```

---

### Cell R-D: Combined final summary across all 6 versions

This cell **loads everything from JSON** (no dependency on in-memory `all_run_results`) so it works cleanly even after a kernel restart between the original session and this one.

```python
# ── Cell R-D: Full R2 summary — load all 6 versions from saved JSONs ─────────
import json, statistics
from pathlib import Path

# Map version → results filename stem (mirrors Cell 16 naming convention)
VERSION_FILE_MAP = {
    "R2-A":             "R2_R2A_results.json",
    "R2-B":             "R2_R2B_results.json",
    "R2-C":             "R2_R2C_results.json",
    "R2-D-MiniLM-L6":  "R2_R2DMiniLML6_results.json",
    "R2-D-MiniLM-L12": "R2_R2DMiniLML12_results.json",
    "R2-D-BGE-v2-m3":  "R2_R2DBGEv2m3_results.json",
}

agg_final = {}
missing   = []

for version, fname in VERSION_FILE_MAP.items():
    fpath = RESULTS_DIR / fname
    if not fpath.exists():
        print(f"  ✗ MISSING: {fpath}")
        missing.append(version)
        continue
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    agg_final[version] = data["aggregated_metrics"]

if missing:
    print(f"\n⚠ Cannot produce full summary — missing: {missing}")
else:
    VERSIONS_ALL = list(VERSION_FILE_MAP.keys())
    print(f"\n{'='*145}")
    print(f"  Stage 1 — R2 Retrieval Strategy Ablation | Fixed: R1-C multi-axis query · k=6 · MedEmbed-large-v0.1 · 3 runs each")
    print(f"{'='*145}")
    header = (
        f"  {'Version':<22} | "
        f"{'CR':>6} ± {'SD':>6} | {'CP':>6} ± {'SD':>6} | "
        f"{'HR@K':>6} ± {'SD':>6} | {'MRR':>6} ± {'SD':>6} | "
        f"{'NDCG@K':>7} ± {'SD':>6} | {'R@K':>6} ± {'SD':>6} | "
        f"{'P@K':>6} ± {'SD':>6} | {'Lat(ms)':>8} ± {'SD':>6}"
    )
    print(header)
    print("-"*145)

    for version in VERSIONS_ALL:
        a = agg_final[version]
        row = (
            f"  {version:<22} | "
            f"{a['context_recall_mean']:>6.4f} ± {a['context_recall_sd']:>6.4f} | "
            f"{a['context_precision_mean']:>6.4f} ± {a['context_precision_sd']:>6.4f} | "
            f"{a['hit_rate_at_k_mean']:>6.4f} ± {a['hit_rate_at_k_sd']:>6.4f} | "
            f"{a['mrr_mean']:>6.4f} ± {a['mrr_sd']:>6.4f} | "
            f"{a['ndcg_at_k_mean']:>7.4f} ± {a['ndcg_at_k_sd']:>6.4f} | "
            f"{a['recall_at_k_mean']:>6.4f} ± {a['recall_at_k_sd']:>6.4f} | "
            f"{a['precision_at_k_mean']:>6.4f} ± {a['precision_at_k_sd']:>6.4f} | "
            f"{a['retrieval_latency_ms_mean']:>8.1f} ± {a['retrieval_latency_ms_sd']:>6.1f}"
        )
        print(row)

    print("="*145)

    # Best per metric
    print(f"\n  Best CR      : {max(VERSIONS_ALL, key=lambda v: agg_final[v]['context_recall_mean'])}")
    print(f"  Best CP      : {max(VERSIONS_ALL, key=lambda v: agg_final[v]['context_precision_mean'])}")
    print(f"  Best HR@6    : {max(VERSIONS_ALL, key=lambda v: agg_final[v]['hit_rate_at_k_mean'])}")
    print(f"  Best MRR     : {max(VERSIONS_ALL, key=lambda v: agg_final[v]['mrr_mean'])}")
    print(f"  Best NDCG@6  : {max(VERSIONS_ALL, key=lambda v: agg_final[v]['ndcg_at_k_mean'])}")
    print(f"  Best R@6     : {max(VERSIONS_ALL, key=lambda v: agg_final[v]['recall_at_k_mean'])}")
    print(f"  Fastest      : {min(VERSIONS_ALL, key=lambda v: agg_final[v]['retrieval_latency_ms_mean'])}")

    # R2-D sub-comparison
    RERANKER_VERSIONS = ["R2-D-MiniLM-L6", "R2-D-MiniLM-L12", "R2-D-BGE-v2-m3"]
    print(f"\n  ── R2-D Reranker Sub-comparison ──")
    for v in RERANKER_VERSIONS:
        a = agg_final[v]
        tag = v.replace("R2-D-", "")
        print(f"    {tag:<14}: CR={a['context_recall_mean']:.4f}  CP={a['context_precision_mean']:.4f}  "
              f"MRR={a['mrr_mean']:.4f}  NDCG={a['ndcg_at_k_mean']:.4f}  Lat={a['retrieval_latency_ms_mean']:.0f}ms")

    best_r2d = max(RERANKER_VERSIONS, key=lambda v: agg_final[v]["context_recall_mean"])
    best_overall = max(VERSIONS_ALL, key=lambda v: agg_final[v]["context_recall_mean"])
    print(f"\n  Best R2-D reranker (by CR) : {best_r2d}")
    print(f"  Best overall (by CR)       : {best_overall}  ← carry forward to R3")
    print(f"\n  Files in: {RESULTS_DIR}")
```

---

### Cell R-E: (Optional) Save updated R2 summary JSON with all 6 versions

```python
# ── Cell R-E: Save complete R2 summary JSON ──────────────────────────────────
# Only run after Cell R-D confirms all 6 versions loaded without missing files.

if missing:
    print("⚠ Skipping — still have missing versions:", missing)
else:
    timestamp_summary = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_payload = {
        "experiment": EXP_ID,
        "title":      "R2 — Retrieval Strategy Ablation (COMPLETE — all 6 versions)",
        "timestamp":  timestamp_summary,
        "config": {
            "k":               K,
            "n_runs":          N_RUNS,
            "embedding_model": EMBED_MODEL,
            "query_strategy":  "R1-C (fixed from R1)",
            "retrieval_variants": LABELS,
            "bm25_weight":     BM25_WEIGHT,
            "dense_weight":    DENSE_WEIGHT,
            "rerank_k_pool":   RERANK_K_POOL,
            "rerankers":       RERANKERS,
            "ragas_llm":       RAGAS_LLM,
            "ragas_embed":     RAGAS_EMBED,
        },
        "results": agg_final,
        "best_by_metric": {
            "context_recall":    max(VERSIONS_ALL, key=lambda v: agg_final[v]["context_recall_mean"]),
            "context_precision": max(VERSIONS_ALL, key=lambda v: agg_final[v]["context_precision_mean"]),
            "hit_rate_at_k":     max(VERSIONS_ALL, key=lambda v: agg_final[v]["hit_rate_at_k_mean"]),
            "mrr":               max(VERSIONS_ALL, key=lambda v: agg_final[v]["mrr_mean"]),
            "ndcg_at_k":         max(VERSIONS_ALL, key=lambda v: agg_final[v]["ndcg_at_k_mean"]),
            "recall_at_k":       max(VERSIONS_ALL, key=lambda v: agg_final[v]["recall_at_k_mean"]),
            "fastest_latency":   min(VERSIONS_ALL, key=lambda v: agg_final[v]["retrieval_latency_ms_mean"]),
        },
        "r2d_best_reranker_by_cr": max(
            RERANKER_VERSIONS, key=lambda v: agg_final[v]["context_recall_mean"]
        ),
        "recommended_for_r3": max(
            VERSIONS_ALL, key=lambda v: agg_final[v]["context_recall_mean"]
        ),
    }

    summary_path = RESULTS_DIR / f"{EXP_ID}_summary_complete.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)
    print(f"Saved complete summary: {summary_path}")
    print("✓ R2 ablation fully complete.")
```

---

## Notes on the R2-D-MiniLM-L12 rerun decision

The original 3rd run hit cascading RAGAS `TimeoutError` on 26+ of the 64 evaluation jobs, leaving nearly half the per-case scores as `NaN`. The aggregated SD (±0.0380 on CP, ±0.0252 on CR) reflects that noise rather than true run-to-run variance. A clean 3-round rerun gives you an honest mean ± SD to report in the ablation table. The saved JSON includes a `"note"` field recording the reason for the rerun for audit trail purposes.

## Expected runtime for remaining work

| Version | Retrieval (3 runs) | RAGAS (3 runs) | Estimated total |
|---------|-------------------|----------------|-----------------|
| R2-D-MiniLM-L12 | ~18 min | ~20–25 min | ~40–45 min |
| R2-D-BGE-v2-m3  | ~25 min | ~20–25 min | ~45–50 min |
| **Total** | | | **~90 min** |

BGE-v2-m3 retrieval is the slowest (~310–350 ms/case × 32 cases × 3 runs). Keep the kernel alive and consider running Cell R-B and Cell R-C in separate sittings if needed — each cell saves immediately on completion.
