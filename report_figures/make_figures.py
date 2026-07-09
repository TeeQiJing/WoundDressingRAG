"""Generate report-ready figures for the FYP2 VLM ablation story (G4-P/A/B/C).
Colorblind-safe Okabe-Ito palette (validated), direct value labels, recessive grid."""
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams

OUT = os.path.dirname(os.path.abspath(__file__)); os.makedirs(OUT, exist_ok=True)
# validated Okabe-Ito subset
BLUE, ORANGE, GREEN, RED = "#0072B2", "#E69F00", "#009E73", "#D55E00"
NEUTRAL, INK, MUTED = "#9AA6B2", "#1a2230", "#5b6b82"
rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#c9d2dc",
                 "axes.linewidth": 0.8, "figure.dpi": 140, "savefig.dpi": 200, "savefig.bbox": "tight"})

def _clean(ax, xgrid=False):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(length=0, colors=MUTED)
    if xgrid: ax.xaxis.grid(True, color="#eef2f7", lw=1, zorder=0); ax.set_axisbelow(True)
    else:     ax.yaxis.grid(True, color="#eef2f7", lw=1, zorder=0); ax.set_axisbelow(True)

# ── Fig 1 — VLM-DISC by prompt (G4-P): only blind cross-validates ──────────────
labels = ["P1 Appearance", "P2 Cross-valid.", "P3 Terse", "P4 Blind"]
vals   = [14, 19, 14, 100]
cols   = [NEUTRAL, NEUTRAL, NEUTRAL, GREEN]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
b = ax.barh(labels, vals, color=cols, height=0.62, zorder=3)
for i, v in enumerate(vals):
    ax.text(v + 2, i, f"{v}%", va="center", ha="left", color=INK, fontweight="bold", fontsize=10.5)
ax.set_xlim(0, 112); ax.invert_yaxis(); _clean(ax, xgrid=True)
ax.set_xlabel("VLM-DISC — discrepancy detection (%)", color=MUTED)
ax.set_title("Only the blind prompt cross-validates the labels", color=INK, fontweight="bold", loc="left", pad=8)
ax.text(0.0, -0.30, "G4-P · 7 adversarial cases × 3 runs · VLM = GPT-4o-mini", transform=ax.transAxes,
        color=MUTED, fontsize=8.5)
fig.savefig(f"{OUT}/fig1_g4p_vlm_disc.png"); plt.close(fig)

# ── Fig 2 — Refusal rate by VLM (G4-B + G4-C): only Gemini refuses ─────────────
m = ["GPT-4o-mini-V", "Gemini-2.5-Flash-V", "Qwen2.5-VL-72B", "Qwen3-VL-235B", "Gemma-3-27B", "Gemma-4-26B"]
r = [0, 41, 1, 0, 1, 0]
kind = ["closed", "closed", "open", "open", "open", "open"]
cols = [RED if v >= 10 else NEUTRAL for v in r]
fig, ax = plt.subplots(figsize=(6.6, 3.4))
b = ax.barh(m, r, color=cols, height=0.64, zorder=3)
for i, v in enumerate(r):
    ax.text(v + 0.8, i, f"{v}%", va="center", ha="left", color=INK, fontweight="bold", fontsize=10)
ax.set_xlim(0, 48); ax.invert_yaxis(); _clean(ax, xgrid=True)
ax.set_xlabel("Clinical-image refusal rate (%)", color=MUTED)
ax.set_title("Only Gemini refuses graphic wound images", color=INK, fontweight="bold", loc="left", pad=8)
ax.text(0.0, -0.26, "G4-B + G4-C · 34 cases × 3 runs · Gemini block is non-configurable (BlockedReason.OTHER)",
        transform=ax.transAxes, color=MUTED, fontsize=8.5)
fig.savefig(f"{OUT}/fig2_refusal_rate.png"); plt.close(fig)

# ── Fig 3 — VLM-DISC vs accuracy (the gameable-DISC finding) ───────────────────
pts = [  # name, infection_acc, vlm_disc, group
    ("GPT-4o-mini", 73, 100, "best"), ("Qwen2.5-VL-72B", 76, 71, "best"),
    ("Qwen3-VL-235B", 68, 57, "mid"), ("Gemma-4-26B", 60, 86, "mid"),
    ("Gemma-3-27B", 49, 100, "overcaller")]
cmap = {"best": GREEN, "mid": BLUE, "overcaller": RED}
fig, ax = plt.subplots(figsize=(6.4, 4.4))
# over-caller region (high DISC + low accuracy = top-left)
ax.axhspan(90, 105, xmin=0, xmax=(60-40)/(85-40), color="#fbe9e2", zorder=0)
ax.text(41, 93, "over-caller zone\n(flags everything)", color=RED, fontsize=8.5, va="center")
for name, x, y, g in pts:
    ax.scatter(x, y, s=130, color=cmap[g], edgecolor="white", lw=1.5, zorder=4)
    dy = -7 if name in ("GPT-4o-mini",) else (9 if g == "overcaller" else 7)
    ax.annotate(name, (x, y), xytext=(0, dy), textcoords="offset points",
                ha="center", va="bottom" if dy > 0 else "top", color=INK, fontsize=9)
ax.set_xlim(40, 85); ax.set_ylim(45, 108); _clean(ax)
ax.set_xlabel("Infection accuracy on clean cases (%)  →  genuine perception", color=MUTED)
ax.set_ylabel("VLM-DISC (%)", color=MUTED)
ax.set_title("VLM-DISC is gameable: high detection ≠ good perception", color=INK, fontweight="bold", loc="left", pad=8)
ax.text(0.0, -0.20, "G4-C · Gemma-3 hits 100% DISC by reading 'Infected' on 95% of clean wounds",
        transform=ax.transAxes, color=MUTED, fontsize=8.5)
fig.savefig(f"{OUT}/fig3_disc_vs_accuracy.png"); plt.close(fig)

# ── Fig 4 — caption accuracy (infection + tissue) by VLM ───────────────────────
models = ["GPT-4o-mini", "Qwen2.5-VL-72B", "Qwen3-VL-235B", "Gemma-4-26B", "Gemma-3-27B"]
infacc = [73, 76, 68, 60, 49]
tisacc = [86, 85, 79, 78, 35]
import numpy as np
x = np.arange(len(models)); w = 0.38
fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.bar(x - w/2, infacc, w, color=BLUE, label="Infection accuracy", zorder=3)
ax.bar(x + w/2, tisacc, w, color=ORANGE, label="Tissue-bucket accuracy", zorder=3)
for i in range(len(models)):
    ax.text(x[i]-w/2, infacc[i]+1.5, f"{infacc[i]}", ha="center", color=INK, fontsize=8.5)
    ax.text(x[i]+w/2, tisacc[i]+1.5, f"{tisacc[i]}", ha="center", color=INK, fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(models, rotation=18, ha="right", fontsize=9)
ax.set_ylim(0, 100); _clean(ax); ax.set_ylabel("Accuracy (%)", color=MUTED)
ax.legend(frameon=False, loc="upper right", fontsize=9)
ax.set_title("Caption accuracy: best open model ≈ GPT-4o-mini", color=INK, fontweight="bold", loc="left", pad=8)
ax.text(0.0, -0.40, "G4-B (GPT-4o-mini) + G4-C (open) · blind prompt · Gemini omitted (41% refusals → partial data)",
        transform=ax.transAxes, color=MUTED, fontsize=8.5)
fig.savefig(f"{OUT}/fig4_caption_accuracy.png"); plt.close(fig)

print("[OK] wrote fig1_g4p_vlm_disc.png · fig2_refusal_rate.png · fig3_disc_vs_accuracy.png · fig4_caption_accuracy.png")
