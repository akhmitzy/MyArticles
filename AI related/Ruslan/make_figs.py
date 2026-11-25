import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#5f5e58", "#e1e0d9"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.6, "lines.markersize": 4.5,
    "pdf.fonttype": 42,
})

# ---------------- Fig 1: E1
e1 = json.load(open(f"{D}/e1.json"))
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.35))
for sa, ls, mk, lab in [(False, "-", "o", "label form, $s'$ sampled"), (True, "--", "s", "label form, $(s',a')$ sampled")]:
    rows = sorted([r for r in e1["stoch"] if r["lam"] == 1.0 and r["sample_a"] == sa], key=lambda r: r["H"])
    H = [r["H"] for r in rows]
    ax[0].errorbar(H, [r["label_q"][0] for r in rows], yerr=[r["label_q"][1] for r in rows], color=ORANGE,
                   ls=ls, marker=mk, capsize=2, label=lab)
rows = sorted([r for r in e1["stoch"] if r["lam"] == 1.0 and r["sample_a"]], key=lambda r: r["H"])
ax[0].plot([r["H"] for r in rows], [max(r["scaled_q"][2], 1e-16) for r in rows], color=BLUE, marker="^",
           label="scaled form (max over seeds)")
ax[0].set_yscale("log"); ax[0].set_ylim(1e-14, 3)
ax[0].set_xlabel("mean transition entropy $H[P(\\cdot|s,a)]$ (nats)")
ax[0].set_ylabel("max rel. error vs. $Q^\\beta\\beta/\\pi$")
ax[0].set_title("(a) stochasticity sweep ($\\lambda=1$)")
ax[0].legend(loc="center right", bbox_to_anchor=(1.0, 0.42), fontsize=7)
for sa, ls, mk in [(False, "-", "o"), (True, "--", "s")]:
    rows = [r for r in e1["scale"] if r["sample_a"] == sa]
    ax[1].plot([r["scale"] for r in rows], [max(r["label_q"][0], 1e-16) for r in rows], color=ORANGE, ls=ls, marker=mk)
    ax[1].plot([r["scale"] for r in rows], [max(r["scaled_q"][2], 1e-16) for r in rows], color=BLUE, ls=ls, marker="^")
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_ylim(1e-16, 3)
ax[1].set_xlabel("reward scale $c$ (rewards $c\\cdot r$)")
ax[1].set_title("(b) reward-scale sweep")
ax[1].text(0.012, 0.5, "label form", color=MUTED, fontsize=7.5)
ax[1].text(0.012, 2e-10, "scaled form", color=MUTED, fontsize=7.5)
fig.tight_layout(); fig.savefig(f"{D}/../fig_e1.pdf"); plt.close(fig)

# ---------------- Fig 2: E2
e2 = json.load(open(f"{D}/e2.json"))
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4), gridspec_kw={"width_ratios": [1.35, 1]})
for gr, ls in [(1.0, "-"), (10.0, "--")]:
    rows = [r for r in e2 if r["goal_r"] == gr]
    sl = [r["slip"] for r in rows]
    ax[0].plot(sl, [r["label_sa"]["maxdiff"] for r in rows], color=ORANGE, ls=ls, marker="s",
               label=f"label form, goal reward {gr:g}")
    ax[0].plot(sl, [r["scaled_sa"]["maxdiff"] for r in rows], color=BLUE, ls=ls, marker="^",
               label=f"scaled form, goal reward {gr:g}")
ax[0].set_xlabel("slip probability"); ax[0].set_ylabel("$\\max_{s,a}|\\hat\\pi(a|s)-\\pi^\\star(a|s)|$")
ax[0].set_title("(a) policy deviation from one-step RL")
ax[0].legend(fontsize=6.8, loc="upper left")
r = [r for r in e2 if r["goal_r"] == 10.0 and r["slip"] == 0.4][0]
ps = np.array(r["pstar"]).ravel()
for key, col, mk, lab, z in [("label_sa", ORANGE, "s", "label", 3), ("scaled_sa", BLUE, "o", "scaled", 4)]:
    ax[1].scatter(ps, np.array(r[key]["pi"]).ravel(), s=22 if key == "scaled_sa" else 14,
                  facecolor="none" if key == "scaled_sa" else col, edgecolor=col, marker=mk,
                  linewidth=1.0 if key == "scaled_sa" else 0.6, label=f"{lab} ($R^2$={r[key]['r2']:.3f})", zorder=z)
ax[1].plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
ax[1].set_xlabel("one-step RL $\\pi^\\star(a|s)$"); ax[1].set_ylabel("critic-reg. $\\hat\\pi(a|s)$")
ax[1].set_title("(b) slip 0.4, goal reward 10")
ax[1].legend(fontsize=6.8, loc="upper left")
fig.tight_layout(); fig.savefig(f"{D}/../fig_e2.pdf"); plt.close(fig)

# ---------------- Fig 3: E3
e3 = json.load(open(f"{D}/e3.json"))
fig, ax = plt.subplots(figsize=(3.4, 2.35))
K = 400
for form, col, mk, lab in [("ce-scaled", BLUE, None, "CE, scaled"), ("lsif-scaled", AQUA, None, "squared (LSIF), scaled"),
                           ("ce-label", ORANGE, None, "CE, label")]:
    c = np.array(e3[form])[:, :, 0]
    x = (np.arange(c.shape[1]) + 1) * K
    med = np.median(c, 0)
    ax.plot(x, med, color=col, label=lab)
    ax.fill_between(x, c.min(0), c.max(0), color=col, alpha=0.18, lw=0)
ax.axhline(e3["label_fp_err"], color=ORANGE, ls=":", lw=1)
ax.text(x[-1], e3["label_fp_err"] * 1.12, "label-form fixed point (Prop. 4)", ha="right", fontsize=6.8, color=MUTED)
ax.set_yscale("log"); ax.set_xlabel("gradient steps (batch 256)")
ax.set_ylabel("median rel. error vs. $Q^\\beta\\beta/\\pi$")
ax.legend(fontsize=6.8, loc="lower left")
fig.tight_layout(); fig.savefig(f"{D}/../fig_e3.pdf"); plt.close(fig)
print("ok")
