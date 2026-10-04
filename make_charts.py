"""Regenerate every chart in the README from results.json and extras.json.

Run:  python make_charts.py
Needs: results.json (from experiments.py) and extras.json (from extras.py).
"""
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE / "charts"
OUT.mkdir(exist_ok=True)

NAVY, BLUE, GREY, RED, GREEN, AMBER = "#1F3864", "#0070C0", "#9AA5B1", "#C00000", "#2E7D32", "#C55A11"
from matplotlib import font_manager as fm
_have = {f.name for f in fm.fontManager.ttflist}
FONT = next((f for f in ("Arial", "Liberation Sans", "DejaVu Sans") if f in _have), "sans-serif")
plt.rcParams.update({
    "font.family": FONT,
    "axes.spines.top": False, "axes.spines.right": False,
})

res = json.load(open(HERE / "results.json"))
ext = json.load(open(HERE / "extras.json"))
A, CI = res["A"], res["CI"]


def r1(v):
    """Round half up to 1 decimal (so 3.15 -> 3.2, matching the README tables)."""
    return str(Decimal(str(round(v, 6))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def finish(fig, name, note):
    fig.text(0.02, 0.012, note, fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / name, dpi=200, facecolor="white")
    plt.close(fig)


# 1. Options pay off only when spare alternate lift is scarce ---------------------------
sweep = ext["sweep"]
keys = ["2.0", "5.0", "8.0"]
b3 = [sweep[k]["B3"] for k in keys]
rr = [sweep[k]["RRL"] for k in keys]
fig, ax = plt.subplots(figsize=(6.2, 4.3))
w = 0.36
ax.bar([i - w / 2 for i in range(3)], b3, w, color=GREY, label="Planner without options")
cols = [BLUE, BLUE, RED]
ax.bar([i + w / 2 for i in range(3)], rr, w, color=cols, label="RRL (with options)")
for i in range(3):
    ax.text(i - w / 2, b3[i] + 0.2, r1(b3[i]), ha="center", fontsize=12, fontweight="bold", color="#404040")
    ax.text(i + w / 2, rr[i] + 0.2, r1(rr[i]), ha="center", fontsize=12, fontweight="bold", color=cols[i])
ax.text(2 + w / 2, rr[2] + 1.2, "RRL loses", ha="center", fontsize=10.5, color=RED, fontweight="bold")
ax.set_xticks(range(3))
ax.set_xticklabels(["2 of 8\n(scarce)", "5 of 8", "8 of 8\n(plentiful)"])
ax.set_xlabel("Spare alternate-route lift while main road is closed (units/day)")
ax.set_ylabel("Stockouts per episode (base-days)")
ax.set_ylim(0, 12.8)
ax.set_title("Options only pay off when spare lift is scarce", fontsize=13, fontweight="bold", color=NAVY)
ax.legend(frameon=False, loc="upper right")
ax.grid(axis="y", alpha=.25)
finish(fig, "options_viability.png", "Simulated network | 20 episodes per setting | identical random events | lower is better")

# 2. Solve time vs network size (true numeric x-axis) ------------------------------------
scal = res["scal"]
x = sorted(int(k) for k in scal)
ms = [scal[str(k)]["t"] * 1000 for k in x]
nv = [scal[str(k)]["vars"] for k in x]
fig, ax = plt.subplots(figsize=(6.4, 4.1))
ax.plot(x, ms, marker="o", color=BLUE, lw=2.5, ms=7)
for xi, yi in zip(x, ms):
    ax.annotate(f"{yi:.0f} ms", (xi, yi), textcoords="offset points", xytext=(0, 9), ha="center",
                fontsize=9, color=NAVY, fontweight="bold")
ax.annotate(f"{nv[0]:,} variables", (x[0], ms[0]), xytext=(30, 40), textcoords="data", fontsize=8.5, color="#444",
            arrowprops=dict(arrowstyle="-", color="#999", lw=0.8))
ax.annotate(f"{nv[-1]:,} variables", (x[-1], ms[-1]), textcoords="offset points", xytext=(-92, -6), fontsize=8.5, color="#444")
ax.text(30, 470, f"{x[-1] // x[0]}x more bases\n-> {ms[-1] / ms[0]:.0f}x solve time", fontsize=11, color=NAVY,
        fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", fc="#EAF2FA", ec="#9DC3E6"))
ax.set_xticks(x); ax.set_xlim(0, 104); ax.set_ylim(0, 700)
ax.set_xlabel("Forward bases"); ax.set_ylabel("Solve time (ms)")
ax.set_title("Solve time grows about linearly with network size", fontsize=12, fontweight="bold", color=NAVY)
ax.grid(axis="y", alpha=.25)
finish(fig, "scalability.png", "One LP solve, 20 scenarios, 6-day horizon, HiGHS, single CPU core. Solve time only.")

# 3. Method comparison: stockouts and cost -------------------------------------------------
order = [("B1", "B1\n(s,S)"), ("B2", "B2\npoint+LP"), ("B4", "B4\nrobust"), ("B3", "B3\nstochastic"), ("RRL", "RRL")]
labels = [l for _, l in order]
so = [A[k]["so_mean"] for k, _ in order]
cost = [A[k]["total"] for k, _ in order]
colors = [GREY, GREY, GREY, GREY, BLUE]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 4.0))
a1.bar(labels, so, color=colors)
for i, v in enumerate(so):
    a1.text(i, v + 0.6, r1(v), ha="center", fontweight="bold", fontsize=10)
a1.set_title("Stockouts per episode (lower is better)", fontsize=11, fontweight="bold", color=NAVY)
a1.set_ylabel("Base-days with unmet demand")
a2.bar(labels, cost, color=colors)
for i, v in enumerate(cost):
    a2.text(i, v + 1.5, f"{v:.0f}", ha="center", fontweight="bold", fontsize=10)
a2.set_title("Simulated cost per episode", fontsize=11, fontweight="bold", color=NAVY)
a2.set_ylabel("Cost units (holding + option fees)")
for a in (a1, a2):
    a.grid(axis="y", alpha=.25); a.tick_params(axis="x", labelsize=8.5)
finish(fig, "method_comparison.png", "50 held-out episodes | identical random events for every method | simulated network")

# 4. Ablations ------------------------------------------------------------------------------
abl = [("RRL\n(full)", "RRL", BLUE), ("no options\n(CVaR only)", "RRL_noOpt", GREY),
       ("no calibration", "RRL_noCal", GREY), ("no edge tree", "RRL_noEdge", GREY)]
vals = [A[k]["so_mean"] for _, k, _ in abl]
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.bar([l for l, _, _ in abl], vals, color=[c for _, _, c in abl])
for i, v in enumerate(vals):
    ax.text(i, v + 0.2, r1(v), ha="center", fontweight="bold", fontsize=11)
ax.set_ylabel("Stockouts per episode")
ax.set_title("Only the options matter in our tests", fontsize=12, fontweight="bold", color=NAVY)
ax.grid(axis="y", alpha=.25)
finish(fig, "ablations.png", "50 held-out episodes | calibration and edge tree: differences within noise (see README table)")

# 5. One base, one closure --------------------------------------------------------------------
tr = ext["trace"]
days = list(range(len(tr["b2"])))
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.plot(days, tr["b2"], color=GREY, lw=2.5, label=f"Conventional plan (B2): {tr['so_b2']} stockout days")
ax.plot(days, tr["rrl"], color=BLUE, lw=2.5, label="RRL: 0 stockout days")
closed = [d for d, o in zip(days, tr["open"]) if o == 0]
if closed:
    ax.axvspan(min(closed) - .5, max(closed) + .5, color=RED, alpha=.08, label="Primary road closed")
ax.axhline(16, color="#555", ls="--", lw=1)
ax.text(len(days) - 1, 17, "survival minimum (16 units)", ha="right", fontsize=8.5, color="#555")
ax.set_xlabel("Day"); ax.set_ylabel("Stock at base (units)")
ax.set_title("Same road closure, two plans", fontsize=13, fontweight="bold", color=NAVY)
ax.legend(frameon=False, fontsize=9, loc="upper right"); ax.grid(axis="y", alpha=.25)
finish(fig, "closure_trace.png", f"Simulated episode seed {tr['seed']}, base {tr['base']} | chosen as the largest gap among 50 test episodes (illustrative, not typical)")

# 6. Scenario count --------------------------------------------------------------------------
sens = res["sens"]
S = sorted(int(k) for k in sens)
so_s = [sens[str(k)]["so"] for k in S]
t_s = [sens[str(k)]["t_med"] for k in S]
fig, ax = plt.subplots(figsize=(6.4, 4.0))
pos = range(len(S))
ax.bar(pos, so_s, color=[BLUE if k == 20 else GREY for k in S])
for i, v in enumerate(so_s):
    ax.text(i, v / 2, f"{v:.2f}".rstrip("0").rstrip("."), ha="center", va="center", fontweight="bold", color="white", fontsize=12)
ax.set_xticks(list(pos)); ax.set_xticklabels([str(k) for k in S])
ax.set_xlabel("Scenarios sampled per re-plan"); ax.set_ylabel("Stockouts per episode")
ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
ax2.plot(list(pos), t_s, color=AMBER, marker="o", lw=2); ax2.set_ylabel("Median solve time (ms)", color=AMBER)
ax.set_title("How many futures to sample?", fontsize=12, fontweight="bold", color=NAVY)
finish(fig, "scenario_count.png", "20 episodes per setting (noisy) | blue bar = scenario count used in the main results")
print("charts written to", OUT)