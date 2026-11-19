"""Publication charts for the VoltRelay analysis."""
import os
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch

BG = "#F7F3EA"
INK = "#1E1C18"
BRICK = "#8E2F2F"
SAGE = "#4E7464"
MUTED = "#8A8578"
GRID = "#D8D0C0"
GOLD = "#B08A4F"
SLATE = "#5B6B73"

D = "S:/project_Y/outputs/chart_data"
OUT = "S:/project_Y/outputs/charts"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "font.family": "Segoe UI",
    "font.size": 10,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "legend.frameon": False,
})


def title(fig, main, sub=None):
    fig.text(0.045, 0.975, main, fontfamily="Georgia", fontsize=20, va="top", color=INK)
    if sub:
        fig.text(0.045, 0.925, sub, fontfamily="Segoe UI", fontsize=10.5, va="top", color=MUTED)


def source(fig, text="Synthetic VoltRelay Energy dataset, Jan 2024 - Jun 2025. Analysis by the author."):
    fig.text(0.045, 0.012, text, fontsize=8, color=MUTED, fontfamily="Consolas")


def style(ax, ygrid=True, xgrid=False):
    ax.tick_params(length=0, labelsize=9)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, alpha=0.85)
        ax.set_axisbelow(True)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, alpha=0.85)
        ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def thousands(x, pos):
    return f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"


def millions(x, pos):
    return f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1000:.0f}k"


# ------------------------------------------------------------------ 01 KPI
m = pd.read_csv(f"{D}/kpi_monthly.csv", parse_dates=["month"])
fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4))
fig.subplots_adjust(hspace=0.42, wspace=0.18, top=0.86, bottom=0.10)
title(fig, "Growth is real. So is the failure bill.",
      "Completed swaps nearly tripled; failures grew twice as fast. Margin per swap still improved.")
panels = [
    ("completed", "Completed swaps per month", SAGE, thousands, "303k in Jun 2025"),
    ("revenue", "Net revenue per month (INR)", SAGE, millions, "INR 19.7M in Jun 2025"),
    ("fail_pct", "Failure rate (%), all attempts", BRICK, lambda x, p: f"{x:.0f}%", "8.6% in Jun 2025"),
    ("cm_full_per_swap", "Contribution margin per swap (INR)", INK, lambda x, p: f"{x:.0f}", "INR 20.47 in Jun 2025"),
]
for ax, (col, lab, color, fmt, note) in zip(axes.ravel(), panels):
    ax.plot(m.month, m[col], color=color, linewidth=2.0, zorder=3)
    ax.fill_between(m.month, 0, m[col], color=color, alpha=0.10, zorder=2)
    ax.set_title(lab, loc="left", fontsize=11, color=INK, pad=8)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%y"))
    ax.set_xlim(m.month.min(), m.month.max())
    ax.set_ylim(0, m[col].max() * 1.28)
    style(ax)
    last = m.iloc[-1]
    ax.scatter([last.month], [last[col]], color=color, s=28, zorder=4)
    ax.annotate(note, (last.month, last[col]), xytext=(-6, 12), textcoords="offset points",
                ha="right", fontsize=8.5, color=color, fontfamily="Consolas")
    if col == "fail_pct":
        for year in (2024, 2025):
            ax.axvspan(pd.Timestamp(f"{year}-04-01"), pd.Timestamp(f"{year}-06-30"),
                       color=BRICK, alpha=0.07, zorder=0)
            ax.annotate("heat season", (pd.Timestamp(f"{year}-05-15"), 15.2), fontsize=7.5,
                        color=BRICK, ha="center", alpha=0.9)
    if col == "cm_full_per_swap":
        ax.axvline(pd.Timestamp("2024-07-01"), color=MUTED, linestyle=":", linewidth=1)
        ax.annotate("price +INR 5", (pd.Timestamp("2024-07-05"), 3.0), fontsize=8,
                    color=MUTED, rotation=90, va="bottom")
source(fig)
save(fig, "01_kpi_growth")

# --------------------------------------------------------------- 02 heat curve
hs = pd.read_csv(f"{D}/heat_station.csv")
hd = pd.read_csv(f"{D}/heat_daily.csv")
hd = hd.set_index("max_temp_bin").loc[["<28", "28-32", "32-36", "36-40", "40+"]].reset_index()
fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4), gridspec_kw={"width_ratios": [1.25, 1]})
fig.subplots_adjust(top=0.78, bottom=0.16, wspace=0.24)
title(fig, "Heat is the hidden failure driver.",
      "Failure is flat near 4% for most of the year, then breaks above 36-38 C. The same threshold appears in station telemetry and city weather.")
ax = axes[0]
ax.axvspan(36, 60, color=BRICK, alpha=0.06)
ax.plot(hs.temp_c, hs.fail_pct, color=BRICK, linewidth=2.2, marker="o", markersize=4.2)
for x in (40, 44, 48):
    row = hs[hs.temp_c == x].iloc[0]
    ax.scatter([x], [row.fail_pct], color=BRICK, s=38, zorder=4)
    ax.annotate(f"{row.fail_pct:.1f}%", (x, row.fail_pct), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=9, color=BRICK)
ax.axvline(36, color=INK, linestyle="--", linewidth=1)
ax.annotate("36 C", (36, 14.6), fontsize=9, color=INK, ha="center")
ax.set_xlabel("Cabinet temperature (C), 2-hour bins")
ax.set_ylabel("Failure rate (%)")
ax.set_ylim(0, 16)
ax.set_xlim(10, 52)
ax.set_title("A. 2W failure by cabinet temperature", loc="left", fontsize=11, pad=8)
style(ax)

ax = axes[1]
colors = [MUTED, MUTED, MUTED, BRICK, BRICK]
bars = ax.bar(hd.max_temp_bin, hd.fail_pct, color=colors, width=0.66)
for b, v in zip(bars, hd.fail_pct):
    ax.annotate(f"{v:.1f}%", (b.get_x() + b.get_width()/2, v), xytext=(0, 5),
                textcoords="offset points", ha="center", fontsize=9)
ax.set_ylabel("Failure rate (%)")
ax.set_xlabel("Daily maximum temperature (C)")
ax.set_ylim(0, 18)
ax.set_title("B. Network failure by city-day maximum", loc="left", fontsize=11, pad=8)
style(ax)
ax.annotate("Heat-alert days: 14.9%\nvs 5.5% on normal days", (0.02, 0.97), xycoords="axes fraction",
            va="top", fontsize=9, color=BRICK, fontfamily="Consolas")
source(fig)
save(fig, "02_heat_curve")

# ---------------------------------------------------------- 03 gen mechanism
gh = pd.read_csv(f"{D}/gen_heat.csv")
gc = pd.read_csv(f"{D}/gen_charge.csv")
bins = ["<35", "35-40", "40-45", "45+"]
fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.2))
fig.subplots_adjust(top=0.75, bottom=0.18, wspace=0.28)
title(fig, "Gen1 cabinets throttle when it gets hot - then stock disappears.",
      "At 45 C+, Gen1 charging time rises 77% and charged 2W inventory falls 64%. Gen3 barely moves. That is the failure mechanism.")
for ax, col, lab, fmt in [
    (axes[0], "fail_pct", "A. Failure rate (%)", lambda x, p: f"{x:.0f}%"),
    (axes[1], "charge_min", "B. Average charge time (minutes)", lambda x, p: f"{x:.0f}"),
    (axes[2], "charged_2w", "C. Charged 2W packs available (avg)", lambda x, p: f"{x:.0f}"),
]:
    for gen, color in [("Gen1", BRICK), ("Gen2", SAGE), ("Gen3", INK)]:
        sub = gh if col == "fail_pct" else gc
        d = sub[sub.gen == gen].set_index("temp_bin").loc[bins]
        y = d[col].values
        ax.plot(range(4), y, color=color, linewidth=2.0, marker="o", markersize=5, label=gen)
        ax.annotate(gen, (3, y[-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=9, color=color, va="center")
    ax.set_xticks(range(4), bins)
    ax.set_xlabel("Cabinet temperature (C)")
    ax.set_title(lab, loc="left", fontsize=10.5, pad=8)
    ax.set_xlim(-0.2, 3.55)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    style(ax)
axes[0].annotate("15.3% vs 5.1%", (2.0, 14.2), fontsize=9, color=BRICK)
axes[1].annotate("156 min", (3, 155.7), xytext=(-8, 6), textcoords="offset points",
                 fontsize=9, color=BRICK, ha="right")
axes[1].annotate("44 min", (3, 43.5), xytext=(-8, -14), textcoords="offset points",
                 fontsize=9, color=INK, ha="right")
axes[2].annotate("3.1 packs", (3, 3.1), xytext=(-8, 8), textcoords="offset points",
                 fontsize=9, color=BRICK, ha="right")
axes[2].annotate("9.3 packs", (3, 9.32), xytext=(-8, 8), textcoords="offset points",
                 fontsize=9, color=INK, ha="right")
axes[1].set_ylim(0, 175)
axes[2].set_ylim(0, 12)
source(fig)
save(fig, "03_gen_mechanism")

# --------------------------------------------------------------- 04 city + CX
ch = pd.read_csv(f"{D}/city_heat.csv").sort_values("fail_hot")
fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})
fig.subplots_adjust(top=0.78, bottom=0.17, wspace=0.28)
title(fig, "Geography is heat exposure, not city brand.",
      "Every city fails at a similar rate in cool hours. The gap opens above 40 C - and complaints follow. Sage = below 35 C, brick = 40 C+.")
ax = axes[0]
for i, row in ch.iterrows():
    ax.plot([row.fail_cool, row.fail_hot], [i, i], color=GRID, linewidth=2.4, zorder=1)
    ax.scatter(row.fail_cool, i, color=SAGE, s=52, zorder=3)
    ax.scatter(row.fail_hot, i, color=BRICK, s=52, zorder=3)
    ax.annotate(f"{row.hot_hours_pct:.0f}% hot hrs", (row.fail_hot, i), xytext=(9, -3),
                textcoords="offset points", fontsize=8, color=MUTED, fontfamily="Consolas")
ax.set_yticks(range(len(ch)), ch.city)
ax.set_xlabel("2W failure rate (%)")
ax.set_xlim(2.5, 17.0)
ax.set_title("A. Cool vs hot-hour failure", loc="left", fontsize=11, pad=8)
style(ax, ygrid=False, xgrid=True)

ax = axes[1]
x = ch.fail_hot.values
y = ch.tickets_per_1k.values
ax.scatter(x, y, color=BRICK, s=58, zorder=3)
for city, xi, yi in zip(ch.city, x, y):
    ax.annotate(city, (xi, yi), xytext=(7, 6), textcoords="offset points", fontsize=8.5, color=INK)
coef = np.polyfit(x, y, 1)
xs = np.linspace(x.min(), x.max(), 50)
ax.plot(xs, np.polyval(coef, xs), color=MUTED, linewidth=1.2, linestyle="--")
r = np.corrcoef(x, y)[0, 1]
ax.annotate(f"r = {r:.2f}", (0.03, 0.95), xycoords="axes fraction", fontsize=10,
            color=MUTED, fontfamily="Consolas")
ax.set_xlabel("Hot-hour failure rate (%)")
ax.set_ylabel("Support tickets per 1,000 completed swaps")
ax.set_title("B. Complaints track hot-hour failure", loc="left", fontsize=11, pad=8)
style(ax)
source(fig)
save(fig, "04_city_cx")

# ----------------------------------------------------------------- 05 battery
fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
fig.subplots_adjust(top=0.77, bottom=0.20, wspace=0.30)
title(fig, "One battery cohort is an outlier: Kyron lots KY-2407/08/09.",
      "All 1,385 Kyron 2W packs retired at 60.5% state of health. The same lots in 3W form sit at 69.5%. Every other cohort is near 80-87%.")
ax = axes[0]
labels = ["Amptek 2W", "Cellora 2W", "Kyron 2W\nKY-2407/08/09", "Amptek 3W", "Cellora 3W",
          "Kyron 3W\nKY-2407/08/09", "Kyron 3W\nlater lots"]
vals = [80.0, 80.0, 60.5, 86.5, 86.5, 69.5, 86.5]
colors = [SAGE, SAGE, BRICK, MUTED, MUTED, BRICK, SAGE]
bars = ax.barh(range(len(vals)), vals, color=colors, height=0.62)
for i, (v, b) in enumerate(zip(vals, bars)):
    ax.annotate(f"{v:.1f}%", (v, i), xytext=(5, -3), textcoords="offset points",
                fontsize=9, color=BRICK if colors[i] == BRICK else INK)
ax.axvline(80, color=INK, linestyle=":", linewidth=1)
ax.annotate("peer cohort level", (80, 6.45), xytext=(4, 0), textcoords="offset points",
            fontsize=8, color=MUTED)
ax.set_yticks(range(len(labels)), labels, fontsize=8.5)
ax.set_xlim(0, 100)
ax.set_xlabel("Current state of health (%)")
ax.set_title("A. Current SOH by cohort", loc="left", fontsize=11, pad=8)
style(ax, ygrid=False, xgrid=True)

ax = axes[1]
br = pd.read_csv(f"{D}/battery_range.csv")
order = ["60-70", "70-80", "80-90", "90+"]
for vc, color in [("2W", BRICK), ("3W", SAGE)]:
    d = br[br.vehicle_class == vc].set_index("soh_bin").loc[order]
    ax.plot(range(len(order)), d.avg_km, color=color, linewidth=2.2, marker="o", markersize=6, label=vc)
    ax.annotate(vc, (3, d.avg_km.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                fontsize=9, color=color, va="center")
ax.set_xticks(range(len(order)), order)
ax.set_xlabel("Delivered state of health (%)")
ax.set_ylabel("Average km since last swap")
ax.set_xlim(-0.2, 3.4)
ax.set_ylim(35, 100)
ax.set_title("B. Range falls with battery health", loc="left", fontsize=11, pad=8)
style(ax)
ax.annotate("2W: 65 km at 90%+\nvs 43 km at 60-70%", (0.04, 0.96), xycoords="axes fraction",
            va="top", fontsize=8.5, color=BRICK, fontfamily="Consolas")
source(fig)
save(fig, "05_battery")

# ------------------------------------------------------------ 06 kyron cost
kw = pd.read_csv(f"{D}/kyron_wear.csv", parse_dates=["month"])
cx = pd.read_csv(f"{D}/supplier_cx.csv")
fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), gridspec_kw={"width_ratios": [1.3, 1]})
fig.subplots_adjust(top=0.78, bottom=0.17, wspace=0.26)
title(fig, "The bad cohort is also the expensive one.",
      "Kyron wear cost per swap is 3.2x the peer average, and its riders raise 1.8x more tickets - with 2.3x the low-range complaints.")
ax = axes[0]
ax.fill_between(kw.month, kw.total_wear/1e6, color=GRID, alpha=0.9, label="All suppliers")
ax.fill_between(kw.month, kw.kyron_wear/1e6, color=BRICK, alpha=0.75, label="Kyron")
ax.plot(kw.month, kw.total_wear/1e6, color=INK, linewidth=1.0)
ax.plot(kw.month, kw.kyron_wear/1e6, color=BRICK, linewidth=1.4)
ax.set_ylabel("Battery wear cost (INR million / month)")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%y"))
ax.set_title("A. Monthly wear cost, Kyron vs network", loc="left", fontsize=11, pad=8)
style(ax)
ax.legend(loc="upper left", fontsize=8)
ax.annotate("Kyron enters\nSep 2024", (pd.Timestamp("2024-09-01"), 2.35), fontsize=8,
            color=BRICK, ha="center",
            bbox=dict(facecolor=BG, edgecolor="none", pad=1.5))
ax.annotate("INR 20.2M\ncumulative", (pd.Timestamp("2025-05-20"), 3.55), fontsize=9,
            color=BRICK, ha="center", fontfamily="Consolas",
            bbox=dict(facecolor=BG, edgecolor="none", pad=1.5))

ax = axes[1]
x = np.arange(len(cx))
w = 0.36
ax.axvspan(-0.45, 0.45, color=BRICK, alpha=0.06, zorder=0)
ax.bar(x - w/2, cx.tickets_per_1k, width=w, color=INK, label="All tickets")
ax.bar(x + w/2, cx.low_range_per_1k, width=w, color=BRICK, label="Low-range tickets")
for i, (a, b) in enumerate(zip(cx.tickets_per_1k, cx.low_range_per_1k)):
    ax.annotate(f"{a:.1f}", (i - w/2, a), xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=8.5)
    ax.annotate(f"{b:.1f}", (i + w/2, b), xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=8.5)
ax.set_xticks(x, cx.supplier)
ax.set_ylabel("Tickets per 1,000 completed swaps")
ax.set_ylim(0, 15.5)
ax.set_title("B. Supplier-linked complaints", loc="left", fontsize=11, pad=8)
style(ax)
ax.legend(handles=[Patch(color=INK, label="All tickets"),
                   Patch(color=BRICK, label="Low-range tickets")],
          loc="upper right", fontsize=8)
source(fig)
save(fig, "06_kyron_cost")

# ------------------------------------------------------------- 07 waterfall
labels = ["Jan 2024\nCM / swap", "Revenue\nper swap", "Energy cost\nper swap", "Battery wear\nper swap",
          "Station cost\nper swap", "Jun 2025\nCM / swap"]
deltas = [6.20, 5.05, 4.98, -4.47, 8.71, None]
fig, ax = plt.subplots(figsize=(11.6, 5.4))
fig.subplots_adjust(top=0.78, bottom=0.17)
title(fig, "Scale and energy costs lifted margin. Battery wear ate part of the gain.",
      "Contribution margin per swap rose from INR 6.20 to INR 20.47. A defective battery cohort added INR 4.5 of wear per swap.")
running = 0
for i, (lab, d) in enumerate(zip(labels, deltas)):
    if i == 0:
        ax.bar(i, d, color=INK, width=0.62)
        ax.annotate(f"{d:.2f}", (i, d), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
        running = d
    elif i == len(labels) - 1:
        ax.bar(i, running, color=INK, width=0.62)
        ax.annotate(f"{running:.2f}", (i, running), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    else:
        bottom = running if d >= 0 else running + d
        ax.bar(i, abs(d), bottom=bottom, color=SAGE if d >= 0 else BRICK, width=0.62)
        ax.annotate(f"{d:+.2f}", (i, running + max(d, 0)), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=SAGE if d >= 0 else BRICK)
        ax.plot([i - 0.31, i + 0.31], [running, running], color=MUTED, linewidth=0.8, linestyle=":")
        running += d
ax.set_xticks(range(len(labels)), labels, fontsize=8.5)
ax.set_ylabel("INR per completed swap")
ax.set_ylim(0, 24)
ax.set_title("Change in contribution margin per swap, Jan 2024 to Jun 2025", loc="left",
             fontsize=11, pad=8)
style(ax)
source(fig)
save(fig, "07_margin_waterfall")

# --------------------------------------------------------------- 08 partner
p = pd.read_csv(f"{D}/partner.csv")
p = p.dropna(subset=["partner_name"])
fig, ax = plt.subplots(figsize=(11.4, 6.2))
fig.subplots_adjust(top=0.79, bottom=0.14)
title(fig, "The largest fleet partner is not the most valuable one.",
      "ZipDrop delivers the most swaps but the lowest contribution margin per swap. FeastFly earns 2.5x more total margin on slightly less volume.")
seg_color = {"food_delivery": SAGE, "quick_commerce": GOLD, "ecom_logistics": SLATE,
             "cargo_3w": INK, "bike_taxi": MUTED}
for _, r in p.iterrows():
    if r.partner_name == "ZipDrop":
        c = BRICK
    elif r.partner_name == "FeastFly":
        c = SAGE
    else:
        c = seg_color.get(r.partner_segment, MUTED)
    ax.scatter(r.completed, r.cm_full, s=max(r.total_cm_full / 7500, 70),
               color=c, alpha=0.78, edgecolor=BG, linewidth=1.2, zorder=3)
offsets = {"ZipDrop": (-10, -18), "FeastFly": (-10, 12), "ParcelNest": (8, -6),
           "Swiggle Go": (-10, -18), "QuickCart": (-12, 10), "RideMitra": (8, -12),
           "CargoTuk": (8, 4), "HaulKing": (8, 4), "DabbaXpress": (8, 4),
           "UrbanErrand": (8, -12), "MetroMove": (8, 4), "LastMileHub": (8, -12)}
for _, r in p.iterrows():
    dx, dy = offsets.get(r.partner_name, (8, 4))
    ax.annotate(r.partner_name, (r.completed, r.cm_full), xytext=(dx, dy),
                textcoords="offset points", fontsize=8, color=INK)
ax.axhline(16.53, color=MUTED, linestyle="--", linewidth=1)
ax.annotate("network average INR 16.53", (335000, 17.0), xytext=(0, 5),
            textcoords="offset points", fontsize=8, color=MUTED)
ax.set_xlabel("Completed swaps")
ax.set_ylabel("Contribution margin per swap (INR)")
ax.xaxis.set_major_formatter(FuncFormatter(thousands))
ax.set_xlim(40000, 610000)
ax.set_ylim(4, 23)
ax.set_title("Bubble size = total contribution margin", loc="left", fontsize=11, pad=8)
style(ax)
source(fig)
save(fig, "08_partner_value")

# ------------------------------------------------------------- 09 retention
ex = pd.read_csv(f"{D}/../retention_exposure.csv")
co = pd.read_csv(f"{D}/../retention_cohorts.csv")
co = co[co.cohort_month <= "2025-03"]
fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.1), gridspec_kw={"width_ratios": [1.1, 1]})
fig.subplots_adjust(top=0.78, bottom=0.17, wspace=0.26)
title(fig, "Retention is stable - until failure exposure piles up.",
      "Among new riders still active in week 4, D60 retention falls 5.6 points at >20% early failure exposure. A single bad first attempt does not matter.")
ax = axes[0]
colors = [MUTED, MUTED, MUTED, GOLD, BRICK]
bars = ax.bar(range(len(ex)), ex.d60_pct, color=colors, width=0.62)
for b, v in zip(bars, ex.d60_pct):
    ax.annotate(f"{v:.1f}%", (b.get_x() + b.get_width()/2, v), xytext=(0, 5),
                textcoords="offset points", ha="center", fontsize=9)
ax.set_xticks(range(len(ex)),
              [f"{b}\n(n={n:,})" for b, n in zip(ex.fail_bucket, ex.riders)], fontsize=8)
ax.set_ylim(80, 100)
ax.set_ylabel("D60 retention (%)")
ax.set_xlabel("Failure rate in first 21 days")
ax.set_title("A. Late-month active riders", loc="left", fontsize=11, pad=8)
style(ax)
ax.annotate("-5.6 pp", (4, 92.2), fontsize=10, color=BRICK, ha="center", fontfamily="Consolas")

ax = axes[1]
co["dt"] = pd.to_datetime(co.cohort_month)
ax.plot(co.dt, co.d60_pct, color=SAGE, linewidth=2.0, marker="o", markersize=4)
ax.set_ylim(77, 93)
ax.set_ylabel("D60 retention (%)")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%y"))
ax.set_title("B. Retention by signup cohort", loc="left", fontsize=11, pad=8)
style(ax)
mean = co.d60_pct.mean()
ax.axhline(mean, color=MUTED, linestyle=":", linewidth=1)
ax.annotate(f"mean {mean:.1f}%", (co.dt.iloc[6], mean), xytext=(0, -5),
            textcoords="offset points", fontsize=8, color=MUTED, va="top", ha="center",
            bbox=dict(facecolor=BG, edgecolor="none", pad=1))
ax.annotate("cohorts after Mar 2025\nexcluded (window incomplete)", (0.98, 0.03),
            xycoords="axes fraction", ha="right", fontsize=7.5, color=MUTED)
source(fig)
save(fig, "09_retention")

# --------------------------------------------------------------- 10 support
con = duckdb.connect()
con.execute("SET memory_limit='1.2GB'; SET threads=2; SET temp_directory='S:/project_Y/tmp';")
P = "S:/project_Y/data/clean/events.parquet"
fail_mix = con.execute(f"""
SELECT event_type, count(*) n, round(sum(queue_wait_sec)/3600.0) rider_hours
FROM read_parquet('{P}') WHERE NOT is_test_station AND NOT is_completed GROUP BY 1 ORDER BY 2 DESC
""").fetchdf()
sup = pd.read_csv(f"{D}/support_category.csv")
fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
fig.subplots_adjust(top=0.78, bottom=0.19, wspace=0.36)
title(fig, "Support sees the symptom. Queue time is the real cost.",
      "58% of failures are empty cabinets, but queue abandonment accounts for 90% of the 14,642 rider-hours lost.")
ax = axes[0]
ax.barh(sup.category[::-1], sup.n[::-1], color=[BRICK if c == "no_battery_available" else MUTED
                                                for c in sup.category[::-1]], height=0.62)
for i, (n, u) in enumerate(zip(sup.n[::-1], sup.unresolved_pct[::-1])):
    ax.annotate(f"{n:,}  ({u:.0f}% unresolved)", (n, i), xytext=(5, -3),
                textcoords="offset points", fontsize=8, color=INK)
ax.set_xlabel("Tickets")
ax.set_xlim(0, 14500)
ax.set_title("A. Ticket volume by category", loc="left", fontsize=11, pad=8)
style(ax, ygrid=False, xgrid=True)
ax = axes[1]
lab = ["Empty cabinet", "Queue abandoned", "Rider cancelled", "System error"]
vals = fail_mix.rider_hours.values / 1000
colors = [BRICK, INK, MUTED, SLATE]
bars = ax.bar(lab, vals, color=colors, width=0.6)
for b, v in zip(bars, vals):
    ax.annotate(f"{v:.1f}k h", (b.get_x() + b.get_width()/2, v), xytext=(0, 5),
                textcoords="offset points", ha="center", fontsize=9)
ax.set_ylabel("Rider-hours lost (thousands)")
ax.set_ylim(0, 15.5)
ax.tick_params(axis="x", labelsize=8)
ax.set_title("B. Time lost by failure type", loc="left", fontsize=11, pad=8)
style(ax)
source(fig)
save(fig, "10_support")

print("all charts written to", OUT)
