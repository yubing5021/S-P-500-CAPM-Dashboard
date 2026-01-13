# ============================================================
# S&P 500 SECTOR CAPM — ANALYSIS SCRIPT
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path.home() / "OneDrive" / "Desktop" / "sp500_outputs"
FIG_DIR = BASE_DIR / "figures"

print(f"Looking for data in: {BASE_DIR}")

if not BASE_DIR.exists():
    raise FileNotFoundError(f"BASE_DIR does not exist: {BASE_DIR}")

print("Files found:")
for f in BASE_DIR.iterdir():
    print(" -", f.name)

BASE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Looking for data in: {BASE_DIR}")
print("Files found:")
for f in BASE_DIR.iterdir():
    print(" -", f.name)
# ============================================================
# LOAD DATA
# ============================================================

panel = pd.read_csv(BASE_DIR / "sp500_stock_panel.csv", parse_dates=["Date"])
sector_returns = pd.read_csv(BASE_DIR / "sector_returns.csv", parse_dates=["Date"])
capm_static = pd.read_csv(BASE_DIR / "sector_capm_static.csv")
capm_rolling = pd.read_csv(BASE_DIR / "sector_capm_rolling.csv", parse_dates=["Date"])

assert not panel.empty, "Panel is empty"
assert not sector_returns.empty, "Sector returns empty"
assert not capm_static.empty, "Static CAPM empty"

# ============================================================
# 1. STATIC SECTOR RISK SUMMARY
# ========
summary = (
    sector_returns
    .groupby("Sector")
    .agg(
        Mean_Return=("Sector_Log_Return", "mean"),
        Volatility=("Sector_Log_Return", "std"),
        Observations=("Sector_Log_Return", "count")
    )
    .merge(capm_static, on="Sector")
    .sort_values("Beta", ascending=False)
)

print("\n=== STATIC SECTOR SUMMARY ===")
print(summary)

summary.to_csv(BASE_DIR / "sector_summary.csv")

# ============================================================
# 2. BETA VS RETURN SCATTER (PROFESSIONAL LEGEND + ANNOTATIONS)
# ============================================================

plt.figure(figsize=(11, 7))

ax = sns.scatterplot(
    data=summary,
    x="Beta",
    y="Mean_Return",
    size="Volatility",
    hue="Sector",
    sizes=(80, 600),
    alpha=0.85,
    legend="brief"
)

# Reference line: market beta
ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
ax.text(
    1.01,
    ax.get_ylim()[1] * 0.95,
    "Market Beta = 1",
    fontsize=9,
    verticalalignment="top"
)

# Axis labels & title
ax.set_xlabel("CAPM Beta (Systematic Risk)")
ax.set_ylabel("Average Weekly Log Return")
ax.set_title("Sector Risk–Return Profile (CAPM)")

# Annotate each sector (THIS MUST BE BEFORE savefig)
for _, row in summary.iterrows():
    ax.text(
        row["Beta"] + 0.015,
        row["Mean_Return"],
        row["Sector"],
        fontsize=9,
        alpha=0.9
    )

# Improve legend placement
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles=handles,
    labels=labels,
    title="Sector / Volatility",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=True
)

plt.tight_layout()
plt.savefig(FIG_DIR / "beta_vs_return.png", dpi=300)
plt.close()

# ============================================================
# 3. ROLLING BETA STABILITY (LEGEND AT BOTTOM)
# ============================================================

pivot_beta = capm_rolling.pivot(
    index="Date",
    columns="Sector",
    values="Rolling_Beta"
)

plt.figure(figsize=(13, 7))

ax = pivot_beta.plot(
    linewidth=1.4,
    alpha=0.85
)

# Market beta reference line
ax.axhline(1.0, color="black", linestyle="--", linewidth=1)

# Market beta label OUTSIDE right edge (no axis shrink)
ax.annotate(
    "Market Beta = 1",
    xy=(1.01, 1.0),
    xycoords=("axes fraction", "data"),
    fontsize=10,
    ha="left",
    va="center",
    color="black",
    fontstyle="italic",
    clip_on=False
)

ax.set_title("Rolling Sector Betas (156-week Window)")
ax.set_xlabel("Date")
ax.set_ylabel("CAPM Beta")

# Bottom legend (key change)
ax.legend(
    title="Sector",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=4,               # adjust based on readability
    frameon=False,
    fontsize=9,
    title_fontsize=10
)

plt.tight_layout()
plt.savefig(FIG_DIR / "rolling_betas.png", dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 3B. ROLLING ALPHA STABILITY
# ============================================================

pivot_alpha = capm_rolling.pivot(
    index="Date",
    columns="Sector",
    values="Rolling_Alpha"
)

fig, ax = plt.subplots(figsize=(12, 6))

pivot_alpha.plot(ax=ax, linewidth=2)

# Market alpha reference line
ax.axhline(0.0, color="black", linestyle="--", linewidth=1)

# Market alpha label OUTSIDE right edge (no plot shrink)
ax.annotate(
    "Market Alpha = 0",
    xy=(1.01, 0.0),
    xycoords=("axes fraction", "data"),
    fontsize=10,
    ha="left",
    va="center",
    fontstyle="italic",
    clip_on=False
)

# Axis labels and title
ax.set_title("Rolling Sector Alphas (156-week Window)")
ax.set_ylabel("CAPM Alpha (Weekly Log Return)")
ax.set_xlabel("Date")

# Legend at bottom (same style as beta chart)
ax.legend(
    title="Sector",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.20),
    ncol=4,
    frameon=False
)

plt.tight_layout()
plt.savefig(FIG_DIR / "rolling_alphas.png", dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 4. SECTOR CORRELATION ANALYSIS
# ============================================================

corr = (
    sector_returns
    .pivot(index="Date", columns="Sector", values="Sector_Log_Return")
    .corr()
)

plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Sector Return Correlation Matrix")
plt.tight_layout()
plt.savefig(FIG_DIR / "sector_correlation.png")
plt.close()

# ============================================================
# 5. CONCENTRATION RISK
# ============================================================

concentration = (
    panel
    .groupby(["Date", "Sector"])
    .apply(lambda x: x["Sector_Weight"].max())
    .reset_index(name="Max_Stock_Weight")
)

print("\n=== CONCENTRATION RISK (TOP STOCK WEIGHT BY SECTOR) ===")
print(
    concentration
    .groupby("Sector")["Max_Stock_Weight"]
    .describe()
)

# ============================================================
# DONE
# ============================================================

print("\nAnalysis complete.")
print(f"Outputs written to:\n{BASE_DIR}")
print(f"Figures written to:\n{FIG_DIR}")

import plotly.graph_objects as go

def round_sig(series, sig=4):
    series = series.astype(float)

    def _round(x):
        if x == 0 or np.isnan(x):
            return x
        return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)

    return series.apply(_round)


# Pivot once
pivot_beta = capm_rolling.pivot(
    index="Date",
    columns="Sector",
    values="Rolling_Beta"
)

pivot_alpha = capm_rolling.pivot(
    index="Date",
    columns="Sector",
    values="Rolling_Alpha"
)

sectors = pivot_beta.columns.tolist()
n_sectors = len(sectors)

fig = go.Figure()

# --- Add BETA traces (visible by default)
for sector in sectors:
    fig.add_trace(
        go.Scatter(
            x=pivot_beta.index,
            y=round_sig(pivot_beta[sector], 4),
            mode="lines",
            name=sector,
            visible=True,
            line=dict(width=2)
        )
    )

# --- Add ALPHA traces (hidden initially)
for sector in sectors:
    fig.add_trace(
        go.Scatter(
            x=pivot_alpha.index,
            y=round_sig(pivot_alpha[sector], 4),
            mode="lines",
            name=sector,
            visible=False,
            line=dict(width=2)
        )
    )

# --- Dropdown buttons
fig.update_layout(
    updatemenus=[
        dict(
            type="dropdown",
            x=0.01,
            y=1.15,
            buttons=[
                dict(
                    label="Rolling Beta",
                    method="update",
                    args=[
                        {"visible": [True]*n_sectors + [False]*n_sectors},
                        {
                            "yaxis": dict(
                                title="CAPM Beta",
                                tickformat=".4f",
                                exponentformat="none",
                                showexponent="none"
                            ),
                            "title": "Rolling Sector Betas (156-week Window)",
                            "shapes": [
                                dict(
                                    type="line",
                                    xref="paper",
                                    x0=0,
                                    x1=1,
                                    y0=1,
                                    y1=1,
                                    yref="y",
                                    line=dict(dash="dash", color="black")
                                )
                            ]
                        }
                    ],
                ),
                dict(
                    label="Rolling Alpha",
                    method="update",
                    args=[
                        {"visible": [False]*n_sectors + [True]*n_sectors},
                        {
                            "yaxis": dict(
                                title="CAPM Alpha (Weekly Log Return)",
                                tickformat=".4f",
                                exponentformat="none",
                                showexponent="none"
                            ),
                            "title": "Rolling Sector Alphas (156-week Window)",
                            "shapes": [
                                dict(
                                    type="line",
                                    xref="paper",
                                    x0=0,
                                    x1=1,
                                    y0=0,
                                    y1=0,
                                    yref="y",
                                    line=dict(dash="dash", color="black")
                                )
                            ]
                        }
                    ],
                )
            ]
        )
    ]
)

                # --- Base layout
fig.update_layout(
    template="plotly_white",
    height=650,
    hovermode="x unified",

    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.25,
        xanchor="center",
        x=0.5
    ),

    xaxis_title="Date",

    yaxis=dict(
        title="CAPM Beta",
        tickformat=".4f",
        exponentformat="none",
        showexponent="none"
    ),

    title="Rolling Sector Betas (156-week Window)"
)

# ============================================================
# SAVE INTERACTIVE PLOTLY OUTPUT
# ============================================================

plotly_path = BASE_DIR / "rolling_sector_beta_alpha_interactive.html"
fig.write_html(plotly_path, include_plotlyjs="cdn")

print(f"Interactive Plotly chart saved to:\n{plotly_path}")

