# ============================================================
# STOCK vs SECTOR vs MARKET — RETURNS, BETAS, ALPHA, DIAGNOSTICS
#   - Total return performance vs Sector and S&P 500 (EW + CW)
#   - Risk-free integration (excess returns)
#   - Static alpha regression (HAC errors)
#   - Cumulative alpha (residual) diagnostic
#   - Rolling alpha t-stats (stability)
#   - Rolling betas (vs market and sector)
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm


# ============================================================
# 1. CONFIG
# ============================================================

BASE_DIR = Path.home() / "OneDrive" / "Desktop" / "S&P 500 Scrapper" / "sp500_outputs"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRADING_WEEKS = 52           # annualize weekly log returns
ROLLING_ALPHA_WINDOW = 52    # 1-year rolling window (weekly)
ROLLING_BETA_WINDOW = 52     # 1-year rolling beta (weekly)
HAC_MAXLAGS = 5

PANEL_FILE = BASE_DIR / "sp500_stock_panel.csv"
SECTOR_FILE = BASE_DIR / "sector_returns.csv"
MARKET_FILE = BASE_DIR / "sp500_market_returns.csv"   # optional


# ============================================================
# 2. LOAD DATA (with validations)
# ============================================================

if not PANEL_FILE.exists():
    raise FileNotFoundError(f"Missing panel file: {PANEL_FILE}")

if not SECTOR_FILE.exists():
    raise FileNotFoundError(f"Missing sector file: {SECTOR_FILE}")

panel = pd.read_csv(PANEL_FILE, parse_dates=["Date"])
sector_returns = pd.read_csv(SECTOR_FILE, parse_dates=["Date"])

if panel.empty:
    raise ValueError("panel is empty")
if sector_returns.empty:
    raise ValueError("sector_returns is empty")

required_panel_cols = {"Date", "Ticker", "Sector", "Log_Return", "Market_Cap", "RF_Log_Return"}
missing = required_panel_cols - set(panel.columns)
if missing:
    raise ValueError(f"sp500_stock_panel.csv missing required columns: {sorted(missing)}")

required_sector_cols = {"Date", "Sector", "Sector_Log_Return"}
missing_s = required_sector_cols - set(sector_returns.columns)
if missing_s:
    raise ValueError(f"sector_returns.csv missing required columns: {sorted(missing_s)}")

# Market returns (EW): load if exists; otherwise build from panel (equal-weight)
if MARKET_FILE.exists():
    market_returns = pd.read_csv(MARKET_FILE, parse_dates=["Date"])
    if market_returns.empty:
        raise ValueError("market_returns is empty")
    if "Market_Log_Return" not in market_returns.columns:
        raise ValueError("sp500_market_returns.csv must contain Market_Log_Return column")
else:
    market_returns = (
        panel.groupby("Date", as_index=False)["Log_Return"]
             .mean()
             .rename(columns={"Log_Return": "Market_Log_Return"})
    )


# ============================================================
# 3. BUILD CAP-WEIGHTED MARKET RETURN (CW)
# ============================================================

cap_weighted_market = (
    panel
    .dropna(subset=["Market_Cap", "Log_Return"])
    .assign(Market_Cap=lambda x: pd.to_numeric(x["Market_Cap"], errors="coerce"))
    .dropna(subset=["Market_Cap"])
    .assign(Weight=lambda x: x["Market_Cap"] / x.groupby("Date")["Market_Cap"].transform("sum"))
    .assign(Weighted_Return=lambda x: x["Weight"] * x["Log_Return"])
    .groupby("Date", as_index=False)["Weighted_Return"]
    .sum()
    .rename(columns={"Weighted_Return": "Market_Log_Return_CW"})
)


# ============================================================
# 4. SAFE INPUT FUNCTIONS
# ============================================================

def select_from_list(options, prompt: str):
    while True:
        try:
            idx = int(input(prompt)) - 1
            if 0 <= idx < len(options):
                return options[idx]
            print("Selection out of range.")
        except ValueError:
            print("Invalid input. Enter a number.")


def select_tickers(available):
    available_set = set(available)
    while True:
        raw = input("Enter tickers (comma-separated): ")
        tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
        if not tickers:
            print("No tickers entered.")
            continue
        invalid = set(tickers) - available_set
        if invalid:
            print(f"Invalid tickers: {', '.join(sorted(invalid))}")
        else:
            return tickers


# ============================================================
# 5. SELECT SECTOR & TICKERS
# ============================================================

sectors = sorted(panel["Sector"].dropna().unique())

print("\nAvailable Sectors:")
for i, s in enumerate(sectors, 1):
    print(f"{i}. {s}")

SELECTED_SECTOR = select_from_list(sectors, "\nSelect sector: ")

sector_panel = panel[panel["Sector"] == SELECTED_SECTOR].copy()
if sector_panel.empty:
    raise ValueError(f"No rows found in panel for sector: {SELECTED_SECTOR}")

available_tickers = (
    sector_panel.groupby("Ticker")["Market_Cap"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

print("\nAvailable tickers (top 30 by avg Market Cap):")
print(", ".join(available_tickers[:30]), "...")

SELECTED_TICKERS = select_tickers(available_tickers)


# ============================================================
# 6. BUILD RETURN PANEL (MASTER MERGE)
# ============================================================

# Stock returns (defensive aggregation)
stock_returns = (
    sector_panel[sector_panel["Ticker"].isin(SELECTED_TICKERS)]
    .groupby(["Date", "Ticker"], as_index=False)["Log_Return"]
    .mean()
)

# Risk-free per ticker-date (same RF repeated, but keep schema stable)
rf_panel = (
    panel[["Date", "Ticker", "RF_Log_Return"]]
    .rename(columns={"RF_Log_Return": "Risk_Free_Return"})
)

# Sector benchmark
sector_benchmark = (
    sector_returns[sector_returns["Sector"] == SELECTED_SECTOR][["Date", "Sector_Log_Return"]]
)

# Market benchmarks
market_benchmark_ew = market_returns[["Date", "Market_Log_Return"]]
market_benchmark_cw = cap_weighted_market[["Date", "Market_Log_Return_CW"]]

# Merge everything; inner join for clean alignment across all series
comparison = (
    stock_returns
    .merge(sector_benchmark, on="Date", how="inner")
    .merge(market_benchmark_ew, on="Date", how="inner")
    .merge(market_benchmark_cw, on="Date", how="inner")
    .merge(rf_panel, on=["Date", "Ticker"], how="inner")
    .dropna()
)

if comparison.empty:
    raise ValueError("Merged comparison panel is empty. Check overlap of dates across series.")


# ============================================================
# 7. EXCESS RETURNS (CAPM-CORRECT)
# ============================================================

comparison["Stock_Excess_Return"] = comparison["Log_Return"] - comparison["Risk_Free_Return"]
comparison["Market_Excess_Return_EW"] = comparison["Market_Log_Return"] - comparison["Risk_Free_Return"]
comparison["Market_Excess_Return_CW"] = comparison["Market_Log_Return_CW"] - comparison["Risk_Free_Return"]
comparison["Sector_Excess_Return"] = comparison["Sector_Log_Return"] - comparison["Risk_Free_Return"]


# ============================================================
# 8. PERFORMANCE PLOTS (TOTAL RETURN, INDEXED TO 100)
# ============================================================

def indexed_from_log_returns(log_ret_series: pd.Series) -> pd.Series:
    return np.exp(log_ret_series.cumsum()) * 100.0


# 8A) Stocks vs Sector
fig_sector = go.Figure()
for ticker in SELECTED_TICKERS:
    df = comparison[comparison["Ticker"] == ticker].sort_values("Date")
    fig_sector.add_trace(go.Scatter(
        x=df["Date"],
        y=indexed_from_log_returns(df["Log_Return"]),
        mode="lines",
        name=ticker
    ))

sector_sorted = sector_benchmark.sort_values("Date")
fig_sector.add_trace(go.Scatter(
    x=sector_sorted["Date"],
    y=indexed_from_log_returns(sector_sorted["Sector_Log_Return"]),
    mode="lines",
    name=f"{SELECTED_SECTOR} Sector",
    line=dict(width=4, dash="dash")
))

fig_sector.update_layout(
    title=f"{SELECTED_SECTOR}: Stock Performance vs Sector (Total Return, Indexed)",
    yaxis_title="Indexed Performance (100 = start)",
    xaxis_title="Date",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5)
)
fig_sector.show()


# 8B) Stocks vs S&P 500 (EW and CW)
fig_mkt = go.Figure()
for ticker in SELECTED_TICKERS:
    df = comparison[comparison["Ticker"] == ticker].sort_values("Date")
    fig_mkt.add_trace(go.Scatter(
        x=df["Date"],
        y=indexed_from_log_returns(df["Log_Return"]),
        mode="lines",
        name=ticker
    ))

mkt_ew = comparison.sort_values("Date").groupby("Date")["Market_Log_Return"].first()
mkt_cw = comparison.sort_values("Date").groupby("Date")["Market_Log_Return_CW"].first()

fig_mkt.add_trace(go.Scatter(
    x=mkt_ew.index, y=indexed_from_log_returns(mkt_ew).values,
    mode="lines", name="S&P 500 (EW)",
    line=dict(width=4, dash="dash")
))
fig_mkt.add_trace(go.Scatter(
    x=mkt_cw.index, y=indexed_from_log_returns(mkt_cw).values,
    mode="lines", name="S&P 500 (CW)",
    line=dict(width=4, dash="dot")
))

fig_mkt.update_layout(
    title="Stock Performance vs S&P 500 (Total Return, Indexed)",
    yaxis_title="Indexed Performance (100 = start)",
    xaxis_title="Date",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5)
)
fig_mkt.show()


# ============================================================
# 9. STATIC ALPHA REGRESSION (EXCESS RETURNS)
#    Two runs: EW market and CW market
# ============================================================

def fit_alpha_model(df: pd.DataFrame, market_excess_col: str):
    y = df["Stock_Excess_Return"]
    X = sm.add_constant(df[[market_excess_col, "Sector_Excess_Return"]])
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_MAXLAGS})


alpha_rows = []
for ticker in SELECTED_TICKERS:
    df = comparison[comparison["Ticker"] == ticker].sort_values("Date").dropna()
    if df.empty:
        continue

    model_ew = fit_alpha_model(df, "Market_Excess_Return_EW")
    model_cw = fit_alpha_model(df, "Market_Excess_Return_CW")

    alpha_rows.append({
        "Ticker": ticker,

        "Alpha_EW_Annualized": float(model_ew.params["const"] * TRADING_WEEKS),
        "Alpha_EW_tstat": float(model_ew.tvalues["const"]),
        "Beta_Mkt_EW": float(model_ew.params["Market_Excess_Return_EW"]),
        "Beta_Sector_EW": float(model_ew.params["Sector_Excess_Return"]),
        "R2_EW": float(model_ew.rsquared),

        "Alpha_CW_Annualized": float(model_cw.params["const"] * TRADING_WEEKS),
        "Alpha_CW_tstat": float(model_cw.tvalues["const"]),
        "Beta_Mkt_CW": float(model_cw.params["Market_Excess_Return_CW"]),
        "Beta_Sector_CW": float(model_cw.params["Sector_Excess_Return"]),
        "R2_CW": float(model_cw.rsquared),
    })

alpha_df = pd.DataFrame(alpha_rows).sort_values("Alpha_CW_Annualized", ascending=False)

print("\n=== STATIC ALPHA REGRESSION (EXCESS RETURNS): EW vs CW MARKET ===")
if alpha_df.empty:
    print("No regression results (insufficient data after merges).")
else:
    print(alpha_df.round(4))


# ============================================================
# 10. CUMULATIVE ALPHA DIAGNOSTIC (RESIDUAL DRIFT)
#     Uses CW market by default
# ============================================================

fig_alpha = go.Figure()

for ticker in SELECTED_TICKERS:
    df = comparison[comparison["Ticker"] == ticker].sort_values("Date").dropna()
    if df.empty:
        continue

    model = fit_alpha_model(df, "Market_Excess_Return_CW")
    resid = model.resid

    cum_alpha = np.exp(pd.Series(resid).cumsum()) * 100.0

    fig_alpha.add_trace(go.Scatter(
        x=df["Date"],
        y=cum_alpha,
        mode="lines",
        name=ticker
    ))

fig_alpha.update_layout(
    title="Cumulative Alpha (Regression Residuals, CW Market + Sector, Indexed to 100)",
    yaxis_title="Alpha Index (100 = start)",
    xaxis_title="Date",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5)
)
fig_alpha.show()


# ============================================================
# 11. ROLLING ALPHA t-STATS (STABILITY)
#     Uses CW market by default
# ============================================================

rolling_alpha_rows = []

for ticker in SELECTED_TICKERS:
    df = (
        comparison[comparison["Ticker"] == ticker]
        .sort_values("Date")
        .reset_index(drop=True)
        .dropna()
    )

    if len(df) < ROLLING_ALPHA_WINDOW:
        print(f"Skipping rolling alpha for {ticker}: not enough data ({len(df)} rows).")
        continue

    for i in range(ROLLING_ALPHA_WINDOW, len(df) + 1):
        w = df.iloc[i - ROLLING_ALPHA_WINDOW:i]

        y = w["Stock_Excess_Return"]
        X = sm.add_constant(w[["Market_Excess_Return_CW", "Sector_Excess_Return"]])

        m = sm.OLS(y, X).fit()  # rolling HAC is slower; this is a stability diagnostic
        rolling_alpha_rows.append({
            "Date": w["Date"].iloc[-1],
            "Ticker": ticker,
            "Alpha_tstat": float(m.tvalues["const"]),
            "Alpha_annualized": float(m.params["const"] * TRADING_WEEKS),
        })

rolling_alpha_df = pd.DataFrame(rolling_alpha_rows)

fig_tstat = go.Figure()
for ticker in SELECTED_TICKERS:
    dft = rolling_alpha_df[rolling_alpha_df["Ticker"] == ticker]
    if dft.empty:
        continue
    fig_tstat.add_trace(go.Scatter(
        x=dft["Date"],
        y=dft["Alpha_tstat"],
        mode="lines",
        name=ticker
    ))

fig_tstat.add_hline(y=2, line=dict(dash="dash", color="gray"))
fig_tstat.add_hline(y=-2, line=dict(dash="dash", color="gray"))

fig_tstat.update_layout(
    title=f"Rolling Alpha t-Statistics ({ROLLING_ALPHA_WINDOW}-Week Window, CW Market + Sector)",
    yaxis_title="Alpha t-stat",
    xaxis_title="Date",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5)
)
fig_tstat.show()


# ============================================================
# 12. ROLLING BETAS (TOTAL RETURN SPACE)
# ============================================================

beta_rows = []

for ticker in SELECTED_TICKERS:
    df = comparison[comparison["Ticker"] == ticker].sort_values("Date").dropna()
    if len(df) < ROLLING_BETA_WINDOW:
        continue

    beta_mkt_ew = (
        df["Log_Return"].rolling(ROLLING_BETA_WINDOW).cov(df["Market_Log_Return"])
        / df["Market_Log_Return"].rolling(ROLLING_BETA_WINDOW).var()
    )

    beta_mkt_cw = (
        df["Log_Return"].rolling(ROLLING_BETA_WINDOW).cov(df["Market_Log_Return_CW"])
        / df["Market_Log_Return_CW"].rolling(ROLLING_BETA_WINDOW).var()
    )

    beta_sec = (
        df["Log_Return"].rolling(ROLLING_BETA_WINDOW).cov(df["Sector_Log_Return"])
        / df["Sector_Log_Return"].rolling(ROLLING_BETA_WINDOW).var()
    )

    beta_rows.append(pd.DataFrame({
        "Date": df["Date"].values,
        "Ticker": ticker,
        "Beta_vs_Market_EW": beta_mkt_ew.values,
        "Beta_vs_Market_CW": beta_mkt_cw.values,
        "Beta_vs_Sector": beta_sec.values,
    }))

if beta_rows:
    beta_df = pd.concat(beta_rows, ignore_index=True).dropna()
else:
    beta_df = pd.DataFrame(columns=["Date", "Ticker", "Beta_vs_Market_EW", "Beta_vs_Market_CW", "Beta_vs_Sector"])

# Plot rolling betas vs CW market (standard)
fig_beta = go.Figure()
for ticker in SELECTED_TICKERS:
    d = beta_df[beta_df["Ticker"] == ticker]
    if d.empty:
        continue
    fig_beta.add_trace(go.Scatter(
        x=d["Date"], y=d["Beta_vs_Market_CW"],
        mode="lines", name=f"{ticker} vs Market (CW)"
    ))

fig_beta.update_layout(
    title=f"{SELECTED_SECTOR}: Rolling {ROLLING_BETA_WINDOW}-Week Beta vs Market (CW)",
    yaxis_title="Beta",
    xaxis_title="Date",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5)
)
fig_beta.show()
