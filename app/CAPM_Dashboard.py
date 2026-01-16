"""
CAPM Dashboard.py

## Live App

The dashboard is deployed on Streamlit Cloud:

👉 https://sp500-capm-dashboard.streamlit.app

## Architecture

- `app/` – Streamlit dashboard application
- `pipeline/` – Data ingestion & preprocessing scripts
- `data/` – Lightweight CSV inputs
- Large datasets are hosted externally (Dropbox) and loaded securely via Streamlit secrets

Streamlit dashboard for S&P 500 analysis using weekly log returns.

# Key features
- Sector multi-select
- Ticker multi-select labeled as: TICKER (Company Name)
- Cumulative $1 growth chart (Market + selected sectors + selected tickers)
- Rolling beta/alpha vs market + vs sector benchmark
- Discount Rate:
    - Summary column: Discount_Rate_Annual_(log)
    - Rolling chart: Rolling Discount Rate (annualized)
- CAPM summary regression diagnostics:
    - Alpha, Beta, t-stats
    - R² and Adjusted R²
- Exports (CSVs) keep FULL precision by default (recommended)

Presentation controls (new)
- Precision control: N significant figures for DISPLAY (tables + charts)
- Percent-format toggle:
    - Display annualized rates and alphas as % (tables + charts)
    - (Cumulative $1 growth remains a level index)

Run (PowerShell):
  cd "$env:USERPROFILE\OneDrive\Desktop\S&P 500 Scrapper"
  streamlit run "CAPM Dashboard.py"
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# 1) PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CAPM Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",  # prevents first-load reflow
)
st.title("S&P 500 CAPM Dashboard")

st.markdown(
    """
    This dashboard implements a weekly **log-return** CAPM workflow. It encompasses the market (S&P 500), a weighted-capped industry sector, and market constituent data from 2014 to current.
    Using excess returns, rolling betas, alpha t-statistics, and Sharpe ratios, it evaluates whether observed performance reflects consistent abnormal returns in correlation to the risk taken.

    """
)

# ============================================================
# 2) PATHS (repo-friendly)
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]   # repo root (since file is in /app)
DATA_DIR = BASE_DIR / "data"

SECTOR_RETURNS_PATH = DATA_DIR / "sector_returns.csv"

if not SECTOR_RETURNS_PATH.exists():
    st.error(f"Missing file: {SECTOR_RETURNS_PATH}")
    st.stop()


# ============================================================
# 3) LOAD DATA
# ============================================================
@st.cache_data(show_spinner=False)
def load_panel_from_url(url: str) -> pd.DataFrame:
    return pd.read_csv(url, parse_dates=["Date"])

@st.cache_data(show_spinner=False)
def load_data(sector_returns_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    panel_url = st.secrets["PANEL_URL"]
    panel_df = load_panel_from_url(panel_url)

    sector_df = pd.read_csv(sector_returns_path, parse_dates=["Date"])
    return panel_df, sector_df

try:
    panel, sector_returns = load_data(SECTOR_RETURNS_PATH)
except Exception as e:
    st.error("Failed to load data.")
    st.exception(e)
    st.stop()

panel = panel.sort_values(["Date", "Ticker"]).reset_index(drop=True)
sector_returns = sector_returns.sort_values(["Date", "Sector"]).reset_index(drop=True)

has_company_name = "Company_Name" in panel.columns


# ============================================================
# 4) SCHEMA VALIDATION
# ============================================================
required_panel_cols = {"Date", "Ticker", "Sector", "Log_Return", "Market_Log_Return", "RF_Log_Return"}
missing_panel = required_panel_cols - set(panel.columns)
if missing_panel:
    st.error(f"sp500_stock_panel.csv missing required columns: {sorted(missing_panel)}")
    st.stop()

required_sector_cols = {"Date", "Sector", "Sector_Log_Return"}
missing_sector = required_sector_cols - set(sector_returns.columns)
if missing_sector:
    st.error(f"sector_returns.csv missing required columns: {sorted(missing_sector)}")
    st.stop()


# ============================================================
# 5) HELPERS
# ============================================================
TRADING_WEEKS = 52


def winsorize(series: pd.Series, p: float = 0.01) -> pd.Series:
    """Optional outlier dampener."""
    s = series.dropna()
    if s.empty:
        return series
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return series.clip(lo, hi)


def cumulative_from_log_returns(log_ret: pd.Series) -> pd.Series:
    """$1 growth index from log returns."""
    return np.exp(log_ret.cumsum())


def annualize_log_mean(mean_weekly_log: float) -> float:
    """Annualized mean log return approximation."""
    return float(mean_weekly_log) * TRADING_WEEKS


def annualize_vol(std_weekly_log: float) -> float:
    """Annualized vol from weekly vol."""
    return float(std_weekly_log) * np.sqrt(TRADING_WEEKS)


def safe_first_nonnull(series: pd.Series) -> Optional[str]:
    s = series.dropna()
    if s.empty:
        return None
    v = str(s.iloc[0]).strip()
    return v if v else None


def df_to_csv_bytes(df: pd.DataFrame, index: bool = True) -> bytes:
    return df.to_csv(index=index).encode("utf-8")


def capm_ols_metrics(excess_y: pd.Series, excess_x: pd.Series) -> dict:
    """
    Classic OLS: y = alpha + beta*x + eps

    Returns:
      - alpha, beta
      - alpha_t, beta_t (classic OLS t-stats)
      - r2, adj_r2
      - n
    """
    df = pd.DataFrame({"y": excess_y, "x": excess_x}).dropna()
    n = int(len(df))
    if n < 30:
        return {
            "alpha": np.nan, "beta": np.nan,
            "alpha_t": np.nan, "beta_t": np.nan,
            "r2": np.nan, "adj_r2": np.nan,
            "n": n
        }

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    x_bar = float(x.mean())
    y_bar = float(y.mean())

    sxx = float(np.sum((x - x_bar) ** 2))
    if sxx == 0 or np.isnan(sxx):
        return {
            "alpha": np.nan, "beta": np.nan,
            "alpha_t": np.nan, "beta_t": np.nan,
            "r2": np.nan, "adj_r2": np.nan,
            "n": n
        }

    sxy = float(np.sum((x - x_bar) * (y - y_bar)))
    beta = sxy / sxx
    alpha = y_bar - beta * x_bar

    y_hat = alpha + beta * x
    resid = y - y_hat

    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - y_bar) ** 2))

    r2 = 1.0 - (rss / tss) if tss > 0 else np.nan

    # Adjusted R^2 (k=1 predictor)
    k = 1
    if (n - k - 1) > 0 and np.isfinite(r2):
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)
    else:
        adj_r2 = np.nan

    # Classic SEs and t-stats
    dof_err = n - 2
    if dof_err <= 0:
        alpha_t = np.nan
        beta_t = np.nan
    else:
        sigma2 = rss / dof_err
        beta_se = math.sqrt(sigma2 / sxx)
        alpha_se = math.sqrt(sigma2 * (1.0 / n + (x_bar ** 2) / sxx))
        alpha_t = alpha / alpha_se if alpha_se > 0 else np.nan
        beta_t = beta / beta_se if beta_se > 0 else np.nan

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "alpha_t": float(alpha_t) if np.isfinite(alpha_t) else np.nan,
        "beta_t": float(beta_t) if np.isfinite(beta_t) else np.nan,
        "r2": float(r2) if np.isfinite(r2) else np.nan,
        "adj_r2": float(adj_r2) if np.isfinite(adj_r2) else np.nan,
        "n": n,
    }


def rolling_alpha_beta(excess_y: pd.Series, excess_x: pd.Series, window: int) -> pd.DataFrame:
    """
    Rolling CAPM: y = alpha + beta*x + eps
    Returns DataFrame indexed by Date with columns: beta, alpha.
    """
    df = pd.DataFrame({"y": excess_y, "x": excess_x}).dropna()
    if len(df) < window + 5:
        return pd.DataFrame(columns=["beta", "alpha"])

    y = df["y"].to_numpy()
    x = df["x"].to_numpy()
    idx = df.index

    betas, alphas, dates = [], [], []
    for i in range(window, len(df) + 1):
        y_w = y[i - window:i]
        x_w = x[i - window:i]
        var_x = np.var(x_w)
        if var_x == 0 or np.isnan(var_x):
            continue

        beta = np.cov(y_w, x_w, ddof=0)[0, 1] / var_x
        alpha = float(np.mean(y_w) - beta * np.mean(x_w))

        betas.append(float(beta))
        alphas.append(alpha)
        dates.append(idx[i - 1])

    return pd.DataFrame({"beta": betas, "alpha": alphas}, index=pd.Index(dates, name="Date"))


def rolling_capm_tstats(excess_y: pd.Series, excess_x: pd.Series, window: int) -> pd.DataFrame:
    """
    Rolling CAPM with classic OLS t-stats:
      y = alpha + beta*x + eps

    Returns DataFrame indexed by Date with:
      alpha, beta, alpha_t, beta_t
    """
    df = pd.DataFrame({"y": excess_y, "x": excess_x}).dropna()
    if len(df) < window + 5:
        return pd.DataFrame(columns=["alpha", "beta", "alpha_t", "beta_t"])

    y = df["y"].to_numpy()
    x = df["x"].to_numpy()
    idx = df.index.to_numpy()

    alphas, betas, alpha_ts, beta_ts, dates = [], [], [], [], []

    for i in range(window, len(df) + 1):
        y_w = y[i - window:i]
        x_w = x[i - window:i]

        n = window
        x_bar = float(np.mean(x_w))
        y_bar = float(np.mean(y_w))

        sxx = float(np.sum((x_w - x_bar) ** 2))
        if sxx <= 0 or np.isnan(sxx):
            continue

        sxy = float(np.sum((x_w - x_bar) * (y_w - y_bar)))
        beta = sxy / sxx
        alpha = y_bar - beta * x_bar

        y_hat = alpha + beta * x_w
        resid = y_w - y_hat

        rss = float(np.sum(resid ** 2))
        dof = n - 2
        if dof <= 0:
            continue

        sigma2 = rss / dof
        beta_se = math.sqrt(sigma2 / sxx)
        alpha_se = math.sqrt(sigma2 * (1.0 / n + (x_bar ** 2) / sxx))

        alpha_t = alpha / alpha_se if alpha_se > 0 else np.nan
        beta_t = beta / beta_se if beta_se > 0 else np.nan

        alphas.append(float(alpha))
        betas.append(float(beta))
        alpha_ts.append(float(alpha_t) if np.isfinite(alpha_t) else np.nan)
        beta_ts.append(float(beta_t) if np.isfinite(beta_t) else np.nan)
        dates.append(idx[i - 1])

    return pd.DataFrame(
        {"alpha": alphas, "beta": betas, "alpha_t": alpha_ts, "beta_t": beta_ts},
        index=pd.Index(dates, name="Date"),
    )


# ----------------------------
#5B) Presentation formatting
# ----------------------------
def sig_str(x: object, sig: int) -> str:
    """N significant figures as a string; blank for NaN/None."""
    try:
        if x is None:
            return ""
        if isinstance(x, (float, np.floating)) and (np.isnan(x) or np.isinf(x)):
            return ""
        return f"{float(x):.{sig}g}"
    except Exception:
        return str(x)


def sig_pct_str(x: object, sig: int) -> str:
    """N significant figures as percent string; blank for NaN/None."""
    try:
        if x is None:
            return ""
        if isinstance(x, (float, np.floating)) and (np.isnan(x) or np.isinf(x)):
            return ""
        return f"{float(x) * 100:.{sig}g}%"
    except Exception:
        return str(x)


def apply_axis_and_hover_format(fig, sig: int, y_is_percent: bool, y_label: str) -> None:
    """
    Apply consistent tick formatting and hover formatting for a single-y line/scatter figure.
    For percent, we plot scaled values (x100) and add a % suffix.
    """
    fig.update_yaxes(title=y_label, tickformat=f".{sig}g", ticksuffix="%" if y_is_percent else None)
    fig.update_xaxes(tickformat=None)

    if y_is_percent:
        fig.update_traces(hovertemplate=f"Date=%{{x}}<br>{y_label}=%{{y:.{sig}g}}%<extra></extra>")
    else:
        fig.update_traces(hovertemplate=f"Date=%{{x}}<br>{y_label}=%{{y:.{sig}g}}<extra></extra>")
# ============================================================
# 6A) SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("Controls")

# Precision + percent toggles (new)
st.sidebar.subheader("Display Formatting")
DISPLAY_SIG_FIGS = st.sidebar.slider(
    "Significant figures (display)", min_value=2, max_value=6, value=3, step=1
)
DISPLAY_PCT = st.sidebar.checkbox("Display rates as percent (%)", value=True)
FORMAT_EXPORTS = st.sidebar.checkbox(
    "Format CSV exports (rounding) — not recommended", value=False
)

st.sidebar.divider()

# ----------------------------
# 6B) Sectors (default to 1 sector, state-safe)
# ----------------------------
all_sectors = sorted(panel["Sector"].dropna().unique().tolist())

if "selected_sectors" not in st.session_state:
    st.session_state.selected_sectors = all_sectors[:1]  # initial default only

selected_sectors = st.sidebar.multiselect(
    "Sectors (multi-select)",
    options=all_sectors,
    key="selected_sectors",
)

# Do NOT auto-add a sector after the user clears the selection
if not selected_sectors:
    st.info("Select at least one sector.")
    st.stop()

# ----------------------------
# 6C) Ticker sector + name maps (for labeling and sector benchmark logic)
# ----------------------------
ticker_sector_map: Dict[str, str] = (
    panel.groupby("Ticker")["Sector"]
    .agg(lambda s: s.value_counts().index[0] if not s.empty else "Unknown")
    .to_dict()
)

ticker_name_map: Dict[str, Optional[str]] = {}
if has_company_name:
    ticker_name_map = (
        panel.groupby("Ticker")["Company_Name"]
        .agg(safe_first_nonnull)
        .to_dict()
    )

def ticker_label(t: str) -> str:
    """UI label: TICKER (Company Name) if available."""
    nm = ticker_name_map.get(t)
    return f"{t} ({nm})" if nm else t

# ----------------------------
# 6D) Tickers (state-safe: store tickers, not labels)
# ----------------------------
tickers_pool = sorted(
    panel.loc[panel["Sector"].isin(selected_sectors), "Ticker"].dropna().unique().tolist()
)

if not tickers_pool:
    st.error("No tickers available for the selected sector(s). Check your data filters.")
    st.stop()

# Initialize / sanitize selected tickers (stable, because values are tickers)
if "selected_tickers" not in st.session_state:
    st.session_state.selected_tickers = tickers_pool[:1]
else:
    st.session_state.selected_tickers = [t for t in st.session_state.selected_tickers if t in tickers_pool]
    if not st.session_state.selected_tickers:
        st.session_state.selected_tickers = tickers_pool[:1]

selected_tickers = st.sidebar.multiselect(
    "Tickers (TICKER (Company Name))",
    options=tickers_pool,                 # underlying values are tickers
    format_func=ticker_label,             # displayed labels
    key="selected_tickers",
)

if not selected_tickers:
    st.info("Select at least one ticker.")
    st.stop()

# ----------------------------
# 6E) Estimation horizon + winsorization
# ----------------------------
st.sidebar.subheader("Estimation Horizon")
horizon_weeks = st.sidebar.radio(
    "Select horizon (weeks)",
    options=[52, 156],
    index=1,
    help="52w = more responsive but noisier; 156w (~3y) = more stable, cycle-aware estimates.",
)

apply_winsor = st.sidebar.checkbox("Winsorize weekly returns (1%)", value=False)

st.sidebar.divider()
exports_enabled = st.sidebar.checkbox("Enable downloads", value=True)

# ============================================================
# 7) BUILD ALIGNED SERIES (IMPORTANT FIX: groupby mean)
# ============================================================
mkt_rf = (
    panel.groupby("Date", as_index=True)[["Market_Log_Return", "RF_Log_Return"]]
    .mean()
    .sort_index()
)

mkt_rf["RF_Log_Return"] = mkt_rf["RF_Log_Return"].ffill().bfill()
mkt_rf["Market_Log_Return"] = mkt_rf["Market_Log_Return"].ffill().bfill()
mkt_rf = mkt_rf.dropna(subset=["Market_Log_Return", "RF_Log_Return"])

sectors_wide = (
    sector_returns.loc[sector_returns["Sector"].isin(selected_sectors), ["Date", "Sector", "Sector_Log_Return"]]
    .pivot_table(index="Date", columns="Sector", values="Sector_Log_Return", aggfunc="mean")
    .sort_index()
)

stocks_wide = (
    panel.loc[panel["Ticker"].isin(selected_tickers), ["Date", "Ticker", "Log_Return"]]
    .pivot_table(index="Date", columns="Ticker", values="Log_Return", aggfunc="mean")
    .sort_index()
)

common_dates = mkt_rf.index.intersection(sectors_wide.index).intersection(stocks_wide.index)
mkt_rf = mkt_rf.loc[common_dates]
sectors_wide = sectors_wide.loc[common_dates]
stocks_wide = stocks_wide.loc[common_dates]

if common_dates.empty:
    st.error("No overlapping dates across market series, selected sectors, and selected tickers.")
    st.stop()

if apply_winsor:
    mkt_rf["Market_Log_Return"] = winsorize(mkt_rf["Market_Log_Return"])
    mkt_rf["RF_Log_Return"] = winsorize(mkt_rf["RF_Log_Return"])
    for c in sectors_wide.columns:
        sectors_wide[c] = winsorize(sectors_wide[c])
    for c in stocks_wide.columns:
        stocks_wide[c] = winsorize(stocks_wide[c])

excess_market = mkt_rf["Market_Log_Return"] - mkt_rf["RF_Log_Return"]
excess_stocks = stocks_wide.sub(mkt_rf["RF_Log_Return"], axis=0)
excess_sectors = sectors_wide.sub(mkt_rf["RF_Log_Return"], axis=0)

selected_ticker_sector = {t: ticker_sector_map.get(t, "Unknown") for t in selected_tickers}

aligned_panel = pd.concat(
    [
        mkt_rf.rename(columns={"Market_Log_Return": "Market", "RF_Log_Return": "RF"}),
        sectors_wide.add_prefix("Sector: "),
        stocks_wide,
    ],
    axis=1,
)

# ============================================================
# 8) DISCOUNT RATE SETTINGS
# ============================================================
mrp_hist_annual_log = annualize_log_mean(excess_market.mean())

st.sidebar.subheader("Discount Rate")
use_custom_mrp = st.sidebar.checkbox("Use custom Market Risk Premium (annual)", value=False)
custom_mrp = st.sidebar.slider(
    "Market Risk Premium (annual, log approx)",
    min_value=-0.10,
    max_value=0.20,
    value=float(mrp_hist_annual_log),
    step=0.005,
    disabled=not use_custom_mrp,
)
mrp_annual_log = float(custom_mrp) if use_custom_mrp else float(mrp_hist_annual_log)

st.sidebar.caption(
    "Note: Custom MRP impacts discount-rate calculations (pricing assumption). "
    "It does not change realized returns, volatility, or realized Sharpe."
)
# ============================================================
# 9) CUMULATIVE PERFORMANCE ($1 growth)
# ============================================================
st.subheader("Cumulative Performance ($1 growth)")

st.markdown(
    """ 
 How much is $1 invested in the market, sectors, and selected companies in 2014 worth now?
"""    
)

# --- Build cumulative $1 growth series from weekly log returns
cum = pd.DataFrame(index=common_dates)

cum["Market (S&P 500)"] = cumulative_from_log_returns(mkt_rf["Market_Log_Return"])

for sec in sectors_wide.columns:
    cum[f"Sector: {sec}"] = cumulative_from_log_returns(sectors_wide[sec])

for t in stocks_wide.columns:
    cum[ticker_label(t)] = cumulative_from_log_returns(stocks_wide[t])

# --- Long format for Plotly
cum_long = (
    cum
    .reset_index()
    .melt(id_vars="Date", var_name="Series", value_name="Growth")
)

# --- Plot
fig_cum = px.line(
    cum_long,
    x="Date",
    y="Growth",
    color="Series",
    title=None,
)

# --- Enforce stable initial render (prevents blank chart on load)
fig_cum.update_layout(
    autosize=True,
    height=600,
    margin=dict(l=40, r=40, t=60, b=40),
)

fig_cum.update_yaxes(tickformat=f".{DISPLAY_SIG_FIGS}g")
fig_cum.update_traces(
    hovertemplate=f"Date=%{{x}}<br>Growth=%{{y:.{DISPLAY_SIG_FIGS}g}}<extra></extra>"
)

st.plotly_chart(
    fig_cum,
    use_container_width=True,
    config={"responsive": True},
)

# --- Export
if exports_enabled:
    export_cum = cum.copy()

    if FORMAT_EXPORTS:
        export_cum = export_cum.applymap(
            lambda v: float(f"{v:.{DISPLAY_SIG_FIGS}g}") if pd.notna(v) else v
        )

    st.download_button(
        "Download cumulative growth (CSV)",
        df_to_csv_bytes(export_cum, True),
        "cumulative_growth.csv",
        "text/csv",
    )

# ============================================================
# 9A) MACRO INPUTS: RISK-FREE RATE & MARKET RISK PREMIUM
# ============================================================
st.subheader("Macro Inputs: Risk-Free Rate & Market Risk Premium")

st.caption(
    "These series provide macro context for CAPM outputs. "
    "Market Risk Premium (MRP) = Market Log Return − RF Log Return using weekly log returns."
)

tab_weekly, tab_roll, tab_cum_rf = st.tabs(
    ["Weekly (log returns)", f"Rolling annualized ({horizon_weeks}w)", "Cumulative $1 growth (Market vs RF)"]
)

with tab_weekly:
    st.subheader("Weekly Risk-Free Rate and Market Risk Premium")
    st.caption(
        "The weekly excess return of the market (S&P 500) relative to the risk-free rate (3M T-Bill)."
    )

    df_weekly = pd.DataFrame(
        {
            "Date": mkt_rf.index,
            "Risk-Free": mkt_rf["RF_Log_Return"].values,
            "Market Risk Premium": excess_market.values,
        }
    ).dropna()

    if DISPLAY_PCT:
        df_weekly["Risk-Free"] *= 100.0
        df_weekly["Market Risk Premium"] *= 100.0
        y_label = "Weekly return (%)"
        y_is_pct = True
    else:
        y_label = "Weekly log return"
        y_is_pct = False

    df_weekly_melt = df_weekly.melt(
        id_vars="Date", var_name="Series", value_name="Value"
    )

    fig_weekly = px.line(
        df_weekly_melt,
        x="Date",
        y="Value",
        color="Series"
    )

    apply_axis_and_hover_format(
        fig_weekly,
        DISPLAY_SIG_FIGS,
        y_is_percent=y_is_pct,
        y_label=y_label
    )

    st.plotly_chart(fig_weekly, use_container_width=True)


with tab_roll:
    st.subheader(f"Rolling Annualized Risk-Free Rate and Market Risk Premium ({horizon_weeks}-week mean × 52)")
    st.caption(
        "Rolling annualized averages smooth short-term fluctuations to highlight changes in the market risk environment over time."
    )

    rf_roll_ann = mkt_rf["RF_Log_Return"].rolling(horizon_weeks).mean() * 52.0
    mrp_roll_ann = excess_market.rolling(horizon_weeks).mean() * 52.0

    df_roll = pd.DataFrame(
        {
            "Date": mkt_rf.index,
            "Risk-Free (annualized)": rf_roll_ann.values,
            "MRP (annualized)": mrp_roll_ann.values,
        }
    ).dropna()

    if DISPLAY_PCT:
        df_roll["Risk-Free (annualized)"] *= 100.0
        df_roll["MRP (annualized)"] *= 100.0
        y_label = "Annualized (%, log approx)"
        y_is_pct = True
    else:
        y_label = "Annualized (log)"
        y_is_pct = False

    df_roll_melt = df_roll.melt(
        id_vars="Date", var_name="Series", value_name="Value"
    )

    fig_roll = px.line(
        df_roll_melt,
        x="Date",
        y="Value",
        color="Series"
    )

    apply_axis_and_hover_format(
        fig_roll,
        DISPLAY_SIG_FIGS,
        y_is_percent=y_is_pct,
        y_label=y_label
    )

    st.plotly_chart(fig_roll, use_container_width=True)


with tab_cum_rf:
    st.subheader("Cumulative Growth of $1: Market vs Risk-Free")
    st.caption(
        "The total growth of a $1 investment in the market (S&P 500) against the risk-free asset (3M T-bill) from 2014 to present."
    )

    df_cum_rf = pd.DataFrame(
        {
            "Date": mkt_rf.index,
            "Market ($1 growth)": cumulative_from_log_returns(mkt_rf["Market_Log_Return"]).values,
            "Risk-Free ($1 growth)": cumulative_from_log_returns(mkt_rf["RF_Log_Return"]).values,
        }
    ).dropna()

    df_cum_rf_melt = df_cum_rf.melt(
        id_vars="Date", var_name="Series", value_name="Value"
    )

    fig_cum_rf = px.line(
        df_cum_rf_melt,
        x="Date",
        y="Value",
        color="Series"
    )

    apply_axis_and_hover_format(
        fig_cum_rf,
        DISPLAY_SIG_FIGS,
        y_is_percent=False,
        y_label="$1 growth"
    )

    st.plotly_chart(fig_cum_rf, use_container_width=True)

# ============================================================
# 10) ROLLING BETA/ALPHA + ROLLING DISCOUNT RATE
# ============================================================
st.subheader("Rolling Beta, Alpha, and Discount Rate (Excess Returns)")

tab_mkt, tab_sec, tab_disc, tab_tstat = st.tabs(
    [
        "Rolling vs Market",
        "Rolling vs Sector Benchmark (per ticker)",
        "Rolling Discount Rate",
        "Rolling Alpha t-Stats",
    ]
)

with tab_mkt:
    rows = []
    for t in excess_stocks.columns:
        r = rolling_alpha_beta(excess_stocks[t], excess_market, horizon_weeks)
        if not r.empty:
            rr = r.reset_index()
            rr["Ticker"] = t
            rr["Label"] = ticker_label(t)
            rows.append(rr)

    if not rows:
        st.warning("No rolling results vs market. Try a smaller rolling window (e.g., 52).")
    else:
        df_rm = pd.concat(rows, ignore_index=True)

        fig_beta = px.line(df_rm, x="Date", y="beta", color="Label", title="Rolling Beta vs Market")
        apply_axis_and_hover_format(fig_beta, DISPLAY_SIG_FIGS, y_is_percent=False, y_label="Beta")
        st.plotly_chart(fig_beta, use_container_width=True)

        df_alpha = df_rm.copy()
        if DISPLAY_PCT:
            df_alpha["alpha_disp"] = df_alpha["alpha"] * 100.0
            ycol, ylab, is_pct = "alpha_disp", "Alpha (weekly)", True
        else:
            ycol, ylab, is_pct = "alpha", "Alpha (weekly)", False

        fig_alpha = px.line(df_alpha, x="Date", y=ycol, color="Label", title="Rolling Alpha vs Market (weekly)")
        apply_axis_and_hover_format(fig_alpha, DISPLAY_SIG_FIGS, y_is_percent=is_pct, y_label=ylab)
        st.plotly_chart(fig_alpha, use_container_width=True)

        if exports_enabled:
            export_rm = df_rm.copy()
            if FORMAT_EXPORTS:
                for col in ["beta", "alpha"]:
                    export_rm[col] = export_rm[col].map(lambda v: float(f"{v:.{DISPLAY_SIG_FIGS}g}") if pd.notna(v) else v)
            st.download_button(
                "Download rolling vs market (CSV)",
                export_rm.to_csv(index=False).encode("utf-8"),
                "rolling_vs_market.csv",
                "text/csv",
            )

with tab_sec:
    rows = []
    for t in excess_stocks.columns:
        sec = selected_ticker_sector.get(t)
        if sec not in excess_sectors.columns:
            continue

        r = rolling_alpha_beta(excess_stocks[t], excess_sectors[sec], horizon_weeks)
        if not r.empty:
            rr = r.reset_index()
            rr["Ticker"] = t
            rr["Label"] = ticker_label(t)
            rr["Sector_Benchmark"] = sec
            rows.append(rr)

    if not rows:
        st.warning("No rolling results vs sector benchmark. Try a smaller rolling window (e.g., 52).")
    else:
        df_rs = pd.concat(rows, ignore_index=True)

        fig_beta_s = px.line(
            df_rs, x="Date", y="beta", color="Label",
            title="Rolling Beta vs Sector Benchmark",
            hover_data=["Sector_Benchmark"],
        )
        apply_axis_and_hover_format(fig_beta_s, DISPLAY_SIG_FIGS, y_is_percent=False, y_label="Beta")
        st.plotly_chart(fig_beta_s, use_container_width=True)

        df_alpha_s = df_rs.copy()
        if DISPLAY_PCT:
            df_alpha_s["alpha_disp"] = df_alpha_s["alpha"] * 100.0
            ycol, ylab, is_pct = "alpha_disp", "Alpha (weekly)", True
        else:
            ycol, ylab, is_pct = "alpha", "Alpha (weekly)", False

        fig_alpha_s = px.line(
            df_alpha_s, x="Date", y=ycol, color="Label",
            title="Rolling Alpha vs Sector Benchmark (weekly)",
            hover_data=["Sector_Benchmark"],
        )
        apply_axis_and_hover_format(fig_alpha_s, DISPLAY_SIG_FIGS, y_is_percent=is_pct, y_label=ylab)
        st.plotly_chart(fig_alpha_s, use_container_width=True)

        if exports_enabled:
            export_rs = df_rs.copy()
            if FORMAT_EXPORTS:
                for col in ["beta", "alpha"]:
                    export_rs[col] = export_rs[col].map(lambda v: float(f"{v:.{DISPLAY_SIG_FIGS}g}") if pd.notna(v) else v)
            st.download_button(
                "Download rolling vs sector (CSV)",
                export_rs.to_csv(index=False).encode("utf-8"),
                "rolling_vs_sector.csv",
                "text/csv",
            )

with tab_disc:
    rf_roll_ann = mkt_rf["RF_Log_Return"].rolling(horizon_weeks).mean() * TRADING_WEEKS
    mrp_roll_ann = excess_market.rolling(horizon_weeks).mean() * TRADING_WEEKS

    mrp_used_ann = pd.Series(mrp_annual_log, index=common_dates) if use_custom_mrp else mrp_roll_ann

    disc_rows = []
    for t in excess_stocks.columns:
        r = rolling_alpha_beta(excess_stocks[t], excess_market, horizon_weeks)
        if r.empty:
            continue

        beta_roll = r["beta"]
        rf_part = rf_roll_ann.loc[beta_roll.index]
        mrp_part = mrp_used_ann.loc[beta_roll.index]
        disc = rf_part + beta_roll * mrp_part

        disc_rows.append(
            pd.DataFrame({"Date": disc.index, "Label": ticker_label(t), "DiscountRate": disc.values})
        )

    if not disc_rows:
        st.warning(
            "No rolling discount rate available. Try a smaller window (e.g., 52) or choose tickers with longer history."
        )
    else:
        df_disc = pd.concat(disc_rows, ignore_index=True)

        df_disc_plot = df_disc.copy()
        if DISPLAY_PCT:
            df_disc_plot["DiscountRate_disp"] = df_disc_plot["DiscountRate"] * 100.0
            ycol, ylab, is_pct = "DiscountRate_disp", "Discount Rate (annual, log)", True
        else:
            ycol, ylab, is_pct = "DiscountRate", "Discount Rate (annual, log)", False

        # Title + caption in Streamlit (so caption sits *under* the subtitle)
        st.subheader("Rolling Discount Rate (Annualized, log approx)")
        st.caption(
            "Discount rate (annualized, log approx): RF annual + rolling beta × MRP annual. "
            "Rolling RF and rolling MRP are computed over the same rolling window."
        )

        # Plotly chart WITHOUT a title (title is handled by st.subheader above)
        fig_disc = px.line(
            df_disc_plot,
            x="Date",
            y=ycol,
            color="Label",
        )
        apply_axis_and_hover_format(fig_disc, DISPLAY_SIG_FIGS, y_is_percent=is_pct, y_label=ylab)
        st.plotly_chart(fig_disc, use_container_width=True)

        if exports_enabled:
            export_disc = df_disc.copy()
            if FORMAT_EXPORTS:
                export_disc["DiscountRate"] = export_disc["DiscountRate"].map(
                    lambda v: float(f"{v:.{DISPLAY_SIG_FIGS}g}") if pd.notna(v) else v
                )
            st.download_button(
                "Download rolling discount rate (CSV)",
                export_disc.to_csv(index=False).encode("utf-8"),
                "rolling_discount_rate.csv",
                "text/csv",
            )

st.subheader("Rolling Discount Rate (Annualized, log approx)")

st.caption(
    "Discount rate (annualized, log approx): RF annual + rolling beta × MRP annual. "
    "Rolling RF and rolling MRP are computed over the same rolling window."
)

fig_disc = px.line(
    df_disc_plot,
    x="Date",
    y=ycol,
    color="Label",
)

apply_axis_and_hover_format(
    fig_disc,
    DISPLAY_SIG_FIGS,
    y_is_percent=is_pct,
    y_label=ylab
)

st.plotly_chart(fig_disc, use_container_width=True)


        if exports_enabled:
            export_disc = df_disc.copy()
            if FORMAT_EXPORTS:
                export_disc["DiscountRate"] = export_disc["DiscountRate"].map(lambda v: float(f"{v:.{DISPLAY_SIG_FIGS}g}") if pd.notna(v) else v)
            st.download_button(
                "Download rolling discount rate (CSV)",
                export_disc.to_csv(index=False).encode("utf-8"),
                "rolling_discount_rate.csv",
                "text/csv",
            )

with tab_tstat:
    st.caption(
        "Rolling alpha t-statistics from a rolling CAPM regression. "
        "Dashed lines at ±2 indicate a common ~5% significance level."
    )

    model_choice = st.radio("Alpha t-stat model", ["Vs Market", "Vs Sector Benchmark"], horizontal=True)

    rows = []
    if model_choice == "Vs Market":
        for t in excess_stocks.columns:
            r = rolling_capm_tstats(excess_stocks[t], excess_market, horizon_weeks)
            if not r.empty:
                rr = r.reset_index()
                rr["Label"] = ticker_label(t)
                rows.append(rr)

        if not rows:
            st.warning("No rolling alpha t-stats vs market. Try a smaller window (e.g., 52).")
        else:
            df_rt = pd.concat(rows, ignore_index=True)
            fig = px.line(df_rt, x="Date", y="alpha_t", color="Label", title=f"Rolling Alpha t-Statistics ({horizon_weeks}-Week Window, Market)")
            fig.add_hline(y=2, line_dash="dash", line_color="gray")
            fig.add_hline(y=-2, line_dash="dash", line_color="gray")
            fig.update_yaxes(title="Alpha t-stat", tickformat=f".{DISPLAY_SIG_FIGS}g")
            fig.update_traces(hovertemplate=f"Date=%{{x}}<br>Alpha t=%{{y:.{DISPLAY_SIG_FIGS}g}}<extra></extra>")
            st.plotly_chart(fig, use_container_width=True)

            if exports_enabled:
                export_df = df_rt[["Date", "Label", "alpha", "beta", "alpha_t", "beta_t"]].copy()
                if FORMAT_EXPORTS:
                    for c in ["alpha", "beta", "alpha_t", "beta_t"]:
                        export_df[c] = export_df[c].map(lambda v: float(f"{v:.{DISPLAY_SIG_FIGS}g}") if pd.notna(v) else v)
                st.download_button(
                    "Download rolling alpha t-stats vs market (CSV)",
                    export_df.to_csv(index=False).encode("utf-8"),
                    "rolling_alpha_tstats_vs_market.csv",
                    "text/csv",
                )

    else:
        for t in excess_stocks.columns:
            sec = selected_ticker_sector.get(t)
            if sec not in excess_sectors.columns:
                continue

            r = rolling_capm_tstats(excess_stocks[t], excess_sectors[sec], horizon_weeks)
            if not r.empty:
                rr = r.reset_index()
                rr["Label"] = ticker_label(t)
                rr["Sector_Benchmark"] = sec
                rows.append(rr)

        if not rows:
            st.warning("No rolling alpha t-stats vs sector benchmark. Try a smaller window (e.g., 52).")
        else:
            df_rt = pd.concat(rows, ignore_index=True)
            fig = px.line(df_rt, x="Date", y="alpha_t", color="Label", title=f"Rolling Alpha t-Statistics ({horizon_weeks}-Week Window, Sector Benchmark)", hover_data=["Sector_Benchmark"])
            fig.add_hline(y=2, line_dash="dash", line_color="gray")
            fig.add_hline(y=-2, line_dash="dash", line_color="gray")
            fig.update_yaxes(title="Alpha t-stat", tickformat=f".{DISPLAY_SIG_FIGS}g")
            fig.update_traces(hovertemplate=f"Date=%{{x}}<br>Alpha t=%{{y:.{DISPLAY_SIG_FIGS}g}}<extra></extra>")
            st.plotly_chart(fig, use_container_width=True)

            if exports_enabled:
                export_df = df_rt[["Date", "Label", "Sector_Benchmark", "alpha", "beta", "alpha_t", "beta_t"]].copy()
                if FORMAT_EXPORTS:
                    for c in ["alpha", "beta", "alpha_t", "beta_t"]:
                        export_df[c] = export_df[c].map(lambda v: float(f"{v:.{DISPLAY_SIG_FIGS}g}") if pd.notna(v) else v)
                st.download_button(
                    "Download rolling alpha t-stats vs sector (CSV)",
                    export_df.to_csv(index=False).encode("utf-8"),
                    "rolling_alpha_tstats_vs_sector.csv",
                    "text/csv",
                )


# ============================================================
# 11A) SUMMARY METRICS (ANNUALIZED) + DISCOUNT RATE + R2 / ADJ R2
# ============================================================

st.subheader("Summary Metrics (Annualized)")

rf_annual_log = annualize_log_mean(mkt_rf["RF_Log_Return"].mean())

summary_rows: List[dict] = []
for t in stocks_wide.columns:
    r = stocks_wide[t].dropna()
    if r.empty:
        continue

    rf = mkt_rf["RF_Log_Return"].loc[r.index]
    ex = (r - rf).dropna()

    ann_ret = annualize_log_mean(r.mean())
    ann_vol = annualize_vol(r.std(ddof=0))
    ann_ex = annualize_log_mean(ex.mean())
    sharpe = float(ann_ex / ann_vol) if ann_vol and ann_vol != 0 and not np.isnan(ann_vol) else np.nan

    reg = capm_ols_metrics(ex, excess_market.loc[ex.index])
    beta = reg.get("beta", np.nan)

    disc_rate_annual_log = (
        float(rf_annual_log + beta * mrp_annual_log)
        if np.isfinite(beta)
        else np.nan
    )

    summary_rows.append(
        {
            "Label": ticker_label(t),
            "Obs": reg.get("n", len(ex)),
            "Discount_Rate_Annual_(log)": disc_rate_annual_log,
            "Ann_Return_(log)": float(ann_ret),
            "Ann_Vol": float(ann_vol),
            "Ann_Excess_Return_(log)": float(ann_ex),
            "Sharpe": sharpe,
            "Beta_vs_Market": reg.get("beta", np.nan),
            "Alpha_vs_Market_(weekly)": reg.get("alpha", np.nan),
            "Beta_tstat_vs_Market": reg.get("beta_t", np.nan),
            "Alpha_tstat_vs_Market": reg.get("alpha_t", np.nan),
            "R2_vs_Market": reg.get("r2", np.nan),
            "Adj_R2_vs_Market": reg.get("adj_r2", np.nan),
        }
    )

summary_df = pd.DataFrame(summary_rows).set_index("Label").sort_index()

# ------------------------------------------------------------
# 11B) Display formatting (percent vs numeric)
# ------------------------------------------------------------
summary_df_display = summary_df.copy()

rate_cols = {
    "Discount_Rate_Annual_(log)",
    "Ann_Return_(log)",
    "Ann_Vol",
    "Ann_Excess_Return_(log)",
    "Alpha_vs_Market_(weekly)",
}

for col in summary_df_display.columns:
    if pd.api.types.is_numeric_dtype(summary_df_display[col]):
        if DISPLAY_PCT and col in rate_cols:
            summary_df_display[col] = summary_df_display[col].map(
                lambda v: sig_pct_str(v, DISPLAY_SIG_FIGS)
            )
        else:
            summary_df_display[col] = summary_df_display[col].map(
                lambda v: sig_str(v, DISPLAY_SIG_FIGS)
            )

# ------------------------------------------------------------
# 11C) Render table with column-level hover tooltips
# ------------------------------------------------------------
st.dataframe(
    summary_df_display,
    use_container_width=True,
    column_config={
        "Obs": st.column_config.NumberColumn(
            "Obs",
            help="Number of weekly observations used in the CAPM regression."
        ),
        "Discount_Rate_Annual_(log)": st.column_config.NumberColumn(
            "Discount Rate (Annual, log)",
            help="The required return on equity implied by the stock’s exposure to systematic market risk."
        ),
        "Ann_Return_(log)": st.column_config.NumberColumn(
            "Annual Return (log)",
            help="Average realized annual return computed from weekly log returns."
        ),
        "Ann_Vol": st.column_config.NumberColumn(
            "Annualized Volatility",
            help="The standard deviation of a ticker’s annualized returns i.e., how much returns typically fluctuate around the mean."
        ),
        "Ann_Excess_Return_(log)": st.column_config.NumberColumn(
            "Annual Excess Return (log)",
            help="Average annual return in excess of the risk-free rate."
        ),
        "Sharpe": st.column_config.NumberColumn(
            "Sharpe Ratio",
            help="Excess return per unit of total risk (volatility). Higher = Better"
        ),
        "Beta_vs_Market": st.column_config.NumberColumn(
            "Beta vs Market",
            help="CAPM beta measuring sensitivity to market excess returns; a beta of 1 indicates market-level risk."
        ),
        "Alpha_vs_Market_(weekly)": st.column_config.NumberColumn(
            "Alpha (Weekly)",
            help="CAPM alpha representing the average weekly excess return unexplained by market risk."
        ),
        "Beta_tstat_vs_Market": st.column_config.NumberColumn(
            "Beta t-stat",
            help="  ∣t∣≥1.96→ indicates beta is statistically different from zero at ~5%, implying reliable exposure to market risk."
        ),
        "Alpha_tstat_vs_Market": st.column_config.NumberColumn(
            "Alpha t-stat",
            help="  ∣t∣≥1.96→ indicates alpha is statistically different from zero at ~5%, implying statistically significant abnormal returns."
        ),
        "R2_vs_Market": st.column_config.NumberColumn(
            "R² vs Market",
            help="Fraction of return variance explained by market movements."
        ),
        "Adj_R2_vs_Market": st.column_config.NumberColumn(
            "Adjusted R² vs Market",
            help="R² adjusted for model complexity; nearly identical to R² in a single-factor CAPM."
        ),
    },
)

# ------------------------------------------------------------
# 11D) Footnote: Market Risk Premium used
# ------------------------------------------------------------
mrp_note = (
    sig_pct_str(mrp_annual_log, DISPLAY_SIG_FIGS)
    if DISPLAY_PCT
    else sig_str(mrp_annual_log, DISPLAY_SIG_FIGS)
)

st.caption(
    f"Market Risk Premium used (annual, log approx): {mrp_note} "
    f"({'custom' if use_custom_mrp else 'historical from aligned data'})"
)

# ============================================================
# 12) FOOTNOTE
# ============================================================
st.caption(
    "Equity prices, tickers, and sector classifications are sourced from Yahoo Finance and based off of the S&P 500. "
    "The risk-free rate is proxied using the 3-month U.S. Treasury bill sourced from the Federal Reserve Economic Data (FRED). "
    "Returns are weekly log returns. CAPM beta/alpha are estimated on excess returns (return minus RF). "
    "Discount rate is computed via CAPM: RF + beta * MRP (annualized log approx). "
    "R² measures variance explained by the market; Adj R² penalizes overfitting. "
    "Display formatting controls affect presentation only (exports keep full precision by default)."
)

st.markdown(
    """

    **Interpretation order (recommended):**
    1. **Macro Inputs** — the Risk-Free Rate (RF) and Market Risk Premium (MRP = Market − RF)
    2. **Cumulative Performance** — growth of $1 for the market, sectors, and selected tickers from 2014 to current
    3. **Rolling CAPM** — beta and alpha estimated on a trailing window (selected in the sidebar)
    4. **Discount Rate** — implied annualized discount rate = RF + β × MRP

    Use **52 weeks** for a more “current” view and **156 weeks** for a more “structural” view.
    """
)































