"""
Mutual Fund Scrapper (Monthly, 2010 → Present) + 10Y Analytics + RF Exports
==========================================================================
AUDIT-READY VERSION (Streamlit Cloud / Python 3.13 compatible)

0) SUMMARY
----------
This Streamlit app:
- Downloads monthly prices (Adj Close) for user-entered fund tickers from Yahoo Finance (yfinance)
- Computes monthly log returns
- Downloads the 30-year Treasury constant maturity yield (FRED: DGS30) as *raw daily data*
- Samples DGS30 to month-end and converts it to a monthly log risk-free proxy
- Builds a tidy panel and stores outputs in st.session_state (robust across reruns)
- Computes 10-year (most recent) fund stats, annualized from monthly data
- Estimates annualized variance-covariance matrix and monthly correlation matrix
- Computes return/volatility for user-defined portfolios
- Exports panel + raw RF (daily) + month-end yield via download buttons (+ optional ephemeral disk write)

1) DEPLOYMENT REQUIREMENTS (repo root: requirements.txt)
--------------------------------------------------------
streamlit
pandas
numpy
yfinance
requests

2) DATA SOURCES
---------------
- Yahoo Finance (prices): yfinance
- FRED (DGS30 yield): official CSV endpoint (no pandas_datareader):
  https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS30

3) AUDIT-RELEVANT DEFINITIONS & ASSUMPTIONS
-------------------------------------------
3.1 Fund monthly log return:
    r_t = log(AdjClose_t / AdjClose_{t-1})

3.2 Risk-free proxy:
    - DGS30 is an annualized yield (%) observed daily.
    - Month-end yield is last daily observation of each month.
    - Monthly log risk-free return proxy:
      RF_Log_Return = log(1 + (DGS30/100)/12)
    NOTE: This is a *rate proxy*, not a bond total return index.

3.3 Annualization (computed from monthly data, then annualized — per user spec):
    - Geometric mean annual return:
      R_geo_ann = exp(12 * mean(monthly_log_returns)) - 1
    - Annualized volatility:
      sigma_ann = std(monthly_log_returns) * sqrt(12)
    - Annualized covariance:
      Cov_ann = Cov_monthly * 12
    - Correlation is computed on monthly returns (dimensionless).

3.4 10-year window:
    - Uses most recent available month in panel, then last 120 months.

4) ROBUSTNESS POLICY (IMPORTANT)
-------------------------------
- Uses st.session_state as the single source of truth:
  st.session_state["panel"], ["audit_panel"], ["rf_daily_raw"], ["rf_month_end"]
  so reruns do not cause NameError or partial state failures.
- Uses @st.cache_data for external pulls (Yahoo/FRED) to reduce rate limits.
- FRED schema robustness:
  Accepts date column variants: DATE or observation_date or date.

"""

from __future__ import annotations

import math
import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


# ============================================================
# 1) CONFIG (AUDIT: centralized parameters)
# ============================================================

@dataclass(frozen=True)
class Config:
    # Default user date range (user can override in sidebar)
    start_date: str = "2010-01-01"
    end_date: str = datetime.today().strftime("%Y-%m-%d")

    # Yahoo Finance sampling
    interval: str = "1mo"

    # Risk-free series (FRED)
    rf_series: str = "DGS30"

    # Outputs (Streamlit Cloud filesystem is ephemeral; downloads are primary)
    output_dir: Path = Path("outputs")
    panel_out: str = "mutual_fund_panel_monthly.csv"
    rf_daily_out: str = "DGS30_daily_raw.csv"
    rf_monthly_out: str = "DGS30_month_end.csv"

    # Networking
    http_timeout_s: int = 20
    user_agent: str = "Mozilla/5.0 (compatible; MutualFundScraper/1.0)"


CFG = Config()


# ============================================================
# 2) AUDIT + DATA UTILITIES
# ============================================================

def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def audit_kv(audit: Dict[str, str], k: str, v: object) -> None:
    audit[k] = str(v)


def normalize_symbol_for_yahoo(symbol: str) -> str:
    # Audit note: Yahoo uses '-' not '.' for some share classes (e.g., BRK-B).
    return str(symbol).strip().replace(".", "-")


def parse_tickers(raw: str) -> List[str]:
    # Accept comma/newline-separated inputs.
    parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
    parts = [p for p in parts if p]
    return [normalize_symbol_for_yahoo(p) for p in parts]


def enforce_month_end(dt: pd.Series) -> pd.Series:
    # Audit note: enforce month-end keys for stable joins.
    return pd.to_datetime(dt).dt.to_period("M").dt.to_timestamp("M")


def monthly_log_return_from_price(price: pd.Series) -> pd.Series:
    # r_t = log(P_t / P_{t-1})
    return np.log(price / price.shift(1))


# ============================================================
# 3) FRED FETCH (NO pandas_datareader; schema-robust)
# ============================================================

@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_fred_series_csv(
    series_id: str,
    start_date: str,
    end_date: str,
    timeout_s: int,
    user_agent: str,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Fetch a FRED series via official CSV endpoint.

    AUDIT NOTES:
    - This is the primary source for DGS30 used in this app.
    - FRED CSV date column has schema variants; we accept:
      'DATE' OR 'observation_date' OR 'date'
    - Only cleaning applied:
      * parse date to datetime
      * coerce series values to numeric
      * drop rows with missing values after coercion
    """
    audit: Dict[str, str] = {}

    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    params = {"id": series_id, "cosd": start_date, "coed": end_date}
    headers = {"User-Agent": user_agent}

    resp = requests.get(url, params=params, headers=headers, timeout=timeout_s)
    resp.raise_for_status()

    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))

    audit_kv(audit, "fred_raw_columns", list(df.columns))

    date_col = None
    for c in ("DATE", "observation_date", "date"):
        if c in df.columns:
            date_col = c
            break

    if date_col is None:
        raise ValueError(f"Unexpected FRED CSV schema (no date column). Columns: {list(df.columns)}")

    if series_id not in df.columns:
        raise ValueError(f"Unexpected FRED CSV schema (missing '{series_id}'). Columns: {list(df.columns)}")

    audit_kv(audit, "fred_date_column_used", date_col)

    df["Date"] = pd.to_datetime(df[date_col])
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")

    out = df[["Date", series_id]].dropna().sort_values("Date").reset_index(drop=True)
    audit_kv(audit, "fred_rows_after_dropna", len(out))

    return out, audit


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_rf_dgs30_all(start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    Fetch and prepare risk-free data.

    Returns:
      rf_daily_raw:  Date, DGS30 (daily, % annualized)      [RAW export + audit]
      rf_month_end:  Date, DGS30 (month-end sampled yield)  [export + traceability]
      rf_monthly_rf: Date, RF_Log_Return                    [used in panel merge]
      audit_rf:      audit metadata (schema, rowcounts, conversion)

    AUDIT NOTES:
    - rf_daily_raw is "raw" in the sense: it reflects the FRED series values after
      minimal parsing/coercion and NA removal (no resampling or transformations).
    - rf_month_end is derived via month-end sampling: resample('M').last()
    - rf_monthly_rf is derived using the rate proxy conversion described in header.
    """
    audit_rf: Dict[str, str] = {}

    rf_daily, fred_audit = fetch_fred_series_csv(
        series_id=CFG.rf_series,
        start_date=start_date,
        end_date=end_date,
        timeout_s=CFG.http_timeout_s,
        user_agent=CFG.user_agent,
    )
    for k, v in fred_audit.items():
        audit_kv(audit_rf, k, v)

    # Daily raw series (standardize column name to DGS30 for exports)
    rf_daily_raw = rf_daily.rename(columns={CFG.rf_series: "DGS30"}).copy()
    rf_daily_raw["Date"] = pd.to_datetime(rf_daily_raw["Date"])
    rf_daily_raw = rf_daily_raw.sort_values("Date").reset_index(drop=True)

    # Month-end yield sampling
    rf_idx = rf_daily_raw.set_index("Date").sort_index()
    rf_month_end = rf_idx.resample("M").last().reset_index()
    rf_month_end["Date"] = enforce_month_end(rf_month_end["Date"])

    # Monthly RF log proxy
    rf_monthly_rf = rf_month_end.copy()
    rf_monthly_rf["RF_Log_Return"] = np.log1p((rf_monthly_rf["DGS30"] / 100.0) / 12.0)
    rf_monthly_rf = rf_monthly_rf[["Date", "RF_Log_Return"]].dropna().sort_values("Date").reset_index(drop=True)

    audit_kv(audit_rf, "rf_daily_rows", len(rf_daily_raw))
    audit_kv(audit_rf, "rf_month_end_rows", len(rf_month_end))
    audit_kv(audit_rf, "rf_monthly_rows", len(rf_monthly_rf))
    audit_kv(audit_rf, "rf_conversion", "RF_Log_Return = log(1 + (DGS30/100)/12)")

    return rf_daily_raw, rf_month_end, rf_monthly_rf, audit_rf


# ============================================================
# 4) YFINANCE PRICE DOWNLOADS (MONTHLY)
# ============================================================

@st.cache_data(show_spinner=False, ttl=60 * 60)
def safe_yf_fund_name(ticker: str) -> Optional[str]:
    """
    Best-effort fund name lookup.

    AUDIT NOTE:
    - Yahoo metadata may be incomplete or rate-limited.
    - Missing Fund_Name is acceptable; it does not affect return calculations.
    """
    try:
        info = yf.Ticker(ticker).info or {}
        return info.get("longName") or info.get("shortName") or None
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=60 * 60)
def download_monthly_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Download monthly prices and return tidy long format:

      Date, Ticker, Adj_Close

    AUDIT NOTE:
    - Uses 'Adj Close' if available, then renamed to Adj_Close.
    - Forces month-end timestamps for stable time-series keys.
    """
    px = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=CFG.interval,
        auto_adjust=False,
        group_by="ticker",
        threads=False,
        progress=False,
    )

    if px is None or len(px) == 0:
        return pd.DataFrame()

    # MultiIndex columns when multiple tickers
    if isinstance(px.columns, pd.MultiIndex):
        out = (
            px.stack(level=0)
              .rename_axis(["Date", "Ticker"])
              .reset_index()
        )
    else:
        # Single ticker returns flat columns
        out = px.reset_index()
        out["Ticker"] = tickers[0]

    if "Adj Close" not in out.columns:
        raise ValueError("Yahoo returned data without 'Adj Close' column.")

    out = out[["Date", "Ticker", "Adj Close"]].dropna()
    out = out.rename(columns={"Adj Close": "Adj_Close"})
    out["Ticker"] = out["Ticker"].astype(str).map(normalize_symbol_for_yahoo)
    out["Date"] = enforce_month_end(out["Date"])

    return out.sort_values(["Ticker", "Date"]).reset_index(drop=True)


# ============================================================
# 5) PANEL BUILD (PRICES + RETURNS + RF MERGE)
# ============================================================

def build_panel(tickers: List[str], start_date: str, end_date: str) -> Tuple[pd.DataFrame, Dict[str, str], pd.DataFrame, pd.DataFrame]:
    """
    Build the core panel and return:
      panel: tidy long dataframe
      audit: audit dictionary
      rf_daily_raw: raw DGS30 daily yield
      rf_month_end: month-end DGS30 yield

    PANEL SCHEMA:
      Date, Ticker, Fund_Name, Adj_Close, Log_Return, RF_Log_Return
    """
    audit: Dict[str, str] = {}

    # --- Audit metadata
    audit_kv(audit, "run_timestamp_local", now_iso())
    audit_kv(audit, "tickers_input", ", ".join(tickers))
    audit_kv(audit, "start_date", start_date)
    audit_kv(audit, "end_date", end_date)
    audit_kv(audit, "yahoo_interval", CFG.interval)
    audit_kv(audit, "fred_series", CFG.rf_series)
    audit_kv(audit, "month_end_alignment", "prices month-end; DGS30 month-end via resample('M').last()")
    audit_kv(audit, "return_def", "Log_Return = log(Adj_Close / Adj_Close.shift(1))")
    audit_kv(audit, "rf_def", "RF_Log_Return = log(1 + (DGS30/100)/12)")

    # --- Prices
    prices = download_monthly_prices(tickers, start_date, end_date)
    audit_kv(audit, "prices_rows_raw", len(prices))
    audit_kv(audit, "prices_unique_tickers", prices["Ticker"].nunique() if not prices.empty else 0)

    if prices.empty:
        raise ValueError("No price data returned from Yahoo Finance for the tickers/date range.")

    # --- Monthly log returns (drop first observation per ticker)
    prices["Log_Return"] = prices.groupby("Ticker")["Adj_Close"].transform(monthly_log_return_from_price)
    before = len(prices)
    prices = prices.dropna(subset=["Log_Return"]).copy()
    audit_kv(audit, "prices_rows_after_return_drop", len(prices))
    audit_kv(audit, "prices_rows_dropped_first_obs", before - len(prices))

    # --- Fund names (best effort)
    meta = pd.DataFrame({"Ticker": tickers, "Fund_Name": [safe_yf_fund_name(t) for t in tickers]})
    prices = prices.merge(meta, on="Ticker", how="left")

    # --- Risk-free (daily raw + month-end + monthly RF proxy)
    rf_daily_raw, rf_month_end, rf_monthly_rf, rf_audit = fetch_rf_dgs30_all(start_date, end_date)
    for k, v in rf_audit.items():
        audit_kv(audit, f"rf_{k}", v)

    # --- Merge monthly RF into panel
    panel = prices.merge(rf_monthly_rf, on="Date", how="left")
    audit_kv(audit, "panel_rows_pre_dropna", len(panel))
    audit_kv(audit, "panel_missing_rf_rows", int(panel["RF_Log_Return"].isna().sum()))

    panel = panel.dropna(subset=["RF_Log_Return"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    audit_kv(audit, "panel_rows_final", len(panel))

    # --- Data quality: monotonic date check per ticker
    violations = 0
    for t, g in panel.groupby("Ticker"):
        if not g["Date"].is_monotonic_increasing:
            violations += 1
    audit_kv(audit, "monotonic_date_violations_tickers", violations)

    # --- Coverage preview
    cov = panel.groupby("Ticker")["Date"].agg(["min", "max", "count"]).reset_index()
    audit_kv(audit, "coverage_preview", cov.head(10).to_string(index=False))

    # Stable schema order
    panel = panel[["Date", "Ticker", "Fund_Name", "Adj_Close", "Log_Return", "RF_Log_Return"]]
    return panel, audit, rf_daily_raw, rf_month_end


def try_write_csv(df: pd.DataFrame, path: Path) -> Tuple[bool, Optional[str]]:
    """
    Attempt to write CSV to disk.

    AUDIT NOTE:
    - On Streamlit Cloud, disk writes are ephemeral; do not treat as durable storage.
    - Download buttons are the authoritative export mechanism.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return True, None
    except Exception as e:
        return False, str(e)


# ============================================================
# 6) ANALYTICS (10Y STATS + COV/CORR + PORTFOLIOS)
# ============================================================

def compute_10y_analytics(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    Compute 10-year analytics.

    Returns:
      stats: per-fund annual geometric mean & annualized volatility (from monthly data)
      cov_ann: annualized covariance matrix (monthly cov × 12)
      corr: correlation matrix (monthly)
      audit_stats: audit dictionary for window & annualization rules
    """
    audit_stats: Dict[str, str] = {}

    df = panel.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    last_dt = df["Date"].max()
    start_10y = (last_dt.to_period("M") - 120).to_timestamp("M")

    audit_kv(audit_stats, "window_last_date", last_dt.date())
    audit_kv(audit_stats, "window_start_date", start_10y.date())
    audit_kv(audit_stats, "window_months_target", 120)
    audit_kv(audit_stats, "annualization_return", "exp(12*mean(log_r_m)) - 1")
    audit_kv(audit_stats, "annualization_vol", "std(log_r_m)*sqrt(12)")
    audit_kv(audit_stats, "annualization_cov", "cov_m*12")

    df_10y = df[df["Date"] > start_10y].copy()

    # Monthly log return matrix
    R = (
        df_10y.pivot_table(index="Date", columns="Ticker", values="Log_Return", aggfunc="mean")
        .sort_index()
    )

    audit_kv(audit_stats, "returns_matrix_rows", R.shape[0])
    audit_kv(audit_stats, "returns_matrix_cols", R.shape[1])

    # Fund stats from monthly log returns
    mu_m_log = R.mean(skipna=True)
    sig_m = R.std(ddof=1, skipna=True)

    geo_ann = np.expm1(12.0 * mu_m_log)
    sig_ann = sig_m * math.sqrt(12.0)

    stats = pd.DataFrame({
        "Geometric_Mean_Return_Annual": geo_ann,
        "Volatility_Annual": sig_ann,
        "Mean_LogReturn_Monthly": mu_m_log,
        "Std_LogReturn_Monthly": sig_m,
        "Months_Available": R.notna().sum(),
    }).sort_index()

    cov_m = R.cov()          # monthly covariance (pairwise by default)
    cov_ann = cov_m * 12.0   # annualize covariance
    corr = R.corr()          # monthly correlation

    return stats, cov_ann, corr, audit_stats


def parse_portfolios(text: str) -> Dict[str, Dict[str, float]]:
    """
    Parse portfolios from text.

    Format:
      PORT1: VFINX=0.6, FCNTX=0.4
      PORT2: VFINX=0.5, FCNTX=0.5
    """
    ports: Dict[str, Dict[str, float]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            name, rhs = line.split(":", 1)
            name = name.strip()
        else:
            name, rhs = f"PORT{len(ports) + 1}", line

        w: Dict[str, float] = {}
        for part in rhs.split(","):
            part = part.strip()
            if not part or "=" not in part:
                continue
            t, val = part.split("=", 1)
            w[t.strip()] = float(val.strip())
        if w:
            ports[name] = w
    return ports


def compute_portfolios(
    stats: pd.DataFrame,
    cov_ann: pd.DataFrame,
    portfolios: Dict[str, Dict[str, float]],
    normalize_weights: bool,
) -> pd.DataFrame:
    """
    Compute portfolio annual return & volatility.

    AUDIT NOTES:
    - Portfolio mean return is estimated as w' * fund_geo_ann, where fund_geo_ann is
      computed from monthly returns and annualized.
    - Portfolio volatility is sqrt(w' * Cov_ann * w).
    """
    tickers = list(stats.index)
    mu = stats["Geometric_Mean_Return_Annual"]
    covA = cov_ann.loc[tickers, tickers]

    rows = []
    for name, wdict in portfolios.items():
        w = pd.Series(0.0, index=tickers)
        ignored = []
        for t, wt in wdict.items():
            if t in w.index:
                w.loc[t] = wt
            else:
                ignored.append(t)

        if normalize_weights:
            s = float(w.sum())
            if s != 0:
                w = w / s

        rp = float((w * mu).sum())
        varp = float(w.values.T @ covA.values @ w.values)
        volp = math.sqrt(varp) if varp >= 0 else float("nan")

        rows.append({
            "Portfolio": name,
            "Return_Annual_Geometric_Est": rp,
            "Volatility_Annual": volp,
            "Weights_Sum": float(w.sum()),
            "Num_Funds_Used": int((w != 0).sum()),
            "Ignored_Tickers": ", ".join(ignored) if ignored else "",
        })

    return pd.DataFrame(rows).set_index("Portfolio")


# ============================================================
# 7) STREAMLIT UI (ROBUST: session_state single source of truth)
# ============================================================

st.set_page_config(page_title="Mutual Fund Scrapper", layout="wide")
st.title("Mutual Fund Scrapper — Monthly Data + 10Y Analytics + RF Exports")

# 7.1 Session state initialization (audit: deterministic state keys)
for k in ("panel", "audit_panel", "rf_daily_raw", "rf_month_end"):
    if k not in st.session_state:
        st.session_state[k] = None

with st.sidebar:
    st.header("7) Inputs")

    tickers_text = st.text_area(
        "7.1 Enter tickers (comma or newline separated)",
        value="VFINX, FCNTX",
        height=110,
        help="Enter mutual fund / ETF tickers supported by Yahoo Finance."
    )

    start_date = st.text_input("7.2 Start date (YYYY-MM-DD)", CFG.start_date)
    end_date = st.text_input("7.3 End date (YYYY-MM-DD)", CFG.end_date)

    c1, c2 = st.columns(2)
    with c1:
        run_btn = st.button("7.4 Pull Data", use_container_width=True)
    with c2:
        clear_btn = st.button("7.5 Clear Results", use_container_width=True)

if clear_btn:
    for k in ("panel", "audit_panel", "rf_daily_raw", "rf_month_end"):
        st.session_state[k] = None
    st.success("Cleared stored results.")

# 7.2 Run build on demand
if run_btn:
    tickers = parse_tickers(tickers_text)
    if not tickers:
        st.error("Please enter at least one ticker.")
        st.stop()

    st.write(f"**Tickers:** {', '.join(tickers)}")
    st.write(f"**Date range:** {start_date} → {end_date}")
    st.write(f"**Risk-free:** FRED {CFG.rf_series} (30Y), daily raw + month-end + monthly RF proxy")

    with st.spinner("Building panel (Yahoo Finance + FRED)…"):
        try:
            panel, audit, rf_daily_raw, rf_month_end = build_panel(tickers, start_date, end_date)
        except Exception as e:
            st.error("Failed to build panel.")
            st.exception(e)
            st.stop()

    # Store results in session_state (most robust pattern)
    st.session_state["panel"] = panel
    st.session_state["audit_panel"] = audit
    st.session_state["rf_daily_raw"] = rf_daily_raw
    st.session_state["rf_month_end"] = rf_month_end

    # Optional ephemeral writes
    ok_panel, err_panel = try_write_csv(panel, CFG.output_dir / CFG.panel_out)
    ok_rf_d, err_rf_d = try_write_csv(rf_daily_raw, CFG.output_dir / CFG.rf_daily_out)
    ok_rf_m, err_rf_m = try_write_csv(rf_month_end, CFG.output_dir / CFG.rf_monthly_out)

    st.success("Build completed.")
    if not ok_panel:
        st.warning(f"Could not write panel to disk (download still available). Details: {err_panel}")
    if not ok_rf_d:
        st.warning(f"Could not write DGS30 daily to disk (download still available). Details: {err_rf_d}")
    if not ok_rf_m:
        st.warning(f"Could not write DGS30 month-end to disk (download still available). Details: {err_rf_m}")

# ============================================================
# 8) OUTPUTS (PANEL + AUDIT + DOWNLOADS)
# ============================================================

panel = st.session_state.get("panel")
audit_panel = st.session_state.get("audit_panel")
rf_daily_raw = st.session_state.get("rf_daily_raw")
rf_month_end = st.session_state.get("rf_month_end")

if panel is None:
    st.info("Click **7.4 Pull Data** to build the panel and enable analytics/exports.")
    st.stop()

st.header("8) Panel Output")

st.subheader("8.1 Panel Preview")
st.dataframe(panel.head(25), use_container_width=True)

st.subheader("8.2 Panel Audit Notes")
if isinstance(audit_panel, dict) and audit_panel:
    st.code("\n".join([f"{k}: {v}" for k, v in audit_panel.items()]))
else:
    st.caption("No audit dictionary found in session_state (unexpected).")

st.subheader("8.3 Download Panel CSV")
st.download_button(
    "Download mutual fund panel CSV",
    data=panel.to_csv(index=False).encode("utf-8"),
    file_name=CFG.panel_out,
    mime="text/csv",
)

# ============================================================
# 9) RISK-FREE EXPORTS (RAW DAILY + MONTH-END YIELD)
# ============================================================

st.header("9) Risk-Free Exports (30Y Treasury DGS30)")

if rf_daily_raw is None or rf_month_end is None:
    st.warning("Risk-free series is not available in session_state (unexpected). Re-run Pull Data.")
else:
    st.subheader("9.1 DGS30 Daily Raw (as received/cleaned)")
    st.dataframe(rf_daily_raw.head(25), use_container_width=True)
    st.download_button(
        "Download DGS30 daily raw CSV",
        data=rf_daily_raw.to_csv(index=False).encode("utf-8"),
        file_name=CFG.rf_daily_out,
        mime="text/csv",
    )

    st.subheader("9.2 DGS30 Month-End Yield (resample M last)")
    st.dataframe(rf_month_end.head(25), use_container_width=True)
    st.download_button(
        "Download DGS30 month-end yield CSV",
        data=rf_month_end.to_csv(index=False).encode("utf-8"),
        file_name=CFG.rf_monthly_out,
        mime="text/csv",
    )

# ============================================================
# 10) 10-YEAR FUND STATISTICS (MONTHLY → ANNUALIZED)
# ============================================================

st.header("10) 10-Year Fund Statistics (Monthly → Annualized)")

stats, cov_ann, corr, audit_stats = compute_10y_analytics(panel)

st.subheader("10.1 Coverage (months available in 10-year window)")
coverage = stats[["Months_Available"]].sort_values("Months_Available", ascending=False)
st.dataframe(coverage, use_container_width=True)

min_months = st.slider("10.2 Minimum months required (filter funds)", 24, 120, 108)
stats_f = stats[stats["Months_Available"] >= min_months].copy()

if stats_f.empty:
    st.error("No funds meet the minimum months threshold. Lower the threshold or add more tickers.")
    st.stop()

keep = stats_f.index.tolist()
stats_f = stats_f.drop(columns=["Months_Available"])

cov_ann_f = cov_ann.loc[keep, keep]
corr_f = corr.loc[keep, keep]

st.subheader("10.3 Per-Fund Annual Statistics (computed from monthly data)")
st.dataframe(
    stats_f.style.format({
        "Geometric_Mean_Return_Annual": "{:.4%}",
        "Volatility_Annual": "{:.4%}",
        "Mean_LogReturn_Monthly": "{:.6f}",
        "Std_LogReturn_Monthly": "{:.6f}",
    }),
    use_container_width=True
)

st.subheader("10.4 Analytics Audit Notes")
st.code("\n".join([f"{k}: {v}" for k, v in audit_stats.items()]))

# ============================================================
# 11) MATRICES (ANNUALIZED COV; MONTHLY CORR)
# ============================================================

st.header("11) Matrices")

st.subheader("11.1 Annualized Variance–Covariance Matrix (monthly cov × 12)")
st.dataframe(cov_ann_f, use_container_width=True)

st.subheader("11.2 Correlation Matrix (monthly)")
st.dataframe(corr_f, use_container_width=True)

# ============================================================
# 12) PORTFOLIOS (RETURN + VOLATILITY)
# ============================================================

st.header("12) Portfolio Return & Volatility")

st.markdown(
    """
**Input format:** one portfolio per line:

- `PORT1: VFINX=0.60, FCNTX=0.40`
- `PORT2: VFINX=0.50, FCNTX=0.50`

**Audit note:** Portfolio mean return is estimated as a weighted sum of fund-level
annual geometric mean returns (computed from monthly data). Portfolio volatility
uses the annualized covariance matrix.
"""
)

normalize_weights = st.checkbox("12.1 Normalize weights to sum to 1", value=True)

port_text = st.text_area(
    "12.2 Portfolio definitions",
    value="PORT1: VFINX=0.60, FCNTX=0.40\nPORT2: VFINX=0.50, FCNTX=0.50",
    height=140,
)

ports = parse_portfolios(port_text)
if not ports:
    st.info("Add portfolio definitions above to compute portfolio statistics.")
else:
    # Use filtered tickers set for portfolio computations
    port_df = compute_portfolios(stats_f, cov_ann_f, ports, normalize_weights)

    st.subheader("12.3 Portfolio Results")
    st.dataframe(
        port_df.style.format({
            "Return_Annual_Geometric_Est": "{:.4%}",
            "Volatility_Annual": "{:.4%}",
            "Weights_Sum": "{:.4f}",
        }),
        use_container_width=True
    )
