"""
Mutual Fund Scrapper (Monthly, 2010 → Present) — Audit-Ready + Analytics (10Y)

WHAT THIS APP DOES
------------------
1) Downloads monthly mutual fund/ETF prices from Yahoo Finance (Adj Close).
2) Computes monthly log returns per fund.
3) Downloads risk-free proxy using FRED DGS30 (30Y Treasury yield).
4) Converts DGS30 to a monthly log risk-free return approximation.
5) Builds a tidy panel and stores it in Streamlit session_state (robust across reruns).
6) Computes 10-year (most recent) statistics:
   - geometric mean return (annualized from monthly log returns)
   - annualized volatility (monthly std * sqrt(12))
   - variance-covariance matrix (annualized = monthly cov * 12)
   - correlation matrix
7) Computes portfolio return/vol for user-provided weight definitions.

DEPLOYMENT NOTE (IMPORTANT)
---------------------------
Streamlit Cloud frequently runs Python 3.13+. pandas_datareader can fail under
Python 3.12+ due to distutils removal. Therefore, FRED is pulled via the official
CSV endpoint using requests + pandas.

AUDIT-RELEVANT ASSUMPTIONS
--------------------------
- Monthly fund log return: r_t = log(AdjClose_t / AdjClose_{t-1})
- DGS30 is an annualized yield (%) at daily frequency. We approximate monthly RF log
  return as: RF_Log_Return = log(1 + (DGS30/100)/12).
  This is a RATE proxy, not a bond total return index.
- Annualization:
  * Geometric mean annual return = exp(12 * mean(monthly_log_returns)) - 1
  * Annualized volatility = std(monthly_log_returns) * sqrt(12)
  * Annualized covariance = cov(monthly_log_returns) * 12
- 10-year window: last available month in the downloaded panel minus 120 months.

REQUIREMENTS (repo root requirements.txt)
-----------------------------------------
streamlit
pandas
numpy
yfinance
requests
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
    start_date: str = "2010-01-01"
    end_date: str = datetime.today().strftime("%Y-%m-%d")

    interval: str = "1mo"      # Yahoo monthly sampling
    rf_series: str = "DGS30"   # FRED 30Y constant maturity yield (%)

    # On Streamlit Cloud, filesystem is ephemeral. We write if possible for traceability,
    # but the download button is the primary export mechanism.
    output_dir: Path = Path("outputs")
    panel_out: str = "mutual_fund_panel_monthly.csv"

    http_timeout_s: int = 20
    user_agent: str = "Mozilla/5.0 (compatible; MutualFundScraper/1.0)"


CFG = Config()


# ============================================================
# 2) AUDIT UTILITIES
# ============================================================

def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def audit_kv(audit: Dict[str, str], k: str, v: object) -> None:
    audit[k] = str(v)


def normalize_symbol_for_yahoo(symbol: str) -> str:
    # Yahoo uses '-' for certain share classes; keep this normalization consistent.
    return str(symbol).strip().replace(".", "-")


def parse_tickers(raw: str) -> List[str]:
    parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
    parts = [p for p in parts if p]
    return [normalize_symbol_for_yahoo(p) for p in parts]


def enforce_month_end(dt: pd.Series) -> pd.Series:
    # Enforce month-end timestamps to avoid merge/key mismatches.
    return pd.to_datetime(dt).dt.to_period("M").dt.to_timestamp("M")


def monthly_log_return_from_price(price: pd.Series) -> pd.Series:
    return np.log(price / price.shift(1))


# ============================================================
# 3) FRED (Python 3.13 compatible; schema-robust)
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
    Fetch a FRED series via the official CSV endpoint.

    Robustness fix:
    - FRED date column can vary: 'DATE' or 'observation_date' (or 'date').
      We accept all to prevent schema breakage.

    Returns:
    - dataframe with columns: Date, <series_id>
    - audit dict with schema details
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

    out = df[["Date", series_id]].dropna().sort_values("Date")
    audit_kv(audit, "fred_rows_after_dropna", len(out))

    return out, audit


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_rf_dgs30_monthly(start_date: str, end_date: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Convert DGS30 daily yield (%) → month-end → monthly log risk-free return proxy.

    RF_Log_Return = log(1 + (DGS30/100)/12)

    Audit note:
    This is a proxy series. It approximates a monthly continuously-compounded risk-free
    return derived from an annualized yield.
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

    rf_daily = rf_daily.set_index("Date").sort_index()
    rf_m = rf_daily.resample("M").last().reset_index()

    rf_m["RF_Log_Return"] = np.log1p((rf_m[CFG.rf_series] / 100.0) / 12.0)
    rf_m["Date"] = enforce_month_end(rf_m["Date"])

    rf_out = rf_m[["Date", "RF_Log_Return"]].dropna().sort_values("Date")

    audit_kv(audit_rf, "rf_rows_monthly", len(rf_out))
    audit_kv(audit_rf, "rf_conversion", "RF_Log_Return = log(1 + (DGS30/100)/12)")

    return rf_out, audit_rf


# ============================================================
# 4) YFINANCE (MONTHLY PRICES)
# ============================================================

@st.cache_data(show_spinner=False, ttl=60 * 60)
def safe_yf_fund_name(ticker: str) -> Optional[str]:
    """
    Best-effort metadata lookup from Yahoo.
    Cached for stability; Yahoo metadata can be rate-limited.
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

    if isinstance(px.columns, pd.MultiIndex):
        out = (
            px.stack(level=0)
              .rename_axis(["Date", "Ticker"])
              .reset_index()
        )
    else:
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
# 5) PANEL BUILD (AUDIT + QUALITY CHECKS)
# ============================================================

def build_panel(tickers: List[str], start_date: str, end_date: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    audit: Dict[str, str] = {}

    audit_kv(audit, "run_timestamp_local", now_iso())
    audit_kv(audit, "tickers_input", ", ".join(tickers))
    audit_kv(audit, "start_date", start_date)
    audit_kv(audit, "end_date", end_date)
    audit_kv(audit, "yahoo_interval", CFG.interval)
    audit_kv(audit, "fred_series", CFG.rf_series)
    audit_kv(audit, "month_end_alignment", "prices coerced to month-end; FRED resample('M').last()")

    # Prices
    prices = download_monthly_prices(tickers, start_date, end_date)
    audit_kv(audit, "prices_rows_raw", len(prices))
    audit_kv(audit, "prices_unique_tickers", prices["Ticker"].nunique() if not prices.empty else 0)

    if prices.empty:
        raise ValueError("No price data returned from Yahoo Finance for the tickers/date range.")

    # Returns (log)
    prices["Log_Return"] = prices.groupby("Ticker")["Adj_Close"].transform(monthly_log_return_from_price)
    before = len(prices)
    prices = prices.dropna(subset=["Log_Return"]).copy()
    audit_kv(audit, "prices_rows_after_return_drop", len(prices))
    audit_kv(audit, "prices_rows_dropped_first_obs", before - len(prices))

    # Fund names (best effort)
    meta = pd.DataFrame({"Ticker": tickers, "Fund_Name": [safe_yf_fund_name(t) for t in tickers]})
    prices = prices.merge(meta, on="Ticker", how="left")

    # Risk-free
    rf, rf_audit = fetch_rf_dgs30_monthly(start_date, end_date)
    for k, v in rf_audit.items():
        audit_kv(audit, f"rf_{k}", v)

    if rf.empty:
        raise ValueError("Risk-free series returned empty after processing.")

    # Merge
    panel = prices.merge(rf, on="Date", how="left")
    audit_kv(audit, "panel_rows_pre_dropna", len(panel))
    audit_kv(audit, "panel_missing_rf_rows", int(panel["RF_Log_Return"].isna().sum()))

    panel = panel.dropna(subset=["RF_Log_Return"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    audit_kv(audit, "panel_rows_final", len(panel))

    # Date monotonicity check
    violations = 0
    for t, g in panel.groupby("Ticker"):
        if not g["Date"].is_monotonic_increasing:
            violations += 1
    audit_kv(audit, "monotonic_date_violations_tickers", violations)

    # Coverage summary
    cov = panel.groupby("Ticker")["Date"].agg(["min", "max", "count"]).reset_index()
    audit_kv(audit, "coverage_preview", cov.head(10).to_string(index=False))

    panel = panel[["Date", "Ticker", "Fund_Name", "Adj_Close", "Log_Return", "RF_Log_Return"]]
    return panel, audit


def try_write_csv(panel: pd.DataFrame) -> Tuple[bool, Optional[Path], Optional[str]]:
    try:
        CFG.output_dir.mkdir(parents=True, exist_ok=True)
        path = CFG.output_dir / CFG.panel_out
        panel.to_csv(path, index=False)
        return True, path, None
    except Exception as e:
        return False, None, str(e)


# ============================================================
# 6) ANALYTICS (10Y stats + cov/corr + portfolios)
#    NOTE: This function requires a built panel and is called
#    only when panel exists in session_state.
# ============================================================

def compute_10y_analytics(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    Returns:
    - stats: per-fund annual geometric mean and annualized vol (computed from monthly data)
    - cov_ann: annualized covariance matrix (monthly cov * 12)
    - corr: correlation matrix (monthly)
    - R_10y: the 10y monthly log return matrix used
    - audit_stats: audit dictionary describing windowing + missing policy
    """
    audit_stats: Dict[str, str] = {}

    panel_stats = panel.copy()
    panel_stats["Date"] = pd.to_datetime(panel_stats["Date"])

    last_dt = panel_stats["Date"].max()
    ten_years_ago = (last_dt.to_period("M") - 120).to_timestamp("M")  # 120 months
    audit_kv(audit_stats, "window_last_date", last_dt.date())
    audit_kv(audit_stats, "window_start_date", ten_years_ago.date())
    audit_kv(audit_stats, "window_months_target", 120)

    panel_10y = panel_stats[panel_stats["Date"] > ten_years_ago].copy()

    R = (
        panel_10y.pivot_table(index="Date", columns="Ticker", values="Log_Return", aggfunc="mean")
        .sort_index()
    )
    audit_kv(audit_stats, "returns_matrix_rows", R.shape[0])
    audit_kv(audit_stats, "returns_matrix_cols", R.shape[1])

    # Per-fund stats from monthly log returns:
    mu_m_log = R.mean(skipna=True)
    sig_m = R.std(ddof=1, skipna=True)

    geo_ann = np.expm1(12.0 * mu_m_log)       # annual geometric mean return
    sig_ann = sig_m * math.sqrt(12.0)         # annualized volatility

    stats = pd.DataFrame({
        "Geometric_Mean_Return_Annual": geo_ann,
        "Volatility_Annual": sig_ann,
        "Mean_LogReturn_Monthly": mu_m_log,
        "Std_LogReturn_Monthly": sig_m,
        "Months_Available": R.notna().sum(),
    }).sort_index()

    # Covariance/correlation from monthly data then annualize covariance
    cov_m = R.cov()          # pairwise by default
    cov_ann = cov_m * 12.0
    corr = R.corr()

    audit_kv(audit_stats, "annualization_rule_return", "exp(12*mean(log_r_m)) - 1")
    audit_kv(audit_stats, "annualization_rule_vol", "std(log_r_m)*sqrt(12)")
    audit_kv(audit_stats, "annualization_rule_cov", "cov_m*12")

    return stats, cov_ann, corr, R, audit_stats


def parse_portfolios(text: str) -> Dict[str, Dict[str, float]]:
    """
    Parse portfolios from text area.

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
            name, rhs = f"PORT{len(ports)+1}", line

        weights: Dict[str, float] = {}
        for part in rhs.split(","):
            part = part.strip()
            if not part or "=" not in part:
                continue
            t, w = part.split("=", 1)
            t = t.strip()
            weights[t] = float(w.strip())
        if weights:
            ports[name] = weights
    return ports


def compute_portfolios(
    stats: pd.DataFrame,
    cov_ann: pd.DataFrame,
    portfolios: Dict[str, Dict[str, float]],
    normalize_weights: bool,
) -> pd.DataFrame:
    """
    Portfolio analytics using annualized covariance and annual geometric mean estimates.

    Audit note:
    - Portfolio mean return is estimated as w' * fund_geo_ann
      (fund_geo_ann computed from monthly log returns).
    - Portfolio vol is sqrt(w' * Cov_ann * w).
    """
    tickers_used = list(stats.index)
    covA = cov_ann.loc[tickers_used, tickers_used]
    muA = stats["Geometric_Mean_Return_Annual"]

    rows = []
    for pname, wdict in portfolios.items():
        w = pd.Series(0.0, index=tickers_used)

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

        rp = float((w * muA).sum())
        var_p = float(w.values.T @ covA.values @ w.values)
        vol_p = math.sqrt(var_p) if var_p >= 0 else float("nan")

        rows.append({
            "Portfolio": pname,
            "Return_Annual_Geometric_Est": rp,
            "Volatility_Annual": vol_p,
            "Weights_Sum": float(w.sum()),
            "Num_Funds_Used": int((w != 0).sum()),
            "Ignored_Tickers": ", ".join(ignored) if ignored else "",
        })

    out = pd.DataFrame(rows).set_index("Portfolio")
    return out


# ============================================================
# 7) STREAMLIT UI (ROBUST: session_state as the single source of truth)
# ============================================================

st.set_page_config(page_title="Mutual Fund Monthly Scraper", layout="wide")
st.title("Mutual Fund Monthly Scraper (Monthly, 2010 → Present)")

# --- Initialize session_state keys (audit: deterministic app state)
if "panel" not in st.session_state:
    st.session_state["panel"] = None
if "audit_panel" not in st.session_state:
    st.session_state["audit_panel"] = None

with st.sidebar:
    st.header("Inputs")
    tickers_text = st.text_area(
        "Enter tickers (comma or newline separated)",
        value="VFINX, FCNTX",
        height=120,
        help="Works for many mutual funds and ETFs supported by Yahoo Finance.",
    )
    start_date = st.text_input("Start date (YYYY-MM-DD)", CFG.start_date)
    end_date = st.text_input("End date (YYYY-MM-DD)", CFG.end_date)

    colA, colB = st.columns(2)
    with colA:
        run_btn = st.button("Pull Data", use_container_width=True)
    with colB:
        clear_btn = st.button("Clear Results", use_container_width=True)

if clear_btn:
    st.session_state["panel"] = None
    st.session_state["audit_panel"] = None
    st.success("Cleared panel + audit state.")

if run_btn:
    tickers = parse_tickers(tickers_text)
    if not tickers:
        st.error("Please enter at least one ticker.")
        st.stop()

    st.write(f"**Tickers:** {', '.join(tickers)}")
    st.write(f"**Date range:** {start_date} → {end_date}")
    st.write(f"**Risk-free:** FRED {CFG.rf_series} (30Y), month-end sampling + monthly conversion")

    with st.spinner("Building panel (Yahoo Finance + FRED)…"):
        try:
            panel, audit = build_panel(tickers, start_date, end_date)
        except Exception as e:
            st.error("Failed to build panel.")
            st.exception(e)
            st.stop()

    # ROBUST STATE STORE: everything below reads from session_state (no NameError)
    st.session_state["panel"] = panel
    st.session_state["audit_panel"] = audit

    ok, path, err = try_write_csv(panel)
    st.success("Panel built successfully.")
    if ok and path is not None:
        st.caption(f"Local CSV written to: {path} (ephemeral on Streamlit Cloud)")
    elif err:
        st.warning(f"Could not write CSV to disk (download still available). Details: {err}")

# --- Read panel from state (single source of truth)
panel = st.session_state.get("panel", None)
audit_panel = st.session_state.get("audit_panel", None)

if panel is None:
    st.info("Click **Pull Data** to build the panel. Analytics will appear after a successful run.")
    st.stop()

# ============================================================
# 8) PANEL OUTPUTS (preview, audit, download)
# ============================================================

st.subheader("Panel Preview")
st.dataframe(panel.head(25), use_container_width=True)

st.subheader("Panel Audit Notes")
if audit_panel:
    st.code("\n".join([f"{k}: {v}" for k, v in audit_panel.items()]))
else:
    st.caption("No audit dictionary stored (unexpected).")

st.download_button(
    "Download panel CSV",
    data=panel.to_csv(index=False).encode("utf-8"),
    file_name=CFG.panel_out,
    mime="text/csv",
)

# ============================================================
# 9) 10-YEAR ANALYTICS (MONTHLY → ANNUALIZED)
# ============================================================

st.header("10-Year Fund Statistics (Monthly → Annualized)")

stats, cov_ann, corr, R_10y, audit_stats = compute_10y_analytics(panel)

# Coverage
st.subheader("Coverage (months available in the 10-year window)")
coverage = stats[["Months_Available"]].sort_values("Months_Available", ascending=False)
st.dataframe(coverage, use_container_width=True)

min_months = st.slider("Minimum months required (filter funds)", 24, 120, 108)
stats_f = stats[stats["Months_Available"] >= min_months].copy()

if stats_f.empty:
    st.error("No funds meet the minimum months threshold. Lower the threshold or add more tickers.")
    st.stop()

keep_tickers = stats_f.index.tolist()
stats_f = stats_f.drop(columns=["Months_Available"])

# Filter matrices to kept tickers
cov_ann_f = cov_ann.loc[keep_tickers, keep_tickers]
corr_f = corr.loc[keep_tickers, keep_tickers]

st.subheader("Per-Fund Annual Statistics (computed from monthly data)")
st.dataframe(
    stats_f.style.format({
        "Geometric_Mean_Return_Annual": "{:.4%}",
        "Volatility_Annual": "{:.4%}",
        "Mean_LogReturn_Monthly": "{:.6f}",
        "Std_LogReturn_Monthly": "{:.6f}",
    }),
    use_container_width=True
)

st.subheader("Audit Notes (10-year analytics)")
st.code("\n".join([f"{k}: {v}" for k, v in audit_stats.items()]))

st.subheader("Annualized Variance–Covariance Matrix (monthly cov × 12)")
st.dataframe(cov_ann_f, use_container_width=True)

st.subheader("Correlation Matrix (monthly)")
st.dataframe(corr_f, use_container_width=True)

# ============================================================
# 10) PORTFOLIOS
# ============================================================

st.header("Portfolio Return & Volatility")

st.markdown(
    """
**Input format:** one portfolio per line

- `PORT1: VFINX=0.60, FCNTX=0.40`
- `PORT2: VFINX=0.50, FCNTX=0.50`

**Audit note:** Portfolio return is estimated as a weighted sum of the **fund-level**
annual geometric mean returns computed from monthly data. Portfolio volatility uses
the annualized covariance matrix.
"""
)

normalize_weights = st.checkbox("Normalize weights to sum to 1", value=True)

port_text = st.text_area(
    "Portfolios (one per line)",
    value="PORT1: VFINX=0.60, FCNTX=0.40\nPORT2: VFINX=0.50, FCNTX=0.50",
    height=140,
)

ports = parse_portfolios(port_text)

if not ports:
    st.info("Add portfolio definitions above to compute portfolio stats.")
else:
    # Use the filtered tickers set for portfolio computations
    stats_use = stats_f.copy()
    cov_use = cov_ann_f.copy()

    port_df = compute_portfolios(stats_use, cov_use, ports, normalize_weights)

    st.subheader("Portfolio Results")
    st.dataframe(
        port_df.style.format({
            "Return_Annual_Geometric_Est": "{:.4%}",
            "Volatility_Annual": "{:.4%}",
            "Weights_Sum": "{:.4f}",
        }),
        use_container_width=True
    )
