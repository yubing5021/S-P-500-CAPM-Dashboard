"""
Mutual Fund Scrapper (Monthly, 2010 → Present) — Audit-Ready

PURPOSE
-------
This Streamlit app downloads monthly mutual fund (or any Yahoo Finance symbol)
price history, computes monthly log returns, and merges a risk-free proxy based on
the 30-year US Treasury constant maturity yield (FRED: DGS30).

WHY THIS VERSION
----------------
- Streamlit Cloud often runs Python 3.13; pandas_datareader currently breaks due
  to distutils removal. Therefore, this script uses the official FRED CSV endpoint.
- Adds audit annotations and quality checks for reproducibility and reviewability.

DATA SOURCES
------------
1) Prices: Yahoo Finance via yfinance (monthly interval, Adj Close)
2) Risk-Free: FRED DGS30 via https://fred.stlouisfed.org/graph/fredgraph.csv

ASSUMPTIONS (AUDIT-RELEVANT)
---------------------------
- Monthly log return for fund: log(AdjClose_t / AdjClose_{t-1})
- DGS30 is an annualized yield (%) at daily frequency.
  We approximate monthly risk-free log return as log(1 + (DGS30/100)/12).
  This is a standard simple-to-period conversion, NOT a bond total return series.
- Month-end alignment:
  - Fund prices are coerced to month-end timestamp.
  - DGS30 is sampled at month-end (last available daily observation in month).

OUTPUTS
-------
- Panel CSV (tidy/long): Date, Ticker, Fund_Name, Adj_Close, Log_Return, RF_Log_Return
- Download button provides the panel as CSV.

REQUIREMENTS (put in requirements.txt at repo root)
---------------------------------------------------
streamlit
pandas
numpy
yfinance
requests
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import streamlit as st


# ============================================================
# 1) CONFIG (AUDIT: centralized parameters)
# ============================================================

@dataclass(frozen=True)
class Config:
    # Default date range
    start_date: str = "2010-01-01"
    end_date: str = datetime.today().strftime("%Y-%m-%d")

    # Yahoo interval
    interval: str = "1mo"

    # FRED series for 30Y Treasury yield (% annualized)
    rf_series: str = "DGS30"

    # Cloud-safe output folder (relative to repo/app working dir)
    output_dir: Path = Path("outputs")

    # Output file name (written into output_dir if possible)
    panel_out: str = "mutual_fund_panel_monthly.csv"

    # Basic request settings
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
    """
    Audit note: Yahoo uses '-' for class tickers (e.g., BRK-B).
    """
    return str(symbol).strip().replace(".", "-")


def parse_tickers(raw: str) -> List[str]:
    parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
    parts = [p for p in parts if p]
    return [normalize_symbol_for_yahoo(p) for p in parts]


def enforce_month_end(dt: pd.Series) -> pd.Series:
    """
    Force timestamps to month-end for stable merges and grouping.
    """
    return pd.to_datetime(dt).dt.to_period("M").dt.to_timestamp("M")


def monthly_log_return_from_price(price: pd.Series) -> pd.Series:
    return np.log(price / price.shift(1))


# ============================================================
# 3) FRED (NO pandas_datareader; Python 3.13 compatible)
# ============================================================

@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_fred_series_csv(series_id: str, start_date: str, end_date: str,
                          timeout_s: int, user_agent: str) -> pd.DataFrame:
    """
    Fetch a FRED series through the official CSV endpoint.

    Audit notes:
    - Uses "fredgraph.csv" which is stable and widely used.
    - Converts values to numeric; '.' or missing values become NaN.
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    params = {"id": series_id, "cosd": start_date, "coed": end_date}
    headers = {"User-Agent": user_agent}

    resp = requests.get(url, params=params, headers=headers, timeout=timeout_s)
    resp.raise_for_status()

    # Read CSV from text for reliability
    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))

    # Expected schema: DATE, <series_id>
    if "DATE" not in df.columns or series_id not in df.columns:
        raise ValueError(f"Unexpected FRED CSV schema. Columns: {list(df.columns)}")

    df["Date"] = pd.to_datetime(df["DATE"])
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")

    out = df[["Date", series_id]].dropna().sort_values("Date")
    return out


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_rf_dgs30_monthly(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Convert DGS30 daily yield (%) to month-end and then to monthly log risk-free.

    Audit assumption:
    RF_Log_Return = log(1 + (DGS30/100)/12)

    This is an approximation. It is NOT a bond total return index.
    """
    rf_daily = fetch_fred_series_csv(
        series_id=CFG.rf_series,
        start_date=start_date,
        end_date=end_date,
        timeout_s=CFG.http_timeout_s,
        user_agent=CFG.user_agent,
    )

    rf_daily = rf_daily.set_index("Date").sort_index()
    rf_m = rf_daily.resample("M").last().reset_index()

    rf_m["RF_Log_Return"] = np.log1p((rf_m[CFG.rf_series] / 100.0) / 12.0)
    rf_m["Date"] = enforce_month_end(rf_m["Date"])

    return rf_m[["Date", "RF_Log_Return"]].dropna().sort_values("Date")


# ============================================================
# 4) YFINANCE (MONTHLY PRICES)
# ============================================================

@st.cache_data(show_spinner=False, ttl=60 * 60)
def safe_yf_fund_name(ticker: str) -> Optional[str]:
    """
    Best-effort metadata lookup.
    Audit note: info() can be slow / rate-limited; cached for stability.
    """
    try:
        info = yf.Ticker(ticker).info or {}
        return info.get("longName") or info.get("shortName") or None
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=60 * 60)
def download_monthly_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Download monthly price history and return tidy long format.

    Output schema: Date, Ticker, Adj_Close

    Audit notes:
    - Uses Adj Close (dividend/split adjusted) if available.
    - Ensures month-end timestamps for stable merges.
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
        # Single ticker returns a flat column index
        out = px.reset_index()
        out["Ticker"] = tickers[0]

    if "Adj Close" not in out.columns:
        raise ValueError(
            "Yahoo Finance returned data without 'Adj Close'. "
            "This can happen for some instruments or schema changes."
        )

    out = out[["Date", "Ticker", "Adj Close"]].dropna()
    out = out.rename(columns={"Adj Close": "Adj_Close"})
    out["Ticker"] = out["Ticker"].astype(str).map(normalize_symbol_for_yahoo)
    out["Date"] = enforce_month_end(out["Date"])

    out = out.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return out


# ============================================================
# 5) PANEL BUILD (QUALITY CHECKS)
# ============================================================

def build_panel(tickers: List[str], start_date: str, end_date: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    audit: Dict[str, str] = {}

    audit_kv(audit, "run_timestamp_local", now_iso())
    audit_kv(audit, "start_date", start_date)
    audit_kv(audit, "end_date", end_date)
    audit_kv(audit, "yahoo_interval", CFG.interval)
    audit_kv(audit, "fred_series", CFG.rf_series)
    audit_kv(audit, "rf_conversion", "RF_Log_Return = log(1 + (DGS30/100)/12)")
    audit_kv(audit, "month_end_alignment", "prices month-end; DGS30 resample M last()")

    # ---- Prices
    prices = download_monthly_prices(tickers, start_date, end_date)
    audit_kv(audit, "prices_rows_raw", len(prices))
    audit_kv(audit, "prices_unique_tickers", prices["Ticker"].nunique() if not prices.empty else 0)

    if prices.empty:
        raise ValueError("No price data returned from Yahoo Finance for the provided tickers/date range.")

    # Compute log returns (drop first month per ticker)
    prices["Log_Return"] = prices.groupby("Ticker")["Adj_Close"].transform(monthly_log_return_from_price)
    before_drop = len(prices)
    prices = prices.dropna(subset=["Log_Return"]).copy()
    audit_kv(audit, "prices_rows_after_return_drop", len(prices))
    audit_kv(audit, "prices_rows_dropped_for_first_obs", before_drop - len(prices))

    # Fund names (best effort)
    meta = pd.DataFrame({"Ticker": tickers, "Fund_Name": [safe_yf_fund_name(t) for t in tickers]})
    prices = prices.merge(meta, on="Ticker", how="left")

    # ---- Risk-free
    rf = fetch_rf_dgs30_monthly(start_date, end_date)
    audit_kv(audit, "rf_rows_monthly", len(rf))

    if rf.empty:
        # Audit note: fail hard; RF is a core series
        raise ValueError("Risk-free series (DGS30) returned empty after processing.")

    # ---- Merge
    panel = prices.merge(rf, on="Date", how="left")
    audit_kv(audit, "panel_rows_pre_dropna", len(panel))

    # RF must exist for each panel row
    missing_rf = panel["RF_Log_Return"].isna().sum()
    audit_kv(audit, "panel_missing_rf_rows", missing_rf)

    panel = panel.dropna(subset=["RF_Log_Return"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    audit_kv(audit, "panel_rows_final", len(panel))

    # Quality checks
    # 1) Date monotonicity within ticker
    mono_viol = 0
    for t, g in panel.groupby("Ticker"):
        if not g["Date"].is_monotonic_increasing:
            mono_viol += 1
    audit_kv(audit, "monotonic_date_violations_tickers", mono_viol)

    # 2) Coverage summary (earliest & latest per ticker)
    cov = panel.groupby("Ticker")["Date"].agg(["min", "max", "count"]).reset_index()
    audit_kv(audit, "coverage_table_preview", cov.head(10).to_string(index=False))

    # Final column order (audit: stable schema)
    panel = panel[["Date", "Ticker", "Fund_Name", "Adj_Close", "Log_Return", "RF_Log_Return"]]

    return panel, audit


def try_write_csv(panel: pd.DataFrame) -> Tuple[bool, Optional[Path], Optional[str]]:
    """
    Attempt to write CSV to outputs/ folder. On Streamlit Cloud, filesystem writes are allowed
    but should be treated as ephemeral. We still do it for traceability when possible.
    """
    try:
        CFG.output_dir.mkdir(parents=True, exist_ok=True)
        path = CFG.output_dir / CFG.panel_out
        panel.to_csv(path, index=False)
        return True, path, None
    except Exception as e:
        return False, None, str(e)


# ============================================================
# 6) STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Mutual Fund Monthly Scraper", layout="wide")
st.title("Mutual Fund Monthly Scraper (Monthly, 2010 → Present)")

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

    run_btn = st.button("Pull Data")


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

    # Try filesystem write (optional; download always provided)
    ok, path, err = try_write_csv(panel)

    st.success("Panel built successfully.")
    if ok and path is not None:
        st.caption(f"Local CSV written to: {path} (ephemeral on Streamlit Cloud)")
    elif err:
        st.warning(f"Could not write CSV to disk (download still available). Details: {err}")

    # Preview
    st.subheader("Preview")
    st.dataframe(panel.head(25), use_container_width=True)

    # Audit block
    st.subheader("Audit Notes")
    st.code("\n".join([f"{k}: {v}" for k, v in audit.items()]))

    # Download
    st.download_button(
        "Download panel CSV",
        data=panel.to_csv(index=False).encode("utf-8"),
        file_name=CFG.panel_out,
        mime="text/csv",
    )
