"""
mutual_fund_monthly_scraper.py

Monthly mutual fund downloader + risk-free series (30Y Treasury, FRED DGS30).

Key features
- User enters fund tickers in a sidebar "popup-style" text box
- Pulls monthly adjusted prices from Yahoo Finance (interval=1mo)
- Computes monthly log returns
- Pulls 30-year Treasury yield from FRED (DGS30) and converts to monthly log RF
- Exports a tidy panel CSV

Notes
- Many mutual funds have limited history on Yahoo; if a ticker has <2010 data, you'll get whatever is available.
- DGS30 is an annualized yield (%). We approximate monthly log RF as log(1 + y/12).
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

import streamlit as st


# ============================================================
# 1) CONFIG
# ============================================================

@dataclass(frozen=True)
class Config:
    start_date: str = "2010-01-01"
    end_date: str = datetime.today().strftime("%Y-%m-%d")

    # Yahoo monthly interval
    interval: str = "1mo"

    # Risk-free: 30Y Treasury constant maturity rate (annualized %)
    rf_series: str = "DGS30"

    # Output directory + files
    output_dir: Path = Path.home() / "OneDrive" / "Desktop" / "MutualFundOutputs"
    panel_out: str = "mutual_fund_panel_monthly.csv"
    rf_cache: str = "rf_dgs30_monthly.csv"


CFG = Config()


# ============================================================
# 2) HELPERS
# ============================================================

def normalize_symbol_for_yahoo(symbol: str) -> str:
    # Keep similar behavior as your S&P script: BRK.B -> BRK-B, etc.
    return str(symbol).strip().replace(".", "-")


def monthly_log_return_from_price(price: pd.Series) -> pd.Series:
    return np.log(price / price.shift(1))


def fetch_rf_dgs30_monthly(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Pull daily DGS30 from FRED, convert to month-end, then to monthly log return approx.
    DGS30 is annualized yield in percent.
    """
    rf_raw = pdr.DataReader(CFG.rf_series, "fred", start_date, end_date)

    # Month-end sampling
    rf_m = rf_raw.resample("M").last().reset_index()
    rf_m = rf_m.rename(columns={"DATE": "Date"}) if "DATE" in rf_m.columns else rf_m

    # Convert annualized % yield to monthly log RF approximation
    # monthly simple rate ≈ (y/100)/12
    rf_m["RF_Log_Return"] = np.log1p((rf_m[CFG.rf_series] / 100.0) / 12.0)

    return rf_m[["Date", "RF_Log_Return"]].dropna().sort_values("Date")


def safe_yf_fund_name(ticker: str) -> Optional[str]:
    try:
        info = yf.Ticker(ticker).info or {}
        return info.get("longName") or info.get("shortName") or None
    except Exception:
        return None


def download_monthly_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Returns long/tidy: Date, Ticker, Adj Close
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

    # Handle 1 ticker vs many tickers
    if isinstance(px.columns, pd.MultiIndex):
        out = (
            px.stack(level=0)
              .rename_axis(["Date", "Ticker"])
              .reset_index()
        )
    else:
        # single ticker: columns are OHLC etc
        out = px.reset_index()
        out["Ticker"] = tickers[0]

    # Normalize schema
    if "Adj Close" not in out.columns:
        raise ValueError("Yahoo Finance returned data without 'Adj Close' column (schema changed?)")

    out = out[["Date", "Ticker", "Adj Close"]].dropna()
    out["Ticker"] = out["Ticker"].astype(str).map(normalize_symbol_for_yahoo)

    # Ensure month-end labels (Yahoo monthly is usually month-end, but enforce)
    out["Date"] = pd.to_datetime(out["Date"]).dt.to_period("M").dt.to_timestamp("M")

    return out.sort_values(["Ticker", "Date"]).reset_index(drop=True)


# ============================================================
# 3) STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Mutual Fund Monthly Scraper", layout="wide")
st.title("Mutual Fund Monthly Scraper (2010 → Present)")

with st.sidebar:
    st.header("Inputs")
    tickers_text = st.text_area(
        "Enter mutual fund tickers (comma or newline separated)",
        value="VFINX, FCNTX",
        height=120,
        help="Example: VFINX, FCNTX, PRGFX (availability depends on Yahoo Finance).",
    )
    start_date = st.text_input("Start date (YYYY-MM-DD)", CFG.start_date)
    end_date = st.text_input("End date (YYYY-MM-DD)", CFG.end_date)

    run_btn = st.button("Pull Data")


def parse_tickers(raw: str) -> List[str]:
    parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
    parts = [p for p in parts if p]
    return [normalize_symbol_for_yahoo(p) for p in parts]


if run_btn:
    tickers = parse_tickers(tickers_text)
    if not tickers:
        st.error("Please enter at least one ticker.")
        st.stop()

    st.write(f"**Tickers:** {', '.join(tickers)}")
    st.write(f"**Date range:** {start_date} → {end_date}")
    st.write(f"**Risk-free:** FRED {CFG.rf_series} (30Y), month-end sampling")

    # Output dir
    out_dir = CFG.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = out_dir / CFG.panel_out
    rf_cache_path = out_dir / CFG.rf_cache

    # ----------------------------
    # A) Prices + returns
    # ----------------------------
    st.info("Downloading monthly prices from Yahoo Finance…")
    prices = download_monthly_prices(tickers, start_date, end_date)
    if prices.empty:
        st.error("No price data returned from Yahoo Finance for those tickers.")
        st.stop()

    prices["Log_Return"] = prices.groupby("Ticker")["Adj Close"].transform(monthly_log_return_from_price)
    prices = prices.dropna(subset=["Log_Return"])

    # Add fund names (best effort)
    meta = pd.DataFrame({
        "Ticker": tickers,
        "Fund_Name": [safe_yf_fund_name(t) for t in tickers],
    })
    prices = prices.merge(meta, on="Ticker", how="left")

    # ----------------------------
    # B) Risk-free (DGS30)
    # ----------------------------
    st.info("Pulling 30Y Treasury (DGS30) from FRED and converting to monthly log RF…")
    rf = None
    last_err = None
    for attempt in range(1, 4):
        try:
            rf = fetch_rf_dgs30_monthly(start_date, end_date)
            if rf is not None and not rf.empty:
                rf.to_csv(rf_cache_path, index=False)
                break
        except Exception as e:
            last_err = e

    if rf is None or rf.empty:
        if rf_cache_path.exists():
            st.warning("FRED failed; using cached RF series.")
            rf = pd.read_csv(rf_cache_path, parse_dates=["Date"]).sort_values("Date")
        else:
            st.warning(f"FRED failed and no cache found; defaulting RF_Log_Return to 0. Last error: {last_err}")
            rf = prices[["Date"]].drop_duplicates().copy()
            rf["RF_Log_Return"] = 0.0

    # ----------------------------
    # C) Merge to panel
    # ----------------------------
    panel = prices.merge(rf, on="Date", how="left")
    panel = panel.dropna(subset=["RF_Log_Return"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # ----------------------------
    # D) Export + preview
    # ----------------------------
    panel.to_csv(panel_path, index=False)

    st.success(f"Saved panel CSV to: {panel_path}")
    st.subheader("Preview")
    st.dataframe(panel.head(25), use_container_width=True)

    st.download_button(
        "Download panel CSV",
        data=panel.to_csv(index=False).encode("utf-8"),
        file_name=CFG.panel_out,
        mime="text/csv",
    )
