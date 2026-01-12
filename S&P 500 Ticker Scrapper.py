r"""
S&P 500 Ticker Scrapper.py

End-to-end pipeline that:
  1) Loads S&P 500 constituents (robust multi-source)
  2) Downloads weekly price history for constituents (Yahoo Finance)
  3) Builds a weekly stock panel with:
       - Log_Return
       - Company_Name
       - Sector
       - Market_Cap
       - Sector_Weight (capped within sector)
       - Market_Log_Return (S&P 500 index weekly log return)
       - RF_Log_Return (3M T-bill weekly log return)
       - Weighted_Log_Return (stock return * sector weight)
  4) Aggregates sector cap-weighted returns
  5) Computes sector CAPM alpha/beta (static + rolling) vs market (excess returns)
  6) Writes outputs to:
       C:\Users\<you>\OneDrive\Desktop\S&P 500 Scrapper\sp500_outputs

AUDIT NOTES
- If you previously saw Saturday dates:
    pandas Period -> Timestamp conversion defaults to PERIOD START.
    For week anchored "W-FRI", the period start is Saturday.
  Fix implemented:
    I forced `to_timestamp(how="end")` so week labels are Friday (weekday=4).
- All series (stocks, market, risk-free) are aligned to the same Friday week label.

Run (PowerShell):
  cd "$env:USERPROFILE\OneDrive\Desktop\S&P 500 Scrapper"
  python "S&P 500 Ticker Scrapper.py"
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr


# ============================================================
# 1) CONFIG
# ============================================================

@dataclass(frozen=True)
class Config:
    start_date: str = "2014-01-01"
    end_date: str = datetime.today().strftime("%Y-%m-%d")

    # Yahoo interval:
    interval: str = "1wk"

    # All dates are labeled as week-ending Friday
    week_anchor: str = "W-FRI"

    # Sector weighting cap
    sector_weight_cap: float = 0.10

    # Sector rolling CAPM window
    rolling_window_weeks: int = 156  # ~3 years

    # Market and RF sources
    market_ticker: str = "^GSPC"
    rf_series: str = "DTB3"

    # Outputs
    output_dir: Path = Path.home() / "OneDrive" / "Desktop" / "S&P 500 Scrapper" / "sp500_outputs"

    # Constituents sources (try in order)
    constituents_urls: Tuple[str, ...] = (
        "https://datahub.io/core/s-and-p-500-companies/_r/-/data/constituents.csv",
        "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv",
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
    )
    wikipedia_url: str = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


CFG = Config()


# ============================================================
# 2) PATHS
# ============================================================

OUT_DIR = CFG.output_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

PANEL_OUT = OUT_DIR / "sp500_stock_panel.csv"
SECTOR_RETURNS_OUT = OUT_DIR / "sector_returns.csv"
SECTOR_CAPM_STATIC_OUT = OUT_DIR / "sector_capm_static.csv"
SECTOR_CAPM_ROLLING_OUT = OUT_DIR / "sector_capm_rolling.csv"
RF_CACHE_PATH = OUT_DIR / "rf_dtb3_weekly.csv"

print(f"[INFO] Outputs directory: {OUT_DIR}")


# ============================================================
# 3) HELPERS
# ============================================================

def normalize_symbol_for_yahoo(symbol: str) -> str:
    """BRK.B -> BRK-B, BF.B -> BF-B, etc."""
    return str(symbol).strip().replace(".", "-")


def to_week_ending_friday(dt: pd.Series) -> pd.Series:
    """
    AUDIT-CRITICAL:
    Ensure week labels are Friday (weekday=4).

    pandas default to_timestamp() labels period START.
    For W-FRI, period start is Saturday, which creates 'Saturday dates'.

    Fix: force period END.
    """
    return (
        pd.to_datetime(dt)
          .dt.to_period(CFG.week_anchor)
          .dt.to_timestamp(how="end")
          .dt.normalize()
    )


def weekly_log_return_from_price(price: pd.Series) -> pd.Series:
    return np.log(price / price.shift(1))


def capped_weights_from_caps(market_caps: pd.Series, cap: float) -> pd.Series:
    w = market_caps / market_caps.sum()
    w = w.clip(upper=cap)
    return w / w.sum()


def audit_warn_non_friday(label: str, dates: pd.Series) -> None:
    bad = pd.to_datetime(dates).dt.weekday != 4
    if bad.any():
        sample = pd.to_datetime(dates.loc[bad]).drop_duplicates().sort_values().head(10).tolist()
        print(f"[WARN] {label}: Non-Friday labels detected. Sample: {sample}")


def safe_yf_info(ticker: str) -> dict:
    """Best-effort metadata fetch (Company_Name, Sector, Market_Cap)."""
    try:
        info = yf.Ticker(ticker).info or {}
        company_name = info.get("longName") or info.get("shortName") or None
        return {
            "Ticker": ticker,
            "Company_Name": company_name,
            "Sector": info.get("sector", "Unknown"),
            "Market_Cap": info.get("marketCap", np.nan),
        }
    except Exception:
        return {"Ticker": ticker, "Company_Name": None, "Sector": "Unknown", "Market_Cap": np.nan}


def load_constituents() -> pd.DataFrame:
    attempts: List[Tuple[str, str]] = []

    for url in CFG.constituents_urls:
        try:
            df = pd.read_csv(url)
            if "Symbol" in df.columns and len(df) > 400:
                print(f"[INFO] Constituents loaded from CSV: {url}")
                return df
            attempts.append((url, "Loaded but missing 'Symbol' or too few rows"))
        except Exception as e:
            attempts.append((url, repr(e)))

    # Wikipedia fallback
    try:
        tables = pd.read_html(CFG.wikipedia_url)
        df = tables[0].copy()
        df = df.rename(columns={"Security": "Name", "GICS Sector": "Sector"})
        if "Symbol" in df.columns and len(df) > 400:
            print("[INFO] Constituents loaded from Wikipedia.")
            return df
        attempts.append((CFG.wikipedia_url, "Read HTML but missing 'Symbol' or too few rows"))
    except Exception as e:
        attempts.append((CFG.wikipedia_url, repr(e)))

    msg = "Failed to load S&P 500 constituents. Attempts:\n" + "\n".join(
        f"- {src}: {err}" for src, err in attempts
    )
    raise RuntimeError(msg)


def compute_sector_capm_static(sector_returns: pd.DataFrame, mkt_rf: pd.DataFrame) -> pd.DataFrame:
    df = sector_returns.merge(mkt_rf, on="Date", how="inner").dropna()
    out_rows = []

    for sector, g in df.groupby("Sector"):
        ex_s = g["Sector_Log_Return"] - g["RF_Log_Return"]
        ex_m = g["Market_Log_Return"] - g["RF_Log_Return"]

        vx = np.var(ex_m)
        if vx == 0 or np.isnan(vx) or len(ex_m) < 52:
            out_rows.append({"Sector": sector, "Obs": int(len(ex_m)), "Alpha_Weekly": np.nan, "Beta": np.nan})
            continue

        beta = np.cov(ex_s, ex_m, ddof=0)[0, 1] / vx
        alpha = float(ex_s.mean() - beta * ex_m.mean())

        out_rows.append({"Sector": sector, "Obs": int(len(ex_m)), "Alpha_Weekly": float(alpha), "Beta": float(beta)})

    out = pd.DataFrame(out_rows).sort_values("Sector").reset_index(drop=True)
    out["Alpha_Annualized_(log_approx)"] = out["Alpha_Weekly"] * 52.0
    return out


def compute_sector_capm_rolling(sector_returns: pd.DataFrame, mkt_rf: pd.DataFrame, window: int) -> pd.DataFrame:
    df = sector_returns.merge(mkt_rf, on="Date", how="inner").dropna()
    rows = []

    for sector, g in df.groupby("Sector"):
        g = g.sort_values("Date").reset_index(drop=True)
        ex_s = (g["Sector_Log_Return"] - g["RF_Log_Return"]).to_numpy()
        ex_m = (g["Market_Log_Return"] - g["RF_Log_Return"]).to_numpy()
        dates = g["Date"].to_numpy()

        if len(g) < window + 5:
            continue

        for i in range(window, len(g) + 1):
            y = ex_s[i - window:i]
            x = ex_m[i - window:i]

            vx = np.var(x)
            if vx == 0 or np.isnan(vx):
                continue

            beta = np.cov(y, x, ddof=0)[0, 1] / vx
            alpha = float(np.mean(y) - beta * np.mean(x))

            rows.append({
                "Date": dates[i - 1],
                "Sector": sector,
                "Rolling_Alpha_Weekly": float(alpha),
                "Rolling_Beta": float(beta),
                "Window_Obs": window,
            })

    if not rows:
        return pd.DataFrame(columns=["Date", "Sector", "Rolling_Alpha_Weekly", "Rolling_Beta", "Window_Obs"])
    return pd.DataFrame(rows).sort_values(["Sector", "Date"]).reset_index(drop=True)


def fetch_rf_from_fred(start_date: str, end_date: str) -> pd.DataFrame:
    rf_raw = pdr.DataReader(CFG.rf_series, "fred", start_date, end_date)

    rf_raw = rf_raw.resample(CFG.week_anchor).last().reset_index()
    rf_raw = rf_raw.rename(columns={"DATE": "Date"}) if "DATE" in rf_raw.columns else rf_raw

    rf_raw["Date"] = to_week_ending_friday(rf_raw["Date"])

    # Convert annualized percent to weekly log return approx
    rf_raw["RF_Log_Return"] = np.log1p((rf_raw[CFG.rf_series] / 100.0) / 52.0)
    return rf_raw[["Date", "RF_Log_Return"]].dropna().sort_values("Date")


# ============================================================
# 4) LOAD CONSTITUENTS
# ============================================================

constituents = load_constituents()
constituents["Symbol"] = constituents["Symbol"].astype(str).map(normalize_symbol_for_yahoo)

name_fallback: Dict[str, str] = {}
if "Name" in constituents.columns:
    tmp = constituents[["Symbol", "Name"]].dropna()
    name_fallback = dict(zip(tmp["Symbol"], tmp["Name"]))

tickers = sorted(constituents["Symbol"].dropna().unique().tolist())
print(f"[INFO] Constituents tickers loaded: {len(tickers)}")


# ============================================================
# 5) DOWNLOAD WEEKLY PRICES (CONSTITUENTS)
# ============================================================

print("[INFO] Downloading weekly prices (this can take a while)...")

prices_raw = yf.download(
    tickers,
    start=CFG.start_date,
    end=CFG.end_date,
    interval=CFG.interval,
    auto_adjust=False,
    group_by="ticker",
    threads=False,
    progress=True,
)

if prices_raw.empty:
    raise RuntimeError("Yahoo Finance returned empty data for constituents prices.")

prices = (
    prices_raw
    .stack(level=0)
    .rename_axis(["Date", "Ticker"])
    .reset_index()
)

# AUDIT: force Friday week labels
prices["Date"] = to_week_ending_friday(prices["Date"])
audit_warn_non_friday("Constituent prices", prices["Date"])

if "Adj Close" not in prices.columns:
    raise ValueError("Downloaded data missing 'Adj Close' column (Yahoo schema may have changed).")

prices = prices.dropna(subset=["Adj Close"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)

prices["Log_Return"] = prices.groupby("Ticker")["Adj Close"].transform(weekly_log_return_from_price)
prices = prices.dropna(subset=["Log_Return"])

if prices.empty:
    raise RuntimeError("Price dataset empty after computing returns.")


# ============================================================
# 6) METADATA (Company_Name, Sector, Market_Cap)
# ============================================================

print("[INFO] Pulling metadata (Company_Name, Sector, Market_Cap) via yfinance...")

meta = pd.DataFrame([safe_yf_info(t) for t in prices["Ticker"].unique()])

# Company name fallback from constituents file
if name_fallback and "Company_Name" in meta.columns:
    meta["Company_Name"] = meta.apply(
        lambda r: r["Company_Name"]
        if pd.notna(r["Company_Name"]) and str(r["Company_Name"]).strip()
        else name_fallback.get(r["Ticker"], None),
        axis=1
    )

prices = prices.merge(meta, on="Ticker", how="left")

prices["Market_Cap"] = pd.to_numeric(prices["Market_Cap"], errors="coerce")
prices = prices.dropna(subset=["Market_Cap"])

if prices.empty:
    raise RuntimeError("All rows dropped due to missing Market_Cap.")


# ============================================================
# 7) SECTOR WEIGHTS (STATIC, CAPPED)
# ============================================================

weights = (
    prices[["Ticker", "Sector", "Market_Cap"]]
    .drop_duplicates(subset=["Ticker"])
    .copy()
)

weights["Sector_Weight"] = (
    weights.groupby("Sector")["Market_Cap"]
           .transform(lambda s: capped_weights_from_caps(s, CFG.sector_weight_cap))
)

# AUDIT: weights sum to 1 per sector
wcheck = weights.groupby("Sector")["Sector_Weight"].sum()
if not np.allclose(wcheck.values, 1.0):
    bad = wcheck.loc[~np.isclose(wcheck, 1.0)]
    raise RuntimeError(f"Sector weights do not sum to 1 per sector. Examples:\n{bad.head(10)}")

prices = prices.merge(weights[["Ticker", "Sector_Weight"]], on="Ticker", how="left")
prices = prices.dropna(subset=["Sector_Weight"])


# ============================================================
# 8) MARKET RETURNS (^GSPC)
# ============================================================

print(f"[INFO] Downloading market proxy ({CFG.market_ticker})...")

mkt = yf.download(
    CFG.market_ticker,
    start=CFG.start_date,
    end=CFG.end_date,
    interval=CFG.interval,
    auto_adjust=True,
    threads=False,
    progress=False,
)

if mkt.empty:
    raise RuntimeError(f"Yahoo Finance returned empty data for {CFG.market_ticker}.")

if isinstance(mkt.columns, pd.MultiIndex):
    mkt.columns = mkt.columns.get_level_values(0)

mkt = mkt.reset_index()
mkt["Date"] = to_week_ending_friday(mkt["Date"])
audit_warn_non_friday("Market series", mkt["Date"])

close_col = "Close" if "Close" in mkt.columns else mkt.columns[1]
mkt["Market_Log_Return"] = weekly_log_return_from_price(mkt[close_col])
mkt = mkt[["Date", "Market_Log_Return"]].dropna().sort_values("Date")


# ============================================================
# 9) RISK-FREE (FRED DTB3) WITH RETRY + CACHE
# ============================================================

print(f"[INFO] Pulling risk-free series from FRED: {CFG.rf_series}")

rf = None
last_err = None

for attempt in range(1, 4):
    try:
        rf = fetch_rf_from_fred(CFG.start_date, CFG.end_date)
        if rf is not None and not rf.empty:
            print(f"[INFO] Risk-free series loaded from FRED (attempt {attempt}).")
            rf.to_csv(RF_CACHE_PATH, index=False)
            print(f"[INFO] Cached RF series to: {RF_CACHE_PATH}")
            break
    except Exception as e:
        last_err = e
        print(f"[WARN] FRED RF load failed (attempt {attempt}/3): {e}")

if rf is None or rf.empty:
    if RF_CACHE_PATH.exists():
        print(f"[WARN] Using cached RF series: {RF_CACHE_PATH}")
        rf = pd.read_csv(RF_CACHE_PATH, parse_dates=["Date"]).sort_values("Date")
    else:
        print("[WARN] No cached RF series found; defaulting RF_Log_Return to 0.")
        rf = mkt[["Date"]].copy()
        rf["RF_Log_Return"] = 0.0
        print(f"[WARN] Last FRED error: {last_err}")

audit_warn_non_friday("Risk-free series", rf["Date"])


# ============================================================
# 10) MASTER STOCK PANEL
# ============================================================

panel = (
    prices
    .merge(mkt, on="Date", how="left")
    .merge(rf, on="Date", how="left")
)

panel = panel.dropna(subset=["Market_Log_Return", "RF_Log_Return"])
panel["Weighted_Log_Return"] = panel["Log_Return"] * panel["Sector_Weight"]

if panel.empty:
    raise RuntimeError("Panel is empty after merging market and RF series.")

audit_warn_non_friday("Master panel", panel["Date"])


# ============================================================
# 11) SECTOR RETURNS (CAP-WEIGHTED)
# ============================================================

sector_returns = (
    panel
    .groupby(["Date", "Sector"], as_index=False)
    .agg(Sector_Log_Return=("Weighted_Log_Return", "sum"))
)

if sector_returns.empty:
    raise RuntimeError("sector_returns is empty.")

audit_warn_non_friday("Sector returns", sector_returns["Date"])


# ============================================================
# 12) SECTOR CAPM (STATIC + ROLLING)
# ============================================================

mkt_rf = panel[["Date", "Market_Log_Return", "RF_Log_Return"]].drop_duplicates("Date").sort_values("Date")

sector_capm_static = compute_sector_capm_static(sector_returns, mkt_rf)
sector_capm_rolling = compute_sector_capm_rolling(sector_returns, mkt_rf, CFG.rolling_window_weeks)


# ============================================================
# 13) EXPORT
# ============================================================

panel.to_csv(PANEL_OUT, index=False)
sector_returns.to_csv(SECTOR_RETURNS_OUT, index=False)
sector_capm_static.to_csv(SECTOR_CAPM_STATIC_OUT, index=False)
sector_capm_rolling.to_csv(SECTOR_CAPM_ROLLING_OUT, index=False)

print("[INFO] Wrote outputs:")
print(f"  - {PANEL_OUT}")
print(f"  - {SECTOR_RETURNS_OUT}")
print(f"  - {SECTOR_CAPM_STATIC_OUT}")
print(f"  - {SECTOR_CAPM_ROLLING_OUT}")
print(f"  - {RF_CACHE_PATH} (RF cache)")

print("[DONE] Pipeline completed successfully.")
