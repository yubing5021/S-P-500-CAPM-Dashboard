# S&P 500 CAPM Dashboard

A SP 500 return dashboard that analyzes and decompose stock and sector returns into market, sector, and company components. Using excess returns, rolling betas, alpha t-statistics, and Sharpe ratios, it evaluates whether observed performance reflects consistent abnormal returns in correlation to the risk taken. 

Equity prices, tickers, and sector classifications are sourced from Yahoo Finance.
Risk-free rate data is based on the 3-month U.S. Treasury bill, and is sourced from the Federal Reserve Economic Data (FRED) database.

---

## 🚀 Live Demo
> (https://sp500-capm-dashboard-2014tocurrent.streamlit.app/)

---

## 📊 Key Features

- **Sector & Ticker Selection**
  - Multi-select S&P 500 sectors
  - Multi-select tickers labeled as `TICKER (Company Name)`

- **Cumulative Performance**
  - Growth of $1 for:
    - Market
    - Selected sectors
    - Selected individual stocks

- **Rolling CAPM Analysis**
  - Rolling **beta** and **alpha** vs:
    - Market benchmark
    - Sector benchmark
  - Configurable estimation windows:
    - 52 weeks (short-term)
    - 156 weeks (structural)

- **Statistical Diagnostics**
  - Rolling **alpha t-statistics**
  - ±2 significance bands (~5% heuristic)
  - Full regression outputs:
    - Alpha
    - Beta
    - t-stats
    - R² and Adjusted R²

- **Discount Rate Estimation**
  - CAPM discount rate:
    ```
    Discount Rate = Risk-Free Rate + β × Market Risk Premium
    ```
  - Annualized (log-return approximation)
  - Option to use:
    - Historical MRP
    - Custom user-defined MRP

- **Macro Context**
  - Weekly and rolling annualized:
    - Risk-free rate
    - Market risk premium
  - Cumulative growth comparison: Market vs Risk-Free

- **Export Functionality**
  - Download CSVs for:
    - Cumulative returns
    - Rolling CAPM metrics
    - Discount rates
    - Summary statistics

---

## 🧠 Methodology Overview

- **Returns**
  - Weekly **log returns**
  - Excess returns calculated as:
    ```
    Excess Return = Asset Return − Risk-Free Rate
    ```

- **CAPM Estimation**
  - OLS regression:
    ```
    Ri − Rf = α + β (Rm − Rf) + ε
    ```
  - Rolling window estimation for time-varying risk exposure

- **Annualization**
  - Mean log returns × 52
  - Volatility × √52

- **Statistical Interpretation**
  - **Beta**: systematic market exposure
  - **Alpha**: abnormal return beyond CAPM
  - **t-stats**: statistical significance
  - **R² / Adj R²**: explanatory power of the market factor

---

## 🗂 Repository Structure

capm-dashboard/
│
├── CAPM_Dashboard.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── README.md # Project documentation
│
├── data/
│ ├── sp500_stock_panel.csv # Weekly stock-level return panel
│ └── sector_returns.csv # Weekly sector return series
│
└── .streamlit/
└── config.toml # Optional UI theming


Key Terms

CAPM (Capital Asset Pricing Model): The theory of asset valuation. It takes a portfolio view of risk. It combines the reward for bearing the risk an asset (company’s stock) and defines the reward of delayed consumption. 

Discount rate: It’s the required rate of return on equity in relation to the risk held.

Alpha (α): Measures risk a stock’s performance relative to the S&P. In this project, the excess return a company’s stock generates in relation to the amount of risk taken.
Alpha= R_p-[R_f+(R_m-R_f )β]
 
Beta (β): Measures risk one asset adds to the risk of an entire portfolio. In this project, the risk one company’s stock adds to the S&P 500 index.
Beta=  〖Return Covariance〗_Stock/(〖Return Variance〗_(S&P 500 index)  )

Sharpe ratio: risk adjusted measure comparing the return to its risk. The higher the ratio, the higher the return for the risk taken on.
Sharpe Ratio=(risk premium(Annual Excess Return))/(standard deviation (Annual Volaltity))  

Risk Free (Rf): The reward for investors for delayed consumption (measured by 3 month T-bills).

Market Risk Premium (Rm): The reward for investors for the risk taken. 

Variance (σ2): Average standard deviation of a stocks returns from the stock’s mean.

Standard deviation (σ): Square root of variance. 

Covariance: measures how the stock moves with the S&P 500.

T- stat: a standardized value used to determine if there is a significant statistical difference between the sample data and a population mean.
	∣t∣≥1.96→ statistically significant at ~5% -> if greater, than strong evidence of co-movement
  ∣t∣≥2.58→ significant at ~1% - if greater, then there are abnormal returns that exceeds the risk and can’t be explained
When the t-stat is greater than 1.96, then we are confident that the systematic exposure of the company is tied to the market conditions, i.e. when the market rises, the company’s rises as well.
“We are confident Charter’s beta reflects actual exposure to market risk—not a statistical accident (chance or randomness)—because the estimated relationship is far too strong and constant to be explained by randomness.”

R Squared: measure of how much a stock’s excess return variability is explained by the market

Adjusted R Squared: Similar to R squared, but it a more accurate version. 

Annual Volitlity- It is the annualized standard deviation of weekly log returns and represents the company’s historical return volatility. Used to calculate the Sharpe Ratio. Measures total risk, not just market risk.
Metric	Benchmark	Question it answers

Annual Excess Return	Risk-free rate	“Did I beat cash?”

Alpha	Market (or sector)	“Did I beat what CAPM predicts for my risk?”
