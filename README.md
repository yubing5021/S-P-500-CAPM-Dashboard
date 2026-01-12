# S-P-500-CAPM-Dashboard
A SP 500 return dashboard that analyzes and decompose stock and sector returns into market, sector, and company components. Using excess returns, rolling betas, alpha t-statistics, and Sharpe ratios, it evaluates whether observed performance reflects consistent abnormal returns in correlation to the risk taken. 
CAPM Project Summary:
The CAPM project constructs a S&P 500 return dashboard and applies rolling, factor-aware attribution to analyze and decompose stock and sector returns into market, sector, and company components. Using excess returns, rolling betas, alpha t-statistics, and Sharpe ratios, it evaluates whether observed performance reflects consistent abnormal returns in correlation to the risk taken. 

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
