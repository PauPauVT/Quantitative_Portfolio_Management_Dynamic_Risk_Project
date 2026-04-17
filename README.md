# Quantitative Portfolio Management & Dynamic Risk Framework

## Project Overview
This project establishes a high-performance quantitative framework for portfolio construction, optimization, and risk monitoring. Using a universe of S&P 500 assets, the system addresses the "instability" problem of traditional Mean-Variance Optimization (MVO) by implementing advanced statistical shrinkage and Bayesian techniques. Furthermore, it integrates a dynamic risk layer using GARCH models to forecast volatility regimes.

## Key Objectives
- **Robust Optimization:** Overcome MVO limitations using **Ledoit-Wolf Shrinkage** and **Black-Litterman** models.
- **Dynamic Volatility Modeling:** Capture "Volatility Clustering" and leverage-effects in financial time series.
- **Tail Risk Measurement:** Quantify potential losses through **Value at Risk (VaR)** and **Conditional VaR (CVaR)**.
- **Resilience Testing:** Conduct historical **Stress Tests** to evaluate performance during black-swan events (e.g., COVID-19 2020 crash).

## Methodology & Technical Features

### 1. Asset Allocation & Optimization
Standard MVO often leads to extreme, undiversified weights due to estimation errors in the covariance matrix. This project implements:
- **Ledoit-Wolf Shrinkage:** A mathematical approach that "shrinks" the sample covariance matrix towards a target (like a constant correlation model), significantly improving the signal-to-noise ratio.
- **Black-Litterman Model:** Combines market equilibrium with subjective views, resulting in more intuitive and stable asset allocations.

### 2. Risk Management (The GARCH Layer)
Risk is not constant over time. To move beyond static risk metrics, I implemented:
- **GARCH(1,1) Models:** To model and forecast the conditional variance of the portfolio. This allows for an "Adaptive Risk Management" approach where the portfolio exposure is adjusted based on predicted volatility regimes.
- **VaR & CVaR Analysis:** Using parametric and Cornish-Fisher expansions to account for non-normality (skewness and kurtosis) in financial returns.

### 3. Backtesting & Performance
The framework includes a comprehensive backtesting engine that evaluates:
- **Sharpe Ratio & Sortino Ratio:** Assessing risk-adjusted returns.
- **Maximum Drawdown:** Analyzing the largest peak-to-trough decline.
- **Turnover Analysis:** Measuring the cost-efficiency of the rebalancing strategy.

## Tech Stack
- **Languages:** Python (3.10+)
- **Quantitative Libraries:** `PyPortfolioOpt`, `arch` (Econometrics), `scipy.optimize`.
- **Data Engineering:** `pandas`, `numpy`, `yfinance`.
- **Visualization:** `matplotlib`, `seaborn` (Advanced heatmaps and efficient frontier plotting).

## Key Findings
- Models utilizing **Ledoit-Wolf Shrinkage** showed a 30% reduction in portfolio turnover compared to standard MVO.
- **GARCH forecasts** successfully anticipated the volatility spike in March 2020, providing a more realistic VaR estimate than historical or static methods.

---
**Author:** Pau Vendrell – Mathematician & Statistician | FRM Candidate
