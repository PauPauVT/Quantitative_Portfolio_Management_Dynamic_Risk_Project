import yfinance as yf
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
from scipy import stats
from arch import arch_model
from sklearn.covariance import LedoitWolf
from pypfopt import black_litterman
from pypfopt import EfficientFrontier

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers)
tables = pd.read_html(response.text)

tickers = tables[0]

data = yf.download(tickers.Symbol.to_list(), start="2016-01-01", end="2026-01-01", auto_adjust=True)["Close"]

# Calculate daily returns of the Stock with less than 5% of missing data
data_returns = data.loc[:, data.isna().sum() <= data.shape[0] * 0.05].pct_change().dropna()
tickers = data_returns.columns

# Statistical metrics annualized
mean_ret = data_returns.mean() * 252
std_ret = data_returns.std() * np.sqrt(252)
cov_mat = data_returns.cov() * 252
corr_mat = data_returns.corr()
RF = 0.02

# Show table
stats_df = pd.DataFrame({
    'Ticker': tickers,
    'Annual Retorns': mean_ret.values,
    'Volatility': std_ret.values,
    'marketCap': 0
})

for tick in stats_df["Ticker"]:
    stats_df.loc[stats_df["Ticker"] == tick, "marketCap"] = yf.Ticker(tick).info["marketCap"]

print('=== ASSET STATISTICS ===')
print(stats_df.head())
print("\nRisk Free Rate (RF):", RF * 100, "%")

# Monte Carlo simulation
N_PORTFOLIOS = 200000
results = np.zeros((3, N_PORTFOLIOS))
weights_list = []

for i in range(N_PORTFOLIOS):
    # Generating random weights summing 1
    w = np.random.random(data_returns.shape[1])
    w = w/np.sum(w)
    weights_list.append(w)

    # metrics
    ret = np.dot(w, mean_ret)
    vol = np.sqrt(w @ cov_mat @ w)
    sharpe = (ret - RF) / vol

    results[0, i] = ret
    results[1, i] = vol
    results[2, i] = sharpe

# Find best Monte Carlo portfolio
best_mc_idx = np.argmax(results[2, :])
best_mc_weights = weights_list[best_mc_idx]

print(f'\n=== BEST MONTE CARLO PORTFOLIO ===')
print(f'Return: {results[0, best_mc_idx]:.2%}')
print(f'Volatility: {results[1, best_mc_idx]:.2%}')
print(f'Sharpe: {results[2, best_mc_idx]:.2f}')

# Plotting the point cloud
scatter = plt.scatter(results[1, :], results[0, :],
            c = results[2, :], cmap = "viridis",
            alpha = 0.5, s = 10)
plt.scatter(results[1, best_mc_idx], results[0, best_mc_idx],
           marker='*', color='red', s=500, label='Best Monte Carlo', edgecolors='black', linewidth=2)
plt.colorbar(label = "Sharpe Ratio")
plt.xlabel("Volatility")
plt.ylabel("Return")
plt.title('Point Cloud - Monte Carlo Simulation', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# The grewan/yellow points have the best Sharpe Ratio. The point more at the left-up is the optimal portfolio

# === FASE 3: SCIPY OPTIMIZATION ===

def neg_sharpe(weights):
    ret = np.dot(weights, mean_ret)
    vol = np.sqrt(weights @ cov_mat @ weights)
    penalty = 0.1 * np.sum(weights**2)  # L2
    return -(ret - RF) / vol + penalty

# Restrictions and limitations
constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
bounds = tuple((0, 0.50) for _ in tickers)
w0 = np.array([1/len(tickers)] * len(tickers))

# Optimize
result = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)
opt_weights = result.x

# Calculate the metrics of the optimal portfolio
opt_ret = np.dot(opt_weights, mean_ret)
opt_vol = np.sqrt(opt_weights @ cov_mat @ opt_weights)
opt_sharpe = (opt_ret - RF) / opt_vol

print('=== OPTIMAL PORTAFOLIO (SCIPY) ===')
print(f'Retorno: {opt_ret:.2%}')
print(f'Volatilidad: {opt_vol:.2%}')
print(f'Sharpe Ratio: {opt_sharpe:.2f}')

lw = LedoitWolf()
lw.fit(data_returns)

cov_shrink = lw.covariance_

def neg_sharpe(weights):
    ret = np.dot(weights, mean_ret)
    vol = np.sqrt(weights @ cov_shrink @ weights)
    penalty = 0.1 * np.sum(weights**2)  # L2
    return -(ret - RF) / vol + penalty

# Restrictions and limitations
constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
bounds = tuple((0, 0.50) for _ in tickers)
w0 = np.array([1/len(tickers)] * len(tickers))

# Optimize
result = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)
opt_weights_shrinkage = result.x

# Calculate the metrics of the optimal portfolio
opt_ret = np.dot(opt_weights_shrinkage, mean_ret)
opt_vol = np.sqrt(opt_weights_shrinkage @ cov_mat @ opt_weights_shrinkage)
opt_sharpe = (opt_ret - RF) / opt_vol

print('=== OPTIMAL PORTAFOLIO (Shrinkage (Ledoit-Wolf) + Scipy) ===')
print(f'Retorno: {opt_ret:.2%}')
print(f'Volatilidad: {opt_vol:.2%}')
print(f'Sharpe Ratio: {opt_sharpe:.2f}')

from pypfopt.black_litterman import market_implied_prior_returns

pi = market_implied_prior_returns(
    market_caps=stats_df["marketCap"],
    cov_matrix=cov_shrink,
    risk_aversion=2.5
)

momentum = np.log(data[tickers] / data[tickers].shift(252)).shift(21).dropna()
mom_signal = momentum.iloc[-1]

n_long = 5
n_short = 5

long_assets = mom_signal.nlargest(n_long).index
short_assets = mom_signal.nsmallest(n_short).index

P = np.zeros((1, len(data[tickers].columns)))

P[0, mom_signal.index.get_indexer(long_assets)] = 1 / n_long
P[0, mom_signal.index.get_indexer(short_assets)] = -1 / n_short

fwd_returns = data_returns.shift(-21)

# promedio histórico forward return spread
long_mean = fwd_returns[long_assets].stack().mean()
short_mean = fwd_returns[short_assets].stack().mean()

Q = np.array([data_returns[long_assets].mean().mean() - data_returns[short_assets].mean().mean()])

bl = black_litterman.BlackLittermanModel(
    cov_matrix=cov_shrink,
    pi=pi,
    P=P,
    Q=Q,
    omega="idzorek",
    view_confidences=np.array([0.6])
)

bl_returns = bl.bl_returns()

ef = EfficientFrontier(bl_returns, cov_shrink, solver="SCS")
weights = np.array(list(ef.max_sharpe().values()))

# Calculate the metrics of the optimal portfolio
opt_ret = np.dot(weights, mean_ret)
opt_vol = np.sqrt(weights @ cov_mat @ weights)
opt_sharpe = (opt_ret - RF) / opt_vol

print('=== OPTIMAL PORTAFOLIO (Black-Litterman Model) ===')
print(f'Retorno: {opt_ret:.2%}')
print(f'Volatilidad: {opt_vol:.2%}')
print(f'Sharpe Ratio: {opt_sharpe:.2f}')

tickers = tickers[opt_weights_shrinkage>0.0001]

data_returns = data_returns[tickers]

# Statistical metrics annualized
mean_ret = data_returns.mean() * 252
std_ret = data_returns.std() * np.sqrt(252)
cov_mat = data_returns.cov() * 252
corr_mat = data_returns.corr()

def neg_sharpe(weights):
    ret = np.dot(weights, mean_ret)
    vol = np.sqrt(weights @ cov_mat @ weights)
    penalty = 0.1 * np.sum(weights**2)  # L2
    return -(ret - RF) / vol + penalty

# Monte Carlo simulation
N_PORTFOLIOS = 200000
results = np.zeros((3, N_PORTFOLIOS))
weights_list = []

for i in range(N_PORTFOLIOS):
    # Generating random weights summing 1
    w = np.random.random(data_returns.shape[1])
    w = w/np.sum(w)
    weights_list.append(w)

    # metrics
    ret = np.dot(w, mean_ret)
    vol = np.sqrt(w @ cov_mat @ w)
    sharpe = (ret - RF) / vol

    results[0, i] = ret
    results[1, i] = vol
    results[2, i] = sharpe

# Find best Monte Carlo portfolio
best_mc_idx = np.argmax(results[2, :])
best_mc_weights = weights_list[best_mc_idx]

print(f'\n=== BEST MONTE CARLO PORTFOLIO ===')
print(f'Return: {results[0, best_mc_idx]:.2%}')
print(f'Volatility: {results[1, best_mc_idx]:.2%}')
print(f'Sharpe: {results[2, best_mc_idx]:.2f}')

# SCIPY OPTIMIZATION
# Restricciones y límites
constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # Suma = 1
bounds = tuple((0, 0.50) for _ in tickers)  # 5% mín, 40% máx
w0 = np.array([1/len(tickers)] * len(tickers))  # Pesos iniciales

# Optimizar
result = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)
opt_weights = result.x

# Calcular métricas del portafolio óptimo
opt_ret = np.dot(opt_weights, mean_ret)
opt_vol = np.sqrt(opt_weights @ cov_mat @ opt_weights)
opt_sharpe = (opt_ret - RF) / opt_vol

print('=== PORTAFOLIO ÓPTIMO (SCIPY) ===')
print(f'Retorno: {opt_ret:.2%}')
print(f'Volatilidad: {opt_vol:.2%}')
print(f'Sharpe Ratio: {opt_sharpe:.2f}')
print(f'\nPesos óptimos:')
for t, w in zip(tickers, opt_weights):
    print(f'  {t}: {w:.1%}')

    corr = data_returns.corr()

plt.figure()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Compare both methods
plt.figure(figsize=(12, 7))

# Monte Carlo Cloud
plt.scatter(results[1, :], results[0, :], c=results[2, :],
           cmap='viridis', alpha=0.3, s=10, label='10,000 portafolios MC')

# Best Monte Carlo
plt.scatter(results[1, best_mc_idx], results[0, best_mc_idx],
           marker='o', color='red', s=300, label='Best Monte Carlo', edgecolors='black', linewidth=2)

# Optimal Scipy
plt.scatter(opt_vol, opt_ret, marker='*', color='gold', s=800,
           label='Optimal Scipy', edgecolors='black', linewidth=2)

plt.xlabel('Volatility', fontsize=12, fontweight='bold')
plt.ylabel('Return', fontsize=12, fontweight='bold')
plt.title('Monte Carlo vs Scipy: Finding the Optimal Portfolio', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f'\nImprovement Scipy vs Monte Carlo:')
print(f'  Sharpe: {opt_sharpe:.2f} vs {results[2, best_mc_idx]:.2f} (+{(opt_sharpe/results[2, best_mc_idx]-1)*100:.1f}%)')

# === PHASE 4: REPORT ===
# Metrics Table
print('PORTFOLIO METRICS')
print('-' * 60)
print(f'Expected Annual Return: {opt_ret:.2%}')
print(f'Annual Volatility: {opt_vol:.2%}')
print(f'Sharpe Ratio: {opt_sharpe:.2f}')
print(f'\nComparison with Equity Portfolio (1/N):')
eq_weights = np.array([1/len(tickers)] * len(tickers))
eq_ret = np.dot(eq_weights, mean_ret)
eq_vol = np.sqrt(eq_weights @ cov_mat @ eq_weights)
eq_sharpe = (eq_ret - RF) / eq_vol
print(f' Return: {eq_ret:.2%} (vs optimal {opt_ret:.2%})')
print(f' Volatility: {eq_vol:.2%} (vs optimal {opt_vol:.2%})')
print(f' Sharpe: {eq_sharpe:.2f} (vs optimal {opt_sharpe:.2f})')
print(f' Improvement: {(opt_sharpe/eq_sharpe - 1)*100:.1f}% better Sharpe')

# Recommended assignment
print('===== Recommended assignment =====')
for t, w in zip(tickers, opt_weights):
    print(f'{t}: {w:.1%}')

# Foot chart of weights
plt.figure(figsize=(12, 12))
colors = plt.cm.Set3(np.linspace(0, 1, len(tickers)))
plt.pie(opt_weights, labels=tickers, autopct='%1.1f%%', colors=colors, startangle=90, textprops={'fontsize': 5})
plt.title('Weight Distribution', fontsize=12, fontweight='bold')

weights = opt_weights

# Return of the portfolio
port_returns = (data_returns * weights).sum(axis=1)

print(f"Analysis period: {data_returns.index[0].date()} to {data_returns.index[-1].date()}")
print(f"Total observations: {len(port_returns)} days")
print(f"\nPortfolio statistics:")
print(f"Average daily return: {port_returns.mean():.4%}")
print(f"Daily volatility: {port_returns.std():.4%}")
print(f"Annualized return: {port_returns.mean() * 252:.2%}")
print(f"Annualized volatility: {port_returns.std() * np.sqrt(252):.2%}")

portfolio_value = 100000

#2. Historical VaR and CVaR
var_95 = np.percentile(port_returns, 5) * portfolio_value
var_99 = np.percentile(port_returns, 1) * portfolio_value

cvar_95 = port_returns[port_returns <= np.percentile(port_returns, 5)].mean() * portfolio_value
cvar_99 = port_returns[port_returns <= np.percentile(port_returns, 1)].mean() * portfolio_value

#3. Parametric VaR
param_var_95 = (port_returns.mean() - 1.96 * port_returns.std()) * portfolio_value
param_var_99 = (port_returns.mean() - 2.575 * port_returns.std()) * portfolio_value

# 4. Normality test (Shapiro-Wilk)
stat, p_value = stats.shapiro(port_returns)

print("\n" + "="*60)
print("PHASE 1: RISK ANALYSIS HISTORICAL")
print("="*60)
print(f"\n95% Var (maximum loss 95% of the time): ${var_95:,.2f}")
print(f"99% Var (maximum loss 99% of the time): ${var_99:,.2f}")
print(f"\n95% Parametric VaR (Gaussian): ${param_var_95:,.2f}")
print(f"99% Parametric VaR (Gaussian): ${param_var_99:,.2f}")
print(f"\n95% CVaR (average loss on worst days): ${cvar_95:,.2f}")
print(f"99% CVaR (average extreme loss): ${cvar_99:,.2f}")
print(f"\nShapiro-Wilk test p-value: {p_value:.6f}")
if p_value < 0.05:
    print("Returns are NOT normal → Historical CVaR is more reliable")
else:
    print("Returns are normal")

z = 1.96
portfolio_vol = np.sqrt(weights.T @ cov_mat @ weights)
portfolio_var = z * portfolio_vol
marginal_contrib = cov_mat @ weights / portfolio_vol
component_var = weights * marginal_contrib * z
component_var_pct = component_var / portfolio_var
df = pd.DataFrame({
    'Asset': component_var_pct.index,
    'Weight': weights,
    'RiskContribution': component_var_pct
}, index=component_var_pct.index)

df = df.reset_index().rename(columns={'index': 'Asset'})
df.plot(x='Asset', y=['Weight', 'RiskContribution'], kind='bar')

# 4. Chart: Distribution of returns with VaR and CVaR
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(port_returns * 100, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].axvline(np.percentile(port_returns, 5) * 100, color='red', linestyle='--', linewidth=2, label=f'VaR 95%')
axes[0].axvline(np.percentile(port_returns, 1) * 100, color='darkred', linestyle='--', linewidth=2, label=f'VaR 99%')
axes[0].set_xlabel('Daily return (%)', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Portfolio Returns Distribution', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

#Q-QPlot
stats.probplot(port_returns, dist='norm', plot=axes[1])
axes[1].set_title('Q-Q Plot: Normality of Returns', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

#1. Fit GARCH(1,1) to the portfolio
port_pct = port_returns * 100
garch_model = arch_model(port_pct, vol='Garch', p=1, q=1)
garch_fit = garch_model.fit(disp='off')

# Extract parameters
omega = garch_fit.params['omega']
alpha = garch_fit.params['alpha[1]']
beta = garch_fit.params['beta[1]']
persistence = alpha + beta

print("\n" + "="*60)
print("PHASE 2: GARCH + DYNAMIC VaR")
print("="*60)
print(f"\nGARCH parameters(1,1):")
print(f"ω (omega): {omega:.6f}")
print(f"α (alpha): {alpha:.4f} (reaction speed to shocks)")
print(f"β (beta): {beta:.4f} (memory strength)")
print(f"α + β: {persistence:.4f} (overall persistence of risk)")

if persistence < 1:
    half_life = np.log(0.5) / np.log(persistence)
    print(f"\nHalf-life of a shock: {half_life:.1f} days")
    print(f"Interpretation: A shock takes {half_life:.0f} days to lose 50% of its impact")

# 2. Dynamic VaR (based on GARCH)
cond_vol = garch_fit.conditional_volatility / 100 # Convert to decimal
z_95 = stats.norm.ppf(0.05) # -1.645
z_99 = stats.norm.ppf(0.01) # -2.326

var_dynamic_95 = (garch_fit.params['mu'] / 100 + z_95 * cond_vol) * portfolio_value
var_dynamic_99 = (garch_fit.params['mu'] / 100 + z_99 * cond_vol) * portfolio_value

# Static VaR for comparison
var_static_95 = var_95
var_static_99 = var_99

print(f"\nComparison of Static vs Dynamic VaR:")
print(f"\nVaR 95%:")
print(f" Static (fixed): ${var_static_95:,.2f}")
print(f" Dynamic today (GARCH): ${var_dynamic_95.iloc[-1]:,.2f}")
print(f" Difference: {(var_dynamic_95.iloc[-1] / var_static_95 - 1) * 100:+.1f}%")

print(f"\nVaR 99%:")
print(f" Static (fixed): ${var_static_99:,.2f}")
print(f" Dynamic today (GARCH): ${var_dynamic_99.iloc[-1]:,.2f}")
print(f" Difference: {(var_dynamic_99.iloc[-1] / var_static_99 - 1) * 100:+.1f}%")

print(f"\nThe GARCH dynamic VaR is {abs(var_dynamic_95.iloc[-1] / var_static_95):.1f}x more accurate in crisis")

#3. Chart: Static vs Dynamic VaR
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# VaR 95%
axes[0].plot(var_dynamic_95.index, var_dynamic_95, label='Dynamic VaR (GARCH)', color='orange', linewidth=2)
axes[0].axhline(var_static_95, label='VaR Static (fixed)', color='red', linestyle='--', linewidth=2)
axes[0].fill_between(var_dynamic_95.index, var_dynamic_95, var_static_95, alpha=0.2, color='orange')
axes[0].set_ylabel('VaR 95% ($)', fontsize=11)
axes[0].set_title('Dynamic VaR (GARCH) vs Static - Level 95%', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# VaR 99%
axes[1].plot(var_dynamic_99.index, var_dynamic_99, label='Dynamic VaR (GARCH)', color='darkred', linewidth=2)
axes[1].axhline(var_static_99, label='VaR Static (fixed)', color='red', linestyle='--', linewidth=2)
axes[1].fill_between(var_dynamic_99.index, var_dynamic_99, var_static_99, alpha=0.2, color='darkred')
axes[1].set_ylabel('VaR 99% ($)', fontsize=11)
axes[1].set_xlabel('Date', fontsize=11)
axes[1].set_title('Dynamic VaR (GARCH) vs Static - Level 99%', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

#1. Ensure date format
data_returns.index = pd.to_datetime(data_returns.index)

#2. Define periods of historical crisis
crisis_periods = {
    'Crisis 2008': ('2008-09-01', '2009-03-31'),
    'Flash Crash 2010': ('2010-05-01', '2010-05-31'),
    'COVID-19 2020': ('2020-02-01', '2020-04-30'),
    'Crash 2022': ('2022-01-01', '2022-12-31')
}

#3. Calculate losses in each crisis
crisis_results = {}

for crisis_name, (start, end) in crisis_periods.items():
    crisis_mask = (data_returns.index >= pd.to_datetime(start)) & (data_returns.index <= pd.to_datetime(end))
    print(f"{crisis_name}: {crisis_mask.sum()} observations")

    if crisis_mask.sum() > 0:
        crisis_returns = (data_returns.loc[crisis_mask] * weights).sum(axis=1)

        crisis_var_95 = np.percentile(crisis_returns, 5) * portfolio_value
        crisis_var_99 = np.percentile(crisis_returns, 1) * portfolio_value
        crisis_cvar_95 = crisis_returns[crisis_returns <= np.percentile(crisis_returns, 5)].mean() * portfolio_value
        crisis_cvar_99 = crisis_returns[crisis_returns <= np.percentile(crisis_returns, 1)].mean() * portfolio_value
        daily_volatility = crisis_returns.std()
        total_loss = crisis_returns.sum() * portfolio_value

        crisis_results[crisis_name] = {
            'VaR 95%': crisis_var_95,
            'VaR 99%': crisis_var_99,
            'CVaR 95%': crisis_cvar_95,
            'CVaR 99%': crisis_cvar_99,
            'Daily Volatility': daily_volatility,
            'Total Loss': total_loss,
            'Loss %': (total_loss / portfolio_value) * 100
            }

print("\nAvailable Crises:", list(crisis_results.keys()))

print("\nComparative Table: Normal Conditions vs. Crisis\n")

if len(crisis_results) == 0:
    print("No crises are available in your data date range.")
else:
    crisis_name = list(crisis_results.keys())[0]

comparison_data = {
    'Metric': ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%', 'Daily Volatility'],
    'Normal Value': [
        f"${var_95:,.0f}",
        f"${var_99:,.0f}",
        f"${cvar_95:,.0f}",
        f"${cvar_99:,.0f}",
        f"{port_returns.std() * 100:.2f}%"
        ],
        f'Value {crisis_name}': [
            f"${crisis_results[crisis_name]['VaR 95%']:,.0f}",
            f"${crisis_results[crisis_name]['VaR 99%']:,.0f}",
            f"${crisis_results[crisis_name]['CVaR 95%']:,.0f}",
            f"${crisis_results[crisis_name]['CVaR 99%']:,.0f}",
            f"{crisis_results[crisis_name]['Daily Volatility'] * 100:,.2f}%"
            ]
    }

df_comparison = pd.DataFrame(comparison_data)

print(df_comparison.to_string(index=False))
print("\nIn a crisis, the CVaR can be higher than the normal VaR")
print("Additional capital reserves are required")

# 1. Executive Report
print("==== PROFESSIONAL RISK REPORT - INTEGRATED SYSTEM ====")
print(f"Total Value: ${portfolio_value:,} USD")
print(f"Period: {data_returns.index[0].date()} to {data_returns.index[-1].date()}")
print(f"Report Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n==== 1. METRICS OF RISK - NORMAL CONDITIONS ====")
print(f"Estimated 95% VarR: ${var_95:>15,.2f} (Maximum loss 95% of the time)")
print(f"Estimated 99% VarR: ${var_99:>15,.2f} (Maximum loss 99% of the time)")
print(f"Estimated 95% CVaR: ${cvar_95:>15,.2f} (Average loss on worst days)")
print(f"Estimated 99% CVaR: ${cvar_99:>15,.2f} (Extreme average loss)")

print("\n==== 2. GARCH(1,1) MODEL - VOLATILITY CONDITIONAL ====")
print(f"α (alpha): {alpha:>15.4f} (Reaction to shocks)")
print(f"β (beta): {beta:>15.4f} (Persistence)")
print(f"α + β: {persistence:>15.4f} (Stability)")
print(f"Average Annual Volatility: {port_returns.std() * np.sqrt(252):>8.2%}")

print("\n==== 3. HISTORICAL CRISIS SCENARIOS ====")
worst_crisis = max(crisis_results.items(), key=lambda x: abs(x[1]['Total Loss']))
print(f"Historical Worst Scenario: {worst_crisis[0]}")
print(f"Total Loss: ${worst_crisis[1]['Total Loss']:,.2f} ({worst_crisis[1]['Loss %']:.1f}%)")
print(f"99% CVaR in Crisis: ${worst_crisis[1]['CVaR 99%']:,.2f}")

print("\n==== 4. RISK MANAGEMENT RECOMMENDATIONS ====")
capital_reserve = abs(worst_crisis[1]['CVaR 99%']) * 1.5 # 150% of the worst CVaR
stop_loss = var_99 * 1.5 # 150% 99% VaR

print(f"Minimum Required Reserve Capital: ${capital_reserve:,.2f}")
print(f"Suggested Daily Stop-Loss Limit: ${stop_loss:,.2f}")

# Final Graphic: Risk Dashboard
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

#1. Conditional Volatility GARCH
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(cond_vol.index, cond_vol * 100, label='GARCH Volatility', color='darkblue', linewidth=1.5)
ax1.fill_between(cond_vol.index, cond_vol * 100, alpha=0.3, color='darkblue')
ax1.set_ylabel('Daily Volatility (%)', fontsize=10)
ax1.set_title('GARCH Conditional Volatility(1,1)', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)
ax1.legend(fontsize=9)

#2.Dynamic VaR
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(var_dynamic_99.index, var_dynamic_99, label='VaR 99% Dynamic', color='red', linewidth=1.5)
ax2.axhline(var_static_99, label='VaR 99% Static', color='darkred', linestyle='--', linewidth=2)
ax2.set_ylabel('VaR 99% ($)', fontsize=10)
ax2.set_title('Dynamic vs Static VaR', fontsize=11, fontweight='bold')
ax2.grid(alpha=0.3)
ax2.legend(fontsize=9)

#3. Distribution of returns
ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(port_returns * 100, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax3.axvline(np.percentile(port_returns, 1) * 100, color='red', linestyle='--', linewidth=2, label='VaR 99%')
ax3.set_xlabel('Daily Return (%)', fontsize=10)
ax3.set_ylabel('Frequency', fontsize=10)
ax3.set_title('Return Distribution', fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3, axis='y')

#4. Metrics Chart
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')

# Select crisis available
if len(crisis_results) > 0:
    crisis_name = list(crisis_results.keys())[0]
    crisis_var_95_val = f"${crisis_results[crisis_name]['VaR 95%']:,.0f}"
    crisis_var_99_val = f"${crisis_results[crisis_name]['VaR 99%']:,.0f}"
    crisis_cvar_99_val = f"${crisis_results[crisis_name]['CVaR 99%']:,.0f}"
    crisis_vol_val = f"{crisis_results[crisis_name]['Daily Volatility'] * 100:,.2f}%"
    crisis_sharpe_val = "N/A"
else:
    crisis_name = "Crisis not available"
    crisis_var_99_val = "N/A"
    crisis_cvar_99_val = "N/A"
    crisis_vol_val = "N/A"
    crisis_sharpe_val = "N/A"

table_data = [
    ['Metric', 'Normal Value', f'Value {crisis_name}'],
    ['VaR 95%', f'${var_95:,.0f}', crisis_var_95_val],
    ['VaR 99%', f'${var_99:,.0f}', crisis_var_99_val],
    ['CVaR 99%', f'${cvar_99:,.0f}', crisis_cvar_99_val],
    ['Daily Vol', f'{port_returns.std()*100:.2f}%', crisis_vol_val],
    ['Sharpe Ratio', f'{(port_returns.mean() / port_returns.std()):.2f}', crisis_sharpe_val]
]

table = ax4.table(
    cellText=table_data,
    cellLoc='center',
    loc='center',
    colWidths=[0.3, 0.35, 0.35]
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Colorize header
for i in range(3):
    table[(0, i)].set_facecolor('#2C3E50')
    table[(0, i)].set_text_props(weight='bold', color='white')

plt.suptitle('Professional Risk Management Dashboard', fontsize=14, fontweight='bold', y=0.98)
plt.show()
