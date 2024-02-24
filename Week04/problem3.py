from scipy.stats import norm, t
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
import pandas as pd

portfolio = pd.read_csv('portfolio.csv')
daily_prices = pd.read_csv('DailyPrices.csv')

# Filter daily prices to include only the stocks present in the portfolio
unique_stocks = portfolio['Stock'].unique()
filtered_daily_prices = daily_prices[['Date'] + list(unique_stocks)].set_index('Date')

# Calculate the daily values of each portfolio
portfolio_values = pd.DataFrame(index=filtered_daily_prices.index)

for portfolio_name in portfolio['Portfolio'].unique():
    current_portfolio = portfolio[portfolio['Portfolio'] == portfolio_name]

    # Compute the weighted daily values
    daily_values = np.zeros(len(filtered_daily_prices))
    for stock, holding in zip(current_portfolio['Stock'], current_portfolio['Holding']):
        daily_values += filtered_daily_prices[stock] * holding

    portfolio_values[portfolio_name] = daily_values

# calculate total protfolio value
total_holdings = portfolio.groupby('Stock')['Holding'].sum()
total_portfolio_values = np.zeros(len(filtered_daily_prices))
for stock in unique_stocks:
    holding = total_holdings[stock]
    total_portfolio_values += filtered_daily_prices[stock] * holding

portfolio_values['Total'] = total_portfolio_values



# Function to calculate returns based on a given method
def calculate_returns(prices, method="discrete"):
    if method.lower() not in ["discrete", "log"]:
        raise ValueError("method must be either 'discrete' or 'log'")

    if method.lower() == "discrete":
        returns = prices.pct_change().dropna()
    elif method.lower() == "log":
        returns = np.log(prices / prices.shift(1)).dropna()

    return returns


# Calculate the arithmetic returns for all prices
returns = calculate_returns(portfolio_values)
returns = returns - returns.mean()

# VaR using an exponentially weighted moving variance
def calculate_var_ewm(returns, confidence_level=0.05, lambda_=0.94):
    ewm_variance = returns.ewm(alpha=(1-lambda_)).var().dropna().iloc[-1]
    std_dev = np.sqrt(ewm_variance)
    var = norm.ppf(confidence_level, 0, std_dev)  # mean is 0 after
    return var


var_ewm= calculate_var_ewm(returns, confidence_level=0.05, lambda_=0.94)
var_ewm_usd= var_ewm * portfolio_values.iloc[-1]
print(var_ewm, var_ewm_usd)

def calculate_var_normal(returns, confidence_level=0.05):
    mean = np.mean(returns)
    std_dev = np.std(returns)
    var = norm.ppf(confidence_level, mean, std_dev)
    return var

var_normal= calculate_var_normal(returns, confidence_level=0.05)
var_normal_usd= var_normal * portfolio_values.iloc[-1]
print(var_normal, var_normal_usd)