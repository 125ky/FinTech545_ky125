from scipy.stats import norm, t
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
import pandas as pd



# Function to calculate returns based on a given method
def calculate_returns(prices, method="discrete"):
    if method.lower() not in ["discrete", "log"]:
        raise ValueError("method must be either 'discrete' or 'log'")

    if method.lower() == "discrete":
        returns = prices.pct_change().dropna()
    elif method.lower() == "log":
        returns = np.log(prices / prices.shift(1)).dropna()

    return returns


daily_prices = pd.read_csv('DailyPrices.csv')

# Calculate the arithmetic returns for all prices
returns = calculate_returns(daily_prices.drop(columns=['Date']))

# Center the returns of META so that the mean return is 0
returns['META'] = returns['META'] - returns['META'].mean()

print(returns['META'])


# VaR using a normal distribution
def calculate_var_normal(returns, confidence_level=0.05):
    mean = np.mean(returns)
    std_dev = np.std(returns)
    var = norm.ppf(confidence_level, mean, std_dev)
    return var

# VaR using an exponentially weighted moving variance
def calculate_var_ewm(returns, confidence_level=0.05, lambda_=0.94):
    ewm_variance = returns.ewm(alpha=(1-lambda_)).var().dropna().iloc[-1]
    std_dev = np.sqrt(ewm_variance)
    var = norm.ppf(confidence_level, 0, std_dev)  # mean is 0 after centering
    return var

# VaR using a MLE fitted T distribution
def calculate_var_t(returns, confidence_level=0.05):
    params = t.fit(returns)
    var = t.ppf(confidence_level, *params)
    return var

# VaR using a fitted AR(1) model
def calculate_var_ar1(returns, confidence_level=0.05):
    model = AutoReg(returns, lags=1).fit()
    forecast = model.predict(start=len(returns), end=len(returns))
    std_dev = np.sqrt(model.sigma2)
    var = norm.ppf(confidence_level, forecast, std_dev)
    return var

# VaR using a historical simulation
def calculate_var_historical(returns, confidence_level=0.05):
    var = np.percentile(returns, confidence_level * 100)
    return var

# Calculate VaR values for META stock
meta_returns = returns['META']
var_normal = calculate_var_normal(meta_returns)
var_ewm = calculate_var_ewm(meta_returns)
var_t = calculate_var_t(meta_returns)
var_ar1 = calculate_var_ar1(meta_returns)
var_historical = calculate_var_historical(meta_returns)

# Collect all VaR values
var_values = {
    'Normal': var_normal,
    'Exponentially Weighted': var_ewm,
    'MLE Fitted T': var_t,
    'AR(1)': var_ar1,
    'Historical Simulation': var_historical
}

print(var_values)




def calculate_var_ewm(data, alpha=0.05, lambda_ew=0.94, method="Normal"):
    if method == "Normal":
        mean, std = data.mean(), data.std()
        var = norm.ppf(alpha, mean, std)

    elif method == "Normal with EW":
        variance = data.ewm(alpha=lambda_ew).var()
        last_variance = variance.iloc[-1]
        var = norm.ppf(alpha, 0, np.sqrt(last_variance))

    elif method == "T Distribution":
        params = t.fit(data)
        var = t.ppf(alpha, *params)

    elif method == "historical":
        var = data.quantile(alpha)

    return var
