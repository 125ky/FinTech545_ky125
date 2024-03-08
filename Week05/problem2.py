import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import t

data = []
with open('problem1.csv', 'r') as file:
    next(file)
    for line in file:
        data.append(float(line.strip()))

mean_value = sum(data) / len(data)
std_value = np.std(data)


## ES formula
def ES(x, alpha=0.05):
    xs = np.sort(x)
    n = alpha * len(xs)
    idn = int(np.floor(n))
    ES = np.mean(xs[:idn])
    return -ES


# EWMA
def exp_weighted_cov(input, lambda_=0.97):
    ror = input
    ror_mean = np.mean(ror, axis=0)
    dev = ror - ror_mean
    times = dev.shape[0]
    weights = np.zeros(times)

    for i in range(times):
        weights[times - i - 1] = (1 - lambda_) * lambda_ ** i

    weights_mat = np.diag(weights / sum(weights))

    cov = np.transpose(dev) @ weights_mat @ dev
    return cov


exp_simu_returns = np.random.normal(mean_value, np.sqrt(exp_weighted_cov(data, lambda_=0.97)), 10000)

VaR_ew = -np.percentile(exp_simu_returns, 5)

es_normal = ES(exp_simu_returns)

print("EW normal - VaR at 5% significance level:", VaR_ew)
print("Expected Shortfall (ES) under normal distribution:", es_normal)


# Fitted T

def MLE_t(params, returns):
    df, loc, scale = params
    neg_LL = -1 * np.sum(stats.t.logpdf(returns, df=df, loc=loc, scale=scale))
    return (neg_LL)


def cal_VaR_MLE_t(returns, n=10000, alpha=0.05):
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0] - 1},
        {'type': 'ineq', 'fun': lambda x: x[2]}
    ]

    res = minimize(MLE_t, x0=[10, sum(returns) / len(returns), np.std(returns)], args=(returns,),
                   constraints=constraints)

    df, loc, scale = res.x
    simu_returns = stats.t.rvs(df, loc=loc, scale=scale, size=n)
    simu_returns.sort()
    n = alpha * simu_returns.size
    iup = int(np.ceil(n))
    idn = int(np.floor(n))
    VaR = (simu_returns[iup] + simu_returns[idn]) / 2

    ES = np.mean(simu_returns[0:idn])

    return -VaR, -ES, simu_returns


VaR_t, ES_t, simu_t = cal_VaR_MLE_t(data)

print("MLE T- VaR at 5% significance level:", VaR_t)
print("Expected Shortfall (ES) under T-distribution:", ES_t)

#### VaR hist
VaR_hist = -np.percentile(data, 5)
es_hist = ES(data)

print("VaR historical at 5% significance level:", VaR_hist)
print("Expected Shortfall (ES) under historical method:", es_hist)
