# FinTech545_ky125
Assignment for FinTech 545

# all libraries needed：
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from scipy.stats import t, norm
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import kurtosis, skew


# run problem 1.py
import numpy as np
from scipy.stats import kurtosis, skew
import pandas as pd
file_path = 'problem1.csv'
sample = pd.read_csv(file_path)
# problem 1 question a
n = len(sample)
# sample mean
x_bar = sample['x'].mean()
x_i = sample['x'].values
print(x_bar)
# Sample variance σ̂²(x)
sample_variance = ((x_i - x_bar)**2).sum() / (n - 1)
biased_variance = ((x_i - x_bar)**2).sum() / n
print(sample_variance)
# sample skewness
sample_skewness = (((x_i - x_bar)**3).sum()) / n / (biased_variance**1.5)
print(sample_skewness)
# sample kurtosis
sample_kurtosis = (((x_i - x_bar)**4).sum()) / n / (biased_variance**2)
print(sample_kurtosis)

# problem 1 question b
# Calculate the mean
mean = float(np.mean(sample))
# Calculate the variance (unbiased)
variance = float(np.var(sample, ddof=1))
# Calculate the skewness (biased)
skewness = float(skew(sample, bias=True))
# Calculate the kurtosis (biased, Fisher's definition which subtracts 3)
# To get the non-excess kurtosis, set 'fisher=True'
kurt = float(kurtosis(sample, bias=True, fisher=False))
print(mean)
print(variance)
print(skewness)
print(kurt)

# problem 1 question c
if mean == x_bar:
    print ('The mean is unbiased.')
else:
    print('The mean is biased.')

if variance == sample_variance:
    print ('The variance is unbiased.')
else:
    print('The variance is biased.')
if skewness == sample_skewness:
    print ('The skewness is biased.')
if kurt == sample_kurtosis:
    print ('The kurtosis is biased.')

Output for problem 1.py
![image](https://github.com/125ky/FinTech545_ky125/assets/157724459/341b929f-31f8-497a-9d43-f1f5064ee840)

# run problem 2.py
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t, norm
from scipy.optimize import minimize
import numpy as np

data_path = 'problem2.csv'
data = pd.read_csv(data_path)

X = sm.add_constant(data['x'])
y = data['y']
# problem 2 question a
# Fit the OLS model
model_ols = sm.OLS(y, X).fit()

# Get the beta coefficients and standard deviation of the errors for OLS
beta_ols = model_ols.params[:2]
std_dev_ols_error = np.std(model_ols.resid)
print("ols for beta:", beta_ols, "ols for sigma:", std_dev_ols_error, model_ols.summary())

# MLE for regression for normal distribution
def regression_log_likelihood(params, X, y):
    beta = params[:-1]
    s = params[-1]

    n = len(y)
    e = y - X.dot(beta)
    return -n/2 * np.log(s**2 * 2 * np.pi) - np.sum(e**2) / (2 * s**2)

# MLE optimization
initial_guess = np.concatenate([np.zeros(X.shape[1]), [1]])
result = minimize(lambda params: -regression_log_likelihood(params, X, y),
                  initial_guess, bounds=[(None, None)]*(X.shape[1] + 1))

# get result
if result.success:
    beta_hat = result.x[:-1]
    sigma_hat = result.x[-1]
    print("MLE for beta_hat：", beta_hat)
    print("MLE for sigma_hat：", sigma_hat)
else:
    print("Failure：", result.message)

# problem 2 question b
# mle for regression for t distribution
def t_distribution_log_likelihood(params, X, y):
    beta = params[:-2]
    sigma = params[-2]
    df = params[-1]  # degree of freedom
    if sigma <= 0 or df <= 2:
        return -np.inf

    n = len(y)
    e = y - X.dot(beta)
    ll = np.sum(t.logpdf(e, df, scale=sigma))
    return ll

# MLE optimization
initial_guess_t = np.concatenate([np.zeros(X.shape[1]), [1, 3]])
result_t = minimize(lambda params: -t_distribution_log_likelihood(params, X, y),
                  initial_guess_t, bounds=[(None, None)]*(X.shape[1] + 2))

# get result
if result_t.success:
    beta_hat_t = result_t.x[:-2]
    sigma_hat_t = result_t.x[-2]
    df_hat_t = result_t.x[-1]
    print("MLE for beta_hat_t：", beta_hat_t)
    print("MLE for sigma_hat_t：", sigma_hat_t)
    print("MLE for df_hat_t：", df_hat_t)
else:
    print("Failure：", result_t.message)

# Calculating AIC values for each model
n = len(y)
k_ols = len(model_ols.params)
# aic_ols = 2 * k_ols - 2 * model_ols.llf
k_normal = len(result.x)
aic_normal = 2 * k_normal - 2 * (-result.fun)
k_t = len(result_t.x)
aic_t = 2 * k_t - 2 * (-result_t.fun)
print("Log likelihood_normal:", result.fun,"Log likelihood_t:", result_t.fun)
print("AIC_normal:", aic_normal,"AIC_t:", aic_t)

# problem 2 question c
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
data_x = pd.read_csv('problem2_x.csv')
data_x1 = pd.read_csv('problem2_x1.csv')

x_mean = data_x.mean()
x_cov = data_x.cov()

# Function to compute conditional distribution parameters for X2 given X1
def conditional_distribution(x1_obs, mean, cov):
    # Mean and covariance for X1 and X2
    mu_x1, mu_x2 = mean
    sigma_x1, sigma_x2 = np.sqrt(cov[0,0]), np.sqrt(cov[1,1])
    sigma_x1x2 = cov[0,1]
    # Conditional mean and variance of X2 given X1
    mu_cond = mu_x2 + sigma_x1x2 * (1/sigma_x1**2) * (x1_obs - mu_x1)
    sigma_cond = sigma_x2**2 - sigma_x1x2**2 * (1/sigma_x1**2)
    return mu_cond, sigma_cond

# Compute the conditional distribution for each observed value in data_x1
conditional_means = []
conditional_variances = []
for x1_obs in data_x1['x1']:
    mu_cond, sigma_cond = conditional_distribution(x1_obs, x_mean, x_cov.values)
    conditional_means.append(mu_cond)
    conditional_variances.append(sigma_cond)

# Convert to arrays for plotting
conditional_means = np.array(conditional_means)
conditional_variances = np.array(conditional_variances)
confidence_intervals = 1.96 * np.sqrt(conditional_variances)

# Plot with shaded 95% confidence interval
plt.figure(figsize=(10, 6))

# Plot the expected value line for X2
plt.plot(data_x1['x1'], conditional_means, label='Expected X2')

# Shade the area for the 95% confidence interval
plt.fill_between(data_x1['x1'],
                 conditional_means - confidence_intervals,
                 conditional_means + confidence_intervals,
                 color='green' ,alpha=0.5, label='95% Confidence Interval')

plt.title('Expected X2 with 95% Confidence Interval Given X1')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)
plt.show()

# Output for problem 2.py
![image](https://github.com/125ky/FinTech545_ky125/assets/157724459/22c65840-89a1-48ed-9549-24629e938a6c)
![image](https://github.com/125ky/FinTech545_ky125/assets/157724459/f16333d4-ede1-4d5e-942c-13bff0d3685d)
![image](https://github.com/125ky/FinTech545_ky125/assets/157724459/8e72b90e-4d2e-4f03-9447-30d1c45016ca)


# run problem 3.py
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

data = pd.read_csv('problem3.csv')
# Function to fit ARIMA models and calculate AIC and BIC
def fit_arima(data, ar_order, ma_order):
    model = ARIMA(data, order=(ar_order, 0, ma_order))
    model_fit = model.fit()
    return model_fit.aic, model_fit.bic

# Fit AR models with orders from 1 to 3
ar_aic_bic_values = [fit_arima(data.x, r, 0) for r in range(1, 4)]

# Fit MA models with orders from 1 to 3
ma_aic_bic_values = [fit_arima(data.x, 0, r) for r in range(1, 4)]

print(ar_aic_bic_values)
print(ma_aic_bic_values)


# Output for problem 3.py
![image](https://github.com/125ky/FinTech545_ky125/assets/157724459/38276dcc-dca8-4f54-996b-e5ad75f06d08)

