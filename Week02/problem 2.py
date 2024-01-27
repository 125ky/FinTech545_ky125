import pandas as pd
import statsmodels.api as sm
from scipy.stats import t, norm
from scipy.optimize import minimize
import numpy as np

data_path = 'problem2.csv'
data = pd.read_csv(data_path)

# Define the predictor and response variables
X = sm.add_constant(data['x'])
y = data['y']

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



# question c
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
data_x = pd.read_csv('problem2_x.csv')
data_x1 = pd.read_csv('problem2_x1.csv')

# Estimate the parameters for the multivariate normal distribution
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