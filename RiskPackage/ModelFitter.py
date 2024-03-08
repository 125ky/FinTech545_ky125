import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
from scipy.stats import skew, kurtosis, norm, t, multivariate_normal, gaussian_kde

# Fitted Models
class FittedModel:
    def __init__(self, beta, error_model, eval_func, errors, u):
        self.beta = beta
        self.error_model = error_model
        self.eval_func = eval_func
        self.errors = errors
        self.u = u


# General t sum ll function
def general_t_ll(mu, s, nu, x):
    """
        Calculate the sum of the logarithms of the probability density function (pdf)
        values of a scaled and shifted t-distribution for a given set of data points.

        Parameters:
        :param: mu (float): The location parameter (mean) of the t-distribution.
        :param: s (float): The scale (sigma) factor applied to the t-distribution.
        :param: nu (int): The degrees of freedom for the t-distribution.
        :param: x (array-like): An array or list of data points to evaluate the t-distribution.

        Returns:
        :return: log_sum (float): The sum of the logarithms of the pdf values for the data points in 'x'.
        """
    # Scale and shift the t-distribution
    scaled_pdf = lambda x_val: t.pdf((x_val - mu) / s, nu) / s
    # Apply the scaled pdf to each element in x and sum their logs
    log_sum = np.sum(np.log(scaled_pdf(x)))
    return log_sum


def fit_general_t(x):
    # Approximate values based on moments
    start_m = np.mean(x)
    start_nu = 6.0 / kurtosis(x, fisher=False, bias=False) + 4
    start_s = np.sqrt(np.var(x) * (start_nu - 2) / start_nu)

    # Objective function to maximize (log-likelihood)
    def objective(mu, s, nu):
        return -general_t_ll(mu, s, nu, x)  # Negated for minimization

    # Initial parameters
    initial_params = np.array([start_m, start_s, start_nu])

    # Bounds for s and nu
    bounds = [(None, None), (1e-6, None), (2.0001, None)]

    # Optimization
    result = minimize(lambda params: objective(*params), initial_params, bounds=bounds)

    m, s, nu = result.x
    error_model = lambda val: t.pdf(val, nu, loc=m, scale=s)
    errors = x - m
    u = t.cdf(x, nu, loc=m, scale=s)

    # Quantile function
    def eval(u_val):
        return t.ppf(u_val, nu, loc=m, scale=s)

    # Return fitted model and parameters
    fitted_model = FittedModel(None, error_model, eval, errors, u)
    return fitted_model, (m, s, nu, error_model)


def fit_regression_t(y, x):
    """
    Fit a regression models with t-distributed errors
    :param y: 1-D array or similar iterable: The dependent variable
    :param x: 2-D array or similar iterable: The independent variable. Each row represents an observation and each
    column represents a different independent variable.
    :return: fitted_model_instance: Instance of FittedModel class
    """
    n = x.shape[0]

    global __x, __y
    __x = np.hstack((np.ones((n, 1)), x))
    __y = y

    nB = __x.shape[1]

    # Initial values based on moments and OLS
    b_start = np.linalg.inv(__x.T @ __x) @ __x.T @ __y
    e = __y - __x @ b_start
    start_m = np.mean(e)
    start_nu = 6.0 / stats.kurtosis(e, fisher=False, bias=False) + 4
    start_s = np.sqrt(np.var(e) * (start_nu - 2) / start_nu)

    # Optimization function
    def objective(params):
        m, s, nu = params[:3]
        B = params[3:]
        return -general_t_ll(m, s, nu, __y - __x @ B)

    # Initial parameters for optimization
    initial_params = np.concatenate(([start_m, start_s, start_nu], b_start))

    # Constraints for s and nu
    bounds = [(None, None), (1e-6, None), (2.0001, None)] + [(None, None)] * nB

    # Optimization
    result = minimize(objective, initial_params, bounds=bounds)

    m, s, nu = result.x[:3]
    beta = result.x[3:]

    # Fitted error model
    errorModel = lambda u: t.ppf(u, nu) * s + m

    # Function to evaluate the model for given x and u
    def eval_model(x, u):
        n = x.shape[0]
        _temp = np.hstack((np.ones((n, 1)), x))
        return _temp @ beta + errorModel(u)

    # Calculate the regression errors and their U values
    errors = y - eval_model(x, np.full(x.shape[0], 0.5))
    u = t.cdf(errors, nu) * s + m

    fitted_model_instance = FittedModel(beta, errorModel, eval_model, errors, u)
    return fitted_model_instance, (m, s, nu)


def fit_normal(x):
    # Calculate mean and standard deviation
    m = np.mean(x)
    s = np.std(x)

    # Create the error model based on the normal distribution
    error_model = lambda val: norm.pdf(val, m, s)

    # Calculate errors and CDF values
    errors = x - m
    u = norm.cdf(x, m, s)

    # Function to evaluate the quantile
    def eval(u):
        return norm.ppf(u, m, s)

    # Return the FittedModel object
    fitted_model = FittedModel(None, error_model, eval, errors, u)
    return fitted_model, (u, m, s)


def VaR(a, alpha=0.05):
    """
    Calculate the Value at Risk (VaR) for a given array of financial data. Used for Historic Simulation on a single
    return series in Project_04.

    Parameters:
    :param: a (array-like): An array of historical financial data (e.g., returns or prices).
    :param: alpha (float, optional): The confidence level, representing the probability of
                             exceeding the VaR. Default value is 0.05 (95% confidence level).

    Returns:
    :return: -v: float: The calculated Value at Risk (VaR). The value is returned as a negative
           number, indicating a potential loss in the context of the given confidence level.
    """
    x = np.sort(a)
    nup = int(np.ceil(len(a) * alpha))
    ndn = int(np.floor(len(a) * alpha))
    v = 0.5 * (x[nup] + x[ndn])

    return -v


def VaR_norm(sigma, mu=0, alpha=0.05):
    """
    Calculates the VaR for a normal distribution. Use for fitting data to a normal distribution (i.e., with fit_normal).
    Output of fit_normal is a fitted model instance and params.
    :param sigma: float: the standard deviation of the fitted model instance (params[2])
    :param mu: float: the mean. For relative / VaR difference from mean use default of 0. For absolute VaR use
    the mu of the fitted model instance (params[1])
    :param alpha: (float, optional): The confidence level, representing the probability of
                             exceeding the VaR. Default value is 0.05 (95% confidence level).
    :return: var: float: The calculated Value at Risk.
    """
    var = -norm.ppf(alpha, mu, sigma)
    return var


def VaR_t(nu, sigma, mu=0, alpha=0.05):
    """Calculates the VaR for a t-distribution. Use for fitting data to a general t-distribution (i.e., with
    fit_general_t). Out of fit_general_t is a fitted model instance and params.
    :param: nu: float: the degrees of freedom of the fitted model instance (params[2])
    :param: sigma: float: the standard deviation of the fitted model instance (params[1])
    :param: mu: float: the mean. For relative / VaR difference from mean use default of 0. For absolute VaR use the mu
    of the fitted model instance (params[0])
    :param: alpha: (float, optional): The confidence level, representing the probability of
                             exceeding the VaR. Default value is 0.05 (95% confidence level).
    :return: var: float: The calculated Value at Risk.
    """
    var = -t.ppf(alpha, nu, loc=mu, scale=sigma)
    return var

def ES(a, alpha=0.05):
    """
    Same as VaR, except that it returns the expected shortfall (a.k.a. conditional VaR) for an input series
    :param a: (array-like): An array of historical financial data (e.g., returns or prices).
    :param alpha: (float, optional): The confidence level, representing the probability of
                             exceeding the VaR. Default value is 0.05 (95% confidence level).
    :return: -es: float: The expected shortfall or expectation of the VaR.
    """
    x = np.sort(a)
    nup = int(np.ceil(len(a) * alpha))
    ndn = int(np.floor(len(a) * alpha))
    v = 0.5 * (x[nup] + x[ndn])
    es = np.mean(x[x<=v])
    return -es


def ES_norm(sigma, mu=0, alpha=0.05):
    """
    Same as VaR_norm, except that it returns the ES. Same use case.
    :param sigma: float: the standard deviation of the fitted model instance (params[2])
    :param mu: float: the mean. For relative / ES difference from mean use default of 0. For absolute ES use
    the mu of the fitted model instance (params[1])
    :param alpha: (float, optional): The confidence level, representing the probability of
                             exceeding the VaR. Default value is 0.05 (95% confidence level).
    :return: es: float: the ES value for the fitted model instance
    """
    z_score = norm.ppf(alpha)
    es = -mu + sigma * norm.pdf(z_score) / alpha
    return es


def ES_t(nu, sigma, mu=0, alpha=0.05):
    """
    Same as VaR_tan, except that it returns the ES. Same use case.
    :param nu: float: the degree of freedom of the fitted model instance (params[2])
    :param sigma: float: the standard deviation of the fitted model instance (params[1])
    :param mu: float: the mean. For relative / ES difference from mean use default of 0. For absolute ES use
    the mu of the fitted model instance (params[0])
    :param alpha: (float, optional): The confidence level, representing the probability of
                             exceeding the VaR. Default value is 0.05 (95% confidence level).
    :return: es: float: the ES value for the fitted model instance
    """
    # Calculate the VaR
    var = t.ppf(alpha, nu, loc=mu, scale=sigma)

    # Calculate the ES
    t_sim = t.rvs(nu, loc=mu, scale=sigma, size=10000)
    es = -np.mean(t_sim[t_sim <= var])
    return es



