import pandas as pd
import numpy as np
from scipy.stats import norm, t


def pca_simulation(df, N=100000, mean=None, seed=1234, pctExp=0.99):
    # Error Checking
    m, n = df.shape
    if n != m:
        raise ValueError(f"Covariance Matrix is not square ({n},{m})")

    # Initialize the output
    out = np.zeros((N, n))

    # Set mean
    if mean is None:
        mean = np.zeros(n)
    else:
        if len(mean) != n:
            raise ValueError(f"Mean ({len(mean)}) is not the size of cov ({n},{n})")

    eigenvalues, eigenvectors = np.linalg.eig(df)

    # Get the indices that would sort eigenvalues in descending order
    indices = np.argsort(eigenvalues)[::-1]
    # Sort eigenvalues
    eigenvalues = eigenvalues[indices]
    # Sort eigenvectors according to the same order
    eigenvectors = eigenvectors[:, indices]

    tv = np.sum(eigenvalues)
    posv = np.where(eigenvalues >= 1e-8)[0]
    if pctExp <= 1:
        nval = 0
        pct = 0.0
        # How many factors needed
        for i in posv:
            pct += eigenvalues[i] / tv
            nval += 1
            if pct >= pctExp:
                break

    # If nval is less than the number of positive eigenvalues, truncate posv
    if nval < len(posv):
        posv = posv[:nval]

    # Filter eigenvalues based on posv
    eigenvalues = eigenvalues[posv]
    eigenvectors = eigenvectors[:, posv]

    B = eigenvectors @ np.diag(np.sqrt(eigenvalues))

    np.random.seed(seed)
    rand_normals = np.random.normal(0.0, 1.0, size=(N, len(posv)))
    out = np.dot(rand_normals, B.T) + mean

    return out.T


def simulateCopula(portfolio, returns):
    portfolio['CurrentValue'] = portfolio['Holding'] * portfolio['Starting Price']
    models = {}
    uniform = pd.DataFrame()
    standard_normal = pd.DataFrame()

    for stock in returns.columns:
        # If the distribution for the model is normal, fit the data with normal distribution.
        if portfolio.loc[portfolio['Stock'] == stock, 'Distribution'].iloc[0] == 'Normal':
            models[stock] = norm.fit(returns[stock])
            mu, sigma = norm.fit(returns[stock])

            # Transform the observation vector into a uniform vector using CDF.
            uniform[stock] = norm.cdf(returns[stock], loc=mu, scale=sigma)

            # Transform the uniform vector into a Standard Normal vector usig the normal quantile function.
            standard_normal[stock] = norm.ppf(uniform[stock])

        # If the distribution for the model is t, fit the data with normal t.
        elif portfolio.loc[portfolio['Stock'] == stock, 'Distribution'].iloc[0] == 'T':
            models[stock] = t.fit(returns[stock])
            nu, mu, sigma = t.fit(returns[stock])

            # Transform the observation vector into a uniform vector using CDF.
            uniform[stock] = t.cdf(returns[stock], df=nu, loc=mu, scale=sigma)

            # Transform the uniform vector into a Standard Normal vector usig the normal quantile function.
            standard_normal[stock] = norm.ppf(uniform[stock])

    # Calculate Spearman's correlation matrix
    spearman_corr_matrix = standard_normal.corr(method='spearman')

    nSim = 10000

    # Use the PCA to simulate the multivariate normal.
    simulations = pca_simulation(spearman_corr_matrix, N=nSim)
    simulations = pd.DataFrame(simulations.T, columns=[stock for stock in returns.columns])

    # Transform the simulations into uniform variables using standard normal CDF.
    uni = norm.cdf(simulations)
    uni = pd.DataFrame(uni, columns=[stock for stock in returns.columns])

    simulatedReturns = pd.DataFrame()
    # Transform the uniform variables into the desired data using quantile.
    for stock in returns.columns:
        # If the distribution for the model is normal, use the quantile of the normal distribution.
        if portfolio.loc[portfolio['Stock'] == stock, 'Distribution'].iloc[0] == 'Normal':
            mu, sigma = models[stock]
            simulatedReturns[stock] = norm.ppf(uni[stock], loc=mu, scale=sigma)

        # If the distribution for the model is t, use the quantile of the t distribution.
        elif portfolio.loc[portfolio['Stock'] == stock, 'Distribution'].iloc[0] == 'T':
            nu, mu, sigma = models[stock]
            simulatedReturns[stock] = t.ppf(uni[stock], df=nu, loc=mu, scale=sigma)

    simulatedValue = pd.DataFrame()
    pnl = pd.DataFrame()
    # Calculate the daily prices for each stock
    for stock in returns.columns:
        currentValue = portfolio.loc[portfolio['Stock'] == stock, 'CurrentValue'].iloc[0]
        simulatedValue[stock] = currentValue * (1 + simulatedReturns[stock])
        pnl[stock] = simulatedValue[stock] - currentValue

    risk = pd.DataFrame(columns=["Stock", "VaR95", "ES95", "VaR95_Pct", "ES95_Pct"])
    w = pd.DataFrame()

    for stock in pnl.columns:
        i = risk.shape[0]
        risk.loc[i, "Stock"] = stock
        risk.loc[i, "VaR95"] = -np.percentile(pnl[stock], 5)
        risk.loc[i, "VaR95_Pct"] = risk.loc[i, "VaR95"] / \
                                   portfolio.loc[portfolio['Stock'] == stock, 'CurrentValue'].iloc[0]
        risk.loc[i, "ES95"] = -pnl[stock][pnl[stock] <= -risk.loc[i, "VaR95"]].mean()
        risk.loc[i, "ES95_Pct"] = risk.loc[i, "ES95"] / portfolio.loc[portfolio['Stock'] == stock, 'CurrentValue'].iloc[
            0]

        # Determine the weights for the two stock
        w.at['Weight', stock] = portfolio.loc[portfolio['Stock'] == stock, 'CurrentValue'].iloc[0] / portfolio[
            'CurrentValue'].sum()

    # Calculate the total pnl.
    pnl['Total'] = 0
    for stock in returns.columns:
        pnl['Total'] += pnl[stock]

    i = risk.shape[0]
    risk.loc[i, "Stock"] = 'Total'
    risk.loc[i, "VaR95"] = -np.percentile(pnl['Total'], 5)
    risk.loc[i, "VaR95_Pct"] = risk.loc[i, "VaR95"] / portfolio['CurrentValue'].sum()
    risk.loc[i, "ES95"] = -pnl['Total'][pnl['Total'] <= -risk.loc[i, "VaR95"]].mean()
    risk.loc[i, "ES95_Pct"] = risk.loc[i, "ES95"] / portfolio['CurrentValue'].sum()
    return risk