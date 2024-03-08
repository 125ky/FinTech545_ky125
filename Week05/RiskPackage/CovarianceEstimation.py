import numpy as np

# compute exponentially weighted covariance matrix of a dataframe
def ewCovar(x, lambda_):
    m, n = np.shape(x)
    weights = np.zeros(m)
    #  Remove means
    x_bar = np.mean(x, axis=0)
    x = x - x_bar
    # Calculate weights
    for i in range(m):
        weights[i] = (1 - lambda_) * lambda_ ** (m - i - 1)
    #Create a diagonal matrix from the normalized weights.
    weights_mat = np.diag(weights / sum(weights))
    # Calculate the weighted covariance matrix
    cov_matrix = np.transpose(x.values) @ weights_mat @ x.values
    return cov_matrix