import numpy as np


# Non-PSD fixes for correlation matrix
def chol_psd(root, a):
    n = a.shape[0]
    # Initialize the root matrix with 0 values
    root.fill(0.0)
    # Loop over columns
    for j in range(n):
        s = 0.0
        # If we are not on the first column, calculate the dot product of the preceding row values.
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])
        # Diagonal element
        temp = a[j, j] - s
        if not np.isfinite(temp):
            raise ValueError("Encountered non-finite temp, requires adjustment")
        if 0 >= temp >= -1e-8:
            temp = 0.0
        if temp < 0:
            # Temp is negative, should trigger fix_method instead of proceeding
            raise ValueError("Matrix not PSD, requires adjustment.")
        root[j, j] = np.sqrt(temp)

        # Check for the 0 eigenvalue. Just set the column to 0 if we have one.
        if root[j, j] == 0.0:
            root[j, j + 1:n] = 0.0
        else:
            # Update off-diagonal rows of the column
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (a[i, j] - s) * ir



def near_psd(a, epsilon=0.0):
 # The function returns the modified matrix "out," which is a near PSD matrix.
    n = a.shape[0]
    invSD = None
    out = a.copy()
    # Calculate the correlation matrix if we got a covariance matrix
    if not np.allclose(np.diag(out), 1.0):
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD
    # SVD, update the eigenvalue and scale
    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    T = 1.0 / (vecs * vecs @ vals)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T

    # Add back the variance
    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        out = invSD @ out @ invSD

    return out




#  Higham's (2002) nearest PSD correlation function
def _getAplus(A):
    vals, vecs = np.linalg.eigh(A)
    vals[vals < 0] = 0
    return vecs @ np.diag(vals) @ vecs.T


def _getPS(A, W):
    # return: Scaled nearest PSD matrix to 'W05 * A * W05'
    W05 = np.sqrt(W)
    iW = np.linalg.inv(W05)
    return iW @ _getAplus(W05 @ A @ W05) @ iW


def _getPu(A, W):
 # return an Adjusted "A" matrix with 1 on the diagonal.
    Aret = A.copy()
    np.fill_diagonal(Aret, 1)
    return Aret


def wgtNorm(A, W):
    # return: Weighted norm of A
    W05 = np.sqrt(W) @ A @ np.sqrt(W)
    return np.sum(W05 * W05)


def higham_nearestPSD(pc, W=None, epsilon=1e-9, maxIter=100, tol=1e-9):
    n = pc.shape[0]
    if W is None:
        W = np.diag(np.ones(n))

    deltaS = 0
    invSD = None
    Yk = pc.copy()

    # Calculate the correlation matrix if we got a covariance
    if not np.allclose(np.diag(Yk), 1.0):
        invSD = np.diag(1.0 / np.sqrt(np.diag(Yk)))
        Yk = invSD @ Yk @ invSD

    Yo = Yk.copy()

    norml = np.finfo(np.float64).max
    i = 1

    while i <= maxIter:
        Rk = Yk - deltaS
        Xk = _getPS(Rk, W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W)
        norm = wgtNorm(Yk - Yo, W)
        minEigVal = np.min(np.real(np.linalg.eigvals(Yk)))

        if abs(norm - norml) < tol and minEigVal > -epsilon:
            break
        norml = norm
        i += 1

    if i < maxIter:
        print(f"Converged in {i} iterations.")
    else:
        print("Convergence failed after {} iterations".format(i - 1))

    # Add back the variance
    if invSD is not None:
        invSD = np.diag(1 / np.diag(invSD))
        Yk = invSD @ Yk @ invSD

    return Yk


def is_psd(A, tol=1e-9):
    # Returns true if A is a PSD matrix
    eigenvalues = np.linalg.eigvalsh(A)
    return np.all(eigenvalues > -tol)



# simulation
def simulate_normal(N, cov, mean=None, seed=1234, fix_method=near_psd):
    n, m = cov.shape
    if n != m:
        raise ValueError(f"Covariance Matrix is not square ({n},{m})")

    if mean is None:
        mean = np.zeros(n)
    elif len(mean) != n:
        raise ValueError(f"Mean ({len(mean)}) is not the size of cov ({n},{n})")

    # Cholesky Decomposition
    l = np.zeros_like(cov)  # Initialize l for modification in chol_psd
    # Attempt standard Cholesky decomposition
    try:
        l = np.linalg.cholesky(cov)  # NumPy returns upper triangular
    except np.linalg.LinAlgError:  # If not PD check PSD and then use chol_psd()
        print("Standard Cholesky Failed: nonPD matrix input")
        try:
            chol_psd(l, cov)  # Try chol_psd with the original covariance matrix
        except ValueError:  # If chol_psd fails, matrix is not positive semi-definite
            print("PSD Cholesky Failed: nonPSD matrix input")
            fixed_cov = fix_method(cov)  # Use fix_method to approximate a PSD matrix
            chol_psd(l, fixed_cov)  # Retry chol_psd with the fixed covariance matrix

    # Initialize out matrix
    # out = np.zeros((n, N))
    # Generate random standard normals
    np.random.seed(seed)
    out = np.random.normal(0.0, 1.0, size=(n, N))

    # Apply the Cholesky root and transpose
    out = np.dot(l, out)

    # Add the mean to each column
    for i in range(n):
        out[:, i] += mean[i]

    return out


# Simulate from PCA
def simulate_pca(a, nsim, pctExp=1, mean=None, seed=1234):
    n = a.shape[0]

    if mean is None:
        _mean = np.zeros(n)
    else:
        _mean = np.array(mean)

    # Eigenvalue decomposition
    vals, vecs = np.linalg.eigh(a)
    vals = np.real(vals)
    vecs = np.real(vecs)
    # Sort values and vectors in descending order
    idx = vals.argsort()[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    tv = np.sum(vals)

    posv = np.where(vals >= 1e-8)[0]
    if pctExp < 1:
        pct = 0.0
        for i in posv:
            pct += vals[i] / tv
            if pct >= pctExp:
                posv = posv[:np.where(posv == i)[0][0] + 1]
                break
    vals = vals[posv]
    vecs = vecs[:, posv]

    # Construct B matrix
    B = vecs @ np.diag(np.sqrt(vals))

    # Generate random samples
    np.random.seed(seed)
    r = np.random.randn(vals.shape[0], nsim)
    print(B.shape, r.shape)
    out = (B @ r)

    # Add the mean
    for i in range(n):
        out[:, i] += _mean[i]

    return out