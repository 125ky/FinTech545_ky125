import numpy as np

def missing_cov(x, skip_miss=True, fun=np.cov):
    n, m = x.shape
    n_miss = x.isnull().sum(axis=0)
    # if nothing missing, then directly calculate it
    if n_miss.sum() == 0:
        return fun(x, rowvar=False)
    idx_missing = [set(x.index[x[col].isnull()]) for col in x.columns]
    if skip_miss:
        # Skipping Missing rows to calculate cov
        rows = set(range(n))
        for c in range(m):
            rows -= idx_missing[c]
        rows = sorted(rows)
        return fun(x.iloc[rows, :], rowvar=False)
    else:
        # Pairwise for each cell to calculate cov
        cov = np.empty((m, m))
        for i in range(m):
            for j in range(i + 1):
                rows = set(range(n))
                for c in (i, j):
                    rows -= idx_missing[c]
                rows = sorted(rows)
                sub_matrix = fun(x.iloc[rows, [i, j]], rowvar=False)
                cov[i, j] = sub_matrix[0, 1]
                if i != j:
                    cov[j, i] = cov[i, j]
        return cov