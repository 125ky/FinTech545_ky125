import numpy as np
from scipy.stats import kurtosis, skew
import pandas as pd

# Load the data from the CSV file
file_path = 'problem1.csv'
sample = pd.read_csv(file_path)

# question a
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
# Biased estimator for variance
sample_kurtosis = (((x_i - x_bar)**4).sum()) / n / (biased_variance**2)
print(sample_kurtosis)


# question b
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

# question c
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