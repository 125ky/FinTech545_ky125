import numpy as np

n = 1000000
sigma = 1
price = 100
returns = np.random.normal(0, sigma, n)

classical_price = price + returns
arithmetic_price = price * (1+returns)
log_price = price * np.exp(returns)

classical_mean = np.mean(classical_price)
classical_std = np.std(classical_price)

arithmetic_mean = np.mean(arithmetic_price)
arithmetic_std = np.std(arithmetic_price)

log_mean = np.mean(log_price)
log_std = np.std(log_price)
print(classical_mean, classical_std, arithmetic_mean, arithmetic_std, log_mean, log_std)
