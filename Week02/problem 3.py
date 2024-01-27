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