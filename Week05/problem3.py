from RiskPackage.CalculateReturn import return_calc
from RiskPackage.simulateCopula import simulateCopula
import pandas as pd

prices = pd.read_csv('DailyPrices.csv')
returns = return_calc(prices)
returns -= returns.mean(numeric_only=True)

returns = returns.copy()
returns = returns.drop('Date', axis=1)

portfolio = pd.read_csv('portfolio.csv')
for stock in portfolio["Stock"]:
    portfolio.loc[portfolio['Stock'] == stock, 'Starting Price'] = prices.iloc[-1][stock]

portfolio.loc[portfolio['Portfolio'].isin(['A', 'B']), 'Distribution'] = 'T'
portfolio.loc[portfolio['Portfolio'] == 'C', 'Distribution'] = 'Normal'

portfolio_A = portfolio[portfolio['Portfolio'] == 'A'].copy()
portfolio_A = portfolio_A.drop('Portfolio', axis=1)
portfolio_B = portfolio[portfolio['Portfolio'] == 'B'].copy()
portfolio_B = portfolio_B.drop('Portfolio', axis=1)
portfolio_C = portfolio[portfolio['Portfolio'] == 'C'].copy()
portfolio_C = portfolio_C.drop('Portfolio', axis=1)

stocks_in_A = portfolio_A['Stock'].tolist()
returns_A = returns[stocks_in_A]
stocks_in_B = portfolio_B['Stock'].tolist()
returns_B = returns[stocks_in_B]
stocks_in_C = portfolio_C['Stock'].tolist()
returns_C = returns[stocks_in_C]

risk_A = simulateCopula(portfolio_A, returns_A)
print('Portfolio A:')
print(risk_A.iloc[-1] if isinstance(risk_A, pd.DataFrame) else risk_A[-1])

risk_B = simulateCopula(portfolio_B, returns_B)
print('Portfolio B:')
print(risk_B.iloc[-1] if isinstance(risk_B, pd.DataFrame) else risk_B[-1])

risk_C = simulateCopula(portfolio_C, returns_C)
print('Portfolio C:')
print(risk_C.iloc[-1] if isinstance(risk_C, pd.DataFrame) else risk_C[-1])