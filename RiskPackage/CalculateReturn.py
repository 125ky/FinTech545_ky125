import pandas as pd
import numpy as np

def return_calc(prices_df, method="DISCRETE", date_column="Date"):
    # Check if the date column is in the DataFrame
    if date_column not in prices_df.columns:
        raise ValueError(f"dateColumn: {date_column} not in DataFrame.")
    # Selecting columns except the date column
    assets = [col for col in prices_df.columns if col != date_column]

    # Convert prices to a numpy matrix for calculations
    p = prices_df[assets].values

    # Calculating the price ratios
    p2 = p[1:] / p[:-1]

    # Applying the selected return calculation method
    if method.upper() == "DISCRETE":
        p2 = p2 - 1
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in (\"LOG\", \"DISCRETE\")")

    # Aligning the dates with the returns

    dates = prices_df[date_column].iloc[1:]

    # Creating a DataFrame from the returns

    returns_df = pd.DataFrame(p2, columns=assets, index=dates.index)

    # Merging the returns with the dates

    out = pd.concat([prices_df[date_column], returns_df], axis=1).dropna()

    return out