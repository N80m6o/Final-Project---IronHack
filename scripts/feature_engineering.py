import pandas as pd

def add_technical_indicators(df):
    """
    Add common technical indicators such as:
    - Simple Moving Average (SMA)
    - Exponential Moving Average (EMA)
    - Relative Strength Index (RSI)
    """
    print("Adding technical indicators...")

    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()

    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()

    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    return df

def add_lagged_features(df, column, n_lags=5):
    """
    Add lagged features to the DataFrame for the specified column.
    
    Parameters:
    - column (str): Column to create lags for (e.g., 'close')
    - n_lags (int): Number of lags to generate
    """
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df[column].shift(lag)
    return df
