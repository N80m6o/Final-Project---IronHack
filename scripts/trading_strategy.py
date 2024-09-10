import numpy as np
import pandas as pd

def trading_strategy(df, y_pred, initial_balance=1000):
    """
    Implement a simple trading strategy based on predicted prices.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing 'close' and other features
    y_pred (np.array): Predicted prices from the model
    initial_balance (float): Initial trading capital
    
    Returns:
    df: DataFrame with signals, portfolio value
    final_portfolio_value: Final portfolio value at the end of the simulation
    """
    df['predicted_close'] = y_pred
    df['signal'] = 0
    df.loc[df['predicted_close'] > df['close'], 'signal'] = 1  # Buy signal
    df.loc[df['predicted_close'] < df['close'], 'signal'] = -1  # Sell signal

    position = 0  # Initial position (no holdings)
    portfolio_value = initial_balance  # Initial portfolio value

    portfolio_values = []
    for i, row in df.iterrows():
        signal = row['signal']
        close_price = row['close']
        
        if signal == 1 and position == 0:  # Buy
            position = portfolio_value / close_price  # Buy the coin with all capital
            portfolio_value = 0  # Capital is now in the coin
        elif signal == -1 and position > 0:  # Sell
            portfolio_value = position * close_price  # Sell all holdings
            position = 0  # Reset position to 0
        
        current_portfolio_value = portfolio_value if position == 0 else position * close_price
        portfolio_values.append(current_portfolio_value)

    df['portfolio_value'] = portfolio_values
    final_portfolio_value = portfolio_values[-1]
    return df, final_portfolio_value
