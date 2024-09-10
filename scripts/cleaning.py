import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Cleaning and Wrangling

def clean_data(df):
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values(by='timestamp')
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values (forward fill and backward fill)
    df = df.ffill()  # Forward fill missing values
    df = df.bfill()  # Backward fill remaining missing values
    
    # Ensure all data types are correct (timestamp is datetime, others are floats)
    df = df.astype({
        'open': 'float',
        'high': 'float',
        'low': 'float',
        'close': 'float',
        'volume': 'float'
    })
    
    return df

def identify_data_quality_issues(df):
    """
    Identify and summarize data quality issues such as missing values, duplicates, and irrelevant columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    None: Prints out data quality issues and their impact on analysis.
    """
    print("### Data Quality Issues ###")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"Missing values:\n{missing_values}\n")
    if missing_values.any():
        print("Impact: Missing values can distort model training or affect accuracy. Imputation or removal is needed.")
    
    # Check for duplicates
    duplicate_rows = df.duplicated().sum()
    print(f"Duplicate rows: {duplicate_rows}\n")
    if duplicate_rows > 0:
        print("Impact: Duplicates can bias the analysis, leading to incorrect conclusions or overfitting.")
    
    # Check for irrelevant columns (example: columns with constant values)
    constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    print(f"Irrelevant columns (constant values): {constant_columns}\n")
    if constant_columns:
        print("Impact: Constant columns do not contribute useful information to the analysis and can be removed.")

    print("### Data Quality Summary Completed ###")


def comprehensive_data_cleaning(df):
    """
    Clean the data by addressing all identified data quality issues.
    - Handles missing values
    - Removes duplicates
    - Drops unnecessary columns
    - Formats data appropriately
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    # Handle missing values (fill forward and backward)
    df = df.ffill()  # Forward fill
    df = df.bfill()  # Backward fill

    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Drop irrelevant columns
    irrelevant_columns = [col for col in df.columns if df[col].nunique() == 1]
    df.drop(columns=irrelevant_columns, inplace=True)
    
    # Convert timestamp to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df

def impute_missing_values(df):
    """
    Impute missing values in the DataFrame using appropriate strategies.
    - Numeric columns: Fill with the mean value
    - Time-series columns: Forward and backward fill
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: DataFrame with imputed missing values
    """
    # Handle missing numeric data by filling with the mean
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # For time series data, apply forward and backward fill
    df = df.ffill()  # Forward fill
    df = df.bfill()  # Backward fill
    
    return df
