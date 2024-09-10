import pandas as pd

def identify_data_quality_issues(df):
    """
    Identify common data quality issues such as missing values and duplicates.
    """
    print("Checking for missing values...")
    print(df.isnull().sum())
    
    print("\nChecking for duplicates...")
    print(df.duplicated().sum())

def comprehensive_data_cleaning(df):
    """
    Perform comprehensive data cleaning including:
    - Removing duplicates
    - Handling missing values
    - Formatting columns
    """
    print("Cleaning data...")
    df = df.drop_duplicates()
    
    # Handling missing values (forward-fill and backward-fill as an example)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    return df

def impute_missing_values(df, strategy='ffill'):
    """
    Impute missing values in the DataFrame using the given strategy (default is 'forward fill').
    """
    print(f"Imputing missing values using {strategy} method.")
    if strategy == 'ffill':
        df.fillna(method='ffill', inplace=True)
    elif strategy == 'bfill':
        df.fillna(method='bfill', inplace=True)
    elif strategy == 'mean':
        df.fillna(df.mean(), inplace=True)
    
    return df
