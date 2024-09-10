from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model and return the fitted model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(y_test, y_pred):
    """
    Evaluate model performance and return key metrics: RMSE, MAE, R-squared.
    """
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R-squared': r2
    }
