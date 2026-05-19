from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return mae, rmse

def mape(y_true, y_pred):
    """Calcula o Mean Absolute Percentage Error."""
    return (np.abs((y_true - y_pred) / y_true)).mean() * 100