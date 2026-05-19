import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data():
    # Carrega o CSV e garante que estamos usando apenas a coluna de valores numéricos
    df = pd.read_csv('data/raw/data.csv', usecols=['Open', 'High', 'Low', 'Close', 'Volume'])
    return df


def scale_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return scaled, scaler


def create_sequences(data, window_size=60):
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        # Assumindo que 'Close' é a 4ª coluna (índice 3)
        y.append(data[i, 3])

    return np.array(X), np.array(y)


def train_test_split(X, y, split=0.8):
    split_index = int(len(X) * split)

    return (
        X[:split_index],
        X[split_index:],
        y[:split_index],
        y[split_index:]
    )