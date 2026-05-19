import yfinance as yf
import pandas as pd


def download_data(symbol='PETR4.SA', start='2018-01-01', end=None):
    if end is None:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')

    df = yf.download(symbol, start=start, end=end)

    # Corrigir colunas MultiIndex
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Selecionar features
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    df.to_csv('data/raw/data.csv')

    return df


if __name__ == "__main__":
    download_data()