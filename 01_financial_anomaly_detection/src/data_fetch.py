import yfinance as yf
import pandas as pd

def fetch_data(tickers, start, end):
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end)
        df.dropna(inplace=True)
        data[ticker] = df
    return data
