# data/fetch_data.py

import yfinance as yf
import pandas as pd

def get_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch historical stock data using yfinance.
    """
    data = yf.download(ticker, start=start, end=end)
    data.dropna(inplace=True)
    return data
