# strategies/sma_crossover.py

import pandas as pd

def apply_sma_strategy(data: pd.DataFrame, short_window: int = 20, long_window: int = 50):
    data = data.copy()
    data["SMA_Short"] = data["Close"].rolling(window=short_window).mean()
    data["SMA_Long"] = data["Close"].rolling(window=long_window).mean()

    # Signal: 1 = Buy, -1 = Sell
    data["Signal"] = 0
    data.loc[data["SMA_Short"] > data["SMA_Long"], "Signal"] = 1
    data.loc[data["SMA_Short"] < data["SMA_Long"], "Signal"] = -1
    data["Position"] = data["Signal"].shift()

    return data
