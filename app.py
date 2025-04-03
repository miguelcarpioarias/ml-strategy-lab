import streamlit as st
from data.fetch_data import get_stock_data
from strategies.sma_crossover import apply_sma_strategy
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="ML Strategy Lab", layout="wide")
st.title(" ML-Driven Strategy Lab")

# Sidebar inputs
st.sidebar.header("Select Stock and Date Range")
ticker = st.sidebar.text_input("Stock Ticker (e.g. AAPL)", value="AAPL")
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")

# SMA Strategy Parameters
st.sidebar.header("SMA Crossover Strategy Settings")
short_window = st.sidebar.slider("Short SMA Window", min_value=5, max_value=50, value=20)
long_window = st.sidebar.slider("Long SMA Window", min_value=20, max_value=200, value=50)

# Load data and apply strategy
if ticker and start_date and end_date:
    data = get_stock_data(ticker, str(start_date), str(end_date))

    if not data.empty:
        # Apply SMA Crossover Strategy
        strategy_data = apply_sma_strategy(data, short_window, long_window)

        st.subheader(f"Price & SMA Crossover for {ticker}")

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(strategy_data.index, strategy_data["Close"], label="Close Price", alpha=0.5)
        ax.plot(strategy_data.index, strategy_data["SMA_Short"], label=f"SMA {short_window}", linestyle="--")
        ax.plot(strategy_data.index, strategy_data["SMA_Long"], label=f"SMA {long_window}", linestyle="--")

        # Plot buy/sell signals
        buy_signals = strategy_data[strategy_data["Position"] == 1]
        sell_signals = strategy_data[strategy_data["Position"] == -1]

        ax.plot(buy_signals.index, buy_signals["Close"], "^", markersize=10, color='green', label='Buy Signal')
        ax.plot(sell_signals.index, sell_signals["Close"], "v", markersize=10, color='red', label='Sell Signal')

        ax.legend()
        ax.set_title(f"{ticker} SMA Crossover Strategy")
        ax.set_ylabel("Price")
        st.pyplot(fig)

        # Basic Strategy Performance
        st.subheader(" Strategy Performance")
        strategy_data["Returns"] = strategy_data["Close"].pct_change()
        strategy_data["Strategy_Returns"] = strategy_data["Returns"] * strategy_data["Position"].shift()

        total_return = strategy_data["Strategy_Returns"].sum()
        total_trades = strategy_data["Position"].diff().abs().sum() / 2  # count entries

        st.write(f"**Total Strategy Return:** {round(total_return * 100, 2)}%")
        st.write(f"**Number of Trades Executed:** {int(total_trades)}")

    else:
        st.warning("No data found. Please check your inputs.")
