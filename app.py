import streamlit as st
from data.fetch_data import get_stock_data
from strategies.sma_crossover import apply_sma_strategy
from models.train_model import train_model
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="ML Strategy Lab", layout="wide")
st.title("ML-Driven Strategy Lab")

# Sidebar inputs
st.sidebar.header("Select Stock and Date Range")
ticker = st.sidebar.text_input("Stock Ticker (e.g. AAPL)", value="AAPL")
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")

# Tabs for different strategies
tab1, tab2 = st.tabs(["SMA Crossover Strategy", " ML-Enhanced Strategy"])

# Load data once
data = None
if ticker and start_date and end_date:
    data = get_stock_data(ticker, str(start_date), str(end_date))

# ----------------------------
# Tab 1: SMA Crossover Strategy
# ----------------------------
with tab1:
    st.sidebar.subheader("SMA Strategy Settings")
    short_window = st.sidebar.slider("Short SMA", 5, 50, 20)
    long_window = st.sidebar.slider("Long SMA", 20, 200, 50)

    if data is not None and not data.empty:
        strategy_data = apply_sma_strategy(data, short_window, long_window)

        st.subheader(f"{ticker} Price & SMA Crossover")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(strategy_data.index, strategy_data["Close"], label="Close Price", alpha=0.6)
        ax.plot(strategy_data.index, strategy_data["SMA_Short"], label=f"SMA {short_window}", linestyle="--")
        ax.plot(strategy_data.index, strategy_data["SMA_Long"], label=f"SMA {long_window}", linestyle="--")

        # Signals
        buy_signals = strategy_data[strategy_data["Position"] == 1]
        sell_signals = strategy_data[strategy_data["Position"] == -1]
        ax.plot(buy_signals.index, buy_signals["Close"], "^", markersize=10, color='green', label='Buy')
        ax.plot(sell_signals.index, sell_signals["Close"], "v", markersize=10, color='red', label='Sell')

        ax.legend()
        ax.set_title(f"SMA Strategy for {ticker}")
        st.pyplot(fig)

        # Performance
        st.subheader("Strategy Performance")
        strategy_data["Returns"] = strategy_data["Close"].pct_change()
        strategy_data["Strategy_Returns"] = strategy_data["Returns"] * strategy_data["Position"].shift()

        total_return = strategy_data["Strategy_Returns"].sum()
        total_trades = strategy_data["Position"].diff().abs().sum() / 2

        st.write(f"**Total Return:** {round(total_return * 100, 2)}%")
        st.write(f"**Trades Executed:** {int(total_trades)}")
    else:
        st.warning("No data found. Please check your inputs.")

# ----------------------------
# Tab 2: ML-Enhanced Strategy
# ----------------------------
with tab2:
    if data is not None and not data.empty:
        st.subheader("Training Machine Learning Model...")
        model, ml_data, accuracy = train_model(data)
        st.success(f"Model trained with accuracy: **{round(accuracy * 100, 2)}%**")

        # Filter: Buy when prediction == 1
        ml_data["Position"] = ml_data["Prediction"].shift()  # simulate acting on yesterday's prediction
        ml_data["Returns"] = ml_data["Close"].pct_change()
        ml_data["Strategy_Returns"] = ml_data["Returns"] * ml_data["Position"]
        # Calculate cumulative returns
        ml_data["Cumulative_Strategy"] = (1 + ml_data["Strategy_Returns"]).cumprod()
        ml_data["Cumulative_BuyHold"] = (1 + ml_data["Returns"]).cumprod()

        st.subheader("📉 Cumulative Return Comparison")
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(ml_data.index, ml_data["Cumulative_Strategy"], label="ML Strategy")
        ax3.plot(ml_data.index, ml_data["Cumulative_BuyHold"], label="Buy & Hold", linestyle="--")
        ax3.legend()
        ax3.set_title("Cumulative Returns: ML Strategy vs Buy & Hold")
        st.pyplot(fig3)


        # Plot results
        st.subheader(f"{ticker} Price & ML Buy Predictions")
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(ml_data.index, ml_data["Close"], label="Close Price", alpha=0.6)

        buy_preds = ml_data[ml_data["Position"] == 1]
        ax2.plot(buy_preds.index, buy_preds["Close"], "^", markersize=10, color='blue', label='ML Buy')

        ax2.legend()
        ax2.set_title("ML-Enhanced Buy Signal Strategy")
        st.pyplot(fig2)

        # Show performance
        st.subheader("ML Strategy Performance")
        total_return = ml_data["Strategy_Returns"].sum()
        trade_count = ml_data["Position"].sum()

        st.write(f"**Total Return:** {round(total_return * 100, 2)}%")
        st.write(f"**Predicted Buy Days:** {int(trade_count)}")
        st.write(f"**Cumulative Return (ML):** {round((ml_data['Cumulative_Strategy'].iloc[-1] - 1) * 100, 2)}%")
        st.write(f"**Cumulative Return (Buy & Hold):** {round((ml_data['Cumulative_BuyHold'].iloc[-1] - 1) * 100, 2)}%")


    else:
        st.warning("No data found. Please check your inputs.")

# Execuation Interactive Brokers
from execution.ibkr_interface import connect_ibkr, place_order

st.subheader("Paper Trade via IBKR")
qty = st.number_input("Quantity", min_value=1, value=10)
trigger_trade = st.button("Place Buy Order (ML Confirmed)")

if trigger_trade:
    ib = connect_ibkr()
    trade = place_order(ib, ticker, qty=qty, action="BUY")
    st.success(f"Order Placed: {trade}")

