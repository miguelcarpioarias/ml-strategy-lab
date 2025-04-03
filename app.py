# app.py

import streamlit as st
from data.fetch_data import get_stock_data
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Strategy Lab", layout="wide")

st.title("ML-Driven Strategy Lab")

# Sidebar inputs
st.sidebar.header("Select Stock and Date Range")
ticker = st.sidebar.text_input("Stock Ticker (e.g. AAPL)", value="AAPL")
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")

if ticker and start_date and end_date:
    data = get_stock_data(ticker, str(start_date), str(end_date))

    if not data.empty:
        st.subheader(f"Price Chart for {ticker}")
        st.line_chart(data["Close"])
    else:
        st.warning("No data found. Please check your inputs.")
