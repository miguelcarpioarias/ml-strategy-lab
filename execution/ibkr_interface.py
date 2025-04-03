#ibkr_interface.py
# execution/ibkr_interface.py

from ib_insync import *

def connect_ibkr():
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=1)  # Localhost, paper trading port
    return ib

def place_order(ib, symbol, qty=1, action="BUY"):
    contract = Stock(symbol, 'SMART', 'USD')
    ib.qualifyContracts(contract)

    order = MarketOrder(action, qty)
    trade = ib.placeOrder(contract, order)
    ib.sleep(1)  # Give it a moment to process
    return trade
