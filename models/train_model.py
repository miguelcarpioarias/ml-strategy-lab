# models/train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import ta  # Technical Analysis library

def add_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["MACD"] = ta.trend.MACD(df["Close"]).macd_diff()
    df["Return"] = df["Close"].pct_change()
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)  # 1 if price goes up tomorrow

    df.dropna(inplace=True)
    return df

def train_model(data: pd.DataFrame):
    df = add_features(data)
    features = ["SMA_10", "SMA_50", "RSI", "MACD", "Return"]
    X = df[features]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    df["Prediction"] = model.predict(X)

    return model, df, accuracy
