import yfinance as yf
import pandas as pd

def get_stock_data(ticker="AAPL", start="2018-01-01", end="2025-06-26"):
    # Download data from Yahoo Finance df has columns like 'Open', 'High', 'Low', 'Close', 'Adj Close', and 'Volume'.
    df = yf.download(ticker, start=start, end=end)
    
    # Return only the 'Close' price at the end of each trading day
    return df[['Close']]

import matplotlib.pyplot as plt

def preprocess(df):
    # Add a new column called 'Prediction' which is shifted 'Close' by 1 step
    df['Prediction'] = df['Close'].shift(-1)
    
    # Remove the last row (because its 'Prediction' is NaN)
    df = df.dropna()
    
    return df

def visualize_data(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df['Close'], label="Close Price", color='blue')
    plt.title("Historical Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

