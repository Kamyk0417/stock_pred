import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def fetch_stock_data(ticker_symbol, period, interval):
    """
    Fetch historical stock data for a given ticker symbol.

    Parameters:
    ticker_symbol (str): The stock ticker symbol.
    period (str): The period over which to fetch data (default is '1mo').
    interval (str): The data interval (default is '1d').

    Returns:
    DataFrame: A pandas DataFrame containing the historical stock data.
    """
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period, interval)
    return hist

def prepare_data(ticker_symbol, period, interval):
    """
    Prepare the stock data by adding technical indicators and target variable.

    Parameters:
    data (DataFrame): The historical stock data.

    Returns:
    DataFrame: The prepared data with features and target variable.
    """
    data = fetch_stock_data(ticker_symbol, period, interval)
    data['Date'] = pd.to_datetime(data.index)
    data.set_index('Date', inplace=True)

    diffs = data['Close'] - data['Open']
    ups = [int((1 - np.sign(d)) / 2) for d in diffs]
    data['Up=0/Down=1'] = ups
    data.insert(loc=4, column='Diff', value=diffs)
    
    for i in range(10):
        data[f'Close_lag_{i+1}'] = data['Close'].shift(i+1)

    data['Return_1'] = data['Close'].pct_change(1)
    data['Return_5'] = data['Close'].pct_change(5)
    data['Return_15'] = data['Close'].pct_change(15)
    data['SMA_10'] = data['Close'].rolling(10).mean()
    data['SMA_30'] = data['Close'].rolling(30).mean()
    data['SMA_ratio'] = data['SMA_10'] / data['SMA_30']
    data['Volatility_10'] = data['Close'].pct_change().rolling(10).std()
    data['Volatility_30'] = data['Close'].pct_change().rolling(30).std()
    
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    data['Updown_next'] = data['Up=0/Down=1'].shift(-1)

    data = data.dropna()

    X = data[['Return_1', 'Return_5', 'Return_15', 'SMA_ratio', 'Volatility_10', 'Volatility_30', 'RSI']]
    y = data['Updown_next']


    return X,y

def train_prep(ticker_symbol, period, interval):
    X,y = prepare_data(ticker_symbol, period, interval)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test