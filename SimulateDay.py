import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import datetime
import warnings
import logging
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from lightgbm import LGBMClassifier
import os
from sklearn.model_selection import cross_val_score

# Ignore warnings and set up logging
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Change the working directory to the script's directory


def predict_action(data: dict, model):
    """
    Predicts the action to take based on the input data and model.
    !! This function might break if given models without the predict method !!

    Parameters:
    data (dict): Dictionary containing the input data
    model (xgb.Booster): XGBoost model used for prediction
    """
    features = ['Volume', 'MA_10', 'MA_20', 'MA_50', 'MA_200', 'std_10',
                'std_20', 'std_50', 'std_200', 'upper_band_10', 'lower_band_10',
                'upper_band_20', 'lower_band_20', 'upper_band_50', 'lower_band_50',
                'upper_band_200', 'lower_band_200', 'Golden_Cross_Short', 'Golden_Cross_Medium',
                'Golden_Cross_Long', 'Death_Cross_Short', 'Death_Cross_Medium', 'Death_Cross_Long',
                'ROC', 'AVG_Volume_10', 'AVG_Volume_20', 'AVG_Volume_50', 'AVG_Volume_200', 'Doji',
                'Bullish_Engulfing', 'Bearish_Engulfing', 'MACD', 'Signal', 'MACD_Hist', 'TR', 'ATR',
                'RSI_10_Day', '10_Day_ROC', 'Resistance_10_Day', 'Support_10_Day', 'Resistance_20_Day',
                'Support_20_Day', 'Resistance_50_Day', 'Support_50_Day', 'Volume_MA_10', 'Volume_MA_20',
                'Volume_MA_50', 'OBV', 'Z-score']

    if type(model) == xgb.core.Booster:

        data_df = pd.DataFrame([data])

        # Select only the features used in training
        data_df = data_df[features]

        # Convert the DataFrame to DMatrix (required by XGBoost)
        dmatrix = xgb.DMatrix(data_df)

        # Predict using the loaded model
        prediction = model.predict(dmatrix)[0]
    else:
        data_df = pd.DataFrame([data])
        data_df = data_df[features]
        prediction = model.predict(data_df)[0]

    if prediction == 0:
        return 'Buy'
    elif prediction == 1:
        return 'Sell'
    else:
        return 'Hold'


def stock_market_simulation(model,
                            initial_cash: int,
                            days: int,
                            stock: pd.DataFrame,
                            existing_shares=0,
                            oneDay=False,
                            print_results=False,
                            masstrades=False):
    """
    Simulates the stock market using the given model and stock data.

    Parameters:
    model: Model used for prediction
    initial_cash: Initial cash available for trading
    days: Number of days to simulate
    stock: DataFrame containing stock data
    existing_shares: Number of shares already held
    oneDay: If True, only simulate for one day, otherwise pass the day number
    print_results: If True, print the results of the simulation
    masstrades: If True, allow mass trades (buy/sell 5 shares at once)

    Returns:
    modelDecisionDf: DataFrame containing the model decisions for each day
    cash: Final cash available after the simulation
    """
    # Initialize variables
    cash = initial_cash
    invested = cash
    shares_held = existing_shares
    portfolio_value = []
    scaled = scale_data(stock)
    modelDecisionDf = pd.DataFrame(
        columns=['Stock Name', 'Day', 'Action', 'Cash',
                 'Shares Held', 'Portfolio Value', 'Stock Price'])

    days = min(days, len(stock))

    # Go through each day and make a decision based on the model
    for i in range(days):
        # Get the stock price, strategy and day for the current day
        stock_price = stock['Close'].iloc[i]
        strategy = predict_action(scaled.iloc[i].to_dict(), model)
        day = oneDay if oneDay else i

        # Buy shares if the strategy is 'Buy' and cash is sufficient
        if strategy == 'Buy' and cash >= stock_price:
            # If the last 5 actions were 'Buy', buy 5 shares and mass trade is enabled
            if (len(modelDecisionDf) >= 5 and
                    (modelDecisionDf['Action'].tail(5) == 'Buy').all()
                    and cash >= stock_price * 5 and masstrades):
                # * 0.99  # Apply a 1% fee, if applicable
                cash -= (stock_price * 5)
                shares_held += 5
            # Otherwise, buy one share
            else:
                cash -= stock_price
                shares_held += 1
            if print_results:
                print(f"Day {day}: Bought 1 share at {
                      stock_price}, Cash left: {cash}")

        # Buy fractional shares if cash is insufficient for a full share
        elif strategy == 'Buy' and cash < stock_price:
            fractional_shares = cash / stock_price
            shares_held += fractional_shares
            cash = 0
            if print_results:
                print(f"Day {day}: Bought {fractional_shares} shares at {
                      stock_price}, Cash left: {cash}")

        # Sell shares if the strategy is 'Sell' and shares are held
        elif strategy == 'Sell' and shares_held > 0:
            # If the last 5 actions were 'Sell', sell 5 shares and mass trade is enabled
            if (modelDecisionDf['Action'].tail(5) == 'Sell').all() and shares_held >= 5 and masstrades:
                # * 0.99  # Apply a 1% fee, if applicable
                cash += (stock_price * 5)
                shares_held -= 5
            else:
                cash += stock_price
                shares_held -= 1
            if print_results:
                print(f"Day {day}: Sold shares at {stock_price}, Cash: {cash}")

        # Hold the current position if the strategy is 'Hold'
        elif strategy == 'Hold':
            if print_results:
                print(f"Day {day}: Holding, Cash: {
                      cash}, Shares held: {shares_held}")

        # Calculate the total portfolio value (cash + stock holdings)
        portfolio_value_at_time = cash + (shares_held * stock_price)
        portfolio_value.append(portfolio_value_at_time)
        stock_name = stock['Symbol'].iloc[0]

        # Add the decision to the DataFrame
        new_row = pd.DataFrame({
            'Stock Name': [stock_name],
            'Day': [day],
            'Date': [stock['Date'].iloc[i]],
            'Action': [strategy],
            'Stock Price': [stock_price],
            'Cash': [cash],
            'Shares Held': [shares_held],
            'Portfolio Value': [portfolio_value_at_time]
        })
        modelDecisionDf = pd.concat(
            [modelDecisionDf, new_row], ignore_index=True)

    # Final results
    final_portfolio_value = cash + (shares_held * stock['Close'].iloc[-1])
    if print_results:
        print(f'Total cash invested: {invested}')
        print(f'Stock {stock["Symbol"].iloc[0]}')
        print(f"Final Portfolio Value: {final_portfolio_value}")
        print(f"Cash: {cash}, Shares held: {shares_held}")

    return modelDecisionDf, cash


def determine_action(row):
    """
    Determines the action to take based on the stock data.
    Action determined based on the following conditions:
    - Buy: Golden Cross, MACD > Signal, RSI between 50 and 70
    - Sell: Death Cross, MACD < Signal, RSI > 80, Daily Return < -1%
    - Hold: All other cases
    """
    try:
        if row['Close'] != 0 and (((row['Close'] - row['close_lag1'])/(row['Close'])*100) > 1):
            return 2
        elif (row['Golden_Cross_Short'] == 1 or
              row['MACD'] > row['Signal'] or
              50 < row['RSI_10_Day'] < 70):
            return 0  # Buy
        elif (row['Death_Cross_Short'] == 1 or
              row['MACD'] < row['Signal'] or
              row['RSI_10_Day'] > 80 and
              row['Daily_Return'] < -0.01):
            return 1  # Sell
        else:
            return 2  # Hold
    except:
        return 2


def calculate_rsi(stock_df, window=10):
    """
    Calculates the Relative Strength Index (RSI) for the stock data.
    The RSI is a momentum oscillator that measures the speed and change of price movements.
    """
    # Calculate daily price changes
    delta = stock_df['Close'].diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate the average gain and average loss
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    # Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss

    # Calculate the RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi


def is_doji(row):
    """
    Returns True if the row has a doji candlestick pattern, False otherwise.
    A doji is a candlestick pattern that forms when the open and close are equal or very close to each other.
    """
    return abs(row['Close'] - row['Open']) <= (row['High'] - row['Low']) * 0.1


def is_bullish_engulfing(current_row, previous_row):
    """
    Returns True if the current row has a bullish engulfing pattern, False otherwise.
    A bullish engulfing pattern is a candlestick pattern that forms when a
    small red candle is followed by a large green candle that completely engulfs the previous candle.
    """
    # Example logic for identifying a bullish engulfing pattern
    if previous_row['Close'] < previous_row['Open'] and current_row['Close'] > current_row['Open'] and current_row['Close'] > previous_row['Open'] and current_row['Open'] < previous_row['Close']:
        return True
    return False


def is_bearish_engulfing(current_row, previous_row):
    """
    Returns True if the current row has a bearish engulfing pattern, False otherwise.
    A bearish engulfing pattern is a candlestick pattern that forms when a 
    small green candle is followed by a large red candle that completely engulfs the previous candle.
    """
    if previous_row['Close'] > previous_row['Open'] and current_row['Close'] < current_row['Open'] and current_row['Close'] < previous_row['Open'] and current_row['Open'] > previous_row['Close']:
        return True
    return False


def add_columns(stock_df):
    """
    Adds new columns to the stock data DataFrame.

    Parameters:
    stock_df (DataFrame): DataFrame containing stock data    
    """
    print(f'Adding columns...')
    # Create new columns with Returns
    stock_df['1_Day_Return'] = (
        stock_df['Close'] - stock_df['Close'].shift(1)) / stock_df['Close'].shift(1) * 100
    stock_df['5_Day_Return'] = (
        stock_df['Close'] - stock_df['Close'].shift(5)) / stock_df['Close'].shift(5) * 100
    stock_df['10_Day_Return'] = (
        stock_df['Close'] - stock_df['Close'].shift(10)) / stock_df['Close'].shift(10) * 100
    stock_df['20_Day_Return'] = (
        stock_df['Close'] - stock_df['Close'].shift(20)) / stock_df['Close'].shift(20) * 100
    stock_df['50_Day_Return'] = (
        stock_df['Close'] - stock_df['Close'].shift(50)) / stock_df['Close'].shift(50) * 100
    stock_df['200_Day_Return'] = (
        stock_df['Close'] - stock_df['Close'].shift(200)) / stock_df['Close'].shift(200) * 100

    stock_df['Best_Return_Window'] = stock_df[['1_Day_Return', '5_Day_Return',
                                               '10_Day_Return', '20_Day_Return', '50_Day_Return', '200_Day_Return']].idxmax(axis=1)
    stock_df['Best_Return'] = stock_df[['1_Day_Return', '5_Day_Return',
                                        '10_Day_Return', '20_Day_Return', '50_Day_Return', '200_Day_Return']].max(axis=1)
    stock_df['Best_Return_Window'] = stock_df['Best_Return_Window'].replace(
        '_Day_Return', '', regex=True)

    # Create lag columns
    stock_df['close_lag1'] = stock_df['Close'].shift(1)
    stock_df['close_lag2'] = stock_df['Close'].shift(2)
    stock_df['close_lag3'] = stock_df['Close'].shift(3)
    stock_df['close_lag4'] = stock_df['Close'].shift(5)
    stock_df['close_lag5'] = stock_df['Close'].shift(10)

    stock_df['volume_lag1'] = stock_df['Volume'].shift(1)
    stock_df['volume_lag2'] = stock_df['Volume'].shift(2)
    stock_df['volume_lag3'] = stock_df['Volume'].shift(3)
    stock_df['volume_lag4'] = stock_df['Volume'].shift(5)
    stock_df['volume_lag5'] = stock_df['Volume'].shift(10)

    # Create new columns with Moving Averages and Standard Deviations
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])

    stock_df['MA_10'] = stock_df.groupby('Symbol')['Close'].rolling(
        window=10).mean().reset_index(level=0, drop=True)
    stock_df['MA_20'] = stock_df.groupby('Symbol')['Close'].rolling(
        window=20).mean().reset_index(level=0, drop=True)
    stock_df['MA_50'] = stock_df.groupby('Symbol')['Close'].rolling(
        window=50).mean().reset_index(level=0, drop=True)
    stock_df['MA_200'] = stock_df.groupby('Symbol')['Close'].rolling(
        window=200).mean().reset_index(level=0, drop=True)

    stock_df['std_10'] = stock_df.groupby('Symbol')['Close'].rolling(
        window=10).std().reset_index(level=0, drop=True)
    stock_df['std_20'] = stock_df.groupby('Symbol')['Close'].rolling(
        window=20).std().reset_index(level=0, drop=True)
    stock_df['std_50'] = stock_df.groupby('Symbol')['Close'].rolling(
        window=50).std().reset_index(level=0, drop=True)
    stock_df['std_200'] = stock_df.groupby('Symbol')['Close'].rolling(
        window=200).std().reset_index(level=0, drop=True)

    # Create new columns with Bollinger Bands for each Moving Average
    stock_df['upper_band_10'] = stock_df['MA_10'] + (stock_df['std_10'] * 2)
    stock_df['lower_band_10'] = stock_df['MA_10'] - (stock_df['std_10'] * 2)

    stock_df['upper_band_20'] = stock_df['MA_20'] + (stock_df['std_20'] * 2)
    stock_df['lower_band_20'] = stock_df['MA_20'] - (stock_df['std_20'] * 2)

    stock_df['upper_band_50'] = stock_df['MA_50'] + (stock_df['std_50'] * 2)
    stock_df['lower_band_50'] = stock_df['MA_50'] - (stock_df['std_50'] * 2)

    stock_df['upper_band_200'] = stock_df['MA_200'] + (stock_df['std_200'] * 2)
    stock_df['lower_band_200'] = stock_df['MA_200'] - (stock_df['std_200'] * 2)

    # Create new columns Indicating Golden Cross and Death Cross
    stock_df['Golden_Cross_Short'] = np.where((stock_df['MA_10'] > stock_df['MA_20']) & (
        stock_df['MA_10'].shift(1) <= stock_df['MA_20'].shift(1)), 1, 0)
    stock_df['Golden_Cross_Medium'] = np.where((stock_df['MA_20'] > stock_df['MA_50']) & (
        stock_df['MA_20'].shift(1) <= stock_df['MA_50'].shift(1)), 1, 0)
    stock_df['Golden_Cross_Long'] = np.where((stock_df['MA_50'] > stock_df['MA_200']) & (
        stock_df['MA_50'].shift(1) <= stock_df['MA_200'].shift(1)), 1, 0)

    stock_df['Death_Cross_Short'] = np.where((stock_df['MA_10'] < stock_df['MA_20']) & (
        stock_df['MA_10'].shift(1) >= stock_df['MA_20'].shift(1)), 1, 0)
    stock_df['Death_Cross_Medium'] = np.where((stock_df['MA_20'] < stock_df['MA_50']) & (
        stock_df['MA_20'].shift(1) >= stock_df['MA_50'].shift(1)), 1, 0)
    stock_df['Death_Cross_Long'] = np.where((stock_df['MA_50'] < stock_df['MA_200']) & (
        stock_df['MA_50'].shift(1) >= stock_df['MA_200'].shift(1)), 1, 0)

    # Create new columns with Rate of Change and Average Volume

    stock_df['ROC'] = (
        (stock_df['Close'] - stock_df['Close'].shift(1)) / stock_df['Close'].shift(1)) * 100

    stock_df['AVG_Volume_10'] = stock_df.groupby('Symbol')['Volume'].rolling(
        window=10).mean().reset_index(level=0, drop=True)
    stock_df['AVG_Volume_20'] = stock_df.groupby('Symbol')['Volume'].rolling(
        window=20).mean().reset_index(level=0, drop=True)
    stock_df['AVG_Volume_50'] = stock_df.groupby('Symbol')['Volume'].rolling(
        window=50).mean().reset_index(level=0, drop=True)
    stock_df['AVG_Volume_200'] = stock_df.groupby('Symbol')['Volume'].rolling(
        window=200).mean().reset_index(level=0, drop=True)

    print(f'Halfway There...')

    # Doji Candlestick Pattern, identified by a small body and long wicks
    stock_df['Doji'] = stock_df.apply(is_doji, axis=1)

    # Bullish and Bearish Engulfing Candlestick Patterns, identified by a large body that engulfs the previous candle
    try:
        stock_df['Bullish_Engulfing'] = stock_df.apply(
            lambda row: is_bullish_engulfing(row, stock_df.shift(1).loc[row.name]), axis=1)
        stock_df['Bearish_Engulfing'] = stock_df.apply(
            lambda row: is_bearish_engulfing(row, stock_df.shift(1).loc[row.name]), axis=1)
    except:
        stock_df['Bullish_Engulfing'] = 0
        stock_df['Bearish_Engulfing'] = 0

    stock_df['EMA_short'] = stock_df['Close'].ewm(span=12, adjust=False).mean()

    # Calculate the long-term EMA
    stock_df['EMA_long'] = stock_df['Close'].ewm(span=26, adjust=False).mean()

    # Calculate the MACD line
    stock_df['MACD'] = stock_df['EMA_short'] - stock_df['EMA_long']

    # Calculate the Signal line
    stock_df['Signal'] = stock_df['MACD'].ewm(span=9, adjust=False).mean()

    # Calculate the MACD histogram
    stock_df['MACD_Hist'] = stock_df['MACD'] - stock_df['Signal']

    # Create new columns for Average True Range (ATR) and True Range (TR)

    stock_df['Previous_Close'] = stock_df['Close'].shift(1)

    # True Range, Shows the volatility of the stock
    stock_df['TR'] = stock_df.apply(
        lambda row: max(
            row['High'] - row['Low'],  # High - Low
            # |High - Previous Close|
            abs(row['High'] - row['Previous_Close']),
            abs(row['Low'] - row['Previous_Close'])  # |Low - Previous Close|
        ), axis=1
    )

    # Average True Range, Shows the average volatility of the stock
    stock_df['ATR'] = stock_df['TR'].rolling(window=10).mean()

    # Create new columns for Relative Strength Index (RSI) and Rate of Change (ROC)
    stock_df['RSI_10_Day'] = calculate_rsi(stock_df)
    stock_df['10_Day_ROC'] = (
        (stock_df['Close'] - stock_df['Close'].shift(10)) / stock_df['Close'].shift(10)) * 100
    stock_df['20_Day_ROC'] = (
        (stock_df['Close'] - stock_df['Close'].shift(20)) / stock_df['Close'].shift(20)) * 100
    stock_df['50_Day_ROC'] = (
        (stock_df['Close'] - stock_df['Close'].shift(50)) / stock_df['Close'].shift(50)) * 100

    # Create new columns for 10,20,50 day resistance and support levels
    stock_df['Resistance_10_Day'] = stock_df['Close'].rolling(window=10).max()
    stock_df['Support_10_Day'] = stock_df['Close'].rolling(window=10).min()
    stock_df['Resistance_20_Day'] = stock_df['Close'].rolling(window=20).max()
    stock_df['Support_20_Day'] = stock_df['Close'].rolling(window=20).min()
    stock_df['Resistance_50_Day'] = stock_df['Close'].rolling(window=50).max()
    stock_df['Support_50_Day'] = stock_df['Close'].rolling(window=50).min()

    # Create new columns for 10,20,50 day Volume Indicators
    stock_df['Volume_MA_10'] = stock_df['Volume'].rolling(window=10).mean()
    stock_df['Volume_MA_20'] = stock_df['Volume'].rolling(window=20).mean()
    stock_df['Volume_MA_50'] = stock_df['Volume'].rolling(window=50).mean()
    # Use a smoothed version of 'Close' to detect peaks and troughs
    stock_df['Smoothed_Close'] = stock_df['Close'].rolling(window=20).mean()

    # Find local minima (buy points) and local maxima (sell points)
    # Local minima (buy points)
    stock_df['Buy_Signal'] = (stock_df['Smoothed_Close'].shift(1) > stock_df['Smoothed_Close']) & (
        stock_df['Smoothed_Close'].shift(-1) > stock_df['Smoothed_Close'])

    # Local maxima (sell points)
    stock_df['Sell_Signal'] = (stock_df['Smoothed_Close'].shift(1) < stock_df['Smoothed_Close']) & (
        stock_df['Smoothed_Close'].shift(-1) < stock_df['Smoothed_Close'])

    # Initialize 'Optimal_Action' column with 'Hold'
    stock_df['Optimal_Action'] = 'Hold'

    # Assign 'Buy' where Buy_Signal is True
    stock_df.loc[stock_df['Buy_Signal'], 'Optimal_Action'] = 'Buy'

    # Assign 'Sell' where Sell_Signal is True
    stock_df.loc[stock_df['Sell_Signal'], 'Optimal_Action'] = 'Sell'

    # Clean up: drop the temporary signals if needed
    stock_df.drop(['Buy_Signal', 'Sell_Signal',
                  'Smoothed_Close'], axis=1, inplace=True)

    stock_df['Action'] = stock_df.apply(determine_action, axis=1)
    stock_df['Z-score'] = (stock_df['Close'] -
                           stock_df['Close'].mean()) / stock_df['Close'].std()

    stock_df.fillna(0, inplace=True)

    stock_df['OBV'] = 0
    for i in range(1, len(stock_df)):
        if stock_df['Close'].iloc[i] > stock_df['Close'].iloc[i - 1]:
            stock_df.loc[stock_df.index[i],
                         'OBV'] = stock_df['OBV'].iloc[i - 1] + stock_df['Volume'].iloc[i]
        elif stock_df['Close'].iloc[i] < stock_df['Close'].iloc[i - 1]:
            stock_df.loc[stock_df.index[i],
                         'OBV'] = stock_df['OBV'].iloc[i - 1] - stock_df['Volume'].iloc[i]
        else:
            stock_df.loc[stock_df.index[i],
                         'OBV'] = stock_df['OBV'].iloc[i - 1]

    return stock_df


def scale_data(stock_df):
    """
    Scales the data using MinMaxScaler for Model Training.
    Columns Added Within this function. 

    Parameters:
    stock_df (DataFrame): DataFrame containing stock data
    """
    features = ['Volume', 'MA_10', 'MA_20', 'MA_50', 'MA_200', 'std_10',
                'std_20', 'std_50', 'std_200', 'upper_band_10', 'lower_band_10',
                'upper_band_20', 'lower_band_20', 'upper_band_50', 'lower_band_50',
                'upper_band_200', 'lower_band_200', 'Golden_Cross_Short', 'Golden_Cross_Medium',
                'Golden_Cross_Long', 'Death_Cross_Short', 'Death_Cross_Medium', 'Death_Cross_Long',
                'ROC', 'AVG_Volume_10', 'AVG_Volume_20', 'AVG_Volume_50', 'AVG_Volume_200', 'Doji',
                'Bullish_Engulfing', 'Bearish_Engulfing', 'MACD', 'Signal', 'MACD_Hist', 'TR', 'ATR',
                'RSI_10_Day', '10_Day_ROC', 'Resistance_10_Day', 'Support_10_Day', 'Resistance_20_Day',
                'Support_20_Day', 'Resistance_50_Day', 'Support_50_Day', 'Volume_MA_10', 'Volume_MA_20',
                'Volume_MA_50', 'OBV', 'Z-score']

    min_max_scaler = MinMaxScaler()

    if features[10] not in stock_df.columns:
        stock_df = add_columns(stock_df)

    stock_df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)

    # Drop rows with NaN values
    stock_df.dropna(subset=features, inplace=True)
    stock_df[features] = min_max_scaler.fit_transform(stock_df[features])
    return stock_df[features]


def scale_and_obtain_data(symbol, test_size=0.2, length=1000):
    """
    Scales the data using MinMaxScaler for Model Training.
    Columns Added Within this function. 

    Parameters:
    symbol (str): Stock symbol to train the model for
    test_size (float): Size of the test set

    Returns:
    X_train (DataFrame): Training data
    y_train (DataFrame): Training labels
    X_test (DataFrame): Testing data
    y_test (DataFrame): Testing labels
    """
    features = ['Volume', 'MA_10', 'MA_20', 'MA_50', 'MA_200', 'std_10',
                'std_20', 'std_50', 'std_200', 'upper_band_10', 'lower_band_10',
                'upper_band_20', 'lower_band_20', 'upper_band_50', 'lower_band_50',
                'upper_band_200', 'lower_band_200', 'Golden_Cross_Short', 'Golden_Cross_Medium',
                'Golden_Cross_Long', 'Death_Cross_Short', 'Death_Cross_Medium', 'Death_Cross_Long',
                'ROC', 'AVG_Volume_10', 'AVG_Volume_20', 'AVG_Volume_50', 'AVG_Volume_200', 'Doji',
                'Bullish_Engulfing', 'Bearish_Engulfing', 'MACD', 'Signal', 'MACD_Hist', 'TR', 'ATR',
                'RSI_10_Day', '10_Day_ROC', 'Resistance_10_Day', 'Support_10_Day', 'Resistance_20_Day',
                'Support_20_Day', 'Resistance_50_Day', 'Support_50_Day', 'Volume_MA_10', 'Volume_MA_20',
                'Volume_MA_50', 'OBV', 'Z-score']

    stock_df = get_stock_data(symbol).tail(length)

    # stock_df = pd.read_csv('data/sp500_stocks.csv')
    # stock_df = stock_df[stock_df['Symbol'] == symbol]

    min_max_scaler = MinMaxScaler()
    stock_df = add_columns(stock_df)
    stock_df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)

    # Drop rows with NaN values
    stock_df.dropna(subset=features, inplace=True)
    stock_df[features] = min_max_scaler.fit_transform(stock_df[features])
    X = stock_df[features]
    y = stock_df['Action']
    print(f'Splitting data...')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def preprocess_data(stock_df, test_size=0.2):
    """
    This function preprocesses the stock data and splits it into training and testing sets.

    Parameters:
    stock_df (DataFrame): DataFrame containing stock data
    test_size (float): Size of the test set

    Returns:
    X_train (DataFrame): Training data
    y_train (DataFrame): Training labels
    X_test (DataFrame): Testing data
    y_test (DataFrame): Testing labels
    """
    features = ['Volume', 'MA_10', 'MA_20', 'MA_50', 'MA_200', 'std_10',
                'std_20', 'std_50', 'std_200', 'upper_band_10', 'lower_band_10',
                'upper_band_20', 'lower_band_20', 'upper_band_50', 'lower_band_50',
                'upper_band_200', 'lower_band_200', 'Golden_Cross_Short', 'Golden_Cross_Medium',
                'Golden_Cross_Long', 'Death_Cross_Short', 'Death_Cross_Medium', 'Death_Cross_Long',
                'ROC', 'AVG_Volume_10', 'AVG_Volume_20', 'AVG_Volume_50', 'AVG_Volume_200', 'Doji',
                'Bullish_Engulfing', 'Bearish_Engulfing', 'MACD', 'Signal', 'MACD_Hist', 'TR', 'ATR',
                'RSI_10_Day', '10_Day_ROC', 'Resistance_10_Day', 'Support_10_Day', 'Resistance_20_Day',
                'Support_20_Day', 'Resistance_50_Day', 'Support_50_Day', 'Volume_MA_10', 'Volume_MA_20',
                'Volume_MA_50', 'OBV', 'Z-score']

    stock_df.dropna(subset=features, inplace=True)
    min_max_scaler = MinMaxScaler()

    stock_df[features] = min_max_scaler.fit_transform(stock_df[features])
    X = stock_df[features]
    y = stock_df['Action']

    print(f'Splitting data...')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def _select_stock():
    """
    Selects a stock from the S&P 500 dataset.
    """
    stock_df = pd.read_csv('data/sp500_stocks.csv')
    company_name = input('Enter the name of the company: ')
    return stock_df[stock_df['Symbol'] == company_name]


def fixFuckUp(symbol):
    """
    Adds second to last day if day was skipped.
    """
    stock_df = pd.read_csv('data/sp500_stocks.csv')
    stock_df = stock_df[stock_df['Symbol'] == symbol]
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period='5d', interval='1d')
    except:
        stock = yf.Ticker(symbol)

        data = stock.history(period='1d', interval='1d')

    if not data.empty:
        latest_data = data.tail(2).iloc[0]
        time = latest_data.name
        open_price = latest_data['Open']
        high = latest_data['High']
        low = latest_data['Low']
        close = latest_data['Close']
        volume = latest_data['Volume']
        new_row = pd.DataFrame({
            'Symbol': [symbol],
            'Date': [datetime.datetime.strftime(time, '%Y-%m-%d')],
            'Open': [open_price],
            'High': [high],
            'Low': [low],
            'Close': [close],
            'Volume': [volume]
        })

        new_row = new_row.reset_index(drop=True)

        stock_df = pd.concat([stock_df, new_row], ignore_index=True).fillna(0)
        if not (stock_df['Date'].tail(2).isin(new_row['Date'].values)).all():
            row = pd.DataFrame({
                'Date': [datetime.datetime.strftime(time, '%Y-%m-%d')],
                'Symbol': [symbol],
                'Adj Close': [close],
                'Close': [close],
                'High': [high],
                'Low': [low],
                'Open': [open_price],
                'Volume': [volume]
            })
            row.to_csv('data/sp500_stocks.csv',
                       index=False, mode='a', header=False)
        return stock_df.sort_values('Date')


def get_stock_data(symbol):
    """
    Gets the stock data for a given symbol and adds new columns to the DataFrame.

    Parameters:
    symbol (str): Stock symbol to get data for
    """
    stock_df = pd.read_csv('data/sp500_stocks.csv')
    stock_df = stock_df[stock_df['Symbol'] == symbol]
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period='5d', interval='1d')
    except:
        stock = yf.Ticker(symbol)

        data = stock.history(period='1d', interval='1d')

    if not data.empty:
        latest_data = data.iloc[-1]
        time = latest_data.name
        open_price = latest_data['Open']
        high = latest_data['High']
        low = latest_data['Low']
        close = latest_data['Close']
        volume = latest_data['Volume']
        new_row = pd.DataFrame({
            'Symbol': [symbol],
            'Date': [datetime.datetime.strftime(time, '%Y-%m-%d')],
            'Open': [open_price],
            'High': [high],
            'Low': [low],
            'Close': [close],
            'Volume': [volume]
        })

        new_row = new_row.reset_index(drop=True)

        stock_df = pd.concat([stock_df, new_row], ignore_index=True).fillna(0)
        # Check if the last two dates in stock_df are the same as the date in new_row
        if not (stock_df['Date'].tail(2).isin(new_row['Date'].values)).all():
            row = pd.DataFrame({
                'Date': [datetime.datetime.strftime(time, '%Y-%m-%d')],
                'Symbol': [symbol],
                'Adj Close': [close],
                'Close': [close],
                'High': [high],
                'Low': [low],
                'Open': [open_price],
                'Volume': [volume]
            })
            row.to_csv('data/sp500_stocks.csv',
                       index=False, mode='a', header=False)
        return stock_df


def tune_hyperparameters(X, y):
    param_grid = {
        'num_leaves': [31, 50],
        'min_data_in_leaf': [20, 50],
        'max_depth': [-1, 10],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200]
    }
    model = LGBMClassifier(random_state=42, verbose=-1)
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0
    )
    grid_search.fit(X, y)
    return grid_search.best_params_


def train_models():
    """
    Trains and tunes models with LGBMClassifier and saves them to the models folder.
    """
    company_df = pd.read_csv('data/sp500_companies.csv')
    # Iterate through each stock and train the model
    for i, stock in enumerate(company_df['Symbol'].unique()):
        if f"{stock}_model.pkl" not in os.listdir('models/LGBMmodels'):
            # Obtain and scale data for the stock
            X_train, X_test, y_train, y_test = scale_and_obtain_data(stock)
            print(f"Training model for {
                  stock} ({i+1}/{len(company_df['Symbol'].unique())})")

            # Hyperparameter tuning (optional)
            logging.info(f"Tuning hyperparameters for {stock}...")
            best_params = tune_hyperparameters(X_train, y_train)
            logging.info(f"Best parameters for {stock}: {best_params}")

            # Define and fit the model
            model = LGBMClassifier(random_state=42, **best_params)
            model.fit(X_train, y_train)

            # Cross-validation for better evaluation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=skf, scoring='accuracy')
            logging.info(
                f"Cross-validation accuracy for {stock}: {cv_scores.mean():.4f}")

            # Save the model
            joblib.dump(model, f'models/LGBMmodels/{stock}_model.pkl')
            logging.info(f"Model for {stock} saved successfully")


def train_model_incrementally():
    """
    Trains a model incrementally on all stocks in the S&P 500 dataset.
    """
    features = ['Volume', 'MA_10', 'MA_20', 'MA_50', 'MA_200', 'std_10',
                'std_20', 'std_50', 'std_200', 'upper_band_10', 'lower_band_10',
                'upper_band_20', 'lower_band_20', 'upper_band_50', 'lower_band_50',
                'upper_band_200', 'lower_band_200', 'Golden_Cross_Short', 'Golden_Cross_Medium',
                'Golden_Cross_Long', 'Death_Cross_Short', 'Death_Cross_Medium', 'Death_Cross_Long',
                'ROC', 'AVG_Volume_10', 'AVG_Volume_20', 'AVG_Volume_50', 'AVG_Volume_200', 'Doji',
                'Bullish_Engulfing', 'Bearish_Engulfing', 'MACD', 'Signal', 'MACD_Hist', 'TR', 'ATR',
                'RSI_10_Day', '10_Day_ROC', 'Resistance_10_Day', 'Support_10_Day', 'Resistance_20_Day',
                'Support_20_Day', 'Resistance_50_Day', 'Support_50_Day', 'Volume_MA_10', 'Volume_MA_20',
                'Volume_MA_50', 'OBV', 'Z-score']

    stock_df = pd.read_csv('data/sp500_stocks.csv')
    print("Data Loaded")

    # Initialize an empty DMatrix for incremental training
    initial_model = None
    num_round = 100  # Number of boosting rounds

    for symbol in stock_df['Symbol'].unique():
        print(f"Processing {symbol}...")

        # Filter the data for the current stock
        stock_data = stock_df[stock_df['Symbol'] == symbol].copy()

        print(f"Adding columns for {symbol}...")
        stock_data = add_columns(stock_data)

        print(f"Preprocessing data for {symbol}...")
        preprocessed = scale_data(stock_data)

        print(f"Splitting data for {symbol}...")
        X = preprocessed[features]
        y = stock_data['Action']

        # Split into training and testing data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # If it's the first iteration, initialize the model
        if initial_model is None:
            print(f"Initializing model for {symbol}...")
            # Number of unique actions (e.g., Buy, Sell, Hold)
            unique_classes = stock_data['Action'].nunique()

            # Initial model training
            initial_model = xgb.train(params={
                'objective': 'multi:softmax',  # Use 'multi:softprob' for probability outputs
                'eval_metric': 'mlogloss',
                'num_class': unique_classes},  # Specify the number of classes
                dtrain=dtrain,
                num_boost_round=num_round,
                evals=[(dtest, 'eval')],
                early_stopping_rounds=10)
        else:
            print(f"Continuing training on {symbol} data...")

            # Continue training with the same parameters
            initial_model = xgb.train(params={
                'objective': 'multi:softmax',  # Same objective as in the initial training
                'eval_metric': 'mlogloss',
                'num_class': unique_classes},  # Specify the number of classes
                dtrain=dtrain,
                num_boost_round=num_round,
                evals=[(dtest, 'eval')],
                xgb_model=initial_model,  # Continue training from the previous model
                early_stopping_rounds=10)

    # Save the final model
    print("Saving the final model...")
    initial_model.save_model(
        'models/XGBmodels/all_stocks_incremental_model.model')
    print("Model trained on all stocks and saved.")


def simulate_days(days, cash=10000, existing_shares=0, to_file=False, massTrade=False):
    """
    Simulates a number of days of trading for all stocks using the specific model.
    Models used: {symbol}_model.pkl LGBMClassifier

    Parameters:
    days (int): Number of days to simulate
    cash (int): Initial cash amount
    existing_shares (int): Number of shares already held
    to_file (bool): Save the results to a file
    massTrade (bool): Simulate mass trading
    """
    # Initialize empty dataframes for storing new decisions
    all_decisions_s = pd.DataFrame(columns=[
        'Stock Name', 'Day', 'Action', 'Stock Price', 'Cash', 'Shares Held', 'Portfolio Value'])
    stock_data = pd.read_csv('data/sp500_stocks.csv')
    sp500_stocks = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'XOM', 'META',
                'INTC', 'T', 'DIS', 'MMM', 'VZ', 'CCL',
                'KO', 'JNJ', 'PG', 'WMT', 'MCD', 'PFE']
    # Loop through each stock symbol
    for symbol in sp500_stocks:
        try:
            # Get the most recent stock_data
            updated_stock_df = get_stock_data(symbol)
            updated_stock_df = fixFuckUp(symbol)
            updated_stock_df = updated_stock_df.tail(days)

            # updated_stock_df = stock_data[stock_data['Symbol'] == symbol].tail(1)

            # Load the specific model for the stock, or fallback to the general model if it doesn't exist
            try:
                specific_model = joblib.load(
                    f'models/LGBMmodels/{symbol}_model.pkl')
                print(f"Using model for {symbol}")
            except Exception:
                general_model = xgb.Booster()
                general_model.load_model(
                    'models/XGBmodels/all_stocks_incremental_model.pkl')
                specific_model = general_model
                print(f"Using general model for {symbol}")

            # Simulate a day of trading for the stock with the specific model
            new_decisions_s, _ = stock_market_simulation(
                model=specific_model,
                initial_cash=cash,
                days=days,  # Simulate only one day
                stock=updated_stock_df,
                oneDay=False,
                existing_shares=existing_shares,
                masstrades=massTrade
            )
            # Append the new decisions to the all_decisions dataframes
            all_decisions_s = pd.concat(
                [all_decisions_s, new_decisions_s], ignore_index=True)
            # if existing_shares == 1:
            #     continue
        except Exception as e:
            print(f"Error: {e}")
            print("====================================")
            print(f"Error for {symbol}. Skipping...")
            print("====================================")
            continue

    if to_file:
        all_decisions_s.to_csv('simResults/LGBM_model_decisions.csv',
                               index=False)
    return all_decisions_s


def simulate_day_general(day):
    """
    Simulates a day of trading for all stocks using the general model.
    Models used: all_stocks_incremental_model.pkl XGBClassifier
    """
    # Load the general model
    general_model = xgb.Booster()
    general_model.load_model(
        'models/XGBmodels/all_stocks_incremental_model.pkl')

    # Load company data
    company_df = pd.read_csv('data/sp500_companies.csv')

    general_decision_df = pd.read_csv('simResults/general_model_decisions.csv')

    all_decisions_g = pd.DataFrame(columns=[
        'Stock Name', 'Day', 'Action', 'Stock Price', 'Cash', 'Shares Held', 'Portfolio Value'])

    for symbol in company_df['Symbol'].unique():
        try:
            # Get the most recent cash and shares held for the general model
            if symbol in general_decision_df['Stock Name'].unique():
                last_row_g = general_decision_df[general_decision_df['Stock Name']
                                                 == symbol].iloc[-1]
            else:
                last_row_g = {'Cash': 10000, 'Shares Held': 0,
                              'Day': 0}  # Initialize if no previous data

            cash_g = last_row_g['Cash']
            existing_shares = last_row_g['Shares Held']

            # Get the stock data for the symbol
            # messupday = fixFuckUp(symbol).tail(1)
            updated_stock_df = get_stock_data(symbol)
            # updated_stock_df = pd.concat([messupday, updated_stock_df]).sort_values(by='Date')
            updated_stock_df = updated_stock_df.tail(1)

            print(f"Using general model for {symbol}")

            new_decisions_g, _ = stock_market_simulation(
                model=general_model,
                initial_cash=cash_g,
                days=1,  # Simulate only one day
                stock=updated_stock_df,
                oneDay=False,
                existing_shares=existing_shares,
            )

            all_decisions_g = pd.concat(
                [all_decisions_g, new_decisions_g], ignore_index=True)
        except Exception as e:
            print(f"Error: {e}")
            print("====================================")
            print(f"Error for {symbol}. Skipping...")
            print("====================================")
            continue
    all_decisions_g.to_csv('simResults/general_model_decisions.csv',
                           mode='a', header=False, index=False)


def simulate_day_specific(day, model_type='LGBM'):
    """
    Simulates a day of trading for all stocks using the specific model.
    Models used: {symbol}_model.pkl XGBClassifier
    """
    # Load company data
    company_df = pd.read_csv('data/sp500_companies.csv')

    # Load the previous decision dataframes
    specific_decision_df = pd.read_csv(
        'simResults/specific_model_decisions.csv')

    # Initialize empty dataframes for storing new decisions
    all_decisions_s = pd.DataFrame(columns=[
        'Stock Name', 'Day', 'Action', 'Stock Price', 'Cash', 'Shares Held', 'Portfolio Value'])

    # Loop through each stock symbol
    for symbol in company_df['Symbol'].unique():
        try:
            # Get the most recent cash and shares held for the specific model
            if symbol in specific_decision_df['Stock Name'].unique():
                last_row_s = specific_decision_df[specific_decision_df['Stock Name']
                                                  == symbol].iloc[0]
            else:
                last_row_s = {'Cash': 10000, 'Shares Held': 0,
                              'Day': 0}  # Initialize if no previous data

            # Set the starting cash and shares for the current simulation
            cash_s = last_row_s['Cash']
            existing_shares = last_row_s['Shares Held']

            updated_stock_df = get_stock_data(symbol)
            updated_stock_df = updated_stock_df.tail(1)

            # updated_stock_df = stock_data[stock_data['Symbol'] == symbol].tail(1)

            # Load the specific model for the stock, or fallback to the general model if it doesn't exist
            try:
                specific_model = joblib.load(
                    f'models/{model_type}models/{symbol}_model.pkl')
                print(f"Using model for {symbol}")
            except Exception:
                general_model = xgb.Booster()
                general_model.load_model(
                    'models/all_stocks_incremental_model.pkl')
                specific_model = general_model
                print(f"Using general model for {symbol}")

            # Simulate a day of trading for the stock with the specific model
            new_decisions_s, _ = stock_market_simulation(
                model=specific_model,
                initial_cash=cash_s,
                days=1,  # Simulate only one day
                stock=updated_stock_df,
                oneDay=day,
                existing_shares=existing_shares
            )
            # Append the new decisions to the all_decisions dataframes
            all_decisions_s = pd.concat(
                [all_decisions_s, new_decisions_s], ignore_index=True)
            # if existing_shares == 1:
            #     continue
        except Exception as e:
            print(f"Error: {e}")
            print("====================================")
            print(f"Error for {symbol}. Skipping...")
            print("====================================")
            continue

    if model_type == 'XGB':
        # Save the new decisions
        all_decisions_s.to_csv('simResults/specific_model_decisions.csv',
                               mode='a', header=False, index=False)
    else:
        all_decisions_s.to_csv(f'simResults/{model_type}_model_decisions.csv',
                               mode='a', header=False, index=False)


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    warnings.filterwarnings('ignore')
    # Simulate one day for all stocks, continuing from previous cash balances
    # simulate_day_specific(7, 'XGB')
    # simulate_day_general(7)
    # train_models()
    simulate_days(252, to_file=True, massTrade=True)
