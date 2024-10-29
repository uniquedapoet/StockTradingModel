import os
import sys

# Add the main directory to the Python path
sys.path.append('/Users/eduardobenjamin/Desktop/Repos/StockTradingModel')

from SimulateDay import stock_market_simulation, train_model, get_stock_data
import pandas as pd
import joblib


def simulate_day_for_cash_app(day):
    portfolio = pd.read_csv("CashAppIntegration/portfolio.csv")
    for stock in portfolio['Stock Name'].unique():
        try:
            model = joblib.load(f"models/LGBMmodels/{stock}_model.pkl")
        except FileNotFoundError:
            print(f"Training model for {stock}...")
            model = train_model(stock)
        
        stock_data = get_stock_data(stock).tail(1)
        new_row, _ = stock_market_simulation(
            model,
            initial_cash=portfolio[portfolio['Stock Name'] == stock]['Cash'].iloc[-1],
            days=1,
            stock=stock_data,
            existing_shares=portfolio[portfolio['Stock Name'] == stock]['Shares Held'].iloc[-1],
            oneDay=day,
            descision=portfolio[portfolio['Stock Name'] == stock].tail(1)
        )
        portfolio = pd.concat([portfolio, new_row])
    
    # Save the updated portfolio back to the CSV file
    portfolio.to_csv("CashAppIntegration/portfolio.csv", index=False)
    return portfolio


if __name__ == '__main__':
    simulate_day_for_cash_app(1)