from SimulateDay import stock_market_simulation, train_model, get_stock_data
import pandas as pd
import joblib


def simulate_day_for_cash_app(day):
    portfolio = pd.read_csv("CashAppIntegration/portfolio.csv")
    for stock in portfolio['Stock'].unique():
        try:
            model = joblib.load(f"models/{stock}.pkl")
        except:
            print(f"Training model for {stock}...")
            model = train_model(stock)
        stock_data = get_stock_data(stock).tail(1)
        stock_market_simulation(model,
                                initial_cash=portfolio[portfolio['Stock']
                                                       == stock]['Cash'].iloc[-1],
                                days=1,
                                stock=stock_data,
                                existing_shares=portfolio[portfolio['Stock']
                                                          == stock]['Shares'].iloc[-1],
                                oneDay=day,
                                )
