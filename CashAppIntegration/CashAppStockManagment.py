import joblib
import pandas as pd
from SimulateDay import stock_market_simulation, train_model, get_stock_data
import os
import sys

# Add the main directory to the Python path
sys.path.append('/Users/eduardobenjamin/Desktop/Repos/StockTradingModel')


def simulate_day_for_cash_app():
    portfolio = pd.read_csv("CashAppIntegration/portfolio.csv")
    daysimulation = pd.DataFrame(columns=[
                                 'Stock Name', 'Day', 'Action', 'Stock Price', 'Cash', 'Shares Held', 'Portfolio Value', 'Date'])
    cash = portfolio['Cash'].iloc[-1]
    for stock in portfolio['Stock Name'].unique():
        try:
            model = joblib.load(f"models/LGBMmodels/{stock}_model.pkl")
        except FileNotFoundError:
            print(f"Training model for {stock}...")
            model = train_model(stock)
            joblib.dump(model, f"models/LGBMmodels/{stock}_model.pkl")

        updated_stock_df = get_stock_data(stock)
        updated_stock_df = updated_stock_df.tail(1)

        day = (portfolio[portfolio['Stock Name'] == stock]['Day'].iloc[-1]) + 1

        new_row, _ = stock_market_simulation(
            model,
            initial_cash=cash,
            days=1,
            stock=updated_stock_df,
            existing_shares=portfolio[portfolio['Stock Name']
                                      == stock]['Shares Held'].iloc[-1],
            oneDay=day,
            descision=portfolio[portfolio['Stock Name'] == stock]
        )
        cash = new_row['Cash'].iloc[-1]
        daysimulation = pd.concat([daysimulation, new_row])

    # Save the updated portfolio back to the CSV file
    daysimulation['Cash'] = daysimulation['Cash'].iloc[-1]
    daysimulation.fillna(0, inplace=True)
    daysimulation.to_csv("CashAppIntegration/portfolio.csv",
                         index=False, mode='a', header=False)
    return portfolio


if __name__ == '__main__':
    simulate_day_for_cash_app()
