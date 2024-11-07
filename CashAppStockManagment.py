import joblib
import pandas as pd
from SimulateDay import stock_market_simulation, train_model, get_stock_data


def simulate_day_for_cash_app():
    """
    !!! Currently has an issue with making decisions when given only one day.
    Current fix ==  feeding it 5 days and only selecting most recent day to record!!!
    """
    portfolio = pd.read_csv("CashAppIntegration/portfolio.csv")
    daysimulation = pd.DataFrame(columns=['Stock Name', 'Day', 'Action', 'Stock Price',
                                 'Cash', 'Shares Held', 'Portfolio Value', 'Date', 'Actual Sell'])
    cash = portfolio['Cash'].iloc[-1]
    for stock in portfolio['Stock Name'].unique():
        updated_stock_df = get_stock_data(stock).drop_duplicates()
        updated_stock_df = updated_stock_df.reset_index(drop=True)
        updated_stock_df = updated_stock_df.tail(5)

        day = (portfolio[portfolio['Stock Name'] == stock]['Day'].iloc[-1]) + 1

        # Load the specific model for the stock, or fallback to the general model if it doesn't exist
        try:
            specific_model = joblib.load(
                f'models/LGBMmodels/{stock}_model.pkl')
            print(f"Using model for {stock}")
        except Exception:
            print(f"Model for {stock} not found. Training a new model...")
            specific_model = train_model(stock)
            print(f"Model trained for {stock}")
        # Simulate a day of trading for the stock with the specific model
        new_row, _ = stock_market_simulation(
            model=specific_model,
            initial_cash=cash,
            days=5,
            stock=updated_stock_df,
            oneDay=day,
            existing_shares=portfolio[portfolio['Stock Name']
                                      == stock]['Shares Held'].iloc[-1],
            masstrades=True,
            descision=portfolio[portfolio['Stock Name'] == stock],
            brokeBitch=True,
            brokeBitchLimiter=0.2,
        )
        cash += new_row['Cash'].iloc[-1]
        daysimulation = pd.concat([daysimulation, new_row])

    # Save the updated portfolio back to the CSV file
    daysimulation['Cash'] = daysimulation['Cash'].iloc[-1]
    daysimulation.fillna(0, inplace=True)
    daysimulation.to_csv("CashAppIntegration/portfolio.csv",
                         index=False, mode='a', header=False)
    return portfolio


if __name__ == '__main__':
    simulate_day_for_cash_app()
