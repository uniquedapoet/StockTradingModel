import random
import pandas as pd


class StockModel:
    def __init__(self, model):
        self.model = model

    def predict(self, stock_price):
        return random.choice(['buy', 'sell', 'hold'])


def stock_market_simulation(model, initial_cash, days, stock):
    cash = initial_cash
    shares_held = 0
    portfolio_value = []
    
    for i in range(days):
        stock_price = stock['Close'].iloc[i]
        strategy = model.predict(stock_price)
        
        if strategy == 'buy' and cash >= stock_price:
            # Buy one share if cash is sufficient
            cash -= stock_price
            shares_held += 1
            print(f"Day {i}: Bought 1 share at {stock_price}, Cash left: {cash}")
            
        elif strategy == 'sell' and shares_held > 0:
            # Sell one share if we hold any
            cash += stock_price
            shares_held -= 1
            print(f"Day {i}: Sold 1 share at {stock_price}, Cash: {cash}")
            
        # Calculate the total portfolio value (cash + stock holdings)
        portfolio_value_at_time = cash + (shares_held * stock_price)
        portfolio_value.append(portfolio_value_at_time)
    
    # Final results
    final_portfolio_value = cash + (shares_held * stock['Close'].iloc[-1])
    print(f"Final Portfolio Value: {final_portfolio_value}")
    print(f"Cash: {cash}, Shares held: {shares_held}")

    return portfolio_value, final_portfolio_value

# Example usage with random data
stock_data = pd.DataFrame({
    'Close': [100, 105, 102, 98, 97, 101, 103, 107, 110, 108]
})

# Initialize a random stock model
model = StockModel(None)

# Simulate for 10 days with an initial cash balance of $1000
portfolio_value, final_value = stock_market_simulation(model, initial_cash=1000, days=10, stock=stock_data)
