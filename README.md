# Stock Trading Model Using Machine Learning and Technical Indicators

## Project Overview

This project showcases a **machine learning-driven stock trading model** tailored for S&P 500 stocks, with an emphasis on profitability through careful data analysis, feature engineering, and conservative trading strategies. Utilizing historical stock data and a robust selection of technical indicators, the model aims to identify profitable opportunities with minimized risk. This repository serves as a demonstration of the methodology, analysis, and results, rather than as a standalone Python module.

Key highlights include:
- **Historical Data and Simulation**: Data spanning over a decade from Yahoo Finance.
- **Trading Strategy**: Conservative trading approach limited to one share per trade day.
- **Performance**: Backtested to achieve a simulated average return of **23% profit** over one year.

---

## Features and Methodology

### Data Collection
Historical stock price data was sourced from **Yahoo Finance** using the `yfinance` library, covering key indicators over a multi-year period to ensure robustness. This extensive data enables the model to account for various market conditions.

### Technical Indicators and Feature Engineering
Key technical indicators were engineered to provide insight into market trends, momentum, and volatility, including:
- **Moving Averages (MA)**: MA_10, MA_50, MA_200 to smooth out price data and identify trends.
- **Relative Strength Index (RSI)**: Measures momentum, identifying overbought or oversold conditions.
- **Bollinger Bands**: Defines volatility boundaries, aiding in the identification of extreme price movements.
- **MACD (Moving Average Convergence Divergence)**: Used for trend and momentum detection.
- **Crossover Signals**: Golden Cross and Death Cross signals for bullish and bearish trends.

The `add_columns` function generates 49 features, creating a comprehensive set of inputs for the machine learning model. Additional indicators include candlestick patterns, rate of change (ROC), and various volatility metrics, which collectively enhance the model’s predictive capabilities.

### Model Training
The core model leverages **LightGBM** and **XGBoost** for predictive classification:
- **Model Optimization**: Hyperparameters are fine-tuned via `GridSearchCV` to maximize accuracy.
- **Cross-Validation**: Ensures model reliability across different timeframes and stock symbols.

The model predicts buy, sell, or hold actions, and each stock in the S&P 500 was trained individually to optimize for distinct trading patterns. Using LightGBM enables efficient, high-performance training even with a large feature set.

### Backtesting and Simulation
The `stock_market_simulation` function simulates real-world trading conditions by acting as a "virtual market," querying the model for daily decisions, and updating the portfolio based on predicted buy or sell signals. Key backtesting results include:
- **Annual Average Return**: Achieved a **23% profit** over a 365-day test period.
- **Portfolio Performance**: Maintained a steady portfolio value growth with a low-risk trading approach.

Simulation includes performance tracking for each S&P 500 stock, allowing analysis across diverse market conditions, from high-growth stocks like Meta to stable performers like Coca-Cola.

### Key Results
The simulation demonstrates the model’s ability to handle both volatile and stable stocks. Stocks like **Meta (META)** and **Vistra Corp (VST)** showed high returns, while conservative choices like **Johnson & Johnson (JNJ)** contributed stability, reflecting the model’s adaptability.

---

## Project Structure

- **`SimulateDay.py`**: Core functions for data preprocessing, feature engineering, model training, and backtesting.
- **`CashAppIntegration/`**: Scripts for portfolio updates and day-to-day simulation.
- **`models/`**: Trained models for each stock symbol.
- **`data/`**: Historical stock data and preprocessed feature data.

### Usage Instructions

This project offers a structured pathway from data collection to analysis. The following steps guide you through setup and usage:

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Simulation**:
   To view data collection, model training, and backtesting results, execute:
   ```bash
   python SimulateDay.py
   ```

---

## Dependencies

The project relies on Python libraries optimized for data science and machine learning:
- **pandas** and **numpy**: Data manipulation and numerical computing.
- **yfinance**: Accessing historical stock data.
- **scikit-learn**: Machine learning utilities, including `GridSearchCV`.
- **lightgbm** and **xgboost**: Gradient boosting libraries for classification tasks.

For a complete list, refer to `requirements.txt`.

---

## Visualization

Key results are visualized to provide insights into model performance:
- **Portfolio Value vs. Stock Price**: Visualizes how portfolio value fluctuates with stock movements.
- **Profit & Loss Analysis**: Highlights gains and losses across stocks in the test period.

---

## Project Insights and Conclusion

This stock trading model underscores the potential of machine learning in the stock market, demonstrating consistent returns across various S&P 500 stocks. While the model yields positive results with a **23% average annual return**, future improvements could include:
- **Transaction Cost Integration**: To reflect real-world trading expenses.
- **Enhanced Risk Management**: Implementing stop-loss or trailing stop strategies.
- **Sequential Data Models**: LSTM or GRU for capturing time dependencies in stock prices.

This repository provides a solid foundation for further research and development in algorithmic trading using machine learning.

---