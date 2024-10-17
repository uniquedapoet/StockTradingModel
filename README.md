# StockTradingModel

This repository contains a stock market forecasting model designed to predict buy, sell, or hold actions using technical indicators and classification models. The model aims to maximize returns by analyzing historical and real-time stock data.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Data](#data)
6. [Model Architecture](#model-architecture)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

## Overview
The project builds a machine learning model that uses various technical indicators and stock market patterns to forecast the best trading actions for individual stocks. It currently delivers an average annual return of 20-30%. The model uses Python and libraries such as Pandas, Plotly, and Scikit-learn for data processing, visualization, and modeling.

## Features
- **Technical Indicators**: Includes moving averages (MA10, MA20, MA50, MA200), Relative Strength Index (RSI), MACD, and ATR among others.
- **Regression Models**: Implements regression-based models to determine stock actions rather than classification models.
- **Backtesting**: Historical data backtesting to validate and optimize model performance.
- **Interactive Visualization**: Utilizes Plotly for dynamic visualizations of stock trends and technical indicators.
- **Comprehensive Evaluation**: Evaluates the model’s performance with annualized return and accuracy metrics.

## Installation
Ensure you have Python 3.x installed. Clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/stock-market-model.git
cd stock-market-model
pip install -r requirements.txt
```

## Usage
To run the model, follow these steps:

1. Download stock data using your preferred data source (e.g., Finnhub API or Alpha Vantage).
2. Save the data as a CSV file or directly connect the data source.
3. Run the model using the command:

```bash
python main.py --data_path path/to/your/data.csv
```

## Data
The model works with stock data that includes the following columns:

- `Date`: Date of the trading day
- `Symbol`: Stock symbol
- `Open`, `Close`, `High`, `Low`, `Volume`: Standard stock price metrics
- `MA_10`, `MA_20`, `MA_50`, `MA_200`: Moving averages
- `RSI_10_Day`: Relative Strength Index
- `MACD`, `Signal`, `MACD_Hist`: MACD indicators
- `ATR`: Average True Range
- And other columns for technical analysis

The dataset should be structured with at least these columns. Adjust the indicators as needed based on your dataset.

## Model Architecture
The model uses various regression techniques to predict stock actions:

- **Linear Regression**: For simple trends and pattern recognition.
- **Decision Trees & Ensemble Methods**: For more complex and non-linear relationships.
- **Neural Network**: To capture deeper patterns and dependencies in the data.

The model is trained on historical data and tested using a validation set to fine-tune hyperparameters and minimize errors.

## Results
The model currently achieves a return of 20-30% annually. An alternative model in development has shown an 8% average profit. Ongoing optimizations and testing aim to refine these results further. The model's detailed performance metrics and results can be found in the `results` folder.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests with improvements. Please ensure any new code is properly tested and documented.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Let me know if you'd like any adjustments!