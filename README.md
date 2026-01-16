# Stock Prediction

A machine learning project for predicting stock price movements using technical indicators and historical data.

## Overview

Stock Prediction is a Python package that trains an XGBoost classifier to predict whether a stock's price will go up or down on the next trading day. The model uses technical indicators derived from historical price data and various machine learning features.

## Features

- **Data Collection**: Fetches historical stock/commodity data using yfinance
- **Feature Engineering**: Generates technical indicators including:
  - Returns (1-day, 5-day, 15-day)
  - Simple Moving Averages (SMA)
  - Volatility measures
  - Relative Strength Index (RSI)
  - Lagged close prices
- **Model Training**: XGBoost classifier for binary classification (up/down movement)
- **Prediction**: Make next-day return predictions for any configured ticker
- **Model Versioning**: Automatic timestamped model and argument saving

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Kamyk0417/stock_pred.git
cd stock_pred
```

2. Install the package and dependencies:
```bash
pip install -e .
pip install -r requirements.txt
```

## Usage

### Training a Model

Configure your training parameters in `stock_pred/trainer/train.py`:

```python
args = {
    "ticker_symbol": "Gold",  # Stock ticker or commodity symbol
    "period": "1y",           # Historical period (e.g., "1y", "6mo", "1mo")
    "interval": "1d"          # Data interval (e.g., "1d", "1h", "5m")
}
```

Then run:
```bash
python stock_pred/trainer/train.py
```

This will:
- Fetch historical data for the specified ticker
- Prepare and scale the data
- Train the XGBoost classifier
- Save the model and training arguments with timestamps

### Making Predictions

Use the prediction API to forecast the next trading day's return:

```python
from stock_pred.app.predict import predict_next_return

prediction = predict_next_return()
# Returns: 0 (price expected to go up) or 1 (price expected to go down)
```

## Project Structure

```
stock_pred/
├── app/                          # Prediction application
│   └── predict.py               # Prediction interface
├── trainer/                      # Model training modules
│   ├── train.py                 # Training script
│   ├── prep_data.py             # Data preparation and feature engineering
│   ├── utils.py                 # Utility functions for model saving/loading
│   ├── worker.py                # Background worker (optional)
│   └── data_eda.ipynb           # Exploratory Data Analysis notebook
├── models/                       # Trained models and arguments (versioned)
│   ├── YYYYMMDD_model.pkl       # Trained model files
│   └── YYYYMMDD_args.json       # Training configuration
└── __init__.py
```

## Requirements

- Python 3.7+
- See [requirements.txt](requirements.txt) for all dependencies

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This project is for educational and research purposes only. Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always conduct thorough research and consult with financial advisors before making investment decisions.
