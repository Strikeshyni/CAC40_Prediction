
# CAC40 Stock Prediction

This project aims to predict the next-day closing prices of stocks in the CAC40 index using machine learning and deep learning techniques. The workflow includes data collection, preprocessing, model training, hyperparameter tuning, simulation of trading strategies, and performance evaluation.

## Table of Contents
- [CAC40 Stock Prediction](#cac40-stock-prediction)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Project Structure](#project-structure)
  - [Installation \& Setup](#installation--setup)
  - [Data Collection](#data-collection)
  - [Model Training \& Prediction](#model-training--prediction)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Trading Simulation](#trading-simulation)
  - [Results \& Visualization](#results--visualization)
  - [Customization](#customization)
  - [License](#license)

---

## Project Overview
This repository provides a full pipeline to:
- Download and preprocess historical CAC40 stock data
- Train and tune deep learning models (LSTM) to predict next-day closing prices
- Simulate trading strategies (buy/sell/hold) based on model predictions
- Evaluate and visualize the results

## Project Structure

```
CAC40_stock_prediction/
│   README.md
│   requirements.txt
│   __init__.py
│
├── dataset/                  # Downloaded and processed datasets
├── v9_no_test_data/         # Scripts and outputs for main experiments
├── web_scrapper/            # Data collection scripts
│   └── scrapper.py
├── model_v1.py              # Main model class (LSTM, training, prediction)
├── buying_simulation.py     # Trading simulation logic
└── ...                      # Other scripts and experiment folders
```

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repo_url>
   cd CAC40_stock_prediction
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Data Collection

- Data is downloaded using the `yfinance` library and stored in the `dataset/` folder.
- The script `web_scrapper/scrapper.py` fetches historical closing prices for CAC40 stocks.
- Data is saved as CSV files with two columns: `Date` (index) and `Close`.

## Model Training & Prediction

- The main model is an LSTM neural network implemented in `model_v1.py`.
- Data is preprocessed (scaling, sequence creation) before training.
- The model predicts the next day's closing price based on a window of previous days (`time_step`).
- Training, validation, and test splits are handled automatically.
- Models and results are saved in experiment-specific folders.

## Hyperparameter Tuning

- Hyperparameter search is performed using `keras-tuner` (Hyperband algorithm).
- Tunable parameters include LSTM units, dense units, activation functions, etc.
- The best model is selected based on validation loss.

## Trading Simulation

- The script `buying_simulation.py` simulates trading strategies using model predictions.
- The `placement` function decides whether to buy, sell, or hold based on predicted and actual prices, current balance, and stocks owned.
- The simulation tracks portfolio value, number of stocks, and actions over time.

## Results & Visualization

- Predictions and actual prices are exported to CSV for analysis.
- Visualization scripts plot:
  - Actual vs. predicted prices
  - Trading actions and portfolio evolution
  - Input sequences and model predictions for interpretability

## Customization

- **Change stock:** Edit the `stock_name` variable in your scripts.
- **Change time window:** Adjust the `time_step` parameter.
- **Change simulation period:** Modify `from_date` and `to_date`.
- **Add new strategies:** Extend `buying_simulation.py` or add new scripts.

## License

This project is for educational and research purposes. Please check the license file for details.
