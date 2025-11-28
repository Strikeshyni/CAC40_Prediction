# CAC40 Stock Prediction & Trading Simulation

This project is a comprehensive system for predicting stock prices of CAC40 companies and simulating trading strategies. It features a robust FastAPI backend, multiple state-of-the-art machine learning models, and a real-time simulation engine.

## 🚀 Key Features

*   **Multi-Model Architecture**:
    *   **Bi-LSTM (v2)**: Bidirectional LSTM with Dropout for capturing temporal dependencies (Default).
    *   **Transformer**: Transformer Encoder with Multi-Head Attention.
    *   **XGBoost**: Gradient boosting regressor with randomized search tuning.
    *   **LSTM (v1)**: Legacy implementation.
*   **FastAPI Backend**:
    *   Asynchronous training and simulation jobs.
    *   WebSocket support for real-time progress tracking.
    *   RESTful endpoints for model management and predictions.
*   **Advanced Simulation**:
    *   **Historical Simulation**: Test strategies on past data with "time-travel" model training.
    *   **Multiple Strategies**: Simple, Threshold, Percentage, Conservative, and Aggressive.
    *   **Visualizations**: Detailed plots of price action, buy/sell signals, and portfolio evolution.
*   **Benchmarking System**:
    *   Compare all models side-by-side on recursive forecasting tasks.

## 📂 Project Structure

```
CAC40_stock_prediction/
├── api/                     # FastAPI application
│   ├── main.py             # Server entry point
│   ├── services.py         # Business logic (Training, Simulation)
│   ├── models.py           # Pydantic data models
│   └── ...
├── models/                  # Machine Learning Model Definitions
│   ├── model_lstm_v2.py    # Bi-LSTM (Current Standard)
│   ├── model_transformer.py # Transformer Architecture
│   ├── model_xgboost.py    # XGBoost Implementation
│   └── ...
├── real_time_simulation/    # Simulation Logic & Visualization
│   ├── buy_simulation_v2.py
│   └── visual_utils.py
├── benchmark/               # Model Comparison Tools
│   └── benchmark_system.py
├── dataset/                 # Cached Stock Data (CSV)
└── web_scrapper/           # Data Fetching Utilities
```

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone <repo_url>
    cd CAC40_stock_prediction
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install -r api/requirements_api.txt
    ```

## 🚦 Usage

### 1. Running the API Server
The core of the project is the API. Start it with:

```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8002
```

*   **Swagger UI**: `http://localhost:8002/docs`
*   **API Documentation**: See `api/README_API.md` for detailed endpoint usage.

### 2. Running the Example Client
Test the full pipeline (Training -> Prediction -> Simulation) using the example script:

```bash
python api/api_example_client.py
```

### 3. Benchmarking Models
To compare the performance of Bi-LSTM, Transformer, and XGBoost:

```bash
python3 benchmark/benchmark_system.py
```
This will generate a performance plot in the `benchmark/` directory.

## 📊 Trading Strategies

The simulation engine supports 5 distinct strategies:
*   **Simple**: Buy if predicted > actual.
*   **Threshold**: Buy if predicted > actual * (1 + threshold).
*   **Percentage**: Based on % change.
*   **Conservative**: High confidence requirements.
*   **Aggressive**: Frequent trading with tight stop-losses.

See `api/STRATEGIES_GUIDE.md` for details.

## 🔧 Configuration

*   **Training**: Configurable epochs, batch size, and hyperparameter tuning (enabled/disabled).
*   **Simulation**: Adjustable initial balance, risk ratios, and stop-loss/take-profit levels.

## 📝 License

This project is for educational and research purposes.
