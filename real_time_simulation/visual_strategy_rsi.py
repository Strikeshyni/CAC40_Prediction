import sys
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from real_time_simulation.visual_utils import plot_simulation
from web_scrapper.scrapper import get_closing_prices
import matplotlib.pyplot as plt

def load_latest_model():
    """Finds and loads the most recent model from api_models/"""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../api_models'))
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Directory {base_dir} not found.")
    model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not model_dirs:
        raise FileNotFoundError("No model directories found in api_models/")
    latest_dir = sorted(model_dirs)[-1]
    model_path = os.path.join(base_dir, latest_dir)
    pkl_files = [f for f in os.listdir(model_path) if f.endswith('.pkl')]
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl file found in {model_path}")
    full_path = os.path.join(model_path, pkl_files[0])
    print(f"Loading model from: {full_path}")
    with open(full_path, 'rb') as f:
        return pickle.load(f)

def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    gains = np.maximum(deltas, 0)
    losses = np.abs(np.minimum(deltas, 0))
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0: return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def run_rsi_simulation():
    print("--- Starting RSI Strategy Simulation ---")
    try:
        model_wrapper = load_latest_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    stock_name = model_wrapper.stock_name
    buffer_days = model_wrapper.time_step * 2 + 50 # Extra buffer for RSI
    fetch_start = start_date - timedelta(days=buffer_days)
    
    df = get_closing_prices(fetch_start.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), stock_name)
    data = df.values
    dates = df.index
    
    balance = 10000.0
    stocks_owned = 0
    initial_balance = balance
    last_buy_price = 0
    
    base_threshold = 0.01
    target_profit = 0.05
    stop_loss = 0.10
    
    sim_dates = []
    actual_prices = []
    predicted_prices = []
    buy_signals = []
    sell_signals = []
    balance_history = []
    buy_threshold_lines = [] # To plot the dynamic threshold
    
    scaler = model_wrapper.scaler
    model = model_wrapper.best_model
    time_step = model_wrapper.time_step
    
    print("Running simulation...")
    
    # Start loop
    start_idx = time_step + 20 # Ensure enough data for RSI
    
    for i in range(start_idx, len(data) - 1):
        current_price = data[i][0]
        current_date = dates[i]
        
        # Prediction
        input_data = data[i-time_step+1 : i+1]
        if len(input_data) != time_step: continue
        input_scaled = scaler.transform(input_data)
        input_reshaped = input_scaled.reshape(1, time_step, 1)
        pred_scaled = model.predict(input_reshaped, verbose=0)
        predicted_price = scaler.inverse_transform(pred_scaled)[0][0]
        
        # RSI Calculation
        # Use last 15 days of prices up to today
        recent_prices = data[i-15:i+1].flatten()
        rsi = calculate_rsi(recent_prices)
        
        # Dynamic Threshold Logic
        current_threshold = base_threshold
        if rsi < 30:
            current_threshold = base_threshold / 2 # More aggressive buy
        
        buy_trigger_price = current_price * (1 + current_threshold)
        
        sim_dates.append(current_date)
        actual_prices.append(current_price)
        predicted_prices.append(predicted_price)
        buy_threshold_lines.append(buy_trigger_price)
        
        # Strategy
        if stocks_owned == 0:
            if predicted_price > buy_trigger_price:
                stocks_to_buy = int(balance // current_price)
                if stocks_to_buy > 0:
                    balance -= stocks_to_buy * current_price
                    stocks_owned += stocks_to_buy
                    last_buy_price = current_price
                    buy_signals.append((len(sim_dates)-1, current_price))
        
        elif stocks_owned > 0:
            if current_price >= last_buy_price * (1 + target_profit):
                balance += stocks_owned * current_price
                stocks_owned = 0
                sell_signals.append((len(sim_dates)-1, current_price))
            elif current_price <= last_buy_price * (1 - stop_loss):
                balance += stocks_owned * current_price
                stocks_owned = 0
                sell_signals.append((len(sim_dates)-1, current_price))
                
        balance_history.append(balance + (stocks_owned * current_price))

    if stocks_owned > 0:
        balance += stocks_owned * actual_prices[-1]
        stocks_owned = 0
        
    print(f"Final Balance: {balance}")
    print(f"Return: {((balance - initial_balance) / initial_balance) * 100:.2f}%")
    
    # Custom Plot for RSI version to include threshold line
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'visuals/strategy_rsi.png'))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(sim_dates, actual_prices, label='Actual Price', color='blue', alpha=0.6, marker='.', markersize=3, linewidth=1)
    ax1.plot(sim_dates, predicted_prices, label='Predicted Price', color='orange', alpha=0.6, linestyle='--', marker='.', markersize=3, linewidth=1)
    ax1.plot(sim_dates, buy_threshold_lines, label='Buy Threshold', color='gray', alpha=0.4, linestyle=':')
    
    if buy_signals:
        buy_indices, buy_prices = zip(*buy_signals)
        ax1.scatter(np.array(sim_dates)[list(buy_indices)], buy_prices, color='green', marker='^', s=150, label='Buy', zorder=5, edgecolors='black')
    if sell_signals:
        sell_indices, sell_prices = zip(*sell_signals)
        ax1.scatter(np.array(sim_dates)[list(sell_indices)], sell_prices, color='red', marker='v', s=150, label='Sell', zorder=5, edgecolors='black')
        
    ax1.set_title('RSI Strategy Simulation')
    ax1.legend()
    ax1.grid(True, which='major', linestyle='-', alpha=0.5)
    ax1.grid(True, which='minor', linestyle=':', alpha=0.2)
    ax1.minorticks_on()
    
    ax2.plot(sim_dates, balance_history, label='Portfolio Value', color='purple', linewidth=1.5)
    ax2.set_title('Portfolio Performance')
    ax2.legend()
    ax2.grid(True, which='major', linestyle='-', alpha=0.5)
    ax2.grid(True, which='minor', linestyle=':', alpha=0.2)
    ax2.minorticks_on()
    
    plt.xticks(rotation=45)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Visual saved to {output_file}")

    # Save Zoomed Version (Last 60 points)
    if len(sim_dates) > 60:
        ax1.set_xlim(sim_dates[-60], sim_dates[-1])
        # Re-scale y-axis for the zoomed range
        last_60_prices = actual_prices[-60:]
        if len(last_60_prices) > 0:
            min_p = min(last_60_prices) * 0.98
            max_p = max(last_60_prices) * 1.02
            ax1.set_ylim(min_p, max_p)
            
        zoomed_filename = output_file.replace('.png', '_zoomed.png')
        plt.savefig(zoomed_filename, dpi=300)
        print(f"Zoomed visual saved to {zoomed_filename}")
    
    plt.close()

if __name__ == "__main__":
    run_rsi_simulation()
