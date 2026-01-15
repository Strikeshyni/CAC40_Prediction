import sys
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from real_time_simulation.visual_utils import plot_simulation
from web_scrapper.scrapper import get_closing_prices

def load_latest_model():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../api_models'))
    if not os.path.exists(base_dir): raise FileNotFoundError(f"Directory {base_dir} not found.")
    model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not model_dirs: raise FileNotFoundError("No model directories found.")
    latest_dir = sorted(model_dirs)[-1]
    model_path = os.path.join(base_dir, latest_dir)
    pkl_files = [f for f in os.listdir(model_path) if f.endswith('.pkl')]
    if not pkl_files: raise FileNotFoundError(f"No .pkl file found in {model_path}")
    full_path = os.path.join(model_path, pkl_files[0])
    print(f"Loading model from: {full_path}")
    with open(full_path, 'rb') as f: return pickle.load(f)

def calculate_bollinger_bands(prices, period=20, num_std=2):
    sma = pd.Series(prices).rolling(window=period).mean()
    std = pd.Series(prices).rolling(window=period).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return sma, upper_band, lower_band

def run_bollinger_simulation():
    print("--- Starting Bollinger Bands Strategy Simulation ---")
    try:
        model_wrapper = load_latest_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    stock_name = model_wrapper.stock_name
    buffer_days = model_wrapper.time_step * 2 + 50
    fetch_start = start_date - timedelta(days=buffer_days)
    
    df = get_closing_prices(fetch_start.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), stock_name)
    data = df.values
    dates = df.index
    
    balance = 10000.0
    stocks_owned = 0
    initial_balance = balance
    
    sim_dates = []
    actual_prices = []
    predicted_prices = []
    buy_signals = []
    sell_signals = []
    balance_history = []
    
    upper_bands = []
    lower_bands = []
    smas = []
    
    scaler = model_wrapper.scaler
    model = model_wrapper.best_model
    time_step = model_wrapper.time_step
    
    print("Running simulation...")
    
    start_idx = time_step + 25
    
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
        
        # Bollinger Calculation
        recent_prices = data[i-25:i+1].flatten()
        sma_series, upper_series, lower_series = calculate_bollinger_bands(recent_prices)
        
        current_upper = upper_series.iloc[-1]
        current_lower = lower_series.iloc[-1]
        current_sma = sma_series.iloc[-1]
        
        sim_dates.append(current_date)
        actual_prices.append(current_price)
        predicted_prices.append(predicted_price)
        upper_bands.append(current_upper)
        lower_bands.append(current_lower)
        smas.append(current_sma)
        
        # Strategy: 
        # Buy if Predicted Price < Lower Band (Rebound expectation) OR 
        # Buy if Predicted Price > SMA (Trend following) - Let's use Rebound logic combined with prediction
        # Logic: If Prediction says UP and Price is near Lower Band -> Strong Buy
        
        # Simple Bollinger Logic:
        # Buy if Price < Lower Band AND Prediction > Price (Confirmation)
        if stocks_owned == 0:
            if current_price < current_lower and predicted_price > current_price:
                stocks_to_buy = int(balance // current_price)
                if stocks_to_buy > 0:
                    balance -= stocks_to_buy * current_price
                    stocks_owned += stocks_to_buy
                    buy_signals.append((len(sim_dates)-1, current_price))
        
        elif stocks_owned > 0:
            # Sell if Price > Upper Band
            if current_price > current_upper:
                balance += stocks_owned * current_price
                stocks_owned = 0
                sell_signals.append((len(sim_dates)-1, current_price))
                
        balance_history.append(balance + (stocks_owned * current_price))

    if stocks_owned > 0:
        balance += stocks_owned * actual_prices[-1]
        stocks_owned = 0
        
    print(f"Final Balance: {balance}")
    print(f"Return: {((balance - initial_balance) / initial_balance) * 100:.2f}%")
    
    # Custom Plot
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'visuals/strategy_bollinger.png'))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(sim_dates, actual_prices, label='Actual Price', color='blue', alpha=0.6, marker='.', markersize=3, linewidth=1)
    ax1.plot(sim_dates, predicted_prices, label='Predicted Price', color='orange', alpha=0.6, linestyle='--', marker='.', markersize=3, linewidth=1)
    
    # Plot Bands
    ax1.plot(sim_dates, upper_bands, color='gray', alpha=0.3)
    ax1.plot(sim_dates, lower_bands, color='gray', alpha=0.3)
    ax1.fill_between(sim_dates, lower_bands, upper_bands, color='gray', alpha=0.1, label='Bollinger Bands')
    
    if buy_signals:
        buy_indices, buy_prices = zip(*buy_signals)
        ax1.scatter(np.array(sim_dates)[list(buy_indices)], buy_prices, color='green', marker='^', s=150, label='Buy', zorder=5, edgecolors='black')
    if sell_signals:
        sell_indices, sell_prices = zip(*sell_signals)
        ax1.scatter(np.array(sim_dates)[list(sell_indices)], sell_prices, color='red', marker='v', s=150, label='Sell', zorder=5, edgecolors='black')
        
    ax1.set_title('Bollinger Bands Strategy Simulation')
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
    run_bollinger_simulation()
