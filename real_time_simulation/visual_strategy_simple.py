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

def load_latest_model():
    """Finds and loads the most recent model from api_models/"""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../api_models'))
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Directory {base_dir} not found.")
        
    # Find all model directories
    model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not model_dirs:
        raise FileNotFoundError("No model directories found in api_models/")
        
    # Sort by name (assuming date is in name like from_YYYY-MM-DD...)
    # Or just pick the first one for now if sorting is complex
    latest_dir = sorted(model_dirs)[-1]
    model_path = os.path.join(base_dir, latest_dir)
    
    # Find the .pkl file inside
    pkl_files = [f for f in os.listdir(model_path) if f.endswith('.pkl')]
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl file found in {model_path}")
        
    full_path = os.path.join(model_path, pkl_files[0])
    print(f"Loading model from: {full_path}")
    
    with open(full_path, 'rb') as f:
        model_instance = pickle.load(f)
        
    return model_instance

def run_simple_simulation(nb_days=365):
    print("--- Starting Simple Strategy Simulation ---")
    
    # 1. Load Model
    try:
        model_wrapper = load_latest_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Prepare Data
    # We need data that the model hasn't seen, or just recent data.
    # Let's simulate on the last nb_days days.
    end_date = datetime.now()
    start_date = end_date - timedelta(days=nb_days)
    
    stock_name = model_wrapper.stock_name
    print(f"Fetching data for {stock_name}...")
    
    # Fetch a bit more data for the time_step buffer
    buffer_days = model_wrapper.time_step * 2 
    fetch_start = start_date - timedelta(days=buffer_days)
    
    df = get_closing_prices(fetch_start.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), stock_name)
    data = df.values
    dates = df.index
    
    # 3. Simulation Parameters
    balance = 10000.0
    stocks_owned = 0
    initial_balance = balance
    
    threshold = 0.005  # 0.5% predicted increase to buy
    target_profit = 0.05
    stop_loss = 0.10
    
    # History for plotting
    sim_dates = []
    actual_prices = []
    predicted_prices = []
    buy_signals = []
    sell_signals = []
    balance_history = []
    
    # 4. Loop
    # We start loop where we have enough history for one prediction
    start_idx = model_wrapper.time_step
    
    scaler = model_wrapper.scaler
    model = model_wrapper.best_model
    time_step = model_wrapper.time_step
    
    print("Running simulation...")
    
    for i in range(start_idx, len(data) - 1):
        # Prepare input for prediction
        # We need the last 'time_step' days
        input_data = data[i-time_step:i]
        input_scaled = scaler.transform(input_data)
        input_reshaped = input_scaled.reshape(1, time_step, 1)
        
        # Predict next day (i)
        # Note: In real-time, at day i-1 (close), we predict for day i.
        # Here 'i' is the "current day" we are simulating.
        # Wait, standard LSTM predicts t+1 given t.
        # So at index i, we have data up to i. We predict i+1.
        
        # Let's align:
        # We are at day 'i'. We know price[i].
        # We want to decide to buy/sell for tomorrow 'i+1'.
        # We predict price[i+1] using data up to i.
        
        current_price = data[i][0]
        current_date = dates[i]
        
        # Predict for i+1
        pred_scaled = model.predict(input_reshaped, verbose=0)
        predicted_price = scaler.inverse_transform(pred_scaled)[0][0]
        
        # Store for plot
        sim_dates.append(current_date)
        actual_prices.append(current_price)
        predicted_prices.append(predicted_price)
        
        # Logic
        last_buy_price = 0 # Track this properly
        
        # Simple Strategy Logic
        action = "HOLD"
        
        # BUY
        if stocks_owned == 0:
            # If predicted rise > threshold
            if predicted_price > current_price * (1 + threshold):
                # Buy max
                stocks_to_buy = int(balance // current_price)
                if stocks_to_buy > 0:
                    cost = stocks_to_buy * current_price
                    balance -= cost
                    stocks_owned += stocks_to_buy
                    last_buy_price = current_price
                    buy_signals.append((len(sim_dates)-1, current_price))
                    action = "BUY"
        
        # SELL
        elif stocks_owned > 0:
            # We need to track the price we bought at. 
            # For simplicity in this loop, let's assume FIFO or just track average cost if multiple buys.
            # But here we buy all at once.
            # We need to store 'last_buy_price' in a persistent variable outside loop?
            # Yes, let's fix the variable scope.
            pass 
            
    # Rerunning loop with proper state
    balance = 10000.0
    stocks_owned = 0
    last_buy_price = 0
    
    sim_dates = []
    actual_prices = []
    predicted_prices = []
    buy_signals = []
    sell_signals = []
    balance_history = []

    for i in range(start_idx, len(data) - 1):
        current_price = data[i][0]
        current_date = dates[i]
        
        # Input: last 'time_step' days ending at i
        input_data = data[i-time_step+1 : i+1] # shape (time_step, 1)
        if len(input_data) != time_step:
            continue
            
        input_scaled = scaler.transform(input_data)
        input_reshaped = input_scaled.reshape(1, time_step, 1)
        
        pred_scaled = model.predict(input_reshaped, verbose=0)
        predicted_price = scaler.inverse_transform(pred_scaled)[0][0]
        
        sim_dates.append(current_date)
        actual_prices.append(current_price)
        predicted_prices.append(predicted_price)
        
        # Strategy
        if stocks_owned == 0:
            if predicted_price > current_price * (1 + threshold):
                stocks_to_buy = int(balance // current_price)
                if stocks_to_buy > 0:
                    balance -= stocks_to_buy * current_price
                    stocks_owned += stocks_to_buy
                    last_buy_price = current_price
                    buy_signals.append((len(sim_dates)-1, current_price))
        
        elif stocks_owned > 0:
            # Target Profit
            if current_price >= last_buy_price * (1 + target_profit):
                balance += stocks_owned * current_price
                stocks_owned = 0
                sell_signals.append((len(sim_dates)-1, current_price))
            # Stop Loss
            elif current_price <= last_buy_price * (1 - stop_loss):
                balance += stocks_owned * current_price
                stocks_owned = 0
                sell_signals.append((len(sim_dates)-1, current_price))
                
        # Update portfolio value
        current_val = balance + (stocks_owned * current_price)
        balance_history.append(current_val)

    # Final Sell to calculate total
    if stocks_owned > 0:
        balance += stocks_owned * actual_prices[-1]
        stocks_owned = 0
        
    print(f"Initial Balance: {initial_balance}")
    print(f"Final Balance: {balance}")
    print(f"Return: {((balance - initial_balance) / initial_balance) * 100:.2f}%")
    
    # Plot
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'visuals/strategy_simple.png'))
    plot_simulation(sim_dates, actual_prices, predicted_prices, buy_signals, sell_signals, 
                   balance_history, "Simple Strategy (Threshold 1%)", output_file)

if __name__ == "__main__":
    run_simple_simulation()
