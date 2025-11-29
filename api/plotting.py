import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime

def generate_simulation_chart(sim_id, daily_results, stock_name, output_dir='api_simulations_plots'):
    """
    Generates and saves a simulation plot based on daily results.
    """
    # Filter out error results
    valid_results = [r for r in daily_results if "error" not in r]
    
    if not valid_results:
        return None

    dates = [datetime.strptime(r["date"], "%Y-%m-%d") for r in valid_results]
    actual_prices = [r["actual_price"] for r in valid_results]
    predicted_prices = [r["predicted_price"] for r in valid_results]
    portfolio_values = [r["portfolio_value"] for r in valid_results]
    
    # Extract buy/sell signals
    buy_signals = []
    sell_signals = []
    
    for i, r in enumerate(valid_results):
        if r["action"] == "buy":
            buy_signals.append((i, r["actual_price"]))
        elif r["action"] == "sell":
            sell_signals.append((i, r["actual_price"]))
            
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{sim_id}.png")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # --- Plot 1: Stock Price & Predictions ---
    ax1.plot(dates, actual_prices, label='Actual Price', color='blue', alpha=0.6, marker='.', markersize=3, linewidth=1)
    ax1.plot(dates, predicted_prices, label='Predicted Price', color='orange', alpha=0.6, linestyle='--', marker='.', markersize=3, linewidth=1)
    
    # Plot Buy Signals
    if buy_signals:
        buy_indices, buy_prices = zip(*buy_signals)
        dates_array = np.array(dates)
        ax1.scatter(dates_array[list(buy_indices)], buy_prices, color='green', marker='^', s=150, label='Buy Signal', zorder=5, edgecolors='black')
        
    # Plot Sell Signals
    if sell_signals:
        sell_indices, sell_prices = zip(*sell_signals)
        dates_array = np.array(dates)
        ax1.scatter(dates_array[list(sell_indices)], sell_prices, color='red', marker='v', s=150, label='Sell Signal', zorder=5, edgecolors='black')
        
    ax1.set_title(f'Simulation {stock_name} - Price Action')
    ax1.set_ylabel('Price (€)')
    ax1.legend()
    ax1.grid(True, which='major', linestyle='-', alpha=0.5)
    ax1.grid(True, which='minor', linestyle=':', alpha=0.2)
    ax1.minorticks_on()
    
    # --- Plot 2: Portfolio Value ---
    ax2.plot(dates, portfolio_values, label='Portfolio Value', color='purple', linewidth=1.5)
    ax2.set_title('Portfolio Performance')
    ax2.set_ylabel('Value (€)')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, which='major', linestyle='-', alpha=0.5)
    ax2.grid(True, which='minor', linestyle=':', alpha=0.2)
    ax2.minorticks_on()
    
    # Rotate date labels
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    
    return filename
