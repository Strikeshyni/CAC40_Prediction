import matplotlib.pyplot as plt
import os

def plot_simulation(dates, actual_prices, predicted_prices, buy_signals, sell_signals, 
                   balance_history, title, filename):
    """
    Generates and saves a simulation plot.
    
    Args:
        dates: List of dates corresponding to prices
        actual_prices: List of actual stock prices
        predicted_prices: List of predicted prices
        buy_signals: List of tuples (index, price)
        sell_signals: List of tuples (index, price)
        balance_history: List of portfolio value over time
        title: Title of the chart
        filename: Output filename (full path)
    """
    
    # Create a figure with two subplots (Price and Portfolio Value)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # --- Plot 1: Stock Price & Predictions ---
    # Added markers and increased linewidth for better visibility
    ax1.plot(dates, actual_prices, label='Actual Price', color='blue', alpha=0.6, marker='.', markersize=3, linewidth=1)
    ax1.plot(dates, predicted_prices, label='Predicted Price', color='orange', alpha=0.6, linestyle='--', marker='.', markersize=3, linewidth=1)
    
    # Plot Buy Signals
    if buy_signals:
        buy_indices, buy_prices = zip(*buy_signals)
        # Convert dates to numpy array for indexing with list
        import numpy as np
        dates_array = np.array(dates)
        ax1.scatter(dates_array[list(buy_indices)], buy_prices, color='green', marker='^', s=150, label='Buy Signal', zorder=5, edgecolors='black')
        
    # Plot Sell Signals
    if sell_signals:
        sell_indices, sell_prices = zip(*sell_signals)
        # Convert dates to numpy array for indexing with list
        import numpy as np
        dates_array = np.array(dates)
        ax1.scatter(dates_array[list(sell_indices)], sell_prices, color='red', marker='v', s=150, label='Sell Signal', zorder=5, edgecolors='black')
        
    ax1.set_title(f'{title} - Price Action')
    ax1.set_ylabel('Price (€)')
    ax1.legend()
    # Enhanced grid
    ax1.grid(True, which='major', linestyle='-', alpha=0.5)
    ax1.grid(True, which='minor', linestyle=':', alpha=0.2)
    ax1.minorticks_on()
    
    # --- Plot 2: Portfolio Value ---
    ax2.plot(dates, balance_history, label='Portfolio Value', color='purple', linewidth=1.5)
    ax2.set_title('Portfolio Performance')
    ax2.set_ylabel('Value (€)')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, which='major', linestyle='-', alpha=0.5)
    ax2.grid(True, which='minor', linestyle=':', alpha=0.2)
    ax2.minorticks_on()
    
    # Rotate date labels
    plt.xticks(rotation=45)
    
    # Save High Resolution
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300) # Increased DPI for better zoom
    print(f"Visual saved to {filename}")
    
    # Save Zoomed Version (Last 60 points)
    if len(dates) > 60:
        ax1.set_xlim(dates[-60], dates[-1])
        # Re-scale y-axis for the zoomed range
        last_60_prices = actual_prices[-60:]
        if len(last_60_prices) > 0:
            min_p = min(last_60_prices) * 0.98
            max_p = max(last_60_prices) * 1.02
            ax1.set_ylim(min_p, max_p)
            
        zoomed_filename = filename.replace('.png', '_zoomed.png')
        plt.savefig(zoomed_filename, dpi=500)
        print(f"Zoomed visual saved to {zoomed_filename}")
    
    plt.close()
