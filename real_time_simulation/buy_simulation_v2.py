def placement(actual_price, predicted_for_last_day, predicted_next_day_price, stocks_owned, balance_value, last_buying_value, threshold=0.01, target_profit=0.05, stop_loss=0.10, risk_ratio=0.5):
    """
    Simulates a single action (buy, sell, or hold) based on the actual price, predicted price, and current portfolio state.

    Parameters:
        actual_price (float): The actual price of the stock for the current day.
        predicted_next_day_price (float): The predicted price of the stock for the next day.
        stocks_owned (int): The number of stocks currently owned.
        balance_value (float): The current balance in cash.
        threshold (float): The threshold for buying (default: 1%).
        target_profit (float): The target profit for selling (default: 5%).
        stop_loss (float): The stop-loss threshold for selling (default: 10%).

    Returns:
        tuple: Updated balance_value and stocks_owned.
    """
    print(f"Prediction difference: {predicted_for_last_day - actual_price} (predicted:{predicted_for_last_day} - actual:{actual_price})")
    print(f"Actual Price: {actual_price}, Predicted Price: {predicted_next_day_price}")
    print(actual_price * (1 + threshold))
    print(predicted_next_day_price > actual_price * (1 + threshold))
    # Buy condition
    if (predicted_next_day_price > predicted_for_last_day * (1 + threshold) or  predicted_next_day_price > actual_price * (1 + threshold)) and stocks_owned == 0:
        # Buy stocks with all available money
        stocks_to_buy = (balance_value * risk_ratio) // actual_price
        balance_value -= stocks_to_buy * actual_price
        stocks_owned += stocks_to_buy
        last_buying_value = actual_price
        print(f"Action: BUY | Stocks Bought: {stocks_to_buy} | New Balance: {balance_value} | Buying Price: {actual_price}")

    # Sell condition: Target profit or stop-loss
    elif stocks_owned > 0:
        # Check target profit
        if actual_price >= stocks_owned * last_buying_value * (1 + target_profit):
            balance_value += stocks_owned * actual_price
            print(f"Action: SELL (Target Profit) | Stocks Sold: {stocks_owned} | New Balance: {balance_value} | Selling Price: {actual_price}")
            stocks_owned = 0  # Reset stocks owned
            last_buying_value = None  # Reset last buying value

        # Check stop-loss
        elif actual_price <= stocks_owned * last_buying_value * (1 - stop_loss):
            balance_value += stocks_owned * actual_price
            print(f"Action: SELL (Stop Loss) | Stocks Sold: {stocks_owned} | New Balance: {balance_value} | Selling Price: {actual_price}")
            stocks_owned = 0  # Reset stocks owned
            last_buying_value = None  # Reset last buying value

    # Hold condition
    else:
        print(f"Action: HOLD | Balance: {balance_value} | Stocks Owned: {stocks_owned}")

    return balance_value, stocks_owned, last_buying_value


import numpy as np

def calculate_rsi(prices, period=14):
    """Calculates the Relative Strength Index (RSI)"""
    deltas = np.diff(prices)
    gains = np.maximum(deltas, 0)
    losses = np.abs(np.minimum(deltas, 0))
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_sma(prices, period=14):
    """Calculates the Simple Moving Average (SMA)"""
    return np.mean(prices[-period:])

def placement_v2(actual_price, predicted_next_day_price, stocks_owned, balance_value, last_buying_value, 
              price_history, threshold=0.01, target_profit=0.05, stop_loss=0.10, risk_ratio=0.5):
    """
    Enhanced trading strategy incorporating RSI, SMA, and risk management.

    Parameters:
        actual_price (float): The actual price of the stock for the current day.
        predicted_next_day_price (float): The predicted price of the stock for the next day.
        stocks_owned (int): The number of stocks currently owned.
        balance_value (float): The current balance in cash.
        last_buying_value (float): Last buying price of the stock.
        price_history (list): A list of previous actual prices for technical analysis.
        threshold (float): Minimum percentage increase in prediction to trigger a buy.
        target_profit (float): Target profit percentage before selling.
        stop_loss (float): Maximum allowable loss percentage before selling.
        risk_ratio (float): Percentage of balance to use for buying.

    Returns:
        tuple: Updated balance_value, stocks_owned, and last_buying_value.
    """
    print(price_history)
    
    if len(price_history) < 14:  # Ensure we have enough data for indicators
        print("Not enough data for technical analysis. Using basic strategy.")
        return balance_value, stocks_owned, last_buying_value

    # Calculate indicators
    rsi = calculate_rsi(price_history)
    sma = calculate_sma(price_history)
    
    print(f"RSI: {rsi}, SMA: {sma}, Actual Price: {actual_price}, Predicted Price: {predicted_next_day_price}")

    # Dynamic Threshold Adjustments
    if rsi < 30:  # Oversold condition -> more aggressive buying
        dynamic_threshold = threshold * 0.5
    elif rsi > 70:  # Overbought condition -> more cautious
        dynamic_threshold = threshold
    else:
        dynamic_threshold = threshold

    # Buy Condition
    if predicted_next_day_price > actual_price * (1 + dynamic_threshold) and stocks_owned == 0:
        stocks_to_buy = (balance_value * risk_ratio) // actual_price
        balance_value -= stocks_to_buy * actual_price
        stocks_owned += stocks_to_buy
        last_buying_value = actual_price
        print(f"BUY | {stocks_to_buy} stocks at {actual_price} | New Balance: {balance_value}")

    # Sell Conditions
    elif stocks_owned > 0:
        # Trailing Stop-Loss (if stock is rising, adjust stop-loss dynamically)
        trailing_stop = max(last_buying_value * (1 - stop_loss), last_buying_value * 1.03)  # Adjust dynamically
        
        if actual_price >= last_buying_value * (1 + target_profit):  # Target profit reached
            balance_value += stocks_owned * actual_price
            print(f"SELL (Target Profit) | {stocks_owned} stocks at {actual_price} | New Balance: {balance_value}")
            stocks_owned = 0
            last_buying_value = None

        elif actual_price <= trailing_stop:  # Stop-loss triggered
            balance_value += stocks_owned * actual_price
            print(f"SELL (Stop-Loss) | {stocks_owned} stocks at {actual_price} | New Balance: {balance_value}")
            stocks_owned = 0
            last_buying_value = None

    else:
        print(f"HOLD | Balance: {balance_value} | Stocks Owned: {stocks_owned}")

    return balance_value, stocks_owned, last_buying_value
