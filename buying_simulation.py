def buying_simulation(stock_data, predictions):
    # Parameters
    initial_money = 1000  # Initial virtual money
    threshold = 0.01  # Threshold for buying/selling (1%)
    target_profit = 0.05  # Target profit (5%)
    stop_loss = 0.10  # Stop-loss (10%)

    # Limit the simulation to 365 days
    months = 12
    days_to_simulate = min(len(predictions), months * 30)
    limited_predictions = predictions[:days_to_simulate]

    def simulate_strategy(threshold, target_profit, stop_loss):
        money = initial_money
        stocks_owned = 0  # Ensure this is a scalar integer
        buying_price = None  # Track the buying price for stop-loss and target profit
        money_over_time = []  # To track the evolution of money over time

        # Iterate through predictions
        for i in range(len(limited_predictions) - 1):
            current_predicted_price = limited_predictions[i][0]
            next_predicted_price = limited_predictions[i + 1][0]
            actual_price = stock_data.iloc[len(stock_data) - len(predictions) + i]  # Actual price for the current day

            # Buy condition
            if next_predicted_price > current_predicted_price * (1 + threshold) and stocks_owned == 0:
                # Buy stocks with all available money
                stocks_to_buy = money // actual_price
                money -= stocks_to_buy * actual_price
                stocks_owned += stocks_to_buy  # Update as a scalar integer
                buying_price = actual_price  # Record the buying price

            # Sell condition: Target profit or stop-loss
            elif stocks_owned > 0:  # Ensure this is a scalar comparison
                # Check target profit
                if actual_price >= buying_price * (1 + target_profit):
                    money += stocks_owned * actual_price
                    stocks_owned = 0  # Reset to scalar integer
                    buying_price = None  # Reset buying price

                # Check stop-loss
                elif actual_price <= buying_price * (1 - stop_loss):
                    money += stocks_owned * actual_price
                    stocks_owned = 0  # Reset to scalar integer
                    buying_price = None  # Reset buying price

            # Track money over time
            money_over_time.append(money + stocks_owned * actual_price)

        # Final value of portfolio
        final_value = money + stocks_owned * stock_data.iloc[-1]

        print(f"\nConfig: Threshold={threshold}, Target Profit={target_profit}, Stop Loss={stop_loss}")
        print(f"Initial money: {initial_money}€")
        print(f"Final portfolio value after {days_to_simulate} days: {final_value}€\n")
        return final_value


    thresholds = [i/10000 for i in range(0, 305, 5)]  # 0%, ..., 0.03%
    target_profits = [i/100 for i in range(2, 11, 1)]  # 2%, ..., 10%
    stop_losses = [i/100 for i in range(3, 21, 1)]  # 3%, ..., 20%

    best_config = None
    best_final_value = 0

    for threshold in thresholds:
        for target_profit in target_profits:
            for stop_loss in stop_losses:
                # Simulate the strategy with the current configuration$
                final_value = simulate_strategy(threshold, target_profit, stop_loss)
                if final_value > best_final_value:
                    best_final_value = final_value
                    best_config = (threshold, target_profit, stop_loss)
    benefit = best_final_value - initial_money
    print(f"Best Configuration: Threshold={best_config[0]}, Target Profit={best_config[1]}, Stop Loss={best_config[2]}\n\tWith Final Value: {best_final_value}€\n\tBenefits: {benefit}€\n\tProfit Ratio: {((best_final_value - initial_money) / initial_money) * 100:.2f}%")
    return best_config, benefit