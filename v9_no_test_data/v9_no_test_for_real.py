import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import pickle

import sys
import os
sys.path.append('/home/abel/personnal_projects/CAC40_stock_prediction/')
from models.model_v1 import model_v1
from real_time_simulation.buying_simulation import buying_simulation
from real_time_simulation.buy_simulation_v2 import placement, placement_v2

actual_prices = []
predicted_prices = []
# Parameters
initial_balance = 100  # Initial cash balance
balance_value = initial_balance  # Initial cash balance
stocks_owned = 0  # No stocks owned initially
nb_years_data = 10
next_day_predict = None
last_buying_value = None

for nb_days_simulation in range(20, 0, -1):
    to_date = datetime.today() - timedelta(days=nb_days_simulation)
    from_date = datetime.today() - timedelta(days=365 * nb_years_data)
    to_date_str = to_date.strftime('%Y-%m-%d')
    from_date_str = from_date.strftime('%Y-%m-%d')
    print(f"Collecting data from {from_date_str} to {to_date_str}")
    stock_name = 'ENGI.PA'
    time_step = 300
    model_manager = None

    model_name = f"from_{from_date_str}_to_{to_date_str}_predict_model_monotunning"
    output_dir = f'/home/abel/personnal_projects/CAC40_stock_prediction/{model_name}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the directory if it doesn't exist

    model_file_path = f'{output_dir}{stock_name}_{from_date_str}_{to_date_str}_model.pkl'

    try:
        if os.path.exists(model_file_path):
            with open(model_file_path, 'rb') as file:
                model_manager = pickle.load(file)
                print(f"Class instance loaded from {model_file_path}")
        elif model_manager is None:
            print("Creating new model")
            model_manager = model_v1(model_name, stock_name, from_date_str, to_date_str, train_size_percent=0.8, val_size_percent=0.2, time_step_train_split=time_step, global_tuning=True)
            # Save the class instance to a file
            with open(model_file_path, 'wb') as file:
                pickle.dump(model_manager, file)

            print(f"Class instance saved to {model_file_path}")

        #print(model_manager.best_model.summary())

        all_for_next_day_predict = np.append(model_manager.X[-1][1:], model_manager.y[-1])
        #print(model_manager.scaler.inverse_transform(all_for_next_day_predict.reshape(-1, 1)))
        predicted_last_day = model_manager.best_model.predict(model_manager.X[-1].reshape(1, -1))
        next_day_predict = model_manager.best_model.predict(all_for_next_day_predict.reshape(1, -1))

        predicted_last_day_dollar = model_manager.scaler.inverse_transform(predicted_last_day).reshape(-1, 1).flatten()[0]
        actual_price_dollar = model_manager.scaler.inverse_transform(model_manager.y[-1].reshape(-1, 1)).flatten()[0]
        predicted_price_dollar = model_manager.scaler.inverse_transform(next_day_predict).reshape(-1, 1).flatten()[0]
        #print(predicted_price_dollar)


        actual_prices.append(actual_price_dollar)
        predicted_prices.append(predicted_price_dollar)
        balance_value, stocks_owned, last_buying_value = placement(actual_price_dollar, predicted_last_day_dollar, predicted_price_dollar, stocks_owned, balance_value, last_buying_value)
        #balance_value, stocks_owned, last_buying_value = placement_v2(actual_price_dollar, predicted_price_dollar, stocks_owned, balance_value, last_buying_value, model_manager.scaler.inverse_transform(model_manager.y.reshape(-1, 1)).flatten())
        print(f"Balance: {balance_value}")
        print(f"Stocks owned: {stocks_owned}")            
    except ValueError as e:
        print(e)
    print(f"\n======================================\n")

print(actual_prices)
print(predicted_prices)
balance_value += stocks_owned * actual_price_dollar
stocks_owned = 0
print(f"Balance Final: {balance_value}€")
print(f"Benefit: {balance_value - initial_balance}€")
    # Predict the next 30 days
    # predictions = model_manager.best_model.predict(model_manager.X_test)
    # predictions = model_manager.scaler.inverse_transform(predictions)
    # #print(predictions)

    # # Plot the predictions
    # plt.figure(figsize=(14, 5))
    # plt.plot(model_manager.scaler.inverse_transform(model_manager.y_test.reshape(-1, 1)), label='Actual Price')
    # plt.plot(predictions, label='Predicted Price')
    # plt.title(f'{stock_name} Price Prediction')
    # plt.xlabel('Days')
    # plt.ylabel('Price')
    # plt.legend()
    # plt.show()

    # Buying simulation
    #best_config, benefit = buying_simulation(model_manager.stock_data, predictions)

    # print("Simulation Results:")
    # print(f"{benefit}€")

    # Export predictions and actual values to a CSV file
    # dates = model_manager.stock_data.index[-len(model_manager.y_test):]  # Get the corresponding dates for the test set
    # actual_prices = model_manager.scaler.inverse_transform(model_manager.y_test.reshape(-1, 1)).flatten()  # Actual prices
    # predicted_prices = predictions.flatten()  # Predicted prices

    # # Create a DataFrame
    # results_df = pd.DataFrame({
    #     'Date': dates,
    #     'Actual Price': actual_prices,
    #     'Predicted Price': predicted_prices
    # })

    # # Save to CSV
    # output_file = f'{output_dir}{stock_name}_predictions_vs_actual.csv'
    # results_df.to_csv(output_file, index=False)
    # print(f"Predictions and actual values saved to {output_file}")

    # import matplotlib.pyplot as plt

    # def predict_and_visualize_first_sequence(model_manager):
    #     # Extract the first sequence
    #     first_sequence = model_manager.X_test[-1].reshape(1, -1, 1)  # Reshape to match model input shape
    #     first_target = model_manager.y_test[-1]  # Actual target value

    #     # Predict the target value
    #     predicted_target = model_manager.best_model.predict(first_sequence)[0][0]
        
    #     predicted_target = model_manager.scaler.inverse_transform(predicted_target.reshape(-1, 1)).flatten()
    #     first_target = model_manager.scaler.inverse_transform(first_target.reshape(-1, 1)).flatten()

    #     # Print the actual and predicted values
    #     print(f"Actual Target Value (Y[0]): {first_target}")
    #     print(f"Predicted Target Value: {predicted_target}")

    #     # Plot the sequence and predictions
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(range(len(first_sequence[0])), model_manager.scaler.inverse_transform(first_sequence[0]), label='Input Sequence (X[0])', marker='o')
    #     plt.axhline(y=first_target, color='r', linestyle='--', label='Actual Target Value (Y[0])')
    #     plt.axhline(y=predicted_target, color='g', linestyle='--', label='Predicted Target Value')
    #     plt.title('Visualization of the First Input Sequence (X[0]) and Predictions')
    #     plt.xlabel('Time Steps')
    #     plt.ylabel('Scaled Value')
    #     plt.legend()
    #     plt.grid()
    #     plt.show()

    # # Call the function after initializing the model
    # predict_and_visualize_first_sequence(model_manager)