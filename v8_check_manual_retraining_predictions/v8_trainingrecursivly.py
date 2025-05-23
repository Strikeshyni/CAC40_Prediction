import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import pickle

import sys
import os
sys.path.append('/home/abel/personnal_projects/CAC40_stock_prediction/')
from model_v1 import model_v1
from buying_simulation import buying_simulation
sys.path.append('/home/abel/personnal_projects/CAC40_stock_prediction/v8_check_manual_retraining_predictions/')

# Parameters
nb_years = 10
to_date = (datetime.today() - timedelta(days=0)).strftime('%Y-%m-%d')
from_date = (datetime.today() - timedelta(days=365*nb_years)).strftime('%Y-%m-%d')
print(f"Collecting data from {from_date} to {to_date}")
stock_name = 'ENGI.PA'
time_step = 300
model_manager = None

if os.path.exists(f'{stock_name}_{from_date}_{to_date}_model.pkl'):
    with open(f'{stock_name}_{from_date}_{to_date}_model.pkl', 'rb') as file:
        model_manager = pickle.load(file)
        print("Class instance loaded from 'model_v1_instance.pkl'")
elif model_manager is None:
    print("Creating new model")
    model_manager = model_v1(stock_name, from_date, to_date, time_step_train_split=time_step)
    # Save the class instance to a file
    with open(f'{stock_name}_{from_date}_{to_date}_model.pkl', 'wb') as file:
        pickle.dump(model_manager, file)

    print(f"Class instance saved to '{stock_name}_{from_date}_{to_date}_model.pkl''")

print(model_manager.best_model.summary())
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
# output_file = f'{stock_name}_predictions_vs_actual.csv'
# results_df.to_csv(output_file, index=False)
# print(f"Predictions and actual values saved to {output_file}")

import matplotlib.pyplot as plt

def predict_and_visualize_first_sequence(model_manager):
    # Extract the first sequence
    first_sequence = model_manager.X_test[-1].reshape(1, -1, 1)  # Reshape to match model input shape
    first_target = model_manager.y_test[-1]  # Actual target value

    # Predict the target value
    predicted_target = model_manager.best_model.predict(first_sequence)[0][0]
    
    predicted_target = model_manager.scaler.inverse_transform(predicted_target.reshape(-1, 1)).flatten()
    first_target = model_manager.scaler.inverse_transform(first_target.reshape(-1, 1)).flatten()

    # Print the actual and predicted values
    print(f"Actual Target Value (Y[0]): {first_target}")
    print(f"Predicted Target Value: {predicted_target}")

    # Plot the sequence and predictions
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(first_sequence[0])), model_manager.scaler.inverse_transform(first_sequence[0]), label='Input Sequence (X[0])', marker='o')
    plt.axhline(y=first_target, color='r', linestyle='--', label='Actual Target Value (Y[0])')
    plt.axhline(y=predicted_target, color='g', linestyle='--', label='Predicted Target Value')
    plt.title('Visualization of the First Input Sequence (X[0]) and Predictions')
    plt.xlabel('Time Steps')
    plt.ylabel('Scaled Value')
    plt.legend()
    plt.grid()
    plt.show()

# Call the function after initializing the model
predict_and_visualize_first_sequence(model_manager)