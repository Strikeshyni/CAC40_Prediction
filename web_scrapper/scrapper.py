import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

nb_years = 12

def get_yesterday_closing_prices():
    today = datetime.today().strftime('%Y-%m-%d')
    yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    from_date = (datetime.today() - timedelta(days=365*nb_years)).strftime('%Y-%m-%d')
    return get_closing_prices(from_date, yesterday)

import os

def get_closing_prices(from_date, to_date, stock_name=None):
    # Add one day to to_date
    to_date_dt = datetime.strptime(to_date, '%Y-%m-%d')  # Convert to datetime object
    to_date_plus_one = (to_date_dt + timedelta(days=1)).strftime('%Y-%m-%d')  # Add one day and convert back to string

    # Define the relative path for the dataset directory
    dataset_dir = 'dataset'
    # Ensure the directory exists
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)  # Create the directory if it doesn't exist

    # Define the file path for saving the data
    file_path = f'{dataset_dir}/{stock_name}_closing_prices_{from_date}_to_{to_date}.csv'

    if os.path.exists(file_path):
        print(f"Data already exists at {file_path}. Loading from file.\n")
        closing_prices = pd.read_csv(file_path)
        closing_prices.set_index('Date', inplace=True)
        #print(closing_prices.head())
        return closing_prices

    print(f"Collecting data from {from_date} to close of {to_date}\n")
    data = yf.download(stock_name, start=from_date, end=to_date_plus_one)
    # print(data.head())
    # print(data.index[-1])
    if data.empty:
        raise ValueError(f"No data found for {stock_name} from {from_date} to {to_date}")
    last_date_in_data = data.index[-1].strftime('%Y-%m-%d')
    if last_date_in_data != to_date:
        raise ValueError(f"Last date in data ({last_date_in_data}) does not match to_date ({to_date})")
    # Extract the 'Close' prices
    closing_prices = data['Close']

    # Save the data to the CSV file
    closing_prices.to_csv(file_path)
    print(f"Data saved to {file_path}\n")

    return closing_prices

if __name__ == "__main__":
    # Example usage
    from_date = '2020-01-01'
    to_date = '2025-03-25'
    stock_name = 'ENGI.PA'
    closing_prices = get_closing_prices(from_date, to_date, stock_name)
    print(closing_prices.head())