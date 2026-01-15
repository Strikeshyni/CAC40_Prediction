import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

nb_years = 12

def get_yesterday_closing_prices():
    today = datetime.today().strftime('%Y-%m-%d')
    yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    from_date = (datetime.today() - timedelta(days=365*nb_years)).strftime('%Y-%m-%d')
    return get_closing_prices(from_date, yesterday)

def get_closing_prices(from_date, to_date, stock_name=None):
    file_path = f'/home/abel/personnal_projects/CAC40_stock_prediction/dataset/{stock_name}_closing_prices_{from_date}_to_{to_date}.csv'

    # Check if the data already exists in a CSV file
    if os.path.exists(file_path):
        print(f"Loading data from {file_path}")
        stock_data = pd.read_csv(file_path)
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])  # Ensure Date is in datetime format
        stock_data.set_index('Date', inplace=True)  # Set Date as the index
        return stock_data[['Close']]  # Return only the Close column

    # If the data is not cached, fetch it from Yahoo Finance
    print(f"Collecting data from {from_date} to {to_date}")
    data = yf.download(stock_name, start=from_date, end=to_date)

    # Ensure the data contains the Close column
    if isinstance(data.columns, pd.MultiIndex):
        # Handle multi-level columns
        stock_data = data['Close'].reset_index()
        stock_data.rename(columns={stock_name: 'Close'}, inplace=True)
    else:
        # Handle single-level columns
        stock_data = data[['Close']].reset_index()

    # Prepare the dataset
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])  # Ensure Date is in datetime format
    stock_data.set_index('Date', inplace=True)  # Set Date as the index

    # Save the dataset to a CSV file for future use
    stock_data.to_csv(file_path)
    print(f"Data saved to {file_path}")

    return stock_data[['Close']]

# to_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
# from_date = (datetime.today() - timedelta(days=365*nb_years)).strftime('%Y-%m-%d')
# print(f"Collecting data from {from_date} to {to_date}")
# stock_name = 'ENGI.PA'
# data = get_closing_prices(from_date, to_date, stock_name)
# print(data.head())