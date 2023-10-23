import os
import pandas as pd
import numpy as np
import alpha_vantage
import financial_modeling_prep
import yahoo_finance
from dotenv import load_dotenv
import logging

load_dotenv()
logging_level = os.environ.get("LOGGING_LEVEL", logging.INFO)
logging.basicConfig(level=logging_level)
logger = logging.getLogger(__name__)

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the script's directory
parent_dir = os.path.dirname(script_dir)

# Construct the path to the data directory
data_dir = os.path.join(parent_dir, 'data')

# Construct the path to the stock prices directory inside the data directory
stock_prices_dir = os.path.join(data_dir, 'stock_prices')

# Construct the path to the market caps directory inside the data directory
stock_market_caps_dir = os.path.join(data_dir, 'stock_market_caps')

# Construct the path to the S&P500 prices directory inside the data directory
sp500tr_dir = os.path.join(data_dir, 'sp500_prices')

# Construct the path to the treasury bill rate inside the data directory
treasury_bill_rate_dir = os.path.join(data_dir, 'treasury_bill_rate')

# Construct the path to the S&P500 components directory inside the data directory
sp500_components_dir = os.path.join(data_dir, 'sp500_components')

intraday_frequency = os.environ.get("INTRADAY_FREQUENCY")


def check_directory_for_csv(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        logger.info(f"The directory '{directory}' does not exist.")
        return False

    # Check if the directory contains any CSV files
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    if not files:
        logger.info(f"The directory '{directory}' does not contain any CSV files.")
        return False

    return True

def load_all_csv_to_dataframe(directory):
    logger.info(f"Loading csv from {directory} into dataframe.")
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    dataframes = []

    for f in files:
        df = pd.read_csv(os.path.join(directory, f), index_col=0, parse_dates=True)
        df.columns = [f.replace('.csv', '')]  # renaming the column as the file name (without the .csv extension)
        dataframes.append(df)

    large_df = pd.concat(dataframes, axis=1)  # concatenate the dataframes horizontally
    large_df = large_df.ffill()

    return large_df

def extract_unique_tickers(start_date_ts,
                           end_date_ts):
    if check_directory_for_csv(sp500_components_dir):
        tickers_df = load_all_csv_to_dataframe(sp500_components_dir)
    else:
        raise ValueError(f"S&P 500 historical components must be downloaded and located in /data/sp500_components. Please visit https://github.com/fja05680/sp500")

    # Convert index to datetime
    tickers_df.index = pd.to_datetime(tickers_df.index)

    # Find the closest date before the specified start date
    closest_start_date = tickers_df[tickers_df.index <= start_date_ts].index.max()
    if pd.isnull(closest_start_date):
        logger.error(f"No S&P 500 historical components available before the specified start date: {start_date_ts}")
        raise ValueError(f"No S&P 500 historical components.")

    # Filter rows between start_date and end_date
    filtered_tickers_df = tickers_df[(tickers_df.index  >= closest_start_date) & (tickers_df.index  <= end_date_ts)].copy()

    # Split tickers strings into lists
    filtered_tickers_df['split_tickers'] = filtered_tickers_df[
        'S&P 500 Historical Components & Changes(08-01-2023)'].str.split(',')

    # Flatten list of lists and get unique tickers
    unique_tickers = pd.unique([ticker for sublist in filtered_tickers_df['split_tickers'] for ticker in sublist])

    return list(unique_tickers)

def get_stock_prices():
    str_start_date = "2000-01-01"
    str_end_date = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    max_calls_per_minute = 75

    if check_directory_for_csv(stock_prices_dir):
        stock_prices_df = load_all_csv_to_dataframe(stock_prices_dir)
    else:
        # Prompt the user for a response
        user_response = input("Do you want to save stock prices to CSV? (Y/N): ").strip().upper()

        if user_response == 'Y':
            unique_tickers_list = extract_unique_tickers(pd.to_datetime("1999-12-31"),
                                                         pd.to_datetime("2023-06-30"))
            alpha_vantage.save_stock_prices_to_csv(unique_tickers_list, str_start_date, str_end_date, max_calls_per_minute)
            stock_prices_df = load_all_csv_to_dataframe(stock_prices_dir)
        else:
            stock_prices_df = None

    # Use pd.Timestamp date
    stock_prices_df.index = pd.to_datetime(stock_prices_df.index)

    return stock_prices_df

def get_stock_market_caps():
    str_start_date = "2000-01-01"
    str_end_date = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    if check_directory_for_csv(stock_market_caps_dir):
        stock_market_caps_df = load_all_csv_to_dataframe(stock_market_caps_dir)
    else:
        # Prompt the user for a response
        user_response = input("Do you want to save market caps to CSV? (Y/N): ").strip().upper()

        if user_response == 'Y':
            unique_tickers_list = extract_unique_tickers(pd.to_datetime("1999-12-31"),
                                                         pd.to_datetime("2023-06-30"))
            financial_modeling_prep.save_stock_market_caps_to_csv(unique_tickers_list, str_start_date, str_end_date)
            stock_market_caps_df = load_all_csv_to_dataframe(stock_market_caps_dir)
        else:
            stock_market_caps_df = None

    # Use pd.Timestamp date
    stock_market_caps_df.index = pd.to_datetime(stock_market_caps_df.index)

    return stock_market_caps_df

def get_sp500tr_prices():
    str_start_date = "2000-01-01"
    str_end_date = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    csv_file = os.path.join(sp500tr_dir, 'SP500TR.csv')

    if os.path.exists(csv_file):
        sp500tr_prices = pd.read_csv(csv_file, index_col=0, parse_dates=True)  # Assuming the first column is a date index
    else:
        # Prompt the user for a response
        user_response = input("Do you want to save SP500TR data to CSV? (Y/N): ").strip().upper()

        if user_response == 'Y':
            yahoo_finance.save_sp500tr_prices_to_csv(str_start_date, str_end_date)
            sp500tr_prices = pd.read_csv(csv_file, index_col=0, parse_dates=True)  # Reload the saved CSV
        else:
            sp500tr_prices = None

    # Use pd.Timestamp date
    sp500tr_prices.index = pd.to_datetime(sp500tr_prices.index)

    return sp500tr_prices


def get_treasury_bill_rate():
    csv_file = os.path.join(treasury_bill_rate_dir, 'DTB3.csv')

    if os.path.exists(csv_file):
        # Read CSV and treat '.' as NaN
        treasury_bill_rate = pd.read_csv(csv_file, index_col=0, parse_dates=True, na_values=['.'])
    else:
        raise ValueError(f"Risk-free rate csv must be downloaded. Please visit https://fred.stlouisfed.org/series/DTB3")

    # Use pd.Timestamp date
    treasury_bill_rate.index = pd.to_datetime(treasury_bill_rate.index)

    # Convert 'DTB3' column to float
    treasury_bill_rate['DTB3'] = treasury_bill_rate['DTB3'].astype(float)

    # Divide 'DTB3' values by 100
    treasury_bill_rate['DTB3'] = treasury_bill_rate['DTB3'] / 100.0

    # Sort the index
    treasury_bill_rate = treasury_bill_rate.sort_index()

    # For a T-bill with:
    # Face Value (F): $100
    # Purchase Price (P): $99
    # Days to Maturity (TTM): 20 days
    # Discount Rate (DR) = ($100 - $99)/$100 * 360/20, i.e., (F-P)/F*360/TTM
    # The actual ann. ytm is (F-P)/P*365/TTM
    # This means
    # P = F * (1 - DR*TTM/360)
    # and
    # ytm = (DR * TTM / 360) / (1 - DR * TTM / 360) * 365 / TTM
    # Bond equivalent yield is
    # (1 + ytm / 2)^2 - 1

    return treasury_bill_rate

def get_market_data():
    stock_prices_df = get_stock_prices()
    stock_simple_returns_df = stock_prices_df.pct_change()
    stock_log_returns_df = np.log(stock_prices_df / stock_prices_df.shift(1))
    stock_market_caps_df = get_stock_market_caps()
    sp500_prices_df = get_sp500tr_prices()
    sp500_simple_returns_df = sp500_prices_df.pct_change()
    treasury_bill_rate_df = get_treasury_bill_rate()

    return {"stock_prices_df": stock_prices_df,
            "stock_simple_returns_df": stock_simple_returns_df,
            "stock_log_returns_df": stock_log_returns_df,
            "stock_market_caps_df": stock_market_caps_df,
            "sp500_prices_df": sp500_prices_df,
            "sp500_simple_returns_df": sp500_simple_returns_df,
            "treasury_bill_rate_df": treasury_bill_rate_df}
