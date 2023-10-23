import os
import logging
import yfinance as yf
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()
logging_level = os.getenv("LOGGING_LEVEL", logging.INFO)
logging.basicConfig(level=logging_level)
logger = logging.getLogger(__name__)

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the grandparent directory (project main directory) of the script's directory
grandparent_dir = os.path.dirname(os.path.dirname(script_dir))

# Construct the path to the data directory
data_dir = os.path.join(grandparent_dir, 'data')

# Construct the path to the vix directory inside the data directory
vix_dir = os.path.join(data_dir, 'vix_prices')

# Construct the path to the S&P500 directory inside the data directory
sp500tr_dir = os.path.join(data_dir, 'sp500_prices')

# Create the data directory if it does not exist
os.makedirs(vix_dir, exist_ok=True)
os.makedirs(sp500tr_dir, exist_ok=True)

def save_sp500tr_prices_to_csv(start_date,
                               end_date):
    # Adjust the end date to fetch data inclusive of the given end_date
    end_date_adjusted = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')

    logger.info(f"Fetching SP500TR data from {start_date} to {end_date}")

    # Fetch data from Yahoo Finance
    sp500tr_data = yf.download('^SP500TR', start=start_date, end=end_date_adjusted)

    if sp500tr_data.empty:
        logger.warning(f"No SP500TR data available for the given date range")
        return

    # Select only the "Adj Close" column
    sp500tr_adj_close = sp500tr_data["Adj Close"]
    sp500tr_adj_close.name = 'S&P 500'

    # Save the DataFrame to CSV file
    sp500tr_adj_close.to_csv(os.path.join(sp500tr_dir, 'SP500TR.csv'), header=True)

    logger.info("SP500TR data fetching complete")

def save_vix_prices_to_csv(start_date,
                           end_date):
    # Adjust the end date to fetch data inclusive of the given end_date
    end_date_adjusted = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')

    logger.info(f"Fetching VIX data from {start_date} to {end_date}")

    # Fetch data from Yahoo Finance
    vix_data = yf.download('^VIX', start=start_date, end=end_date_adjusted)

    if vix_data.empty:
        logger.warning(f"No VIX data available for the given date range")
        return

    # Select only the "Adj Close" column
    vix_adj_close = vix_data["Adj Close"]
    vix_adj_close.name = 'VIX'

    # Save the DataFrame to CSV file
    vix_adj_close.to_csv(os.path.join(vix_dir, 'VIX.csv'), header=True)

    logger.info("VIX data fetching complete")