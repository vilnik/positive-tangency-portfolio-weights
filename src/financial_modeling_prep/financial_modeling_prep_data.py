import os
import logging
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging_level = os.getenv("LOGGING_LEVEL", logging.INFO)
logging.basicConfig(level = logging_level)
logger = logging.getLogger(__name__)

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the grandparent directory (project main directory) of the script's directory
grandparent_dir = os.path.dirname(os.path.dirname(script_dir))

# Construct the path to the data directory
data_dir = os.path.join(grandparent_dir, 'data')

# Construct the path to the market caps directory inside the data directory
stock_market_caps_dir = os.path.join(data_dir, 'stock_market_caps')

# Create the data directory if it does not exist
os.makedirs(data_dir, exist_ok=True)

fmp_key = os.getenv("FINANCIAL_MODELING_PREP_KEY")
if fmp_key is None:
    logger.error("Missing FINANCIAL_MODELING_PREP_KEY from environment")
    raise ValueError("Missing FINANCIAL_MODELING_PREP_KEY from environment")


def save_stock_market_caps_to_csv(tickers,
                                  start_date,
                                  end_date):
    # Create the market caps directory if it does not exist
    os.makedirs(stock_market_caps_dir, exist_ok = True)

    total_tickers = len(tickers)

    for index, ticker in enumerate(tickers, start = 1):
        logger.info(f"Fetching market cap data for {ticker} ({index}/{total_tickers})")

        stock_market_caps = []
        stock_market_caps_dates = []

        url = f"https://financialmodelingprep.com/api/v3/historical-market-capitalization/{ticker}?limit=10000&apikey={fmp_key}"

        try:
            data = requests.get(url).json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch market cap data for {ticker}: {e}")
            continue

        for item in data:
            stock_market_caps_dates.append(item["date"])
            stock_market_caps.append(item["marketCap"])

        stock_market_cap_data = pd.DataFrame({ticker: stock_market_caps}, index = stock_market_caps_dates)

        # Convert the index to datetime
        stock_market_cap_data.index = pd.to_datetime(stock_market_cap_data.index)

        # Create a mask for dates within the range
        mask = (stock_market_cap_data.index >= start_date) & (stock_market_cap_data.index <= end_date)

        # Apply the mask
        stock_market_cap_data = stock_market_cap_data.loc[mask]

        # If the DataFrame is not empty, sort it and save to a CSV file
        if not stock_market_cap_data.empty:
            # Sort the index
            stock_market_cap_data.sort_index(inplace = True)

            # Save the DataFrame to csv file
            stock_market_cap_data.to_csv(os.path.join(stock_market_caps_dir, f'{ticker}.csv'))
        else:
            logger.warning(f"No market cap data available for {ticker} in the given date range")

    logger.info("Market cap data fetching complete")