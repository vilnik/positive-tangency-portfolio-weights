import os
from dotenv import load_dotenv
import logging
import requests
import pandas as pd
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta

load_dotenv()
logging_level = os.environ.get("LOGGING_LEVEL", logging.INFO)
logging.basicConfig(level=logging_level)
logger = logging.getLogger(__name__)

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the grandparent directory (project main directory) of the script's directory
grandparent_dir = os.path.dirname(os.path.dirname(script_dir))

# Construct the path to the data directory
data_dir = os.path.join(grandparent_dir, 'data')

# Construct the path to the intraday prices directory inside the data directory
stock_intraday_prices_dir = os.path.join(data_dir, 'stock_intraday_prices')

# Construct the path to the daily prices directory inside the data directory
stock_prices_dir = os.path.join(data_dir, 'stock_prices')

# Create the data directory if it does not exist
os.makedirs(data_dir, exist_ok=True)

alpha_vantage_key = os.environ.get("ALPHA_VANTAGE_KEY")
if alpha_vantage_key is None:
    logger.error("Missing ALPHA_VANTAGE_KEY from environment")
    raise ValueError("Missing ALPHA_VANTAGE_KEY from environment")

def save_stock_intraday_prices_to_csv(tickers,
                                      start_date,
                                      end_date,
                                      frequency,
                                      max_calls_per_minute):
    # Create the intraday prices directory if it does not exist
    os.makedirs(stock_intraday_prices_dir, exist_ok = True)

    start_date = datetime.strptime(start_date, "%Y-%m")
    end_date = datetime.strptime(end_date, "%Y-%m")

    request_counter = 0
    start_time = time.time()

    total_tickers = len(tickers)

    for index, ticker in enumerate(tickers, start=1):
        current_date = start_date
        stock_intraday_price_data = None  # Initialize an empty DataFrame for the ticker
        while current_date <= end_date:
            month = current_date.strftime("%Y-%m")
            try:
                if request_counter == max_calls_per_minute:  # we've hit the limit, need to wait
                    time_elapsed = time.time() - start_time
                    if time_elapsed < 60:  # less than a minute has passed
                        time.sleep(61 - time_elapsed)  # sleep until one minute and one second has passed
                    start_time = time.time()  # reset the start time
                    request_counter = 0  # reset the counter

                logger.info(f"Fetching data for {ticker} ({index}/{total_tickers}) on {month}")
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval={frequency}&month={month}&outputsize=full&apikey={alpha_vantage_key}"
                r = requests.get(url)
                r.raise_for_status()  # Will raise an exception if the request failed

                data = r.json()
                time_series = data.get(f"Time Series ({frequency})", None)

                if not time_series:
                    logger.warning(f"Unexpected data format for {ticker} on {month}")
                    logger.warning(data)
                    logger.warning("skipping...")
                    current_date += relativedelta(months=12)  # Increase 12 months when there is missing data
                    request_counter += 1
                    continue  # Skip to the next iteration

                # Extract close prices and timestamps
                timestamps = pd.to_datetime(list(time_series.keys()))
                closes = [float(values['4. close']) for values in time_series.values()]

                # Create a DataFrame for this ticker and append it to the ticker's DataFrame
                stock_monthly_price_data = pd.DataFrame({f'{ticker}': closes}, index=timestamps)
                if stock_intraday_price_data is None:
                    stock_intraday_price_data = stock_monthly_price_data
                else:
                    stock_intraday_price_data = stock_intraday_price_data.append(stock_monthly_price_data)

                request_counter += 1  # increment the counter after successful request
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to fetch data for {ticker} on {month}: {e}")

            current_date += relativedelta(months=1)

        # Save the ticker's DataFrame to a csv file in the intraday-data directory
        if stock_intraday_price_data is not None:
            stock_intraday_price_data.sort_index(inplace=True)  # Sort the DataFrame by index
            stock_intraday_price_data.to_csv(os.path.join(stock_intraday_prices_dir, f'{ticker}.csv'))

    logger.info("Data fetching complete")


def save_stock_prices_to_csv(tickers,
                             start_date,
                             end_date,
                             max_calls_per_minute):
    # Create the daily prices directory if it does not exist
    os.makedirs(stock_prices_dir, exist_ok = True)

    request_counter = 0
    start_time = time.time()

    total_tickers = len(tickers)

    for index, ticker in enumerate(tickers, start=1):
        try:
            if request_counter == max_calls_per_minute:  # we've hit the limit, need to wait
                time_elapsed = time.time() - start_time
                if time_elapsed < 60:  # less than a minute has passed
                    time.sleep(61 - time_elapsed)  # sleep until one minute and one second has passed
                start_time = time.time()  # reset the start time
                request_counter = 0  # reset the counter

            logger.info(f"Fetching data for {ticker} ({index}/{total_tickers})")
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={alpha_vantage_key}"
            r = requests.get(url)
            r.raise_for_status()  # Will raise an exception if the request failed

            data = r.json()

            time_series = data.get('Time Series (Daily)', None)

            if not time_series:
                logger.warning(f"Unexpected data format for {ticker}")
                logger.warning(data)
                logger.warning("skipping...")
                request_counter += 1
                continue  # Skip to the next iteration

            # Create DataFrame from time_series
            stock_price_data = pd.DataFrame(time_series).T  # Transpose the DataFrame since keys are dates

            # Rename the '5. adjusted close' column to 'Adjusted Close'
            stock_price_data.rename(columns={'5. adjusted close': 'Adjusted Close'}, inplace=True)

            # Convert 'Adjusted Close' to float
            stock_price_data['Adjusted Close'] = stock_price_data['Adjusted Close'].astype(float)

            # Convert the index to DateTime
            stock_price_data.index = pd.to_datetime(stock_price_data.index)

            # Apply date filter based on the start_date and end_date
            mask = (stock_price_data.index >= start_date) & (stock_price_data.index <= end_date)
            stock_price_data = stock_price_data.loc[mask]

            # Fetch only the 'Adjusted Close' column and reverse the dataframe
            stock_price_data = stock_price_data[['Adjusted Close']].iloc[::-1]

            # Save the DataFrame to csv file
            stock_price_data.to_csv(os.path.join(stock_prices_dir, f'{ticker}.csv'))

            request_counter += 1  # increment the counter after successful request
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch data for {ticker}: {e}")