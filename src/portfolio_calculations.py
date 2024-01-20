import os
import math
import pandas as pd
import pymc as pm
import xarray as xr
import arviz as az
import pytensor.tensor as pt
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import black_litterman
from pypfopt.black_litterman import BlackLittermanModel
from datetime import timedelta
from scipy.stats import gamma
import numpy as np
import data_handling
from dotenv import load_dotenv
import logging
import csv
import matplotlib.pyplot as plt
import seaborn as sns

# Logging setup
load_dotenv()
logging_level = os.environ.get("LOGGING_LEVEL", logging.INFO)
logging.basicConfig(level=logging_level)
logger = logging.getLogger(__name__)

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the script's directory
parent_dir = os.path.dirname(script_dir)

# Construct the path to the MCMC results
mcmc_results = os.path.join(parent_dir, 'mcmc_results')
os.makedirs(mcmc_results, exist_ok=True)

def time_periods_per_year(portfolio_spec):
    if portfolio_spec["rebalancing_frequency"] == "daily":
        frequency = 252
    elif portfolio_spec["rebalancing_frequency"] == "weekly":
        frequency = 52
    elif portfolio_spec["rebalancing_frequency"] == "monthly":
        frequency = 12
    else:
        logger.error(f"Unknown rebalancing frequency")
        raise ValueError(f"Unknown rebalancing frequency")

    return frequency

def save_dict_as_csv(data_dict, csv_file):
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in data_dict.items():
            writer.writerow([key, value])

def read_dict_from_csv(csv_file):
    data_dict = {}
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            key, value = row
            key = key.strip().replace('"', '')
            if '(' in value and ')' in value:
                values = [v.strip().replace("'", "") for v in value.strip('()').split(',')]
                values = [int(v) if v.isdigit() or (v.startswith('-') and v[1:].isdigit()) else v for v in values]
                data_dict[key] = tuple(values)
            elif value.replace('.', '', 1).isdigit():
                data_dict[key] = float(value)
            else:
                data_dict[key] = value.replace('"', '').strip()
    return data_dict

def calculate_simple_returns_from_prices(stock_prices_df):
    logger.info(f"Calculating simple returns.")

    # Calculate the percentage change for each stock
    stock_simple_returns_df = stock_prices_df.pct_change()

    # Drop NaN values, which occur for the first data point
    stock_simple_returns_df.dropna(inplace=True)

    return stock_simple_returns_df


def calculate_log_returns_from_prices(stock_prices_df):
    logger.info(f"Calculating log returns.")

    # Calculate the log returns for each stock
    stock_log_returns_df = np.log(stock_prices_df / stock_prices_df.shift(1))

    # Drop NaN values, which occur for the first data point
    stock_log_returns_df.dropna(inplace=True)

    return stock_log_returns_df

def calculate_bond_equivalent_yield_from_DTB3(treasury_bill_rate_df):
    days_to_maturity = 90
    ann_ytm = (treasury_bill_rate_df * days_to_maturity / 360) / (1 - treasury_bill_rate_df * days_to_maturity / 360) * 365 / days_to_maturity
    treasury_bill_rate_bey_df = (1 + ann_ytm / 2)**2 - 1

    # 2. Comparison with Original Data
    differences = (treasury_bill_rate_bey_df['DTB3'] - treasury_bill_rate_df['DTB3']).abs()
    if differences.mean() > 0.01:
        logger.error("Average difference between BEY and original data is too large. Please inspect the results.")
        raise ValueError("Average difference between BEY and original data is too large. Please inspect the results.")

    return treasury_bill_rate_bey_df

def calculate_excess_returns_from_prices(portfolio_spec,
                                         stock_prices_df,
                                         treasury_bill_rate_df,
                                         log_returns):
    logger.info(f"Calculating excess log returns.")

    if portfolio_spec["rebalancing_frequency"] == "daily":
        trading_days_between_rebalancing = 1
        calendar_days_between_rebalancing = 1.4
    elif portfolio_spec["rebalancing_frequency"] == "weekly":
        trading_days_between_rebalancing = 5
        calendar_days_between_rebalancing = 7
    elif portfolio_spec["rebalancing_frequency"] == "monthly":
        trading_days_between_rebalancing = 21
        calendar_days_between_rebalancing = 31
    else:
        # Log an error and raise an exception if the rebalancing frequency is unknown
        logger.error("Unknown rebalancing frequency.")
        raise RuntimeError("Unknown rebalancing frequency.")

    average_timedelta_between_stock_prices = (stock_prices_df.index[1:] - stock_prices_df.index[:-1]).mean()
    average_days_between_stock_prices  = average_timedelta_between_stock_prices.total_seconds() / 86400
    if abs(average_days_between_stock_prices - calendar_days_between_rebalancing) > 1.5:
        stock_prices_df = daily_prices_to_rebalancing_frequency_and_window(portfolio_spec,
                                                                    trading_date_ts,
                                                                    stock_prices_df)

    if log_returns:
        # Calculate the log returns for each stock
        stock_returns_df = calculate_log_returns_from_prices(stock_prices_df)
    else:
        # Calculate simple returns
        stock_returns_df = calculate_simple_returns_from_prices(stock_prices_df)

    # Adjust risk-free rate
    treasury_bill_rate_bey_df = calculate_bond_equivalent_yield_from_DTB3(treasury_bill_rate_df)
    treasury_bill_rate_bey_adjusted_df = (1 + treasury_bill_rate_bey_df) ** (trading_days_between_rebalancing / 252) - 1

    # Resample and interpolate risk-free rates to match stock returns' dates
    treasury_bill_rate_bey_resampled_df = treasury_bill_rate_bey_adjusted_df.reindex(stock_returns_df.index, method='ffill')

    if log_returns:
        treasury_bill_rate_bey_resampled_df = np.log(1 + treasury_bill_rate_bey_resampled_df)

    # Calculate the excess returns
    stock_excess_returns_df = stock_returns_df - treasury_bill_rate_bey_resampled_df.values

    # Drop NaN values, which occur for the first data point
    stock_excess_returns_df.dropna(inplace=True)

    return stock_excess_returns_df

def calculate_portfolio_variance(portfolio_comp_df, 
                                 covariance_matrix_df):
    logger.info(f"Calculating portfolio variance.")

    # Sort the portfolio DataFrame by index (stock symbols)
    sorted_portfolio_comp_df = portfolio_comp_df.sort_index()
    sorted_weights_np = sorted_portfolio_comp_df['Weight'].to_numpy()

    # Sort the covariance DataFrame by stock symbols and convert to a numpy array
    sorted_keys = sorted_portfolio_comp_df.index
    sorted_covariance_matrix_df = covariance_matrix_df.loc[sorted_keys, sorted_keys]
    sorted_covariance_matrix_np = sorted_covariance_matrix_df.to_numpy()

    # Compute the portfolio variance as w^T * S * w
    portfolio_variance = np.dot(sorted_weights_np.T, np.dot(sorted_covariance_matrix_np, sorted_weights_np))
    # Same as portfolio_comp_df["Weight"].T.dot(covariance_matrix_df.dot(portfolio_comp_df["Weight"]))

    return portfolio_variance

def calculate_rolling_window_frequency_adjusted(portfolio_spec):
    logger.info(f"Calculating rolling window frequency adjusted.")

    # Check the rebalancing frequency specified in portfolio_spec
    if portfolio_spec["rebalancing_frequency"] == "daily":
        rolling_window_frequency_adjusted = portfolio_spec["rolling_window_days"]
    elif portfolio_spec["rebalancing_frequency"] == "weekly":
        rolling_window_frequency_adjusted = math.floor(portfolio_spec["rolling_window_days"] / 5)
    elif portfolio_spec["rebalancing_frequency"] == "monthly":
        rolling_window_frequency_adjusted = math.floor(portfolio_spec["rolling_window_days"] / 21)
    else:
        # Log an error and raise an exception if the rebalancing frequency is unknown
        logger.error("Unknown rebalancing frequency.")
        raise RuntimeError("Unknown rebalancing frequency.")

    return rolling_window_frequency_adjusted


def daily_prices_to_rebalancing_frequency_and_window(portfolio_spec,
                                                     trading_date_ts,
                                                     stock_prices_df):
    logger.info(f"Adjusting daily prices to rebalancing frequency and rolling window.")

    # Calculate the rolling window size based on the portfolio's rebalancing frequency
    rolling_window_frequency_adjusted = calculate_rolling_window_frequency_adjusted(portfolio_spec)

    # Adjust the stock prices DataFrame based on the rebalancing frequency
    if portfolio_spec["rebalancing_frequency"] == "weekly":
        stock_prices_df_frequency_adjusted = stock_prices_df.iloc[::-1][::5][::-1]
    elif portfolio_spec["rebalancing_frequency"] == "monthly":
        stock_prices_df_frequency_adjusted = stock_prices_df.iloc[::-1][::21][::-1]
    else:
        logger.error("Unknown rebalancing frequency.")
        raise RuntimeError("Unknown rebalancing frequency.")

    # Find the position of the current trading date in the adjusted DataFrame
    position_current_date = stock_prices_df_frequency_adjusted.index.get_loc(trading_date_ts)

    # Calculate the start position for the rolling window
    start_position = position_current_date - (rolling_window_frequency_adjusted - 1)

    # Check for invalid start position
    if start_position < 0:
        logger.error(f"Start position is smaller than 0.")
        raise ValueError(f"Start position is smaller than 0.")

    # Slice the DataFrame to only include data within the rolling window
    stock_prices_frequency_and_window_adjusted_df = stock_prices_df_frequency_adjusted.iloc[
                                                      start_position:position_current_date + 1]

    return stock_prices_frequency_and_window_adjusted_df

def calculate_prior_w(portfolio_spec,
                    trading_date_ts,
                    k_stock_prices_df,
                    k_stock_market_caps_df,
                    treasury_bill_rate_df):
    logger.info(f"Calculating prior w.")

    # Count the number of stocks in the portfolio
    num_stocks = len(k_stock_prices_df.columns)
    # Initialize equal weights for each stock
    if portfolio_spec["prior_weights"] == "empty":
        portfolio_comp_df = pd.DataFrame({
            'Weight': [0] * num_stocks
        }, index=k_stock_prices_df.columns)

        portfolio_comp_df.index.name = 'Stock'
    elif portfolio_spec["prior_weights"] == "value_weighted":
        portfolio_comp_df = calculate_value_weighted_portfolio(portfolio_spec,
                                                               trading_date_ts,
                                                               k_stock_market_caps_df)
    elif portfolio_spec["prior_weights"] == "equally_weighted":
        portfolio_comp_df = calculate_equally_weighted_portfolio(portfolio_spec,
                                                                 k_stock_prices_df)
    else:
        logger.error(f"Unknown portfolio prior weights.")
        raise ValueError("Unknown portfolio prior weights.")

    return portfolio_comp_df

def get_k_largest_stocks_market_caps(stock_market_caps_df,
                                     stock_prices_df,
                                     trading_date_ts,
                                     portfolio_size,
                                     rolling_window_days,
                                     rebalancing_frequency):
    # Get S&P 500 components for the current date
    tickers_list = data_handling.extract_unique_tickers(trading_date_ts,
                                          trading_date_ts)

    # Identify tickers that are present in stock_market_caps_df.columns
    present_tickers = [ticker for ticker in tickers_list if ticker in stock_market_caps_df.columns]
    missing_fraction = (len(tickers_list) - len(present_tickers)) / len(tickers_list)
    logger.info(f"Fraction of tickers missing from stock_market_caps_df: {missing_fraction:.2%}")

    eligible_stocks = [
        stock for stock in stock_prices_df.columns
        if (
                stock in stock_market_caps_df.columns and
                stock in tickers_list and
                stock_prices_df.loc[trading_date_ts, stock] is not None and
                stock_prices_df[stock].loc[:trading_date_ts].tail(rolling_window_days).notna().all()
        )
    ]

    # From these available stocks, get the portfolio_size largest based on market caps for the current date
    if trading_date_ts in stock_market_caps_df.index:
        daily_market_caps = stock_market_caps_df.loc[trading_date_ts, eligible_stocks].dropna()
        k_stock_market_caps_df = daily_market_caps.nlargest(portfolio_size)
        return k_stock_market_caps_df
    else:
        logger.error(f"The trading date {trading_date_ts} does not exist in the market capitalizations data.")
        raise ValueError(f"The trading date {trading_date_ts} does not exist in the market capitalizations data.")


def calculate_equally_weighted_portfolio(portfolio_spec,
                                         k_stock_prices_df):
    # Logging the calculation step
    logger.info(f"Calculating equally weighted portfolio")

    # Determine the number of stocks in the portfolio
    num_stocks = portfolio_spec["portfolio_size"]

    # Assign equal weight to each stock and create the resulting dataframe
    portfolio_comp_df = pd.DataFrame({
        'Weight': [1 / num_stocks] * num_stocks
    }, index=k_stock_prices_df.columns)

    # Rename the index to 'Stock'
    portfolio_comp_df.index.name = 'Stock'

    return portfolio_comp_df

def calculate_value_weighted_portfolio(portfolio_spec, 
                                       trading_date_ts, 
                                       k_stock_market_caps_df):
    logger.info(f"Calculating market cap portfolio weights.")
    k_stock_market_caps_series = k_stock_market_caps_df.iloc[-1].sort_values(ascending=False)

    # Total market cap of the k largest stocks
    total_market_cap = k_stock_market_caps_series.sum()

    # Calculate value weights
    portfolio_comp_df = pd.DataFrame(k_stock_market_caps_series / total_market_cap)

    # Fix labels
    portfolio_comp_df.index.name = 'Stock'
    portfolio_comp_df.columns = ['Weight']

    return portfolio_comp_df

def calculate_black_litterman_portfolio(portfolio_spec,
                                        trading_date_ts,
                                        k_stock_market_caps_df,
                                        k_stock_prices_df,
                                        sp500_prices_df,
                                        treasury_bill_rate_df):
    logger.info(f"Calculating Black-Litterman portfolio weights.")
    k_stock_prices_frequency_and_window_adjusted_df = daily_prices_to_rebalancing_frequency_and_window(
        portfolio_spec,
        trading_date_ts,
        k_stock_prices_df)
    k_stock_excess_log_returns_frequency_and_window_adjusted = calculate_excess_returns_from_prices(
        portfolio_spec, k_stock_prices_frequency_and_window_adjusted_df, treasury_bill_rate_df, True)

    k_stock_market_caps_latest_df = k_stock_market_caps_df.iloc[-1].sort_values(ascending=False)

    # Covariance matrix
    covariance_log_returns_df = risk_models.CovarianceShrinkage(
        k_stock_excess_log_returns_frequency_and_window_adjusted,
        returns_data=True,
        frequency = time_periods_per_year(portfolio_spec)).ledoit_wolf()

    viewdict = {}
    market_prior_excess = black_litterman.market_implied_prior_returns(k_stock_market_caps_latest_df.squeeze(),
                                                                portfolio_spec["risk_aversion"],
                                                                covariance_log_returns_df,
                                                                risk_free_rate = 0)

    bl = BlackLittermanModel(covariance_log_returns_df, pi=market_prior_excess, absolute_views=viewdict)
    bl_mean_log_returns_series = bl.bl_returns()
    bl_covariance_log_returns_df = bl.bl_cov()

    # Add risk-free asset
    bl_mean_log_returns_with_risk_free_asset_series = bl_mean_log_returns_series.copy()
    bl_mean_log_returns_with_risk_free_asset_series["RISK_FREE"] = 0

    bl_covariance_log_returns_with_risk_free_asset_df = bl_covariance_log_returns_df.copy()
    bl_covariance_log_returns_with_risk_free_asset_df["RISK_FREE"] = 0
    bl_covariance_log_returns_with_risk_free_asset_df.loc["RISK_FREE"] = 0

    ef = EfficientFrontier(bl_mean_log_returns_with_risk_free_asset_series, bl_covariance_log_returns_with_risk_free_asset_df, weight_bounds=(0, 1))
    raw_portfolio_comp = ef.max_quadratic_utility(risk_aversion=portfolio_spec["risk_aversion"])

    # Convert cleaned weights to DataFrame
    portfolio_comp_df = pd.DataFrame(list(ef.clean_weights().items()), columns=['Stock', 'Weight'])
    portfolio_comp_df.set_index('Stock', inplace=True)
    portfolio_comp_df = portfolio_comp_df.drop("RISK_FREE")

    return portfolio_comp_df

def calculate_log_normal_portfolio(portfolio_spec,
                                  trading_date_ts,
                                  k_stock_market_caps_df,
                                  k_stock_prices_df,
                                  treasury_bill_rate_df):
    logger.info(f"Calculating portfolio weights using log normal prior.")

    # Prior weights
    prior_weights = portfolio_spec["prior_weights"]
    prior_w_df = calculate_prior_w(portfolio_spec,
                                trading_date_ts,
                                k_stock_prices_df,
                                k_stock_market_caps_df,
                                treasury_bill_rate_df)
    prior_w = prior_w_df.squeeze().values
    # Portfolio size
    p = portfolio_spec["portfolio_size"]

    # Number of observations
    n = calculate_rolling_window_frequency_adjusted(portfolio_spec)

    # Scale
    scale = portfolio_spec["scale"]

    # Observed
    # Adjust the stock prices DataFrame based on the portfolio's rebalancing frequency and rolling window
    k_stock_prices_frequency_and_window_adjusted_df = daily_prices_to_rebalancing_frequency_and_window(
        portfolio_spec, trading_date_ts, k_stock_prices_df
    )

    # Calculate the excess log returns for the adjusted stock prices
    k_stock_excess_log_returns_frequency_and_window_adjusted = calculate_excess_returns_from_prices(
        portfolio_spec, k_stock_prices_frequency_and_window_adjusted_df, treasury_bill_rate_df, True)

    # Convert trading_date_ts to string format
    formatted_date = trading_date_ts.strftime('%Y-%m-%d')

    mcmc_trace_path = os.path.join(mcmc_results, f"trace_log_normal_{scale}_{prior_weights}_{formatted_date}_{p}.nc")
    mcmc_stocks_path = os.path.join(mcmc_results, f"stocks_log_normal_{prior_weights}_{formatted_date}_{p}.csv")

    if os.path.exists(mcmc_stocks_path):
        stocks_df = pd.read_csv(mcmc_stocks_path)
        if list(stocks_df['Stocks']) == list(prior_w_df.index):
            stocks_mathcing = True
        else:
            logger.info(f"Stocks are not matching previous MCMC calculations.")
            stocks_mathcing = False
        if "GOOG" in stocks_df:
            raise ValueError(f"GOOG should not be included.")
    else:
        stocks_mathcing = False
    if os.path.exists(mcmc_trace_path) and stocks_mathcing:
        trace = az.from_netcdf(mcmc_trace_path)
    else:
        with pm.Model() as model:
            # Cholesky decomposition for the covariance matrix
            chol, corr, sigmas = pm.LKJCholeskyCov('packed_L', n=p, eta=2, sd_dist=pm.HalfCauchy.dist(beta = 5, shape = p), shape=(p * (p + 1) // 2))
            Omega = pm.Deterministic('Omega', chol.dot(chol.T))  # Precision matrix

            # Positive weights
            log_nu = pm.MvNormal('log_nu', mu=pt.log(prior_w), cov=scale*pt.eye(len(prior_w)))
            nu = pm.Deterministic('nu', pt.exp(log_nu))

            # Likelihood for X
            # Convert from natural parameters to mean and covariance
            Sigma = pm.Deterministic('Sigma', pt.nlinalg.matrix_inverse(Omega) + pt.eye(p) * 1e-32)
            mu = pm.Deterministic('mu', pt.reshape(pt.dot(Sigma, nu), (p,)))
            observed_data = k_stock_excess_log_returns_frequency_and_window_adjusted.values
            likelihood = pm.MvNormal('obs', mu=mu, cov=Sigma, observed=observed_data)

            # Sample
            trace = pm.sample(1000, tune=1000, chains=2, target_accept=0.9)

        # Save trace
        az.to_netcdf(trace, mcmc_trace_path)

        # Save stocks
        prior_w_df.index.to_series().to_csv(mcmc_stocks_path, header=['Stocks'])

    # Posterior mean of nu
    posterior_mean_nu = np.exp(trace.posterior['log_nu'].mean(dim=('chain', 'draw')).values)

    if formatted_date == portfolio_spec["posterior_nu_plot_date"]:
        log_prior_w_df = np.log(prior_w_df)
        data = trace.posterior['log_nu']
        # Convert the data to a DataFrame for easier handling
        data_array = np.concatenate((data[0], data[1]), axis = 0)
        df = pd.DataFrame(data_array, columns = log_prior_w_df.index)

        # 1. Posterior Distribution Plots with Prior Weights Marked
        plt.figure(figsize = (12, 6))
        sns.boxplot(data = df)
        plt.xticks(rotation = 45)

        # Overlay the prior weights as horizontal lines on the boxplot
        line_width = 0.5  # set the width of the horizontal lines
        for idx, stock in enumerate(df.columns):
            y = log_prior_w_df.loc[stock, 'Weight']
            plt.hlines(y, idx - line_width / 2, idx + line_width / 2, color = 'red', linestyle = '-', linewidth = 2,
                       label = 'Value-Weighted Prior' if idx == 0 else "")

        # Ensure only one "Prior" label appears in the legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc = 'upper right', bbox_to_anchor=(1, 1.05))

        plt.suptitle("Posterior Distributions of Log Portfolio Weights", y=0.96, fontweight="bold", fontsize=14, color="black")

        plt.ylabel("Log Weight", fontweight="bold")
        plt.xlabel("")

        # Save the figure
        mcmc_posterior_path = os.path.join(mcmc_results, f"log_normal_{scale}_{prior_weights}_{formatted_date}_{p}_posterior.png")
        plt.savefig(mcmc_posterior_path, dpi = 400, bbox_inches = 'tight')
        plt.close()

    portfolio_comp_df = pd.DataFrame({
        'Weight': 1 / portfolio_spec["risk_aversion"] * posterior_mean_nu
    }, index=prior_w_df.index)

    # Rename the index to 'Stock'
    portfolio_comp_df.index.name = 'Stock'

    return portfolio_comp_df

def calculate_jorion_hyperparameter_portfolio(portfolio_spec,
                                              trading_date_ts,
                                              k_stock_prices_df,
                                              treasury_bill_rate_df):

    logger.info(f"Calculating Jorion hyperparameter portfolio.")

    # Adjust the stock prices DataFrame based on the portfolio's rebalancing frequency and rolling window
    k_stock_prices_frequency_and_window_adjusted_df = daily_prices_to_rebalancing_frequency_and_window(
        portfolio_spec, trading_date_ts, k_stock_prices_df
    )

    # Calculate the log returns for the adjusted stock prices
    k_stock_excess_log_returns_frequency_and_window_adjusted = calculate_excess_returns_from_prices(
        portfolio_spec, k_stock_prices_frequency_and_window_adjusted_df, treasury_bill_rate_df, True
    )

    # Using notations of Bayesian Portfolio Analysis (2010) by Avramov and Zhou
    N = len(k_stock_prices_df.columns)
    T = len(k_stock_excess_log_returns_frequency_and_window_adjusted)

    # Sample mean
    mu_hat_df = k_stock_excess_log_returns_frequency_and_window_adjusted.mean().to_frame()

    # Sample covariance
    V_hat_df = k_stock_excess_log_returns_frequency_and_window_adjusted.cov()

    # Shrinkage
    V_bar_df = T / (T - N - 2) * V_hat_df
    V_bar_inverse_df = pd.DataFrame(np.linalg.inv(V_bar_df.to_numpy()), index=V_bar_df.index, columns=V_bar_df.columns)
    one_N_df = pd.DataFrame(np.ones(N), index=V_bar_inverse_df.index)
    mu_hat_g = (one_N_df.T.dot(V_bar_inverse_df).dot(mu_hat_df) / one_N_df.T.dot(V_bar_inverse_df).dot(one_N_df)).values[0,0]

    mu_hat_difference = mu_hat_df.sub(mu_hat_g * one_N_df.values, axis=0)
    lambda_hat = (N + 2) / (mu_hat_difference.T.dot(V_bar_inverse_df).dot(mu_hat_difference)).values[0, 0]

    v_hat = (N + 2) / ((N + 2) + T * mu_hat_difference.T.dot(V_bar_inverse_df).dot(mu_hat_difference)).values[0, 0]
    V_hat_PJ_df = (1 + 1 / (T + lambda_hat)) * V_bar_df + lambda_hat / (T * (T + 1 + lambda_hat)) * one_N_df.dot(one_N_df.T) / (one_N_df.T.dot(V_bar_inverse_df).dot(one_N_df)).values[0, 0]
    mu_hat_PJ_df = (1 - v_hat) * mu_hat_df + v_hat * mu_hat_g * one_N_df

    # Add risk free asset
    mu_hat_PJ_with_risk_free_asset_series = mu_hat_PJ_df.squeeze().copy()
    mu_hat_PJ_with_risk_free_asset_series["RISK_FREE"] = 0

    V_hat_PJ_with_risk_free_asset_df = V_hat_PJ_df.copy()
    V_hat_PJ_with_risk_free_asset_df["RISK_FREE"] = 0
    V_hat_PJ_with_risk_free_asset_df.loc["RISK_FREE"] = 0

    ef = EfficientFrontier(mu_hat_PJ_with_risk_free_asset_series, V_hat_PJ_with_risk_free_asset_df, weight_bounds=(0, 1))
    raw_portfolio_comp = ef.max_quadratic_utility(risk_aversion=portfolio_spec["risk_aversion"])

    # Convert cleaned weights to DataFrame
    portfolio_comp_df = pd.DataFrame(list(ef.clean_weights().items()), columns=['Stock', 'Weight'])
    portfolio_comp_df.set_index('Stock', inplace=True)
    portfolio_comp_df = portfolio_comp_df.drop("RISK_FREE")

    return portfolio_comp_df

def calculate_shrinkage_portfolio(portfolio_spec,
                                  trading_date_ts,
                                  k_stock_prices_df,
                                  treasury_bill_rate_df):

        logger.info(f"Calculating shrinkage portfolio weights.")

        # Adjust the stock prices DataFrame based on the portfolio's rebalancing frequency and rolling window
        k_stock_prices_frequency_and_window_adjusted_df = daily_prices_to_rebalancing_frequency_and_window(
            portfolio_spec, trading_date_ts, k_stock_prices_df
        )

        # Calculate excess returns
        k_stock_excess_log_returns_frequency_and_window_adjusted = calculate_excess_returns_from_prices(
            portfolio_spec, k_stock_prices_frequency_and_window_adjusted_df, treasury_bill_rate_df, True)

        # Mean return
        shrinkage_mean_log_returns_series = expected_returns.mean_historical_return(k_stock_excess_log_returns_frequency_and_window_adjusted,
                                                                                      returns_data=True,
                                                                                      compounding = False,
                                                                                      frequency=time_periods_per_year(portfolio_spec))

        # Covariance matrix
        shrinkage_covariance_log_returns_df = risk_models.CovarianceShrinkage(k_stock_excess_log_returns_frequency_and_window_adjusted,
                                                                                 returns_data=True,
                                                                                frequency=time_periods_per_year(
                                                                                    portfolio_spec)).ledoit_wolf()

        # Add risk free asset
        shrinkage_mean_log_returns_with_risk_free_asset_series = shrinkage_mean_log_returns_series.copy()
        shrinkage_mean_log_returns_with_risk_free_asset_series["RISK_FREE"] = 0

        shrinkage_covariance_log_returns_with_risk_free_asset_df = shrinkage_covariance_log_returns_df.copy()
        shrinkage_covariance_log_returns_with_risk_free_asset_df["RISK_FREE"] = 0
        shrinkage_covariance_log_returns_with_risk_free_asset_df.loc["RISK_FREE"] = 0

        ef = EfficientFrontier(shrinkage_mean_log_returns_with_risk_free_asset_series,
                               shrinkage_covariance_log_returns_with_risk_free_asset_df, weight_bounds=(0, 1))
        raw_portfolio_comp = ef.max_quadratic_utility(risk_aversion=portfolio_spec["risk_aversion"])

        # Convert cleaned weights to DataFrame
        portfolio_comp_df = pd.DataFrame(list(ef.clean_weights().items()), columns=['Stock', 'Weight'])
        portfolio_comp_df.set_index('Stock', inplace=True)
        portfolio_comp_df = portfolio_comp_df.drop("RISK_FREE")

        return portfolio_comp_df

def calculate_min_variance_portfolio(portfolio_spec,
                                      trading_date_ts,
                                      k_stock_prices_df,
                                      treasury_bill_rate_df):
    logger.info(f"Calculating min variance portfolio weights.")

    # Adjust the stock prices DataFrame based on the portfolio's rebalancing frequency and rolling window
    k_stock_prices_frequency_and_window_adjusted_df = daily_prices_to_rebalancing_frequency_and_window(
        portfolio_spec, trading_date_ts, k_stock_prices_df
    )

    # Calculate excess returns
    k_stock_excess_log_returns_frequency_and_window_adjusted = calculate_excess_returns_from_prices(
        portfolio_spec, k_stock_prices_frequency_and_window_adjusted_df, treasury_bill_rate_df, True)

    # Mean return
    min_variance_mean_log_returns_series = expected_returns.mean_historical_return(
        k_stock_excess_log_returns_frequency_and_window_adjusted,
        returns_data=True,
        compounding=False,
        frequency=time_periods_per_year(portfolio_spec))

    # Covariance matrix
    min_variance_covariance_log_returns_df = risk_models.CovarianceShrinkage(
        k_stock_excess_log_returns_frequency_and_window_adjusted,
        returns_data=True,
        frequency=time_periods_per_year(
            portfolio_spec)).ledoit_wolf()

    ef = EfficientFrontier(min_variance_mean_log_returns_series,
                           min_variance_covariance_log_returns_df, weight_bounds=(0, 1))
    raw_portfolio_comp = ef.min_volatility()

    # Convert cleaned weights to DataFrame
    portfolio_comp_df = pd.DataFrame(list(ef.clean_weights().items()), columns=['Stock', 'Weight'])
    portfolio_comp_df.set_index('Stock', inplace=True)

    return portfolio_comp_df

def calculate_portfolio_weights(trading_date_ts,
                                portfolio_spec,
                                market_data):
    # Unpack market data
    stock_market_caps_df = market_data["stock_market_caps_df"]
    stock_prices_df = market_data["stock_prices_df"]
    treasury_bill_rate_df = market_data["treasury_bill_rate_df"]
    sp500_prices_df = market_data["sp500_prices_df"]

    # Get k largest stocks and market caps at trading_date_ts
    k_stock_market_caps_trading_date_df = get_k_largest_stocks_market_caps(stock_market_caps_df,
                                                              stock_prices_df,
                                                              trading_date_ts,
                                                              portfolio_spec["portfolio_size"],
                                                              portfolio_spec["rolling_window_days"],
                                                              portfolio_spec["rebalancing_frequency"])

    # Filter all the data to only include data until current date. Very important!
    # Filter market caps
    k_stock_market_caps_df = stock_market_caps_df[k_stock_market_caps_trading_date_df.index.intersection(stock_market_caps_df.columns)]
    k_stock_market_caps_df = k_stock_market_caps_df.loc[:trading_date_ts]

    # Filter stock prices
    k_stock_prices_df = stock_prices_df[k_stock_market_caps_trading_date_df.index.intersection(stock_prices_df.columns)]
    k_stock_prices_df = k_stock_prices_df.loc[:trading_date_ts]

    # Filter S&P 500 prices
    sp500_prices_df = sp500_prices_df.loc[sp500_prices_df.index <= trading_date_ts]

    # Check for NA values in the filtered DataFrame
    if k_stock_prices_df.tail(portfolio_spec["rolling_window_days"]).isna().any().any():
        logger.error(f"Found NA values in the filtered stock prices.")
        raise ValueError(f"The filtered stock prices contain NA values.")

    if portfolio_spec["weights_spec"] == "value_weighted":
        portfolio_comp_df = calculate_value_weighted_portfolio(portfolio_spec,
                                                                trading_date_ts,
                                                                k_stock_market_caps_df)

    elif portfolio_spec["weights_spec"] == "equally_weighted":
        portfolio_comp_df = calculate_equally_weighted_portfolio(portfolio_spec,
                                                                k_stock_prices_df)

    elif portfolio_spec["weights_spec"] == "black_litterman":
        portfolio_comp_df = calculate_black_litterman_portfolio(portfolio_spec,
                                                                 trading_date_ts,
                                                                 k_stock_market_caps_df,
                                                                 k_stock_prices_df,
                                                                 sp500_prices_df,
                                                                 treasury_bill_rate_df)

    elif portfolio_spec["weights_spec"] == "log_normal":
        portfolio_comp_df = calculate_log_normal_portfolio(portfolio_spec,
                                                          trading_date_ts,
                                                          k_stock_market_caps_df,
                                                          k_stock_prices_df,
                                                          treasury_bill_rate_df)


    elif portfolio_spec["weights_spec"] == "jorion_hyper":
        portfolio_comp_df = calculate_jorion_hyperparameter_portfolio(portfolio_spec,
                                                                      trading_date_ts,
                                                                      k_stock_prices_df,
                                                                      treasury_bill_rate_df)

    elif portfolio_spec["weights_spec"] == "shrinkage":
        portfolio_comp_df = calculate_shrinkage_portfolio(portfolio_spec,
                                                          trading_date_ts,
                                                          k_stock_prices_df,
                                                          treasury_bill_rate_df)

    elif portfolio_spec["weights_spec"] == "min_variance":
        portfolio_comp_df = calculate_min_variance_portfolio(portfolio_spec,
                                                          trading_date_ts,
                                                          k_stock_prices_df,
                                                          treasury_bill_rate_df)

    else:
        logger.error(f"Unknown weights spec.")
        raise ValueError(f"Unknown weights spec.")

    return portfolio_comp_df

def compute_portfolio_turnover(portfolio_comp_before_df,
                               portfolio_comp_after_df):

    # Merging the old and new weights with a suffix to differentiate them
    portfolio_comp_before_after_df = portfolio_comp_before_df.merge(portfolio_comp_after_df,
                                                how='outer',
                                                left_index=True,
                                                right_index=True,
                                                suffixes=('_before', '_after'))

    # Fill missing values with 0s (for new stocks or those that have been removed)
    portfolio_comp_before_after_df.fillna(0, inplace=True)

    # Calculate absolute difference for each stock and then compute turnover
    portfolio_comp_before_after_df['weight_diff'] = abs(portfolio_comp_before_after_df['Weight_before'] - portfolio_comp_before_after_df['Weight_after'])

    # Calculate turnover corresponding to risk free asset
    risk_free_turnover = abs(portfolio_comp_before_df['Weight'].sum() - portfolio_comp_after_df['Weight'].sum())

    # Calculate total turnover
    turnover = (portfolio_comp_before_after_df['weight_diff'].sum() + risk_free_turnover) / 2

    return turnover

class Portfolio:

    def get_portfolio_simple_returns_series(self):
        return self.portfolio_simple_returns_series

    def get_portfolio_turnover(self):
        return self.portfolio_turnover_series

    def __init__(self,
                 ts_start_date,
                 portfolio_spec):
        self.ts_start_date = ts_start_date
        self.portfolio_spec = portfolio_spec
        self.portfolio_simple_returns_series = pd.Series(dtype = "float64", name = portfolio_spec["display_name"])
        self.portfolio_turnover_series = pd.Series(dtype = "float64", name = portfolio_spec["display_name"])
        self.last_rebalance_date_ts = None

    def update_portfolio(self,
                         trading_date_ts,
                         market_data):

        # Calculate daily portfolio return
        if self.ts_start_date != trading_date_ts:
            # Filter out stocks not in the portfolio
            filtered_stock_simple_returns_series = market_data["stock_simple_returns_df"].loc[trading_date_ts].reindex(self.portfolio_comp_df.index)

            # Multiply returns by weights element-wise and then sum to get the portfolio return
            portfolio_simple_return = (filtered_stock_simple_returns_series * self.portfolio_comp_df['Weight']).sum()

            # Add risk-free return
            treasury_bill_rate_df = market_data["treasury_bill_rate_df"]
            treasury_bill_rate_bey_df = calculate_bond_equivalent_yield_from_DTB3(treasury_bill_rate_df)
            most_recent_treasury_bill_rate_bey = treasury_bill_rate_bey_df.asof(trading_date_ts).iloc[0]
            risk_free_daily_return = ((most_recent_treasury_bill_rate_bey + 1) ** (1 / 252) - 1)
            portfolio_simple_return += (1 - self.portfolio_comp_df['Weight'].sum()) * risk_free_daily_return

            self.portfolio_simple_returns_series[trading_date_ts] = portfolio_simple_return

            # Update weight for the risk-free asset
            current_risk_free_weight = 1 - self.portfolio_comp_df['Weight'].sum()
            updated_risk_free_weight = current_risk_free_weight * (1 + risk_free_daily_return)

            # Update weights for the stocks
            self.portfolio_comp_df['Weight'] = (
                        self.portfolio_comp_df['Weight'] * (1 + filtered_stock_simple_returns_series))


            # Update the total invested value by adding the updated risk-free weight
            total_value = self.portfolio_comp_df['Weight'].sum() + updated_risk_free_weight

            # Normalize the weights so they sum up to 1
            self.portfolio_comp_df['Weight'] = self.portfolio_comp_df['Weight'] / total_value

            # Check that weights sum to 1
            if abs((self.portfolio_comp_df['Weight'].values.sum() + updated_risk_free_weight / total_value) - 1) > 1e-5:
                logger.error(f"Weights do not sum to 1.")
                raise ValueError(f"Weights do not sum to 1.")

        if self.last_rebalance_date_ts is None:
            rebalance = True
        elif self.portfolio_spec["rebalancing_frequency"] == "daily":
            rebalance = True
        elif self.portfolio_spec["rebalancing_frequency"] == "weekly":
            rebalance = trading_date_ts.weekday() == 2 or (trading_date_ts - self.last_rebalance_date_ts).days > 7
        elif self.portfolio_spec["rebalancing_frequency"] == "monthly":
            rebalance = trading_date_ts.month != self.last_rebalance_date_ts.month
        else:
            logger.error(f"Unknown rebalancing frequency.")
            raise ValueError(f"Unknown rebalancing frequency.")

        if rebalance:
            if not self.last_rebalance_date_ts is None:
                # Make a copy of the current weights to calculate turnover later
                portfolio_comp_before_df = self.portfolio_comp_df.copy()

            # Calculate the new portfolio weights
            self.portfolio_comp_df = calculate_portfolio_weights(trading_date_ts,
                                                                 self.portfolio_spec,
                                                                 market_data)

            if not self.last_rebalance_date_ts is None:
                turnover = compute_portfolio_turnover(portfolio_comp_before_df, self.portfolio_comp_df)
                self.portfolio_turnover_series[trading_date_ts] = turnover
                turnover_cost = self.portfolio_spec["turnover_cost_bps"] / 10000 * turnover
                self.portfolio_simple_returns_series[trading_date_ts] -= turnover_cost

            logger.info(f"Portfolio size {trading_date_ts}: {len(self.portfolio_comp_df.index)}")
            self.last_rebalance_date_ts = trading_date_ts

def backtest_portfolio(portfolio_spec,
                       ts_start_date,
                       ts_end_date,
                       market_data):

    # Trading dates
    trading_date_ts = [pd.Timestamp(ts) for ts in market_data["stock_prices_df"].index]
    trading_date_ts = [ts for ts in trading_date_ts if ts_start_date <= ts <= ts_end_date]

    portfolio = Portfolio(trading_date_ts[0], portfolio_spec)

    for trading_date_ts in trading_date_ts:
        portfolio.update_portfolio(trading_date_ts,
                                   market_data)

    return {"portfolio_simple_returns_series": portfolio.get_portfolio_simple_returns_series(),
            "portfolio_turnover_series": portfolio.get_portfolio_turnover()}
