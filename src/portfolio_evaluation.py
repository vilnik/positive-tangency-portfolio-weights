import os
import pandas as pd
import quantstats as qs
from dotenv import load_dotenv
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

load_dotenv()
logging_level = os.environ.get("LOGGING_LEVEL", logging.INFO)
logging.basicConfig(level=logging_level)
logger = logging.getLogger(__name__)

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the script's directory
parent_dir = os.path.dirname(script_dir)

def format_pct_axis(x, pos):
    return f"{x * 100:.0f}%"

def plot_performance(portfolio_setups_simple_returns_df,
                     portfolio_setups_excess_simple_returns_df):

    # Plot daily returns
    qs.plots.returns(portfolio_setups_simple_returns_df,
                           savefig = f"../results/returns")

    # Plot yearly returns
    qs.plots.yearly_returns(portfolio_setups_simple_returns_df,
                           savefig = f"../results/yearly_returns")

    # Plot rolling Sharpe
    qs.plots.rolling_sharpe(portfolio_setups_excess_simple_returns_df,
                           savefig = f"../results/rolling_sharpe")

    # Plot rolling Sharpe
    qs.plots.rolling_sortino(portfolio_setups_excess_simple_returns_df,
                           savefig = f"../results/rolling_sortino")

    # Plot rolling volatility
    qs.plots.rolling_volatility(portfolio_setups_simple_returns_df,
                           savefig = f"../results/rolling_volatility")

    # Plot drawdown
    qs.plots.drawdown(portfolio_setups_simple_returns_df,
                      savefig=f"../results/drawdown")

def process_and_highlight_values(metrics_df):
    # Set of metrics that are better when higher
    higher_is_better = {
        'Cum. return', 'CAGR', 'Sharpe Ratio',
        'Sortino Ratio', 'Calmar Ratio', 'Max. DD',
        'Avg. Loss', 'Avg. Return', 'Avg. Win', 'Best Day',
        'Worst Day', 'Daily VaR'
    }

    # Set of metrics that are better when lower
    lower_is_better = {
        'Ann. Vol.', 'Avg. Turnover'
    }

    # Set of metrics that should not be converted to percentages
    not_percentage = {'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'}

    for row_label, row_series in metrics_df.iterrows():
        if row_label not in higher_is_better and row_label not in lower_is_better:
            raise ValueError(f"Unexpected row label: {row_label}")

        # Convert and round the values if necessary before finding max/min
        processed_values = [round(100 * val, 3) if row_label not in not_percentage else round(val, 3) for val in
                            row_series]
        max_value = max(processed_values)
        min_value = min(processed_values)

        new_values = []
        for proc_val in processed_values:
            str_val = f"{proc_val:.3f}%" if row_label not in not_percentage else f"{proc_val:.3f}"
            if row_label in higher_is_better:
                if proc_val == max_value:
                    new_values.append(f"{str_val} (Best)")
                elif proc_val == min_value:
                    new_values.append(f"{str_val} (Worst)")
                else:
                    new_values.append(str_val)
            elif row_label in lower_is_better:
                if proc_val == min_value:
                    new_values.append(f"{str_val} (Best)")
                elif proc_val == max_value:
                    new_values.append(f"{str_val} (Worst)")
                else:
                    new_values.append(str_val)

        metrics_df.loc[row_label] = new_values

    return metrics_df


def performance_metrics(portfolio_setups_simple_returns_df,
                         portfolio_setups_excess_simple_returns_df,
                        portfolio_setups_turnover):
    # Df to store portfolio metrics
    portfolio_setups_metrics_df = pd.DataFrame(columns=portfolio_setups_simple_returns_df.columns)

    # Calculate the cumulative return for each portfolio and add it as a row
    for column_name in portfolio_setups_simple_returns_df.columns:
        portfolio_simple_returns_series = portfolio_setups_simple_returns_df[column_name]

        portfolio_comp_return = qs.stats.comp(portfolio_simple_returns_series)
        portfolio_setups_metrics_df.at['Cum. return', column_name] = portfolio_comp_return

    # Calculate the CAGR for each portfolio and add it as a row
    for column_name in portfolio_setups_simple_returns_df.columns:
        portfolio_simple_returns_series = portfolio_setups_simple_returns_df[column_name]

        portfolio_cagr = qs.stats.cagr(portfolio_simple_returns_series)
        portfolio_setups_metrics_df.at['CAGR', column_name] = portfolio_cagr

    # Calculate the Sharpe ratio for each portfolio and add it as a row
    for column_name in portfolio_setups_excess_simple_returns_df.columns:
        portfolio_excess_simple_returns_series = portfolio_setups_excess_simple_returns_df[column_name]

        portfolio_sharpe_ratio = qs.stats.sharpe(portfolio_excess_simple_returns_series)
        portfolio_setups_metrics_df.at['Sharpe Ratio', column_name] = portfolio_sharpe_ratio

    # Calculate the Sortino ratio for each portfolio and add it as a row
    for column_name in portfolio_setups_excess_simple_returns_df.columns:
        portfolio_excess_simple_returns_series = portfolio_setups_excess_simple_returns_df[column_name]

        portfolio_sortino_ratio = qs.stats.sortino(portfolio_excess_simple_returns_series)
        portfolio_setups_metrics_df.at['Sortino Ratio', column_name] = portfolio_sortino_ratio

    # Calculate the Calmar ratio for each portfolio and add it as a row
    for column_name in portfolio_setups_excess_simple_returns_df.columns:
        portfolio_excess_simple_returns_series = portfolio_setups_excess_simple_returns_df[column_name]

        portfolio_calmar_ratio = qs.stats.calmar(portfolio_excess_simple_returns_series)
        portfolio_setups_metrics_df.at['Calmar Ratio', column_name] = portfolio_calmar_ratio

    # Calculate the max drawdown for each portfolio and add it as a row
    for column_name in portfolio_setups_simple_returns_df.columns:
        portfolio_simple_returns_series = portfolio_setups_simple_returns_df[column_name]

        portfolio_max_drawdown = qs.stats.max_drawdown(portfolio_simple_returns_series)
        portfolio_setups_metrics_df.at['Max. DD', column_name] = portfolio_max_drawdown

    # Calculate the avg loss for each portfolio and add it as a row
    for column_name in portfolio_setups_simple_returns_df.columns:
        portfolio_simple_returns_series = portfolio_setups_simple_returns_df[column_name]

        portfolio_avg_loss = qs.stats.avg_loss(portfolio_simple_returns_series)
        portfolio_setups_metrics_df.at['Avg. Loss', column_name] = portfolio_avg_loss

    # Calculate the avg return for each portfolio and add it as a row
    for column_name in portfolio_setups_simple_returns_df.columns:
        portfolio_simple_returns_series = portfolio_setups_simple_returns_df[column_name]

        portfolio_avg_return = qs.stats.avg_return(portfolio_simple_returns_series)
        portfolio_setups_metrics_df.at['Avg. Return', column_name] = portfolio_avg_return

    # Calculate the avg win for each portfolio and add it as a row
    for column_name in portfolio_setups_simple_returns_df.columns:
        portfolio_simple_returns_series = portfolio_setups_simple_returns_df[column_name]

        portfolio_avg_win = qs.stats.avg_win(portfolio_simple_returns_series)
        portfolio_setups_metrics_df.at['Avg. Win', column_name] = portfolio_avg_win

    # Calculate the best day for each portfolio and add it as a row
    for column_name in portfolio_setups_simple_returns_df.columns:
        portfolio_simple_returns_series = portfolio_setups_simple_returns_df[column_name]

        portfolio_best = qs.stats.best(portfolio_simple_returns_series)
        portfolio_setups_metrics_df.at['Best Day', column_name] = portfolio_best

    # Calculate the worst day for each portfolio and add it as a row
    for column_name in portfolio_setups_simple_returns_df.columns:
        portfolio_simple_returns_series = portfolio_setups_simple_returns_df[column_name]

        portfolio_worst = qs.stats.worst(portfolio_simple_returns_series)
        portfolio_setups_metrics_df.at['Worst Day', column_name] = portfolio_worst

    # Calculate the volatility for each portfolio and add it as a row
    for column_name in portfolio_setups_simple_returns_df.columns:
        portfolio_simple_returns_series = portfolio_setups_simple_returns_df[column_name]

        portfolio_volatility = qs.stats.volatility(portfolio_simple_returns_series)
        portfolio_setups_metrics_df.at['Ann. Vol.', column_name] = portfolio_volatility

    # Calculate the VaR for each portfolio and add it as a row
    for column_name in portfolio_setups_simple_returns_df.columns:
        portfolio_simple_returns_series = portfolio_setups_simple_returns_df[column_name]

        portfolio_VaR = qs.stats.value_at_risk(portfolio_simple_returns_series)
        portfolio_setups_metrics_df.at['Daily VaR', column_name] = portfolio_VaR

    # Calculate the turnover for each portfolio and add it as a row
    for column_name in portfolio_setups_turnover.columns:
        portfolio_turnover_series = portfolio_setups_turnover[column_name]
        portfolio_turnover = portfolio_turnover_series.mean()
        portfolio_setups_metrics_df.at['Avg. Turnover', column_name] = portfolio_turnover

    highlighted_metrics_df = process_and_highlight_values(portfolio_setups_metrics_df)
    highlighted_metrics_df.to_csv(f"../results/metrics.csv", index=True)

def compute_excess_returns(portfolio_simple_returns_series,
                           treasury_bill_rate_df):

    days_to_maturity = 90
    ann_ytm = (treasury_bill_rate_df * days_to_maturity / 360) / (1 - treasury_bill_rate_df * days_to_maturity / 360) * 365 / days_to_maturity
    treasury_bill_rate_bey_df = (1 + ann_ytm / 2)**2 - 1

    # 2. Comparison with Original Data
    differences = (treasury_bill_rate_bey_df['DTB3'] - treasury_bill_rate_df['DTB3']).abs()
    if differences.mean() > 0.01:
        logger.error("Average difference between BEY and original data is too large. Please inspect the results.")
        raise ValueError("Average difference between BEY and original data is too large. Please inspect the results.")

    # Reindex treasury_bill_rate_df to match the dates of portfolio_simple_returns_series, then forward-fill
    matching_treasury_bill_bey_rates_series = treasury_bill_rate_bey_df['DTB3'].reindex(portfolio_simple_returns_series.index).ffill()

    # For any NaNs at the beginning, use backfill to fill them with the next valid value
    matching_treasury_bill_bey_rates_series = matching_treasury_bill_bey_rates_series.bfill()

    # Compute the excess returns
    portfolio_excess_simple_returns_series = portfolio_simple_returns_series - (
            (matching_treasury_bill_bey_rates_series + 1) ** (1 / 252) - 1)

    # Rename the series to match the original portfolio name
    portfolio_excess_simple_returns_series.name = portfolio_simple_returns_series.name

    return portfolio_excess_simple_returns_series

def full_evaluation(portfolio_setups_simple_returns_df,
                    portfolio_setups_turnover,
                    treasury_bill_rate_df):

    # Get excess returns
    portfolio_setups_excess_simple_returns_df = pd.DataFrame(columns=portfolio_setups_simple_returns_df.columns)
    for column_name in portfolio_setups_simple_returns_df.columns:
        portfolio_simple_returns_series = portfolio_setups_simple_returns_df[column_name]

        portfolio_excess_simple_returns_series = compute_excess_returns(portfolio_simple_returns_series,
                                                                        treasury_bill_rate_df)

        # Add portfolio_excess_simple_returns_series to the new DataFrame
        portfolio_setups_excess_simple_returns_df[column_name] = portfolio_excess_simple_returns_series

    # Performance metrics
    performance_metrics(portfolio_setups_simple_returns_df,
                     portfolio_setups_excess_simple_returns_df,
                    portfolio_setups_turnover)

    # Plot performance
    plot_performance(portfolio_setups_simple_returns_df,
                     portfolio_setups_excess_simple_returns_df)