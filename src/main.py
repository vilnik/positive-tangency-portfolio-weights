import os
import pandas as pd
import data_handling
import portfolio_evaluation as evaluation
from dotenv import load_dotenv
import logging
import portfolio_calculations
from portfolio_specs import EVAL_PORTFOLIO_SPECS

load_dotenv()
logging_level = os.environ.get("LOGGING_LEVEL", logging.INFO)
logging.basicConfig(level=logging_level)
logger = logging.getLogger(__name__)

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the script's directory
parent_dir = os.path.dirname(script_dir)

# Construct the path to the results directory
results_dir = os.path.join(parent_dir, 'results')

os.makedirs(results_dir, exist_ok=True)

def main():
    str_start_date = "2006-01-01"
    str_end_date = "2023-06-30"
    ts_start_date = pd.Timestamp(str_start_date)
    ts_end_date = pd.Timestamp(str_end_date)

    # Get market data
    market_data = data_handling.get_market_data()
    portfolio_setups_simple_returns = {}
    portfolio_setups_turnover = {}

    # Backtest portfolios
    for portfolio_spec_name, portfolio_spec in EVAL_PORTFOLIO_SPECS.items():
        simple_returns_results_file = os.path.join(results_dir, f'{portfolio_spec_name}_simple_returns_{str_start_date}_{str_end_date}.csv')
        turnover_returns_file = os.path.join(results_dir, f'{portfolio_spec_name}_turnover_{str_start_date}_{str_end_date}.csv')

        if os.path.exists(simple_returns_results_file) and \
            os.path.exists(turnover_returns_file):
            portfolio_setups_simple_returns[portfolio_spec_name] = pd.read_csv(simple_returns_results_file,
                                                                    index_col=0,
                                                                    parse_dates=True)
            portfolio_setups_turnover[portfolio_spec_name] = pd.read_csv(turnover_returns_file,
                                                                    index_col=0,
                                                                    parse_dates=True)
        else:
            portfolio_performance = portfolio_calculations.backtest_portfolio(portfolio_spec,
                                                                               ts_start_date,
                                                                               ts_end_date,
                                                                               market_data)

            # Store the results
            portfolio_setups_simple_returns[portfolio_spec_name] = portfolio_performance["portfolio_simple_returns_series"]
            portfolio_setups_turnover[portfolio_spec_name] = portfolio_performance["portfolio_turnover_series"]

            # Save to CSV
            portfolio_performance["portfolio_simple_returns_series"].to_csv(simple_returns_results_file, header=True)
            portfolio_performance["portfolio_turnover_series"].to_csv(turnover_returns_file, header=True)

    # Simple returns
    portfolio_setups_simple_returns = pd.concat(list(portfolio_setups_simple_returns.values()), axis=1)

    # Turnovers
    portfolio_setups_turnover = pd.concat(list(portfolio_setups_turnover.values()), axis=1)

    # Portfolio metrics
    evaluation.full_evaluation(portfolio_setups_simple_returns,
                            portfolio_setups_turnover,
                            market_data["treasury_bill_rate_df"])

if __name__ == "__main__":
    main()