# Constructing Bayesian tangency portfolios under short-selling restrictions

This repository contains code related to the research project 'Constructing Bayesian Tangency Portfolios under Short-Selling Restrictions'. 

## Research Overview
The associated paper addresses the challenge of constructing tangency portfolios in the context of short-selling restrictions in equity markets. Utilizing Bayesian techniques, we reparameterize the asset return model, enabling direct determination of priors for the tangency portfolio weights. This facilitates the integration of non-negative weight constraints into an investor's prior beliefs, resulting in a posterior distribution focused exclusively on nonnegative values. Portfolio weight estimators are subsequently derived via the Markov Chain Monte Carlo (MCMC) methodology. Our novel Bayesian approach is empirically illustrated using the largest stocks by market capitalization in the S\&P 500 index. By effectively combining prior beliefs and historical data, this method showcases promising results in terms of risk-adjusted returns and interpretability. 

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **API Keys:**
  - **Alpha Vantage:** Obtain your API key by signing up [here](https://www.alphavantage.co/).
  - **Financial Modeling Prep:** Sign up [here](https://financialmodelingprep.com/developer/docs/) to get your API key.

### Manual Data Collection
In addition to the data automatically gathered through the APIs, there is some data you'll need to collect and place manually:

- **S&P 500 Historical Components:**
  - Download `S&P 500 Historical Components & Changes(08-01-2023).csv` from [this repository](https://github.com/fja05680/sp500).
  - Place the downloaded file in `data/sp500_components`.
- **Treasury Bill Rate (DTB3):**
  - Download the DTB3 data from [FRED](https://fred.stlouisfed.org/series/DTB3).
  - Save the data as `DTB3.csv` and place it in `data/treasury_bill_rate`.

## Setup

### Environment Variables

1. Create a `.env` file in your project directory.
2. Insert your API keys into the `.env` file as shown below:

    ```
    ALPHA_VANTAGE_KEY=your_alpha_vantage_api_key
    FINANCIAL_MODELING_PREP_KEY=your_financial_modeling_prep_api_key
    ```

    Replace `your_alpha_vantage_api_key` and `your_financial_modeling_prep_api_key` with your actual keys.

### Dependencies Installation

To install dependencies, navigate to the project directory in your terminal and run:

    ```shell
    pip3 install -r requirements.txt
    ```

## Running the Project

After setting up your API keys, downloading the necessary data, and installing dependencies, you can execute the project. Running the main script will download relevant data, perform portfolio backtesting for the period between January 2006, and June 2023 (as discussed in the paper), save the results, generate performance metrics, and create visualizations to aid in understanding the portfolio's performance over time. Navigate to the project directory in your terminal and execute the main script:

    ```shell
    python3 main.py
    ```
