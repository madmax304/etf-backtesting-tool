import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
import streamlit as st
import logging
import time

logger = logging.getLogger(__name__)

def validate_tickers(tickers):
    """
    Validate if the provided tickers exist on Yahoo Finance
    Returns tuple of (valid_tickers, invalid_tickers)
    """
    valid_tickers = []
    invalid_tickers = []
    
    for ticker in tickers:
        try:
            # Add delay between requests
            time.sleep(0.1)  # 100ms delay
            logger.info(f"Validating ticker: {ticker}")
            etf = yf.Ticker(ticker)
            # Try to fetch info to validate ticker
            if etf.info:
                valid_tickers.append(ticker)
                logger.info(f"Successfully validated {ticker}")
            else:
                invalid_tickers.append(ticker)
                logger.warning(f"Invalid ticker: {ticker} - no info available")
        except Exception as e:
            if "rate limit" in str(e).lower():
                st.error("Rate limit reached. Please try again in a few minutes.")
                return [], tickers
            invalid_tickers.append(ticker)
            logger.error(f"Error validating {ticker}: {str(e)}", exc_info=True)
    
    return valid_tickers, invalid_tickers

def calculate_cagr(start_price, end_price, years):
    """Calculate the Compound Annual Growth Rate"""
    try:
        if years == 0:
            return 0
        return (((end_price / start_price) ** (1 / years)) - 1) * 100
    except Exception as e:
        logger.error(f"Error calculating CAGR: {str(e)}", exc_info=True)
        return 0

def convert_date_for_yfinance(date_obj):
    """
    Convert date/datetime objects to datetime format required by yfinance
    Ensures consistent date handling throughout the application
    """
    if isinstance(date_obj, datetime):
        return date_obj
    return datetime.combine(date_obj, datetime.min.time())

@st.cache_data(ttl=3600, show_spinner=False)
def calculate_metrics(tickers, start_date, end_date, initial_investment, allocations, rebalance_frequency="None"):
    """
    Calculate performance metrics for the given tickers with optional rebalancing
    
    Parameters:
    tickers (list): List of ticker symbols
    start_date (datetime): Start date
    end_date (datetime): End date
    initial_investment (float): Initial investment amount
    allocations (dict): Dictionary of ticker:allocation_percentage pairs
    rebalance_frequency (str): None, Monthly, Quarterly, or Annually
    """
    # Create a cache key based on all inputs
    cache_key = f"{tickers}-{start_date}-{end_date}-{initial_investment}-{allocations}-{rebalance_frequency}"
    st.session_state['last_calculation'] = cache_key
    
    # Convert dates to proper format
    start_date = convert_date_for_yfinance(start_date)
    end_date = convert_date_for_yfinance(end_date)
    
    logger.info(f"Calculating metrics for {len(tickers)} tickers with {rebalance_frequency} rebalancing")
    results = []
    portfolio_data = pd.DataFrame()
    
    # Get data for all tickers first
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(data) > 0:
                # Check if start date is before first available data
                first_date = data.index[0].date()
                if first_date > start_date.date():
                    st.warning(f"{ticker} data only available from {first_date}. Using this as start date for {ticker}.")
                portfolio_data[ticker] = data['Close']
            else:
                st.error(f"No data available for {ticker} in the selected date range.")
                continue
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {str(e)}")
            continue
    
    if portfolio_data.empty:
        return pd.DataFrame()
    
    # Initialize portfolio
    shares = {}
    for ticker in portfolio_data.columns:
        initial_investment_per_ticker = initial_investment * (allocations[ticker] / 100)
        shares[ticker] = initial_investment_per_ticker / portfolio_data[ticker].iloc[0]
    
    # Calculate daily portfolio value
    portfolio_value = pd.Series(0.0, index=portfolio_data.index)
    
    # Function to rebalance portfolio
    def rebalance(date):
        nonlocal shares
        current_value = sum(shares[ticker] * portfolio_data[ticker][date] for ticker in shares)
        for ticker in shares:
            target_value = current_value * (allocations[ticker] / 100)
            shares[ticker] = target_value / portfolio_data[ticker][date]
    
    # Process portfolio value with rebalancing
    last_rebalance = portfolio_data.index[0]
    
    for date in portfolio_data.index:
        # Check if rebalancing is needed
        if rebalance_frequency != "None":
            rebalance_needed = False
            
            if rebalance_frequency == "Monthly":
                rebalance_needed = date.month != last_rebalance.month
            elif rebalance_frequency == "Quarterly":
                rebalance_needed = (date.month - 1) // 3 != (last_rebalance.month - 1) // 3
            elif rebalance_frequency == "Annually":
                rebalance_needed = date.year != last_rebalance.year
            
            if rebalance_needed:
                rebalance(date)
                last_rebalance = date
        
        # Calculate portfolio value for this day
        portfolio_value[date] = sum(shares[ticker] * portfolio_data[ticker][date] for ticker in shares)
    
    # Calculate final metrics for each ticker
    for ticker in portfolio_data.columns:
        initial_price = portfolio_data[ticker].iloc[0]
        current_price = portfolio_data[ticker].iloc[-1]
        current_shares = shares[ticker]
        ticker_initial_investment = initial_investment * (allocations[ticker] / 100)
        current_value = current_shares * current_price
        total_return = ((current_value / ticker_initial_investment) - 1) * 100
        
        # Calculate years for CAGR
        years = (portfolio_data.index[-1] - portfolio_data.index[0]).days / 365.25
        cagr = calculate_cagr(ticker_initial_investment, current_value, years)
        
        results.append({
            'Ticker': ticker,
            'Allocation (%)': round(allocations[ticker], 1),
            'Shares': round(current_shares, 4),
            'Initial Investment': round(ticker_initial_investment, 2),
            'Current Value': round(current_value, 2),
            'Initial Price': round(initial_price, 2),
            'Current Price': round(current_price, 2),
            'Total Return (%)': round(total_return, 2),
            'CAGR (%)': round(cagr, 2)
        })
    
    return pd.DataFrame(results)

def calculate_risk_metrics(data):
    """Calculate risk metrics like Sharpe ratio, volatility, etc."""
    returns = data.pct_change()
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))  # Assuming risk-free rate of 0
    return volatility, sharpe_ratio 