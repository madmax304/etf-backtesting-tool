import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
import streamlit as st
import logging

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

def calculate_metrics(tickers, start_date, initial_investment, allocations):
    """
    Calculate performance metrics for the given tickers
    
    Parameters:
    tickers (list): List of ticker symbols
    start_date (datetime): Purchase date
    initial_investment (float): Initial investment amount
    allocations (dict): Dictionary of ticker:allocation_percentage pairs
    """
    logger.info(f"Calculating metrics for {len(tickers)} tickers")
    results = []
    
    for ticker in tickers:
        try:
            logger.info(f"Processing {ticker}")
            # Calculate investment amount for this ticker
            ticker_investment = initial_investment * (allocations[ticker] / 100)
            
            # Fetch historical data
            data = yf.download(ticker, start=start_date, progress=False)
            
            # Debug information
            logger.info(f"Downloaded data for {ticker}:")
            logger.info(f"Columns: {data.columns.tolist()}")
            logger.info(f"Data shape: {data.shape}")
            
            if len(data) == 0:
                logger.warning(f"No data available for {ticker}")
                st.warning(f"No data available for {ticker} from the selected date.")
                continue
            
            # Use 'Close' instead of 'Adj Close' since that's what we're getting
            close_col = ('Close', ticker) if ('Close', ticker) in data.columns else 'Close'
            
            if close_col not in data.columns:
                logger.error(f"Missing Close price data for {ticker}")
                column_names = [str(col) for col in data.columns]
                st.error(f"Price data not available for {ticker}. Available columns: {', '.join(column_names)}")
                continue
            
            initial_price = data[close_col].iloc[0]
            current_price = data[close_col].iloc[-1]
            shares = ticker_investment / initial_price
            current_value = shares * current_price
            total_return = ((current_price / initial_price) - 1) * 100
            
            # Calculate years for CAGR
            years = (data.index[-1] - data.index[0]).days / 365.25
            cagr = calculate_cagr(initial_price, current_price, years)
            
            results.append({
                'Ticker': ticker,
                'Allocation (%)': round(allocations[ticker], 1),
                'Shares': round(shares, 4),
                'Initial Investment': round(ticker_investment, 2),
                'Current Value': round(current_value, 2),
                'Initial Price': round(initial_price, 2),
                'Current Price': round(current_price, 2),
                'Total Return (%)': round(total_return, 2),
                'CAGR (%)': round(cagr, 2)
            })
            
            logger.info(f"Successfully processed {ticker}")
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}", exc_info=True)
            st.error(f"Error processing {ticker}: {str(e)}")
            continue
    
    if not results:
        logger.warning("No results generated")
        return pd.DataFrame()
    
    logger.info("Successfully generated metrics")
    return pd.DataFrame(results) 