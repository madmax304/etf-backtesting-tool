import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from utils import calculate_metrics, validate_tickers
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    st.title("ETF Backtesting Tool")
    st.write("Analyze historical performance of ETFs using a buy-and-hold strategy")

    # User inputs
    st.sidebar.header("Input Parameters")
    
    # Portfolio value input
    initial_investment = st.sidebar.number_input(
        "Initial Investment ($)",
        min_value=100,
        max_value=10000000,
        value=10000,
        step=100,
        help="Enter the total amount you want to invest"
    )
    
    # Text input for ETF tickers
    tickers_input = st.sidebar.text_area(
        "Enter ETF tickers (comma-separated)",
        "SPY, QQQ, VTI",
        help="Example: SPY, QQQ, VTI"
    )
    
    # Portfolio allocation
    st.sidebar.subheader("Portfolio Allocation")
    tickers = [ticker.strip() for ticker in tickers_input.split(",")]
    allocations = {}
    
    # Equal weight by default
    default_weight = 100 / len(tickers)
    total_allocation = 0
    
    for ticker in tickers:
        allocation = st.sidebar.slider(
            f"{ticker} Allocation (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(default_weight),
            step=0.1,
            key=f"allocation_{ticker}"
        )
        allocations[ticker] = allocation
        total_allocation += allocation
    
    # Warning for invalid allocation
    if abs(total_allocation - 100) > 0.1:
        st.sidebar.error(f"Total allocation must be 100% (currently {total_allocation:.1f}%)")
    
    # Date selection
    min_date = datetime.now() - timedelta(days=365*20)  # 20 years ago
    max_date = datetime.now() - timedelta(days=1)  # Yesterday
    
    purchase_date = st.sidebar.date_input(
        "Select Purchase Date",
        value=datetime.now() - timedelta(days=365),
        min_value=min_date,
        max_value=max_date
    )

    if st.sidebar.button("Analyze ETFs"):
        logger.info(f"Starting analysis for tickers: {tickers}")
        
        if abs(total_allocation - 100) > 0.1:
            st.error("Please adjust allocations to total 100% before analyzing")
            return
        
        try:
            # Validate tickers
            valid_tickers, invalid_tickers = validate_tickers(tickers)
            
            if invalid_tickers:
                st.error(f"Invalid tickers found: {', '.join(invalid_tickers)}")
            
            if valid_tickers:
                logger.info(f"Processing valid tickers: {valid_tickers}")
                try:
                    # Calculate metrics for valid tickers
                    results_df = calculate_metrics(
                        valid_tickers, 
                        purchase_date, 
                        initial_investment,
                        {t: allocations[t] for t in valid_tickers}
                    )
                    
                    if not results_df.empty:
                        # Display results
                        st.header("Performance Metrics")
                        st.dataframe(results_df)
                        
                        # Create price chart
                        st.header("Price Performance")
                        fig = create_price_chart(valid_tickers, purchase_date)
                        st.plotly_chart(fig)
                        
                        # Display total portfolio value
                        if 'Current Value' in results_df.columns:
                            total_current_value = results_df['Current Value'].sum()
                            total_return = ((total_current_value / initial_investment) - 1) * 100
                            
                            st.header("Portfolio Summary")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Initial Investment", f"${initial_investment:,.2f}")
                            col2.metric("Current Value", f"${total_current_value:,.2f}")
                            col3.metric("Total Return", f"{total_return:.2f}%")
                    
                except Exception as e:
                    logger.error(f"Error in analysis: {str(e)}", exc_info=True)
                    st.error(f"An error occurred during analysis: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error in ticker validation: {str(e)}", exc_info=True)
            st.error(f"An error occurred during ticker validation: {str(e)}")

def create_price_chart(tickers, start_date):
    fig = go.Figure()
    
    for ticker in tickers:
        try:
            # Fetch historical data
            data = yf.download(ticker, start=start_date, progress=False)
            
            if len(data) == 0:
                logger.warning(f"No data available for {ticker}")
                st.warning(f"No data available for {ticker} from the selected date.")
                continue
            
            # Use 'Close' instead of 'Adj Close'
            close_col = ('Close', ticker) if ('Close', ticker) in data.columns else 'Close'
            
            if close_col not in data.columns:
                logger.error(f"Missing Close price data for {ticker}")
                st.error(f"Price data not available for {ticker}")
                continue
            
            # Normalize prices to 100 at start
            normalized_prices = (data[close_col] / data[close_col].iloc[0]) * 100
            
            # Add trace for each ETF
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=normalized_prices,
                    name=ticker,
                    mode='lines'
                )
            )
            
        except Exception as e:
            logger.error(f"Error plotting {ticker}: {str(e)}", exc_info=True)
            st.error(f"Error plotting {ticker}: {str(e)}")
            continue
    
    fig.update_layout(
        title="Normalized Price Performance (Starting Value = 100)",
        xaxis_title="Date",
        yaxis_title="Normalized Price",
        hovermode='x unified'
    )
    
    return fig

if __name__ == "__main__":
    main() 