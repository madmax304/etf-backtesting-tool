import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from utils import calculate_metrics, validate_tickers, convert_date_for_yfinance
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_ticker_symbols():
    """Load common ticker symbols and their names"""
    common_tickers = {
        # Major ETFs
        "SPY": "SPDR S&P 500 ETF",
        "QQQ": "Invesco QQQ Trust",
        "VTI": "Vanguard Total Stock Market ETF",
        "VOO": "Vanguard S&P 500 ETF",
        "IVV": "iShares Core S&P 500 ETF",
        "VEA": "Vanguard FTSE Developed Markets ETF",
        "BND": "Vanguard Total Bond Market ETF",
        "VWO": "Vanguard FTSE Emerging Markets ETF",
        "AGG": "iShares Core U.S. Aggregate Bond ETF",
        "VIG": "Vanguard Dividend Appreciation ETF",
        # Add more common ETFs and stocks as needed
    }
    return common_tickers

def filter_tickers(search_text, ticker_dict):
    """Filter tickers based on search text"""
    search_text = search_text.upper()
    return {
        k: v for k, v in ticker_dict.items() 
        if search_text in k.upper() or search_text in v.upper()
    }

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
    
    # Initialize tickers in session state if not present
    if 'tickers' not in st.session_state:
        st.session_state.tickers = ["SPY", "QQQ", "VTI"]  # Default tickers
    if 'add_ticker_clicked' not in st.session_state:
        st.session_state.add_ticker_clicked = False

    # Get current tickers from session state
    tickers = st.session_state.tickers

    # Add near the top of the file after imports
    common_tickers = load_ticker_symbols()

    # Create two columns for input
    input_col1, input_col2 = st.sidebar.columns([3, 1])

    # Function to handle ticker addition
    def add_ticker():
        """Handle adding a new ticker"""
        if st.session_state.new_ticker:
            new_ticker = st.session_state.new_ticker.strip().upper()
            if new_ticker and new_ticker not in st.session_state.tickers:
                try:
                    # Validate ticker
                    ticker_obj = yf.Ticker(new_ticker)
                    info = ticker_obj.info
                    if info:
                        st.session_state.tickers.append(new_ticker)
                        st.session_state.new_ticker = ""  # Clear input
                        st.rerun()
                    else:
                        st.sidebar.error(f"Could not validate {new_ticker}")
                except Exception as e:
                    st.sidebar.error(f"Invalid ticker: {new_ticker}")
            elif new_ticker in st.session_state.tickers:
                st.sidebar.warning(f"{new_ticker} is already in your portfolio")
                st.session_state.new_ticker = ""  # Clear on duplicate

    # Simple ticker input
    new_ticker = input_col1.text_input(
        "Add Ticker",
        key="new_ticker",
        help="Enter a ticker symbol and press Enter",
        on_change=add_ticker
    )

    # Add button as alternative to Enter
    if input_col2.button("Add", use_container_width=True):
        add_ticker()

    # Replace the current tickers display section with this:
    st.sidebar.write("Current Tickers:")
    
    # Update the CSS to include tooltip styling
    st.markdown("""
    <style>
        .ticker-container {
            display: flex;
            align-items: center;
            padding: 0.5rem;
            margin: 0.25rem 0;
            background-color: #f0f2f6;
            border-radius: 0.5rem;
        }
        .ticker-symbol {
            font-weight: bold;
            min-width: 80px;
        }
        .ticker-name {
            flex-grow: 1;
            color: #666;
            font-size: 0.9em;
            margin-left: 0.5rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            position: relative;
        }
        .ticker-name:hover {
            cursor: help;
        }
        .remove-button {
            padding: 0.2rem 0.5rem !important;
            font-size: 0.8rem !important;
            line-height: 1 !important;
            height: auto !important;
            margin-left: 0.5rem !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Update the ticker display section
    for i, ticker in enumerate(st.session_state.tickers):
        # Create container for each ticker
        with st.sidebar.container():
            # Create three columns: ticker, name, remove button
            col1, col2, col3 = st.sidebar.columns([1, 3, 1])
            
            # Ticker symbol
            col1.markdown(f"**{ticker}**")
            
            try:
                # Get company info
                ticker_info = yf.Ticker(ticker).info
                company_name = ticker_info.get('shortName', '')
                # Company name
                col2.write(company_name if company_name else ticker)
            except Exception as e:
                # Fallback to just showing ticker
                logger.error(f"Error getting info for {ticker}: {str(e)}")
                col2.write(ticker)
            
            # Remove button with unique key using position
            if col3.button("✕", 
                          key=f"remove_btn_{i}",  # Simpler unique key
                          help=f"Remove {ticker}", 
                          use_container_width=True):
                # Store the ticker to be removed
                ticker_to_remove = ticker  # Use the current ticker
                # First remove from allocations
                reallocate_after_removal(ticker_to_remove)
                # Then remove from tickers list
                st.session_state.tickers.remove(ticker_to_remove)
                st.rerun()
    
    # Add a divider
    st.sidebar.markdown("---")

    # Initialize allocations in session state if not present
    if 'allocations' not in st.session_state:
        # Start with equal weight
        default_weight = 100 / len(tickers)
        st.session_state.allocations = {ticker: default_weight for ticker in tickers}
    
    # Handle new tickers with proportional allocation
    if any(ticker not in st.session_state.allocations for ticker in tickers):
        # Get existing total allocation
        existing_total = sum(st.session_state.allocations.get(t, 0) for t in st.session_state.allocations)
        new_tickers = [t for t in tickers if t not in st.session_state.allocations]
        
        if existing_total == 0:
            # If no existing allocations, distribute equally
            new_weight = 100 / len(tickers)
            for ticker in tickers:
                st.session_state.allocations[ticker] = new_weight
        else:
            # Reduce existing allocations proportionally
            reduction_factor = 0.8  # Keep 80% of current allocations
            for ticker in st.session_state.allocations:
                if ticker in tickers:  # Only adjust if ticker is still in list
                    st.session_state.allocations[ticker] *= reduction_factor
            
            # Distribute remaining allocation among new tickers
            remaining_allocation = 100 - sum(st.session_state.allocations.values())
            new_weight = remaining_allocation / len(new_tickers)
            for ticker in new_tickers:
                st.session_state.allocations[ticker] = new_weight
    
    # Remove any old tickers
    st.session_state.allocations = {k: v for k, v in st.session_state.allocations.items() if k in tickers}
    
    def adjust_other_allocations(changed_ticker):
        """Adjust allocations when a slider changes"""
        # Check if the ticker still exists
        if changed_ticker not in st.session_state.tickers:
            return
        
        # Get the new value from session state
        new_value = st.session_state.get(f"allocation_{changed_ticker}")
        old_value = st.session_state.allocations.get(changed_ticker, 0)
        
        if new_value is None:  # If the slider no longer exists
            return
        
        delta = new_value - old_value
        
        if delta == 0:
            return
        
        # Get other tickers and their allocations
        other_tickers = [t for t in st.session_state.tickers if t != changed_ticker]
        other_total = sum(st.session_state.allocations.get(t, 0) for t in other_tickers)
        
        if other_total == 0:
            # If other allocations are 0, distribute equally
            for ticker in other_tickers:
                st.session_state.allocations[ticker] = (100 - new_value) / len(other_tickers)
        else:
            # Adjust other allocations proportionally
            for ticker in other_tickers:
                current_alloc = st.session_state.allocations.get(ticker, 0)
                proportion = current_alloc / other_total
                st.session_state.allocations[ticker] = max(0, current_alloc - (delta * proportion))
        
        st.session_state.allocations[changed_ticker] = new_value

    # Display sliders and handle changes
    total_allocation = 0
    for ticker in tickers:
        col1, col2 = st.sidebar.columns([4, 1])
        
        # Slider in first column
        new_allocation = col1.slider(
            f"{ticker}",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.allocations[ticker]),
            step=0.1,
            key=f"allocation_{ticker}",
            on_change=adjust_other_allocations,
            args=(ticker,)  # Only pass the ticker name
        )
        
        # Percentage in second column
        col2.write(f"{new_allocation:.1f}%")
        
        total_allocation += new_allocation
    
    # Show total allocation
    st.sidebar.write(f"Total Allocation: {total_allocation:.1f}%")
    if abs(total_allocation - 100) > 0.1:
        st.sidebar.error("Total allocation must be 100%")
    
    # Update allocations dict for use in calculations
    allocations = st.session_state.allocations

    # Date selection
    min_date = datetime.now() - timedelta(days=365*20)  # 20 years ago
    max_date = datetime.now() - timedelta(days=1)  # Yesterday
    
    col1, col2 = st.sidebar.columns(2)
    
    # Initialize the dates in session state if they don't exist
    if 'start_date' not in st.session_state:
        st.session_state.start_date = datetime.now() - timedelta(days=365)
    if 'end_date' not in st.session_state:
        st.session_state.end_date = datetime.now() - timedelta(days=1)
    
    start_date = col1.date_input(
        "Start Date",
        value=st.session_state.start_date,
        min_value=min_date,
        max_value=max_date,
        key='start_date_input'
    )
    
    # Update session state
    st.session_state.start_date = start_date
    
    end_date = col2.date_input(
        "End Date",
        value=st.session_state.end_date,
        min_value=start_date,
        max_value=max_date,
        key='end_date_input'
    )
    
    # Update session state
    st.session_state.end_date = end_date

    # Add chart controls
    st.sidebar.subheader("Chart Options")
    show_volume = st.sidebar.checkbox("Show Volume", value=False)
    log_scale = st.sidebar.checkbox("Logarithmic Scale", value=False)

    # Rebalancing frequency
    rebalance_frequency = st.sidebar.selectbox(
        "Rebalancing Frequency",
        ["None", "Monthly", "Quarterly", "Annually"]
    )

    # At the top of main()
    theme = st.sidebar.selectbox(
        "Chart Theme",
        ["Default", "Dark", "Light"]
    )

    # Add some CSS to make it look better
    st.markdown("""
    <style>
        .stButton>button {
            height: 1.5rem;
            padding: 0rem 0.5rem;
        }
        .stButton>button:hover {
            background-color: #ff4b4b;
            color: white;
        }
        div[data-testid="stCode"] {
            background-color: #f0f2f6;
            padding: 0.25rem 0.75rem;
            margin: 0;
            line-height: 1.5;
            border-radius: 0.25rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Remove the "Analyze ETFs" button and automatically process when inputs change
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
                with st.spinner('Fetching and analyzing ETF data...'):
                    # Calculate metrics for valid tickers
                    results_df = calculate_metrics(
                        valid_tickers, 
                        start_date,
                        end_date,
                        initial_investment,
                        allocations,
                        rebalance_frequency
                    )
                    
                    if not results_df.empty:
                        # Display results
                        st.header("Performance Metrics")
                        st.dataframe(results_df)
                        
                        # Add download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="etf_analysis_results.csv",
                            mime="text/csv"
                        )
                        
                        # Create price chart
                        st.header("Price Performance")
                        fig = create_price_chart(
                            valid_tickers, 
                            start_date,
                            end_date,
                            show_volume=show_volume, 
                            log_scale=log_scale,
                            theme=theme
                        )
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
                
                        # Add HYSA comparison
                        st.header("Benchmark Comparison")
                        hysa_results = calculate_hysa_returns(initial_investment, start_date, end_date)
                        
                        if hysa_results:
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric(
                                "HYSA Final Value", 
                                f"${hysa_results['Current Value']:,.2f}",
                                f"{hysa_results['Total Return (%)']:.2f}%"
                            )
                            col2.metric("HYSA APY", f"{hysa_results['APY (%)']:.2f}%")
                            
                            # Compare with portfolio performance
                            portfolio_value = results_df['Current Value'].sum()
                            difference = portfolio_value - hysa_results['Current Value']
                            difference_pct = ((portfolio_value / hysa_results['Current Value']) - 1) * 100
                            
                            col3.metric(
                                "Portfolio vs HYSA", 
                                f"${difference:,.2f}",
                                f"{difference_pct:+.2f}%",
                                delta_color="normal"
                            )
                            
                            # Add to chart for comparison with actual historical values
                            fig.add_trace(
                                go.Scatter(
                                    x=hysa_results['Dates'],
                                    y=[(v / initial_investment) * 100 for v in hysa_results['Values']],
                                    name='High-Yield Savings',
                                    line=dict(dash='dash'),
                                    opacity=0.7
                                )
                            )
                            
                            # Add rate information
                            st.sidebar.markdown("### Historical HYSA Rates")
                            st.sidebar.markdown("""
                            - 2019: 2.0% APY
                            - 2020 (COVID): 1.0% APY
                            - 2021: 0.5% APY
                            - 2022 Q1: 1.5% APY
                            - 2022 Q3: 2.5% APY
                            - 2022 Q4: 3.5% APY
                            - 2023 Q2: 4.0% APY
                            - 2023 Q3-Present: 4.5% APY
                            """)
                
                            # Add inflation comparison
                            inflation_results = calculate_inflation_metrics(initial_investment, start_date, end_date)
                            if inflation_results:
                                st.header("Risk-Adjusted Performance")
                                
                                # Calculate real returns (inflation-adjusted)
                                portfolio_real_return = total_return - inflation_results['Total Inflation (%)']
                                hysa_real_return = hysa_results['Total Return (%)'] - inflation_results['Total Inflation (%)']
                                
                                col1, col2, col3 = st.columns(3)
                                
                                # Show real returns
                                col1.metric(
                                    "Portfolio Real Return",
                                    f"{portfolio_real_return:.2f}%",
                                    f"{total_return:.2f}% nominal"
                                )
                                
                                col2.metric(
                                    "HYSA Real Return",
                                    f"{hysa_real_return:.2f}%",
                                    f"{hysa_results['Total Return (%)']:.2f}% nominal"
                                )
                                
                                col3.metric(
                                    "Total Inflation",
                                    f"{inflation_results['Total Inflation (%)']:.2f}%"
                                )
                                
                                # Add inflation line to chart
                                fig.add_trace(
                                    go.Scatter(
                                        x=inflation_results['Dates'],
                                        y=[(v / initial_investment) * 100 for v in inflation_results['Values']],
                                        name='Inflation',
                                        line=dict(dash='dot', color='red'),
                                        opacity=0.5
                                    )
                                )
                                
                                # Add inflation information to sidebar
                                st.sidebar.markdown("### Historical Inflation Rates")
                                st.sidebar.markdown("""
                                - 2019: 2.5%
                                - 2020: 1.4%
                                - 2021: 7.0%
                                - 2022: 6.5%
                                - 2023: 3.4%
                                - 2024: 3.1%
                                """)
                                
                                # Add risk metrics
                                st.subheader("Risk Analysis")
                                risk_df = pd.DataFrame({
                                    'Metric': ['Real Return (%)', 'Risk Level', 'Liquidity', 'Principal Protection'],
                                    'Portfolio': [f"{portfolio_real_return:.2f}%", 'Moderate-High', 'High', 'No'],
                                    'HYSA': [f"{hysa_real_return:.2f}%", 'Very Low', 'High', 'Yes (FDIC)'],
                                }).set_index('Metric')
                                
                                st.dataframe(risk_df)
                
            except Exception as e:
                logger.error(f"Error in analysis: {str(e)}", exc_info=True)
                st.error(f"An error occurred during analysis: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error in ticker validation: {str(e)}", exc_info=True)
        st.error(f"An error occurred during ticker validation: {str(e)}")

def create_price_chart(tickers, start_date, end_date, show_volume=False, log_scale=False, theme="Default"):
    # Convert dates to proper format
    start_date = convert_date_for_yfinance(start_date)
    end_date = convert_date_for_yfinance(end_date)
    
    fig = go.Figure()
    
    # Define theme colors
    if theme == "Dark":
        background_color = "#1f1f1f"
        text_color = "#ffffff"
        grid_color = "#333333"
    elif theme == "Light":
        background_color = "#ffffff"
        text_color = "#000000"
        grid_color = "#e5e5e5"
    else:  # Default theme
        background_color = "#ffffff"
        text_color = "#000000"
        grid_color = "#e5e5e5"
    
    for ticker in tickers:
        try:
            # Fetch historical data with end date
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
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
            
            # Add price trace for each ETF
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=normalized_prices,
                    name=ticker,
                    mode='lines'
                )
            )

            # Add volume trace if requested
            if show_volume:
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data['Volume'],
                        name=f"{ticker} Volume",
                        yaxis="y2",
                        opacity=0.3
                    )
                )
            
        except Exception as e:
            logger.error(f"Error plotting {ticker}: {str(e)}", exc_info=True)
            st.error(f"Error plotting {ticker}: {str(e)}")
            continue
    
    # Update layout based on options and theme
    layout_updates = {
        "title": "Normalized Price Performance (Starting Value = 100)",
        "xaxis_title": "Date",
        "yaxis_title": "Normalized Price",
        "hovermode": 'x unified',
        "plot_bgcolor": background_color,
        "paper_bgcolor": background_color,
        "font": {"color": text_color},
        "xaxis": {
            "gridcolor": grid_color,
            "zerolinecolor": grid_color
        },
        "yaxis": {
            "gridcolor": grid_color,
            "zerolinecolor": grid_color
        }
    }

    # Add log scale if selected
    if log_scale:
        layout_updates["yaxis_type"] = "log"

    # Add volume axis if showing volume
    if show_volume:
        layout_updates.update({
            "yaxis2": {
                "title": "Volume",
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
                "gridcolor": grid_color,
                "zerolinecolor": grid_color
            }
        })

    fig.update_layout(**layout_updates)
    
    return fig

def reallocate_after_removal(removed_ticker):
    """Redistribute the allocation of a removed ticker proportionally among remaining tickers"""
    try:
        # Get the allocation of the removed ticker (default to 0 if not found)
        removed_allocation = st.session_state.allocations.pop(removed_ticker, 0)
        remaining_tickers = [t for t in st.session_state.tickers if t != removed_ticker]
        
        if not remaining_tickers:
            return
        
        # Get total of remaining allocations
        remaining_total = sum(st.session_state.allocations.get(t, 0) for t in remaining_tickers)
        
        if remaining_total == 0:
            # If all remaining allocations are 0, distribute equally
            equal_weight = 100 / len(remaining_tickers)
            for ticker in remaining_tickers:
                st.session_state.allocations[ticker] = equal_weight
        else:
            # Distribute proportionally
            for ticker in remaining_tickers:
                current_alloc = st.session_state.allocations.get(ticker, 0)
                proportion = current_alloc / remaining_total if remaining_total > 0 else 1/len(remaining_tickers)
                st.session_state.allocations[ticker] = current_alloc + (removed_allocation * proportion)
        
        # Clean up any old tickers from allocations
        st.session_state.allocations = {k: v for k, v in st.session_state.allocations.items() 
                                      if k in remaining_tickers}
        
    except Exception as e:
        logger.error(f"Error in reallocation: {str(e)}")
        # Reset allocations to equal weight if something goes wrong
        remaining_tickers = [t for t in st.session_state.tickers if t != removed_ticker]
        if remaining_tickers:
            equal_weight = 100 / len(remaining_tickers)
            st.session_state.allocations = {ticker: equal_weight for ticker in remaining_tickers}

def calculate_hysa_returns(initial_investment, start_date, end_date):
    """Calculate returns if the money was in a high-yield savings account using historical rates"""
    try:
        # Convert dates
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Create monthly date range for calculations
        dates = pd.date_range(start=start_date, end=end_date, freq='ME')  # Changed from 'M' to 'ME'
        if dates.empty:
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Historical HYSA rates (approximate based on Fed Funds Rate + spread)
        # Key dates and rates (simplified model)
        historical_rates = {
            '2019-01-01': 0.020,  # 2.0% APY
            '2020-03-15': 0.010,  # COVID cut
            '2021-01-01': 0.005,  # Low rate environment
            '2022-03-01': 0.015,  # Rate hikes begin
            '2022-07-01': 0.025,
            '2022-12-01': 0.035,
            '2023-05-01': 0.040,
            '2023-08-01': 0.045,
            '2024-01-01': 0.045,  # Current rate
        }
        
        # Convert to DataFrame for easier date matching
        rates_df = pd.DataFrame(
            list(historical_rates.items()), 
            columns=['date', 'rate']
        ).set_index('date')
        rates_df.index = pd.to_datetime(rates_df.index)
        
        # Calculate monthly values with changing rates
        values = []
        current_value = initial_investment
        
        for i in range(len(dates)):
            current_date = dates[i]
            # Get applicable rate (use last known rate)
            applicable_rate = rates_df[rates_df.index <= current_date].iloc[-1]['rate']
            monthly_rate = applicable_rate / 12  # Convert annual rate to monthly
            
            # Compound monthly
            current_value *= (1 + monthly_rate)
            values.append(current_value)
        
        final_value = values[-1] if values else initial_investment
        total_return = ((final_value / initial_investment) - 1) * 100
        
        # Get current rate for display
        current_rate = rates_df.iloc[-1]['rate']
        
        return {
            'Investment Type': 'High-Yield Savings',
            'Initial Investment': initial_investment,
            'Current Value': final_value,
            'Total Return (%)': total_return,
            'APY (%)': current_rate * 100,
            'Values': values,
            'Dates': dates,
            'Risk Level': 'Very Low'
        }
        
    except Exception as e:
        logger.error(f"Error calculating HYSA returns: {str(e)}")
        return None

def calculate_inflation_metrics(initial_investment, start_date, end_date):
    """Calculate inflation metrics and inflation-adjusted returns"""
    try:
        # Convert dates
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Historical CPI data (monthly) - simplified version
        # You could also fetch this from FRED API for more accurate data
        historical_cpi = {
            '2019-01-01': 2.5,  # Annual inflation rate %
            '2020-01-01': 1.4,
            '2021-01-01': 7.0,
            '2022-01-01': 6.5,
            '2023-01-01': 3.4,
            '2024-01-01': 3.1,  # Current rate
        }
        
        # Convert to DataFrame
        cpi_df = pd.DataFrame(
            list(historical_cpi.items()), 
            columns=['date', 'rate']
        ).set_index('date')
        cpi_df.index = pd.to_datetime(cpi_df.index)
        
        # Calculate monthly values with changing inflation rates
        dates = pd.date_range(start=start_date, end=end_date, freq='ME')  # Changed from 'M' to 'ME'
        if dates.empty:
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        values = []
        current_value = initial_investment
        
        for date in dates:
            # Get applicable inflation rate
            inflation_rate = cpi_df[cpi_df.index <= date].iloc[-1]['rate']
            monthly_rate = inflation_rate / 12 / 100  # Convert annual % to monthly decimal
            current_value *= (1 + monthly_rate)
            values.append(current_value)
        
        final_value = values[-1] if values else initial_investment
        total_inflation = ((final_value / initial_investment) - 1) * 100
        
        return {
            'Type': 'Inflation',
            'Initial Value': initial_investment,
            'Current Value': final_value,
            'Total Inflation (%)': total_inflation,
            'Values': values,
            'Dates': dates
        }
        
    except Exception as e:
        logger.error(f"Error calculating inflation metrics: {str(e)}")
        return None

if __name__ == "__main__":
    main() 