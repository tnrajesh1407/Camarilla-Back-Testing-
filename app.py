import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Camarilla Technical Trader Analyst",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Camarilla Technical Trader Analyst")
st.markdown("---")

# Camarilla Pivot Point Calculations
def calculate_camarilla_pivots(df):
    """
    Calculate Camarilla Pivot Points
    H4 = Close + ((High - Low) * 1.1/2)
    H3 = Close + ((High - Low) * 1.1/4)
    L3 = Close - ((High - Low) * 1.1/4)
    L4 = Close - ((High - Low) * 1.1/2)
    """
    df = df.copy()
    
    # Get previous day's data for pivot calculation
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_High'] = df['High'].shift(1)
    df['Prev_Low'] = df['Low'].shift(1)
    
    # Calculate range
    df['Range'] = df['Prev_High'] - df['Prev_Low']
    
    # Camarilla Pivot Points
    df['H4'] = df['Prev_Close'] + (df['Range'] * 1.1 / 2)
    df['H3'] = df['Prev_Close'] + (df['Range'] * 1.1 / 4)
    df['L3'] = df['Prev_Close'] - (df['Range'] * 1.1 / 4)
    df['L4'] = df['Prev_Close'] - (df['Range'] * 1.1 / 2)
    
    # Calculate gaps
    df['Gap_High_H4'] = df['High'] - df['H4']
    df['Gap_High_H3'] = df['High'] - df['H3']
    df['Gap_Low_L4'] = df['L4'] - df['Low']
    df['Gap_Low_L3'] = df['L3'] - df['Low']
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

# Fetch data from Yahoo Finance
def get_yahoo_data(symbol, timeframe, days):
    """Fetch data from Yahoo Finance"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=timeframe)
        
        if df.empty:
            st.error(f"No data found for symbol: {symbol}")
            return None
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Rename columns to standard format
        df.columns = [col.replace('Date', 'Time') if col == 'Date' or col == 'Datetime' else col for col in df.columns]
        
        return df
    except Exception as e:
        st.error(f"Error fetching Yahoo Finance data: {str(e)}")
        return None

# Fetch data from MetaTrader 5
def get_mt5_data(symbol, timeframe, days):
    """Fetch data from MetaTrader 5"""
    try:
        # Initialize MT5
        if not mt5.initialize():
            st.error("MetaTrader 5 initialization failed")
            return None
        
        # Get data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        
        if rates is None or len(rates) == 0:
            st.error(f"No data found for symbol: {symbol}")
            mt5.shutdown()
            return None
        
        # Create DataFrame
        df = pd.DataFrame(rates)
        df['Time'] = pd.to_datetime(df['time'], unit='s')
        
        # Rename columns to match standard format
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Volume'
        })
        
        mt5.shutdown()
        
        return df
    except Exception as e:
        st.error(f"Error fetching MT5 data: {str(e)}")
        if MT5_AVAILABLE:
            mt5.shutdown()
        return None

# Calculate statistics for different periods
def calculate_period_stats(df, period_days, metric_col):
    """Calculate average for a specific period"""
    cutoff_date = df['Time'].max() - timedelta(days=period_days)
    period_df = df[df['Time'] >= cutoff_date]
    
    if len(period_df) == 0:
        return None
    
    return period_df[metric_col].mean()

# Backtesting Functions
def calculate_rolling_stoploss(df, window_days, gap_col):
    """Calculate rolling average stoploss based on last month"""
    df = df.copy()
    df['Stoploss_Value'] = df[gap_col].rolling(window=window_days, min_periods=1).mean()
    return df

def calculate_ideal_stoploss_enhanced(trades_df):
    """
    Enhanced stoploss calculation considering multiple factors
    
    Returns:
        dict with 'simple', 'enhanced', 'winners_avg', 'losers_adjusted', 'volatility_adjusted'
    """
    if len(trades_df) == 0:
        return {
            'simple': None,
            'enhanced': None,
            'winners_avg': None,
            'losers_adjusted': None,
            'volatility_adjusted': None
        }
    
    # Method 1: Average of winners (current/simple method)
    winners = trades_df[trades_df['PnL'] > 0]
    winners_sl = winners['Stoploss_Distance'].mean() if len(winners) > 0 else None
    
    # Method 2: Analyze losers that hit stoploss
    # If we had used a wider SL, could we have saved them?
    losers = trades_df[trades_df['PnL'] < 0]
    losers_hit_sl = losers[losers['Hit_Stoploss'] == True]
    
    if len(losers_hit_sl) > 0:
        # Try 20% wider SL to potentially save losing trades
        losers_adjusted_sl = losers_hit_sl['Stoploss_Distance'].mean() * 1.2
    else:
        losers_adjusted_sl = winners_sl if winners_sl else None
    
    # Method 3: Volatility-adjusted (mean + 1 std dev for safety margin)
    all_sl_distances = trades_df['Stoploss_Distance']
    if len(all_sl_distances) > 1:
        volatility_adjusted_sl = all_sl_distances.mean() + all_sl_distances.std()
    else:
        volatility_adjusted_sl = all_sl_distances.mean() if len(all_sl_distances) > 0 else None
    
    # Method 4: Enhanced combined approach
    # Weight: 50% winners, 30% losers adjusted, 20% volatility
    if winners_sl is not None and losers_adjusted_sl is not None and volatility_adjusted_sl is not None:
        enhanced_sl = (winners_sl * 0.5 + 
                      losers_adjusted_sl * 0.3 + 
                      volatility_adjusted_sl * 0.2)
    else:
        enhanced_sl = winners_sl  # Fallback to simple method
    
    return {
        'simple': winners_sl,
        'enhanced': enhanced_sl,
        'winners_avg': winners_sl,
        'losers_adjusted': losers_adjusted_sl,
        'volatility_adjusted': volatility_adjusted_sl
    }

def rank_scenarios(all_results, initial_capital, profit_target_pct):
    """
    Rank scenarios based on multiple performance criteria
    
    Returns:
        DataFrame with ranked scenarios and scores
    """
    if not all_results:
        return None
    
    rankings = []
    
    for scenario_name, metrics in all_results.items():
        if metrics['total_trades'] == 0:
            continue
        
        # Calculate individual scores (0-100 scale)
        
        # 1. ROI Score (higher is better)
        roi_pct = (metrics['net_profit'] / initial_capital * 100) if initial_capital > 0 else 0
        roi_score = min(100, max(0, roi_pct * 2))  # Scale: 50% ROI = 100 points
        
        # 2. Days to Target Score (fewer days is better)
        if metrics['days_to_target'] is not None:
            # Assuming 90 days (3 months) as baseline
            days_diff = (metrics['days_to_target'] - metrics.get('first_trade_date', metrics['days_to_target'])).days
            days_score = max(0, 100 - (days_diff / 90 * 100))  # 90 days = 0 points, 0 days = 100 points
        else:
            days_score = 0  # Didn't reach target
        
        # 3. Win Rate Score (higher is better)
        win_rate_score = metrics['win_rate']  # Already 0-100
        
        # 4. Risk Score (lower risk is better)
        # Consider days below $90k and days with >5% loss
        days_below_90k_penalty = metrics['days_below_90k'] * 5  # 5 points penalty per day
        days_5pct_loss_penalty = metrics['days_below_5pct_loss'] * 3  # 3 points penalty per day
        risk_score = max(0, 100 - days_below_90k_penalty - days_5pct_loss_penalty)
        
        # 5. Profit Factor Score
        profit_factor = metrics['total_profit'] / metrics['total_loss'] if metrics['total_loss'] > 0 else 10
        profit_factor_score = min(100, profit_factor * 40)  # 2.5 PF = 100 points
        
        # 6. Enhanced SL Quality Score (lower is better, tighter risk control)
        if metrics['ideal_stoploss_enhanced'] is not None:
            # Normalize based on typical range (assume 1-5 is common range)
            sl_score = max(0, 100 - (metrics['ideal_stoploss_enhanced'] * 20))
        else:
            sl_score = 50  # Neutral score if not available
        
        # Calculate weighted composite score
        # Prioritize: ROI (30%), Days to Target (25%), Risk (20%), Win Rate (15%), Profit Factor (10%)
        composite_score = (
            roi_score * 0.30 +
            days_score * 0.25 +
            risk_score * 0.20 +
            win_rate_score * 0.15 +
            profit_factor_score * 0.10
        )
        
        rankings.append({
            'Scenario': scenario_name,
            'Composite Score': round(composite_score, 2),
            'ROI (%)': round(roi_pct, 2),
            'Days to Target': str(days_diff) if metrics['days_to_target'] else 'N/A',
            'Win Rate (%)': round(metrics['win_rate'], 2),
            'Profit Factor': round(profit_factor, 2),
            'Net Profit': round(metrics['net_profit'], 2),
            'Days <$90k': int(metrics['days_below_90k']),
            'Days >5% Loss': int(metrics['days_below_5pct_loss']),
            'Enhanced SL': str(round(metrics['ideal_stoploss_enhanced'], 5)) if metrics['ideal_stoploss_enhanced'] else 'N/A',
            'Total Trades': int(metrics['total_trades']),
            'ROI Score': round(roi_score, 1),
            'Days Score': round(days_score, 1),
            'Risk Score': round(risk_score, 1),
            'Win Rate Score': round(win_rate_score, 1),
            'PF Score': round(profit_factor_score, 1)
        })
    
    # Sort by composite score (descending)
    rankings_df = pd.DataFrame(rankings)
    if len(rankings_df) > 0:
        rankings_df = rankings_df.sort_values('Composite Score', ascending=False)
        rankings_df = rankings_df.reset_index(drop=True)
        rankings_df.index = rankings_df.index + 1  # Start ranking from 1
    
    return rankings_df

def backtest_scenario(df, scenario_config, initial_capital=100000, profit_target_pct=0.10, dollar_per_tick=1):
    """
    Run backtest for a specific scenario
    
    Parameters:
    - df: DataFrame with OHLC and Camarilla levels
    - scenario_config: dict with 'entry_level', 'stoploss_gap', 'target_level', 'trade_type'
    - initial_capital: Starting capital
    - profit_target_pct: Target profit percentage (0.10 = 10%)
    - dollar_per_tick: Multiplier for position sizing ($1-$10 per tick)
    """
    
    df = df.copy()
    entry_level = scenario_config['entry_level']
    stoploss_gap_col = scenario_config['stoploss_gap']
    target_level = scenario_config['target_level']
    trade_type = scenario_config['trade_type']  # 'long' or 'short'
    
    # Calculate rolling 30-day average stoploss
    df = calculate_rolling_stoploss(df, window_days=30, gap_col=stoploss_gap_col)
    
    # Initialize tracking variables
    capital = initial_capital
    profit_target_amount = initial_capital * profit_target_pct
    
    trades = []
    capital_history = []
    
    for idx, row in df.iterrows():
        # Check if we can enter trade (price reaches entry level)
        can_enter = False
        
        if trade_type == 'long':
            # For long: enter when Low touches or goes below entry level
            if row['Low'] <= row[entry_level]:
                can_enter = True
                entry_price = row[entry_level]
        else:  # short
            # For short: enter when High touches or goes above entry level
            if row['High'] >= row[entry_level]:
                can_enter = True
                entry_price = row[entry_level]
        
        if can_enter:
            # Calculate stoploss level
            stoploss_distance = abs(row['Stoploss_Value'])
            
            if trade_type == 'long':
                stoploss_price = entry_price - stoploss_distance
                target_price = row[target_level]
            else:  # short
                stoploss_price = entry_price + stoploss_distance
                target_price = row[target_level]
            
            # Determine trade outcome
            hit_target = False
            hit_stoploss = False
            exit_price = row['Close']
            
            if trade_type == 'long':
                # Check if target was hit
                if row['High'] >= target_price:
                    hit_target = True
                    exit_price = target_price
                # Check if stoploss was hit
                elif row['Low'] <= stoploss_price:
                    hit_stoploss = True
                    exit_price = stoploss_price
            else:  # short
                # Check if target was hit
                if row['Low'] <= target_price:
                    hit_target = True
                    exit_price = target_price
                # Check if stoploss was hit
                elif row['High'] >= stoploss_price:
                    hit_stoploss = True
                    exit_price = stoploss_price
            
            # Calculate P&L
            if trade_type == 'long':
                price_change = exit_price - entry_price
            else:  # short
                price_change = entry_price - exit_price
            
            # Calculate position size
            # dollar_per_tick defines how much each 1-point price movement is worth
            # If dollar_per_tick = $1, then 1 point move = $1 profit/loss
            # If dollar_per_tick = $5, then 1 point move = $5 profit/loss
            # Formula: P&L = price_change * dollar_per_tick
            pnl_amount = price_change * dollar_per_tick
            
            # Update capital
            capital += pnl_amount
            
            # Record trade
            trades.append({
                'Date': row['Time'],
                'Type': trade_type.upper(),
                'Entry': entry_price,
                'Exit': exit_price,
                'Stoploss': stoploss_price,
                'Target': target_price,
                'Hit_Target': hit_target,
                'Hit_Stoploss': hit_stoploss,
                'PnL': pnl_amount,
                'Capital': capital,
                'Stoploss_Distance': stoploss_distance,
                'Daily_Loss_Pct': (pnl_amount / (capital - pnl_amount) * 100) if pnl_amount < 0 else 0
            })
        
        capital_history.append({
            'Date': row['Time'],
            'Capital': capital
        })
    
    # Create results DataFrame
    trades_df = pd.DataFrame(trades)
    capital_df = pd.DataFrame(capital_history)
    
    # Calculate metrics
    if len(trades_df) > 0:
        total_profit = trades_df[trades_df['PnL'] > 0]['PnL'].sum()
        total_loss = abs(trades_df[trades_df['PnL'] < 0]['PnL'].sum())
        
        # Days to reach profit target
        cumulative_profit = trades_df['PnL'].cumsum()
        days_to_target = None
        first_trade_date = trades_df.iloc[0]['Date'] if len(trades_df) > 0 else None
        
        if (cumulative_profit >= profit_target_amount).any():
            days_to_target = trades_df[cumulative_profit >= profit_target_amount].iloc[0]['Date']
        
        # Days when capital fell below $90,000
        days_below_90k = len(trades_df[trades_df['Capital'] < 90000])
        
        # Days when daily loss exceeded 5%
        days_below_5pct = len(trades_df[trades_df['Daily_Loss_Pct'] <= -5])
        
        # Calculate ideal stoploss using both methods
        stoploss_results = calculate_ideal_stoploss_enhanced(trades_df)
        
        metrics = {
            'total_trades': len(trades_df),
            'winning_trades': len(trades_df[trades_df['PnL'] > 0]),
            'losing_trades': len(trades_df[trades_df['PnL'] < 0]),
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_profit': total_profit - total_loss,
            'final_capital': capital,
            'days_to_target': days_to_target,
            'first_trade_date': first_trade_date,
            'days_below_90k': days_below_90k,
            'days_below_5pct_loss': days_below_5pct,
            'ideal_stoploss': stoploss_results['simple'],  # Keep original for compatibility
            'ideal_stoploss_enhanced': stoploss_results['enhanced'],
            'ideal_stoploss_winners': stoploss_results['winners_avg'],
            'ideal_stoploss_losers_adj': stoploss_results['losers_adjusted'],
            'ideal_stoploss_volatility': stoploss_results['volatility_adjusted'],
            'win_rate': len(trades_df[trades_df['PnL'] > 0]) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        }
    else:
        metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0,
            'total_loss': 0,
            'net_profit': 0,
            'final_capital': initial_capital,
            'days_to_target': None,
            'first_trade_date': None,
            'days_below_90k': 0,
            'days_below_5pct_loss': 0,
            'ideal_stoploss': None,
            'ideal_stoploss_enhanced': None,
            'ideal_stoploss_winners': None,
            'ideal_stoploss_losers_adj': None,
            'ideal_stoploss_volatility': None,
            'win_rate': 0
        }
    
    return trades_df, capital_df, metrics

# Create tabs
tab1, tab2, tab3 = st.tabs(["📈 Analysis", "🔄 Backtesting", "🏆 Best Scenario Finder"])

# Tab 1: Original Analysis
with tab1:
    # Sidebar for inputs
    st.sidebar.header("Analysis Configuration")

    # Data Source Selection
    data_sources = ["Yahoo Finance"]
    if MT5_AVAILABLE:
        data_sources.append("MetaTrader 5")

    data_source = st.sidebar.selectbox(
        "Select Data Source",
        data_sources,
        key="analysis_source"
    )

    # Symbol input based on data source
    if data_source == "Yahoo Finance":
        symbol = st.sidebar.text_input(
            "Enter Symbol (e.g., AAPL, EURUSD=X, BTC-USD)",
            value="AAPL",
            key="analysis_symbol"
        )
        timeframe = st.sidebar.selectbox(
            "Select Timeframe",
            ["1d", "1h", "4h"],
            key="analysis_timeframe"
        )
    else:  # MetaTrader 5
        symbol = st.sidebar.text_input(
            "Enter Symbol (e.g., EURUSD, GBPUSD)",
            value="EURUSD",
            key="analysis_symbol_mt5"
        )
        timeframe_options = {
            "1D": mt5.TIMEFRAME_D1 if MT5_AVAILABLE else None,
            "4H": mt5.TIMEFRAME_H4 if MT5_AVAILABLE else None,
            "1H": mt5.TIMEFRAME_H1 if MT5_AVAILABLE else None
        }
        timeframe_display = st.sidebar.selectbox(
            "Select Timeframe",
            list(timeframe_options.keys()),
            key="analysis_timeframe_mt5"
        )
        timeframe = timeframe_options[timeframe_display]

    # Date range
    lookback_days = st.sidebar.number_input(
        "Lookback Days",
        min_value=30,
        max_value=1825,
        value=365,
        step=30,
        key="analysis_lookback"
    )

    # Main execution
    if st.sidebar.button("Analyze", type="primary", key="analyze_btn"):
        with st.spinner("Fetching and analyzing data..."):
            # Fetch data based on source
            if data_source == "Yahoo Finance":
                df = get_yahoo_data(symbol, timeframe, lookback_days)
            else:  # MetaTrader 5
                df = get_mt5_data(symbol, timeframe, lookback_days)
            
            if df is not None and not df.empty:
                # Calculate Camarilla Pivots
                df_camarilla = calculate_camarilla_pivots(df)
                
                if df_camarilla is not None and not df_camarilla.empty:
                    # Store in session state for use in backtesting
                    st.session_state['analysis_data'] = df_camarilla
                    st.session_state['symbol'] = symbol
                    
                    st.success(f"Successfully loaded {len(df_camarilla)} bars of data")
                    
                    # Display summary statistics
                    st.header("📈 Camarilla Pivot Analysis")
                    
                    # Create metrics display
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("📅 Year (365 days)")
                    with col2:
                        st.subheader("📅 Month (30 days)")
                    with col3:
                        st.subheader("📅 Week (7 days)")
                    
                    # Gap High - H4
                    st.markdown("### 🔴 Average Gap: High vs H4")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        year_h4 = calculate_period_stats(df_camarilla, 365, 'Gap_High_H4')
                        if year_h4 is not None:
                            st.metric("Year Average", f"{year_h4:.5f}")
                        else:
                            st.info("Insufficient data")
                    
                    with col2:
                        month_h4 = calculate_period_stats(df_camarilla, 30, 'Gap_High_H4')
                        if month_h4 is not None:
                            st.metric("Month Average", f"{month_h4:.5f}")
                        else:
                            st.info("Insufficient data")
                    
                    with col3:
                        week_h4 = calculate_period_stats(df_camarilla, 7, 'Gap_High_H4')
                        if week_h4 is not None:
                            st.metric("Week Average", f"{week_h4:.5f}")
                        else:
                            st.info("Insufficient data")
                    
                    st.markdown("---")
                    
                    # Gap High - H3
                    st.markdown("### 🟠 Average Gap: High vs H3")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        year_h3 = calculate_period_stats(df_camarilla, 365, 'Gap_High_H3')
                        if year_h3 is not None:
                            st.metric("Year Average", f"{year_h3:.5f}")
                        else:
                            st.info("Insufficient data")
                    
                    with col2:
                        month_h3 = calculate_period_stats(df_camarilla, 30, 'Gap_High_H3')
                        if month_h3 is not None:
                            st.metric("Month Average", f"{month_h3:.5f}")
                        else:
                            st.info("Insufficient data")
                    
                    with col3:
                        week_h3 = calculate_period_stats(df_camarilla, 7, 'Gap_High_H3')
                        if week_h3 is not None:
                            st.metric("Week Average", f"{week_h3:.5f}")
                        else:
                            st.info("Insufficient data")
                    
                    st.markdown("---")
                    
                    # Gap Low - L4
                    st.markdown("### 🔵 Average Gap: Low vs L4")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        year_l4 = calculate_period_stats(df_camarilla, 365, 'Gap_Low_L4')
                        if year_l4 is not None:
                            st.metric("Year Average", f"{year_l4:.5f}")
                        else:
                            st.info("Insufficient data")
                    
                    with col2:
                        month_l4 = calculate_period_stats(df_camarilla, 30, 'Gap_Low_L4')
                        if month_l4 is not None:
                            st.metric("Month Average", f"{month_l4:.5f}")
                        else:
                            st.info("Insufficient data")
                    
                    with col3:
                        week_l4 = calculate_period_stats(df_camarilla, 7, 'Gap_Low_L4')
                        if week_l4 is not None:
                            st.metric("Week Average", f"{week_l4:.5f}")
                        else:
                            st.info("Insufficient data")
                    
                    st.markdown("---")
                    
                    # Gap Low - L3
                    st.markdown("### 🟢 Average Gap: Low vs L3")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        year_l3 = calculate_period_stats(df_camarilla, 365, 'Gap_Low_L3')
                        if year_l3 is not None:
                            st.metric("Year Average", f"{year_l3:.5f}")
                        else:
                            st.info("Insufficient data")
                    
                    with col2:
                        month_l3 = calculate_period_stats(df_camarilla, 30, 'Gap_Low_L3')
                        if month_l3 is not None:
                            st.metric("Month Average", f"{month_l3:.5f}")
                        else:
                            st.info("Insufficient data")
                    
                    with col3:
                        week_l3 = calculate_period_stats(df_camarilla, 7, 'Gap_Low_L3')
                        if week_l3 is not None:
                            st.metric("Week Average", f"{week_l3:.5f}")
                        else:
                            st.info("Insufficient data")
                    
                    st.markdown("---")
                    
                    # Display detailed data table
                    st.header("📊 Detailed Data Table")
                    
                    # Select relevant columns for display
                    display_cols = ['Time', 'Open', 'High', 'Low', 'Close', 
                                   'H4', 'H3', 'L3', 'L4',
                                   'Gap_High_H4', 'Gap_High_H3', 'Gap_Low_L4', 'Gap_Low_L3']
                    
                    # Show latest 100 rows
                    st.dataframe(
                        df_camarilla[display_cols].tail(100).sort_values('Time', ascending=False),
                        width='stretch',
                        height=400
                    )
                    
                    # Download option
                    st.header("💾 Download Data")
                    csv = df_camarilla.to_csv(index=False)
                    st.download_button(
                        label="Download Full Dataset as CSV",
                        data=csv,
                        file_name=f"camarilla_analysis_{symbol}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                    # Summary Statistics
                    st.header("📉 Summary Statistics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Resistance Levels (High vs H3/H4)")
                        summary_resistance = pd.DataFrame({
                            'Metric': ['High vs H4', 'High vs H3'],
                            'Mean': [df_camarilla['Gap_High_H4'].mean(), df_camarilla['Gap_High_H3'].mean()],
                            'Std Dev': [df_camarilla['Gap_High_H4'].std(), df_camarilla['Gap_High_H3'].std()],
                            'Min': [df_camarilla['Gap_High_H4'].min(), df_camarilla['Gap_High_H3'].min()],
                            'Max': [df_camarilla['Gap_High_H4'].max(), df_camarilla['Gap_High_H3'].max()]
                        })
                        st.dataframe(summary_resistance, width='stretch')
                    
                    with col2:
                        st.subheader("Support Levels (Low vs L3/L4)")
                        summary_support = pd.DataFrame({
                            'Metric': ['Low vs L4', 'Low vs L3'],
                            'Mean': [df_camarilla['Gap_Low_L4'].mean(), df_camarilla['Gap_Low_L3'].mean()],
                            'Std Dev': [df_camarilla['Gap_Low_L4'].std(), df_camarilla['Gap_Low_L3'].std()],
                            'Min': [df_camarilla['Gap_Low_L4'].min(), df_camarilla['Gap_Low_L3'].min()],
                            'Max': [df_camarilla['Gap_Low_L4'].max(), df_camarilla['Gap_Low_L3'].max()]
                        })
                        st.dataframe(summary_support, width='stretch')
                else:
                    st.error("Failed to calculate Camarilla pivots. Please check your data.")
            else:
                st.error("Failed to fetch data. Please check your inputs.")

# Tab 2: Backtesting
with tab2:
    st.header("🔄 Strategy Backtesting")
    st.markdown("Test various Camarilla-based trading scenarios with defined entry, stoploss, and profit targets.")
    
    # Backtest Configuration
    st.sidebar.header("Backtest Configuration")
    
    bt_data_sources = ["Yahoo Finance"]
    if MT5_AVAILABLE:
        bt_data_sources.append("MetaTrader 5")
    
    bt_data_source = st.sidebar.selectbox(
        "Select Data Source",
        bt_data_sources,
        key="bt_source"
    )
    
    if bt_data_source == "Yahoo Finance":
        bt_symbol = st.sidebar.text_input(
            "Enter Symbol",
            value="AAPL",
            key="bt_symbol"
        )
        bt_timeframe = st.sidebar.selectbox(
            "Select Timeframe",
            ["1d", "1h", "4h"],
            key="bt_timeframe"
        )
    else:
        bt_symbol = st.sidebar.text_input(
            "Enter Symbol",
            value="EURUSD",
            key="bt_symbol_mt5"
        )
        bt_timeframe_options = {
            "1D": mt5.TIMEFRAME_D1 if MT5_AVAILABLE else None,
            "4H": mt5.TIMEFRAME_H4 if MT5_AVAILABLE else None,
            "1H": mt5.TIMEFRAME_H1 if MT5_AVAILABLE else None
        }
        bt_timeframe_display = st.sidebar.selectbox(
            "Select Timeframe",
            list(bt_timeframe_options.keys()),
            key="bt_timeframe_mt5"
        )
        bt_timeframe = bt_timeframe_options[bt_timeframe_display]
    
    bt_initial_capital = st.sidebar.number_input(
        "Initial Capital ($)",
        min_value=10000,
        max_value=1000000,
        value=100000,
        step=10000,
        key="bt_capital"
    )
    
    bt_profit_target = st.sidebar.number_input(
        "Profit Target (%)",
        min_value=1.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        key="bt_target"
    )
    
    # Backtest period selection
    backtest_periods = {
        "5 Years": 1825,
        "3 Years": 1095,
        "1 Year": 365,
        "3 Months": 90,
        "1 Month": 30
    }
    
    selected_period = st.sidebar.selectbox(
        "Backtest Period",
        list(backtest_periods.keys()),
        index=1,  # Default to 3 Years
        key="bt_period"
    )
    
    bt_lookback_days = backtest_periods[selected_period]
    
    # Dollar per tick selection
    dollar_per_tick = st.sidebar.selectbox(
        "Dollar per Tick ($)",
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        index=0,  # Default to $1
        key="bt_tick_value",
        help="Multiplier for P&L calculation. Higher values = more aggressive position sizing"
    )
    
    st.sidebar.info(f"Period: {selected_period} ({bt_lookback_days} days)\nTick Value: ${dollar_per_tick}/tick")
    
    # Scenario selection
    scenarios_config = {
        "Scenario 1: L3→H3 (Long)": {
            'entry_level': 'L3',
            'stoploss_gap': 'Gap_Low_L3',
            'target_level': 'H3',
            'trade_type': 'long',
            'description': 'Enter at L3, SL based on avg(L3-Low), Target H3'
        },
        "Scenario 2: L3→H4 (Long)": {
            'entry_level': 'L3',
            'stoploss_gap': 'Gap_Low_L3',
            'target_level': 'H4',
            'trade_type': 'long',
            'description': 'Enter at L3, SL based on avg(L3-Low), Target H4'
        },
        "Scenario 3: H3→L3 (Short)": {
            'entry_level': 'H3',
            'stoploss_gap': 'Gap_High_H3',
            'target_level': 'L3',
            'trade_type': 'short',
            'description': 'Enter at H3, SL based on avg(High-H3), Target L3'
        },
        "Scenario 4: H3→L4 (Short)": {
            'entry_level': 'H3',
            'stoploss_gap': 'Gap_High_H3',
            'target_level': 'L4',
            'trade_type': 'short',
            'description': 'Enter at H3, SL based on avg(High-H3), Target L4'
        },
        "Scenario 5: L4→H3 (Long)": {
            'entry_level': 'L4',
            'stoploss_gap': 'Gap_Low_L4',
            'target_level': 'H3',
            'trade_type': 'long',
            'description': 'Enter at L4, SL based on avg(L4-Low), Target H3'
        },
        "Scenario 6: L4→H4 (Long)": {
            'entry_level': 'L4',
            'stoploss_gap': 'Gap_Low_L4',
            'target_level': 'H4',
            'trade_type': 'long',
            'description': 'Enter at L4, SL based on avg(L4-Low), Target H4'
        },
        "Scenario 7: H4→L3 (Short)": {
            'entry_level': 'H4',
            'stoploss_gap': 'Gap_High_H4',
            'target_level': 'L3',
            'trade_type': 'short',
            'description': 'Enter at H4, SL based on avg(High-H4), Target L3'
        },
        "Scenario 8: H4→L4 (Short)": {
            'entry_level': 'H4',
            'stoploss_gap': 'Gap_High_H4',
            'target_level': 'L4',
            'trade_type': 'short',
            'description': 'Enter at H4, SL based on avg(High-H4), Target L4'
        }
    }
    
    selected_scenario = st.sidebar.selectbox(
        "Select Trading Scenario",
        list(scenarios_config.keys()),
        key="bt_scenario"
    )
    
    st.sidebar.info(scenarios_config[selected_scenario]['description'])
    
    # Run backtest button
    if st.sidebar.button("Run Backtest", type="primary", key="run_backtest"):
        with st.spinner("Running backtest..."):
            # Fetch data
            if bt_data_source == "Yahoo Finance":
                bt_df = get_yahoo_data(bt_symbol, bt_timeframe, bt_lookback_days)
            else:
                bt_df = get_mt5_data(bt_symbol, bt_timeframe, bt_lookback_days)
            
            if bt_df is not None and not bt_df.empty:
                # Calculate Camarilla Pivots
                bt_df_camarilla = calculate_camarilla_pivots(bt_df)
                
                if bt_df_camarilla is not None and not bt_df_camarilla.empty:
                    st.success(f"Loaded {len(bt_df_camarilla)} bars for backtesting")
                    
                    # Run backtest
                    scenario_cfg = scenarios_config[selected_scenario]
                    trades_df, capital_df, metrics = backtest_scenario(
                        bt_df_camarilla,
                        scenario_cfg,
                        initial_capital=bt_initial_capital,
                        profit_target_pct=bt_profit_target / 100,
                        dollar_per_tick=dollar_per_tick
                    )
                    
                    # Display results
                    st.header(f"📊 Results: {selected_scenario}")
                    
                    # Key Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Final Capital",
                            f"${metrics['final_capital']:,.2f}",
                            f"${metrics['net_profit']:,.2f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Total Trades",
                            f"{metrics['total_trades']}",
                            f"{metrics['win_rate']:.1f}% Win Rate"
                        )
                    
                    with col3:
                        st.metric(
                            "Winning Trades",
                            f"{metrics['winning_trades']}",
                            f"+${metrics['total_profit']:,.2f}"
                        )
                    
                    with col4:
                        st.metric(
                            "Losing Trades",
                            f"{metrics['losing_trades']}",
                            f"-${metrics['total_loss']:,.2f}"
                        )
                    
                    st.markdown("---")
                    
                    # Answer specific questions
                    st.subheader("📋 Analysis Questions")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### ✅ Days to Achieve Profit Target")
                        if metrics['days_to_target'] is not None:
                            st.success(f"Target of ${bt_initial_capital * bt_profit_target / 100:,.2f} achieved on: **{metrics['days_to_target'].strftime('%Y-%m-%d')}**")
                            
                            # Calculate number of trading days
                            if len(trades_df) > 0:
                                first_trade = trades_df.iloc[0]['Date']
                                trading_days = (metrics['days_to_target'] - first_trade).days
                                st.info(f"**{trading_days} trading days** from first trade")
                        else:
                            st.warning("Profit target not achieved during backtest period")
                        
                        st.markdown("#### 💰 Ideal Stoploss Comparison")
                        
                        # Simple Method (Original)
                        if metrics['ideal_stoploss'] is not None:
                            st.info(f"**Simple Method (Winners Avg):** {metrics['ideal_stoploss']:.5f}")
                        else:
                            st.warning("Simple Method: No profitable trades")
                        
                        # Enhanced Method
                        if metrics['ideal_stoploss_enhanced'] is not None:
                            st.success(f"**Enhanced Method (Recommended):** {metrics['ideal_stoploss_enhanced']:.5f}")
                            
                            # Show breakdown with expander
                            with st.expander("📊 See Calculation Breakdown"):
                                col_a, col_b, col_c = st.columns(3)
                                
                                with col_a:
                                    st.metric("Winners Average", 
                                             f"{metrics['ideal_stoploss_winners']:.5f}" if metrics['ideal_stoploss_winners'] else "N/A",
                                             help="Average SL from winning trades (50% weight)")
                                
                                with col_b:
                                    st.metric("Losers Adjusted", 
                                             f"{metrics['ideal_stoploss_losers_adj']:.5f}" if metrics['ideal_stoploss_losers_adj'] else "N/A",
                                             help="120% of losers' SL to potentially save them (30% weight)")
                                
                                with col_c:
                                    st.metric("Volatility Adjusted", 
                                             f"{metrics['ideal_stoploss_volatility']:.5f}" if metrics['ideal_stoploss_volatility'] else "N/A",
                                             help="Mean + 1 Std Dev for safety margin (20% weight)")
                                
                                st.markdown("""
                                **Enhanced Formula:**  
                                `Enhanced SL = (Winners × 50%) + (Losers Adj × 30%) + (Volatility × 20%)`
                                
                                **Benefits:**
                                - ✅ Considers winning trades (what worked)
                                - ✅ Accounts for losers (what didn't work)
                                - ✅ Adjusts for market volatility
                                - ✅ More robust than simple average
                                """)
                        else:
                            st.warning("Enhanced Method: Insufficient data")
                    
                    with col2:
                        st.markdown("#### ⚠️ Days Below $90,000")
                        if metrics['days_below_90k'] > 0:
                            st.error(f"Capital fell below $90,000 on **{metrics['days_below_90k']} days**")
                            
                            # Show when it first happened
                            if len(trades_df) > 0:
                                below_90k = trades_df[trades_df['Capital'] < 90000]
                                if len(below_90k) > 0:
                                    first_below = below_90k.iloc[0]['Date']
                                    st.warning(f"First occurred on: {first_below.strftime('%Y-%m-%d')}")
                        else:
                            st.success("Capital never fell below $90,000! 🎉")
                        
                        st.markdown("#### 📉 Days with >5% Loss")
                        if metrics['days_below_5pct_loss'] > 0:
                            st.error(f"Daily loss exceeded 5% on **{metrics['days_below_5pct_loss']} days**")
                            
                            # Show worst loss day
                            if len(trades_df) > 0:
                                worst_day = trades_df[trades_df['Daily_Loss_Pct'] <= -5].nsmallest(1, 'Daily_Loss_Pct')
                                if len(worst_day) > 0:
                                    worst_loss = worst_day.iloc[0]['Daily_Loss_Pct']
                                    worst_date = worst_day.iloc[0]['Date']
                                    st.warning(f"Worst day: {worst_loss:.2f}% on {worst_date.strftime('%Y-%m-%d')}")
                        else:
                            st.success("No single day had a loss exceeding 5%! 🎉")
                    
                    st.markdown("---")
                    
                    # Equity Curve
                    st.subheader("📈 Equity Curve")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=capital_df['Date'],
                        y=capital_df['Capital'],
                        mode='lines',
                        name='Capital',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Add horizontal lines
                    fig.add_hline(y=bt_initial_capital, line_dash="dash", line_color="green", 
                                  annotation_text="Initial Capital")
                    fig.add_hline(y=90000, line_dash="dash", line_color="red",
                                  annotation_text="$90,000 Threshold")
                    
                    if metrics['days_to_target']:
                        target_amount = bt_initial_capital + (bt_initial_capital * bt_profit_target / 100)
                        fig.add_hline(y=target_amount, line_dash="dash", line_color="gold",
                                      annotation_text="Profit Target")
                    
                    fig.update_layout(
                        title="Capital Over Time",
                        xaxis_title="Date",
                        yaxis_title="Capital ($)",
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # Trade Details
                    st.subheader("📝 Trade Log")
                    
                    if len(trades_df) > 0:
                        # Format the trades dataframe for display
                        display_trades = trades_df.copy()
                        display_trades['Date'] = display_trades['Date'].dt.strftime('%Y-%m-%d')
                        display_trades['Entry'] = display_trades['Entry'].round(5)
                        display_trades['Exit'] = display_trades['Exit'].round(5)
                        display_trades['PnL'] = display_trades['PnL'].round(2)
                        display_trades['Capital'] = display_trades['Capital'].round(2)
                        display_trades['Daily_Loss_Pct'] = display_trades['Daily_Loss_Pct'].round(2)
                        
                        st.dataframe(
                            display_trades[['Date', 'Type', 'Entry', 'Exit', 'PnL', 'Capital', 
                                          'Daily_Loss_Pct', 'Hit_Target', 'Hit_Stoploss']],
                            width='stretch',
                            height=400
                        )
                        
                        # Download trades
                        csv_trades = trades_df.to_csv(index=False)
                        st.download_button(
                            label="Download Trade Log as CSV",
                            data=csv_trades,
                            file_name=f"backtest_{selected_scenario.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No trades executed during backtest period")
                    
                    # Performance Summary
                    st.subheader("📊 Performance Summary")
                    
                    summary_data = {
                        'Metric': [
                            'Backtest Period',
                            'Dollar per Tick',
                            'Total Trades',
                            'Winning Trades',
                            'Losing Trades',
                            'Win Rate',
                            'Total Profit',
                            'Total Loss',
                            'Net Profit',
                            'ROI',
                            'Final Capital',
                            'Days Below $90k',
                            'Days with >5% Loss',
                            '─── Stoploss Analysis ───',
                            'Simple SL (Winners Avg)',
                            'Enhanced SL (Recommended)',
                            'Winners Component',
                            'Losers Adjusted Component',
                            'Volatility Component'
                        ],
                        'Value': [
                            f"{selected_period} ({bt_lookback_days} days)",
                            f"${dollar_per_tick}/tick",
                            str(metrics['total_trades']),
                            str(metrics['winning_trades']),
                            str(metrics['losing_trades']),
                            f"{metrics['win_rate']:.2f}%",
                            f"${metrics['total_profit']:,.2f}",
                            f"-${metrics['total_loss']:,.2f}",
                            f"${metrics['net_profit']:,.2f}",
                            f"{(metrics['net_profit'] / bt_initial_capital * 100):.2f}%",
                            f"${metrics['final_capital']:,.2f}",
                            str(metrics['days_below_90k']),
                            str(metrics['days_below_5pct_loss']),
                            '──────────────',
                            f"{metrics['ideal_stoploss']:.5f}" if metrics['ideal_stoploss'] else "N/A",
                            f"{metrics['ideal_stoploss_enhanced']:.5f}" if metrics['ideal_stoploss_enhanced'] else "N/A",
                            f"{metrics['ideal_stoploss_winners']:.5f}" if metrics['ideal_stoploss_winners'] else "N/A",
                            f"{metrics['ideal_stoploss_losers_adj']:.5f}" if metrics['ideal_stoploss_losers_adj'] else "N/A",
                            f"{metrics['ideal_stoploss_volatility']:.5f}" if metrics['ideal_stoploss_volatility'] else "N/A"
                        ]
                    }
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, width='stretch', hide_index=True)
                    
                else:
                    st.error("Failed to calculate Camarilla pivots for backtest")
            else:
                st.error("Failed to fetch data for backtesting")

# Tab 3: Best Scenario Finder
with tab3:
    st.header("🏆 Best Scenario Finder")
    st.markdown("Automatically test all 8 scenarios and find the best performer based on multiple criteria.")
    
    # Configuration
    st.sidebar.header("Scenario Finder Config")
    
    finder_data_sources = ["Yahoo Finance"]
    if MT5_AVAILABLE:
        finder_data_sources.append("MetaTrader 5")
    
    finder_data_source = st.sidebar.selectbox(
        "Select Data Source",
        finder_data_sources,
        key="finder_source"
    )
    
    if finder_data_source == "Yahoo Finance":
        finder_symbol = st.sidebar.text_input(
            "Enter Symbol",
            value="AAPL",
            key="finder_symbol"
        )
        finder_timeframe = st.sidebar.selectbox(
            "Select Timeframe",
            ["1d", "1h", "4h"],
            key="finder_timeframe"
        )
    else:
        finder_symbol = st.sidebar.text_input(
            "Enter Symbol",
            value="EURUSD",
            key="finder_symbol_mt5"
        )
        finder_timeframe_options = {
            "1D": mt5.TIMEFRAME_D1 if MT5_AVAILABLE else None,
            "4H": mt5.TIMEFRAME_H4 if MT5_AVAILABLE else None,
            "1H": mt5.TIMEFRAME_H1 if MT5_AVAILABLE else None
        }
        finder_timeframe_display = st.sidebar.selectbox(
            "Select Timeframe",
            list(finder_timeframe_options.keys()),
            key="finder_timeframe_mt5"
        )
        finder_timeframe = finder_timeframe_options[finder_timeframe_display]
    
    finder_initial_capital = st.sidebar.number_input(
        "Initial Capital ($)",
        min_value=10000,
        max_value=1000000,
        value=100000,
        step=10000,
        key="finder_capital"
    )
    
    finder_profit_target = st.sidebar.number_input(
        "Profit Target (%)",
        min_value=1.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        key="finder_target"
    )
    
    # Use 3 months as recommended period
    finder_lookback_days = 90  # 3 months
    st.sidebar.info("📅 Using 3-Month Period (Recommended for live trading)")
    
    finder_dollar_per_tick = st.sidebar.selectbox(
        "Dollar per Tick ($)",
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        index=0,
        key="finder_tick_value"
    )
    
    # All scenarios config (same as before)
    all_scenarios_config = {
        "Scenario 1: L3→H3 (Long)": {
            'entry_level': 'L3',
            'stoploss_gap': 'Gap_Low_L3',
            'target_level': 'H3',
            'trade_type': 'long',
            'description': 'Enter at L3, SL based on avg(L3-Low), Target H3'
        },
        "Scenario 2: L3→H4 (Long)": {
            'entry_level': 'L3',
            'stoploss_gap': 'Gap_Low_L3',
            'target_level': 'H4',
            'trade_type': 'long',
            'description': 'Enter at L3, SL based on avg(L3-Low), Target H4'
        },
        "Scenario 3: H3→L3 (Short)": {
            'entry_level': 'H3',
            'stoploss_gap': 'Gap_High_H3',
            'target_level': 'L3',
            'trade_type': 'short',
            'description': 'Enter at H3, SL based on avg(High-H3), Target L3'
        },
        "Scenario 4: H3→L4 (Short)": {
            'entry_level': 'H3',
            'stoploss_gap': 'Gap_High_H3',
            'target_level': 'L4',
            'trade_type': 'short',
            'description': 'Enter at H3, SL based on avg(High-H3), Target L4'
        },
        "Scenario 5: L4→H3 (Long)": {
            'entry_level': 'L4',
            'stoploss_gap': 'Gap_Low_L4',
            'target_level': 'H3',
            'trade_type': 'long',
            'description': 'Enter at L4, SL based on avg(L4-Low), Target H3'
        },
        "Scenario 6: L4→H4 (Long)": {
            'entry_level': 'L4',
            'stoploss_gap': 'Gap_Low_L4',
            'target_level': 'H4',
            'trade_type': 'long',
            'description': 'Enter at L4, SL based on avg(L4-Low), Target H4'
        },
        "Scenario 7: H4→L3 (Short)": {
            'entry_level': 'H4',
            'stoploss_gap': 'Gap_High_H4',
            'target_level': 'L3',
            'trade_type': 'short',
            'description': 'Enter at H4, SL based on avg(High-H4), Target L3'
        },
        "Scenario 8: H4→L4 (Short)": {
            'entry_level': 'H4',
            'stoploss_gap': 'Gap_High_H4',
            'target_level': 'L4',
            'trade_type': 'short',
            'description': 'Enter at H4, SL based on avg(High-H4), Target L4'
        }
    }
    
    if st.sidebar.button("🔍 Find Best Scenario", type="primary", key="find_best"):
        with st.spinner("Testing all 8 scenarios... This may take a minute..."):
            # Fetch data
            if finder_data_source == "Yahoo Finance":
                finder_df = get_yahoo_data(finder_symbol, finder_timeframe, finder_lookback_days)
            else:
                finder_df = get_mt5_data(finder_symbol, finder_timeframe, finder_lookback_days)
            
            if finder_df is not None and not finder_df.empty:
                # Calculate Camarilla Pivots
                finder_df_camarilla = calculate_camarilla_pivots(finder_df)
                
                if finder_df_camarilla is not None and not finder_df_camarilla.empty:
                    st.success(f"Data loaded: {len(finder_df_camarilla)} bars")
                    
                    # Run all scenarios
                    all_results = {}
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, (scenario_name, scenario_cfg) in enumerate(all_scenarios_config.items()):
                        status_text.text(f"Testing {scenario_name}...")
                        
                        trades_df, capital_df, metrics = backtest_scenario(
                            finder_df_camarilla,
                            scenario_cfg,
                            initial_capital=finder_initial_capital,
                            profit_target_pct=finder_profit_target / 100,
                            dollar_per_tick=finder_dollar_per_tick
                        )
                        
                        all_results[scenario_name] = metrics
                        progress_bar.progress((idx + 1) / len(all_scenarios_config))
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Rank scenarios
                    rankings_df = rank_scenarios(all_results, finder_initial_capital, finder_profit_target / 100)
                    
                    if rankings_df is not None and len(rankings_df) > 0:
                        st.success("✅ Analysis Complete!")
                        
                        # Display winner
                        st.markdown("---")
                        winner = rankings_df.iloc[0]
                        
                        st.markdown("## 🥇 WINNER: Best Overall Scenario")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "🏆 Top Scenario",
                                winner['Scenario'].replace('Scenario ', 'S'),
                                f"Score: {winner['Composite Score']}/100"
                            )
                        
                        with col2:
                            st.metric(
                                "💰 ROI",
                                f"{winner['ROI (%)']}%",
                                f"${winner['Net Profit']:,.0f}"
                            )
                        
                        with col3:
                            st.metric(
                                "📅 Days to Target",
                                f"{winner['Days to Target']}" if winner['Days to Target'] != 'N/A' else "Not Reached",
                                "From first trade"
                            )
                        
                        with col4:
                            st.metric(
                                "✅ Win Rate",
                                f"{winner['Win Rate (%)']}%",
                                f"{winner['Total Trades']} trades"
                            )
                        
                        # Winner details
                        st.markdown("### 📊 Winner Details")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**✅ Strengths:**")
                            st.info(f"""
                            - Enhanced SL: **{winner['Enhanced SL']}**
                            - Profit Factor: **{winner['Profit Factor']}**
                            - Days <$90k: **{winner['Days <$90k']}**
                            - Days >5% Loss: **{winner['Days >5% Loss']}**
                            """)
                        
                        with col2:
                            st.markdown("**📈 Performance Scores:**")
                            st.success(f"""
                            - ROI Score: **{winner['ROI Score']}/100**
                            - Days Score: **{winner['Days Score']}/100**
                            - Risk Score: **{winner['Risk Score']}/100**
                            - Win Rate Score: **{winner['Win Rate Score']}/100**
                            - Profit Factor Score: **{winner['PF Score']}/100**
                            """)
                        
                        st.markdown("---")
                        
                        # Full rankings table
                        st.subheader("📋 All Scenarios Ranked")
                        
                        # Display key columns
                        display_cols = ['Scenario', 'Composite Score', 'ROI (%)', 'Days to Target', 
                                       'Win Rate (%)', 'Net Profit', 'Days <$90k', 'Days >5% Loss', 'Enhanced SL']
                        
                        st.dataframe(
                            rankings_df[display_cols],
                            width='stretch',
                            height=400
                        )
                        
                        # Download rankings
                        csv_rankings = rankings_df.to_csv(index=True)
                        st.download_button(
                            label="📥 Download Full Rankings CSV",
                            data=csv_rankings,
                            file_name=f"scenario_rankings_{finder_symbol}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                        
                        st.markdown("---")
                        
                        # Scoring methodology
                        with st.expander("ℹ️ How Scenarios Are Scored"):
                            st.markdown("""
                            ### Composite Score Calculation
                            
                            Each scenario is evaluated across 5 key dimensions:
                            
                            **1. ROI Score (30% weight)**
                            - Measures return on investment
                            - Higher ROI = Higher score
                            - Scale: 50% ROI = 100 points
                            
                            **2. Days to Target Score (25% weight)**
                            - How quickly profit target is reached
                            - Fewer days = Higher score
                            - Scale: 0 days = 100 points, 90 days = 0 points
                            
                            **3. Risk Score (20% weight)**
                            - Based on days below $90k and days with >5% loss
                            - Lower risk = Higher score
                            - Penalties: 5 points per day <$90k, 3 points per day >5% loss
                            
                            **4. Win Rate Score (15% weight)**
                            - Percentage of winning trades
                            - Direct mapping (65% win rate = 65 points)
                            
                            **5. Profit Factor Score (10% weight)**
                            - Total wins / Total losses
                            - Higher PF = Higher score
                            - Scale: 2.5 PF = 100 points
                            
                            **Final Composite Score:**
                            ```
                            Score = (ROI × 0.30) + (Days × 0.25) + (Risk × 0.20) + 
                                   (WinRate × 0.15) + (PF × 0.10)
                            ```
                            
                            **Why This Matters:**
                            - Top-ranked scenario balances all factors
                            - Not just highest ROI (might be too risky)
                            - Not just fastest (might have low ROI)
                            - **Best overall risk-adjusted performance**
                            """)
                        
                        # Recommendation
                        st.markdown("---")
                        st.markdown("### 💡 Recommendation")
                        
                        st.success(f"""
                        **For Live Trading on {finder_symbol}:**
                        
                        ✅ Use **{winner['Scenario']}**
                        
                        **Settings:**
                        - Entry: As per scenario definition
                        - Stoploss: Use 3-Month Enhanced SL = **{winner['Enhanced SL']}**
                        - Target: As per scenario definition
                        - Position Size: ${finder_dollar_per_tick}/tick
                        
                        **Expected Performance:**
                        - ROI: ~{winner['ROI (%)']}% over 3 months
                        - Win Rate: ~{winner['Win Rate (%)']}%
                        - Days to {finder_profit_target}% profit: ~{winner['Days to Target']} days
                        
                        **Risk Profile:**
                        - Days below $90k: {winner['Days <$90k']} ({"Low" if winner['Days <$90k'] < 3 else "Moderate" if winner['Days <$90k'] < 7 else "High"} risk)
                        - Days with >5% loss: {winner['Days >5% Loss']} ({"Excellent" if winner['Days >5% Loss'] == 0 else "Good" if winner['Days >5% Loss'] < 3 else "Fair"})
                        
                        ⚠️ **Remember:** Past performance doesn't guarantee future results. Always start with small position sizes and paper trade first!
                        """)
                        
                    else:
                        st.error("No valid scenarios to rank. All scenarios may have failed.")
                else:
                    st.error("Failed to calculate Camarilla pivots")
            else:
                st.error("Failed to fetch data")

# Information section
with st.sidebar:
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Camarilla Pivot Points** are advanced support and resistance levels used in technical analysis.
    
    **Formula:**
    - H4 = Close + ((High - Low) × 1.1/2)
    - H3 = Close + ((High - Low) × 1.1/4)
    - L3 = Close - ((High - Low) × 1.1/4)
    - L4 = Close - ((High - Low) × 1.1/2)
    
    **Gap Metrics:**
    - Positive gap: Price exceeded the level
    - Negative gap: Price didn't reach the level
    """)
    
    st.markdown("---")
    st.markdown("### 📚 Tips")
    st.markdown("""
    - Use **H3/H4** as resistance levels
    - Use **L3/L4** as support levels
    - Larger gaps indicate stronger trends
    - Compare different timeframes for better insights
    - Test multiple scenarios before live trading
    """)
