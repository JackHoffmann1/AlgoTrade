import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
import streamlit as st

# Function to fetch historical stock data
def get_data(tickerSymbol, data_period):
    try:
        # Fetch historical data
        tickerData = yf.Ticker(tickerSymbol)
        df = tickerData.history(period=data_period)
        return df
    except Exception as e:
        st.error(f"An error occurred while fetching data: {e}")
        return pd.DataFrame()

# Function to calculate moving averages
def calculate_moving_averages(df, short_window, long_window):
    # Calculate moving averages
    df['SMA_Short'] = df['Close'].rolling(window=short_window, min_periods=1).mean()
    df['SMA_Long'] = df['Close'].rolling(window=long_window, min_periods=1).mean()
    return df

# Function to generate signals for Moving Average Crossover
def generate_ma_signals(df, short_window):
    # Generate signals
    df['Signal'] = 0.0
    df['Signal'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1.0, 0.0)
    df['Positions'] = df['Signal'].diff()
    return df

# Function to generate signals for RSI Strategy
def generate_rsi_signals(df, rsi_period, rsi_overbought, rsi_oversold):
    # Calculate RSI
    df['RSI'] = talib.RSI(df['Close'], timeperiod=rsi_period)
    df['Signal'] = 0.0
    df['Signal'] = np.where(df['RSI'] < rsi_oversold, 1.0, df['Signal'])
    df['Signal'] = np.where(df['RSI'] > rsi_overbought, -1.0, df['Signal'])
    df['Positions'] = df['Signal'].diff()
    return df

# Function to backtest the strategy
def backtest_strategy(df):
    # Backtest the strategy
    df['Market Returns'] = df['Close'].pct_change()
    df['Strategy Returns'] = df['Market Returns'] * df['Signal'].shift(1)
    df['Cumulative Market Returns'] = (1 + df['Market Returns']).cumprod()
    df['Cumulative Strategy Returns'] = (1 + df['Strategy Returns']).cumprod()
    return df

def main():
    st.title("Algorithmic Trading App")
    st.write("""
    This app allows you to perform backtesting on stock trading strategies.
    Select a stock, choose a strategy, and customize parameters to see the results.
    """)

    # Sidebar for inputs
    st.sidebar.header('User Input Parameters')

    tickerSymbol = st.sidebar.text_input("Ticker Symbol (e.g., AAPL)", "AAPL").upper()
    data_period = st.sidebar.selectbox("Historical Data Period", ['1y', '6mo', '3mo', '1mo', '5d'])

    strategy_choice = st.sidebar.selectbox("Select a Strategy", ['Moving Average Crossover', 'RSI Overbought/Oversold'])

    if strategy_choice == 'Moving Average Crossover':
        short_window = st.sidebar.number_input("Short-term Moving Average Window", min_value=1, value=20, step=1)
        long_window = st.sidebar.number_input("Long-term Moving Average Window", min_value=1, value=50, step=1)
        if short_window >= long_window:
            st.error("Error: Short-term window must be less than long-term window.")
            return
    else:
        rsi_period = st.sidebar.number_input("RSI Period", min_value=1, value=14, step=1)
        rsi_overbought = st.sidebar.slider("RSI Overbought Level", min_value=50, max_value=100, value=70)
        rsi_oversold = st.sidebar.slider("RSI Oversold Level", min_value=0, max_value=50, value=30)
        if rsi_overbought <= rsi_oversold:
            st.error("Error: Overbought level must be greater than oversold level.")
            return

    # Fetch data
    df = get_data(tickerSymbol, data_period)
    if df.empty:
        st.error(f"No data found for {tickerSymbol}. Please check the ticker symbol and try again.")
        return

    # Strategy execution
    if strategy_choice == 'Moving Average Crossover':
        # Calculate moving averages
        df = calculate_moving_averages(df, short_window, long_window)
        # Generate signals
        df = generate_ma_signals(df, short_window)
        # Plot signals
        st.subheader('Price Chart with Moving Averages and Signals')
        fig, ax = plt.subplots(figsize=(14,7))
        ax.plot(df.index, df['Close'], label='Close Price', alpha=0.5)
        ax.plot(df.index, df['SMA_Short'], label=f'SMA {short_window}', alpha=0.9)
        ax.plot(df.index, df['SMA_Long'], label=f'SMA {long_window}', alpha=0.9)
        ax.scatter(df.index[df['Positions'] == 1], df['Close'][df['Positions'] == 1],
                   label='Buy Signal', marker='^', color='green', s=100)
        ax.scatter(df.index[df['Positions'] == -1], df['Close'][df['Positions'] == -1],
                   label='Sell Signal', marker='v', color='red', s=100)
        ax.set_title(f'{tickerSymbol} - Moving Average Crossover Strategy')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid()
        st.pyplot(fig)
    else:
        # Generate RSI signals
        df = generate_rsi_signals(df, rsi_period, rsi_overbought, rsi_oversold)
        # Plot RSI signals
        st.subheader('Price and RSI Chart with Signals')
        fig, (ax1, ax2) = plt.subplots(2, figsize=(14,10), sharex=True)

        # Plot price
        ax1.plot(df.index, df['Close'], label='Close Price')
        ax1.set_title(f'{tickerSymbol} - Price Chart')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid()

        # Plot RSI
        ax2.plot(df.index, df['RSI'], label='RSI', color='orange')
        ax2.axhline(y=rsi_overbought, color='red', linestyle='--', label='Overbought Level')
        ax2.axhline(y=rsi_oversold, color='green', linestyle='--', label='Oversold Level')
        ax2.scatter(df.index[df['Positions'] == 2], df['RSI'][df['Positions'] == 2],
                    label='Buy Signal', marker='^', color='green', s=100)
        ax2.scatter(df.index[df['Positions'] == -2], df['RSI'][df['Positions'] == -2],
                    label='Sell Signal', marker='v', color='red', s=100)
        ax2.set_title(f'{tickerSymbol} - RSI ({rsi_period})')
        ax2.set_ylabel('RSI Value')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid()
        st.pyplot(fig)

    # Backtest strategy
    df = backtest_strategy(df)
    # Plot cumulative returns
    st.subheader('Cumulative Returns')
    fig2, ax3 = plt.subplots(figsize=(14,7))
    ax3.plot(df.index, df['Cumulative Market Returns'], label='Market Returns')
    ax3.plot(df.index, df['Cumulative Strategy Returns'], label='Strategy Returns')
    ax3.set_title(f'{tickerSymbol} - Cumulative Returns')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Cumulative Returns')
    ax3.legend()
    ax3.grid()
    st.pyplot(fig2)

    # Display performance metrics
    total_market_return = df['Cumulative Market Returns'].iloc[-1]
    total_strategy_return = df['Cumulative Strategy Returns'].iloc[-1]
    st.write(f"**Total Market Return over the period:** {(total_market_return - 1) * 100:.2f}%")
    st.write(f"**Total Strategy Return over the period:** {(total_strategy_return - 1) * 100:.2f}%")

if __name__ == "__main__":
    main()
