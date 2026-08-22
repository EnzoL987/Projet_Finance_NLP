import yfinance as yf
import pandas as pd
import logging

# Configure basic logging to track the pipeline's health
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_market_data(ticker: str, period: str = "1mo") -> pd.DataFrame:
    """
    Fetches historical OHLCV market data for a given financial asset.
    
    Args:
        ticker (str): The stock symbol (e.g., 'AAPL' for Apple).
        period (str): The time period to download (e.g., '1mo', '1y', 'max').
        
    Returns:
        pd.DataFrame: A clean DataFrame containing Open, High, Low, Close, and Volume.
                      Returns an empty DataFrame if the extraction fails.
    """
    logging.info(f"Starting data extraction for ticker: {ticker} over period: {period}")
    
    try:
        # Initialize the Ticker object
        asset = yf.Ticker(ticker)
        
        # Download historical data
        df = asset.history(period=period)
        
        # Check if the dataframe is empty (e.g., invalid ticker)
        if df.empty:
            logging.warning(f"No data retrieved for {ticker}. Please check the ticker symbol.")
            return pd.DataFrame()
            
        # Filter only the required columns for our ML project
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        df_clean = df[required_cols].copy()
        
        # Standardize the index (Date) to ensure timezone consistency across assets
        if isinstance(df_clean.index, pd.DatetimeIndex):
            df_clean.index = df_clean.index.tz_convert(None)
            
        logging.info(f"Successfully extracted {len(df_clean)} trading days for {ticker}.")
        return df_clean
        
    except Exception as e:
        # Catch and log any network or API errors to prevent pipeline crashes
        logging.error(f"An error occurred while fetching data for {ticker}: {e}")
        return pd.DataFrame()