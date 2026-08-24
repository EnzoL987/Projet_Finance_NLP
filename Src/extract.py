import yfinance as yf
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configure basic logging to track the extraction process in the terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API key for NewsAPI (Replace with your actual key if needed)
NEWS_API_KEY = "6bef262549ab4b92a72ab2642be1d7c0" 

def fetch_market_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """
    Fetches historical OHLCV market data for a given financial asset.
    
    Args:
        ticker (str): The stock symbol (e.g., 'AAPL').
        period (str): The time period to download.
        
    Returns:
        pd.DataFrame: A clean DataFrame containing Open, High, Low, Close, and Volume.
    """
    logging.info(f"Starting market data extraction for {ticker} (Period: {period})")
    
    try:
        asset = yf.Ticker(ticker)
        df = asset.history(period=period)
        
        if df.empty:
            logging.warning(f"No market data retrieved for {ticker}.")
            return pd.DataFrame()
            
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        df_clean = df[required_cols].copy()
        
        # Standardize timezone to avoid SQL join issues later
        if isinstance(df_clean.index, pd.DatetimeIndex):
            df_clean.index = df_clean.index.tz_convert(None)
            
        logging.info(f"Successfully extracted {len(df_clean)} trading days for {ticker}.")
        return df_clean
        
    except Exception as e:
        logging.error(f"Error fetching market data for {ticker}: {e}")
        return pd.DataFrame()

def fetch_financial_news(query: str, days_back: int = 7) -> pd.DataFrame:
    """
    Fetches recent news articles related to a specific company.
    
    Args:
        query (str): The search term (e.g., "Apple OR AAPL").
        days_back (int): Number of days to look back.
        
    Returns:
        pd.DataFrame: Dataframe with Date, Headline, and Summary.
    """
    logging.info(f"Starting news extraction for query: '{query}' (Last {days_back} days)")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': query,
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'language': 'en',
        'sortBy': 'relevancy',
        'apiKey': NEWS_API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            extracted_data = []
            
            for item in articles:
                if item.get('title') and item.get('description'):
                    extracted_data.append({
                        'Date': item['publishedAt'][:10],
                        'Headline': item['title'],
                        'Summary': item['description']
                    })
                    
            df_news = pd.DataFrame(extracted_data)
            logging.info(f"Successfully extracted {len(df_news)} articles for '{query}'.")
            return df_news
        else:
            logging.error(f"NewsAPI Error: {response.status_code} - {response.text}")
            return pd.DataFrame()
            
    except Exception as e:
        logging.error(f"Network error during news extraction: {e}")
        return pd.DataFrame()

# ==========================================
# TEST BLOCK
# ==========================================
if __name__ == "__main__":
    # This block only runs if you execute THIS script directly.
    # It will NOT run when we import these functions into another file later.
    print("--- Testing Extraction Module ---")
    
    test_market = fetch_market_data("AAPL", period="1mo")
    print(f"Market Data shape: {test_market.shape}")
    
    test_news = fetch_financial_news("Apple OR AAPL", days_back=3)
    print(f"News Data shape: {test_news.shape}")