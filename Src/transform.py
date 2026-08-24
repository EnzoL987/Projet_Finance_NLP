import pandas as pd
import numpy as np
from transformers import pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_financial_sentiment(df: pd.DataFrame, text_column: str = 'Headline') -> pd.DataFrame:
    """
    Applies the FinBERT NLP model to evaluate the sentiment of financial texts.
    
    Args:
        df (pd.DataFrame): The dataframe containing the news articles.
        text_column (str): The column name containing the text to analyze.
        
    Returns:
        pd.DataFrame: Dataframe enriched with 'Sentiment_Label' and 'Sentiment_Score'.
    """
    if df.empty:
        logging.warning("The news dataframe is empty. Skipping sentiment analysis.")
        return df
        
    logging.info("Loading FinBERT model... (This may take a moment)")
    
    try:
        # Load the pre-trained FinBERT model
        sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        
        texts = df[text_column].tolist()
        logging.info(f"Analyzing sentiment for {len(texts)} articles...")
        
        results = sentiment_analyzer(texts)
        
        labels = [res['label'] for res in results]
        scores = [res['score'] for res in results]
        
        df_result = df.copy()
        df_result['Sentiment_Label'] = labels
        df_result['Sentiment_Score'] = scores
        
        logging.info("Sentiment analysis completed successfully.")
        return df_result
        
    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}")
        return df

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates essential technical indicators: SMA, EMA, Volatility, and RSI.
    
    Args:
        df (pd.DataFrame): Financial dataset containing a 'Close' column.
        
    Returns:
        pd.DataFrame: A new dataframe enriched with technical features.
    """
    if df.empty or 'Close' not in df.columns:
        logging.warning("Invalid dataframe for technical indicators.")
        return df
        
    logging.info("Calculating technical indicators (SMA, EMA, Volatility, RSI)...")
    df_feat = df.copy()
    
    try:
        # 1. Moving Averages (20 days window)
        df_feat['SMA_20'] = df_feat['Close'].rolling(window=20).mean()
        df_feat['EMA_20'] = df_feat['Close'].ewm(span=20, adjust=False).mean()
        
        # 2. Daily Returns & Historical Volatility (14 days)
        df_feat['Daily_Return'] = df_feat['Close'].pct_change()
        df_feat['Volatility_14'] = df_feat['Daily_Return'].rolling(window=14).std() * np.sqrt(252)
        
        # 3. Relative Strength Index (RSI - 14 days)
        delta = df_feat['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        df_feat['RSI_14'] = 100 - (100 / (1 + rs))
        
        logging.info("Technical indicators calculated successfully.")
        return df_feat
        
    except Exception as e:
        logging.error(f"Error calculating technical indicators: {e}")
        return df

def create_target_variable(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Creates a binary target variable for Machine Learning classification.
    Target = 1 if Future_Close > Current Close, else 0.
    """
    logging.info(f"Creating binary target variable with a {horizon}-day horizon...")
    df_target = df.copy()
    
    try:
        df_target['Future_Close'] = df_target['Close'].shift(-horizon)
        df_target['Target'] = (df_target['Future_Close'] > df_target['Close']).astype(int)
        
        # Drop rows with missing future values
        df_target = df_target.dropna(subset=['Future_Close'])
        
        logging.info("Target variable created successfully.")
        return df_target
    except Exception as e:
        logging.error(f"Error creating target variable: {e}")
        return df

# ==========================================
# TEST BLOCK
# ==========================================
if __name__ == "__main__":
    print("--- Testing Transform Module ---")
    # A simple mock dataframe to test the mathematical functions
    mock_data = {'Close': [100, 102, 101, 105, 107, 106, 108, 110, 109, 112, 115, 114, 116, 118, 120]}
    df_mock = pd.DataFrame(mock_data)
    
    df_tech = calculate_technical_indicators(df_mock)
    print("Technical features added successfully.")