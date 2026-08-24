import sqlite3
import pandas as pd
import logging

# Configure logging for the database operations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_database(db_name: str = "finance_nlp.db"):
    """
    Connects to the SQLite database and initializes the star schema tables.
    Matches the project specifications exactly.
    """
    logging.info(f"Connecting to database: {db_name}")
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        # 1. Dimension Table: dim_assets
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS dim_assets (
            asset_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker VARCHAR(10) NOT NULL UNIQUE,
            company_name VARCHAR(100),
            sector VARCHAR(50)
        );
        ''')
        
        # 2. Fact Table: fact_market_data
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS fact_market_data (
            date DATE,
            asset_id INTEGER,
            open_price DECIMAL(10,2),
            close_price DECIMAL(10,2),
            volume BIGINT,
            FOREIGN KEY (asset_id) REFERENCES dim_assets(asset_id),
            PRIMARY KEY (date, asset_id)
        );
        ''')
        
        # 3. Fact Table: fact_news_sentiment
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS fact_news_sentiment (
            news_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE,
            asset_id INTEGER,
            headline TEXT,
            sentiment_score DECIMAL(5,4),
            sentiment_label VARCHAR(10),
            FOREIGN KEY (asset_id) REFERENCES dim_assets(asset_id)
        );
        ''')
        
        conn.commit()
        logging.info("Database schema initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize database: {e}")
    finally:
        if conn:
            conn.close()

def insert_asset_dimension(db_name: str, ticker: str, company_name: str, sector: str) -> int:
    """
    Inserts a company into dim_assets if it doesn't exist.
    Returns the asset_id.
    """
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR IGNORE INTO dim_assets (ticker, company_name, sector) 
            VALUES (?, ?, ?)
        ''', (ticker, company_name, sector))
        conn.commit()
        
        cursor.execute('SELECT asset_id FROM dim_assets WHERE ticker = ?', (ticker,))
        result = cursor.fetchone()
        return result[0] if result else None
    except Exception as e:
        logging.error(f"Error inserting asset {ticker}: {e}")
        return None
    finally:
        if conn:
            conn.close()

def insert_market_data(db_name: str, df_market: pd.DataFrame, asset_id: int):
    """
    Inserts daily OHLCV data into fact_market_data.
    """
    if df_market.empty:
        logging.warning("Market dataframe is empty. Nothing to insert.")
        return
        
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        records = []
        
        for date, row in df_market.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            records.append((date_str, asset_id, float(row['Open']), float(row['Close']), int(row['Volume'])))
            
        cursor.executemany('''
            INSERT OR IGNORE INTO fact_market_data (date, asset_id, open_price, close_price, volume)
            VALUES (?, ?, ?, ?, ?)
        ''', records)
        
        conn.commit()
        logging.info(f"Inserted {cursor.rowcount} market records for asset_id {asset_id}.")
    except Exception as e:
        logging.error(f"Error inserting market data: {e}")
    finally:
        if conn:
            conn.close()

def insert_news_sentiment(db_name: str, df_news: pd.DataFrame, asset_id: int):
    """
    Inserts news articles and FinBERT sentiment scores into fact_news_sentiment.
    """
    if df_news.empty or 'Sentiment_Label' not in df_news.columns:
        logging.warning("News dataframe is empty or missing sentiment. Nothing to insert.")
        return
        
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        records = []
        
        for _, row in df_news.iterrows():
            records.append((
                row['Date'], asset_id, row['Headline'], 
                float(row['Sentiment_Score']), row['Sentiment_Label']
            ))
            
        cursor.executemany('''
            INSERT INTO fact_news_sentiment (date, asset_id, headline, sentiment_score, sentiment_label)
            VALUES (?, ?, ?, ?, ?)
        ''', records)
        
        conn.commit()
        logging.info(f"Inserted {cursor.rowcount} news sentiment records for asset_id {asset_id}.")
    except Exception as e:
        logging.error(f"Error inserting news sentiment: {e}")
    finally:
        if conn:
            conn.close()

# ==========================================
# TEST BLOCK
# ==========================================
if __name__ == "__main__":
    print("--- Testing Load Module ---")
    # Initialize the database file
    initialize_database("test_finance_nlp.db")
    print("Test database 'test_finance_nlp.db' created successfully. You can delete it later.")