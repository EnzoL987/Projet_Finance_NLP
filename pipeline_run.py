import os
import sys
import logging

# 1. Define absolute paths to ensure the script never loses its files
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Add the project root to Python's memory so it can find the 'Src' folder
sys.path.append(PROJECT_ROOT)

# Now we can import the modules we just built!
from Src.extract import fetch_market_data, fetch_financial_news
from Src.transform import analyze_financial_sentiment
from Src.load import initialize_database, insert_asset_dimension, insert_market_data, insert_news_sentiment
from Src.train import fetch_ml_data, train_and_evaluate_models

# Configure pipeline logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_daily_pipeline(ticker: str = "AAPL", company_name: str = "Apple Inc."):
    """
    Executes the end-to-end Data Science pipeline: Extract, Transform, Load, and Train/Predict.
    """
    logging.info("=== Starting Financial NLP Pipeline ===")
    
    # Define file paths at the root of the project
    os.makedirs(os.path.join(PROJECT_ROOT, "SQL"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "Data"), exist_ok=True)
    
    db_path = os.path.join(PROJECT_ROOT, "SQL", "finance_nlp.db")
    model_path = os.path.join(PROJECT_ROOT, "Data", "champion_model.pkl")
    # ---------------------------------------------------------
    # PHASE 1: LOAD (Database Setup)
    # ---------------------------------------------------------
    logging.info("--- Phase 1: Database Initialization ---")
    initialize_database(db_name=db_path)
    asset_id = insert_asset_dimension(db_path, ticker, company_name, "Technology")
    
    # ---------------------------------------------------------
    # PHASE 2: EXTRACT
    # ---------------------------------------------------------
    logging.info("--- Phase 2: Data Extraction ---")
    df_market = fetch_market_data(ticker, period="6mo")
    df_news = fetch_financial_news(f"{ticker} OR {company_name}", days_back=7)
    
    # ---------------------------------------------------------
    # PHASE 3: TRANSFORM
    # ---------------------------------------------------------
    logging.info("--- Phase 3: Data Transformation (NLP) ---")
    df_news_sentiment = analyze_financial_sentiment(df_news)
    
    # ---------------------------------------------------------
    # PHASE 4: LOAD (Insert daily data)
    # ---------------------------------------------------------
    logging.info("--- Phase 4: Storing Data in SQL ---")
    insert_market_data(db_path, df_market, asset_id)
    insert_news_sentiment(db_path, df_news_sentiment, asset_id)
    
    # ---------------------------------------------------------
    # PHASE 5: MACHINE LEARNING (Train & Evaluate)
    # ---------------------------------------------------------
    logging.info("--- Phase 5: Machine Learning Execution ---")
    # Fetch the newly combined dataset from SQL
    df_ml_ready = fetch_ml_data(db_path, ticker)
    
    # Train the models and save the best one
    if not df_ml_ready.empty:
        train_and_evaluate_models(df_ml_ready, save_path=model_path)
    else:
        logging.warning("No data available for Machine Learning training.")
        
    logging.info("=== Pipeline Execution Completed Successfully ===")

if __name__ == "__main__":
    # Execute the main pipeline for Apple
    run_daily_pipeline(ticker="AAPL", company_name="Apple Inc.")