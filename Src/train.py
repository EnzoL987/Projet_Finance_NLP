import sqlite3
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
import logging
import os
import sys

# Add the current directory to the path so we can import our transform module
sys.path.append(os.path.dirname(__file__))
from transform import calculate_technical_indicators, create_target_variable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_ml_data(db_name: str, ticker: str) -> pd.DataFrame:
    """
    Fetches the combined raw dataset (Prices + Aggregated NLP Sentiment) from SQL.
    """
    logging.info(f"Extracting unified dataset for {ticker} from {db_name}...")
    
    query = """
    WITH DailySentiment AS (
        SELECT 
            date,
            asset_id,
            AVG(CASE 
                WHEN sentiment_label = 'positive' THEN sentiment_score 
                WHEN sentiment_label = 'negative' THEN -sentiment_score 
                ELSE 0 
            END) as avg_daily_sentiment,
            COUNT(news_id) as article_count
        FROM fact_news_sentiment
        GROUP BY date, asset_id
    )
    SELECT 
        m.date,
        m.open_price AS Open,
        m.close_price AS Close,
        m.volume AS Volume,
        COALESCE(s.avg_daily_sentiment, 0) as daily_sentiment,
        COALESCE(s.article_count, 0) as news_volume
    FROM fact_market_data m
    JOIN dim_assets d ON m.asset_id = d.asset_id
    LEFT JOIN DailySentiment s ON m.date = s.date AND m.asset_id = s.asset_id
    WHERE d.ticker = ?
    ORDER BY m.date ASC
    """
    
    try:
        conn = sqlite3.connect(db_name)
        df_train = pd.read_sql_query(query, conn, params=(ticker,))
        df_train['date'] = pd.to_datetime(df_train['date'])
        df_train.set_index('date', inplace=True)
        return df_train
    except Exception as e:
        logging.error(f"SQL Extraction failed: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def train_and_evaluate_models(df: pd.DataFrame, save_path: str = "best_model.pkl"):
    """
    Splits the data, trains both XGBoost and Random Forest, compares them, 
    and saves the best performing model to disk.
    """
    logging.info("Preparing features and target variable...")
    df_ready = calculate_technical_indicators(df)
    df_ready = create_target_variable(df_ready, horizon=5)
    df_ready = df_ready.dropna()
    
    if df_ready.empty:
        logging.error("Dataset is empty after transformation. Cannot train.")
        return
        
    features = ['Open', 'Close', 'Volume', 'SMA_20', 'EMA_20', 
                'Daily_Return', 'Volatility_14', 'RSI_14', 
                'daily_sentiment', 'news_volume']
                
    X = df_ready[features]
    y = df_ready['Target']
    
    # Chronological Split (80/20)
    split_idx = int(len(df_ready) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    # 1. Train XGBoost
    logging.info("Training XGBoost...")
    xgb_grid = {'max_depth': [3, 4], 'learning_rate': [0.05, 0.1], 'n_estimators': [100]}
    xgb_search = GridSearchCV(xgb.XGBClassifier(random_state=42, eval_metric='logloss'), 
                              param_grid=xgb_grid, cv=tscv, scoring='accuracy')
    xgb_search.fit(X_train, y_train)
    xgb_acc = accuracy_score(y_test, xgb_search.predict(X_test))
    logging.info(f"XGBoost Accuracy: {xgb_acc * 100:.2f}%")
    
    # 2. Train Random Forest
    logging.info("Training Random Forest...")
    rf_grid = {'max_depth': [4, 6], 'n_estimators': [100, 200]}
    rf_search = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), 
                             param_grid=rf_grid, cv=tscv, scoring='accuracy')
    rf_search.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf_search.predict(X_test))
    logging.info(f"Random Forest Accuracy: {rf_acc * 100:.2f}%")
    
    # 3. Compare and Save
    if xgb_acc >= rf_acc:
        logging.info("XGBoost is the champion! Saving model...")
        best_model = xgb_search.best_estimator_
    else:
        logging.info("Random Forest is the champion! Saving model...")
        best_model = rf_search.best_estimator_
        
    # Save the model to disk using joblib
    joblib.dump(best_model, save_path)
    logging.info(f"Model saved successfully as {save_path}")

# ==========================================
# TEST BLOCK
# ==========================================
if __name__ == "__main__":
    print("--- Testing Train Module ---")
    # Assuming 'finance_nlp.db' is in the root directory (one level above Src/)
    # We will adjust the path for testing
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "finance_nlp.db")
    
    if os.path.exists(db_path):
        df_raw = fetch_ml_data(db_path, "AAPL")
        train_and_evaluate_models(df_raw, save_path="test_model.pkl")
    else:
        print(f"Database not found at {db_path} for testing.")