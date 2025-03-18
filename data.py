import sqlite3
import threading
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import asdict
import os
from utils.logger import logger

class CryptoDatabase:
    """Database handler for cryptocurrency market data, analysis and predictions"""
    
    def __init__(self, db_path: str = "data/crypto_history.db"):
        """Initialize database connection and create tables if they don't exist"""
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.local = threading.local()  # Thread-local storage
        self._initialize_database()
    
    @property
    def conn(self):
        """Thread-safe connection property - returns the connection for current thread"""
        conn, _ = self._get_connection()
        return conn
        
    @property
    def cursor(self):
        """Thread-safe cursor property - returns the cursor for current thread"""
        _, cursor = self._get_connection()
        return cursor

    def _get_connection(self):
        """Get database connection, creating it if necessary - thread-safe version"""
        # Check if this thread has a connection
        if not hasattr(self.local, 'conn') or self.local.conn is None:
            # Create a new connection for this thread
            self.local.conn = sqlite3.connect(self.db_path)
            self.local.conn.row_factory = sqlite3.Row
            self.local.cursor = self.local.conn.cursor()
        
        return self.local.conn, self.local.cursor

    def _initialize_database(self):
        """Create necessary tables if they don't exist"""
        conn, cursor = self._get_connection()
        
        try:
            # Market Data Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    chain TEXT NOT NULL,
                    price REAL NOT NULL,
                    volume REAL NOT NULL,
                    price_change_24h REAL,
                    market_cap REAL,
                    ath REAL,
                    ath_change_percentage REAL
                )
            """)

# Posted Content Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS posted_content (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    content TEXT NOT NULL,
                    sentiment JSON NOT NULL,
                    trigger_type TEXT NOT NULL,
                    price_data JSON NOT NULL,
                    meme_phrases JSON NOT NULL,
                    is_prediction BOOLEAN DEFAULT 0,
                    prediction_data JSON,
                    timeframe TEXT DEFAULT '1h'
                )
            """)

            # Chain Mood History
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mood_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    chain TEXT NOT NULL,
                    mood TEXT NOT NULL,
                    indicators JSON NOT NULL
                )
            """)
            
            # Smart Money Indicators Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS smart_money_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    chain TEXT NOT NULL,
                    volume_z_score REAL,
                    price_volume_divergence BOOLEAN,
                    stealth_accumulation BOOLEAN,
                    abnormal_volume BOOLEAN,
                    volume_vs_hourly_avg REAL,
                    volume_vs_daily_avg REAL,
                    volume_cluster_detected BOOLEAN,
                    unusual_trading_hours JSON,
                    raw_data JSON
                )
            """)
            
            # Token Market Comparison Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS token_market_comparison (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token TEXT NOT NULL,
                    vs_market_avg_change REAL,
                    vs_market_volume_growth REAL,
                    outperforming_market BOOLEAN,
                    correlations JSON
                )
            """)
            
            # Token Correlations Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS token_correlations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token TEXT NOT NULL,
                    avg_price_correlation REAL NOT NULL,
                    avg_volume_correlation REAL NOT NULL,
                    full_data JSON NOT NULL
                )
            """)
            
            # Generic JSON Data Table for flexible storage
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generic_json_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    data_type TEXT NOT NULL,
                    data JSON NOT NULL
                )
            """)
            
            # NEW TABLE: Replied Tweets Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS replied_tweets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tweet_id TEXT NOT NULL UNIQUE,
                    author_username TEXT NOT NULL,
                    tweet_text TEXT NOT NULL,
                    reply_text TEXT NOT NULL,
                    timestamp DATETIME NOT NULL
                )
            """)            
# PREDICTION TABLES
            
            # Predictions Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS price_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    prediction_type TEXT NOT NULL,
                    prediction_value REAL NOT NULL,
                    confidence_level REAL NOT NULL,
                    lower_bound REAL,
                    upper_bound REAL,
                    prediction_rationale TEXT,
                    method_weights JSON,
                    model_inputs JSON,
                    technical_signals JSON,
                    expiration_time DATETIME NOT NULL
                )
            """)

            # Prediction Outcomes Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER NOT NULL,
                    actual_outcome REAL NOT NULL,
                    accuracy_percentage REAL NOT NULL,
                    was_correct BOOLEAN NOT NULL,
                    evaluation_time DATETIME NOT NULL,
                    deviation_from_prediction REAL NOT NULL,
                    market_conditions JSON,
                    FOREIGN KEY (prediction_id) REFERENCES price_predictions(id)
                )
            """)

            # Prediction Performance Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    prediction_type TEXT NOT NULL,
                    total_predictions INTEGER NOT NULL,
                    correct_predictions INTEGER NOT NULL,
                    accuracy_rate REAL NOT NULL,
                    avg_deviation REAL NOT NULL,
                    updated_at DATETIME NOT NULL
                )
            """)
            
            # Technical Indicators Table with timeframe support
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token TEXT NOT NULL,
                    timeframe TEXT NOT NULL DEFAULT '1h',
                    rsi REAL,
                    macd_line REAL,
                    macd_signal REAL,
                    macd_histogram REAL, 
                    bb_upper REAL,
                    bb_middle REAL,
                    bb_lower REAL,
                    stoch_k REAL,
                    stoch_d REAL,
                    obv REAL,
                    adx REAL,
                    ichimoku_data JSON,
                    pivot_points JSON,
                    overall_trend TEXT,
                    trend_strength REAL,
                    volatility REAL,
                    raw_data JSON
                )
            """)
            
            # Statistical Models Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS statistical_forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    forecast_value REAL NOT NULL,
                    confidence_80_lower REAL,
                    confidence_80_upper REAL,
                    confidence_95_lower REAL,
                    confidence_95_upper REAL,
                    model_parameters JSON,
                    input_data_summary JSON
                )
            """)
            
            # Machine Learning Models Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ml_forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    forecast_value REAL NOT NULL,
                    confidence_80_lower REAL,
                    confidence_80_upper REAL,
                    confidence_95_lower REAL,
                    confidence_95_upper REAL,
                    feature_importance JSON,
                    model_parameters JSON
                )
            """)
            
            # Claude AI Predictions Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS claude_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    claude_model TEXT NOT NULL,
                    prediction_value REAL NOT NULL,
                    confidence_level REAL,
                    sentiment TEXT,
                    rationale TEXT,
                    key_factors JSON,
                    input_data JSON
                )
            """)
            
            # Timeframe Metrics Table - New table to track metrics by timeframe
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS timeframe_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    avg_accuracy REAL,
                    total_count INTEGER,
                    correct_count INTEGER,
                    model_weights JSON,
                    best_model TEXT,
                    last_updated DATETIME NOT NULL
                )
            """# Create indices for better query performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_chain ON market_data(chain)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_posted_content_timestamp ON posted_content(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_replied_tweets_tweet_id ON replied_tweets(tweet_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_replied_tweets_timestamp ON replied_tweets(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_replied_tweets_author ON replied_tweets(author_username)")
            
            # HERE'S THE FIX: Check if timeframe column exists in posted_content before creating index
            try:
                # Try to get column info
                cursor.execute("PRAGMA table_info(posted_content)")
                columns = [column[1] for column in cursor.fetchall()]
                
                # Check if timeframe column exists
                if 'timeframe' not in columns:
                    # Add the timeframe column if it doesn't exist
                    cursor.execute("ALTER TABLE posted_content ADD COLUMN timeframe TEXT DEFAULT '1h'")
                    conn.commit()
                    logger.logger.info("Added missing timeframe column to posted_content table")
                
                # Now it's safe to create the index
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_posted_content_timeframe ON posted_content(timeframe)")
            except Exception as e:
                logger.log_error("Timeframe Column Check", str(e))
            
            # Check if timeframe column exists in technical_indicators before creating index
            try:
                # Try to get column info
                cursor.execute("PRAGMA table_info(technical_indicators)")
                columns = [column[1] for column in cursor.fetchall()]
                
                # Check if timeframe column exists
                if 'timeframe' not in columns:
                    # Add the timeframe column if it doesn't exist
                    cursor.execute("ALTER TABLE technical_indicators ADD COLUMN timeframe TEXT DEFAULT '1h'")
                    conn.commit()
                    logger.logger.info("Added missing timeframe column to technical_indicators table")
                
                # Now it's safe to create the index
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_technical_indicators_timeframe ON technical_indicators(timeframe)")
            except Exception as e:
                logger.log_error("Timeframe Column Check for technical_indicators", str(e))
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_mood_history_timestamp ON mood_history(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_mood_history_chain ON mood_history(chain)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_smart_money_timestamp ON smart_money_indicators(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_smart_money_chain ON smart_money_indicators(chain)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_generic_json_timestamp ON generic_json_data(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_generic_json_type ON generic_json_data(data_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_token_market_comparison_timestamp ON token_market_comparison(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_token_market_comparison_token ON token_market_comparison(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_token_correlations_timestamp ON token_correlations(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_token_correlations_token ON token_correlations(token)")
            
            # Prediction indices - Enhanced for timeframe support
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_predictions_token ON price_predictions(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_predictions_timeframe ON price_predictions(timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_predictions_timestamp ON price_predictions(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_predictions_expiration ON price_predictions(expiration_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_predictions_token_timeframe ON price_predictions(token, timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction_outcomes_prediction_id ON prediction_outcomes(prediction_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction_performance_token ON prediction_performance(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction_performance_timeframe ON prediction_performance(timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction_performance_token_timeframe ON prediction_performance(token, timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_technical_indicators_token ON technical_indicators(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_technical_indicators_timestamp ON technical_indicators(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_statistical_forecasts_token ON statistical_forecasts(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_statistical_forecasts_timeframe ON statistical_forecasts(timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ml_forecasts_token ON ml_forecasts(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ml_forecasts_timeframe ON ml_forecasts(timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_claude_predictions_token ON claude_predictions(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_claude_predictions_timeframe ON claude_predictions(timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeframe_metrics_token ON timeframe_metrics(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeframe_metrics_timeframe ON timeframe_metrics(timeframe)")

            conn.commit()
            logger.logger.info("Database initialized successfully")

        except Exception as e:
            logger.log_error("Database Initialization", str(e))
            raise

    #########################
    # CORE DATA STORAGE METHODS
    #########################

    def store_market_data(self, chain: str, data: Dict[str, Any]) -> None:
        """Store market data for a specific chain"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                INSERT INTO market_data (
                    timestamp, chain, price, volume, price_change_24h, 
                    market_cap, ath, ath_change_percentage
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                chain,
                data['current_price'],
                data['volume'],
                data['price_change_percentage_24h'],
                data['market_cap'],
                data['ath'],
                data['ath_change_percentage']
            ))
            conn.commit()
        except Exception as e:
            logger.log_error(f"Store Market Data - {chain}", str(e))
            conn.rollback())

    def store_token_correlations(self, token: str, correlations: Dict[str, Any]) -> None:
        """Store token-specific correlation data"""
        conn, cursor = self._get_connection()
        try:
            # Extract average correlations
            avg_price_corr = correlations.get('avg_price_correlation', 0)
            avg_volume_corr = correlations.get('avg_volume_correlation', 0)
            
            cursor.execute("""
                INSERT INTO token_correlations (
                    timestamp, token, avg_price_correlation, avg_volume_correlation, full_data
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                token,
                avg_price_corr,
                avg_volume_corr,
                json.dumps(correlations)
            ))
            conn.commit()
            logger.logger.debug(f"Stored correlation data for {token}")
        except Exception as e:
            logger.log_error(f"Store Token Correlations - {token}", str(e))
            conn.rollback()
            
    def store_token_market_comparison(self, token: str, vs_market_avg_change: float,
                                    vs_market_volume_growth: float, outperforming_market: bool,
                                    correlations: Dict[str, Any]) -> None:
        """Store token vs market comparison data"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                INSERT INTO token_market_comparison (
                    timestamp, token, vs_market_avg_change, vs_market_volume_growth,
                    outperforming_market, correlations
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                token,
                vs_market_avg_change,
                vs_market_volume_growth,
                1 if outperforming_market else 0,
                json.dumps(correlations)
            ))
            conn.commit()
            logger.logger.debug(f"Stored market comparison data for {token}")
        except Exception as e:
            logger.log_error(f"Store Token Market Comparison - {token}", str(e))
            conn.rollback()

    def store_posted_content(self, content: str, sentiment: Dict, 
                           trigger_type: str, price_data: Dict, 
                           meme_phrases: Dict, is_prediction: bool = False,
                           prediction_data: Dict = None, timeframe: str = "1h") -> None:
        """Store posted content with metadata and timeframe"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                INSERT INTO posted_content (
                    timestamp, content, sentiment, trigger_type, 
                    price_data, meme_phrases, is_prediction, prediction_data, timeframe
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                content,
                json.dumps(sentiment),
                trigger_type,
                json.dumps(price_data),
                json.dumps(meme_phrases),
                1 if is_prediction else 0,
                json.dumps(prediction_data) if prediction_data else None,
                timeframe
            ))
            conn.commit()
        except Exception as e:
            logger.log_error("Store Posted Content", str(e))
            conn.rollback()

    def store_mood(self, chain: str, mood: str, indicators: Dict) -> None:
        """Store mood data for a specific chain"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                INSERT INTO mood_history (
                    timestamp, chain, mood, indicators
                ) VALUES (?, ?, ?, ?)
            """, (
                datetime.now(),
                chain,
                mood,
                json.dumps(asdict(indicators) if hasattr(indicators, '__dataclass_fields__') else indicators)
            ))
            conn.commit()
        except Exception as e:
            logger.log_error(f"Store Mood - {chain}", str(e))
            conn.rollback()
            
    def store_smart_money_indicators(self, chain: str, indicators: Dict[str, Any]) -> None:
        """Store smart money indicators for a chain"""
        conn, cursor = self._get_connection()
        try:
            # Extract values with defaults for potential missing keys
            volume_z_score = indicators.get('volume_z_score', 0.0)
            price_volume_divergence = 1 if indicators.get('price_volume_divergence', False) else 0
            stealth_accumulation = 1 if indicators.get('stealth_accumulation', False) else 0
            abnormal_volume = 1 if indicators.get('abnormal_volume', False) else 0
            volume_vs_hourly_avg = indicators.get('volume_vs_hourly_avg', 0.0)
            volume_vs_daily_avg = indicators.get('volume_vs_daily_avg', 0.0)
            volume_cluster_detected = 1 if indicators.get('volume_cluster_detected', False) else 0
            
            # Convert unusual_trading_hours to JSON if present
            unusual_hours = json.dumps(indicators.get('unusual_trading_hours', []))
            
            # Store all raw data for future reference
            raw_data = json.dumps(indicators)
            
            cursor.execute("""
                INSERT INTO smart_money_indicators (
                    timestamp, chain, volume_z_score, price_volume_divergence,
                    stealth_accumulation, abnormal_volume, volume_vs_hourly_avg,
                    volume_vs_daily_avg, volume_cluster_detected, unusual_trading_hours,
                    raw_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                chain,
                volume_z_score,
                price_volume_divergence,
                stealth_accumulation,
                abnormal_volume,
                volume_vs_hourly_avg,
                volume_vs_daily_avg,
                volume_cluster_detected,
                unusual_hours,
                raw_data
            ))
            conn.commit()
        except Exception as e:
            logger.log_error(f"Store Smart Money Indicators - {chain}", str(e))
            conn.rollback()
            
    def _store_json_data(self, data_type: str, data: Dict[str, Any]) -> None:
        """Generic method to store JSON data in a generic_json_data table"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                INSERT INTO generic_json_data (
                    timestamp, data_type, data
                ) VALUES (?, ?, ?)
            """, (
                datetime.now(),
                data_type,
                json.dumps(data)
            ))
            conn.commit()
        except Exception as e:
            logger.log_error(f"Store JSON Data - {data_type}", str(e))
            conn.rollback()

    # NEW METHODS FOR REPLIED TWEETS
    def store_replied_tweet(self, tweet_id: str, author_username: str, tweet_text: str, reply_text: str) -> bool:
        """Store information about a tweet we've replied to"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                INSERT INTO replied_tweets (
                    tweet_id, author_username, tweet_text, reply_text, timestamp
                ) VALUES (?, ?, ?, ?, ?)
            """, (tweet_id, author_username, tweet_text, reply_text, datetime.now()))
            conn.commit()
            logger.logger.info(f"Stored reply to tweet {tweet_id} by @{author_username}")
            return True
        except sqlite3.IntegrityError:
            # This can happen if we try to insert a duplicate tweet_id
            logger.logger.warning(f"Already replied to tweet {tweet_id}")
            return False
        except Exception as e:
            logger.log_error(f"Store Replied Tweet - {tweet_id}", str(e))
            conn.rollback()
            return False
        
    def get_replied_tweet_ids(self, hours: int = 24) -> List[str]:
        """Get IDs of tweets we've replied to within the given time period"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                SELECT tweet_id FROM replied_tweets
                WHERE timestamp >= datetime('now', '-' || ? || ' hours')
            """, (hours,))
            return [row['tweet_id'] for row in cursor.fetchall()]
        except Exception as e:
            logger.log_error("Get Replied Tweet IDs", str(e))
            return []
    
    def get_reply_stats(self) -> Dict[str, Any]:
        """Get statistics about our replied tweets"""
        conn, cursor = self._get_connection()
        try:
            stats = {}
            
            # Total replies
            cursor.execute("SELECT COUNT(*) as count FROM replied_tweets")
            stats['total_replies'] = cursor.fetchone()['count']
            
            # Replies in last 24 hours
            cursor.execute("""
                SELECT COUNT(*) as count FROM replied_tweets
                WHERE timestamp >= datetime('now', '-24 hours')
            """)
            stats['replies_last_24h'] = cursor.fetchone()['count']
            
            # Top 5 authors we've replied to
            cursor.execute("""
                SELECT author_username, COUNT(*) as count FROM replied_tweets
                GROUP BY author_username
                ORDER BY count DESC
                LIMIT 5
            """)
            stats['top_authors'] = [dict(row) for row in cursor.fetchall()]
            
            # Most recent replies
            cursor.execute("""
                SELECT tweet_id, author_username, tweet_text, reply_text, timestamp
                FROM replied_tweets
                ORDER BY timestamp DESC
                LIMIT 5
            """)
            stats['recent_replies'] = [dict(row) for row in cursor.fetchall()]
            
            return stats
        except Exception as e:
            logger.log_error("Get Reply Stats", str(e))
            return {
                'total_replies': 0,
                'replies_last_24h': 0,
                'top_authors': [],
                'recent_replies': []
            }

    #########################
    # DATA RETRIEVAL METHODS
    #########################

    def get_recent_market_data(self, chain: str, hours: int = 24) -> List[Dict]:
        """Get recent market data for a specific chain"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                SELECT * FROM market_data 
                WHERE chain = ? 
                AND timestamp >= datetime('now', '-' || ? || ' hours')
                ORDER BY timestamp DESC
            """, (chain, hours))
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.log_error(f"Get Recent Market Data - {chain}", str(e))
            return []
            
    def get_token_correlations(self, token: str, hours: int = 24) -> List[Dict]:
        """Get token-specific correlation data"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                SELECT * FROM token_correlations 
                WHERE token = ?
                AND timestamp >= datetime('now', '-' || ? || ' hours')
                ORDER BY timestamp DESC
            """, (token, hours))
            
            results = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON field
            for result in results:
                result["full_data"] = json.loads(result["full_data"]) if result["full_data"] else {}
                
            return results
        except Exception as e:
            logger.log_error(f"Get Token Correlations - {token}", str(e))
            return []
            
    def get_token_market_comparison(self, token: str, hours: int = 24) -> List[Dict]:
        """Get token vs market comparison data"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                SELECT * FROM token_market_comparison 
                WHERE token = ?
                AND timestamp >= datetime('now', '-' || ? || ' hours')
                ORDER BY timestamp DESC
            """, (token, hours))
            
            results = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON field
            for result in results:
                result["correlations"] = json.loads(result["correlations"]) if result["correlations"] else {}
                
            return results
        except Exception as e:
            logger.log_error(f"Get Token Market Comparison - {token}", str(e))
            return []
        
    def get_recent_posts(self, hours: int = 24, timeframe: str = None) -> List[Dict]:
        """
        Get recent posted content
        Can filter by timeframe
        """
        conn, cursor = self._get_connection()
        try:
            query = """
                SELECT * FROM posted_content 
                WHERE timestamp >= datetime('now', '-' || ? || ' hours')
            """
            params = [hours]
            
            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)
                
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            
            results = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON fields
            for result in results:
                result["sentiment"] = json.loads(result["sentiment"]) if result["sentiment"] else {}
                result["price_data"] = json.loads(result["price_data"]) if result["price_data"] else {}
                result["meme_phrases"] = json.loads(result["meme_phrases"]) if result["meme_phrases"] else {}
                result["prediction_data"] = json.loads(result["prediction_data"]) if result["prediction_data"] else None
                
            return results
        except Exception as e:
            logger.log_error("Get Recent Posts", str(e))
            return []

    def get_chain_stats(self, chain: str, hours: int = 24) -> Dict[str, Any]:
        """Get statistical summary for a chain"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                SELECT 
                    AVG(price) as avg_price,
                    MAX(price) as max_price,
                    MIN(price) as min_price,
                    AVG(volume) as avg_volume,
                    MAX(volume) as max_volume,
                    AVG(price_change_24h) as avg_price_change
                FROM market_data 
                WHERE chain = ? 
                AND timestamp >= datetime('now', '-' || ? || ' hours')
            """, (chain, hours))
            result = cursor.fetchone()
            if result:
                return dict(result)
            return {}
        except Exception as e:
            logger.log_error(f"Get Chain Stats - {chain}", str(e))
            return {}
            
    def get_smart_money_indicators(self, chain: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent smart money indicators for a chain"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                SELECT * FROM smart_money_indicators
                WHERE chain = ? 
                AND timestamp >= datetime('now', '-' || ? || ' hours')
                ORDER BY timestamp DESC
            """, (chain, hours))
            
            results = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON fields
            for result in results:
                result["unusual_trading_hours"] = json.loads(result["unusual_trading_hours"]) if result["unusual_trading_hours"] else []
                result["raw_data"] = json.loads(result["raw_data"]) if result["raw_data"] else {}
                
            return results
        except Exception as e:
            logger.log_error(f"Get Smart Money Indicators - {chain}", str(e))
            return []
            
    def get_token_market_stats(self, token: str, hours: int = 24) -> Dict[str, Any]:
        """Get statistical summary of token vs market performance"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                SELECT 
                    AVG(vs_market_avg_change) as avg_performance_diff,
                    AVG(vs_market_volume_growth) as avg_volume_growth_diff,
                    SUM(CASE WHEN outperforming_market = 1 THEN 1 ELSE 0 END) as outperforming_count,
                    COUNT(*) as total_records
                FROM token_market_comparison
                WHERE token = ?
                AND timestamp >= datetime('now', '-' || ? || ' hours')
            """, (token, hours))
            result = cursor.fetchone()
            if result:
                result_dict = dict(result)
                
                # Calculate percentage of time outperforming
                if result_dict['total_records'] > 0:
                    result_dict['outperforming_percentage'] = (result_dict['outperforming_count'] / result_dict['total_records']) * 100
                else:
                    result_dict['outperforming_percentage'] = 0
                    
                return result_dict
            return {}
        except Exception as e:
            logger.log_error(f"Get Token Market Stats - {token}", str(e))
            return {}

    def get_latest_smart_money_alert(self, chain: str) -> Optional[Dict[str, Any]]:
        """Get the most recent smart money alert for a chain"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                SELECT * FROM smart_money_indicators
                WHERE chain = ? 
                AND (abnormal_volume = 1 OR stealth_accumulation = 1 OR volume_cluster_detected = 1)
                ORDER BY timestamp DESC
                LIMIT 1
            """, (chain,))
            result = cursor.fetchone()
            if result:
                result_dict = dict(result)
                
                # Parse JSON fields
                result_dict["unusual_trading_hours"] = json.loads(result_dict["unusual_trading_hours"]) if result_dict["unusual_trading_hours"] else []
                result_dict["raw_data"] = json.loads(result_dict["raw_data"]) if result_dict["raw_data"] else {}
                
                return result_dict
            return None
        except Exception as e:
            logger.log_error(f"Get Latest Smart Money Alert - {chain}", str(e))
            return None
    
    def get_volume_trend(self, chain: str, hours: int = 24) -> Dict[str, Any]:
        """Get volume trend analysis for a chain"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                SELECT 
                    timestamp,
                    volume
                FROM market_data
                WHERE chain = ? 
                AND timestamp >= datetime('now', '-' || ? || ' hours')
                ORDER BY timestamp ASC
            """, (chain, hours))
            
            results = cursor.fetchall()
            if not results:
                return {'trend': 'insufficient_data', 'change': 0}
                
            # Calculate trend
            volumes = [row['volume'] for row in results]
            earliest_volume = volumes[0] if volumes else 0
            latest_volume = volumes[-1] if volumes else 0
            
            if earliest_volume > 0:
                change_pct = ((latest_volume - earliest_volume) / earliest_volume) * 100
            else:
                change_pct = 0
                
            # Determine trend description
            if change_pct >= 15:
                trend = "significant_increase"
            elif change_pct <= -15:
                trend = "significant_decrease"
            elif change_pct >= 5:
                trend = "moderate_increase"
            elif change_pct <= -5:
                trend = "moderate_decrease"
            else:
                trend = "stable"
                
            return {
                'trend': trend,
                'change': change_pct,
                'earliest_volume': earliest_volume,
                'latest_volume': latest_volume,
                'data_points': len(volumes)
            }
            
        except Exception as e:
            logger.log_error(f"Get Volume Trend - {chain}", str(e))
            return {'trend': 'error', 'change': 0}    
        
    def get_top_performing_tokens(self, hours: int = 24, limit: int = 5) -> List[Dict[str, Any]]:
        """Get list of top performing tokens based on price change"""
        conn, cursor = self._get_connection()
        try:
            # Get unique tokens in database
            cursor.execute("""
                SELECT DISTINCT chain
                FROM market_data
                WHERE timestamp >= datetime('now', '-' || ? || ' hours')
            """, (hours,))
            tokens = [row['chain'] for row in cursor.fetchall()]
            
            results = []
            for token in tokens:
                # Get latest price and 24h change
                cursor.execute("""
                    SELECT price, price_change_24h
                    FROM market_data
                    WHERE chain = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (token,))
                data = cursor.fetchone()
                
                if data:
                    results.append({
                        'token': token,
                        'price': data['price'],
                        'price_change_24h': data['price_change_24h']
                    })
            
            # Sort by price change (descending)
            results.sort(key=lambda x: x.get('price_change_24h', 0), reverse=True)
            
            # Return top N tokens
            return results[:limit]
            
        except Exception as e:
            logger.log_error("Get Top Performing Tokens", str(e))
            return []

    def get_tokens_by_prediction_accuracy(self, timeframe: str = "1h", min_predictions: int = 5) -> List[Dict[str, Any]]:
        """
        Get tokens sorted by prediction accuracy for a specific timeframe
        Only includes tokens with at least min_predictions number of predictions
        """
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                SELECT token, accuracy_rate, total_predictions, correct_predictions
                FROM prediction_performance
                WHERE timeframe = ? AND total_predictions >= ?
                ORDER BY accuracy_rate DESC
            """, (timeframe, min_predictions))
            
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.log_error(f"Get Tokens By Prediction Accuracy - {timeframe}", str(e))
            return []

    #########################
    # DUPLICATE DETECTION METHODS
    #########################
    
    def check_content_similarity(self, content: str, timeframe: str = None) -> bool:
        """
        Check if similar content was recently posted
        Can filter by timeframe
        """
        conn, cursor = self._get_connection()
        try:
            query = """
                SELECT content FROM posted_content 
                WHERE timestamp >= datetime('now', '-1 hour')
            """
            
            params = []
            
            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)
                
            cursor.execute(query, params)
            recent_posts = [row['content'] for row in cursor.fetchall()]
            
            # Simple similarity check - can be enhanced later
            return any(content.strip() == post.strip() for post in recent_posts)
        except Exception as e:
            logger.log_error("Check Content Similarity", str(e))
            return False
            
    def check_exact_content_match(self, content: str, timeframe: str = None) -> bool:
        """
        Check for exact match of content within recent posts
        Can filter by timeframe
        """
        conn, cursor = self._get_connection()
        try:
            query = """
                SELECT COUNT(*) as count FROM posted_content 
                WHERE content = ? 
                AND timestamp >= datetime('now', '-3 hours')
            """
            
            params = [content]
            
            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)
                
            cursor.execute(query, params)
            result = cursor.fetchone()
            return result['count'] > 0 if result else False
        except Exception as e:
            logger.log_error("Check Exact Content Match", str(e))
            return False
            
    def check_content_similarity_with_timeframe(self, content: str, hours: int = 1, timeframe: str = None) -> bool:
        """
        Check if similar content was posted within a specified timeframe
        Can filter by prediction timeframe
        """
        conn, cursor = self._get_connection()
        try:
            query = """
                SELECT content FROM posted_content 
                WHERE timestamp >= datetime('now', '-' || ? || ' hours')
            """
            
            params = [hours]
            
            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)
                
            cursor.execute(query, params)
            recent_posts = [row['content'] for row in cursor.fetchall()]
            
            # Split content into main text and hashtags
            content_main = content.split("\n\n#")[0].lower() if "\n\n#" in content else content.lower()
            
            for post in recent_posts:
                post_main = post.split("\n\n#")[0].lower() if "\n\n#" in post else post.lower()
                
                # Calculate similarity based on word overlap
                content_words = set(content_main.split())
                post_words = set(post_main.split())
                
                if content_words and post_words:
                    overlap = len(content_words.intersection(post_words))
                    similarity = overlap / max(len(content_words), len(post_words))
                    
                    # Consider similar if 70% or more words overlap
                    if similarity > 0.7:
                        return True
            
            return False
        except Exception as e:
            logger.log_error("Check Content Similarity With Timeframe", str(e))
            return False

    #########################
    # PREDICTION METHODS
    #########################
    
    def store_prediction(self, token: str, prediction_data: Dict[str, Any], timeframe: str = "1h") -> int:
        """
        Store a prediction in the database
        Returns the ID of the inserted prediction
        """
        conn, cursor = self._get_connection()
        prediction_id = None
        
        try:
            # Extract prediction details
            prediction = prediction_data.get("prediction", {})
            rationale = prediction_data.get("rationale", "")
            sentiment = prediction_data.get("sentiment", "NEUTRAL")
            key_factors = json.dumps(prediction_data.get("key_factors", []))
            model_weights = json.dumps(prediction_data.get("model_weights", {}))
            model_inputs = json.dumps(prediction_data.get("inputs", {}))
            
            # Calculate expiration time based on timeframe
            if timeframe == "1h":
                expiration_time = datetime.now() + timedelta(hours=1)
            elif timeframe == "24h":
                expiration_time = datetime.now() + timedelta(hours=24)
            elif timeframe == "7d":
                expiration_time = datetime.now() + timedelta(days=7)
            else:
                expiration_time = datetime.now() + timedelta(hours=1)  # Default to 1h
                
            cursor.execute("""
                INSERT INTO price_predictions (
                    timestamp, token, timeframe, prediction_type,
                    prediction_value, confidence_level, lower_bound, upper_bound,
                    prediction_rationale, method_weights, model_inputs, technical_signals,
                    expiration_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now(),
                    token,
                    timeframe,
                    "price",
                    prediction["price"],
                    prediction["confidence"],
                    prediction["lower_bound"],
                    prediction["upper_bound"],
                    rationale,
                    model_weights,
                    model_inputs,
                    key_factors,
                    expiration_time
                ))
                
            conn.commit()
            prediction_id = cursor.lastrowid
            logger.logger.debug(f"Stored {timeframe} prediction for {token} with ID {prediction_id}")
            
            # Also store in specialized tables based on the prediction models used
            
            # Store Claude prediction if it was used
            if prediction_data.get("model_weights", {}).get("claude_enhanced", 0) > 0:
                self._store_claude_prediction(token, prediction_data, timeframe)
                
            # Store technical analysis if available
            if "inputs" in prediction_data and "technical_analysis" in prediction_data["inputs"]:
                self._store_technical_indicators(token, prediction_data["inputs"]["technical_analysis"], timeframe)
                
            # Store statistical forecast if available
            if "inputs" in prediction_data and "statistical_forecast" in prediction_data["inputs"]:
                self._store_statistical_forecast(token, prediction_data["inputs"]["statistical_forecast"], timeframe)
                
            # Store ML forecast if available
            if "inputs" in prediction_data and "ml_forecast" in prediction_data["inputs"]:
                self._store_ml_forecast(token, prediction_data["inputs"]["ml_forecast"], timeframe)
                
            # Update timeframe metrics
            self._update_timeframe_metrics(token, timeframe, prediction_data)
            
        except Exception as e:
            logger.log_error(f"Store Prediction - {token} ({timeframe})", str(e))
            conn.rollback()
            
        return prediction_id

    def close(self):
        """Close database connection"""
        if hasattr(self, 'local') and hasattr(self.local, 'conn') and self.local.conn:
            self.local.conn.close()
            self.local.conn = None
            self.local.cursor = None    

    def _store_technical_indicators(self, token: str, technical_analysis: Dict[str, Any], timeframe: str = "1h") -> None:
        """Store technical indicator data with timeframe support"""
        conn, cursor = self._get_connection()
        try:
            # Extract indicator values
            overall_trend = technical_analysis.get("overall_trend", "neutral")
            trend_strength = technical_analysis.get("trend_strength", 50)
            signals = technical_analysis.get("signals", {})
            
            # Extract individual indicators if available
            indicators = technical_analysis.get("indicators", {})
            
            # Get RSI
            rsi = indicators.get("rsi", None)
            
            # Get MACD
            macd = indicators.get("macd", {})
            macd_line = macd.get("macd_line", None)
            signal_line = macd.get("signal_line", None)
            histogram = macd.get("histogram", None)
            
            # Get Bollinger Bands
            bb = indicators.get("bollinger_bands", {})
            bb_upper = bb.get("upper", None)
            bb_middle = bb.get("middle", None)
            bb_lower = bb.get("lower", None)
            
            # Get Stochastic
            stoch = indicators.get("stochastic", {})
            stoch_k = stoch.get("k", None)
            stoch_d = stoch.get("d", None)
            
            # Get OBV
            obv = indicators.get("obv", None)
            
            # Get ADX
            adx = indicators.get("adx", None)
            
            # Get additional timeframe-specific indicators
            ichimoku_data = json.dumps(indicators.get("ichimoku", {}))
            pivot_points = json.dumps(indicators.get("pivot_points", {}))
            
            # Get volatility
            volatility = technical_analysis.get("volatility", None)
            
            # Store in database
            cursor.execute("""
                INSERT INTO technical_indicators (
                    timestamp, token, timeframe, rsi, macd_line, macd_signal, 
                    macd_histogram, bb_upper, bb_middle, bb_lower,
                    stoch_k, stoch_d, obv, adx, ichimoku_data, pivot_points,
                    overall_trend, trend_strength, volatility, raw_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                token,
                timeframe,
                rsi,
                macd_line,
                signal_line,
                histogram,
                bb_upper,
                bb_middle,
                bb_lower,
                stoch_k,
                stoch_d,
                obv,
                adx,
                ichimoku_data,
                pivot_points,
                overall_trend,
                trend_strength,
                volatility,
                json.dumps(technical_analysis)
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.log_error(f"Store Technical Indicators - {token} ({timeframe})", str(e))
            conn.rollback()
    
    def _store_statistical_forecast(self, token: str, forecast_data: Dict[str, Any], timeframe: str = "1h") -> None:
        """Store statistical forecast data with timeframe support"""
        conn, cursor = self._get_connection()
        try:
            # Extract forecast and confidence intervals
            forecast_value = forecast_data.get("prediction", 0)
            confidence = forecast_data.get("confidence", [0, 0])
            
            # Get model type from model_info if available
            model_info = forecast_data.get("model_info", {})
            model_type = model_info.get("method", "ARIMA")
            
            # Extract model parameters if available
            model_parameters = json.dumps(model_info)
            
            # Store in database
            cursor.execute("""
                INSERT INTO statistical_forecasts (
                    timestamp, token, timeframe, model_type,
                    forecast_value, confidence_80_lower, confidence_80_upper,
                    confidence_95_lower, confidence_95_upper, 
                    model_parameters, input_data_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                token,
                timeframe,
                model_type,
                forecast_value,
                confidence[0],  # 80% confidence lower
                confidence[1],  # 80% confidence upper
                confidence[0] * 0.9,  # Approximate 95% confidence lower
                confidence[1] * 1.1,  # Approximate 95% confidence upper
                model_parameters,
                "{}"   # Input data summary (empty for now)
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.log_error(f"Store Statistical Forecast - {token} ({timeframe})", str(e))
            conn.rollback()

    def _store_ml_forecast(self, token: str, forecast_data: Dict[str, Any], timeframe: str = "1h") -> None:
        """Store machine learning forecast data with timeframe support"""
        conn, cursor = self._get_connection()
        try:
            # Extract forecast and confidence intervals
            forecast_value = forecast_data.get("prediction", 0)
            confidence = forecast_data.get("confidence", [0, 0])
            
            # Get model type and parameters if available
            model_info = forecast_data.get("model_info", {})
            model_type = model_info.get("method", "RandomForest")
            
            # Extract feature importance if available
            feature_importance = json.dumps(forecast_data.get("feature_importance", {}))
            
            # Store model parameters
            model_parameters = json.dumps(model_info)
            
            # Store in database
            cursor.execute("""
                INSERT INTO ml_forecasts (
                    timestamp, token, timeframe, model_type,
                    forecast_value, confidence_80_lower, confidence_80_upper,
                    confidence_95_lower, confidence_95_upper, 
                    feature_importance, model_parameters
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                token,
                timeframe,
                model_type,
                forecast_value,
                confidence[0],  # 80% confidence lower
                confidence[1],  # 80% confidence upper
                confidence[0] * 0.9,  # Approximate 95% confidence lower
                confidence[1] * 1.1,  # Approximate 95% confidence upper
                feature_importance,
                model_parameters
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.log_error(f"Store ML Forecast - {token} ({timeframe})", str(e))
            conn.rollback()
    
    def _store_claude_prediction(self, token: str, prediction_data: Dict[str, Any], timeframe: str = "1h") -> None:
        """Store Claude AI prediction data with timeframe support"""
        conn, cursor = self._get_connection()
        try:
            # Extract prediction details
            prediction = prediction_data.get("prediction", {})
            rationale = prediction_data.get("rationale", "")
            sentiment = prediction_data.get("sentiment", "NEUTRAL")
            key_factors = json.dumps(prediction_data.get("key_factors", []))
            
            # Default Claude model
            claude_model = "claude-3-7-sonnet-20250219"
            
            # Store inputs if available
            input_data = json.dumps(prediction_data.get("inputs", {}))
            
            # Store in database
            cursor.execute("""
                INSERT INTO claude_predictions (
                    timestamp, token, timeframe, claude_model,
                    prediction_value, confidence_level, sentiment,
                    rationale, key_factors, input_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                token,
                timeframe,
                claude_model,
                prediction.get("price", 0),
                prediction.get("confidence", 70),
                sentiment,
                rationale,
                key_factors,
                input_data
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.log_error(f"Store Claude Prediction - {token} ({timeframe})", str(e))
            conn.rollback()
    
    def _update_timeframe_metrics(self, token: str, timeframe: str, prediction_data: Dict[str, Any]) -> None:
        """Update timeframe metrics based on new prediction"""
        conn, cursor = self._get_connection()
        
        try:
            # Get current metrics for this token and timeframe
            cursor.execute("""
                SELECT * FROM timeframe_metrics
                WHERE token = ? AND timeframe = ?
            """, (token, timeframe))
            
            metrics = cursor.fetchone()
            
            # Get prediction performance
            performance = self.get_prediction_performance(token=token, timeframe=timeframe)
            
            if performance:
                avg_accuracy = performance[0]["accuracy_rate"]
                total_count = performance[0]["total_predictions"]
                correct_count = performance[0]["correct_predictions"]
            else:
                avg_accuracy = 0
                total_count = 0
                correct_count = 0
            
            # Extract model weights
            model_weights = prediction_data.get("model_weights", {})
            
            # Determine best model
            if model_weights:
                best_model = max(model_weights.items(), key=lambda x: x[1])[0]
            else:
                best_model = "unknown"
            
            if metrics:
                # Update existing metrics
                cursor.execute("""
                    UPDATE timeframe_metrics
                    SET avg_accuracy = ?,
                        total_count = ?,
                        correct_count = ?,
                        model_weights = ?,
                        best_model = ?,
                        last_updated = ?
                    WHERE token = ? AND timeframe = ?
                """, (
                    avg_accuracy,
                    total_count,
                    correct_count,
                    json.dumps(model_weights),
                    best_model,
                    datetime.now(),
                    token,
                    timeframe
                ))
            else:
                # Insert new metrics
                cursor.execute("""
                    INSERT INTO timeframe_metrics (
                        timestamp, token, timeframe, avg_accuracy,
                        total_count, correct_count, model_weights,
                        best_model, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now(),
                    token,
                    timeframe,
                    avg_accuracy,
                    total_count,
                    correct_count,
                    json.dumps(model_weights),
                    best_model,
                    datetime.now()
                ))
            
            conn.commit()
            
        except Exception as e:
            logger.log_error(f"Update Timeframe Metrics - {token} ({timeframe})", str(e))
            conn.rollback()
    
                    





