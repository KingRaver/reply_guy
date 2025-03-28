#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Any, Union, Tuple
import sys
import os
import time
import requests
import re
import numpy as np
from datetime import datetime, timedelta
import anthropic
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.keys import Keys
import random
import statistics
import threading
import queue
import json

from utils.logger import logger
from utils.browser import browser
from config import config
from coingecko_handler import CoinGeckoHandler
from mood_config import MoodIndicators, determine_advanced_mood, Mood, MemePhraseGenerator
from meme_phrases import MEME_PHRASES
from prediction_engine import PredictionEngine

# Import new modules
from timeline_scraper import TimelineScraper
from reply_handler import ReplyHandler
from content_analyzer import ContentAnalyzer

# First define the CryptoAnalysisBot class
class CryptoAnalysisBot:
    def __init__(self) -> None:
        self.browser = browser
        self.config = config
        self.claude_client = anthropic.Client(api_key=self.config.CLAUDE_API_KEY)
        self.coingecko = CoinGeckoHandler()
    # Your CryptoAnalysisBot implementation here
    # ...

# THEN define the IntelligentReplyBot class, which inherits from CryptoAnalysisBot
class IntelligentReplyBot(CryptoAnalysisBot):
    """
    Enhanced bot that combines crypto analysis with intelligent replies to X posts
    Extends the base CryptoAnalysisBot with timeline scraping and reply capabilities
    """
    
    def __init__(self) -> None:
        """Initialize the intelligent reply bot with additional components"""
        # Initialize the base class first
        super().__init__()
        
        # Initialize timeline scraper
        self.timeline_scraper = TimelineScraper(self.browser, self.config, self.config.db)
        
        # Initialize reply handler
        self.reply_handler = ReplyHandler(
            self.browser, 
            self.config, 
            self.claude_client, 
            self.coingecko,
            self.config.db
        )
        
        # Initialize content analyzer
        self.content_analyzer = ContentAnalyzer(self.config, self.config.db)
        
        # Reply-specific state tracking
        self.last_reply_time = datetime.now()
        self.last_trend_check_time = datetime.now()
        self.last_account_check_time = datetime.now()
        self.active_reply_cycle = False
        
        # Reply configuration
        self.reply_interval_minutes = 60  # Default to hourly
        self.max_replies_per_cycle = 10   # Maximum replies per hour
        self.trend_check_interval_hours = 3  # Check trending topics every 3 hours
        self.account_check_interval_hours = 4  # Check specific accounts every 4 hours
        
        # Market accounts to monitor specifically
        self.market_accounts = []  # Will be populated from config or database
        
        # Initialize reply statistics
        self.reply_stats = {
            'total_replies': 0,
            'successful_replies': 0,
            'market_replies': 0,
            'timeline_replies': 0,
            'trend_replies': 0,
            'account_replies': 0
        }
        
        logger.logger.info("Intelligent Reply Bot initialized")
# --- IntelligentReplyBot Methods ---
    
    def run_reply_cycle(self) -> bool:
        """
        Run a complete reply cycle: scrape timeline, analyze posts, generate and send replies
        
        Returns:
            bool: True if the cycle ran successfully, False otherwise
        """
        try:
            # Prevent overlapping cycles
            if self.active_reply_cycle:
                logger.logger.warning("Reply cycle already in progress, skipping")
                return False
                
            self.active_reply_cycle = True
            logger.logger.info("Starting reply cycle")
            
            # Get latest market data for context in replies
            market_data = self._get_crypto_data()
            if not market_data:
                logger.logger.error("Failed to get market data for reply cycle")
                self.active_reply_cycle = False
                return False
                
            # 1. Scrape timeline posts
            timeline_posts = self.timeline_scraper.scrape_timeline(count=20)
            logger.logger.info(f"Scraped {len(timeline_posts)} posts from timeline")
            
            # 2. Analyze for market-related content
            analyzed_posts = self.content_analyzer.analyze_multiple_posts(timeline_posts)
            
            # 3. Filter for market-related posts
            market_posts = [post for post in analyzed_posts 
                           if post.get('analysis', {}).get('is_market_related', False)]
            logger.logger.info(f"Found {len(market_posts)} market-related posts")
            
            # 4. Filter out posts we've already replied to
            market_posts = self.timeline_scraper.filter_already_replied_posts(market_posts)
            logger.logger.info(f"{len(market_posts)} market-related posts available to reply to")
            
            # 5. Prioritize posts
            prioritized_posts = self.timeline_scraper.prioritize_posts(market_posts)
            
            # 6. Generate and post replies
            successful_replies = self.reply_handler.reply_to_posts(
                prioritized_posts, 
                market_data,
                max_replies=self.max_replies_per_cycle
            )
            
            # 7. Update statistics
            self.reply_stats['total_replies'] += self.max_replies_per_cycle
            self.reply_stats['successful_replies'] += successful_replies
            self.reply_stats['market_replies'] += successful_replies
            self.reply_stats['timeline_replies'] += successful_replies
            
            # 8. Update last reply time
            self.last_reply_time = datetime.now()
            
            logger.logger.info(f"Reply cycle completed: {successful_replies} replies sent")
            self.active_reply_cycle = False
            return True
            
        except Exception as e:
            logger.log_error("Reply Cycle", str(e))
            self.active_reply_cycle = False
            return False
    
    def run_trending_topics_cycle(self) -> bool:
        """
        Run a cycle to find and reply to trending market topics
        
        Returns:
            bool: True if the cycle ran successfully, False otherwise
        """
        try:
            # Check if we should run the trending topics cycle
            if not self._should_check_trending_topics():
                return False
                
            logger.logger.info("Starting trending topics reply cycle")
            
            # Get latest market data for context in replies
            market_data = self._get_crypto_data()
            if not market_data:
                logger.logger.error("Failed to get market data for trending topics")
                return False
                
            # 1. Scrape additional posts for trend analysis
            timeline_posts = self.timeline_scraper.scrape_timeline(count=30)
            
            # 2. Analyze posts for market content
            analyzed_posts = self.content_analyzer.analyze_multiple_posts(timeline_posts)
            
            # 3. Get trending topics
            trending_topics = self.content_analyzer.get_trending_market_topics(min_count=2)
            trending_tokens = self.content_analyzer.get_trending_tokens(min_count=2)
            
            logger.logger.info(f"Found {len(trending_topics)} trending topics and {len(trending_tokens)} trending tokens")
            
            # 4. Find posts related to trending topics/tokens
            trending_posts = []
            
            # Add posts related to trending topics
            for topic, _ in trending_topics:
                for post in analyzed_posts:
                    post_topics = post.get('analysis', {}).get('topics', {})
                    if topic in post_topics and post not in trending_posts:
                        trending_posts.append(post)
                        
            # Add posts mentioning trending tokens
            for token, _ in trending_tokens:
                for post in analyzed_posts:
                    mentioned_tokens = post.get('analysis', {}).get('mentioned_tokens', [])
                    if token in mentioned_tokens and post not in trending_posts:
                        trending_posts.append(post)
            
            # 5. Filter out posts we've already replied to
            trending_posts = self.timeline_scraper.filter_already_replied_posts(trending_posts)
            
            if not trending_posts:
                logger.logger.info("No unreplied trending posts found")
                return False
                
            # 6. Prioritize posts
            prioritized_trending_posts = self.timeline_scraper.prioritize_posts(trending_posts)
            
            # 7. Generate and post replies
            successful_replies = self.reply_handler.reply_to_posts(
                prioritized_trending_posts, 
                market_data,
                max_replies=min(5, len(prioritized_trending_posts))  # Limit to 5 trend replies
            )
            
            # 8. Update statistics
            self.reply_stats['total_replies'] += successful_replies
            self.reply_stats['successful_replies'] += successful_replies
            self.reply_stats['trend_replies'] += successful_replies
            
            # 9. Update last trend check time
            self.last_trend_check_time = datetime.now()
            
            logger.logger.info(f"Trending topics cycle completed: {successful_replies} replies sent")
            return True
            
        except Exception as e:
            logger.log_error("Trending Topics Cycle", str(e))
            return False
    
    def run_market_accounts_cycle(self) -> bool:
        """
        Run a cycle to find and reply to specific market-focused accounts
        
        Returns:
            bool: True if the cycle ran successfully, False otherwise
        """
        try:
            # Check if we should run the market accounts cycle
            if not self._should_check_market_accounts():
                return False
                
            # Check if we have market accounts to monitor
            if not self.market_accounts:
                # Load market accounts from database or config
                self._load_market_accounts()
                
            if not self.market_accounts:
                logger.logger.warning("No market accounts configured for monitoring")
                return False
                
            logger.logger.info(f"Starting market accounts reply cycle for {len(self.market_accounts)} accounts")
            
            # Get latest market data for context in replies
            market_data = self._get_crypto_data()
            if not market_data:
                logger.logger.error("Failed to get market data for market accounts")
                return False
                
            # Variables to track posts and replies
            account_posts = []
            successful_replies = 0
            
            # Process each account
            for account in self.market_accounts:
                try:
                    # Navigate to account page
                    account_url = f"https://twitter.com/{account}"
                    self.browser.driver.get(account_url)
                    time.sleep(3)  # Allow page to load
                    
                    # Scrape recent posts from the account
                    post_elements = self.browser.driver.find_elements(By.CSS_SELECTOR, '[data-testid="tweet"]')
                    
                    # Process up to 5 recent posts per account
                    for i, post_element in enumerate(post_elements[:5]):
                        try:
                            # Extract post data
                            post_data = self.timeline_scraper.extract_post_data(post_element)
                            
                            if post_data:
                                # Analyze for market content
                                analysis = self.content_analyzer.analyze_post(post_data)
                                post_data['analysis'] = analysis
                                
                                # Check if market-related and we haven't replied
                                if (analysis.get('is_market_related', False) and 
                                    not self.timeline_scraper.filter_already_replied_posts([post_data])):
                                    account_posts.append(post_data)
                        except Exception as e:
                            logger.logger.warning(f"Error processing post from {account}: {str(e)}")
                            continue
                            
                except Exception as e:
                    logger.logger.warning(f"Error processing account {account}: {str(e)}")
                    continue
            
            # Check if we found any posts to reply to
            if not account_posts:
                logger.logger.info("No unreplied market account posts found")
                return False
                
            # Prioritize posts
            prioritized_account_posts = self.timeline_scraper.prioritize_posts(account_posts)
            
            # Generate and post replies
            successful_replies = self.reply_handler.reply_to_posts(
                prioritized_account_posts, 
                market_data,
                max_replies=min(5, len(prioritized_account_posts))  # Limit to 5 account replies
            )
            
            # Update statistics
            self.reply_stats['total_replies'] += successful_replies
            self.reply_stats['successful_replies'] += successful_replies
            self.reply_stats['account_replies'] += successful_replies
            
            # Update last account check time
            self.last_account_check_time = datetime.now()
            
            logger.logger.info(f"Market accounts cycle completed: {successful_replies} replies sent")
            return True
            
        except Exception as e:
            logger.log_error("Market Accounts Cycle", str(e))
            return False
    
    def _should_run_reply_cycle(self) -> bool:
        """
        Determine if it's time to run a reply cycle
        
        Returns:
            bool: True if a reply cycle should be run, False otherwise
        """
        # Check time since last reply
        time_since_last_reply = (datetime.now() - self.last_reply_time).total_seconds() / 60
        should_run = time_since_last_reply >= self.reply_interval_minutes
        
        if should_run:
            logger.logger.debug(f"Time to run reply cycle ({time_since_last_reply:.1f} minutes since last cycle)")
        else:
            minutes_to_wait = self.reply_interval_minutes - time_since_last_reply
            logger.logger.debug(f"Not time for reply cycle yet, {minutes_to_wait:.1f} minutes remaining")
            
        return should_run
    
    def _should_check_trending_topics(self) -> bool:
        """
        Determine if it's time to check trending topics
        
        Returns:
            bool: True if trending topics should be checked, False otherwise
        """
        # Check time since last trend check
        time_since_last_check = (datetime.now() - self.last_trend_check_time).total_seconds() / 3600
        return time_since_last_check >= self.trend_check_interval_hours
    
    def _should_check_market_accounts(self) -> bool:
        """
        Determine if it's time to check market accounts
        
        Returns:
            bool: True if market accounts should be checked, False otherwise
        """
        # Check time since last account check
        time_since_last_check = (datetime.now() - self.last_account_check_time).total_seconds() / 3600
        return time_since_last_check >= self.account_check_interval_hours
    
    def _load_market_accounts(self) -> None:
        """Load market accounts from database or config"""
        try:
            # First try to load from database if available
            if self.config.db:
                # Assuming a get_market_accounts method exists or could be added
                accounts = self.config.db.get_market_accounts()
                if accounts:
                    self.market_accounts = accounts
                    logger.logger.info(f"Loaded {len(accounts)} market accounts from database")
                    return
                    
            # Fallback to predefined accounts
            self.market_accounts = [
                'cz_binance',
                'saylor',
                'VitalikButerin',
                'SBF_FTX',
                'Chainlinkgod',
                'CryptoHayes',
                'AltcoinPsycho',
                'CryptoCapo_',
                'galaxy_digital',
                'APompliano'
            ]
            logger.logger.info(f"Loaded {len(self.market_accounts)} default market accounts")
            
        except Exception as e:
            logger.log_error("Load Market Accounts", str(e))
            # Use a minimal default list if all else fails
            self.market_accounts = ['cz_binance', 'VitalikButerin', 'APompliano']

    def start(self) -> None:
        """
        Main bot execution loop with intelligent reply functionality
        Override of the base class start method
        """
        try:
            retry_count = 0
            max_setup_retries = 3
            
            # Start the prediction thread early
            self._start_prediction_thread()
            
            # Load saved timeframe state
            self._load_saved_timeframe_state()
            
            # Initialize the browser and login
            while retry_count < max_setup_retries:
                if not self.browser.initialize_driver():
                    retry_count += 1
                    logger.logger.warning(f"Browser initialization attempt {retry_count} failed, retrying...")
                    time.sleep(10)
                    continue
                    
                if not self._login_to_twitter():
                    retry_count += 1
                    logger.logger.warning(f"Twitter login attempt {retry_count} failed, retrying...")
                    time.sleep(15)
                    continue
                    
                break
            
            if retry_count >= max_setup_retries:
                raise Exception("Failed to initialize bot after maximum retries")

            logger.logger.info("Bot initialized successfully with intelligent reply capabilities")
            
            # Log configuration settings
            logger.logger.info(f"Reply interval: {self.reply_interval_minutes} minutes")
            logger.logger.info(f"Max replies per cycle: {self.max_replies_per_cycle}")
            logger.logger.info(f"Trend check interval: {self.trend_check_interval_hours} hours")
            logger.logger.info(f"Account check interval: {self.account_check_interval_hours} hours")
            
            # Initialize timestamps for scheduling
            self.last_analysis_time = datetime.now()
            last_cleanup_time = datetime.now()
            
            # Main execution loop
            while True:
                try:
                    # 1. Check if we should run a reply cycle
                    if self._should_run_reply_cycle():
                        # Run the reply cycle
                        self.run_reply_cycle()
                    
                    # 2. Check if we should look for trending topics
                    if self._should_check_trending_topics():
                        # Run the trending topics cycle
                        self.run_trending_topics_cycle()
                    
                    # 3. Check if we should monitor market accounts
                    if self._should_check_market_accounts():
                        # Run the market accounts cycle
                        self.run_market_accounts_cycle()
                    
                    # 4. Run regular crypto analysis functionalities on a schedule
                    # Only run analysis if enough time has passed since last analysis
                    time_since_last_analysis = (datetime.now() - self.last_analysis_time).total_seconds()
                    if time_since_last_analysis >= self.config.BASE_INTERVAL:
                        # Get market data
                        market_data = self._get_crypto_data()
                        
                        if market_data:
                            # Run the analysis cycle from the base class
                            self._run_analysis_cycle()
                            
                            # Also check for expired predictions
                            self._evaluate_expired_timeframe_predictions()
                            
                            # Update last analysis time
                            self.last_analysis_time = datetime.now()
                    
                    # 5. Periodically run cleanup (once per day)
                    hours_since_cleanup = (datetime.now() - last_cleanup_time).total_seconds() / 3600
                    if hours_since_cleanup >= 24:
                        # Run database cleanup
                        if self.config.db:
                            self.config.db.cleanup_old_data(days_to_keep=7)
                            
                        # Update cleanup time
                        last_cleanup_time = datetime.now()
                    
                    # Sleep until next check
                    # Use a shorter interval than the base analysis interval to be responsive for replies
                    sleep_time = min(60, self.config.BASE_INTERVAL / 2)
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    logger.log_error("Main Loop", str(e), exc_info=True)
                    time.sleep(60)  # Shorter sleep on error
                    continue

        except KeyboardInterrupt:
            logger.logger.info("Bot stopped by user")
        except Exception as e:
            logger.log_error("Bot Execution", str(e))
        finally:
            self._cleanup()
    
    def get_reply_stats(self) -> Dict[str, Any]:
        """
        Get statistics about reply operations
        
        Returns:
            Dict with reply statistics
        """
        stats = self.reply_stats.copy()
        
        # Calculate percentages
        if stats['total_replies'] > 0:
            stats['success_rate'] = (stats['successful_replies'] / stats['total_replies']) * 100
        else:
            stats['success_rate'] = 0
            
        # Add timing information
        stats['last_reply_time'] = self.last_reply_time
        stats['last_trend_check_time'] = self.last_trend_check_time
        stats['last_account_check_time'] = self.last_account_check_time
        
        # Add configuration settings
        stats['reply_interval_minutes'] = self.reply_interval_minutes
        stats['max_replies_per_cycle'] = self.max_replies_per_cycle
        stats['trend_check_interval_hours'] = self.trend_check_interval_hours
        stats['account_check_interval_hours'] = self.account_check_interval_hours
        
        return stats
    
    def update_reply_settings(self, settings: Dict[str, Any]) -> None:
        """
        Update reply cycle settings
        
        Args:
            settings: Dictionary with settings to update
        """
        try:
            # Update reply interval
            if 'reply_interval_minutes' in settings:
                new_interval = int(settings['reply_interval_minutes'])
                if new_interval >= 10:  # Minimum 10 minutes between cycles
                    self.reply_interval_minutes = new_interval
                    logger.logger.info(f"Reply interval updated to {new_interval} minutes")
                    
            # Update max replies per cycle
            if 'max_replies_per_cycle' in settings:
                new_max = int(settings['max_replies_per_cycle'])
                if new_max >= 1:  # At least 1 reply per cycle
                    self.max_replies_per_cycle = new_max
                    logger.logger.info(f"Max replies per cycle updated to {new_max}")
                    
            # Update trend check interval
            if 'trend_check_interval_hours' in settings:
                new_interval = int(settings['trend_check_interval_hours'])
                if new_interval >= 1:  # At least 1 hour
                    self.trend_check_interval_hours = new_interval
                    logger.logger.info(f"Trend check interval updated to {new_interval} hours")
                    
            # Update account check interval
            if 'account_check_interval_hours' in settings:
                new_interval = int(settings['account_check_interval_hours'])
                if new_interval >= 1:  # At least 1 hour
                    self.account_check_interval_hours = new_interval
                    logger.logger.info(f"Account check interval updated to {new_interval} hours")
                    
            # Update market accounts
            if 'market_accounts' in settings and settings['market_accounts']:
                self.market_accounts = settings['market_accounts']
                logger.logger.info(f"Market accounts updated: {len(self.market_accounts)} accounts")
                
            # Store settings in database if available
            if self.config.db:
                self._store_reply_settings()
                
        except Exception as e:
            logger.log_error("Update Reply Settings", str(e))
    
    def _store_reply_settings(self) -> None:
        """Store reply settings in database"""
        try:
            if not self.config.db:
                return
                
            settings = {
                'reply_interval_minutes': self.reply_interval_minutes,
                'max_replies_per_cycle': self.max_replies_per_cycle,
                'trend_check_interval_hours': self.trend_check_interval_hours,
                'account_check_interval_hours': self.account_check_interval_hours,
                'market_accounts': self.market_accounts,
                'updated_at': datetime.now().isoformat()
            }
            
            # Use the generic JSON storage method
            self.config.db._store_json_data(
                data_type="reply_settings",
                data=settings
            )
            
            logger.logger.debug("Saved reply settings to database")
            
        except Exception as e:
            logger.log_error("Store Reply Settings", str(e))
    
    def _load_reply_settings(self) -> None:
        """Load reply settings from database"""
        try:
            if not self.config.db:
                return
                
            # Query for the latest settings
            conn, cursor = self.config.db._get_connection()
            
            cursor.execute("""
                SELECT data 
                FROM generic_json_data 
                WHERE data_type = 'reply_settings'
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            
            result = cursor.fetchone()
            
            if not result:
                logger.logger.info("No saved reply settings found")
                return
                
            # Parse the settings
            settings_json = result[0]
            settings = json.loads(settings_json)
            
            # Apply settings
            self.update_reply_settings(settings)
            
            logger.logger.info("Loaded reply settings from database")
            
        except Exception as e:
            logger.log_error("Load Reply Settings", str(e))

        def _initialize_reply_database(self) -> None:
             """
             Initialize database tables for reply tracking if they don't exist
             """
        if not self.config.db:  # This line and everything below needs to be indented
            logger.logger.warning("No database connection available for reply tracking")
            return
        
    conn, cursor = self.config.db._get_connection()
        try:
            # Table for tracking replied-to posts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS replied_posts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_id TEXT NOT NULL,
                    post_url TEXT,
                    post_author TEXT,
                    post_text TEXT,
                    reply_text TEXT,
                    reply_time DATETIME NOT NULL,
                    market_related BOOLEAN DEFAULT 0,
                    reply_type TEXT,
                    engagement_data JSON
                )
            """)
            
            # Table for tracking reply performance
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reply_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_id TEXT NOT NULL,
                    reply_id INTEGER,
                    likes INTEGER DEFAULT 0,
                    replies INTEGER DEFAULT 0,
                    reposts INTEGER DEFAULT 0,
                    check_time DATETIME NOT NULL,
                    FOREIGN KEY (reply_id) REFERENCES replied_posts(id)
                )
            """)
            
            # Table for tracking market accounts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_accounts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_handle TEXT NOT NULL UNIQUE,
                    account_name TEXT,
                    account_category TEXT,
                    follower_count INTEGER,
                    last_engagement_score REAL,
                    last_checked DATETIME,
                    active BOOLEAN DEFAULT 1
                )
            """)
            
            # Table for tracking content analysis results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS content_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_id TEXT NOT NULL,
                    post_url TEXT,
                    post_author TEXT,
                    market_relevance REAL,
                    sentiment TEXT,
                    sentiment_score REAL,
                    mentioned_tokens TEXT,
                    topics TEXT,
                    analysis_time DATETIME NOT NULL,
                    raw_analysis JSON
                )
            """)
            
            # Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_replied_posts_post_id ON replied_posts(post_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_replied_posts_author ON replied_posts(post_author)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_replied_posts_time ON replied_posts(reply_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reply_performance_post_id ON reply_performance(post_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_accounts_handle ON market_accounts(account_handle)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_analysis_post_id ON content_analysis(post_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_analysis_author ON content_analysis(post_author)")
            
            conn.commit()
            logger.logger.info("Reply database tables initialized")
            
        except Exception as e:
            logger.log_error("Initialize Reply Database", str(e))
            conn.rollback()
    
    def store_reply(self, post_id: str, post_url: str, post_author: str, 
                   post_text: str, reply_text: str, reply_type: str = "timeline",
                   market_related: bool = True) -> int:
        """
        Store a reply in the database
        
        Args:
            post_id: Unique identifier for the post
            post_url: URL to the post
            post_author: Author of the post
            post_text: Content of the post
            reply_text: Content of our reply
            reply_type: Type of reply (timeline, trend, account)
            market_related: Whether the post was market-related
            
        Returns:
            ID of the stored reply or 0 if storing failed
        """
        if not self.config.db:
            return 0
            
        conn, cursor = self.config.db._get_connection()
        reply_id = 0
        
        try:
            # Prepare engagement data structure
            engagement_data = {
                'initial_likes': 0,
                'initial_replies': 0,
                'initial_reposts': 0,
                'last_checked': datetime.now().isoformat()
            }
            
            # Insert reply record
            cursor.execute("""
                INSERT INTO replied_posts (
                    post_id, post_url, post_author, post_text, reply_text,
                    reply_time, market_related, reply_type, engagement_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                post_id,
                post_url,
                post_author,
                post_text,
                reply_text,
                datetime.now(),
                1 if market_related else 0,
                reply_type,
                json.dumps(engagement_data)
            ))
            
            conn.commit()
            reply_id = cursor.lastrowid
            
            logger.logger.debug(f"Stored reply to {post_author} with ID {reply_id}")
            return reply_id
            
        except Exception as e:
            logger.log_error("Store Reply", str(e))
            conn.rollback()
            return 0
    
    def check_if_post_replied(self, post_id: str, post_url: Optional[str] = None) -> bool:
        """
        Check if we've already replied to a post
        
        Args:
            post_id: Unique identifier for the post
            post_url: URL to the post (optional fallback)
            
        Returns:
            True if we've already replied, False otherwise
        """
        if not self.config.db:
            return False
            
        conn, cursor = self.config.db._get_connection()
        
        try:
            # Check by post ID first
            cursor.execute("""
                SELECT id FROM replied_posts
                WHERE post_id = ?
            """, (post_id,))
            
            result = cursor.fetchone()
            if result:
                return True
                
            # If post_url is provided, check by URL as fallback
            if post_url:
                cursor.execute("""
                    SELECT id FROM replied_posts
                    WHERE post_url = ?
                """, (post_url,))
                
                result = cursor.fetchone()
            if result:
                return True
                    
            return False
            
        except Exception as e:
            logger.log_error("Check Post Replied", str(e))
            return False
    
    def store_content_analysis(self, post_id: str, post_url: str, post_author: str, 
                             post_text: str, analysis_data: Dict[str, Any]) -> int:
        """
        Store content analysis results in database
        
        Args:
            post_id: Unique identifier for the post
            post_url: URL to the post
            post_author: Author of the post
            post_text: Content of the post
            analysis_data: Analysis results
            
        Returns:
            ID of the stored analysis or 0 if storing failed
        """
        if not self.config.db:
            return 0
            
        conn, cursor = self.config.db._get_connection()
        analysis_id = 0
        
        try:
            # Extract key analysis fields
            market_relevance = analysis_data.get('market_relevance', 0.0)
            sentiment = analysis_data.get('sentiment', {}).get('sentiment', 'neutral')
            sentiment_score = analysis_data.get('sentiment', {}).get('score', 0.0)
            
            # Convert list of tokens to comma-separated string
            mentioned_tokens = ','.join(analysis_data.get('mentioned_tokens', []))
            
            # Convert topics dict to comma-separated string of keys
            topics = ','.join(analysis_data.get('topics', {}).keys())
            
            # Insert analysis record
            cursor.execute("""
                INSERT INTO content_analysis (
                    post_id, post_url, post_author, market_relevance,
                    sentiment, sentiment_score, mentioned_tokens, topics,
                    analysis_time, raw_analysis
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                post_id,
                post_url,
                post_author,
                market_relevance,
                sentiment,
                sentiment_score,
                mentioned_tokens,
                topics,
                datetime.now(),
                json.dumps(analysis_data)
            ))
            
            conn.commit()
            analysis_id = cursor.lastrowid
            
            logger.logger.debug(f"Stored content analysis for post by {post_author} with ID {analysis_id}")
            return analysis_id
            
        except Exception as e:
            logger.log_error("Store Content Analysis", str(e))
            conn.rollback()
            return 0
    
    def update_reply_performance(self, reply_id: int, likes: int, replies: int, reposts: int) -> bool:
        """
        Update performance metrics for a reply
        
        Args:
            reply_id: ID of the replied post
            likes: Current number of likes
            replies: Current number of replies
            reposts: Current number of reposts
            
        Returns:
            True if update was successful, False otherwise
        """
        if not self.config.db:
            return False
            
        conn, cursor = self.config.db._get_connection()
        
        try:
            # First get the post_id for the reply
            cursor.execute("""
                SELECT post_id FROM replied_posts
                WHERE id = ?
            """, (reply_id,))
            
            result = cursor.fetchone()
            if not result:
                logger.logger.warning(f"Reply ID {reply_id} not found in database")
                return False
                
            post_id = result[0]
            
            # Insert performance record
            cursor.execute("""
                INSERT INTO reply_performance (
                    post_id, reply_id, likes, replies, reposts, check_time
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                post_id,
                reply_id,
                likes,
                replies,
                reposts,
                datetime.now()
            ))
            
            # Also update the engagement data in the replied_posts table
            cursor.execute("""
                SELECT engagement_data FROM replied_posts
                WHERE id = ?
            """, (reply_id,))
            
            engagement_result = cursor.fetchone()
            if engagement_result:
                try:
                    engagement_data = json.loads(engagement_result[0])
                    
                    # Update with latest metrics
                    engagement_data['current_likes'] = likes
                    engagement_data['current_replies'] = replies
                    engagement_data['current_reposts'] = reposts
                    engagement_data['last_checked'] = datetime.now().isoformat()
                    
                    # Calculate engagement growth
                    initial_likes = engagement_data.get('initial_likes', 0)
                    initial_replies = engagement_data.get('initial_replies', 0)
                    initial_reposts = engagement_data.get('initial_reposts', 0)
                    
                    engagement_data['likes_growth'] = likes - initial_likes
                    engagement_data['replies_growth'] = replies - initial_replies
                    engagement_data['reposts_growth'] = reposts - initial_reposts
                    
                    # Update the record
                    cursor.execute("""
                        UPDATE replied_posts
                        SET engagement_data = ?
                        WHERE id = ?
                    """, (json.dumps(engagement_data), reply_id))
                    
                except Exception as e:
                    logger.logger.warning(f"Error updating engagement data: {str(e)}")
            
            conn.commit()
            logger.logger.debug(f"Updated performance for reply ID {reply_id}")
            return True
            
        except Exception as e:
            logger.log_error("Update Reply Performance", str(e))
            conn.rollback()
            return False
    
    def get_market_accounts(self) -> List[str]:
        """
        Get list of market accounts to monitor
        
        Returns:
            List of account handles
        """
        if not self.config.db:
            return []
            
        conn, cursor = self.config.db._get_connection()
        
        try:
            cursor.execute("""
                SELECT account_handle FROM market_accounts
                WHERE active = 1
                ORDER BY last_engagement_score DESC
            """)
            
            results = cursor.fetchall()
            return [result[0] for result in results]
            
        except Exception as e:
            logger.log_error("Get Market Accounts", str(e))
            return []
    
    def add_market_account(self, account_handle: str, account_name: str = "", 
                         account_category: str = "crypto", follower_count: int = 0) -> bool:
        """
        Add a market account to monitor
        
        Args:
            account_handle: Twitter handle (without @)
            account_name: Display name
            account_category: Category (crypto, finance, etc.)
            follower_count: Number of followers
            
        Returns:
            True if account was added, False otherwise
        """
        if not self.config.db:
            return False
            
        conn, cursor = self.config.db._get_connection()
        
        try:
            # Clean up handle
            if account_handle.startswith('@'):
                account_handle = account_handle[1:]
                
            # Check if account already exists
            cursor.execute("""
                SELECT id FROM market_accounts
                WHERE account_handle = ?
            """, (account_handle,))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing account
                cursor.execute("""
                    UPDATE market_accounts
                    SET account_name = ?,
                        account_category = ?,
                        follower_count = ?,
                        active = 1,
                        last_checked = ?
                    WHERE account_handle = ?
                """, (
                    account_name,
                    account_category,
                    follower_count,
                    datetime.now(),
                    account_handle
                ))
            else:
                # Insert new account
                cursor.execute("""
                    INSERT INTO market_accounts (
                        account_handle, account_name, account_category,
                        follower_count, last_checked, active
                    ) VALUES (?, ?, ?, ?, ?, 1)
                """, (
                    account_handle,
                    account_name,
                    account_category,
                    follower_count,
                    datetime.now()
                ))
            
            conn.commit()
            logger.logger.info(f"Added/updated market account: @{account_handle}")
            return True
            
        except Exception as e:
            logger.log_error("Add Market Account", str(e))
            conn.rollback()
            return False
    
    def get_reply_performance_stats(self) -> Dict[str, Any]:
        """
        Get statistics about reply performance
        
        Returns:
            Dictionary with performance statistics
        """
        if not self.config.db:
            return {}
            
        conn, cursor = self.config.db._get_connection()
        
        try:
            stats = {}
            
            # Get total number of replies
            cursor.execute("SELECT COUNT(*) FROM replied_posts")
            stats['total_replies'] = cursor.fetchone()[0]
            
            # Get average engagements
            cursor.execute("""
                SELECT AVG(likes), AVG(replies), AVG(reposts)
                FROM reply_performance
            """)
            
            avg_result = cursor.fetchone()
            stats['avg_likes'] = round(avg_result[0], 1) if avg_result[0] else 0
            stats['avg_replies'] = round(avg_result[1], 1) if avg_result[1] else 0
            stats['avg_reposts'] = round(avg_result[2], 1) if avg_result[2] else 0
            
            # Get top performing replies
            cursor.execute("""
                SELECT rp.post_author, rp.reply_text, perf.likes, perf.replies, perf.reposts
                FROM reply_performance perf
                JOIN replied_posts rp ON perf.reply_id = rp.id
                ORDER BY (perf.likes + perf.replies * 3 + perf.reposts * 2) DESC
                LIMIT 5
            """)
            
            top_replies = []
            for row in cursor.fetchall():
                top_replies.append({
                    'author': row[0],
                    'reply': row[1],
                    'likes': row[2],
                    'replies': row[3],
                    'reposts': row[4]
                })
            
            stats['top_replies'] = top_replies
            
            # Get reply breakdown by type
            cursor.execute("""
                SELECT reply_type, COUNT(*)
                FROM replied_posts
                GROUP BY reply_type
            """)
            
            reply_types = {}
            for row in cursor.fetchall():
                reply_types[row[0]] = row[1]
            
            stats['reply_types'] = reply_types
            
            # Get reply stats by market relatedness
            cursor.execute("""
                SELECT market_related, COUNT(*)
                FROM replied_posts
                GROUP BY market_related
            """)
            
            market_related = {
                'market_related': 0,
                'non_market_related': 0
            }
            
            for row in cursor.fetchall():
                if row[0] == 1:  # True/1 = market related
                    market_related['market_related'] = row[1]
                else:
                    market_related['non_market_related'] = row[1]
            
            stats['market_related'] = market_related
            
            # Get time-based statistics - replies per day over last 7 days
            cursor.execute("""
                SELECT date(reply_time) as reply_date, COUNT(*) as count
                FROM replied_posts
                WHERE reply_time >= datetime('now', '-7 days')
                GROUP BY reply_date
                ORDER BY reply_date
            """)
            
            daily_replies = {}
            for row in cursor.fetchall():
                daily_replies[row[0]] = row[1]
            
            stats['daily_replies'] = daily_replies
            
            # Get sentiment analysis of replied posts
            cursor.execute("""
                SELECT ca.sentiment, COUNT(*)
                FROM content_analysis ca
                JOIN replied_posts rp ON ca.post_id = rp.post_id
                GROUP BY ca.sentiment
            """)
            
            sentiment_breakdown = {}
            for row in cursor.fetchall():
                sentiment_breakdown[row[0]] = row[1]
            
            stats['sentiment_breakdown'] = sentiment_breakdown
            
            # Get most frequently mentioned tokens in replies
            cursor.execute("""
                SELECT mentioned_tokens, COUNT(*) as count
                FROM content_analysis ca
                JOIN replied_posts rp ON ca.post_id = rp.post_id
                WHERE mentioned_tokens != ''
                GROUP BY mentioned_tokens
                ORDER BY count DESC
                LIMIT 10
            """)
            
            token_mentions = {}
            for row in cursor.fetchall():
                # Parse comma-separated tokens
                tokens = row[0].split(',')
                for token in tokens:
                    if token in token_mentions:
                        token_mentions[token] += row[1]
                    else:
                        token_mentions[token] = row[1]
            
            # Sort by frequency and get top 10
            token_mentions = dict(sorted(token_mentions.items(), key=lambda x: x[1], reverse=True)[:10])
            stats['token_mentions'] = token_mentions
            
            return stats
            
        except Exception as e:
            logger.log_error("Get Reply Performance Stats", str(e))
            return {'error': str(e)}
        
    def check_reply_engagement(self, max_replies: int = 10) -> int:
        """
        Check engagement metrics for recent replies
        
        Args:
            max_replies: Maximum number of replies to check
            
        Returns:
            Number of replies checked
        """
        if not self.config.db or not self.browser:
            return 0
            
        conn, cursor = self.config.db._get_connection()
        checked_count = 0
        
        try:
            # Get recent replies that haven't been checked in the last 24 hours
            cursor.execute("""
                SELECT rp.id, rp.post_url, rp.post_id
                FROM replied_posts rp
                LEFT JOIN reply_performance perf ON rp.id = perf.reply_id
                WHERE perf.id IS NULL OR perf.check_time < datetime('now', '-24 hours')
                ORDER BY rp.reply_time DESC
                LIMIT ?
            """, (max_replies,))
            
            replies_to_check = cursor.fetchall()
            
            for reply in replies_to_check:
                reply_id, post_url, post_id = reply
                
                if not post_url:
                    logger.logger.warning(f"No URL for reply ID {reply_id}, skipping engagement check")
                    continue
                
                try:
                    # Navigate to the post
                    self.browser.driver.get(post_url)
                    time.sleep(3)  # Wait for page to load
                    
                    # Find our reply in the replies section
                    # Note: This requires identifying our own replies, which can be complex
                    # For simplicity, we'll just check engagement on the original post
                    
                    # Get metrics from the original post
                    metrics = {
                        'likes': 0,
                        'replies': 0,
                        'reposts': 0
                    }
                    
                    # Find engagement metrics
                    try:
                        metrics_elements = self.browser.driver.find_elements(
                            By.CSS_SELECTOR, '[data-testid="reply"], [data-testid="retweet"], [data-testid="like"]'
                        )
                        
                        for element in metrics_elements:
                            test_id = element.get_attribute('data-testid')
                            aria_label = element.get_attribute('aria-label')
                            
                            if not aria_label:
                                continue
                                
                            # Extract the number from the aria-label
                            number_match = re.search(r'(\d+)', aria_label)
                            count = int(number_match.group(1)) if number_match else 0
                            
                            if 'reply' in test_id:
                                metrics['replies'] = count
                            elif 'retweet' in test_id:
                                metrics['reposts'] = count
                            elif 'like' in test_id:
                                metrics['likes'] = count
                    except Exception as e:
                        logger.logger.warning(f"Error parsing metrics for {post_url}: {str(e)}")
                    
                    # Update performance in database
                    self.update_reply_performance(
                        reply_id,
                        metrics['likes'],
                        metrics['replies'],
                        metrics['reposts']
                    )
                    
                    checked_count += 1
                    
                except Exception as e:
                    logger.logger.warning(f"Error checking engagement for {post_url}: {str(e)}")
                    continue
            
            logger.logger.info(f"Checked engagement for {checked_count} replies")
            return checked_count
            
        except Exception as e:
            logger.log_error("Check Reply Engagement", str(e))
            return 0
    
    def get_best_performing_replies(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the best performing replies based on engagement
        
        Args:
            limit: Maximum number of replies to return
            
        Returns:
            List of best performing replies with engagement data
        """
        if not self.config.db:
            return []
            
        conn, cursor = self.config.db._get_connection()
        
        try:
            cursor.execute("""
                SELECT 
                    rp.id, rp.post_author, rp.post_text, rp.reply_text,
                    rp.reply_time, perf.likes, perf.replies, perf.reposts,
                    (perf.likes + perf.replies * 3 + perf.reposts * 2) as engagement_score
                FROM replied_posts rp
                JOIN reply_performance perf ON rp.id = perf.reply_id
                ORDER BY engagement_score DESC
                LIMIT ?
            """, (limit,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'post_author': row[1],
                    'post_text': row[2],
                    'reply_text': row[3],
                    'reply_time': row[4],
                    'likes': row[5],
                    'replies': row[6],
                    'reposts': row[7],
                    'engagement_score': row[8]
                })
            
            return results
            
        except Exception as e:
            logger.log_error("Get Best Performing Replies", str(e))
            return []

    def generate_prediction_reply(self, post_data: Dict[str, Any], market_data: Dict[str, Any]) -> Optional[str]:
        """
        Generate a reply that incorporates relevant predictions for tokens mentioned in the post
        
        Args:
            post_data: Post data dictionary with analysis
            market_data: Market data from CoinGecko
            
        Returns:
            Generated reply text or None if generation failed
        """
        try:
            # Extract mentioned tokens
            analysis = post_data.get('analysis', {})
            mentioned_tokens = analysis.get('mentioned_tokens', [])
            
            # If no tokens mentioned, use general market reply
            if not mentioned_tokens:
                return self.reply_handler.generate_reply(post_data, market_data)
                
            # Get predictions for the mentioned tokens
            predictions = {}
            timeframe_priority = ["1h", "24h", "7d"]  # Prioritize shorter timeframes
            
            for token in mentioned_tokens:
                # Check each timeframe for predictions
                token_upper = token.upper()
                for timeframe in timeframe_priority:
                    # Check if we have a prediction for this token and timeframe
                    if token_upper in self.timeframe_predictions.get(timeframe, {}):
                        predictions[token_upper] = {
                            'prediction': self.timeframe_predictions[timeframe][token_upper],
                            'timeframe': timeframe
                        }
                        break
                        
                    # Try to find in database if not in memory
                    active_predictions = self.config.db.get_active_predictions(token=token_upper, timeframe=timeframe)
                    if active_predictions:
                        predictions[token_upper] = {
                            'prediction': active_predictions[0],
                            'timeframe': timeframe
                        }
                        break
            
            # If we found predictions, include them in the prompt
            prediction_context = ""
            if predictions:
                prediction_context = "Predictions for mentioned tokens:\n"
                for token, pred_data in predictions.items():
                    pred = pred_data['prediction'].get('prediction', {})
                    timeframe = pred_data['timeframe']
                    
                    if pred:
                        price = pred.get('price', 0)
                        percent_change = pred.get('percent_change', 0)
                        confidence = pred.get('confidence', 0)
                        
                        prediction_context += f"- {token} {timeframe} prediction: "
                        prediction_context += f"${price:.4f} ({percent_change:+.2f}%) with {confidence}% confidence\n"
            
            # Generate reply with prediction context
            reply_text = self.reply_handler.generate_reply(
                post_data,
                market_data,
                additional_context={"predictions": predictions} if predictions else None
            )
            
            return reply_text
            
        except Exception as e:
            logger.log_error("Generate Prediction Reply", str(e))
            return None
    
    def find_prediction_opportunities(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find posts that might be good opportunities to share relevant predictions
        
        Args:
            market_data: Market data from CoinGecko
            
        Returns:
            List of posts that are good prediction sharing opportunities
        """
        try:
            # Get active predictions for all timeframes
            active_predictions = self._get_all_active_predictions()
            if not active_predictions or not any(active_predictions.values()):
                logger.logger.info("No active predictions available for sharing")
                return []
            
            # Get tokens with predictions
            tokens_with_predictions = set()
            for timeframe, predictions in active_predictions.items():
                tokens_with_predictions.update(predictions.keys())
            
            if not tokens_with_predictions:
                return []
                
            # Scrape timeline to find relevant posts
            timeline_posts = self.timeline_scraper.scrape_timeline(count=30)
            
            # Analyze posts
            analyzed_posts = self.content_analyzer.analyze_multiple_posts(timeline_posts)
            
            # Filter for market-related posts
            market_posts = [post for post in analyzed_posts 
                          if post.get('analysis', {}).get('is_market_related', False)]
            
            # Filter out posts we've already replied to
            unreplied_posts = self.timeline_scraper.filter_already_replied_posts(market_posts)
            
            # Find posts mentioning tokens we have predictions for
            prediction_opportunities = []
            for post in unreplied_posts:
                analysis = post.get('analysis', {})
                mentioned_tokens = [token.upper() for token in analysis.get('mentioned_tokens', [])]
                
                # Check if any mentioned tokens have predictions
                matching_tokens = tokens_with_predictions.intersection(mentioned_tokens)
                if matching_tokens:
                    post['prediction_tokens'] = list(matching_tokens)
                    prediction_opportunities.append(post)
            
            # Sort opportunities by engagement score
            prediction_opportunities = sorted(
                prediction_opportunities,
                key=lambda x: x.get('engagement_score', 0),
                reverse=True
            )
            
            return prediction_opportunities
            
        except Exception as e:
            logger.log_error("Find Prediction Opportunities", str(e))
            return []
    
    def reply_with_predictions(self, market_data: Dict[str, Any], max_replies: int = 3) -> int:
        """
        Find and reply to posts with relevant predictions
        
        Args:
            market_data: Market data from CoinGecko
            max_replies: Maximum number of replies to send
            
        Returns:
            Number of successful replies
        """
        try:
            # Find prediction opportunities
            opportunities = self.find_prediction_opportunities(market_data)
            
            if not opportunities:
                logger.logger.info("No prediction sharing opportunities found")
                return 0
            
            # Limit to max_replies
            opportunities = opportunities[:max_replies]
            
            # Generate and post replies
            successful_replies = 0
            
            for post in opportunities:
                # Generate prediction-focused reply
                reply_text = self.generate_prediction_reply(post, market_data)
                
                if not reply_text:
                    logger.logger.warning(f"Failed to generate prediction reply for post by {post.get('author_handle')}")
                    continue
                
                # Post the reply
                if self.reply_handler.post_reply(post, reply_text):
                    successful_replies += 1
                    
                    # Update statistics
                    self.reply_stats['total_replies'] += 1
                    self.reply_stats['successful_replies'] += 1
                    self.reply_stats['market_replies'] += 1
                    
                    # Add a small delay between replies
                    time.sleep(random.uniform(30, 60))
            
            logger.logger.info(f"Posted {successful_replies} prediction replies")
            return successful_replies
            
        except Exception as e:
            logger.log_error("Reply With Predictions", str(e))
            return 0

    def find_market_questions(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find posts containing market-related questions
        
        Args:
            posts: List of post data dictionaries
            
        Returns:
            Posts containing market questions
        """
        try:
            question_posts = []
            
            for post in posts:
                analysis = post.get('analysis', {})
                
                # Check if post is a question and market-related
                if (analysis.get('is_market_related', False) and 
                    analysis.get('is_question', False)):
                    question_posts.append(post)
            
            # Prioritize questions by engagement score
            question_posts = sorted(question_posts, key=lambda x: x.get('engagement_score', 0), reverse=True)
            
            logger.logger.debug(f"Found {len(question_posts)} market-related questions")
            return question_posts
            
        except Exception as e:
            logger.log_error("Find Market Questions", str(e))
            return []
    
    def generate_question_reply(self, post_data: Dict[str, Any], market_data: Dict[str, Any]) -> Optional[str]:
        """
        Generate a detailed reply to a market-related question
        
        Args:
            post_data: Post data dictionary with analysis
            market_data: Market data from CoinGecko
            
        Returns:
            Generated reply text or None if generation failed
        """
        try:
            # Extract key details from the question
            analysis = post_data.get('analysis', {})
            mentioned_tokens = analysis.get('mentioned_tokens', [])
            topics = analysis.get('topics', {})
            sentiment = analysis.get('sentiment', {})
            
            # Create a more detailed prompt for question answering
            question_text = post_data.get('text', '')
            
            # Get predictions for mentioned tokens if available
            predictions_context = ""
            for token in mentioned_tokens:
                token_upper = token.upper()
                token_predictions = {}
                
                # Check each timeframe for predictions
                for timeframe in self.timeframes:
                    # Check if we have a prediction for this token and timeframe
                    if token_upper in self.timeframe_predictions.get(timeframe, {}):
                        token_predictions[timeframe] = self.timeframe_predictions[timeframe][token_upper]
                        
                    # Try to find in database if not in memory
                    active_predictions = self.config.db.get_active_predictions(token=token_upper, timeframe=timeframe)
                    if active_predictions:
                        token_predictions[timeframe] = active_predictions[0]
                
                if token_predictions:
                    predictions_context += f"\n{token_upper} predictions:\n"
                    for tf, pred in token_predictions.items():
                        pred_data = pred.get('prediction', {})
                        price = pred_data.get('price', 0)
                        percent_change = pred_data.get('percent_change', 0)
                        confidence = pred_data.get('confidence', 0)
                        
                        predictions_context += f"- {tf}: ${price:.4f} ({percent_change:+.2f}%) with {confidence}% confidence\n"
            
            # Get token technical analysis if available
            technical_context = ""
            for token in mentioned_tokens:
                token_upper = token.upper()
                if token_upper in market_data:
                    # Get tech analysis for 1h timeframe as default
                    token_data = market_data[token_upper]
                    
                    # Create price history array for technical analysis
                    prices = [token_data['current_price']]
                    if 'sparkline' in token_data and token_data['sparkline']:
                        prices = token_data['sparkline'] + prices
                    
                    # Run technical analysis
                    tech_analysis = self.prediction_engine._technical_indicators.analyze_technical_indicators(
                        prices=prices, 
                        volumes=[token_data['volume']] * len(prices)
                    )
                    
                    if tech_analysis:
                        technical_context += f"\n{token_upper} technical indicators:\n"
                        technical_context += f"- Overall trend: {tech_analysis.get('overall_trend', 'unknown')}\n"
                        technical_context += f"- Trend strength: {tech_analysis.get('trend_strength', 0)}\n"
                        
                        signals = tech_analysis.get('signals', {})
                        for indicator, signal in signals.items():
                            technical_context += f"- {indicator}: {signal}\n"
            
            # Generate a more comprehensive reply using additional context
            additional_context = {
                "is_question": True,
                "question_topics": topics,
                "predictions": predictions_context,
                "technical_analysis": technical_context
            }
            
            # Use the reply handler to generate the reply
            reply_text = self.reply_handler.generate_reply(
                post_data,
                market_data,
                additional_context=additional_context
            )
            
            return reply_text
            
        except Exception as e:
            logger.log_error("Generate Question Reply", str(e))
            return None
    
    def answer_market_questions(self, market_data: Dict[str, Any], max_replies: int = 3) -> int:
        """
        Find and answer market-related questions
        
        Args:
            market_data: Market data from CoinGecko
            max_replies: Maximum number of questions to answer
            
        Returns:
            Number of successful replies
        """
        try:
            # Scrape timeline
            timeline_posts = self.timeline_scraper.scrape_timeline(count=30)
            
            # Analyze posts
            analyzed_posts = self.content_analyzer.analyze_multiple_posts(timeline_posts)
            
            # Find question posts
            question_posts = self.find_market_questions(analyzed_posts)
            
            # Filter out posts we've already replied to
            unreplied_questions = self.timeline_scraper.filter_already_replied_posts(question_posts)
            
            if not unreplied_questions:
                logger.logger.info("No unreplied market questions found")
                return 0
            
            # Limit to max_replies
            unreplied_questions = unreplied_questions[:max_replies]
            
            # Generate and post replies
            successful_replies = 0
            
            for post in unreplied_questions:
                # Generate detailed question reply
                reply_text = self.generate_question_reply(post, market_data)
                
                if not reply_text:
                    logger.logger.warning(f"Failed to generate question reply for post by {post.get('author_handle')}")
                    continue
                
                # Post the reply
                if self.reply_handler.post_reply(post, reply_text):
                    successful_replies += 1
                    
                    # Update statistics
                    self.reply_stats['total_replies'] += 1
                    self.reply_stats['successful_replies'] += 1
                    self.reply_stats['market_replies'] += 1
                    
                    # Add a small delay between replies
                    time.sleep(random.uniform(30, 60))
            
            logger.logger.info(f"Answered {successful_replies} market questions")
            return successful_replies
            
        except Exception as e:
            logger.log_error("Answer Market Questions", str(e))
            return 0

    def identify_trending_market_topics(self) -> Dict[str, Any]:
        """
        Identify trending market topics from timeline
        
        Returns:
            Dictionary with trending topics, tokens, and sentiment
        """
        try:
            # Scrape a larger sample of timeline posts
            timeline_posts = self.timeline_scraper.scrape_timeline(count=50)
            
            # Analyze posts
            analyzed_posts = self.content_analyzer.analyze_multiple_posts(timeline_posts)
            
            # Filter for market-related posts
            market_posts = [post for post in analyzed_posts 
                          if post.get('analysis', {}).get('is_market_related', False)]
            
            # Extract topics
            all_topics = {}
            all_tokens = {}
            sentiment_counts = {
                'bullish': 0,
                'bearish': 0,
                'neutral': 0
            }
            
            for post in market_posts:
                analysis = post.get('analysis', {})
                
                # Collect topics
                for topic, score in analysis.get('topics', {}).items():
                    if topic in all_topics:
                        all_topics[topic] += score
                    else:
                        all_topics[topic] = score
                
                # Collect tokens
                for token in analysis.get('mentioned_tokens', []):
                    if token in all_tokens:
                        all_tokens[token] += 1
                    else:
                        all_tokens[token] = 1
                
                # Collect sentiment
                sentiment = analysis.get('sentiment', {}).get('sentiment', 'neutral')
                if sentiment in sentiment_counts:
                    sentiment_counts[sentiment] += 1
            
            # Sort topics and tokens by frequency
            trending_topics = sorted(all_topics.items(), key=lambda x: x[1], reverse=True)
            trending_tokens = sorted(all_tokens.items(), key=lambda x: x[1], reverse=True)
            
            # Calculate overall market sentiment
            total_sentiment = sum(sentiment_counts.values())
            if total_sentiment > 0:
                market_sentiment = {
                    'bullish': sentiment_counts['bullish'] / total_sentiment * 100,
                    'bearish': sentiment_counts['bearish'] / total_sentiment * 100,
                    'neutral': sentiment_counts['neutral'] / total_sentiment * 100
                }
                
                # Determine primary sentiment
                primary_sentiment = max(market_sentiment, key=market_sentiment.get)
            else:
                market_sentiment = {'bullish': 0, 'bearish': 0, 'neutral': 100}
                primary_sentiment = 'neutral'
            
            return {
                'trending_topics': trending_topics[:10],
                'trending_tokens': trending_tokens[:10],
                'sentiment_counts': sentiment_counts,
                'market_sentiment': market_sentiment,
                'primary_sentiment': primary_sentiment
            }
            
        except Exception as e:
            logger.log_error("Identify Trending Market Topics", str(e))
            return {
                'trending_topics': [],
                'trending_tokens': [],
                'sentiment_counts': {'bullish': 0, 'bearish': 0, 'neutral': 0},
                'market_sentiment': {'bullish': 0, 'bearish': 0, 'neutral': 100},
                'primary_sentiment': 'neutral'
            }
    
    def post_market_sentiment_update(self, market_data: Dict[str, Any]) -> bool:
        """
        Post an update about market sentiment based on timeline analysis
        
        Args:
            market_data: Market data from CoinGecko
            
        Returns:
            True if post was successful, False otherwise
        """
        try:
            # Identify trending topics
            market_trends = self.identify_trending_market_topics()
            
            if not market_trends['trending_topics'] and not market_trends['trending_tokens']:
                logger.logger.warning("No significant market trends identified")
                return False
            
            # Build update text
            sentiment = market_trends['primary_sentiment']
            sentiment_emoji = "📈" if sentiment == 'bullish' else "📉" if sentiment == 'bearish' else "➡️"
            
            update_text = f"X MARKET SENTIMENT UPDATE {sentiment_emoji}\n\n"
            
            # Add sentiment breakdown
            sentiment_pcts = market_trends['market_sentiment']
            update_text += f"Current sentiment: {sentiment_pcts['bullish']:.1f}% bullish, "
            update_text += f"{sentiment_pcts['bearish']:.1f}% bearish, "
            update_text += f"{sentiment_pcts['neutral']:.1f}% neutral\n\n"
            
            # Add trending topics
            if market_trends['trending_topics']:
                update_text += "Trending topics:\n"
                for topic, score in market_trends['trending_topics'][:5]:
                    update_text += f"- {topic}\n"
                update_text += "\n"
            
            # Add trending tokens with price data
            if market_trends['trending_tokens']:
                update_text += "Trending tokens:\n"
                for token, count in market_trends['trending_tokens'][:5]:
                    token_upper = token.upper()
                    if token_upper in market_data:
                        price = market_data[token_upper]['current_price']
                        change = market_data[token_upper]['price_change_percentage_24h']
                        update_text += f"- #{token_upper}: ${price:.4f} ({change:+.2f}%)\n"
                    else:
                        update_text += f"- #{token_upper}\n"
                        
            # Add prediction context if available
            top_tokens = [t[0].upper() for t in market_trends['trending_tokens'][:3]]
            prediction_added = False
            
            for token in top_tokens:
                # Get most recent prediction for any timeframe
                token_prediction = None
                for timeframe in self.timeframes:
                    if token in self.timeframe_predictions.get(timeframe, {}):
                        token_prediction = {
                            'data': self.timeframe_predictions[timeframe][token],
                            'timeframe': timeframe
                        }
                        break
                
                if token_prediction:
                    if not prediction_added:
                        update_text += "\nLatest prediction:\n"
                        prediction_added = True
                        
                    pred_data = token_prediction['data'].get('prediction', {})
                    timeframe = token_prediction['timeframe']
                    
                    price = pred_data.get('price', 0)
                    percent_change = pred_data.get('percent_change', 0)
                    
                    update_text += f"#{token} {timeframe}: ${price:.4f} ({percent_change:+.2f}%)\n"
                    break  # Only add one prediction for brevity
            
            # Post the update
            if self._post_analysis(update_text, timeframe="sentiment"):
                logger.logger.info("Posted market sentiment update")
                
                # Store in database
                storage_data = {
                    'content': update_text,
                    'sentiment': {'market': sentiment},
                    'trigger_type': 'sentiment_update',
                    'price_data': {},
                    'meme_phrases': {},
                    'timeframe': 'sentiment'
                }
                
                self.config.db.store_posted_content(**storage_data)
                return True
            else:
                logger.logger.error("Failed to post market sentiment update")
                return False
            
        except Exception as e:
            logger.log_error("Post Market Sentiment Update", str(e))
            return False

    def refresh_market_data(self) -> Dict[str, Any]:
        """
        Refresh market data and update predictions for active tokens
        
        Returns:
            Updated market data
        """
        try:
            # Get fresh market data
            market_data = self._get_crypto_data()
            if not market_data:
                logger.logger.error("Failed to refresh market data")
                return {}
            
            # Get trending tokens
            trending_analysis = self.identify_trending_market_topics()
            trending_tokens = [t[0].upper() for t in trending_analysis.get('trending_tokens', [])]
            
            # Determine tokens to refresh predictions for
            tokens_to_refresh = set()
            
            # Add trending tokens
            tokens_to_refresh.update(trending_tokens[:5])
            
            # Add tokens with existing predictions
            for timeframe in self.timeframes:
                tokens_to_refresh.update(self.timeframe_predictions.get(timeframe, {}).keys())
            
            # Limit to tokens with market data
            tokens_to_refresh = [t for t in tokens_to_refresh if t in market_data]
            
            # Refresh predictions for selected tokens
            for token in tokens_to_refresh:
                self._queue_predictions_for_all_timeframes(token, market_data)
            
            logger.logger.info(f"Refreshed market data and queued predictions for {len(tokens_to_refresh)} tokens")
            return market_data
            
        except Exception as e:
            logger.log_error("Refresh Market Data", str(e))
            return {}
    
    def run_reply_cycles(self) -> None:
        """
        Run all reply cycles in sequence
        """
        try:
            # Check if it's time to run reply cycles
            if not self._should_run_reply_cycle():
                return
                
            logger.logger.info("Running all reply cycles")
            
            # Refresh market data
            market_data = self.refresh_market_data()
            if not market_data:
                logger.logger.error("Failed to get market data for reply cycles")
                return
            
            # 1. Run the main reply cycle
            main_cycle_success = self.run_reply_cycle()
            
            # 2. Answer market questions with a smaller limit
            question_replies = self.answer_market_questions(market_data, max_replies=2)
            
            # 3. Share predictions for relevant posts
            prediction_replies = self.reply_with_predictions(market_data, max_replies=2)
            
            # 4. Check trending topics and post sentiment update (less frequently)
            hours_since_trend = (datetime.now() - self.last_trend_check_time).total_seconds() / 3600
            if hours_since_trend >= self.trend_check_interval_hours:
                self.post_market_sentiment_update(market_data)
                self.last_trend_check_time = datetime.now()
            
            # 5. Check engagement for previous replies
            self.check_reply_engagement(max_replies=5)
            
            # Update last reply time
            self.last_reply_time = datetime.now()
            
            logger.logger.info(
                f"Completed reply cycles - Main: {main_cycle_success}, "
                f"Questions: {question_replies}, Predictions: {prediction_replies}"
            )
            
        except Exception as e:
            logger.log_error("Run Reply Cycles", str(e))
    
    def _initialize_reply_database(self) -> None:
        """
        Initialize database tables for reply tracking if they don't exist
        """
        if not self.config.db:
            logger.logger.warning("No database connection available for reply tracking")
            return
            
        conn, cursor = self.config.db._get_connection()
        
        try:
            # Table for tracking replied-to posts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS replied_posts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_id TEXT NOT NULL,
                    post_url TEXT,
                    post_author TEXT,
                    post_text TEXT,
                    reply_text TEXT,
                    reply_time DATETIME NOT NULL,
                    market_related BOOLEAN DEFAULT 0,
                    reply_type TEXT,
                    engagement_data JSON
                )
            """)
            
            # Table for tracking reply performance
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reply_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_id TEXT NOT NULL,
                    reply_id INTEGER,
                    likes INTEGER DEFAULT 0,
                    replies INTEGER DEFAULT 0,
                    reposts INTEGER DEFAULT 0,
                    check_time DATETIME NOT NULL,
                    FOREIGN KEY (reply_id) REFERENCES replied_posts(id)
                )
            """)
            
            # Table for tracking market accounts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_accounts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_handle TEXT NOT NULL UNIQUE,
                    account_name TEXT,
                    account_category TEXT,
                    follower_count INTEGER,
                    last_engagement_score REAL,
                    last_checked DATETIME,
                    active BOOLEAN DEFAULT 1
                )
            """)
            
            # Table for tracking content analysis results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS content_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_id TEXT NOT NULL,
                    post_url TEXT,
                    post_author TEXT,
                    market_relevance REAL,
                    sentiment TEXT,
                    sentiment_score REAL,
                    mentioned_tokens TEXT,
                    topics TEXT,
                    analysis_time DATETIME NOT NULL,
                    raw_analysis JSON
                )
            """)
            
            # Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_replied_posts_post_id ON replied_posts(post_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_replied_posts_author ON replied_posts(post_author)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_replied_posts_time ON replied_posts(reply_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reply_performance_post_id ON reply_performance(post_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_accounts_handle ON market_accounts(account_handle)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_analysis_post_id ON content_analysis(post_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_analysis_author ON content_analysis(post_author)")
            
            conn.commit()
            logger.logger.info("Reply database tables initialized")
            
        except Exception as e:
            logger.log_error("Initialize Reply Database", str(e))
            conn.rollback()
            
    def _cleanup_reply_resources(self) -> None:
        """Cleanup reply-specific resources"""
        try:
            # Store reply statistics
            if self.config.db:
                reply_stats_data = {
                    'timestamp': datetime.now().isoformat(),
                    'stats': self.reply_stats
                }
                
                # Use the generic JSON storage method
                self.config.db._store_json_data(
                    data_type="reply_stats",
                    data=reply_stats_data
                )
                
                logger.logger.debug("Stored reply statistics in database")
                
        except Exception as e:
            logger.log_error("Cleanup Reply Resources", str(e))

    def _cleanup(self) -> None:
        """Override of base class cleanup to include reply-specific resources"""
        try:
            # Clean up reply resources
            self._cleanup_reply_resources()
            
            # Call the base class cleanup
            super()._cleanup()
            
        except Exception as e:
            logger.log_error("Cleanup", str(e))

# Main entry point
if __name__ == "__main__":
    try:
        # Use the intelligent reply bot instead of the base bot
        bot = IntelligentReplyBot()
        bot.start()
    except Exception as e:
        logger.log_error("Bot Startup", str(e))                    
