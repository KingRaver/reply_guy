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
from datetime_utils import strip_timezone, ensure_naive_datetimes, safe_datetime_diff
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
from llm_provider import LLMProvider

from utils.logger import logger
from utils.browser import browser
from config import config
from coingecko_handler import CoinGeckoHandler
from mood_config import MoodIndicators, determine_advanced_mood, Mood, MemePhraseGenerator
from meme_phrases import MEME_PHRASES
from prediction_engine import PredictionEngine

# Import modules for reply functionality
from timeline_scraper import TimelineScraper
from reply_handler import ReplyHandler
from content_analyzer import ContentAnalyzer

class CryptoAnalysisBot:
    """
    Enhanced crypto analysis bot with market and tech content capabilities.
    Handles market analysis, predictions, and social media engagement.
    """
    
    def __init__(self) -> None:
        """
        Initialize the crypto analysis bot with improved configuration and tracking.
        Sets up connections to browser, database, and API services.
        """
        self.browser = browser
        self.config = config
        self.llm_provider = LLMProvider(self.config)  
        self.past_predictions = []
        self.meme_phrases = MEME_PHRASES
        self.last_check_time = strip_timezone(datetime.now())
        self.last_market_data = {}
        self.last_reply_time = strip_timezone(datetime.now())
       
        # Multi-timeframe prediction tracking
        self.timeframes = ["1h", "24h", "7d"]
        self.timeframe_predictions = {tf: {} for tf in self.timeframes}
        self.timeframe_last_post = {tf: strip_timezone(datetime.now() - timedelta(hours=3)) for tf in self.timeframes}
       
        # Timeframe posting frequency controls (in hours)
        self.timeframe_posting_frequency = {
            "1h": 1,    # Every hour
            "24h": 6,   # Every 6 hours
            "7d": 24    # Once per day
        }
       
        # Prediction accuracy tracking by timeframe
        self.prediction_accuracy = {tf: {'correct': 0, 'total': 0} for tf in self.timeframes}
       
        # Initialize prediction engine with database and LLM Provider
        self.prediction_engine = PredictionEngine(
            database=self.config.db,
            llm_provider=self.llm_provider
        )
       
        # Create a queue for predictions to process
        self.prediction_queue = queue.Queue()
       
        # Initialize thread for async prediction generation
        self.prediction_thread = None
        self.prediction_thread_running = False
       
        # Initialize CoinGecko handler with 60s cache duration
        self.coingecko = CoinGeckoHandler(
            base_url=self.config.COINGECKO_BASE_URL,
            cache_duration=60
        )

        # Target chains to analyze
        self.target_chains = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'SOL': 'solana',
            'XRP': 'ripple',
            'BNB': 'binancecoin',
            'AVAX': 'avalanche-2',
            'DOT': 'polkadot',
            'UNI': 'uniswap',
            'NEAR': 'near',
            'AAVE': 'aave',
            'FIL': 'filecoin',
            'POL': 'matic-network',
            'TRUMP': 'official-trump',
            'KAITO': 'kaito'
        }

        # All tokens for reference and comparison
        self.reference_tokens = list(self.target_chains.keys())
       
        # Chain name mapping for display
        self.chain_name_mapping = self.target_chains.copy()
       
        self.CORRELATION_THRESHOLD = 0.75  
        self.VOLUME_THRESHOLD = 0.60  
        self.TIME_WINDOW = 24
       
        # Smart money thresholds
        self.SMART_MONEY_VOLUME_THRESHOLD = 1.5  # 50% above average
        self.SMART_MONEY_ZSCORE_THRESHOLD = 2.0  # 2 standard deviations
       
        # Timeframe-specific triggers and thresholds
        self.timeframe_thresholds = {
            "1h": {
                "price_change": 3.0,    # 3% price change for 1h predictions
                "volume_change": 8.0,   # 8% volume change
                "confidence": 70,       # Minimum confidence percentage
                "fomo_factor": 1.0      # FOMO enhancement factor
            },
            "24h": {
                "price_change": 5.0,    # 5% price change for 24h predictions
                "volume_change": 12.0,  # 12% volume change
                "confidence": 65,       # Slightly lower confidence for longer timeframe
                "fomo_factor": 1.2      # Higher FOMO factor
            },
            "7d": {
                "price_change": 8.0,    # 8% price change for 7d predictions
                "volume_change": 15.0,  # 15% volume change
                "confidence": 60,       # Even lower confidence for weekly predictions
                "fomo_factor": 1.5      # Highest FOMO factor
            }
        }
       
        # Initialize scheduled timeframe posts
        self.next_scheduled_posts = {
            "1h": strip_timezone(datetime.now() + timedelta(minutes=random.randint(10, 30))),
            "24h": strip_timezone(datetime.now() + timedelta(hours=random.randint(1, 3))),
            "7d": strip_timezone(datetime.now() + timedelta(hours=random.randint(4, 8)))
        }
       
        # Initialize reply functionality components
        self.timeline_scraper = TimelineScraper(self.browser, self.config, self.config.db)
        self.reply_handler = ReplyHandler(self.browser, self.config, self.llm_provider, self.coingecko, self.config.db)
        self.content_analyzer = ContentAnalyzer(self.config, self.config.db)
       
        # Reply tracking and control
        self.last_reply_check = strip_timezone(datetime.now() - timedelta(minutes=30))  # Start checking soon
        self.reply_check_interval = 60  # Check for posts to reply to every 60 minutes
        self.max_replies_per_cycle = 10  # Maximum 10 replies per cycle
        self.reply_cooldown = 20  # Minutes between reply cycles
        self.last_reply_time = strip_timezone(datetime.now() - timedelta(minutes=self.reply_cooldown))  # Allow immediate first run
       
        logger.log_startup()

    @ensure_naive_datetimes
    def _check_for_reply_opportunities(self, market_data: Dict[str, Any]) -> bool:
        """
        Check for posts to reply to and generate replies with enhanced content analysis
        
        Args:
            market_data: Current market data dictionary
            
        Returns:
            True if any replies were posted
        """
        now = strip_timezone(datetime.now())

        # Check if it's time to look for posts to reply to
        time_since_last_check = safe_datetime_diff(now, self.last_reply_check) / 60
        if time_since_last_check < self.reply_check_interval:
            logger.logger.debug(f"Skipping reply check, {time_since_last_check:.1f} minutes since last check (interval: {self.reply_check_interval})")
            return False
    
        # Also check cooldown period
        time_since_last_reply = safe_datetime_diff(now, self.last_reply_time) / 60
        if time_since_last_reply < self.reply_cooldown:
            logger.logger.debug(f"In reply cooldown period, {time_since_last_reply:.1f} minutes since last reply (cooldown: {self.reply_cooldown})")
            return False
    
        logger.logger.info("Starting check for posts to reply to")
        self.last_reply_check = now
    
        try:
            # Scrape timeline for posts - get more to allow for better filtering
            posts = self.timeline_scraper.scrape_timeline(count=self.max_replies_per_cycle * 3)
            logger.logger.info(f"Timeline scraping completed - found {len(posts) if posts else 0} posts")
    
            if not posts:
                logger.logger.warning("No posts found during timeline scraping")
                return False

            # Log sample posts for debugging
            for i, post in enumerate(posts[:3]):
                logger.logger.info(f"Sample post {i}: {post.get('text', '')[:100]}...")

            # Enhanced filtering using ContentAnalyzer's comprehensive methods
            # First find market-related posts
            logger.logger.info(f"Finding market-related posts among {len(posts)} scraped posts")
            market_posts = self.content_analyzer.find_market_related_posts(posts)
            logger.logger.info(f"Found {len(market_posts)} market-related posts, checking which ones need replies")
        
            # Filter out posts we've already replied to
            unreplied_posts = self.content_analyzer.filter_already_replied_posts(market_posts)
            logger.logger.info(f"Found {len(unreplied_posts)} unreplied market-related posts")
        
            if not unreplied_posts:
                return False
            
            # Analyze content of each post for engagement metrics
            analyzed_posts = []
            for post in unreplied_posts:
                analysis = self.content_analyzer.analyze_post(post)
                # Add the analysis to the post
                post['content_analysis'] = analysis
                analyzed_posts.append(post)
            
            # Only reply to posts worth replying to based on ContentAnalyzer scoring
            reply_worthy_posts = [post for post in analyzed_posts if post['content_analysis'].get('reply_worthy', False)]
            logger.logger.info(f"Found {len(reply_worthy_posts)} reply-worthy posts")
        
            # Further prioritize high-value posts
            high_value_posts = [post for post in reply_worthy_posts if post['content_analysis'].get('high_value', False)]
        
            # Balance between high value and regular posts to ensure diversity
            posts_to_reply = high_value_posts[:int(self.max_replies_per_cycle * 0.7)]  # 70% high value
            remaining_slots = self.max_replies_per_cycle - len(posts_to_reply)
        
            # Add remaining medium-value posts sorted by reply score
            if remaining_slots > 0:
                medium_value_posts = [p for p in reply_worthy_posts if p not in high_value_posts]
                medium_value_posts.sort(key=lambda x: x['content_analysis'].get('reply_score', 0), reverse=True)
                posts_to_reply.extend(medium_value_posts[:remaining_slots])
        
            if not posts_to_reply:
                logger.logger.info("No posts worth replying to found")
                return False
            
            # Generate and post replies using enhanced reply metadata
            logger.logger.info(f"Starting to reply to {len(posts_to_reply)} prioritized posts")
            successful_replies = self.reply_handler.reply_to_posts(posts_to_reply, market_data, max_replies=self.max_replies_per_cycle)
        
            if successful_replies > 0:
                logger.logger.info(f"Successfully posted {successful_replies} replies")
                self.last_reply_time = now
                return True
            else:
                logger.logger.info("No replies were successfully posted")
                return False
            
        except Exception as e:
            logger.log_error("Check For Reply Opportunities", str(e))
            return False

    def _get_historical_volume_data(self, chain: str, minutes: int = None, timeframe: str = "1h") -> List[Dict[str, Any]]:
        """
        Get historical volume data for the specified window period
        Adjusted based on timeframe for appropriate historical context
        
        Args:
            chain: Token/chain symbol
            minutes: Time window in minutes (if None, determined by timeframe)
            timeframe: Timeframe for the data (1h, 24h, 7d)
            
        Returns:
            List of historical volume data points
        """
        try:
            # Adjust window size based on timeframe if not specifically provided
            if minutes is None:
                if timeframe == "1h":
                    minutes = self.config.VOLUME_WINDOW_MINUTES  # Default (typically 60)
                elif timeframe == "24h":
                    minutes = 24 * 60  # Last 24 hours
                elif timeframe == "7d":
                    minutes = 7 * 24 * 60  # Last 7 days
                else:
                    minutes = self.config.VOLUME_WINDOW_MINUTES
               
            window_start = strip_timezone(datetime.now() - timedelta(minutes=minutes))
            query = """
                SELECT timestamp, volume
                FROM market_data
                WHERE chain = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """
           
            conn = self.config.db.conn
            cursor = conn.cursor()
            cursor.execute(query, (chain, window_start))
            results = cursor.fetchall()
           
            volume_data = [
                {
                    'timestamp': strip_timezone(datetime.fromisoformat(row[0])),
                    'volume': float(row[1])
                }
                for row in results
            ]
           
            logger.logger.debug(
                f"Retrieved {len(volume_data)} volume data points for {chain} "
                f"over last {minutes} minutes (timeframe: {timeframe})"
            )
           
            return volume_data
           
        except Exception as e:
            logger.log_error(f"Historical Volume Data - {chain} ({timeframe})", str(e))
            return []
       
    def _is_duplicate_analysis(self, new_tweet: str, last_posts: List[str], timeframe: str = "1h") -> bool:
        """
        Enhanced duplicate detection with time-based thresholds and timeframe awareness.
        Applies different checks based on how recently similar content was posted:
        - Very recent posts (< 15 min): Check for exact matches
        - Recent posts (15-30 min): Check for high similarity
        - Older posts (> 30 min): Allow similar content
        
        Args:
            new_tweet: The new tweet text to check for duplication
            last_posts: List of recently posted tweets
            timeframe: Timeframe for the post (1h, 24h, 7d)
            
        Returns:
            Boolean indicating if the tweet is a duplicate
        """
        try:
            # Log that we're using enhanced duplicate detection
            logger.logger.info(f"Using enhanced time-based duplicate detection for {timeframe} timeframe")
           
            # Define time windows for different levels of duplicate checking
            # Adjust windows based on timeframe
            if timeframe == "1h":
                VERY_RECENT_WINDOW_MINUTES = 15
                RECENT_WINDOW_MINUTES = 30
                HIGH_SIMILARITY_THRESHOLD = 0.85  # 85% similar for recent posts
            elif timeframe == "24h":
                VERY_RECENT_WINDOW_MINUTES = 120  # 2 hours
                RECENT_WINDOW_MINUTES = 240       # 4 hours
                HIGH_SIMILARITY_THRESHOLD = 0.80  # Slightly lower threshold for daily predictions
            else:  # 7d
                VERY_RECENT_WINDOW_MINUTES = 720  # 12 hours
                RECENT_WINDOW_MINUTES = 1440      # 24 hours
                HIGH_SIMILARITY_THRESHOLD = 0.75  # Even lower threshold for weekly predictions
           
            # 1. Check for exact matches in very recent database entries
            conn = self.config.db.conn
            cursor = conn.cursor()
           
            # Very recent exact duplicates check
            cursor.execute("""
                SELECT content FROM posted_content 
                WHERE timestamp >= datetime('now', '-' || ? || ' minutes')
                AND timeframe = ?
            """, (VERY_RECENT_WINDOW_MINUTES, timeframe))
           
            very_recent_posts = [row[0] for row in cursor.fetchall()]
           
            # Check for exact matches in very recent posts
            for post in very_recent_posts:
                if post.strip() == new_tweet.strip():
                    logger.logger.info(f"Exact duplicate detected within last {VERY_RECENT_WINDOW_MINUTES} minutes for {timeframe}")
                    return True
           
            # 2. Check for high similarity in recent posts
            cursor.execute("""
                SELECT content FROM posted_content 
                WHERE timestamp >= datetime('now', '-' || ? || ' minutes')
                AND timestamp < datetime('now', '-' || ? || ' minutes')
                AND timeframe = ?
            """, (RECENT_WINDOW_MINUTES, VERY_RECENT_WINDOW_MINUTES, timeframe))
           
            recent_posts = [row[0] for row in cursor.fetchall()]
           
            # Calculate similarity for recent posts
            new_content = new_tweet.lower()
           
            for post in recent_posts:
                post_content = post.lower()
               
                # Calculate a simple similarity score based on word overlap
                new_words = set(new_content.split())
                post_words = set(post_content.split())
               
                if new_words and post_words:
                    overlap = len(new_words.intersection(post_words))
                    similarity = overlap / max(len(new_words), len(post_words))
                   
                    # Apply high similarity threshold for recent posts
                    if similarity > HIGH_SIMILARITY_THRESHOLD:
                        logger.logger.info(f"High similarity ({similarity:.2f}) detected within last {RECENT_WINDOW_MINUTES} minutes for {timeframe}")
                        return True
           
            # 3. Also check exact duplicates in last posts from Twitter
            # This prevents double-posting in case of database issues
            for post in last_posts:
                if post.strip() == new_tweet.strip():
                    logger.logger.info(f"Exact duplicate detected in recent Twitter posts for {timeframe}")
                    return True
           
            # If we get here, it's not a duplicate according to our criteria
            logger.logger.info(f"No duplicates detected with enhanced time-based criteria for {timeframe}")
            return False
           
        except Exception as e:
            logger.log_error(f"Duplicate Check - {timeframe}", str(e))
            # If the duplicate check fails, allow the post to be safe
            logger.logger.warning("Duplicate check failed, allowing post to proceed")
            return False

    def _start_prediction_thread(self) -> None:
        """
        Start background thread for asynchronous prediction generation
        """
        if self.prediction_thread is None or not self.prediction_thread.is_alive():
            self.prediction_thread_running = True
            self.prediction_thread = threading.Thread(target=self._process_prediction_queue)
            self.prediction_thread.daemon = True
            self.prediction_thread.start()
            logger.logger.info("Started prediction processing thread")
           
    def _process_prediction_queue(self) -> None:
        """
        Process predictions from the queue in the background
        """
        while self.prediction_thread_running:
            try:
                # Get a prediction task from the queue with timeout
                try:
                    task = self.prediction_queue.get(timeout=10)
                except queue.Empty:
                    # No tasks, just continue the loop
                    continue
                   
                # Process the prediction task
                token, timeframe, market_data = task
               
                logger.logger.debug(f"Processing queued prediction for {token} ({timeframe})")
               
                # Generate the prediction
                prediction = self.prediction_engine.generate_prediction(
                    token=token, 
                    market_data=market_data,
                    timeframe=timeframe
                )
               
                # Store in memory for quick access
                self.timeframe_predictions[timeframe][token] = prediction
               
                # Mark task as done
                self.prediction_queue.task_done()
               
                # Short sleep to prevent CPU overuse
                time.sleep(0.5)
               
            except Exception as e:
                logger.log_error("Prediction Thread Error", str(e))
                time.sleep(5)  # Sleep longer on error
               
        logger.logger.info("Prediction processing thread stopped")

    def _login_to_twitter(self) -> bool:
        """
        Log into Twitter with enhanced verification and detection of existing sessions
        
        Returns:
            Boolean indicating login success
        """
        try:
            logger.logger.info("Starting Twitter login")
            self.browser.driver.set_page_load_timeout(45)
        
            # First navigate to Twitter home page instead of login page directly
            self.browser.driver.get('https://twitter.com')
            time.sleep(5)
        
            # Check if we're already logged in
            already_logged_in = False
            login_indicators = [
                '[data-testid="SideNav_NewTweet_Button"]',
                '[data-testid="AppTabBar_Profile_Link"]',
                '[aria-label="Tweet"]',
                '.DraftEditor-root'  # Tweet composer element
            ]
        
            for indicator in login_indicators:
                try:
                    if self.browser.check_element_exists(indicator):
                        already_logged_in = True
                        logger.logger.info("Already logged into Twitter, using existing session")
                        return True
                except Exception:
                    continue
        
            if not already_logged_in:
                logger.logger.info("Not logged in, proceeding with login process")
                self.browser.driver.get('https://twitter.com/login')
                time.sleep(5)

                username_field = WebDriverWait(self.browser.driver, 20).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "input[autocomplete='username']"))
                )
                username_field.click()
                time.sleep(1)
                username_field.send_keys(self.config.TWITTER_USERNAME)
                time.sleep(2)

                next_button = WebDriverWait(self.browser.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//span[text()='Next']"))
                )
                next_button.click()
                time.sleep(3)

                password_field = WebDriverWait(self.browser.driver, 20).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "input[type='password']"))
                )
                password_field.click()
                time.sleep(1)
                password_field.send_keys(self.config.TWITTER_PASSWORD)
                time.sleep(2)

                login_button = WebDriverWait(self.browser.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//span[text()='Log in']"))
                )
                login_button.click()
                time.sleep(10)

            return self._verify_login()

        except Exception as e:
            logger.log_error("Twitter Login", str(e))
            return False

    def _verify_login(self) -> bool:
        """
        Verify Twitter login success
        
        Returns:
            Boolean indicating if login verification succeeded
        """
        try:
            verification_methods = [
                lambda: WebDriverWait(self.browser.driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="SideNav_NewTweet_Button"]'))
                ),
                lambda: WebDriverWait(self.browser.driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="AppTabBar_Profile_Link"]'))
                ),
                lambda: any(path in self.browser.driver.current_url 
                          for path in ['home', 'twitter.com/home'])
            ]
           
            for method in verification_methods:
                try:
                    if method():
                        return True
                except:
                    continue
           
            return False
           
        except Exception as e:
            logger.log_error("Login Verification", str(e))
            return False

    def _queue_predictions_for_all_timeframes(self, token: str, market_data: Dict[str, Any]) -> None:
        """
        Queue predictions for all timeframes for a specific token
        
        Args:
            token: Token symbol
            market_data: Market data dictionary
        """
        for timeframe in self.timeframes:
            # Skip if we already have a recent prediction
            if (token in self.timeframe_predictions.get(timeframe, {}) and 
               safe_datetime_diff(datetime.now(), self.timeframe_predictions[timeframe].get(token, {}).get('timestamp', 
                                                                                        datetime.now() - timedelta(hours=3))) 
               < 3600):  # Less than 1 hour old
                logger.logger.debug(f"Skipping {timeframe} prediction for {token} - already have recent prediction")
                continue
               
            # Add prediction task to queue
            self.prediction_queue.put((token, timeframe, market_data))
            logger.logger.debug(f"Queued {timeframe} prediction for {token}")

    def _post_analysis(self, tweet_text: str, timeframe: str = "1h") -> bool:
        """
        Post analysis to Twitter with robust button handling
        Tracks post by timeframe
        
        Args:
            tweet_text: Text to post
            timeframe: Timeframe for the analysis
            
        Returns:
            Boolean indicating if posting succeeded
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self.browser.driver.get('https://twitter.com/compose/tweet')
                time.sleep(3)
                
                text_area = WebDriverWait(self.browser.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="tweetTextarea_0"]'))
                )
                text_area.click()
                time.sleep(1)
                
                # Ensure tweet text only contains BMP characters
                safe_tweet_text = ''.join(char for char in tweet_text if ord(char) < 0x10000)
                
                # Simply send the tweet text directly - no handling of hashtags needed
                text_area.send_keys(safe_tweet_text)
                time.sleep(2)

                post_button = None
                button_locators = [
                    (By.CSS_SELECTOR, '[data-testid="tweetButton"]'),
                    (By.XPATH, "//div[@role='button'][contains(., 'Post')]"),
                    (By.XPATH, "//span[text()='Post']")
                ]

                for locator in button_locators:
                    try:
                        post_button = WebDriverWait(self.browser.driver, 5).until(
                            EC.element_to_be_clickable(locator)
                        )
                        if post_button:
                            break
                    except:
                        continue

                if post_button:
                    self.browser.driver.execute_script("arguments[0].scrollIntoView(true);", post_button)
                    time.sleep(1)
                    self.browser.driver.execute_script("arguments[0].click();", post_button)
                    time.sleep(5)
                    
                    # Update last post time for this timeframe
                    self.timeframe_last_post[timeframe] = strip_timezone(datetime.now())
                    
                    # Update next scheduled post time
                    hours_to_add = self.timeframe_posting_frequency.get(timeframe, 1)
                    # Add some randomness to prevent predictable patterns
                    jitter = random.uniform(0.8, 1.2)
                    self.next_scheduled_posts[timeframe] = strip_timezone(datetime.now() + timedelta(hours=hours_to_add * jitter))
                    
                    logger.logger.info(f"{timeframe} tweet posted successfully")
                    logger.logger.debug(f"Next {timeframe} post scheduled for {self.next_scheduled_posts[timeframe]}")
                    return True
                else:
                    logger.logger.error(f"Could not find post button for {timeframe} tweet")
                    retry_count += 1
                    time.sleep(2)
                    
            except Exception as e:
                logger.logger.error(f"{timeframe} tweet posting error, attempt {retry_count + 1}: {str(e)}")
                retry_count += 1
                wait_time = retry_count * 10
                logger.logger.warning(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
        
        logger.log_error(f"Tweet Creation - {timeframe}", "Maximum retries reached")
        return False
   
    @ensure_naive_datetimes
    def _get_last_posts(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get last N posts from timeline with timeframe detection
        
        Args:
            count: Number of posts to retrieve
            
        Returns:
            List of post information including detected timeframe
        """
        max_retries = 3
        retry_count = 0
    
        while retry_count < max_retries:
            try:
                self.browser.driver.get(f'https://twitter.com/{self.config.TWITTER_USERNAME}')
                time.sleep(3)
            
                # Use explicit waits to ensure elements are loaded
                WebDriverWait(self.browser.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="tweetText"]'))
                )
            
                posts = self.browser.driver.find_elements(By.CSS_SELECTOR, '[data-testid="tweetText"]')
            
                # Use an explicit wait for timestamps too
                WebDriverWait(self.browser.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'time'))
                )
            
                timestamps = self.browser.driver.find_elements(By.CSS_SELECTOR, 'time')
            
                # Get only the first count posts
                posts = posts[:count]
                timestamps = timestamps[:count]
            
                result = []
                for i in range(min(len(posts), len(timestamps))):
                    try:
                        post_text = posts[i].text
                        timestamp_str = timestamps[i].get_attribute('datetime') if timestamps[i].get_attribute('datetime') else None
                    
                        # Detect timeframe from post content
                        detected_timeframe = "1h"  # Default
                    
                        # Look for timeframe indicators in the post
                        if "7D PREDICTION" in post_text.upper() or "7-DAY" in post_text.upper() or "WEEKLY" in post_text.upper():
                            detected_timeframe = "7d"
                        elif "24H PREDICTION" in post_text.upper() or "24-HOUR" in post_text.upper() or "DAILY" in post_text.upper():
                            detected_timeframe = "24h"
                        elif "1H PREDICTION" in post_text.upper() or "1-HOUR" in post_text.upper() or "HOURLY" in post_text.upper():
                            detected_timeframe = "1h"
                    
                        post_info = {
                            'text': post_text,
                            'timestamp': strip_timezone(datetime.fromisoformat(timestamp_str)) if timestamp_str else None,
                            'timeframe': detected_timeframe
                        }
                    
                        result.append(post_info)
                    except Exception as element_error:
                        # Skip this element if it's stale or otherwise problematic
                        logger.logger.debug(f"Element error while extracting post {i}: {str(element_error)}")
                        continue
            
                return result
            
            except Exception as e:
                retry_count += 1
                logger.logger.warning(f"Error getting last posts (attempt {retry_count}/{max_retries}): {str(e)}")
                time.sleep(2)  # Add a small delay before retry
            
        # If all retries failed, log the error and return an empty list
        logger.log_error("Get Last Posts", f"Maximum retries ({max_retries}) reached")
        return []

    def _get_last_posts_by_timeframe(self, timeframe: str = "1h", count: int = 5) -> List[str]:
        """
        Get last N posts for a specific timeframe
        
        Args:
            timeframe: Timeframe to filter for
            count: Number of posts to retrieve
            
        Returns:
            List of post text content
        """
        all_posts = self._get_last_posts(count=20)  # Get more posts to filter from
        
        # Filter posts by the requested timeframe
        filtered_posts = [post['text'] for post in all_posts if post['timeframe'] == timeframe]
        
        # Return the requested number of posts
        return filtered_posts[:count]

    @ensure_naive_datetimes
    def _schedule_timeframe_post(self, timeframe: str, delay_hours: float = None) -> None:
        """
        Schedule the next post for a specific timeframe
        
        Args:
            timeframe: Timeframe to schedule for
            delay_hours: Optional override for delay hours (otherwise uses default frequency)
        """
        if delay_hours is None:
            # Use default frequency with some randomness
            base_hours = self.timeframe_posting_frequency.get(timeframe, 1)
            delay_hours = base_hours * random.uniform(0.9, 1.1)
        
        self.next_scheduled_posts[timeframe] = strip_timezone(datetime.now() + timedelta(hours=delay_hours))
        logger.logger.debug(f"Scheduled next {timeframe} post for {self.next_scheduled_posts[timeframe]}")
   
    @ensure_naive_datetimes
    def _should_post_timeframe_now(self, timeframe: str) -> bool:
        """
        Check if it's time to post for a specific timeframe
        
        Args:
            timeframe: Timeframe to check
            
        Returns:
            Boolean indicating if it's time to post
        """
        try:
            # Debug
            logger.logger.debug(f"Checking if should post for {timeframe}")
            logger.logger.debug(f"  Last post: {self.timeframe_last_post.get(timeframe)} ({type(self.timeframe_last_post.get(timeframe))})")
            logger.logger.debug(f"  Next scheduled: {self.next_scheduled_posts.get(timeframe)} ({type(self.next_scheduled_posts.get(timeframe))})")
        
            # Check if enough time has passed since last post
            min_interval = timedelta(hours=self.timeframe_posting_frequency.get(timeframe, 1) * 0.8)
            last_post_time = self._ensure_datetime(self.timeframe_last_post.get(timeframe, datetime.min))
            logger.logger.debug(f"  Last post time (after ensure): {last_post_time} ({type(last_post_time)})")
        
            time_since_last = safe_datetime_diff(datetime.now(), last_post_time) / 3600  # Hours
            if time_since_last < min_interval.total_seconds() / 3600:
                return False
            
            # Check if scheduled time has been reached
            next_scheduled = self._ensure_datetime(self.next_scheduled_posts.get(timeframe, datetime.now()))
            logger.logger.debug(f"  Next scheduled (after ensure): {next_scheduled} ({type(next_scheduled)})")
        
            return datetime.now() >= next_scheduled
        except Exception as e:
            logger.logger.error(f"Error in _should_post_timeframe_now for {timeframe}: {str(e)}")
            # Provide a safe default
            return False
   
    @ensure_naive_datetimes
    def _post_prediction_for_timeframe(self, token: str, market_data: Dict[str, Any], timeframe: str) -> bool:
        """
        Post a prediction for a specific timeframe
        
        Args:
            token: Token symbol
            market_data: Market data dictionary
            timeframe: Timeframe for the prediction
            
        Returns:
            Boolean indicating if posting succeeded
        """
        try:
            # Check if we have a prediction
            prediction = self.timeframe_predictions.get(timeframe, {}).get(token)
        
            # If no prediction exists, generate one
            if not prediction:
                prediction = self.prediction_engine.generate_prediction(
                    token=token,
                    market_data=market_data,
                    timeframe=timeframe
                )
            
                # Store for future use
                if timeframe not in self.timeframe_predictions:
                    self.timeframe_predictions[timeframe] = {}
                self.timeframe_predictions[timeframe][token] = prediction
        
            # Format the prediction for posting
            tweet_text = self._format_prediction_tweet(token, prediction, market_data, timeframe)
        
            # Check for duplicates - make sure we're handling datetime properly
            last_posts = self._get_last_posts_by_timeframe(timeframe=timeframe)
        
            # Ensure datetime compatibility in duplicate check
            if self._is_duplicate_analysis(tweet_text, last_posts, timeframe):
                logger.logger.warning(f"Skipping duplicate {timeframe} prediction for {token}")
                return False
            
            # Post the prediction
            if self._post_analysis(tweet_text, timeframe):
                # Store in database
                sentiment = prediction.get("sentiment", "NEUTRAL")
                price_data = {token: {'price': market_data[token]['current_price'], 
                                    'volume': market_data[token]['volume']}}
            
                # Create storage data
                storage_data = {
                    'content': tweet_text,
                    'sentiment': {token: sentiment},
                    'trigger_type': f"scheduled_{timeframe}_post",
                    'price_data': price_data,
                    'meme_phrases': {token: ""},  # No meme phrases for predictions
                    'is_prediction': True,
                    'prediction_data': prediction,
                    'timeframe': timeframe
                }
            
                # Store in database
                self.config.db.store_posted_content(**storage_data)
            
                # Update last post time for this timeframe with current datetime
                # This is important - make sure we're storing a datetime object
                self.timeframe_last_post[timeframe] = strip_timezone(datetime.now())
            
                logger.logger.info(f"Successfully posted {timeframe} prediction for {token}")
                return True
            else:
                logger.logger.error(f"Failed to post {timeframe} prediction for {token}")
                return False
            
        except Exception as e:
            logger.log_error(f"Post Prediction For Timeframe - {token} ({timeframe})", str(e))
            return False
   
    @ensure_naive_datetimes
    def _post_timeframe_rotation(self, market_data: Dict[str, Any]) -> bool:
        """
        Post predictions in a rotation across timeframes with enhanced token selection
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Boolean indicating if a post was made
        """
        # Debug timeframe scheduling data
        logger.logger.debug("TIMEFRAME ROTATION DEBUG:")
        for tf in self.timeframes:
            try:
                now = strip_timezone(datetime.now())
                last_post_time = strip_timezone(self._ensure_datetime(self.timeframe_last_post.get(tf)))
                next_scheduled_time = strip_timezone(self._ensure_datetime(self.next_scheduled_posts.get(tf)))
            
                time_since_last = safe_datetime_diff(now, last_post_time) / 3600
                time_until_next = safe_datetime_diff(next_scheduled_time, now) / 3600
                logger.logger.debug(f"{tf}: {time_since_last:.1f}h since last post, {time_until_next:.1f}h until next")
            except Exception as e:
                logger.logger.error(f"Error calculating timeframe timing for {tf}: {str(e)}")
        
        # First check if any timeframe is due for posting
        due_timeframes = [tf for tf in self.timeframes if self._should_post_timeframe_now(tf)]

        if not due_timeframes:
            logger.logger.debug("No timeframes due for posting")
            return False
    
        try:
            # Pick the most overdue timeframe
            now = strip_timezone(datetime.now())
        
            chosen_timeframe = None
            max_overdue_time = timedelta(0)
        
            for tf in due_timeframes:
                next_scheduled = strip_timezone(self._ensure_datetime(self.next_scheduled_posts.get(tf, datetime.min)))
                overdue_time = safe_datetime_diff(now, next_scheduled)
            
                if overdue_time > max_overdue_time.total_seconds():
                    max_overdue_time = timedelta(seconds=overdue_time)
                    chosen_timeframe = tf
                
            if not chosen_timeframe:
                logger.logger.warning("Could not find most overdue timeframe, using first available")
                chosen_timeframe = due_timeframes[0]
            
        except ValueError as ve:
            if "arg is an empty sequence" in str(ve):
                logger.logger.warning("No timeframes available for rotation, rescheduling all timeframes")
                # Reschedule all timeframes with random delays
                now = strip_timezone(datetime.now())
                for tf in self.timeframes:
                    delay_hours = self.timeframe_posting_frequency.get(tf, 1) * random.uniform(0.1, 0.3)
                    self.next_scheduled_posts[tf] = now + timedelta(hours=delay_hours)
                return False
            else:
                raise  # Re-raise if it's a different ValueError
        
        logger.logger.info(f"Selected {chosen_timeframe} for timeframe rotation posting")

        # Enhanced token selection using content analysis and reply data
        token_to_post = self._select_best_token_for_timeframe(market_data, chosen_timeframe)
    
        if not token_to_post:
            logger.logger.warning(f"No suitable token found for {chosen_timeframe} timeframe")
            # Reschedule this timeframe for later
            now = strip_timezone(datetime.now())
            self._schedule_timeframe_post(chosen_timeframe, delay_hours=1)
            return False
    
        # Before posting, check if there's active community discussion about this token
        # This helps align our posts with current community interests
        try:
            # Get recent timeline posts to analyze community trends
            recent_posts = self.timeline_scraper.scrape_timeline(count=25)
            if recent_posts:
                # Filter for posts related to our selected token
                token_related_posts = [p for p in recent_posts if token_to_post.upper() in p.get('text', '').upper()]
        
                # If we found significant community discussion, give this token higher priority
                if len(token_related_posts) >= 3:
                    logger.logger.info(f"Found active community discussion about {token_to_post} ({len(token_related_posts)} recent posts)")
                    # Analyze sentiment to make our post more contextually relevant
                    sentiment_stats = {
                        'positive': 0,
                        'negative': 0,
                        'neutral': 0
                    }
            
                    # Simple sentiment analysis of community posts
                    for post in token_related_posts:
                        analysis = self.content_analyzer.analyze_post(post)
                        sentiment = analysis.get('features', {}).get('sentiment', {}).get('label', 'neutral')
                        if sentiment in ['bullish', 'enthusiastic', 'positive']:
                            sentiment_stats['positive'] += 1
                        elif sentiment in ['bearish', 'negative', 'skeptical']:
                            sentiment_stats['negative'] += 1
                        else:
                            sentiment_stats['neutral'] += 1
            
                    # Log community sentiment
                    dominant_sentiment = max(sentiment_stats.items(), key=lambda x: x[1])[0]
                    logger.logger.info(f"Community sentiment for {token_to_post}: {dominant_sentiment} ({sentiment_stats})")
                else:
                    logger.logger.debug(f"Limited community discussion about {token_to_post} ({len(token_related_posts)} posts)")
        except Exception as e:
            logger.logger.warning(f"Error analyzing community trends: {str(e)}")
    
        # Post the prediction
        success = self._post_prediction_for_timeframe(token_to_post, market_data, chosen_timeframe)
    
        # If post failed, reschedule for later
        if not success:
            now = strip_timezone(datetime.now())
            self._schedule_timeframe_post(chosen_timeframe, delay_hours=1)
    
        return success

    def _analyze_tech_topics(self, market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze tech topics for educational content generation
    
        Args:
            market_data: Optional market data for context
        
        Returns:
            Dictionary with tech topic analysis
        """
        try:
            # Get configured tech topics
            tech_topics = self.config.get_tech_topics()
        
            if not tech_topics:
                logger.logger.warning("No tech topics configured or enabled")
                return {'enabled': False}
            
            # Get recent tech posts from database
            tech_posts = {}
            last_tech_post = strip_timezone(datetime.now() - timedelta(days=1))  # Default fallback
        
            if self.config.db:
                try:
                    # Query last 24 hours of content
                    recent_posts = self.config.db.get_recent_posts(hours=24)
                
                    # Filter to tech-related posts
                    for post in recent_posts:
                        if 'tech_category' in post:
                            category = post['tech_category']
                            if category not in tech_posts:
                                tech_posts[category] = []
                            tech_posts[category].append(post)
                        
                            # Update last tech post time
                            post_time = strip_timezone(datetime.fromisoformat(post['timestamp']))
                            if post_time > last_tech_post:
                                last_tech_post = post_time
                except Exception as db_err:
                    logger.logger.warning(f"Error retrieving tech posts: {str(db_err)}")
                
            # Analyze topics for candidacy
            candidate_topics = []
        
            for topic in tech_topics:
                category = topic['category']
                posts_today = len(tech_posts.get(category, []))
            
                # Calculate last post for this category
                category_last_post = last_tech_post
                if category in tech_posts and tech_posts[category]:
                    category_timestamps = [
                        strip_timezone(datetime.fromisoformat(p['timestamp'])) 
                        for p in tech_posts[category]
                    ]
                    if category_timestamps:
                        category_last_post = max(category_timestamps)
            
                # Check if allowed to post about this category
                allowed = self.config.is_tech_post_allowed(category, category_last_post)
            
                if allowed:
                    # Prepare topic metadata
                    topic_metadata = {
                        'category': category,
                        'priority': topic['priority'],
                        'keywords': topic['keywords'][:5],  # Just first 5 for logging
                        'posts_today': posts_today,
                        'hours_since_last_post': safe_datetime_diff(datetime.now(), category_last_post) / 3600,
                        'selected_token': self._select_token_for_tech_topic(category, market_data)
                    }
                
                    # Add to candidates
                    candidate_topics.append(topic_metadata)
        
            # Order by priority and recency
            if candidate_topics:
                candidate_topics.sort(key=lambda x: (x['priority'], x['hours_since_last_post']), reverse=True)
                logger.logger.info(f"Found {len(candidate_topics)} tech topics eligible for posting")
            
                # Return analysis results
                return {
                    'enabled': True,
                    'candidate_topics': candidate_topics,
                    'tech_posts_today': sum(len(posts) for posts in tech_posts.values()),
                    'max_daily_posts': self.config.TECH_CONTENT_CONFIG.get('max_daily_tech_posts', 6),
                    'last_tech_post': last_tech_post
                }
            else:
                logger.logger.info("No tech topics are currently eligible for posting")
                return {
                    'enabled': True,
                    'candidate_topics': [],
                    'tech_posts_today': sum(len(posts) for posts in tech_posts.values()),
                    'max_daily_posts': self.config.TECH_CONTENT_CONFIG.get('max_daily_tech_posts', 6),
                    'last_tech_post': last_tech_post
                }
        
        except Exception as e:
            logger.log_error("Tech Topic Analysis", str(e))
            return {'enabled': False, 'error': str(e)}

    def _select_token_for_tech_topic(self, tech_category: str, market_data: Dict[str, Any] = None) -> str:
        """
        Select an appropriate token to pair with a tech topic
    
        Args:
            tech_category: Tech category for pairing
            market_data: Market data for context
        
        Returns:
            Selected token symbol
        """
        try:
            if not market_data:
                # Default to a popular token if no market data
                return random.choice(['BTC', 'ETH', 'SOL'])
            
            # Define affinity between tech categories and tokens
            tech_token_affinity = {
                'ai': ['ETH', 'SOL', 'DOT'],          # Smart contract platforms
                'quantum': ['BTC', 'XRP', 'AVAX'],    # Security-focused or scaling
                'blockchain_tech': ['ETH', 'SOL', 'BNB', 'NEAR'],  # Advanced platforms
                'advanced_computing': ['SOL', 'AVAX', 'DOT']  # High performance chains
            }
        
            # Get affinity tokens for this category
            affinity_tokens = tech_token_affinity.get(tech_category, self.reference_tokens)
        
            # Filter to tokens with available market data
            available_tokens = [t for t in affinity_tokens if t in market_data]
        
            if not available_tokens:
                # Fall back to reference tokens if no affinity tokens available
                available_tokens = [t for t in self.reference_tokens if t in market_data]
            
            if not available_tokens:
                # Last resort fallback
                return random.choice(['BTC', 'ETH', 'SOL'])
            
            # Select token with interesting market movement if possible
            interesting_tokens = []
            for token in available_tokens:
                price_change = abs(market_data[token].get('price_change_percentage_24h', 0))
                if price_change > 5.0:  # >5% change is interesting
                    interesting_tokens.append(token)
                
            # Use interesting tokens if available, otherwise use all available tokens
            selection_pool = interesting_tokens if interesting_tokens else available_tokens
        
            # Select a token, weighting by market cap if possible
            if len(selection_pool) > 1:
                # Extract market caps
                market_caps = {t: market_data[t].get('market_cap', 1) for t in selection_pool}
                # Create weighted probability
                total_cap = sum(market_caps.values())
                weights = [market_caps[t]/total_cap for t in selection_pool]
                # Select with weights
                return random.choices(selection_pool, weights=weights, k=1)[0]
            else:
                # Just one token available
                return selection_pool[0]
            
        except Exception as e:
            logger.log_error("Token Selection for Tech Topic", str(e))
            # Safe fallback
            return random.choice(['BTC', 'ETH', 'SOL'])

    def _generate_tech_content(self, tech_category: str, token: str, market_data: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate educational tech content for posting
    
        Args:
            tech_category: Tech category to focus on
            token: Token to relate to tech content
            market_data: Market data for context
        
        Returns:
            Tuple of (content_text, metadata)
        """
        try:
            logger.logger.info(f"Generating tech content for {tech_category} related to {token}")
        
            # Get token data if available
            token_data = {}
            if market_data and token in market_data:
                token_data = market_data[token]
            
            # Determine content type - integration or educational
            integration_prob = 0.7  # 70% chance of integration content
            is_integration = random.random() < integration_prob
        
            # Select appropriate audience level
            audience_levels = ['beginner', 'intermediate', 'advanced']
            audience_weights = [0.3, 0.5, 0.2]  # Bias toward intermediate
            audience_level = random.choices(audience_levels, weights=audience_weights, k=1)[0]
        
            # Select appropriate template
            template_type = 'integration_template' if is_integration else 'educational_template'
            template = self.config.get_tech_prompt_template(template_type, audience_level)
        
            # Prepare prompt variables
            prompt_vars = {
                'tech_topic': tech_category.replace('_', ' ').title(),
                'token': token,
                'audience_level': audience_level,
                'min_length': self.config.TWEET_CONSTRAINTS['MIN_LENGTH'],
                'max_length': self.config.TWEET_CONSTRAINTS['MAX_LENGTH']
            }
        
            if is_integration:
                # Integration template - focus on connections
                # Get token sentiment for mood
                mood_words = ['enthusiastic', 'analytical', 'curious', 'balanced', 'thoughtful']
                prompt_vars['mood'] = random.choice(mood_words)
            
                # Add token price data if available
                if token_data:
                    prompt_vars['token_price'] = token_data.get('current_price', 0)
                    prompt_vars['price_change'] = token_data.get('price_change_percentage_24h', 0)
                
                # Get tech status summary
                prompt_vars['tech_status'] = self._get_tech_status_summary(tech_category)
                prompt_vars['integration_level'] = random.randint(3, 8)
            
                # Use tech analysis prompt template if we have market data
                if token_data:
                    prompt = self.config.client_TECH_ANALYSIS_PROMPT.format(
                        tech_topic=prompt_vars['tech_topic'],
                        token=token,
                        price=token_data.get('current_price', 0),
                        change=token_data.get('price_change_percentage_24h', 0),
                        tech_status_summary=prompt_vars['tech_status'],
                        integration_level=prompt_vars['integration_level'],
                        audience_level=audience_level
                    )
                else:
                    # Fall back to simpler template without market data
                    prompt = template.format(**prompt_vars)
                
            else:
                # Educational template - focus on informative content
                # Generate key points for educational content
                key_points = self._generate_tech_key_points(tech_category)
                prompt_vars['key_point_1'] = key_points[0]
                prompt_vars['key_point_2'] = key_points[1]
                prompt_vars['key_point_3'] = key_points[2]
                prompt_vars['learning_objective'] = self._generate_learning_objective(tech_category)
            
                # Format prompt with variables
                prompt = template.format(**prompt_vars)
        
            # Generate content with LLM
            logger.logger.debug(f"Generating {tech_category} content with {template_type}")
            content = self.llm_provider.generate_text(prompt, max_tokens=1000)
        
            if not content:
                raise ValueError("Failed to generate tech content")
            
            # Ensure content meets length requirements
            content = self._format_tech_content(content)
        
            # Prepare metadata for storage
            metadata = {
                'tech_category': tech_category,
                'token': token,
                'is_integration': is_integration,
                'audience_level': audience_level,
                'template_type': template_type,
                'token_data': token_data,
                'timestamp': strip_timezone(datetime.now())
            }
        
            return content, metadata
        
        except Exception as e:
            logger.log_error("Tech Content Generation", str(e))
            # Return fallback content
            fallback_content = f"Did you know that advances in {tech_category.replace('_', ' ')} technology could significantly impact the future of {token} and the broader crypto ecosystem? The intersection of these fields is creating fascinating new possibilities."
            return fallback_content, {'tech_category': tech_category, 'token': token, 'error': str(e)}

    def _post_tech_content(self, content: str, metadata: Dict[str, Any]) -> bool:
        """
        Post tech content to Twitter with proper formatting
    
        Args:
            content: Content to post
            metadata: Content metadata for database storage
        
        Returns:
            Boolean indicating if posting succeeded
        """
        try:
            # Check if content is already properly formatted
            if len(content) > self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH']:
                content = self._format_tech_content(content)
            
            # Format as a tweet
            tweet_text = content
        
            # Add a subtle educational hashtag if there's room
            if len(tweet_text) < self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH'] - 20:
                tech_category = metadata.get('tech_category', 'technology')
                token = metadata.get('token', '')
            
                # Determine if we should add hashtags
                if random.random() < 0.7:  # 70% chance to add hashtags
                    # Potential hashtags
                    tech_tags = {
                        'ai': ['#AI', '#ArtificialIntelligence', '#MachineLearning'],
                        'quantum': ['#QuantumComputing', '#Quantum', '#QuantumTech'],
                        'blockchain_tech': ['#Blockchain', '#Web3', '#DLT'],
                        'advanced_computing': ['#Computing', '#TechInnovation', '#FutureTech']
                    }
                
                    # Get tech hashtags
                    tech_hashtags = tech_tags.get(tech_category, ['#Technology', '#Innovation'])
                
                    # Add tech hashtag and token
                    hashtag = random.choice(tech_hashtags)
                    if len(tweet_text) + len(hashtag) + len(token) + 2 <= self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH']:
                        tweet_text = f"{tweet_text} {hashtag}"
                    
                        # Maybe add token hashtag too
                        if len(tweet_text) + len(token) + 2 <= self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH']:
                            tweet_text = f"{tweet_text} #{token}"
        
            # Post to Twitter
            logger.logger.info(f"Posting tech content about {metadata.get('tech_category', 'tech')} and {metadata.get('token', 'crypto')}")
            if self._post_analysis(tweet_text):
                logger.logger.info("Successfully posted tech content")
            
                # Record in database
                if self.config.db:
                    try:
                        # Extract token price data if available
                        token = metadata.get('token', '')
                        token_data = metadata.get('token_data', {})
                        price_data = {
                            token: {
                                'price': token_data.get('current_price', 0),
                                'volume': token_data.get('volume', 0)
                            }
                        }
                    
                        # Store as content with tech category
                        self.config.db.store_posted_content(
                            content=tweet_text,
                            sentiment={},  # No sentiment for educational content
                            trigger_type=f"tech_{metadata.get('tech_category', 'general')}",
                            price_data=price_data,
                            meme_phrases={},  # No meme phrases for educational content
                            tech_category=metadata.get('tech_category', 'technology'),
                            tech_metadata=metadata,
                            is_educational=True
                        )
                    except Exception as db_err:
                        logger.logger.warning(f"Failed to store tech content: {str(db_err)}")
            
                return True
            else:
                logger.logger.warning("Failed to post tech content")
                return False
            
        except Exception as e:
            logger.log_error("Tech Content Posting", str(e))
            return False

    def _format_tech_content(self, content: str) -> str:
        """
        Format tech content to meet tweet constraints
    
        Args:
            content: Raw content to format
        
        Returns:
            Formatted content
        """
        # Ensure length is within constraints
        if len(content) > self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH']:
            # Find a good sentence break to truncate
            last_period = content[:self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH'] - 3].rfind('.')
            last_question = content[:self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH'] - 3].rfind('?')
            last_exclamation = content[:self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH'] - 3].rfind('!')
        
            # Find best break point
            break_point = max(last_period, last_question, last_exclamation)
        
            if break_point > self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH'] * 0.7:
                # Good sentence break found
                content = content[:break_point + 1]
            else:
                # Find word boundary
                last_space = content[:self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH'] - 3].rfind(' ')
                if last_space > 0:
                    content = content[:last_space] + "..."
                else:
                    # Hard truncate with ellipsis
                    content = content[:self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH'] - 3] + "..."
    
        # Ensure minimum length is met
        if len(content) < self.config.TWEET_CONSTRAINTS['MIN_LENGTH']:
            logger.logger.warning(f"Tech content too short ({len(content)} chars). Minimum: {self.config.TWEET_CONSTRAINTS['MIN_LENGTH']}")
            # We won't try to expand too-short content
    
        return content

    def _get_tech_status_summary(self, tech_category: str) -> str:
        """
        Get a current status summary for a tech category
    
        Args:
            tech_category: Tech category
        
        Returns:
            Status summary string
        """
        # Status summaries by category
        summaries = {
            'ai': [
                "Rapid advancement in multimodal capabilities",
                "Increasing deployment in enterprise settings",
                "Rising concerns about governance and safety",
                "Growing focus on specialized models",
                "Shift toward open models and distributed research",
                "Mainstream adoption accelerating"
            ],
            'quantum': [
                "Steady progress in error correction",
                "Growing number of qubits in leading systems",
                "Early commercial applications emerging",
                "Increasing focus on quantum-resistant cryptography",
                "Major investment from government and private sectors",
                "Hardware diversity expanding beyond superconducting qubits"
            ],
            'blockchain_tech': [
                "Layer 2 solutions gaining momentum",
                "ZK-rollup technology maturing rapidly",
                "Cross-chain interoperability improving",
                "RWA tokenization expanding use cases",
                "Institutional adoption of infrastructure growing",
                "Privacy-preserving technologies advancing"
            ],
            'advanced_computing': [
                "Specialized AI hardware proliferating",
                "Edge computing deployments accelerating",
                "Neuromorphic computing showing early promise",
                "Post-Moore's Law approaches diversifying",
                "High-performance computing becoming more accessible",
                "Increasing focus on energy efficiency"
            ]
        }
    
        # Get summaries for this category
        category_summaries = summaries.get(tech_category, [
            "Steady technological progress",
            "Growing market adoption",
            "Increasing integration with existing systems",
            "Emerging commercial applications",
            "Active research and development"
        ])
    
        # Return random summary
        return random.choice(category_summaries)

    def _generate_tech_key_points(self, tech_category: str) -> List[str]:
        """
        Generate key educational points for a tech category
    
        Args:
            tech_category: Tech category
        
        Returns:
            List of key points for educational content
        """
        # Define key educational points by category
        key_points = {
            'ai': [
                "How large language models process and generate human language",
                "The difference between narrow AI and artificial general intelligence",
                "How multimodal AI combines text, image, audio, and video processing",
                "The concept of prompt engineering and its importance",
                "The role of fine-tuning in customizing AI models",
                "How AI models are trained on massive datasets",
                "The emergence of specialized AI for different industries",
                "The importance of ethical considerations in AI development",
                "How AI models handle context and memory limitations",
                "The computational resources required for modern AI systems"
            ],
            'quantum': [
                "How quantum bits (qubits) differ from classical bits",
                "The concept of quantum superposition and entanglement",
                "Why quantum computing excels at certain types of problems",
                "The challenge of quantum error correction",
                "Different physical implementations of quantum computers",
                "How quantum algorithms provide computational advantages",
                "The potential impact on cryptography and security",
                "The timeline for quantum advantage in practical applications",
                "How quantum computing complements rather than replaces classical computing",
                "The difference between quantum annealing and gate-based quantum computing"
            ],
            'blockchain_tech': [
                "How zero-knowledge proofs enable privacy while maintaining verification",
                "The concept of sharding and its role in blockchain scaling",
                "The difference between optimistic and ZK rollups",
                "How Layer 2 solutions address blockchain scalability challenges",
                "The evolution of consensus mechanisms beyond proof of work",
                "How cross-chain bridges enable interoperability between blockchains",
                "The concept of state channels for off-chain transactions",
                "How smart contracts enable programmable transactions",
                "The role of oracles in connecting blockchains to external data",
                "Different approaches to blockchain governance"
            ],
            'advanced_computing': [
                "How neuromorphic computing mimics brain functions",
                "The concept of edge computing and its advantages",
                "The evolution beyond traditional Moore's Law scaling",
                "How specialized hardware accelerates specific workloads",
                "The rise of heterogeneous computing architectures",
                "How in-memory computing reduces data movement bottlenecks",
                "The potential of optical computing for specific applications",
                "How quantum-inspired algorithms work on classical hardware",
                "The importance of energy efficiency in modern computing",
                "How cloud computing is evolving with specialized hardware"
            ]
        }
    
        # Get points for this category
        category_points = key_points.get(tech_category, [
            "The fundamental principles behind this technology",
            "Current applications and use cases",
            "Future potential developments and challenges",
            "How this technology relates to blockchain and cryptocurrency",
            "The importance of this technology for digital innovation"
        ])
    
        # Select 3 random points without replacement
        selected_points = random.sample(category_points, min(3, len(category_points)))
    
        # Ensure we have 3 points
        while len(selected_points) < 3:
            selected_points.append("How this technology impacts the future of digital assets")
        
        return selected_points

    def _generate_learning_objective(self, tech_category: str) -> str:
        """
        Generate a learning objective for educational tech content
    
        Args:
            tech_category: Tech category
        
        Returns:
            Learning objective string
        """
        # Define learning objectives by category
        objectives = {
            'ai': [
                "how AI technologies are transforming the crypto landscape",
                "the core principles behind modern AI systems",
                "how AI and blockchain technologies can complement each other",
                "the key limitations and challenges of current AI approaches",
                "how AI is being used to enhance trading, security, and analytics in crypto"
            ],
            'quantum': [
                "how quantum computing affects blockchain security",
                "the fundamentals of quantum computing in accessible terms",
                "the timeline and implications of quantum advances for cryptography",
                "how the crypto industry is preparing for quantum computing",
                "the difference between quantum threats and opportunities for blockchain"
            ],
            'blockchain_tech': [
                "how advanced blockchain technologies are addressing scalability",
                "the technical foundations of modern blockchain systems",
                "the trade-offs between different blockchain scaling approaches",
                "how blockchain privacy technologies actually work",
                "the evolution of blockchain architecture beyond first-generation systems"
            ],
            'advanced_computing': [
                "how specialized computing hardware is changing crypto mining",
                "the next generation of computing technologies on the horizon",
                "how computing advances are enabling new blockchain capabilities",
                "the relationship between energy efficiency and blockchain sustainability",
                "how distributed computing and blockchain share foundational principles"
            ]
        }
    
        # Get objectives for this category
        category_objectives = objectives.get(tech_category, [
            "the fundamentals of this technology in accessible terms",
            "how this technology relates to blockchain and cryptocurrency",
            "the potential future impact of this technological development",
            "the current state and challenges of this technology",
            "how this technology might transform digital finance"
        ])
    
        # Return random objective
        return random.choice(category_objectives)

    @ensure_naive_datetimes
    def _should_post_tech_content(self, market_data: Dict[str, Any] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if tech content should be posted, and select a topic
    
        Args:
            market_data: Optional market data for context
        
        Returns:
            Tuple of (should_post, topic_data)
        """
        try:
            # Check if tech content is enabled
            if not self.config.TECH_CONTENT_CONFIG.get('enabled', False):
                return False, {}
            
            # Analyze tech topics
            tech_analysis = self._analyze_tech_topics(market_data)
        
            if not tech_analysis.get('enabled', False):
                return False, {}
            
            # Check if we've hit daily maximum
            if tech_analysis.get('tech_posts_today', 0) >= tech_analysis.get('max_daily_posts', 6):
                logger.logger.info(f"Maximum daily tech posts reached ({tech_analysis.get('tech_posts_today')}/{tech_analysis.get('max_daily_posts')})")
                return False, {}
            
            # Check if we have candidate topics
            candidates = tech_analysis.get('candidate_topics', [])
            if not candidates:
                return False, {}
            
            # Select top candidate
            selected_topic = candidates[0]
        
            # Check if enough time has passed since last tech post
            last_tech_post = tech_analysis.get('last_tech_post', strip_timezone(datetime.now() - timedelta(days=1)))
            hours_since_last = safe_datetime_diff(datetime.now(), last_tech_post) / 3600
            post_frequency = self.config.TECH_CONTENT_CONFIG.get('post_frequency', 4)
        
            if hours_since_last < post_frequency:
                logger.logger.info(f"Not enough time since last tech post ({hours_since_last:.1f}h < {post_frequency}h)")
                return False, selected_topic
            
            # At this point, we should post tech content
            logger.logger.info(f"Will post tech content about {selected_topic['category']} related to {selected_topic['selected_token']}")
            return True, selected_topic
        
        except Exception as e:
            logger.log_error("Tech Content Decision", str(e))
            return False, {}

    def _post_tech_educational_content(self, market_data: Dict[str, Any]) -> bool:
        """
        Generate and post tech educational content
    
        Args:
            market_data: Market data for context
        
        Returns:
            Boolean indicating if content was successfully posted
        """
        try:
            # Check if we should post tech content
            should_post, topic_data = self._should_post_tech_content(market_data)
        
            if not should_post:
                return False
            
            # Generate tech content
            tech_category = topic_data.get('category', 'ai')  # Default to AI if not specified
            token = topic_data.get('selected_token', 'BTC')   # Default to BTC if not specified
        
            content, metadata = self._generate_tech_content(tech_category, token, market_data)
        
            # Post the content
            return self._post_tech_content(content, metadata)
        
        except Exception as e:
            logger.log_error("Tech Educational Content", str(e))
            return False

    def _select_best_token_for_timeframe(self, market_data: Dict[str, Any], timeframe: str) -> Optional[str]:
        """
        Select the best token to use for a specific timeframe post
        Uses momentum scoring, prediction accuracy, and market activity
        
        Args:
            market_data: Market data dictionary
            timeframe: Timeframe to select for
            
        Returns:
            Best token symbol for the timeframe
        """
        candidates = []
        
        # Get tokens with data
        available_tokens = [t for t in self.reference_tokens if t in market_data]
        
        # Score each token
        for token in available_tokens:
            # Calculate momentum score
            momentum_score = self._calculate_momentum_score(token, market_data, timeframe)
            
            # Calculate activity score based on recent volume and price changes
            token_data = market_data.get(token, {})
            volume = token_data.get('volume', 0)
            price_change = abs(token_data.get('price_change_percentage_24h', 0))
            
            # Get volume trend
            volume_trend, _ = self._analyze_volume_trend(volume, 
                                                    self._get_historical_volume_data(token, timeframe=timeframe),
                                                    timeframe=timeframe)
            
            # Get historical prediction accuracy
            perf_stats = self.config.db.get_prediction_performance(token=token, timeframe=timeframe)
            
            # Calculate accuracy score
            accuracy_score = 0
            if perf_stats:
                accuracy = perf_stats[0].get('accuracy_rate', 0)
                total_preds = perf_stats[0].get('total_predictions', 0)
                
                # Only consider accuracy if we have enough data
                if total_preds >= 5:
                    accuracy_score = accuracy * (min(total_preds, 20) / 20)  # Scale by number of predictions up to 20
            
            # Calculate recency score - prefer tokens we haven't posted about recently
            recency_score = 0
            
            # Check when this token was last posted for this timeframe
            recent_posts = self.config.db.get_recent_posts(hours=48, timeframe=timeframe)
            
            token_posts = [p for p in recent_posts if token.upper() in p.get('content', '')]
            
            if not token_posts:
                # Never posted - maximum recency score
                recency_score = 100
            else:
                # Calculate hours since last post
                last_posts_times = [strip_timezone(datetime.fromisoformat(p.get('timestamp', datetime.min.isoformat()))) for p in token_posts]
                if last_posts_times:
                    last_post_time = max(last_posts_times)
                    hours_since = safe_datetime_diff(datetime.now(), last_post_time) / 3600
                    
                    # Scale recency score based on timeframe
                    if timeframe == "1h":
                        recency_score = min(100, hours_since * 10)  # Max score after 10 hours
                    elif timeframe == "24h":
                        recency_score = min(100, hours_since * 2)   # Max score after 50 hours
                    else:  # 7d
                        recency_score = min(100, hours_since * 0.5)  # Max score after 200 hours
            
            # Combine scores with timeframe-specific weightings
            if timeframe == "1h":
                # For hourly, momentum and price action matter most
                total_score = (
                    momentum_score * 0.5 +
                    price_change * 3.0 +
                    volume_trend * 0.7 +
                    accuracy_score * 0.3 +
                    recency_score * 0.4
                )
            elif timeframe == "24h":
                # For daily, balance between momentum, accuracy and recency
                total_score = (
                    momentum_score * 0.4 +
                    price_change * 2.0 +
                    volume_trend * 0.8 +
                    accuracy_score * 0.5 +
                    recency_score * 0.6
                )
            else:  # 7d
                # For weekly, accuracy and longer-term views matter more
                total_score = (
                    momentum_score * 0.3 +
                    price_change * 1.0 +
                    volume_trend * 1.0 +
                    accuracy_score * 0.8 +
                    recency_score * 0.8
                )
            
            candidates.append((token, total_score))
        
        # Sort by total score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        logger.logger.debug(f"Token candidates for {timeframe}: {candidates[:3]}")
        
        return candidates[0][0] if candidates else None

    def _check_for_posts_to_reply(self, market_data: Dict[str, Any]) -> bool:
        """
        Check for posts to reply to and generate replies
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Boolean indicating if any replies were posted
        """
        now = strip_timezone(datetime.now())
    
        # Check if it's time to look for posts to reply to
        time_since_last_check = safe_datetime_diff(now, self.last_reply_check) / 60
        if time_since_last_check < self.reply_check_interval:
            logger.logger.debug(f"Skipping reply check, {time_since_last_check:.1f} minutes since last check (interval: {self.reply_check_interval})")
            return False
        
        # Also check cooldown period
        time_since_last_reply = safe_datetime_diff(now, self.last_reply_time) / 60
        if time_since_last_reply < self.reply_cooldown:
            logger.logger.debug(f"In reply cooldown period, {time_since_last_reply:.1f} minutes since last reply (cooldown: {self.reply_cooldown})")
            return False
        
        logger.logger.info("Starting check for posts to reply to")
        self.last_reply_check = now
    
        try:
            # Scrape timeline for posts
            posts = self.timeline_scraper.scrape_timeline(count=self.max_replies_per_cycle * 2)  # Get more to filter
            logger.logger.info(f"Timeline scraping completed - found {len(posts) if posts else 0} posts")
        
            if not posts:
                logger.logger.warning("No posts found during timeline scraping")
                return False

            # Log sample posts for debugging
            for i, post in enumerate(posts[:3]):  # Log first 3 posts
                logger.logger.info(f"Sample post {i}: {post.get('text', '')[:100]}...")

            # Find market-related posts
            logger.logger.info(f"Finding market-related posts among {len(posts)} scraped posts")
            market_posts = self.content_analyzer.find_market_related_posts(posts)
            logger.logger.info(f"Found {len(market_posts)} market-related posts, checking which ones need replies")
            
            # Filter out posts we've already replied to
            unreplied_posts = self.timeline_scraper.filter_already_replied_posts(market_posts)
            logger.logger.info(f"Found {len(unreplied_posts)} unreplied market-related posts")
            if unreplied_posts:
                for i, post in enumerate(unreplied_posts[:3]):
                    logger.logger.info(f"Sample unreplied post {i}: {post.get('text', '')[:100]}...")
            
            if not unreplied_posts:
                return False
                
            # Prioritize posts (engagement, relevance, etc.)
            prioritized_posts = self.timeline_scraper.prioritize_posts(unreplied_posts)
            
            # Limit to max replies per cycle
            posts_to_reply = prioritized_posts[:self.max_replies_per_cycle]
            
            # Generate and post replies
            logger.logger.info(f"Starting to reply to {len(posts_to_reply)} prioritized posts")
            successful_replies = self.reply_handler.reply_to_posts(posts_to_reply, market_data, max_replies=self.max_replies_per_cycle)
            
            if successful_replies > 0:
                logger.logger.info(f"Successfully posted {successful_replies} replies")
                self.last_reply_time = now                    
                return True
            else:
                logger.logger.info("No replies were successfully posted")
                return False
                
        except Exception as e:
            logger.log_error("Check For Posts To Reply", str(e))
            return False

    @ensure_naive_datetimes
    def _cleanup(self) -> None:
        """Cleanup resources and save state"""
        try:
            # Stop prediction thread if running
            if self.prediction_thread_running:
                self.prediction_thread_running = False
                if self.prediction_thread and self.prediction_thread.is_alive():
                    self.prediction_thread.join(timeout=5)
                logger.logger.info("Stopped prediction thread")
           
            # Close browser
            if self.browser:
                logger.logger.info("Closing browser...")
                try:
                    self.browser.close_browser()
                    time.sleep(1)
                except Exception as e:
                    logger.logger.warning(f"Error during browser close: {str(e)}")
           
            # Save timeframe prediction data to database for persistence
            try:
                timeframe_state = {
                    "predictions": self.timeframe_predictions,
                    "last_post": {tf: ts.isoformat() for tf, ts in self.timeframe_last_post.items()},
                    "next_scheduled": {tf: ts.isoformat() for tf, ts in self.next_scheduled_posts.items()},
                    "accuracy": self.prediction_accuracy
                }
               
                # Store using the generic JSON data storage
                self.config.db._store_json_data(
                    data_type="timeframe_state",
                    data=timeframe_state
                )
                logger.logger.info("Saved timeframe state to database")
            except Exception as e:
                logger.logger.warning(f"Failed to save timeframe state: {str(e)}")
           
            # Close database connection
            if self.config:
                self.config.cleanup()
               
            logger.log_shutdown()
        except Exception as e:
            logger.log_error("Cleanup", str(e))

    @ensure_naive_datetimes
    def _ensure_datetime(self, value) -> datetime:
        """
        Convert value to datetime if it's a string, ensuring timezone-naive datetime
        
        Args:
            value: Value to convert
            
        Returns:
            Datetime object (timezone-naive)
        """
        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value)
                return strip_timezone(dt)
            except ValueError:
                logger.logger.warning(f"Could not parse datetime string: {value}")
                return strip_timezone(datetime.min)
        elif isinstance(value, datetime):
            return strip_timezone(value)
        return strip_timezone(datetime.min)

    def _get_crypto_data(self) -> Optional[Dict[str, Any]]:
        """
        Fetch crypto data from CoinGecko with retries
        
        Returns:
            Market data dictionary or None if failed
        """
        try:
            params = {
                **self.config.get_coingecko_params(),
                'ids': ','.join(self.target_chains.values()), 
                'sparkline': True 
            }
            
            data = self.coingecko.get_market_data(params)
            if not data:
                logger.logger.error("Failed to fetch market data from CoinGecko")
                return None
                
            formatted_data = {
                coin['symbol'].upper(): {
                    'current_price': coin['current_price'],
                    'volume': coin['total_volume'],
                    'price_change_percentage_24h': coin['price_change_percentage_24h'],
                    'sparkline': coin.get('sparkline_in_7d', {}).get('price', []),
                    'market_cap': coin['market_cap'],
                    'market_cap_rank': coin['market_cap_rank'],
                    'total_supply': coin.get('total_supply'),
                    'max_supply': coin.get('max_supply'),
                    'circulating_supply': coin.get('circulating_supply'),
                    'ath': coin.get('ath'),
                    'ath_change_percentage': coin.get('ath_change_percentage')
                } for coin in data
            }
            
            # Map to correct symbol if needed (particularly for POL which might return as MATIC)
            symbol_corrections = {'MATIC': 'POL'}
            for old_sym, new_sym in symbol_corrections.items():
                if old_sym in formatted_data and new_sym not in formatted_data:
                    formatted_data[new_sym] = formatted_data[old_sym]
                    logger.logger.debug(f"Mapped {old_sym} data to {new_sym}")
            
            # Log API usage statistics
            stats = self.coingecko.get_request_stats()
            logger.logger.debug(
                f"CoinGecko API stats - Daily requests: {stats['daily_requests']}, "
                f"Failed: {stats['failed_requests']}, Cache size: {stats['cache_size']}"
            )
            
            # Store market data in database
            for chain, chain_data in formatted_data.items():
                self.config.db.store_market_data(chain, chain_data)
            
            # Check if all data was retrieved
            missing_tokens = [token for token in self.reference_tokens if token not in formatted_data]
            if missing_tokens:
                logger.logger.warning(f"Missing data for tokens: {', '.join(missing_tokens)}")
                
                # Try fallback mechanism for missing tokens
                if 'POL' in missing_tokens and 'MATIC' in formatted_data:
                    formatted_data['POL'] = formatted_data['MATIC']
                    missing_tokens.remove('POL')
                    logger.logger.info("Applied fallback for POL using MATIC data")
                
            logger.logger.info(f"Successfully fetched crypto data for {', '.join(formatted_data.keys())}")
            return formatted_data
                
        except Exception as e:
            logger.log_error("CoinGecko API", str(e))
            return None

    @ensure_naive_datetimes
    def _load_saved_timeframe_state(self) -> None:
        """Load previously saved timeframe state from database with enhanced datetime handling"""
        try:
            # Query the latest timeframe state
            conn, cursor = self.config.db._get_connection()
        
            cursor.execute("""
                SELECT data 
                FROM generic_json_data 
                WHERE data_type = 'timeframe_state'
                ORDER BY timestamp DESC
                LIMIT 1
            """)
        
            result = cursor.fetchone()
        
            if not result:
                logger.logger.info("No saved timeframe state found")
                return
            
            # Parse the saved state
            state_json = result[0]
            state = json.loads(state_json)
        
            # Restore timeframe predictions
            for timeframe, predictions in state.get("predictions", {}).items():
                self.timeframe_predictions[timeframe] = predictions
        
            # Restore last post times with proper datetime handling
            for timeframe, timestamp in state.get("last_post", {}).items():
                try:
                    # Convert string to datetime and ensure it's timezone-naive
                    dt = datetime.fromisoformat(timestamp)
                    self.timeframe_last_post[timeframe] = strip_timezone(dt)
                    logger.logger.debug(f"Restored last post time for {timeframe}: {self.timeframe_last_post[timeframe]}")
                except (ValueError, TypeError) as e:
                    # If timestamp can't be parsed, use a safe default
                    logger.logger.warning(f"Could not parse timestamp for {timeframe} last post: {str(e)}")
                    self.timeframe_last_post[timeframe] = strip_timezone(datetime.now() - timedelta(hours=3))
        
            # Restore next scheduled posts with proper datetime handling
            for timeframe, timestamp in state.get("next_scheduled", {}).items():
                try:
                    # Convert string to datetime and ensure it's timezone-naive
                    dt = datetime.fromisoformat(timestamp)
                    scheduled_time = strip_timezone(dt)
                
                    # If scheduled time is in the past, reschedule
                    now = strip_timezone(datetime.now())
                    if scheduled_time < now:
                        delay_hours = self.timeframe_posting_frequency.get(timeframe, 1) * random.uniform(0.1, 0.5)
                        self.next_scheduled_posts[timeframe] = now + timedelta(hours=delay_hours)
                        logger.logger.debug(f"Rescheduled {timeframe} post for {self.next_scheduled_posts[timeframe]}")
                    else:
                        self.next_scheduled_posts[timeframe] = scheduled_time
                        logger.logger.debug(f"Restored next scheduled time for {timeframe}: {self.next_scheduled_posts[timeframe]}")
                except (ValueError, TypeError) as e:
                    # If timestamp can't be parsed, set a default
                    logger.logger.warning(f"Could not parse timestamp for {timeframe} next scheduled post: {str(e)}")
                    delay_hours = self.timeframe_posting_frequency.get(timeframe, 1) * random.uniform(0.1, 0.5)
                    self.next_scheduled_posts[timeframe] = strip_timezone(datetime.now() + timedelta(hours=delay_hours))
        
            # Restore accuracy tracking
            self.prediction_accuracy = state.get("accuracy", {timeframe: {'correct': 0, 'total': 0} for timeframe in self.timeframes})
        
            # Debug log the restored state
            logger.logger.debug("Restored timeframe state:")
            for tf in self.timeframes:
                last_post = self.timeframe_last_post.get(tf)
                next_post = self.next_scheduled_posts.get(tf)
                logger.logger.debug(f"  {tf}: last={last_post}, next={next_post}")
        
            logger.logger.info("Restored timeframe state from database")
        
        except Exception as e:
            logger.log_error("Load Timeframe State", str(e))
            # Create safe defaults for all timing data
            now = strip_timezone(datetime.now())
            for timeframe in self.timeframes:
                self.timeframe_last_post[timeframe] = now - timedelta(hours=3)
                delay_hours = self.timeframe_posting_frequency.get(timeframe, 1) * random.uniform(0.1, 0.5)
                self.next_scheduled_posts[timeframe] = now + timedelta(hours=delay_hours)
        
            logger.logger.warning("Using default timeframe state due to error")

    def _get_historical_price_data(self, chain: str, hours: int = None, timeframe: str = "1h") -> List[Dict[str, Any]]:
        """
        Get historical price data for the specified time period
        Adjusted based on timeframe for appropriate historical context
        
        Args:
            chain: Token/chain symbol
            hours: Number of hours of historical data to retrieve
            timeframe: Timeframe for the data (1h, 24h, 7d)
            
        Returns:
            List of historical price data points
        """
        try:
            # Adjust time period based on timeframe if not specified
            if hours is None:
                if timeframe == "1h":
                    hours = 24  # Last 24 hours for hourly predictions
                elif timeframe == "24h":
                    hours = 7 * 24  # Last 7 days for daily predictions
                elif timeframe == "7d":
                    hours = 30 * 24  # Last 30 days for weekly predictions
                else:
                    hours = 24
            
            # Query the database
            return self.config.db.get_recent_market_data(chain, hours)
            
        except Exception as e:
            logger.log_error(f"Historical Price Data - {chain} ({timeframe})", str(e))
            return []
   
    @ensure_naive_datetimes
    def _get_token_timeframe_performance(self, token: str) -> Dict[str, Dict[str, Any]]:
        """
        Get prediction performance statistics for a token across all timeframes
        
        Args:
            token: Token symbol
            
        Returns:
            Dictionary of performance statistics by timeframe
        """
        try:
            result = {}
            
            # Gather performance for each timeframe
            for timeframe in self.timeframes:
                perf_stats = self.config.db.get_prediction_performance(token=token, timeframe=timeframe)
                
                if perf_stats:
                    result[timeframe] = {
                        "accuracy": perf_stats[0].get("accuracy_rate", 0),
                        "total": perf_stats[0].get("total_predictions", 0),
                        "correct": perf_stats[0].get("correct_predictions", 0),
                        "avg_deviation": perf_stats[0].get("avg_deviation", 0)
                    }
                else:
                    result[timeframe] = {
                        "accuracy": 0,
                        "total": 0,
                        "correct": 0,
                        "avg_deviation": 0
                    }
            
            # Get cross-timeframe comparison
            cross_comparison = self.config.db.get_prediction_comparison_across_timeframes(token)
            
            if cross_comparison:
                result["best_timeframe"] = cross_comparison.get("best_timeframe", {}).get("timeframe", "1h")
                result["overall"] = cross_comparison.get("overall", {})
            
            return result
            
        except Exception as e:
            logger.log_error(f"Get Token Timeframe Performance - {token}", str(e))
            return {tf: {"accuracy": 0, "total": 0, "correct": 0, "avg_deviation": 0} for tf in self.timeframes}
   
    @ensure_naive_datetimes
    def _get_all_active_predictions(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Get all active predictions organized by timeframe and token
        
        Returns:
            Dictionary of active predictions by timeframe and token
        """
        try:
            result = {tf: {} for tf in self.timeframes}
            
            # Get active predictions from the database
            active_predictions = self.config.db.get_active_predictions()
            
            for prediction in active_predictions:
                timeframe = prediction.get("timeframe", "1h")
                token = prediction.get("token", "")
                
                if timeframe in result and token:
                    result[timeframe][token] = prediction
            
            # Merge with in-memory predictions which might be more recent
            for timeframe, predictions in self.timeframe_predictions.items():
                for token, prediction in predictions.items():
                    result.setdefault(timeframe, {})[token] = prediction
            
            return result
            
        except Exception as e:
            logger.log_error("Get All Active Predictions", str(e))
            return {tf: {} for tf in self.timeframes}

    @ensure_naive_datetimes
    def _evaluate_expired_timeframe_predictions(self) -> Dict[str, int]:
        """
        Find and evaluate expired predictions across all timeframes
        
        Returns:
            Dictionary with count of evaluated predictions by timeframe
        """
        try:
            # Get expired unevaluated predictions
            all_expired = self.config.db.get_expired_unevaluated_predictions()
            
            if not all_expired:
                logger.logger.debug("No expired predictions to evaluate")
                return {tf: 0 for tf in self.timeframes}
                
            # Group by timeframe
            expired_by_timeframe = {tf: [] for tf in self.timeframes}
            
            for prediction in all_expired:
                timeframe = prediction.get("timeframe", "1h")
                if timeframe in expired_by_timeframe:
                    expired_by_timeframe[timeframe].append(prediction)
            
            # Get current market data for evaluation
            market_data = self._get_crypto_data()
            if not market_data:
                logger.logger.error("Failed to fetch market data for prediction evaluation")
                return {tf: 0 for tf in self.timeframes}
            
            # Track evaluated counts
            evaluated_counts = {tf: 0 for tf in self.timeframes}
            
            # Evaluate each prediction by timeframe
            for timeframe, predictions in expired_by_timeframe.items():
                for prediction in predictions:
                    token = prediction["token"]
                    prediction_id = prediction["id"]
                    
                    # Get current price for the token
                    token_data = market_data.get(token, {})
                    if not token_data:
                        logger.logger.warning(f"No current price data for {token}, skipping evaluation")
                        continue
                        
                    current_price = token_data.get("current_price", 0)
                    if current_price == 0:
                        logger.logger.warning(f"Zero price for {token}, skipping evaluation")
                        continue
                        
                    # Record the outcome
                    result = self.config.db.record_prediction_outcome(prediction_id, current_price)
                    
                    if result:
                        logger.logger.debug(f"Evaluated {timeframe} prediction {prediction_id} for {token}")
                        evaluated_counts[timeframe] += 1
                    else:
                        logger.logger.error(f"Failed to evaluate {timeframe} prediction {prediction_id} for {token}")
            
            # Log evaluation summaries
            for timeframe, count in evaluated_counts.items():
                if count > 0:
                    logger.logger.info(f"Evaluated {count} expired {timeframe} predictions")
            
            # Update prediction performance metrics
            self._update_prediction_performance_metrics()
            
            return evaluated_counts
            
        except Exception as e:
            logger.log_error("Evaluate Expired Timeframe Predictions", str(e))
            return {tf: 0 for tf in self.timeframes}

    @ensure_naive_datetimes
    def _update_prediction_performance_metrics(self) -> None:
        """Update in-memory prediction performance metrics from database"""
        try:
            # Get overall performance by timeframe
            for timeframe in self.timeframes:
                performance = self.config.db.get_prediction_performance(timeframe=timeframe)
                
                total_correct = sum(p.get("correct_predictions", 0) for p in performance)
                total_predictions = sum(p.get("total_predictions", 0) for p in performance)
                
                # Update in-memory tracking
                self.prediction_accuracy[timeframe] = {
                    'correct': total_correct,
                    'total': total_predictions
                }
            
            # Log overall performance
            for timeframe, stats in self.prediction_accuracy.items():
                if stats['total'] > 0:
                    accuracy = (stats['correct'] / stats['total']) * 100
                    logger.logger.info(f"{timeframe} prediction accuracy: {accuracy:.1f}% ({stats['correct']}/{stats['total']})")
                    
        except Exception as e:
            logger.log_error("Update Prediction Performance Metrics", str(e))

    def _analyze_volume_trend(self, current_volume: float, historical_data: List[Dict[str, Any]], 
                             timeframe: str = "1h") -> Tuple[float, str]:
        """
        Analyze volume trend over the window period, adjusted for timeframe
        
        Args:
            current_volume: Current volume value
            historical_data: Historical volume data
            timeframe: Timeframe for analysis
            
        Returns:
            Tuple of (percentage_change, trend_description)
        """
        if not historical_data:
            return 0.0, "insufficient_data"
            
        try:
            # Adjust trend thresholds based on timeframe
            if timeframe == "1h":
                SIGNIFICANT_THRESHOLD = self.config.VOLUME_TREND_THRESHOLD  # Default (usually 15%)
                MODERATE_THRESHOLD = 5.0
            elif timeframe == "24h":
                SIGNIFICANT_THRESHOLD = 20.0  # Higher threshold for daily predictions
                MODERATE_THRESHOLD = 10.0
            else:  # 7d
                SIGNIFICANT_THRESHOLD = 30.0  # Even higher for weekly predictions
                MODERATE_THRESHOLD = 15.0
            
            # Calculate average volume excluding the current volume
            historical_volumes = [entry['volume'] for entry in historical_data]
            avg_volume = statistics.mean(historical_volumes) if historical_volumes else current_volume
            
            # Calculate percentage change
            volume_change = ((current_volume - avg_volume) / avg_volume) * 100 if avg_volume > 0 else 0
            
            # Determine trend based on timeframe-specific thresholds
            if volume_change >= SIGNIFICANT_THRESHOLD:
                trend = "significant_increase"
            elif volume_change <= -SIGNIFICANT_THRESHOLD:
                trend = "significant_decrease"
            elif volume_change >= MODERATE_THRESHOLD:
                trend = "moderate_increase"
            elif volume_change <= -MODERATE_THRESHOLD:
                trend = "moderate_decrease"
            else:
                trend = "stable"
                
            logger.logger.debug(
                f"Volume trend analysis ({timeframe}): {volume_change:.2f}% change from average. "
                f"Current: {current_volume:,.0f}, Avg: {avg_volume:,.0f}, "
                f"Trend: {trend}"
            )
            
            return volume_change, trend
            
        except Exception as e:
            logger.log_error(f"Volume Trend Analysis - {timeframe}", str(e))
            return 0.0, "error"
        
                                            
      
