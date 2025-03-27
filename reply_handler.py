#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Any, Union
import time
import random
import re
from datetime import datetime
import anthropic
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
    WebDriverException,
    ElementClickInterceptedException
)
from selenium.webdriver.common.keys import Keys

from utils.logger import logger

class ReplyHandler:
    """
    Handler for generating and posting replies to timeline posts
    """
    
    def __init__(self, browser, config, claude_client, coingecko=None, db=None):
        """
        Initialize the reply handler
        
        Args:
            browser: Browser instance for web interaction
            config: Configuration instance containing settings
            claude_client: Claude API client for generating replies
            coingecko: CoinGecko handler for market data (optional)
            db: Database instance for storing reply data (optional)
        """
        self.browser = browser
        self.config = config
        self.claude_client = claude_client
        self.coingecko = coingecko
        self.db = db
        
        self.max_retries = 3
        self.retry_delay = 5
        self.claude_model = "claude-3-7-sonnet-20250219"  # Using the latest model
        self.max_tokens = 300
        
        # Element selectors for reply interactions
        self.reply_button_selector = '[data-testid="reply"]'
        self.reply_textarea_selector = '[data-testid="tweetTextarea_0"]'
        self.reply_send_button_selector = '[data-testid="tweetButton"]'
        
        # Track posts we've recently replied to (in-memory cache)
        self.recent_replies = []
        self.max_recent_replies = 100
        
        logger.logger.info("Reply handler initialized")
    
    def generate_reply(self, post: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Generate a witty market-related reply using Claude API
        
        Args:
            post: Post data dictionary
            market_data: Market data from CoinGecko (optional)
            
        Returns:
            Generated reply text or None if generation failed
        """
        try:
            # Extract post details
            post_text = post.get('text', '')
            author_name = post.get('author_name', 'someone')
            author_handle = post.get('author_handle', '@someone')
            
            # Prepare market data context if available
            market_context = ""
            market_symbols = []
            
            if market_data:
                # Build market context for top coins
                top_coins = [
                    ('BTC', 'Bitcoin'), ('ETH', 'Ethereum'), ('SOL', 'Solana'),
                    ('XRP', 'Ripple'), ('BNB', 'Binance Coin')
                ]
                
                market_context = "Current market data:\n"
                
                for symbol, name in top_coins:
                    if symbol in market_data:
                        coin_data = market_data[symbol]
                        price = coin_data.get('current_price', 0)
                        change_24h = coin_data.get('price_change_percentage_24h', 0)
                        
                        market_context += f"- {name} ({symbol}): ${price:,.2f} ({change_24h:+.2f}%)\n"
                        market_symbols.append(symbol.lower())
            
            # Look for any crypto/market symbols in the post
            all_symbols = [
                'btc', 'eth', 'sol', 'xrp', 'bnb', 'avax', 'dot', 'uni', 
                'near', 'aave', 'matic', 'fil', 'pol', 'ada', 'doge', 'shib'
            ]
            
            mentioned_symbols = [symbol for symbol in all_symbols if symbol in post_text.lower()]
            
            # Add symbols specifically mentioned in the post
            if mentioned_symbols:
                market_symbols.extend(mentioned_symbols)
                
            # Remove duplicates
            market_symbols = list(set(market_symbols))
            
            # Detect potential market topics in the post
            market_topics = self._detect_market_topics(post_text)
            
            # Build the prompt for Claude
            prompt = f"""You are an intelligent, witty crypto/market commentator replying to posts on social media. 
You specialize in providing informative but humorous replies about cryptocurrency and financial markets.

The post you're replying to:
Author: {author_name} ({author_handle})
Post: "{post_text}"

{market_context}

Your task is to write a brief, intelligent, witty reply with a hint of market knowledge or insight. The reply should:
1. Be conversational and casual in tone
2. Include relevant market insights when appropriate
3. Be humorous but not over-the-top or meme-heavy
4. Be concise (1-3 short sentences, maximum 240 characters)
5. Not use hashtags, emojis, or excessive special characters
6. Sound like a real person, not an automated bot
7. Not appear overly promotional or financial-advice-like
8. Match the tone of the original post when appropriate

{"Specifically reference " + ", ".join(market_symbols[:2]) + " in your reply" if market_symbols else ""}
{"Consider addressing these topics mentioned in the post: " + ", ".join(market_topics) if market_topics else ""}

Your reply (maximum 240 characters):
"""

            # Generate reply using Claude
            logger.logger.debug(f"Generating reply to post by {author_handle}")
            response = self.claude_client.messages.create(
                model=self.claude_model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract reply text
            reply_text = response.content[0].text.strip()
            
            # Make sure the reply isn't too long for Twitter (240 chars max)
            if len(reply_text) > 240:
                # Truncate with preference to complete sentences
                last_period = reply_text[:240].rfind('.')
                last_question = reply_text[:240].rfind('?')
                last_exclamation = reply_text[:240].rfind('!')
                last_punctuation = max(last_period, last_question, last_exclamation)
                
                if last_punctuation > 180:  # If we can get a substantial reply with complete sentence
                    reply_text = reply_text[:last_punctuation+1]
                else:
                    # Find last space to avoid cutting words
                    last_space = reply_text[:240].rfind(' ')
                    if last_space > 180:
                        reply_text = reply_text[:last_space]
                    else:
                        # Hard truncate as last resort
                        reply_text = reply_text[:237] + "..."
            
            logger.logger.info(f"Generated reply ({len(reply_text)} chars)")
            return reply_text
            
        except Exception as e:
            logger.log_error("Reply Generation", str(e))
            
            # Fallback replies if Claude API fails
            fallback_replies = [
                "Interesting perspective on the market. The data seems to tell a different story though.",
                "Not sure if I agree with this take. Market fundamentals suggest otherwise.",
                "Classic crypto market analysis - half right, half wishful thinking.",
                "Hmm, that's one way to interpret what's happening. The charts paint a nuanced picture though.",
                "I see your point, but have you considered the broader market implications?"
            ]
            return random.choice(fallback_replies)
    
    def _detect_market_topics(self, text: str) -> List[str]:
        """
        Detect market-related topics in the post text
        
        Args:
            text: Post text
            
        Returns:
            List of detected market topics
        """
        topics = []
        
        # Topic patterns with regex
        patterns = [
            (r'\b(bull|bullis(h|m))\b', 'bullish sentiment'),
            (r'\b(bear|bearis(h|m))\b', 'bearish sentiment'),
            (r'\b(crash|dump|collapse)\b', 'market downturn'),
            (r'\b(pump|moon|rally|surge)\b', 'price rally'),
            (r'\b(fed|federal reserve|interest rate|inflation)\b', 'macroeconomic factors'),
            (r'\b(hold|hodl|holding)\b', 'investment strategy'),
            (r'\b(buy|buying|bought)\b', 'buying activity'),
            (r'\b(sell|selling|sold)\b', 'selling pressure'),
            (r'\baltcoin season|alt season\b', 'altcoin performance'),
            (r'\btechnical analysis|TA|support|resistance\b', 'technical analysis'),
            (r'\b(volume|liquidity)\b', 'market liquidity'),
            (r'\b(fund|investor|institutional)\b', 'institutional investment'),
            (r'\bregulat(ion|ory|e)\b', 'regulatory discussion'),
            (r'\b(trade|trading)\b', 'trading activity'),
        ]
        
        text_lower = text.lower()
        
        for pattern, topic in patterns:
            if re.search(pattern, text_lower):
                topics.append(topic)
                
        return topics
    
    def post_reply(self, post: Dict[str, Any], reply_text: str) -> bool:
        """
        Navigate to the post and submit a reply
        
        Args:
            post: Post data dictionary
            reply_text: Text to reply with
            
        Returns:
            True if reply was successfully posted, False otherwise
        """
        if not reply_text:
            logger.logger.error("Cannot post empty reply")
            return False
            
        # Check if post has a URL to navigate to
        post_url = post.get('post_url')
        if not post_url:
            logger.logger.error("No URL available for the post")
            return False
            
        post_id = post.get('post_id', 'unknown')
        author_handle = post.get('author_handle', '@unknown')
        
        # Check if we've already replied to this post (memory cache)
        if self._already_replied(post_id):
            logger.logger.info(f"Already replied to post {post_id} by {author_handle}")
            return False
            
        # Check if we've already replied to this post (database)
        if self.db and self.db.check_if_post_replied(post_id, post_url):
            logger.logger.info(f"Already replied to post {post_id} by {author_handle} (DB)")
            return False
            
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                # Navigate to the post
                logger.logger.debug(f"Navigating to post: {post_url}")
                self.browser.driver.get(post_url)
                
                # Wait for page to load
                WebDriverWait(self.browser.driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="tweet"]'))
                )
                
                # Find the reply button and click it
                reply_button = WebDriverWait(self.browser.driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, self.reply_button_selector))
                )
                
                # Click the reply button
                self.browser.js_click(self.reply_button_selector)
                
                # Wait for reply text area to appear
                reply_textarea = WebDriverWait(self.browser.driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, self.reply_textarea_selector))
                )
                
                # Enter reply text
                reply_textarea.click()
                time.sleep(1)
                
                # Clear any existing text
                reply_textarea.clear()
                
                # Enter reply text using the safe method
                self.browser.safe_send_keys(self.reply_textarea_selector, reply_text)
                
                # Wait a moment for the Send button to become active
                time.sleep(2)
                
                # Click the reply/send button
                reply_send_button = WebDriverWait(self.browser.driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, self.reply_send_button_selector))
                )
                
                # Try JavaScript click first
                success = self.browser.js_click(self.reply_send_button_selector)
                
                if not success:
                    # Try direct click as fallback
                    reply_send_button.click()
                
                # Wait for reply to be sent (button disappears or changes)
                time.sleep(5)
                
                # Check if the reply was successfully posted
                if self._verify_reply_posted():
                    logger.logger.info(f"Successfully replied to post by {author_handle}")
                    
                    # Add to recent replies list (memory cache)
                    self._add_to_recent_replies(post_id)
                    
                    # Store in database if available
                    if self.db:
                        self.db.store_reply(
                            post_id=post_id,
                            post_url=post_url,
                            post_author=author_handle,
                            post_text=post.get('text', ''),
                            reply_text=reply_text,
                            reply_time=datetime.now()
                        )
                    
                    return True
                else:
                    logger.logger.warning(f"Reply may not have been posted, verification failed")
                    retry_count += 1
                    time.sleep(self.retry_delay)
                    
            except TimeoutException:
                logger.logger.warning(f"Timeout while trying to reply to post (attempt {retry_count + 1})")
                retry_count += 1
                time.sleep(self.retry_delay)
                
            except (ElementClickInterceptedException, NoSuchElementException) as e:
                logger.logger.warning(f"Element interaction error: {str(e)} (attempt {retry_count + 1})")
                retry_count += 1
                time.sleep(self.retry_delay)
                
            except Exception as e:
                logger.log_error("Reply Posting", str(e))
                retry_count += 1
                time.sleep(self.retry_delay)
        
        logger.logger.error(f"Failed to post reply to {author_handle} after {self.max_retries} attempts")
        return False
    
    def _verify_reply_posted(self) -> bool:
        """
        Verify that a reply was successfully posted
        
        Returns:
            True if verification succeeded, False otherwise
        """
        try:
            # Look for indicators that the reply was posted
            # This could be a success message, or returning to the main tweet page
            
            # Check if reply compose area is no longer visible
            try:
                WebDriverWait(self.browser.driver, 5).until_not(
                    EC.presence_of_element_located((By.CSS_SELECTOR, self.reply_textarea_selector))
                )
                return True
            except TimeoutException:
                # If the reply area is still visible, the reply probably failed
                return False
            
        except Exception as e:
            logger.logger.warning(f"Reply verification error: {str(e)}")
            return False
    
    def prioritize_posts_for_reply(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort posts by priority for replying (engagement, follower count, recency)
        
        Args:
            posts: List of post data dictionaries
            
        Returns:
            Sorted list of posts
        """
        # Define weights for different factors
        engagement_weight = 0.6
        recency_weight = 0.4
        
        # Calculate a combined score for each post
        for post in posts:
            # Engagement score (already calculated)
            engagement_score = post.get('engagement_score', 0)
            
            # Recency score (higher for more recent posts)
            timestamp = post.get('timestamp', datetime.now())
            hours_ago = (datetime.now() - timestamp).total_seconds() / 3600 if timestamp else 24
            recency_score = max(0, 100 - (hours_ago * 5))  # Decreases by 5 points per hour
            
            # Combined score
            combined_score = (
                engagement_score * engagement_weight +
                recency_score * recency_weight
            )
            
            post['reply_priority_score'] = combined_score
            
        # Sort by combined score (highest first)
        sorted_posts = sorted(posts, key=lambda x: x.get('reply_priority_score', 0), reverse=True)
        
        return sorted_posts
    
    def reply_to_posts(self, posts: List[Dict[str, Any]], market_data: Optional[Dict[str, Any]] = None, max_replies: int = 5) -> int:
        """
        Generate and post replies to a list of posts
        
        Args:
            posts: List of post data dictionaries
            market_data: Market data from CoinGecko (optional)
            max_replies: Maximum number of replies to send
            
        Returns:
            Number of successful replies
        """
        if not posts:
            return 0
            
        # Filter out posts we've already replied to
        filtered_posts = [p for p in posts if not self._already_replied(p.get('post_id'))]
        
        if self.db:
            filtered_posts = [p for p in filtered_posts 
                             if not self.db.check_if_post_replied(p.get('post_id'), p.get('post_url'))]
            
        # Prioritize remaining posts
        prioritized_posts = self.prioritize_posts_for_reply(filtered_posts)
        
        # Limit to max_replies
        posts_to_reply = prioritized_posts[:max_replies]
        
        successful_replies = 0
        
        for post in posts_to_reply:
            # Generate reply
            reply_text = self.generate_reply(post, market_data)
            
            if not reply_text:
                logger.logger.warning(f"Failed to generate reply for post by {post.get('author_handle')}")
                continue
                
            # Add slight delay between replies to avoid rate limiting
            if successful_replies > 0:
                delay = random.uniform(10, 20)  # 10-20 seconds between replies
                logger.logger.debug(f"Waiting {delay:.1f} seconds before next reply")
                time.sleep(delay)
                
            # Post reply
            if self.post_reply(post, reply_text):
                successful_replies += 1
            
        return successful_replies
    
    def _already_replied(self, post_id: str) -> bool:
        """
        Check if we've already replied to a post (memory cache)
        
        Args:
            post_id: Unique identifier for the post
            
        Returns:
            True if we've already replied, False otherwise
        """
        return post_id in self.recent_replies
    
    def _add_to_recent_replies(self, post_id: str) -> None:
        """
        Add a post ID to the recent replies list
        
        Args:
            post_id: Unique identifier for the post
        """
        if post_id in self.recent_replies:
            return
            
        self.recent_replies.append(post_id)
        
        # Keep the list at a reasonable size
        if len(self.recent_replies) > self.max_recent_replies:
            self.recent_replies.pop(0)  # Remove oldest entry
