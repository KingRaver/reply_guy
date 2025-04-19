#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Any, Union, Tuple
import time
import random
import re
from datetime import datetime
from datetime_utils import ensure_naive_datetimes, strip_timezone, safe_datetime_diff
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
    WebDriverException,
    ElementClickInterceptedException,
    ElementNotInteractableException,
    InvalidElementStateException
)
from selenium.webdriver.common.keys import Keys

from utils.logger import logger

class ReplyHandler:
    """
    Handler for generating and posting replies to timeline posts
    """
    
    def __init__(self, browser, config, llm_provider, coingecko=None, db=None):
        """
        Initialize the reply handler with multiple selectors for resilience
    
        Args:
            browser: Browser instance for web interaction
            config: Configuration instance containing settings
            llm_provider: LLM provider for generating replies
            coingecko: CoinGecko handler for market data (optional)
            db: Database instance for storing reply data (optional)
        """
        self.browser = browser
        self.config = config
        self.llm_provider = llm_provider
        self.coingecko = coingecko
        self.db = db
    
        # Configure retry settings
        self.max_retries = 3
        self.retry_delay = 5
        self.max_tokens = 300
    
        # Multiple selectors for each element type to increase resilience
        self.reply_button_selectors = [
            '[data-testid="reply"]',
            'div[role="button"] span:has-text("Reply")',
            'button[aria-label*="Reply"]',
            'div[role="button"][aria-label*="Reply"]',
            'div[aria-label*="Reply"]'
        ]
    
        self.reply_textarea_selectors = [
            '[data-testid="tweetTextarea_0"]',
            '[data-testid*="tweetTextarea"]',
            '[contenteditable="true"]',
            'div[role="textbox"]',
            'div.DraftEditor-root'
        ]
    
        self.reply_send_button_selectors = [
            '[data-testid="tweetButton"]',
            'button[data-testid="tweetButton"]',
            'div[role="button"]:has-text("Reply")',
            'div[role="button"]:has-text("Post")',
            'button[type="submit"]'
        ]
    
        # Track posts we've recently replied to (in-memory cache)
        self.recent_replies = []
        self.max_recent_replies = 100
    
        logger.logger.info("Reply handler initialized")

    def _strip_timezone(self, dt):
        """Helper method to convert datetime to naive"""
        if dt is None:
            return dt
        if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
            # Convert to UTC then remove tzinfo
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt

    def _safe_datetime_diff(self, dt1, dt2):
        """Safely calculate time difference between two datetime objects in seconds"""
        dt1 = self._strip_timezone(dt1)
        dt2 = self._strip_timezone(dt2)
        return (dt1 - dt2).total_seconds()

    @ensure_naive_datetimes
    def generate_reply(self, post: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Generate a witty market-related reply using LLM provider
        
        Args:
            post: Post data dictionary
            market_data: Market data from CoinGecko (optional)
            
        Returns:
            Generated reply text or None if generation failed
        """
        try:
            logger.logger.debug(f"Starting reply generation for post: '{post.get('text', '')[:50]}...'")
            
            # Extract post details
            post_text = post.get('text', '')
            author_name = post.get('author_name', 'someone')
            author_handle = post.get('author_handle', '@someone')
            
            logger.logger.debug(f"Generating reply to {author_handle}'s post: '{post_text[:50]}...'")
            
            # Prepare market data context if available
            market_context = ""
            market_symbols = []
            
            # Look for any crypto/market symbols in the post
            all_symbols = [
                'btc', 'eth', 'sol', 'xrp', 'bnb', 'avax', 'dot', 'uni', 
                'near', 'aave', 'matic', 'fil', 'pol', 'ada', 'doge', 'shib'
            ]
            
            mentioned_symbols = [symbol for symbol in all_symbols if symbol in post_text.lower()]
            logger.logger.debug(f"Found market symbols in post: {mentioned_symbols}")
            
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
            
            # Add symbols specifically mentioned in the post
            if mentioned_symbols:
                market_symbols.extend(mentioned_symbols)
                
            # Remove duplicates
            market_symbols = list(set(market_symbols))
            
            # Detect potential market topics in the post
            market_topics = self._detect_market_topics(post_text)
            logger.logger.debug(f"Detected market topics: {market_topics}")
            
            # Build the prompt for the LLM
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

            # Generate reply using LLM provider instead of direct Claude API
            logger.logger.debug(f"Sending prompt to LLM provider for reply generation")
            reply_text = self.llm_provider.generate_text(prompt, max_tokens=self.max_tokens)
            
            if not reply_text:
                logger.logger.warning("LLM provider returned empty response")
                return None
            
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
            
            logger.logger.info(f"Generated reply ({len(reply_text)} chars): {reply_text}")
            return reply_text
            
        except Exception as e:
            logger.log_error("Reply Generation", str(e))
            
            # Fallback replies if LLM provider fails
            fallback_replies = [
                "Interesting perspective on the market. The data seems to tell a different story though.",
                "Not sure if I agree with this take. Market fundamentals suggest otherwise.",
                "Classic crypto market analysis - half right, half wishful thinking.",
                "Hmm, that's one way to interpret what's happening. The charts paint a nuanced picture though.",
                "I see your point, but have you considered the broader market implications?"
            ]
            return random.choice(fallback_replies)

    def reply_to_posts(self, posts: List[Dict[str, Any]], market_data: Optional[Dict[str, Any]] = None, max_replies: int = 5) -> int:
        """
        Generate and post replies to a list of posts
    
        Args:
            posts: List of post data dictionaries
            market_data: Market data for context
            max_replies: Maximum number of replies to post
        
        Returns:
            Number of successful replies
        """
        if not posts:
            logger.logger.info("No posts to reply to")
            return 0
        
        successful_replies = 0
    
        for post in posts[:max_replies]:
            try:
                # Generate reply
                reply_text = self.generate_reply(post, market_data)
            
                if not reply_text:
                    logger.logger.warning(f"Failed to generate reply for post by {post.get('author_handle', 'unknown')}")
                    continue
                
                # Post the reply
                if self.post_reply(post, reply_text):
                    successful_replies += 1
                
                    # Capture metadata
                    metadata = self._capture_reply_metadata(post, reply_text)
                    logger.logger.debug(f"Reply metadata: {metadata}")
                
                    # Allow some time between replies to avoid rate limiting
                    if successful_replies < max_replies:
                        time.sleep(random.uniform(5, 10))
                else:
                    logger.logger.warning(f"Failed to post reply to {post.get('author_handle', 'unknown')}")
                
            except Exception as e:
                logger.log_error("Reply to Post", str(e))
                continue
            
        return successful_replies

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
        Navigate to the post and submit a reply with improved resilience
        
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
                time.sleep(3)  # Allow time for page to load
                
                # Try both twitter.com and x.com URLs if needed
                if "twitter.com" in post_url and "Page not found" in self.browser.driver.title:
                    x_url = post_url.replace("twitter.com", "x.com")
                    logger.logger.debug(f"Trying alternative URL: {x_url}")
                    self.browser.driver.get(x_url)
                    time.sleep(3)
                
                # Wait for page to load - try multiple selectors
                page_loaded = False
                for selector in ['article', '[data-testid="tweet"]', 'div[data-testid="cellInnerDiv"]']:
                    try:
                        WebDriverWait(self.browser.driver, 10).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                        page_loaded = True
                        logger.logger.debug(f"Page loaded, found element with selector: {selector}")
                        break
                    except TimeoutException:
                        continue
                
                if not page_loaded:
                    logger.logger.warning("Could not confirm page loaded with any selector")
                    retry_count += 1
                    time.sleep(self.retry_delay)
                    continue
                
                # Take screenshot for debugging
                try:
                    debug_screenshot = f"reply_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    self.browser.driver.save_screenshot(debug_screenshot)
                    logger.logger.debug(f"Saved reply debugging screenshot to {debug_screenshot}")
                except Exception as ss_error:
                    logger.logger.debug(f"Failed to save debugging screenshot: {str(ss_error)}")
                
                # Find and click the reply button using multiple selectors
                reply_button = None
                for selector in self.reply_button_selectors:
                    try:
                        logger.logger.debug(f"Trying to find reply button with selector: {selector}")
                        reply_button = WebDriverWait(self.browser.driver, 5).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                        if reply_button:
                            logger.logger.debug(f"Found reply button using selector: {selector}")
                            break
                    except Exception as e:
                        logger.logger.debug(f"Could not find reply button with selector '{selector}': {str(e)}")
                        continue
                
                if not reply_button:
                    logger.logger.warning("Could not find reply button with any selector")
                    retry_count += 1
                    time.sleep(self.retry_delay)
                    continue
                
                # Try multiple methods to click the reply button
                click_success = False
                
                # Method 1: JavaScript click
                try:
                    logger.logger.debug("Attempting to click reply button using JavaScript")
                    self.browser.driver.execute_script("arguments[0].click();", reply_button)
                    click_success = True
                    logger.logger.debug("Clicked reply button with JavaScript")
                except Exception as js_error:
                    logger.logger.debug(f"JavaScript click failed: {str(js_error)}")
                
                # Method 2: Standard click if JavaScript failed
                if not click_success:
                    try:
                        logger.logger.debug("Attempting to click reply button using standard click")
                        reply_button.click()
                        click_success = True
                        logger.logger.debug("Clicked reply button with standard click")
                    except Exception as click_error:
                        logger.logger.debug(f"Standard click failed: {str(click_error)}")
                
                # Method 3: Action chains if both methods failed
                if not click_success:
                    try:
                        logger.logger.debug("Attempting to click reply button using ActionChains")
                        ActionChains(self.browser.driver).move_to_element(reply_button).click().perform()
                        click_success = True
                        logger.logger.debug("Clicked reply button with ActionChains")
                    except Exception as action_error:
                        logger.logger.debug(f"ActionChains click failed: {str(action_error)}")
                
                if not click_success:
                    logger.logger.warning("Could not click reply button with any method")
                    retry_count += 1
                    time.sleep(self.retry_delay)
                    continue
                
                # Wait for reply area to appear
                time.sleep(2)
                
                # Find the reply textarea using multiple selectors
                reply_textarea = None
                for selector in self.reply_textarea_selectors:
                    try:
                        logger.logger.debug(f"Trying to find reply textarea with selector: {selector}")
                        reply_textarea = WebDriverWait(self.browser.driver, 5).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                        if reply_textarea:
                            logger.logger.debug(f"Found reply textarea using selector: {selector}")
                            break
                    except Exception as e:
                        logger.logger.debug(f"Could not find textarea with selector '{selector}': {str(e)}")
                        continue
                
                if not reply_textarea:
                    logger.logger.warning("Could not find reply textarea with any selector")
                    retry_count += 1
                    time.sleep(self.retry_delay)
                    continue
                
                # Focus the textarea
                try:
                    reply_textarea.click()
                    time.sleep(1)
                except Exception as click_error:
                    logger.logger.debug(f"Could not click textarea: {str(click_error)}")
                    # Try scrolling to it first
                    try:
                        self.browser.driver.execute_script("arguments[0].scrollIntoView(true);", reply_textarea)
                        time.sleep(1)
                        reply_textarea.click()
                    except Exception as e:
                        logger.logger.debug(f"Could not focus textarea after scroll: {str(e)}")
                        retry_count += 1
                        time.sleep(self.retry_delay)
                        continue
                
                # Clear any existing text
                try:
                    reply_textarea.clear()
                except Exception as clear_error:
                    logger.logger.debug(f"Could not clear textarea: {str(clear_error)}")
                
                # Try multiple methods to enter text
                text_entry_success = False
                
                # Method 1: Direct send_keys
                try:
                    logger.logger.debug("Attempting to enter reply text using send_keys")
                    reply_textarea.send_keys(reply_text)
                    text_entry_success = True
                    logger.logger.debug("Entered text using send_keys")
                except Exception as keys_error:
                    logger.logger.debug(f"send_keys failed: {str(keys_error)}")
                
                # Method 2: JavaScript to set value
                if not text_entry_success:
                    try:
                        logger.logger.debug("Attempting to enter reply text using JavaScript")
                        self.browser.driver.execute_script(
                            "arguments[0].textContent = arguments[1]; arguments[0].dispatchEvent(new Event('input', { bubbles: true }));", 
                            reply_textarea, 
                            reply_text
                        )
                        text_entry_success = True
                        logger.logger.debug("Entered text using JavaScript")
                    except Exception as js_error:
                        logger.logger.debug(f"JavaScript text entry failed: {str(js_error)}")
                
                # Method 3: ActionChains
                if not text_entry_success:
                    try:
                        logger.logger.debug("Attempting to enter reply text using ActionChains")
                        ActionChains(self.browser.driver).move_to_element(reply_textarea).click().send_keys(reply_text).perform()
                        text_entry_success = True
                        logger.logger.debug("Entered text using ActionChains")
                    except Exception as action_error:
                        logger.logger.debug(f"ActionChains text entry failed: {str(action_error)}")
                
                # Method 4: Char by char
                if not text_entry_success:
                    try:
                        logger.logger.debug("Attempting to enter reply text character by character")
                        for char in reply_text:
                            reply_textarea.send_keys(char)
                            time.sleep(0.05)
                        text_entry_success = True
                        logger.logger.debug("Entered text character by character")
                    except Exception as char_error:
                        logger.logger.debug(f"Character by character entry failed: {str(char_error)}")
                
                if not text_entry_success:
                    logger.logger.warning("Could not enter reply text with any method")
                    retry_count += 1
                    time.sleep(self.retry_delay)
                    continue
                
                # Wait for the text to be processed
                time.sleep(2)
                
                # Find the send button using multiple selectors
                send_button = None
                for selector in self.reply_send_button_selectors:
                    try:
                        logger.logger.debug(f"Trying to find send button with selector: {selector}")
                        send_button = WebDriverWait(self.browser.driver, 5).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                        if send_button:
                            logger.logger.debug(f"Found send button using selector: {selector}")
                            break
                    except Exception as e:
                        logger.logger.debug(f"Could not find send button with selector '{selector}': {str(e)}")
                        continue
                
                if not send_button:
                    logger.logger.warning("Could not find send button with any selector")
                    retry_count += 1
                    time.sleep(self.retry_delay)
                    continue
                
                # Try multiple methods to click the send button
                send_success = False
                
                # Method 1: JavaScript click
                try:
                    logger.logger.debug("Attempting to click send button using JavaScript")
                    self.browser.driver.execute_script("arguments[0].click();", send_button)
                    send_success = True
                    logger.logger.debug("Clicked send button with JavaScript")
                except Exception as js_error:
                    logger.logger.debug(f"JavaScript click failed: {str(js_error)}")
                
                # Method 2: Standard click if JavaScript failed
                if not send_success:
                    try:
                        logger.logger.debug("Attempting to click send button using standard click")
                        send_button.click()
                        send_success = True
                        logger.logger.debug("Clicked send button with standard click")
                    except Exception as click_error:
                        logger.logger.debug(f"Standard click failed: {str(click_error)}")
                
                # Method 3: Action chains if both methods failed
                if not send_success:
                    try:
                        logger.logger.debug("Attempting to click send button using ActionChains")
                        ActionChains(self.browser.driver).move_to_element(send_button).click().perform()
                        send_success = True
                        logger.logger.debug("Clicked send button with ActionChains")
                    except Exception as action_error:
                        logger.logger.debug(f"ActionChains click failed: {str(action_error)}")
                
                if not send_success:
                    logger.logger.warning("Could not click send button with any method")
                    retry_count += 1
                    time.sleep(self.retry_delay)
                    continue
                
                # Wait for reply to be sent
                time.sleep(5)
                
                # Verify that the reply was posted successfully
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
                    logger.logger.warning(f"Reply verification failed, may not have been posted")
                    retry_count += 1
                    time.sleep(self.retry_delay)
                    
            except TimeoutException as te:
                logger.logger.warning(f"Timeout while trying to reply to post: {str(te)} (attempt {retry_count + 1})")
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
        Verify that a reply was successfully posted using multiple methods
        
        Returns:
            True if verification succeeded, False otherwise
        """
        try:
            # Method 1: Check if reply compose area is no longer visible
            textarea_gone = False
            for selector in self.reply_textarea_selectors:
                try:
                    WebDriverWait(self.browser.driver, 5).until_not(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    textarea_gone = True
                    logger.logger.debug(f"Verified reply posted - textarea {selector} no longer visible")
                    break
                except TimeoutException:
                    continue
            
            if textarea_gone:
                return True
            
            # Method 2: Check if success indicators are present
            success_indicators = [
                '[data-testid="toast"]',  # Success toast notification
                '[role="alert"]',         # Alert role that might indicate success
                '.css-1dbjc4n[style*="background-color: rgba(0, 0, 0, 0)"]'  # Modal closed
            ]
            
            for indicator in success_indicators:
                try:
                    WebDriverWait(self.browser.driver, 3).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, indicator))
                    )
                    logger.logger.debug(f"Verified reply posted - found success indicator: {indicator}")
                    return True
                except TimeoutException:
                    continue
            
            # Method 3: Check if URL changed
            current_url = self.browser.driver.current_url
            if '/compose/' not in current_url:
                logger.logger.debug("Verified reply posted - no longer on compose URL")
                return True
            
            # Method 4: Check if send button is disabled or gone
            for selector in self.reply_send_button_selectors:
                try:
                    # Check for disabled state
                    send_buttons = self.browser.driver.find_elements(By.CSS_SELECTOR, selector)
                    if not send_buttons:
                        logger.logger.debug(f"Verified reply posted - send button {selector} no longer present")
                        return True
                    
                    # If button exists, check if it's disabled
                    for button in send_buttons:
                        aria_disabled = button.get_attribute('aria-disabled')
                        is_disabled = button.get_attribute('disabled')
                        if aria_disabled == 'true' or is_disabled == 'true':
                            logger.logger.debug(f"Verified reply posted - send button is now disabled")
                            return True
                except Exception:
                    continue
            
            # If we get here, no verification method succeeded
            logger.logger.warning("Could not verify if reply was posted with any method")
            
            # Take screenshot for debugging
            try:
                debug_screenshot = f"reply_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                self.browser.driver.save_screenshot(debug_screenshot)
                logger.logger.debug(f"Saved verification debugging screenshot to {debug_screenshot}")
            except Exception as e:
                logger.logger.debug(f"Failed to save verification screenshot: {str(e)}")
            
            return False
            
        except Exception as e:
            logger.logger.warning(f"Reply verification error: {str(e)}")
            return False
    
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
    
    def _extract_mentioned_tokens(self, post_text: str) -> List[str]:
        """
        Extract cryptocurrency token symbols mentioned in post text
        
        Args:
            post_text: Text content of the post
            
        Returns:
            List of token symbols found in the text
        """
        if not post_text:
            return []
        
        # Common crypto tokens to look for
        tokens = [
            'btc', 'eth', 'sol', 'xrp', 'ada', 'dot', 'doge', 'shib', 'bnb',
            'avax', 'matic', 'link', 'ltc', 'etc', 'bch', 'uni', 'atom'
        ]
        
        # Look for both direct mentions and $-prefixed mentions
        mentioned = []
        post_text_lower = post_text.lower()
        
        # First check for $-prefixed tokens (stronger signal)
        for token in tokens:
            if f'${token}' in post_text_lower:
                mentioned.append(token)
        
        # Then check for word-boundary matches
        for token in tokens:
            # Use regex to find whole-word matches
            if re.search(r'\b' + token + r'\b', post_text_lower):
                if token not in mentioned:  # Avoid duplicates
                    mentioned.append(token)
        
        return mentioned
    
    def _get_reply_sentiment(self, reply_text: str) -> str:
        """
        Determine the sentiment of a reply (bullish, bearish, or neutral)
        
        Args:
            reply_text: Reply text content
            
        Returns:
            Sentiment as string ('bullish', 'bearish', or 'neutral')
        """
        # Bullish words
        bullish_words = [
            'bullish', 'bull', 'buy', 'long', 'moon', 'rally', 'pump', 'uptrend',
            'breakout', 'strong', 'growth', 'profit', 'gain', 'higher', 'up',
            'optimistic', 'momentum', 'support', 'bounce', 'surge', 'uptick'
        ]
        
        # Bearish words
        bearish_words = [
            'bearish', 'bear', 'sell', 'short', 'dump', 'crash', 'correction',
            'downtrend', 'weak', 'decline', 'loss', 'lower', 'down', 'pessimistic',
            'resistance', 'fall', 'drop', 'slump', 'collapse', 'cautious'
        ]
        
        # Count sentiment words
        reply_lower = reply_text.lower()
        
        bullish_count = sum(1 for word in bullish_words if re.search(r'\b' + word + r'\b', reply_lower))
        bearish_count = sum(1 for word in bearish_words if re.search(r'\b' + word + r'\b', reply_lower))
        
        # Check for negation that might flip sentiment
        negation_words = ['not', 'no', 'never', 'doubt', 'unlikely', 'against']
        for neg in negation_words:
            for bull in bullish_words:
                if f"{neg} {bull}" in reply_lower or f"{neg} really {bull}" in reply_lower:
                    bullish_count -= 1
                    bearish_count += 1
            
            for bear in bearish_words:
                if f"{neg} {bear}" in reply_lower or f"{neg} really {bear}" in reply_lower:
                    bearish_count -= 1
                    bullish_count += 1
        
        # Determine overall sentiment
        if bullish_count > bearish_count:
            return 'bullish'
        elif bearish_count > bullish_count:
            return 'bearish'
        else:
            return 'neutral'
    
    def _capture_reply_metadata(self, post: Dict[str, Any], reply_text: str) -> Dict[str, Any]:
        """
        Capture metadata about a reply for analysis and tracking
        
        Args:
            post: Original post data
            reply_text: Generated reply text
            
        Returns:
            Dictionary with metadata about the reply
        """
        # Extract mentioned tokens
        tokens = self._extract_mentioned_tokens(reply_text)
        
        # Determine reply sentiment
        sentiment = self._get_reply_sentiment(reply_text)
        
        # Count relevant metrics
        char_count = len(reply_text)
        word_count = len(reply_text.split())
        
        # Track replied topics
        topics = []
        if 'analysis' in post and 'topics' in post['analysis']:
            topics = list(post['analysis']['topics'].keys())
        
        return {
            'tokens': tokens,
            'sentiment': sentiment,
            'char_count': char_count,
            'word_count': word_count,
            'topics': topics,
            'timestamp': self._strip_timezone(datetime.now())  
        }        
