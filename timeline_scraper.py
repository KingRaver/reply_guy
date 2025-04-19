#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Any, Tuple
import time
import re
import random
import math
from datetime import datetime, timedelta
from datetime_utils import ensure_naive_datetimes, strip_timezone, safe_datetime_diff
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
    WebDriverException,
    ElementClickInterceptedException,
    ElementNotInteractableException
)

from utils.logger import logger

class TimelineScraper:
    """
    Enhanced scraper for X timeline to extract posts for reply targeting
    with improved error handling and resilient content detection
    """
    
    def __init__(self, browser, config, db=None):
        """
        Initialize the timeline scraper with enhanced resilience
        
        Args:
            browser: Browser instance for web interaction
            config: Configuration instance containing settings
            db: Database instance for storing post data (optional)
        """
        self.browser = browser
        self.config = config
        self.db = db
        
        # Timeouts and retry controls
        self.max_retries = 5  # Increased from 3
        self.scroll_pause_time = 2.0  # Increased from 1.5
        self.post_extraction_timeout = 15  # Increased from 10
        self.max_posts_to_scrape = 80  # Increased from 50
        self.max_scroll_attempts = 15  # Added limit to scrolling attempts
        self.scroll_recovery_wait = 3  # Wait time after scroll fails
        self.page_load_timeout = 30  # Maximum time to wait for page load
        
        # Track processed posts to avoid duplicates
        self.already_processed_posts = set()
        
        # Enhanced post selectors with multiple fallbacks for resilience
        # Primary post selectors
        self.timeline_post_selectors = [
            'div[data-testid="cellInnerDiv"]',
            'article[data-testid="tweet"]',
            'div[role="article"]'
        ]
        
        # Content selectors
        self.post_text_selectors = [
            '[data-testid="tweetText"]',
            'div[lang]',  # Language attribute often on text content
            'div.css-901oao'  # Common text class
        ]
        
        self.post_author_selectors = [
            '[data-testid="User-Name"]',
            'div[dir="auto"] > span > span',
            'a[role="link"] > div > div > span'
        ]
        
        self.post_timestamp_selectors = [
            'time',
            'a[href*="/status/"] > time',
            '[data-testid="User-Name"] time'
        ]
        
        self.post_metrics_selectors = [
            '[data-testid="reply"], [data-testid="retweet"], [data-testid="like"]',
            'div[role="group"] > div[role="button"]',
            '[aria-label*="repl"], [aria-label*="Retw"], [aria-label*="Like"]'
        ]
        
        # Navigation and state tracking
        self.last_scroll_height = 0
        self.consecutive_no_new_posts = 0
        self.max_consecutive_no_new = 3  # Abort after 3 consecutive failures
        self.navigation_urls = [
            "https://twitter.com/home",
            "https://x.com/home"
        ]
        
        # Debugging controls
        self.detailed_logging = True  # Enable verbose logging
        self.take_debug_screenshots = True  # Enable screenshots for debugging
        
        logger.logger.info("Timeline scraper initialized with enhanced resilience and multiple detection strategies")

    def _strip_timezone(self, dt):
        return strip_timezone(dt)
        
    def _safe_datetime_diff(self, dt1, dt2):
        return safe_datetime_diff(dt1, dt2)
    
    @ensure_naive_datetimes
    def ensure_naive_datetimes(func):
        """
        Decorator to ensure all datetime objects passed to a function are timezone-naive.
        Place this decorator on methods that compare or operate on datetime objects.
        """
        def wrapper(self, *args, **kwargs):
            # Process args
            new_args = []
            for arg in args:
                if isinstance(arg, datetime):
                    arg = self._strip_timezone(arg)
                new_args.append(arg)
        
            # Process kwargs
            new_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, datetime):
                    value = self._strip_timezone(value)
                new_kwargs[key] = value
        
            # Call the original function with sanitized datetimes
            return func(self, *new_args, **new_kwargs)
        return wrapper

    def navigate_to_home_timeline(self) -> bool:
        """
        Enhanced navigation to home timeline with multiple fallback mechanisms
        
        Returns:
            bool: True if navigation was successful, False otherwise
        """
        logger.logger.info("Attempting to navigate to Twitter/X home timeline")
        
        try:
            # Try multiple navigation URLs with fallbacks
            page_loaded = False
            navigation_errors = []
            
            for attempt, url in enumerate(self.navigation_urls):
                try:
                    logger.logger.info(f"Navigation attempt {attempt+1} using URL: {url}")
                    
                    # Clear cookies every other attempt to help with loading issues
                    if attempt > 0 and attempt % 2 == 0:
                        logger.logger.info("Clearing cookies before navigation retry")
                        try:
                            self.browser.clear_cookies()
                        except:
                            pass
                    
                    # Use existing browser methods for navigation
                    self.browser.driver.get(url)
                    logger.logger.info(f"Page requested, waiting for load: {url}")
                    
                    # Wait for initial page load with longer timeout
                    time.sleep(5)  # Allow for initial load
                    
                    # Check if we're on the correct page
                    current_url = self.browser.driver.current_url
                    logger.logger.info(f"Current URL after navigation: {current_url}")
                    
                    # Take a screenshot for debugging if enabled
                    if self.take_debug_screenshots:
                        screenshot_path = f"timeline_navigation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        try:
                            self.browser.driver.save_screenshot(screenshot_path)
                            logger.logger.info(f"Saved navigation screenshot to {screenshot_path}")
                        except Exception as ss_error:
                            logger.logger.warning(f"Failed to save navigation screenshot: {str(ss_error)}")
                    
                    # Verify we're on Twitter/X
                    if not ("twitter.com" in current_url or "x.com" in current_url):
                        logger.logger.warning(f"Not on Twitter/X: {current_url}")
                        navigation_errors.append(f"URL mismatch: {current_url}")
                        continue
                    
                    # Content verification - try multiple detection methods
                    content_found = self._verify_page_content()
                    
                    if content_found:
                        page_loaded = True
                        logger.logger.info("Twitter/X timeline loaded successfully")
                        break
                    else:
                        logger.logger.warning("Timeline content detection failed")
                        navigation_errors.append("Content detection failed")
                
                except Exception as e:
                    logger.logger.warning(f"Navigation error: {str(e)}")
                    navigation_errors.append(str(e))
                    
                    # Try refreshing the page if already on Twitter/X
                    current_url = self.browser.driver.current_url
                    if "twitter.com" in current_url or "x.com" in current_url:
                        logger.logger.info("Already on Twitter/X, attempting page refresh")
                        try:
                            self.browser.driver.refresh()
                            time.sleep(5)
                            
                            # Check if refresh helped with content loading
                            if self._verify_page_content():
                                page_loaded = True
                                logger.logger.info("Timeline loaded after page refresh")
                                break
                        except:
                            pass
            
            # If we found content, return success
            if page_loaded:
                return True
                
            # If all navigation attempts failed, log detailed error and try emergency detection
            if not page_loaded:
                logger.logger.error(f"All navigation attempts failed: {navigation_errors}")
                
                # Emergency content detection - look for any Twitter-like elements
                emergency_content_found = self._emergency_content_detection()
                if emergency_content_found:
                    logger.logger.info("Emergency content detection succeeded")
                    return True
                
                # If we have an existing timeline URL, try directly
                if hasattr(self.browser.driver, 'current_url'):
                    current_url = self.browser.driver.current_url
                    if "twitter.com" in current_url or "x.com" in current_url:
                        logger.logger.info("Current URL appears to be Twitter/X, proceeding with caution")
                        return True
            
            # Final failure
            logger.logger.error("Failed to navigate to Twitter/X timeline after all attempts")
            return False
            
        except Exception as e:
            logger.log_error("Timeline Navigation", str(e))
            return False
    
    def _verify_page_content(self) -> bool:
        """
        Verify page content using multiple detection methods
        
        Returns:
            bool: True if content detected, False otherwise
        """
        try:
            # Try multiple content detection strategies
            logger.logger.info("Verifying page content with multiple detection methods")
            
            # Method 1: Check for timeline post elements
            for selector in self.timeline_post_selectors:
                try:
                    logger.logger.debug(f"Checking for timeline posts using selector: {selector}")
                    elements = WebDriverWait(self.browser.driver, 10).until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector))
                    )
                    if elements:
                        logger.logger.info(f"Found {len(elements)} timeline posts using selector: {selector}")
                        return True
                except TimeoutException:
                    logger.logger.debug(f"No posts found with selector: {selector}")
                except Exception as e:
                    logger.logger.debug(f"Error checking selector {selector}: {str(e)}")
            
            # Method 2: Check for post text elements
            for selector in self.post_text_selectors:
                try:
                    logger.logger.debug(f"Checking for post text using selector: {selector}")
                    elements = self.browser.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        logger.logger.info(f"Found {len(elements)} post text elements using selector: {selector}")
                        return True
                except Exception as e:
                    logger.logger.debug(f"Error checking text selector {selector}: {str(e)}")
            
            # Method 3: Check for common Twitter UI elements
            common_ui_selectors = [
                '[data-testid="primaryColumn"]',
                '[data-testid="sidebarColumn"]',
                '[data-testid="AppTabBar_Home_Link"]',
                'header[role="banner"]'
            ]
            
            for selector in common_ui_selectors:
                try:
                    elements = self.browser.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        logger.logger.info(f"Found Twitter UI element using selector: {selector}")
                        return True
                except:
                    pass
            
            # Method 4: Look for any Twitter/X specific classes
            twitter_classes = [
                'css-1dbjc4n',
                'r-14lw9ot',
                'r-1tlfku8'
            ]
            
            for cls in twitter_classes:
                try:
                    elements = self.browser.driver.find_elements(By.CSS_SELECTOR, f".{cls}")
                    if len(elements) > 5:  # More than 5 elements with this class
                        logger.logger.info(f"Found multiple Twitter class elements: {cls}")
                        return True
                except:
                    pass
            
            # Method 5: Check page title
            title = self.browser.driver.title.lower()
            if "twitter" in title or "x.com" in title or "home / x" in title or "home / twitter" in title:
                logger.logger.info(f"Twitter/X detected from page title: {title}")
                return True
            
            logger.logger.warning("Failed to verify page content with all detection methods")
            return False
            
        except Exception as e:
            logger.logger.error(f"Error in page content verification: {str(e)}")
            return False

    def _emergency_content_detection(self) -> bool:
        """
        Emergency content detection when standard methods fail
        
        Returns:
            bool: True if any Twitter-like content detected, False otherwise
        """
        try:
            logger.logger.warning("Attempting emergency content detection")
            
            # Method 1: Look for any links to status posts
            try:
                status_links = self.browser.driver.find_elements(By.CSS_SELECTOR, 'a[href*="/status/"]')
                if status_links:
                    logger.logger.info(f"Emergency detection found {len(status_links)} status links")
                    return True
            except:
                pass
            
            # Method 2: Find any substantial text elements
            try:
                # Use XPath to look for any divs with reasonable text content
                text_elements = self.browser.driver.find_elements(
                    By.XPATH, "//div[string-length(text()) > 30]"
                )
                
                if len(text_elements) > 3:
                    logger.logger.info(f"Emergency detection found {len(text_elements)} text elements")
                    return True
            except:
                pass
            
            # Method 3: Check for Twitter-specific SVG icons
            try:
                svg_elements = self.browser.driver.find_elements(By.CSS_SELECTOR, 'svg[viewBox="0 0 24 24"]')
                if len(svg_elements) > 5:
                    logger.logger.info(f"Emergency detection found {len(svg_elements)} Twitter-like SVG icons")
                    return True
            except:
                pass
            
            # Method 4: Look for timeline structure
            try:
                # Try to locate the timeline's main column via various attributes
                main_column_candidates = [
                    self.browser.driver.find_elements(By.CSS_SELECTOR, 'div[aria-label*="Timeline"]'),
                    self.browser.driver.find_elements(By.CSS_SELECTOR, 'div[data-testid="primaryColumn"]'),
                    self.browser.driver.find_elements(By.CSS_SELECTOR, 'section[role="region"]'),
                    self.browser.driver.find_elements(By.CSS_SELECTOR, 'div[data-testid="cellInnerDiv"]')
                ]
                
                for candidates in main_column_candidates:
                    if candidates:
                        logger.logger.info(f"Emergency detection found timeline structure elements")
                        return True
            except:
                pass
            
            # Method 5: Check page source for Twitter/X indicators
            try:
                page_source = self.browser.driver.page_source.lower()
                twitter_indicators = [
                    'twitter.com', 'x.com', 'tweet', 'tweetdeck', 
                    'timeline', 'home timeline', 'retweeted', 'repost',
                    'promoted tweet', 'data-testid'
                ]
                
                found_indicators = [ind for ind in twitter_indicators if ind in page_source]
                if len(found_indicators) >= 3:
                    logger.logger.info(f"Emergency detection found {len(found_indicators)} Twitter indicators in source")
                    return True
            except:
                pass
                
            # Take a screenshot for debugging
            if self.take_debug_screenshots:
                try:
                    screenshot_path = f"emergency_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    self.browser.driver.save_screenshot(screenshot_path)
                    logger.logger.info(f"Saved emergency detection screenshot to {screenshot_path}")
                except:
                    pass
            
            logger.logger.warning("All emergency content detection methods failed")
            return False
            
        except Exception as e:
            logger.logger.error(f"Error in emergency content detection: {str(e)}")
            return False

    def scrape_timeline(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Enhanced timeline scraping with progressive fallbacks and recovery mechanisms
        
        Args:
            count: Number of posts to return
            
        Returns:
            List of post data dictionaries
        """
        logger.logger.info(f"Starting timeline scraping, looking for {count} posts")
        
        # Step 1: Enhanced navigation with validation
        if not self.navigate_to_home_timeline():
            logger.logger.error("Failed to navigate to timeline, cannot scrape posts")
            return []
        
        # Step 2: Initialize scraping variables with enhanced recovery
        retry_count = 0
        posts_data = []
        previously_found_posts = set()  # Track post IDs to detect stalled scrolling
        total_scrolls = 0
        scroll_errors = 0  # Track scroll errors for recovery
        
        # Step 3: Enhanced scraping loop with progressive backoff
        while retry_count < self.max_retries and len(posts_data) < count and total_scrolls < self.max_scroll_attempts:
            try:
                # Log current progress
                logger.logger.debug(f"Scraping attempt {retry_count+1}/{self.max_retries}, found {len(posts_data)}/{count} posts")
                
                # Find posts using multi-strategy approach
                logger.logger.debug("Searching for posts with multi-selector strategy")
                post_elements = self._find_post_elements_multi_strategy()
                
                # Validate post elements
                if not post_elements:
                    logger.logger.warning("No posts found with primary selectors")
                    # Try page refresh on occasional failure
                    if retry_count % 2 == 1:
                        logger.logger.info("Refreshing page to attempt recovery")
                        try:
                            self.browser.driver.refresh()
                            time.sleep(5)
                        except:
                            pass
                    
                    retry_count += 1
                    time.sleep(2)
                    continue
                
                logger.logger.info(f"Found {len(post_elements)} post elements")
                
                # Process posts with advanced error handling and duplicate detection
                new_posts_found = False
                current_post_ids = set()
                post_processing_errors = 0
                
                for i, post_element in enumerate(post_elements):
                    if len(posts_data) >= count:
                        logger.logger.info(f"Reached target count of {count} posts")
                        break
                    
                    try:
                        # Extract unique post ID with resilient method
                        post_id = self._extract_post_id_resilient(post_element)
                        
                        # Skip already processed posts
                        if post_id in self.already_processed_posts:
                            continue
                        
                        # Add to current batch set to track scroll progress
                        current_post_ids.add(post_id)
                        
                        # Add to global processed set
                        self.already_processed_posts.add(post_id)
                        
                        # Extract post data with enhanced error handling
                        post_data = self._extract_post_data_enhanced(post_element)
                        
                        if post_data:
                            posts_data.append(post_data)
                            new_posts_found = True
                            logger.logger.debug(f"Extracted post from {post_data.get('author_handle', 'unknown')}")
                            
                            # Store in database if available
                            if self.db and hasattr(self.db, 'store_post'):
                                try:
                                    self.db.store_post(post_data)
                                except Exception as db_error:
                                    logger.logger.warning(f"Failed to store post in database: {str(db_error)}")
                        else:
                            post_processing_errors += 1
                    
                    except StaleElementReferenceException:
                        logger.logger.debug(f"Stale element reference for post {i+1}")
                        continue
                    
                    except Exception as e:
                        logger.logger.warning(f"Error processing post {i+1}: {str(e)}")
                        post_processing_errors += 1
                        continue
                
                # Check if we found any new posts or if scrolling is stuck
                if not new_posts_found:
                    logger.logger.warning("No new posts found in this scroll cycle")
                    self.consecutive_no_new_posts += 1
                    
                    # Try to unstick scrolling if needed
                    if self.consecutive_no_new_posts >= self.max_consecutive_no_new:
                        logger.logger.warning(f"Scrolling appears stuck after {self.consecutive_no_new_posts} attempts")
                        self._attempt_scroll_recovery()
                        self.consecutive_no_new_posts = 0  # Reset counter after recovery
                    
                    retry_count += 1
                else:
                    self.consecutive_no_new_posts = 0  # Reset counter on success
                
                # Check for too many post processing errors
                if post_processing_errors > len(post_elements) / 2:
                    logger.logger.warning(f"High error rate in post processing: {post_processing_errors}/{len(post_elements)}")
                    retry_count += 1
                
                # Detect if scrolling is making progress by comparing post sets
                current_post_ids_set = set(current_post_ids)
                overlap_with_previous = previously_found_posts.intersection(current_post_ids_set)
                previously_found_posts = current_post_ids_set
                
                # If too much overlap between scrolls, we might be stuck
                if len(post_elements) > 5 and len(overlap_with_previous) > len(post_elements) * 0.8:
                    logger.logger.warning(f"Possible scroll stall: {len(overlap_with_previous)}/{len(post_elements)} posts unchanged")
                    scroll_errors += 1
                    
                    if scroll_errors >= 3:
                        logger.logger.warning("Multiple scroll stalls detected, attempting recovery")
                        self._attempt_scroll_recovery()
                        scroll_errors = 0
                
                # If we need more posts, scroll down
                if len(posts_data) < count:
                    logger.logger.debug(f"Scrolling to find more posts ({len(posts_data)}/{count})")
                    scroll_successful = self._enhanced_scroll_down()
                    
                    if not scroll_successful:
                        logger.logger.warning("Scroll operation failed or no new content loaded")
                        scroll_errors += 1
                    
                    time.sleep(self.scroll_pause_time)
                    total_scrolls += 1
                
            except Exception as e:
                logger.log_error("Timeline Scraping", str(e))
                retry_count += 1
                time.sleep(3)
        
        # Step 4: Final processing and recovery attempts
        if not posts_data and retry_count >= self.max_retries:
            logger.logger.error(f"Failed to scrape any posts after {retry_count} attempts")
            
            # Final emergency attempt with direct DOM inspection
            posts_data = self._emergency_post_extraction()
        
        # Report results
        if posts_data:
            logger.logger.info(f"Successfully scraped {len(posts_data)} posts from timeline")
        else:
            logger.logger.error("Failed to scrape any posts after exhausting all recovery methods")
        
        # Trim to requested count
        result = posts_data[:count]
        return result

    def _find_post_elements_multi_strategy(self) -> List[Any]:
        """
        Find post elements using multiple selectors and strategies
        
        Returns:
            List of WebElements representing posts
        """
        try:
            all_elements = []
            
            # Try each post selector strategy in order
            for selector in self.timeline_post_selectors:
                logger.logger.debug(f"Finding post elements with selector: {selector}")
                try:
                    elements = WebDriverWait(self.browser.driver, 10).until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector))
                    )
                    if elements:
                        logger.logger.info(f"Found {len(elements)} post elements with selector: {selector}")
                        all_elements.extend(elements)
                        break  # Stop if we found enough elements
                except TimeoutException:
                    logger.logger.debug(f"No post elements found with selector: {selector}")
                except Exception as e:
                    logger.logger.debug(f"Error finding post elements with selector {selector}: {str(e)}")
            
            # If primary selectors found nothing, try fallback approaches
            if not all_elements:
                logger.logger.warning("Primary post selectors found no elements, trying fallbacks")
                
                # Fallback 1: Look for tweet text elements and get parents
                for text_selector in self.post_text_selectors:
                    try:
                        text_elements = self.browser.driver.find_elements(By.CSS_SELECTOR, text_selector)
                        logger.logger.debug(f"Found {len(text_elements)} text elements with selector: {text_selector}")
                        
                        if text_elements:
                            parent_elements = []
                            
                            # For each text element, go up a few levels to find post container
                            for text_elem in text_elements:
                                try:
                                    # Try to move up 3-4 levels to find container
                                    parent = text_elem
                                    for _ in range(4):
                                        if parent:
                                            parent = self.browser.driver.execute_script(
                                                "return arguments[0].parentNode;", parent
                                            )
                                    
                                    if parent and parent not in parent_elements:
                                        parent_elements.append(parent)
                                except:
                                    continue
                            
                            if parent_elements:
                                logger.logger.info(f"Fallback found {len(parent_elements)} parent elements")
                                all_elements.extend(parent_elements)
                                break
                    except Exception as e:
                        logger.logger.debug(f"Error in text element fallback with selector {text_selector}: {str(e)}")
                
                # Fallback 2: Use XPath to find potential post containers
                if not all_elements:
                    try:
                        # Look for typical post structure with XPath
                        xpath_candidates = [
                            "//div[contains(@class, 'css-1dbjc4n') and .//a[contains(@href, '/status/')]]",
                            "//article[contains(@data-testid, 'tweet')]/ancestor::div[contains(@class, 'css-1dbjc4n')]",
                            "//div[.//time]/.."  # Posts usually have timestamps
                        ]
                        
                        for xpath in xpath_candidates:
                            elements = self.browser.driver.find_elements(By.XPATH, xpath)
                            if elements:
                                logger.logger.info(f"XPath fallback found {len(elements)} elements")
                                all_elements.extend(elements)
                                break
                    except Exception as e:
                        logger.logger.debug(f"Error in XPath fallback: {str(e)}")
                
                # Fallback 3: Look for any elements with status links
                if not all_elements:
                    try:
                        status_links = self.browser.driver.find_elements(By.CSS_SELECTOR, 'a[href*="/status/"]')
                        if status_links:
                            container_elements = []
                            
                            for link in status_links:
                                try:
                                    # Try to get parent container
                                    container = link
                                    for _ in range(5):  # Go up to 5 levels
                                        if container:
                                            container = self.browser.driver.execute_script(
                                                "return arguments[0].parentNode;", container
                                            )
                                            # Check if we found a large enough container
                                            if container and container.size['height'] > 100:
                                                container_elements.append(container)
                                                break
                                except:
                                    continue
                            
                            if container_elements:
                                logger.logger.info(f"Status link fallback found {len(container_elements)} containers")
                                all_elements.extend(container_elements)
                    except Exception as e:
                        logger.logger.debug(f"Error in status link fallback: {str(e)}")
            
            # Remove duplicates while preserving order
            unique_elements = []
            seen_ids = set()
            
            for elem in all_elements:
                # Create a simple element signature
                try:
                    elem_id = self.browser.driver.execute_script(
                        "return arguments[0].innerHTML.length + '_' + arguments[0].className", elem
                    )
                    if elem_id not in seen_ids:
                        seen_ids.add(elem_id)
                        unique_elements.append(elem)
                except:
                    # If JS fails, just use the element
                    unique_elements.append(elem)
            
            logger.logger.info(f"Found {len(unique_elements)} unique post elements after deduplication")
            return unique_elements
            
        except Exception as e:
            logger.logger.error(f"Error finding post elements: {str(e)}")
            return []
    
    def _extract_post_id_resilient(self, post_element) -> str:
        """
        Extract a unique identifier for a post with multiple fallback methods
        
        Args:
            post_element: The WebElement for the post
            
        Returns:
            A string representing the unique post ID
        """
        try:
            # Strategy 1: Try to get the status ID from any link
            try:
                links = post_element.find_elements(By.CSS_SELECTOR, 'a[href*="/status/"]')
                for link in links:
                    href = link.get_attribute('href')
                    if href:
                        # Extract status ID from the URL
                        match = re.search(r'/status/(\d+)', href)
                        if match:
                            return match.group(1)
            except:
                pass
            
            # Strategy 2: Try attributes that might contain an ID
            attributes = ['data-testid', 'id', 'data-tweet-id', 'data-id', 'data-item-id']
            for attr in attributes:
                try:
                    attr_value = post_element.get_attribute(attr)
                    if attr_value and attr_value.strip():
                        return f"{attr}_{attr_value}"
                except:
                    continue
            
            # Strategy 3: Use element content and timestamp for a composite ID
            composite_parts = []
            
            # Try to get text content
            try:
                for selector in self.post_text_selectors:
                    text_elements = post_element.find_elements(By.CSS_SELECTOR, selector)
                    if text_elements:
                        text = text_elements[0].text
                        if text:
                            # Use a hash of the first few words
                            text_words = ' '.join(text.split()[:5])
                            text_hash = hash(text_words) % 100000
                            composite_parts.append(f"txt{text_hash}")
                            break
            except:
                pass
            
            # Try to get timestamp
            try:
                time_elements = post_element.find_elements(By.TAG_NAME, 'time')
                if time_elements:
                    datetime_attr = time_elements[0].get_attribute('datetime')
                    if datetime_attr:
                        time_hash = hash(datetime_attr) % 100000
                        composite_parts.append(f"time{time_hash}")
            except:
                pass
            
            # Try to get author info
            try:
                for selector in self.post_author_selectors:
                    author_elements = post_element.find_elements(By.CSS_SELECTOR, selector)
                    if author_elements:
                        author_text = author_elements[0].text
                        if author_text:
                            author_hash = hash(author_text) % 100000
                            composite_parts.append(f"auth{author_hash}")
                            break
            except:
                pass
            
            # If we have composite parts, use them
            if composite_parts:
                return "_".join(composite_parts)
            
            # Strategy 4: Last resort - use element attributes and position
            try:
                element_classes = post_element.get_attribute('class') or ''
                element_position = hash(element_classes) % 10000
                timestamp = int(time.time() * 1000)
                return f"elem_{element_position}_{timestamp}"
            except:
                # Ultimate fallback - just use timestamp
                return f"post_{int(time.time() * 1000)}"
            
        except Exception as e:
            logger.logger.debug(f"Error extracting post ID: {str(e)}")
            return f"unknown_{random.randint(10000, 99999)}_{int(time.time())}"
    
    def _enhanced_scroll_down(self) -> bool:
        """
        Enhanced scroll down with height verification and recovery
        
        Returns:
            bool: True if scroll was successful, False otherwise
        """
        try:
            # Get current scroll position before scrolling
            previous_height = self.browser.driver.execute_script("return document.documentElement.scrollHeight")
            
            # Try smooth scrolling first (less likely to be detected as a bot)
            self.browser.driver.execute_script("""
                window.scrollBy({
                    top: 800,
                    left: 0,
                    behavior: 'smooth'
                });
            """)
            
            # Add small random delay to make scrolling more natural
            time.sleep(0.5 + random.random() * 0.5)
            
            # Check if scroll position actually changed
            new_height = self.browser.driver.execute_script("return document.documentElement.scrollHeight")
            scroll_position = self.browser.driver.execute_script("return window.pageYOffset")
            
            # Verify scroll worked correctly
            if new_height > previous_height or scroll_position > self.last_scroll_height:
                logger.logger.debug(f"Scroll successful: Position {scroll_position}, Height change: {new_height - previous_height}")
                self.last_scroll_height = scroll_position
                return True
            else:
                logger.logger.warning("Scroll didn't change position, attempting alternate scroll method")
                
                # Try alternative scrolling method
                self.browser.driver.execute_script("window.scrollBy(0, 1000);")
                time.sleep(1)
                
                # Check if alternate method worked
                alt_position = self.browser.driver.execute_script("return window.pageYOffset")
                if alt_position > self.last_scroll_height:
                    logger.logger.debug(f"Alternative scroll successful, new position: {alt_position}")
                    self.last_scroll_height = alt_position
                    return True
                else:
                    logger.logger.warning("Alternative scroll also failed to change position")
                    return False
                
        except Exception as e:
            logger.logger.warning(f"Scroll error: {str(e)}")
            
            # Try simple scrolling as fallback
            try:
                self.browser.driver.execute_script("window.scrollBy(0, 800);")
                return True
            except:
                return False

    def _attempt_scroll_recovery(self) -> None:
        """
        Attempt various recovery methods when scrolling gets stuck
        """
        from selenium.webdriver.common.keys import Keys  # Add this import
    
        logger.logger.info("Attempting scroll recovery")
    
        # Strategy 1: Try larger scroll amount
        try:
            logger.logger.debug("Recovery: Attempting larger scroll")
            self.browser.driver.execute_script("window.scrollBy(0, 1500);")
            time.sleep(2)
        except:
            pass
    
        # Strategy 2: Try scrolling to bottom then back up
        try:
            logger.logger.debug("Recovery: Scrolling to bottom then back up")
            self.browser.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            self.browser.driver.execute_script("window.scrollTo(0, Math.max(document.body.scrollHeight - 2000, 0));")
            time.sleep(2)
        except:
            pass
    
        # Strategy 3: Try refreshing the page
        try:
            logger.logger.debug("Recovery: Refreshing page")
            self.browser.driver.refresh()
            time.sleep(5)
        
            # Scroll down a bit after refresh to get away from the top
            self.browser.driver.execute_script("window.scrollBy(0, 500);")
            time.sleep(1)
        except:
            pass
    
        # Strategy 4: Try pressing ESC to dismiss any popups (fixed)
        try:
            logger.logger.debug("Recovery: Pressing ESC to dismiss popups")
            actions = ActionChains(self.browser.driver)
            actions.send_keys(Keys.ESCAPE).perform()
            time.sleep(1)
        except:
            pass
    
        # Strategy 5: Try clicking in empty space to dismiss any context menus
        try:
            logger.logger.debug("Recovery: Clicking in empty space")
            empty_areas = self.browser.driver.find_elements(By.TAG_NAME, 'body')
            if empty_areas:
                actions = ActionChains(self.browser.driver)
                actions.move_to_element_with_offset(empty_areas[0], 10, 10).click().perform()
                time.sleep(1)
        except:
            pass
    
        # Take a recovery screenshot for debugging
        if self.take_debug_screenshots:
            try:
                recovery_screenshot = f"recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                self.browser.driver.save_screenshot(recovery_screenshot)
                logger.logger.info(f"Saved scroll recovery screenshot to {recovery_screenshot}")
            except:
                pass
    
        # Reset scroll height tracking
        self.last_scroll_height = 0
        logger.logger.info("Scroll recovery procedures completed")
    
    def _extract_post_data_enhanced(self, post_element) -> Optional[Dict[str, Any]]:
        """
        Enhanced version of post data extraction with better error handling
        
        Args:
            post_element: Selenium WebElement for the post
            
        Returns:
            Dictionary containing post data or None if extraction failed
        """
        logger.logger.debug("Extracting post data with enhanced method")
        
        try:
            # Ensure post element is viewable - scroll it into view if needed
            try:
                self.browser.driver.execute_script(
                    "arguments[0].scrollIntoView({block: 'center', behavior: 'smooth'});", 
                    post_element
                )
                time.sleep(0.5)  # Brief pause for any animations
            except Exception as e:
                logger.logger.debug(f"Scroll into view failed: {str(e)}")
            
            # Extract post ID with resilient method
            post_id = self._extract_post_id_resilient(post_element)
            
            # Extract post URL with multiple fallbacks
            post_url = self._extract_post_url_enhanced(post_element)
            
            # Extract author information with multiple fallbacks
            author_info = self._extract_author_info_enhanced(post_element)
            author_name, author_handle, author_link = author_info
            
            # Extract post content with better text handling
            post_text = self._extract_post_text_enhanced(post_element)
            
            # Extract timestamp with multiple formats support
            timestamp, parsed_time = self._extract_timestamp_enhanced(post_element)
            
            # Extract engagement metrics with robust parsing
            metrics = self._extract_engagement_metrics_enhanced(post_element)
            
            # Calculate engagement score using a weighted algorithm
            engagement_score = self._calculate_engagement_score(metrics)
            
            # Check for media content (images, videos, links)
            has_media = self._check_for_media_enhanced(post_element)
            
            # Analyze tweet content for topics, hashtags, etc.
            content_analysis = self._analyze_tweet_content_enhanced(post_text)
            
            # Assemble all post data
            post_data = {
                'post_id': post_id,
                'post_url': post_url,
                'author_name': author_name,
                'author_handle': author_handle,
                'author_profile_url': author_link,
                'text': post_text,
                'timestamp_text': timestamp,
                'timestamp': parsed_time,
                'metrics': metrics,
                'engagement_score': engagement_score,
                'has_media': has_media,
                'content_analysis': content_analysis,
                'scraped_at': datetime.now()
            }
            
            # Log success with preview of extracted content
            preview = post_text[:50] + "..." if len(post_text) > 50 else post_text
            logger.logger.debug(f"Successfully extracted post from {author_handle}: '{preview}'")
            
            return post_data
            
        except Exception as e:
            logger.log_error("Post Data Extraction", str(e))
            return None
    
    def _extract_post_url_enhanced(self, post_element) -> Optional[str]:
        """
        Extract post URL with multiple fallback methods
        
        Args:
            post_element: WebElement for the post
            
        Returns:
            URL string or None if extraction failed
        """
        try:
            # Strategy 1: Look for status links directly
            status_links = post_element.find_elements(By.CSS_SELECTOR, 'a[href*="/status/"]')
            for link in status_links:
                href = link.get_attribute('href')
                
                # Check if this is a direct link to a tweet status (exclude media links)
                if href and '/status/' in href and '/photo/' not in href and '/video/' not in href:
                    return href
            
            # Strategy 2: Try to find timestamp links (these often link to the tweet)
            try:
                time_elements = post_element.find_elements(By.TAG_NAME, 'time')
                for time_elem in time_elements:
                    # Look for parent link
                    parent_links = self.browser.driver.execute_script(
                        "return arguments[0].closest('a[href*=\"/status/\"]');", time_elem
                    )
                    if parent_links:
                        href = parent_links.get_attribute('href')
                        if href and '/status/' in href:
                            return href
            except:
                pass
            
            # Strategy 3: Try to construct URL from author and status ID
            try:
                # Get author handle
                author_handle = self._extract_author_info_enhanced(post_element)[1]
                if author_handle.startswith('@'):
                    author_handle = author_handle[1:]  # Remove @ prefix
                
                # Extract status ID from any link
                for link in post_element.find_elements(By.TAG_NAME, 'a'):
                    href = link.get_attribute('href') or ''
                    match = re.search(r'/status/(\d+)', href)
                    if match:
                        post_id = match.group(1)
                        # Construct URL if we have both handle and ID
                        if author_handle and post_id:
                            return f"https://twitter.com/{author_handle}/status/{post_id}"
            except:
                pass
            
            # If all else fails, log the issue
            logger.logger.debug("Could not extract post URL with any method")
            return None
            
        except Exception as e:
            logger.logger.debug(f"Error extracting post URL: {str(e)}")
            return None

    def _extract_author_info_enhanced(self, post_element) -> Tuple[str, str, str]:
        """
        Extract author information with multiple fallback methods
        
        Args:
            post_element: WebElement for the post
            
        Returns:
            Tuple of (author_name, author_handle, profile_link)
        """
        try:
            author_name = "Unknown"
            author_handle = "@unknown"
            author_link = ""
            
            # Strategy 1: Try standard approach with multiple selectors
            for selector in self.post_author_selectors:
                try:
                    author_elements = post_element.find_elements(By.CSS_SELECTOR, selector)
                    if author_elements:
                        # Try to get author name and handle from text content
                        full_text = author_elements[0].text
                        if full_text:
                            parts = full_text.split('\n')
                            if len(parts) >= 2:
                                author_name = parts[0].strip()
                                author_handle = parts[1].strip()
                                if not author_handle.startswith('@'):
                                    author_handle = f"@{author_handle}"
                            elif len(parts) == 1:
                                author_name = parts[0].strip()
                        
                        # Try to get link to profile
                        link_elements = author_elements[0].find_elements(By.TAG_NAME, 'a')
                        for link in link_elements:
                            href = link.get_attribute('href')
                            if href and '/status/' not in href and '/media/' not in href:
                                author_link = href
                                # Ensure URL is absolute
                                if author_link.startswith('/'):
                                    author_link = f"https://twitter.com{author_link}"
                                break
                        
                        # If we found both name and handle, we're done
                        if author_name != "Unknown" and author_handle != "@unknown":
                            break
                except:
                    continue
            
            # Strategy 2: Try advanced approach if standard failed
            if author_name == "Unknown" or author_handle == "@unknown":
                try:
                    # Look for any links that might be profile links
                    link_elements = post_element.find_elements(By.TAG_NAME, 'a')
                    for link in link_elements:
                        href = link.get_attribute('href') or ''
                        # Skip status and media links
                        if '/status/' in href or '/photo/' in href or '/video/' in href:
                            continue
                            
                        # Check if this could be a profile link
                        username_match = re.search(r'/([\w]+)$', href)
                        if username_match:
                            potential_username = username_match.group(1)
                            # Check if it's not a Twitter feature page
                            if potential_username not in ['home', 'explore', 'notifications', 'messages', 'settings']:
                                link_text = link.text
                                if link_text:
                                    # Likely a profile link
                                    author_name = link_text
                                    author_handle = f"@{potential_username}"
                                    author_link = href
                                    break
                except:
                    pass
            
            # Strategy 3: Parse any available text if still unknown
            if author_name == "Unknown" or author_handle == "@unknown":
                try:
                    # Get all visible text and look for patterns
                    full_text = post_element.text
                    lines = full_text.split('\n')
                    
                    # Try to find name and handle pattern (@something)
                    for i, line in enumerate(lines):
                        if i < len(lines) - 1 and re.search(r'@\w+', lines[i+1]):
                            author_name = line
                            author_handle = lines[i+1]
                            break
                except:
                    pass
            
            return author_name, author_handle, author_link
            
        except Exception as e:
            logger.logger.debug(f"Error extracting author info: {str(e)}")
            return "Unknown", "@unknown", ""
    
    def _extract_post_text_enhanced(self, post_element) -> str:
        """
        Extract post text content with better handling of formatting
        
        Args:
            post_element: WebElement for the post
            
        Returns:
            String containing post text
        """
        try:
            post_text = ""
            
            # Try each text selector
            for selector in self.post_text_selectors:
                try:
                    text_elements = post_element.find_elements(By.CSS_SELECTOR, selector)
                    if text_elements:
                        text_element = text_elements[0]
                        post_text = text_element.text
                        
                        # If we found text, break
                        if post_text:
                            break
                except:
                    continue
            
            # If primary selectors failed, try alternative approach
            if not post_text:
                try:
                    # Look for any div with substantial text
                    potential_texts = post_element.find_elements(
                        By.XPATH, ".//div[string-length(text()) > 15]"
                    )
                    
                    # Get the element with the most text
                    if potential_texts:
                        post_text = max(potential_texts, key=lambda e: len(e.text)).text
                except:
                    pass
            
            # If we found text, clean it up
            if post_text:
                # Replace Unicode ellipsis
                post_text = post_text.replace('\u2026', '...')
                
                # Replace multiple spaces
                post_text = re.sub(r'\s+', ' ', post_text)
                
                # Normalize newlines
                post_text = post_text.replace('\n', ' ').replace('\r', ' ')
                
                # Trim
                post_text = post_text.strip()
            
            return post_text
            
        except Exception as e:
            logger.logger.debug(f"Error extracting post text: {str(e)}")
            return ""
    
    def _extract_timestamp_enhanced(self, post_element) -> Tuple[str, Optional[datetime]]:
        """
        Extract timestamp with improved parsing and multiple formats
        
        Args:
            post_element: WebElement for the post
            
        Returns:
            Tuple of (timestamp_text, parsed_datetime)
        """
        try:
            # Strategy 1: Find time element and get datetime attribute
            for time_selector in self.post_timestamp_selectors:
                try:
                    time_elements = post_element.find_elements(By.CSS_SELECTOR, time_selector)
                    if time_elements:
                        time_element = time_elements[0]
                        
                        # Get the raw timestamp text
                        timestamp_text = time_element.text
                        
                        # Try to parse the datetime attribute (most reliable)
                        datetime_attr = time_element.get_attribute('datetime')
                        
                        if datetime_attr:
                            # Handle different ISO formats
                            if 'Z' in datetime_attr:
                                parsed_time = datetime.fromisoformat(datetime_attr.replace('Z', '+00:00'))
                            else:
                                parsed_time = datetime.fromisoformat(datetime_attr)
                                
                            # Convert to local time
                            parsed_time = parsed_time.astimezone(None)
                            return timestamp_text, parsed_time
                except:
                    continue
            
            # Strategy 2: Parse the timestamp text itself as fallback
            try:
                # Look for any element that might contain timestamp text
                potential_timestamps = post_element.find_elements(
                    By.XPATH, ".//span[contains(text(), 'h') or contains(text(), 'm') or contains(text(), 's') or contains(text(), 'd')]"
                )
                
                for elem in potential_timestamps:
                    timestamp_text = elem.text.strip()
                    
                    # Check if this looks like a relative timestamp
                    if re.match(r'^(\d+[smhdw]|now|just now)$', timestamp_text.lower()):
                        parsed_time = self._parse_relative_time(timestamp_text)
                        return timestamp_text, parsed_time
            except:
                pass
            
            # If all else fails, use current time
            return "", datetime.now()
            
        except Exception as e:
            logger.logger.debug(f"Error extracting timestamp: {str(e)}")
            return "", datetime.now()
    
    def _parse_relative_time(self, time_text: str) -> datetime:
        """
        Parse relative time expressions with enhanced recognition patterns
        
        Args:
            time_text: The time text to parse (e.g., "5m", "2h", "3d")
            
        Returns:
            datetime object representing the estimated absolute time
        """
        now = datetime.now()
        time_text = time_text.lower().strip()
        
        # Handle "now" or "just now"
        if time_text in ['now', 'just now']:
            return now
        
        # Handle pattern like "5m", "2h", etc.
        match = re.match(r'(\d+)([smhdw])', time_text)
        if match:
            value = int(match.group(1))
            unit = match.group(2)
            
            if unit == 's':  # Seconds
                return now - timedelta(seconds=value)
            elif unit == 'm':  # Minutes
                return now - timedelta(minutes=value)
            elif unit == 'h':  # Hours
                return now - timedelta(hours=value)
            elif unit == 'd':  # Days
                return now - timedelta(days=value)
            elif unit == 'w':  # Weeks
                return now - timedelta(weeks=value)
        
        # Handle full date formats like "Mar 15" or "2023-01-15"
        try:
            # Try common date formats
            formats = ['%b %d', '%Y-%m-%d', '%d %b', '%b %d, %Y', '%B %d, %Y', '%d %B %Y', '%d/%m/%Y', '%m/%d/%Y']
            for fmt in formats:
                try:
                    dt = datetime.strptime(time_text, fmt)
                    # If year is not in the format, use current year
                    if dt.year == 1900:
                        dt = dt.replace(year=now.year)
                        # If this date is in the future, it's probably from last year
                        if dt > now:
                            dt = dt.replace(year=now.year - 1)
                    return dt
                except ValueError:
                    continue
        except:
            pass
        
        # If all parsing fails, return current time
        return now
    
    def _extract_engagement_metrics_enhanced(self, post_element) -> Dict[str, int]:
        """
        Extract engagement metrics with improved parsing and multiple strategies
        
        Args:
            post_element: WebElement for the post
            
        Returns:
            Dictionary with engagement metrics
        """
        metrics = {
            'replies': 0,
            'reposts': 0,
            'likes': 0,
            'views': 0
        }
        
        try:
            # Strategy 1: Look for metrics by attribute
            try:
                for metric_test_id in ['reply', 'retweet', 'like', 'view']:
                    selector = f'[data-testid="{metric_test_id}"]'
                    metric_elements = post_element.find_elements(By.CSS_SELECTOR, selector)
                    
                    for element in metric_elements:
                        aria_label = element.get_attribute('aria-label') or element.text
                        
                        # Extract the number using regex
                        number_match = re.search(r'(\d+(?:,\d+)*)', aria_label)
                        if number_match:
                            count_str = number_match.group(1)
                            count = int(count_str.replace(',', ''))
                            
                            # Map to the correct metric
                            if 'repl' in metric_test_id.lower() or 'comment' in aria_label.lower():
                                metrics['replies'] = count
                            elif 'retweet' in metric_test_id.lower() or 'repost' in aria_label.lower():
                                metrics['reposts'] = count
                            elif 'like' in metric_test_id.lower() or 'favorite' in aria_label.lower():
                                metrics['likes'] = count
                            elif 'view' in metric_test_id.lower() or 'seen' in aria_label.lower():
                                metrics['views'] = count
            except:
                pass
            
            # Strategy 2: Try to find metric elements by role and text pattern
            if sum(metrics.values()) == 0:
                try:
                    # Get all potential metric containers
                    metric_containers = post_element.find_elements(By.CSS_SELECTOR, 'div[role="group"] > div[role="button"]')
                    
                    for container in metric_containers:
                        container_text = container.text
                        container_aria = container.get_attribute('aria-label') or ''
                        
                        # First extract the number from either source
                        number_match = re.search(r'(\d+(?:,\d+)*)', container_text) or re.search(r'(\d+(?:,\d+)*)', container_aria)
                        if number_match:
                            count_str = number_match.group(1)
                            count = int(count_str.replace(',', ''))
                            
                            # Determine metric type from text or aria-label
                            combined_text = (container_text + ' ' + container_aria).lower()
                            
                            if 'repl' in combined_text or 'comment' in combined_text:
                                metrics['replies'] = count
                            elif 'retweet' in combined_text or 'repost' in combined_text:
                                metrics['reposts'] = count
                            elif 'like' in combined_text or 'heart' in combined_text or 'favorite' in combined_text:
                                metrics['likes'] = count
                            elif 'view' in combined_text or 'seen' in combined_text or 'impression' in combined_text:
                                metrics['views'] = count
                except:
                    pass
            
            # Strategy 3: Look for SVG icons with adjacent text
            if sum(metrics.values()) == 0:
                try:
                    svg_elements = post_element.find_elements(By.TAG_NAME, 'svg')
                    
                    for svg in svg_elements:
                        # Check if there's a number next to this icon
                        parent = self.browser.driver.execute_script("return arguments[0].parentNode;", svg)
                        if parent:
                            parent_text = parent.text.strip()
                            number_match = re.match(r'^(\d+(?:,\d+)*[KkMm]?)$', parent_text)
                            
                            if number_match:
                                count_str = number_match.group(1)
                                count = self._parse_metric_value(count_str)
                                
                                # Determine metric type from icon
                                svg_path = svg.get_attribute('innerHTML') or ''
                                
                                if 'comment' in svg_path or 'reply' in svg_path:
                                    metrics['replies'] = count
                                elif 'retweet' in svg_path or 'repost' in svg_path:
                                    metrics['reposts'] = count
                                elif 'heart' in svg_path or 'like' in svg_path:
                                    metrics['likes'] = count
                                elif 'view' in svg_path or 'impression' in svg_path:
                                    metrics['views'] = count
                except:
                    pass
            
            return metrics
            
        except Exception as e:
            logger.logger.debug(f"Error extracting engagement metrics: {str(e)}")
            return metrics
    
    def _parse_metric_value(self, value_text: str) -> int:
        """
        Parse metric values like "5.2K" or "1.3M" to actual numbers
        
        Args:
            value_text: The value text to parse
            
        Returns:
            Integer value
        """
        try:
            value_text = value_text.strip().upper()
            
            # Handle values with K, M, B suffixes
            if value_text.endswith('K'):
                return int(float(value_text[:-1]) * 1000)
            elif value_text.endswith('M'):
                return int(float(value_text[:-1]) * 1000000)
            elif value_text.endswith('B'):
                return int(float(value_text[:-1]) * 1000000000)
            else:
                return int(float(value_text))
        except:
            return 0
    
    def _calculate_engagement_score(self, metrics: Dict[str, int]) -> float:
        """
        Calculate a weighted engagement score for better post prioritization
        
        Args:
            metrics: Dictionary of engagement metrics
            
        Returns:
            Float score representing engagement level
        """
        # Weights: replies have highest value, then reposts, then likes
        reply_weight = 3.0
        repost_weight = 2.0
        like_weight = 1.0
        view_weight = 0.1  # Views are less valuable as engagement indicators
        
        score = (
            metrics.get('replies', 0) * reply_weight +
            metrics.get('reposts', 0) * repost_weight +
            metrics.get('likes', 0) * like_weight +
            metrics.get('views', 0) * view_weight
        )
        
        # Apply a log scale for very high values to prevent outliers dominating
        if score > 1000:
            # Log scaling for high values
            score = 1000 * (1 + math.log10(score / 1000))
        
        return score
    
    def _check_for_media_enhanced(self, post_element) -> Dict[str, bool]:
        """
        Check for different types of media in the post with better detection
        
        Args:
            post_element: WebElement for the post
            
        Returns:
            Dictionary with indicators for different media types
        """
        media = {
            'has_image': False,
            'has_video': False,
            'has_link': False,
            'has_card': False,
            'has_any_media': False
        }
        
        try:
            # Check for images
            try:
                images = post_element.find_elements(By.CSS_SELECTOR, 'img[src*="pbs.twimg.com"]')
                if images:
                    media['has_image'] = True
                    media['has_any_media'] = True
            except:
                pass
                
            # Check for videos
            try:
                videos = post_element.find_elements(By.CSS_SELECTOR, 'video')
                video_divs = post_element.find_elements(By.CSS_SELECTOR, 'div[data-testid="videoPlayer"]')
                if videos or video_divs:
                    media['has_video'] = True
                    media['has_any_media'] = True
                    
                # Also check for video-related attributes
                if 'video' in post_element.get_attribute('innerHTML').lower():
                    media['has_video'] = True
                    media['has_any_media'] = True
            except:
                pass
                
            # Check for card links
            try:
                cards = post_element.find_elements(By.CSS_SELECTOR, '[data-testid="card.wrapper"]')
                if cards:
                    media['has_card'] = True
                    media['has_any_media'] = True
            except:
                pass
                
            # Check for embedded article links
            try:
                article_links = post_element.find_elements(By.CSS_SELECTOR, 'a[role="link"][href*="http"]')
                non_twitter_links = [link for link in article_links if 
                                    'twitter.com' not in link.get_attribute('href') and 
                                    'x.com' not in link.get_attribute('href')]
                if non_twitter_links:
                    media['has_link'] = True
                    media['has_any_media'] = True
            except:
                pass
                
            return media
            
        except Exception as e:
            logger.logger.debug(f"Error checking for media: {str(e)}")
            return {'has_any_media': False}
    
    def _analyze_tweet_content_enhanced(self, text: str) -> Dict[str, Any]:
        """
        Analyze tweet content for topics, hashtags, and entities
        
        Args:
            text: The tweet text to analyze
            
        Returns:
            Dictionary of analysis results
        """
        if not text:
            return {
                'hashtags': [],
                'mentions': [],
                'topics': [],
                'has_question': False,
                'sentiment': 'neutral'
            }
            
        try:
            # Extract hashtags with better pattern matching
            hashtags = re.findall(r'#(\w+)', text)
            
            # Extract mentions
            mentions = re.findall(r'@(\w+)', text)
            
            # Check if tweet contains a question
            has_question = '?' in text
            
            # Basic topic detection
            topics = []
            
            # Financial terms
            financial_terms = ['market', 'stock', 'invest', 'chart', 'trade', 'buy', 'sell', 'profit', 'loss', 'price']
            if any(term in text.lower() for term in financial_terms):
                topics.append('finance')
                
            # Crypto terms
            crypto_terms = ['crypto', 'bitcoin', 'btc', 'eth', 'blockchain', 'token', 'coin', 'wallet', 'defi']
            if any(term in text.lower() for term in crypto_terms):
                topics.append('crypto')
                
            # Tech terms
            tech_terms = ['tech', 'software', 'code', 'programming', 'ai', 'data', 'app', 'web3']
            if any(term in text.lower() for term in tech_terms):
                topics.append('tech')
                
            # Simple sentiment analysis
            sentiment = 'neutral'
            
            positive_terms = ['bull', 'up', 'gain', 'profit', 'moon', 'pump', 'win', 'good', 'great', 'excellent']
            negative_terms = ['bear', 'down', 'loss', 'crash', 'dump', 'bad', 'terrible', 'worry', 'fear']
            
            positive_count = sum(1 for term in positive_terms if term in text.lower())
            negative_count = sum(1 for term in negative_terms if term in text.lower())
            
            if positive_count > negative_count:
                sentiment = 'positive'
            elif negative_count > positive_count:
                sentiment = 'negative'
            
            # Create enhanced result
            return {
                'hashtags': hashtags,
                'mentions': mentions,
                'topics': topics,
                'has_question': has_question,
                'sentiment': sentiment,
                'word_count': len(text.split()),
                'char_count': len(text)
            }
            
        except Exception as e:
            logger.logger.debug(f"Error analyzing tweet content: {str(e)}")
            return {
                'hashtags': [],
                'mentions': [],
                'topics': [],
                'has_question': False,
                'sentiment': 'neutral'
            }
    
    def _emergency_post_extraction(self) -> List[Dict[str, Any]]:
        """
        Last resort method to try to find any posts with raw DOM manipulation
        
        Returns:
            List of post data extracted by alternative means
        """
        logger.logger.warning("Attempting emergency post extraction")
        results = []
        
        try:
            # Try to find any elements that might contain tweet text
            potential_text_elements = self.browser.driver.find_elements(By.XPATH, 
                "//div[contains(@class, 'r-') and string-length(text()) > 20]")
            
            logger.logger.info(f"Found {len(potential_text_elements)} potential text elements")
            
            for i, elem in enumerate(potential_text_elements[:20]):  # Limit to first 20
                try:
                    text = elem.text
                    if len(text) > 20:  # Only consider substantial text
                        # Create minimal post data
                        post_data = {
                            'post_id': f"emergency_{i}_{int(time.time())}",
                            'text': text,
                            'timestamp': datetime.now(),
                            'scraped_at': datetime.now(),
                            'emergency_extraction': True
                        }
                        results.append(post_data)
                        logger.logger.debug(f"Emergency extracted text: {text[:50]}...")
                except:
                    continue
                    
            logger.logger.info(f"Emergency extraction found {len(results)} potential posts")
            return results
            
        except Exception as e:
            logger.logger.error(f"Emergency extraction failed: {str(e)}")
            return []

    def find_market_related_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter posts to only include those related to markets with enhanced detection
        
        Args:
            posts: List of post data dictionaries
            
        Returns:
            Filtered list of market-related posts
        """
        if not posts:
            logger.logger.warning("No posts provided to filter for market relevance")
            return []
            
        logger.logger.info(f"Filtering {len(posts)} posts for market relevance")
        
        # Enhanced set of market-related keywords with categorization
        market_keywords = {
            # Crypto specific
            'crypto': [
                'bitcoin', 'btc', 'eth', 'ethereum', 'sol', 'solana', 'bnb', 'binance', 'xrp', 'ripple',
                'avax', 'avalanche', 'doge', 'dogecoin', 'shib', 'ada', 'cardano', 'dot', 'polkadot',
                'blockchain', 'defi', 'nft', 'crypto', 'token', 'altcoin', 'coin', 'web3', 'wallet',
                'exchange', 'hodl', 'mining', 'staking', 'smart contract', 'airdrop'
            ],
            
            # General finance
            'finance': [
                'market', 'stock', 'trade', 'trading', 'price', 'chart', 'analysis', 'technical',
                'fundamental', 'invest', 'investment', 'fund', 'capital', 'bull', 'bear', 'rally',
                'crash', 'correction', 'resistance', 'support', 'volume', 'liquidity', 'volatility',
                'trend', 'breakout', 'sell', 'buy', 'long', 'short', 'profit', 'loss', 'roi', 'yield'
            ],
            
            # Finance symbols
            'symbols': ['$', '€', '£', '¥']
        }
        
        # Flatten the keywords for efficient matching
        all_keywords = []
        for category in market_keywords.values():
            all_keywords.extend(category)
        
        # Store matched posts with their relevance score
        market_related = []
        
        for post in posts:
            post_text = post.get('text', '').lower()
            
            # Skip posts with no text
            if not post_text:
                continue
                
            # Track matched keywords for logging
            matched_keywords = []
            keyword_categories = set()
            
            # Basic keyword matching with categorization
            for category, keywords in market_keywords.items():
                for keyword in keywords:
                    if keyword in post_text:
                        matched_keywords.append(keyword)
                        keyword_categories.add(category)
            
            # Price pattern matching (e.g., "$45.2K" or "45,000 USDT")
            price_patterns = [
                r'\$\d+[,.]?\d*[KMB]?',  # $45K, $45.2K
                r'\d+[,.]?\d*\s*[$€£¥]',  # 45.2 $, 45,000 €
                r'\d+[,.]?\d*\s*usdt',    # 45000 USDT
                r'\d+[,.]?\d*\s*usd',     # 45000 USD
                r'\d+[,.]\d*\s*eth',      # 0.05 ETH
                r'\d+[,.]\d*\s*btc'       # 0.01 BTC
            ]
            
            for pattern in price_patterns:
                if re.search(pattern, post_text, re.IGNORECASE):
                    matched_keywords.append("price_pattern")
                    keyword_categories.add('finance')
                    break
            
            # Percentage change patterns
            percentage_patterns = [
                r'[+-]?\d+(\.\d+)?%',         # +5%, -10.5%
                r'up\s+\d+(\.\d+)?%',         # up 5%
                r'down\s+\d+(\.\d+)?%',       # down 10.5%
                r'gained\s+\d+(\.\d+)?%',     # gained 5%
                r'lost\s+\d+(\.\d+)?%',       # lost 10.5%
                r'increased\s+\d+(\.\d+)?%',  # increased 5%
                r'decreased\s+\d+(\.\d+)?%',  # decreased 10.5%
            ]
            
            for pattern in percentage_patterns:
                if re.search(pattern, post_text, re.IGNORECASE):
                    matched_keywords.append("percentage_change")
                    keyword_categories.add('finance')
                    break
                    
            # Also check hashtags if available
            if 'content_analysis' in post and 'hashtags' in post['content_analysis']:
                hashtags = post['content_analysis']['hashtags']
                for tag in hashtags:
                    tag = tag.lower()
                    if any(keyword in tag for keyword in all_keywords):
                        matched_keywords.append(f"hashtag:{tag}")
                        # Determine category of the hashtag
                        for cat, keywords in market_keywords.items():
                            if any(keyword in tag for keyword in keywords):
                                keyword_categories.add(cat)
            
            # Calculate relevance score based on matches
            relevance_score = 0
            
            # Base score from number of matches
            relevance_score += min(10, len(matched_keywords))
            
            # Bonus for matching multiple categories
            relevance_score += len(keyword_categories) * 3
            
            # Bonus for having both crypto and finance terms (high relevance)
            if 'crypto' in keyword_categories and 'finance' in keyword_categories:
                relevance_score += 5
            
            # Mark as market related if we found any matches and score is high enough
            if matched_keywords and relevance_score >= 3:
                # Add relevant fields to the post
                post['market_related'] = True
                post['market_keywords'] = matched_keywords
                post['market_categories'] = list(keyword_categories)
                post['market_relevance_score'] = relevance_score
                market_related.append(post)
                
        logger.logger.info(f"Found {len(market_related)} market-related posts out of {len(posts)}")
        
        # Log some example keywords for debugging
        if market_related:
            examples = [p.get('market_keywords', [])[:3] for p in market_related[:3]]
            logger.logger.debug(f"Example keywords matched: {examples}")
        
        # Sort by relevance score (highest first)
        market_related.sort(key=lambda x: x.get('market_relevance_score', 0), reverse=True)
        
        return market_related
    
    def filter_already_replied_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out posts we've already replied to 
        
        Args:
            posts: List of post data dictionaries
            
        Returns:
            Filtered list of posts
        """
        if not posts:
            logger.logger.warning("No posts provided to filter for replies")
            return []
            
        logger.logger.info(f"Filtering {len(posts)} posts for ones we haven't replied to yet")
        
        if not self.db:
            logger.logger.warning("No database connected, cannot filter already replied posts")
            return posts
            
        filtered_posts = []
        
        for post in posts:
            post_id = post.get('post_id')
            post_url = post.get('post_url')
            
            # Skip posts with no ID or URL
            if not post_id and not post_url:
                continue
            
            # Check if we've already replied to this post
            already_replied = False
            
            try:
                already_replied = self.db.check_if_post_replied(post_id, post_url)
            except Exception as e:
                logger.logger.warning(f"Error checking if post was replied to: {str(e)}")
                # Fall back to checking locally
                already_replied = False
            
            if not already_replied:
                filtered_posts.append(post)
        
        logger.logger.info(f"Filtered out {len(posts) - len(filtered_posts)} already replied posts")
        return filtered_posts
    
    @ensure_naive_datetimes
    def prioritize_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort posts by engagement level, relevance, recency with improved ranking
        
        Args:
            posts: List of post data dictionaries
            
        Returns:
            Sorted list of posts by priority score
        """
        if not posts:
            return []
            
        logger.logger.info(f"Prioritizing {len(posts)} posts by engagement, relevance, and recency")
        
        # Calculate a priority score for each post
        scored_posts = []
        
        for post in posts:
            # Base score starts at 0
            score = 0
            
            # Factor 1: Engagement score (0-100 points)
            engagement = min(100, post.get('engagement_score', 0))
            score += engagement * 0.5  # Up to 50 points
            
            # Factor 2: Recency (0-30 points)
            timestamp = post.get('timestamp')
            if timestamp:
                # Calculate hours since posted
                hours_ago = safe_datetime_diff(datetime.now(), timestamp) / 3600
                
                # More recent posts get higher scores (inverse relationship)
                recency_score = max(0, 30 - (hours_ago * 2.5))  # Lose 2.5 points per hour, max 30
                score += recency_score
            
            # Factor 3: Market relevance (0-30 points)
            # Use the market_relevance_score if available
            relevance = min(30, post.get('market_relevance_score', 0) * 2)
            score += relevance
            
            # Factor 4: Has question (bonus 10 points)
            # Questions are good opportunities for helpful replies
            if post.get('content_analysis', {}).get('has_question', False):
                score += 10
                
            # Factor 5: Media content (bonus 5 points)
            # Posts with media often get more engagement
            if post.get('has_media', {}).get('has_any_media', False):
                score += 5
                
            # Factor 6: Sentiment alignment (bonus 5 points)
            # If the post has strong sentiment, it's more interesting to reply to
            sentiment = post.get('content_analysis', {}).get('sentiment', 'neutral')
            if sentiment != 'neutral':
                score += 5
                
            # Store score with post
            post['priority_score'] = score
            scored_posts.append(post)
        
        # Sort by priority score (highest first)
        sorted_posts = sorted(scored_posts, key=lambda x: x.get('priority_score', 0), reverse=True)
        
        # Log top priority posts
        if sorted_posts:
            top_posts = sorted_posts[:3]
            logger.logger.info("Top priority posts:")
            for i, post in enumerate(top_posts):
                preview = post.get('text', '')[:50]
                if len(post.get('text', '')) > 50:
                    preview += "..."
                logger.logger.info(f"  {i+1}. Score: {post.get('priority_score', 0):.1f} - {preview}")
        
        return sorted_posts                    
