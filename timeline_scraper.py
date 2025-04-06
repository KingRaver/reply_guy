#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Any, Tuple
import time
import re
import random
import math
from datetime import datetime, timedelta
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
    WebDriverException
)

from utils.logger import logger

class TimelineScraper:
    """
    Scraper for X timeline to extract posts for reply targeting
    """
    
    def __init__(self, browser, config, db=None):
        """
        Initialize the timeline scraper
        
        Args:
            browser: Browser instance for web interaction
            config: Configuration instance containing settings
            db: Database instance for storing post data (optional)
        """
        self.browser = browser
        self.config = config
        self.db = db
        self.max_retries = 3
        self.scroll_pause_time = 1.5
        self.post_extraction_timeout = 10
        self.max_posts_to_scrape = 50  # We'll scrape more and filter later
        self.already_processed_posts = set()
        
        # Post selectors - use multiple options for better coverage
        self.timeline_post_selector = 'div[data-testid="cellInnerDiv"]'
        self.tweet_selector = 'article[data-testid="tweet"]'
        self.article_selector = 'div[role="article"]'
        
        # Content selectors
        self.post_text_selector = '[data-testid="tweetText"]'
        self.post_author_selector = '[data-testid="User-Name"]'
        self.post_timestamp_selector = 'time'
        self.post_metrics_selector = '[data-testid="reply"], [data-testid="retweet"], [data-testid="like"]'
        
        logger.logger.info("Timeline scraper initialized with multiple detection strategies")
    
    def navigate_to_home_timeline(self) -> bool:
        """
        Navigate to home timeline with enhanced validation
        
        Returns:
            bool: True if navigation was successful, False otherwise
        """
        logger.logger.info("Attempting to navigate to Twitter home timeline")
        
        try:
            # Go to the home page
            self.browser.driver.get("https://twitter.com/home")
            time.sleep(3)  # Allow for initial page load
            
            # Check if we're on the correct page
            current_url = self.browser.driver.current_url
            logger.logger.info(f"Current URL after navigation: {current_url}")
            
            if "twitter.com/home" not in current_url and "x.com/home" not in current_url:
                logger.logger.warning(f"Not on home timeline, current URL: {current_url}")
                # If on a login page, try navigating again after a short wait
                if "login" in current_url or "i/flow" in current_url:
                    logger.logger.info("On login page, waiting to see if redirection occurs")
                    time.sleep(5)
                    current_url = self.browser.driver.current_url
                    if "twitter.com/home" not in current_url and "x.com/home" not in current_url:
                        logger.logger.error("Still not on home timeline after wait, login may be required")
                        return False
            
            # Try to find timeline posts with multiple attempts
            for attempt in range(3):
                try:
                    # Try different selectors to find any timeline content
                    selectors_to_try = [
                        self.timeline_post_selector,
                        self.tweet_selector,
                        self.article_selector,
                        self.post_text_selector  # Even finding any tweet text is a good sign
                    ]
                    
                    for selector in selectors_to_try:
                        try:
                            logger.logger.info(f"Waiting for timeline content with selector: {selector}")
                            WebDriverWait(self.browser.driver, 10).until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                            )
                            logger.logger.info(f"Timeline content found on attempt {attempt+1} with selector: {selector}")
                            return True
                        except TimeoutException:
                            continue
                    
                    # If we get here, none of the selectors worked
                    logger.logger.warning(f"Timeout on attempt {attempt+1}, trying refresh")
                    self.browser.driver.refresh()
                    time.sleep(5)  # Longer wait after refresh
                    
                except Exception as e:
                    logger.logger.warning(f"Error on navigation attempt {attempt+1}: {str(e)}")
                    time.sleep(2)
            
            # Final check for any content
            try:
                page_source = self.browser.driver.page_source
                if "timeline" in page_source.lower() or "tweet" in page_source.lower():
                    logger.logger.info("Timeline terms found in page source, proceeding with caution")
                    return True
                else:
                    # Take a screenshot for debugging
                    screenshot_path = f"timeline_screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    self.browser.driver.save_screenshot(screenshot_path)
                    logger.logger.warning(f"Timeline terms not found in page source. Saved screenshot to {screenshot_path}")
                    return False
            except Exception as e:
                logger.logger.error(f"Error on final content check: {str(e)}")
                return False
                
        except Exception as e:
            logger.log_error("Timeline Navigation", str(e))
            return False
        
    def scrape_timeline(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Scrape X timeline and return post data with enhanced error handling
        
        Args:
            count: Number of posts to return
            
        Returns:
            List of post data dictionaries
        """
        logger.logger.info(f"Starting timeline scraping, looking for {count} posts")

        # Step 1: Navigate to timeline with validation
        if not self.navigate_to_home_timeline():
            logger.logger.error("Failed to navigate to timeline, cannot scrape posts")
            return []
            
        logger.logger.info(f"Timeline navigation successful, current URL: {self.browser.driver.current_url}")
        
        # Step 2: Initialize variables for scraping loop
        retry_count = 0
        posts_data = []
        total_scrolls = 0
        max_scrolls = 10  # Limit scrolling for performance
        
        # Step 3: Main scraping loop
        while retry_count < self.max_retries and len(posts_data) < count and total_scrolls < max_scrolls:
            try:
                # Find posts using multiple strategies
                post_elements = self._find_post_elements()
                
                # Check if we found any posts
                if not post_elements or len(post_elements) == 0:
                    # Log debugging information and try again
                    self._log_debug_info("No posts found with primary selectors")
                    retry_count += 1
                    time.sleep(2)
                    continue
                
                logger.logger.info(f"Found {len(post_elements)} post elements on timeline")
                
                # Process each post
                new_posts_found = False
                logger.logger.debug(f"Starting to process {len(post_elements)} post elements")
                
                for i, post_element in enumerate(post_elements):
                    # Check if we've reached our target count
                    if len(posts_data) >= count:
                        break

                    logger.logger.debug(f"Processing post element {i + 1}/{len(post_elements)}")
                    
                    try:
                        # Extract unique post ID
                        post_id = self._extract_post_id(post_element)
                        
                        # Skip already processed posts
                        if post_id in self.already_processed_posts:
                            continue
                            
                        # Add to processed set
                        self.already_processed_posts.add(post_id)
                        
                        # Extract post data
                        post_data = self.extract_post_data(post_element)
                        
                        if post_data:
                            posts_data.append(post_data)
                            new_posts_found = True
                            
                            # Log first 50 chars of post text for debugging
                            text_preview = post_data.get('text', '')[:50]
                            if len(post_data.get('text', '')) > 50:
                                text_preview += "..."
                                
                            logger.logger.debug(f"Extracted post from {post_data.get('author_handle', 'unknown')}: '{text_preview}'")
                            
                            # Store in database if available and configured
                            if self.db and hasattr(self.db, 'store_post'):
                                self.db.store_post(post_data)
                        else:
                            logger.logger.debug(f"Failed to extract post data from element {i + 1}")
                    
                    except StaleElementReferenceException:
                        # Element reference is no longer valid, skip this post
                        logger.logger.debug(f"Stale element reference for post {i + 1}, skipping")
                        continue
                        
                    except Exception as e:
                        logger.logger.warning(f"Failed to process post {i + 1}: {str(e)}")
                        continue

                # Break if we've found enough posts
                if len(posts_data) >= count:
                    logger.logger.info(f"Found {len(posts_data)} posts, target reached")
                    break
                    
                # If no new posts were found, scroll down
                if not new_posts_found or len(posts_data) < count:
                    logger.logger.debug("Scrolling to find more posts")
                    self._scroll_down()
                    time.sleep(self.scroll_pause_time)
                    total_scrolls += 1
                    
                # Increment retry count if we're still not finding enough posts
                if not new_posts_found:
                    retry_count += 1
                    logger.logger.warning(f"No new posts found in this cycle, retry {retry_count}/{self.max_retries}")
                    
            except Exception as e:
                logger.log_error("Timeline Scraping", str(e))
                retry_count += 1
                time.sleep(2)
        
        # Step 4: Final processing and return
        if posts_data:
            logger.logger.info(f"Successfully scraped {len(posts_data)} posts from timeline")
        else:
            # Log failure and try one last method
            logger.logger.error("Failed to scrape any posts after multiple attempts")
            posts_data = self._emergency_post_extraction()
            
        # Trim to requested count
        result = posts_data[:count]
        logger.logger.info(f"Returning {len(result)} posts for processing")
        return result
    
    def _find_post_elements(self) -> List[Any]:
        """
        Find post elements using multiple selectors
        
        Returns:
            List of WebElements representing posts
        """
        try:
            # Try primary selector first
            post_elements = self.browser.driver.find_elements(By.CSS_SELECTOR, self.timeline_post_selector)
            
            if not post_elements or len(post_elements) == 0:
                logger.logger.debug("No posts found with primary selector, trying tweet selector")
                post_elements = self.browser.driver.find_elements(By.CSS_SELECTOR, self.tweet_selector)
                
            if not post_elements or len(post_elements) == 0:
                logger.logger.debug("No posts found with tweet selector, trying article selector")
                post_elements = self.browser.driver.find_elements(By.CSS_SELECTOR, self.article_selector)
                
            if not post_elements or len(post_elements) == 0:
                # Last resort - try to find any tweet text elements and get their parents
                logger.logger.debug("No posts found with article selector, trying to find tweet text")
                text_elements = self.browser.driver.find_elements(By.CSS_SELECTOR, self.post_text_selector)
                
                if text_elements:
                    # Try to get parent elements that might be posts
                    post_elements = []
                    for text_elem in text_elements:
                        try:
                            # Go up a few levels to find potential post container
                            parent = text_elem
                            for _ in range(3):  # Try going up to 3 levels
                                if parent:
                                    parent = self.browser.driver.execute_script("return arguments[0].parentNode;", parent)
                            
                            if parent:
                                post_elements.append(parent)
                        except:
                            continue
            
            return post_elements
            
        except Exception as e:
            logger.logger.error(f"Error finding post elements: {str(e)}")
            return []
    
    def _log_debug_info(self, message: str) -> None:
        """
        Log detailed debug information for troubleshooting
        
        Args:
            message: Context message describing the issue
        """
        try:
            logger.logger.warning(f"{message} - Gathering debug information")
            
            # Log current URL
            try:
                current_url = self.browser.driver.current_url
                logger.logger.info(f"Current URL: {current_url}")
            except:
                logger.logger.warning("Could not get current URL")
            
            # Log page source preview
            try:
                page_source = self.browser.driver.page_source[:2000]  # First 2000 chars
                logger.logger.info(f"Page source preview: {page_source[:500]}...")
            except:
                logger.logger.warning("Could not get page source")
            
            # Take a screenshot for debugging
            try:
                screenshot_path = f"timeline_screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                self.browser.driver.save_screenshot(screenshot_path)
                logger.logger.info(f"Saved screenshot to {screenshot_path}")
            except Exception as se:
                logger.logger.warning(f"Failed to capture screenshot: {str(se)}")
                
            # Try to identify any visible elements that might help debug
            try:
                visible_elements = self.browser.driver.find_elements(By.CSS_SELECTOR, "h1, h2, div.r-1kihuf0, div.r-18u37iz")
                visible_text = [elem.text for elem in visible_elements if elem.text.strip()]
                logger.logger.info(f"Visible text elements: {visible_text[:10]}")
            except:
                logger.logger.warning("Could not identify visible elements")
                
        except Exception as e:
            logger.logger.error(f"Error logging debug info: {str(e)}")
    
    def _emergency_post_extraction(self) -> List[Dict[str, Any]]:
        """
        Last resort method to try to find any posts
        
        Returns:
            List of post data extracted by alternative means
        """
        logger.logger.info("Attempting emergency post extraction")
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

    def extract_post_data(self, post_element) -> Optional[Dict[str, Any]]:
        """
        Extract relevant data from a post element with enhanced reliability
        
        Args:
            post_element: Selenium WebElement for the post
            
        Returns:
            Dictionary containing post data or None if extraction failed
        """
        logger.logger.debug(f"Extracting post data from element")
        
        try:
            # Step 1: Ensure the post is in the viewport
            try:
                self.browser.driver.execute_script(
                    "arguments[0].scrollIntoView({block: 'center', behavior: 'smooth'});", 
                    post_element
                )
                time.sleep(0.5)  # Brief pause for any animations
            except Exception as e:
                logger.logger.debug(f"Scroll into view failed: {str(e)}")
            
            # Step 2: Extract basic post identifiers
            post_id = self._extract_post_id(post_element)
            post_url = self._extract_post_url(post_element)
            
            # Step 3: Extract author information
            author_info = self._extract_author_info(post_element)
            author_name, author_handle, author_link = author_info
            
            # Step 4: Extract post content
            post_text = ""
            try:
                # Try multiple strategies to get tweet text
                text_element = None
                
                # First try the standard selector
                try:
                    text_element = post_element.find_element(By.CSS_SELECTOR, self.post_text_selector)
                except NoSuchElementException:
                    # Try alternative approaches
                    try:
                        # Try to find any div with substantial text
                        potential_texts = post_element.find_elements(
                            By.XPATH, ".//div[string-length(text()) > 15]"
                        )
                        
                        # Get the element with the most text
                        if potential_texts:
                            text_element = max(potential_texts, key=lambda e: len(e.text))
                    except:
                        pass
                
                if text_element:
                    post_text = text_element.text
                    
                    # Clean up common issues in tweet text
                    post_text = post_text.replace('\u2026', '...')  # Fix ellipsis
                    post_text = re.sub(r'\s+', ' ', post_text)  # Fix multiple spaces
                    post_text = post_text.strip()
            except Exception as e:
                logger.logger.debug(f"Text extraction error: {str(e)}")
                post_text = ""
                
            # Step 5: Extract timestamp and engagement metrics
            timestamp, parsed_time = self._extract_timestamp(post_element)
            metrics = self._extract_engagement_metrics(post_element)
            engagement_score = self._calculate_engagement_score(metrics)
            
            # Step 6: Check for media content
            has_media = self._check_for_media(post_element)
            
            # Step 7: Analyze tweet content
            topics = self._analyze_tweet_content(post_text)
            
            # Step 8: Assemble post data
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
                'scraped_at': datetime.now(),
                'topics': topics
            }
            
            # Log success with preview of extracted content
            preview = post_text[:50] + "..." if len(post_text) > 50 else post_text
            logger.logger.debug(f"Successfully extracted post data for {author_handle}: '{preview}'")
            
            return post_data
            
        except Exception as e:
            logger.log_error("Post Data Extraction", str(e))
            return None
    
    def _extract_post_id(self, post_element) -> str:
        """
        Extract unique identifier for post with improved reliability
        
        Args:
            post_element: The WebElement for the post
            
        Returns:
            A string representing the unique post ID
        """
        try:
            # Strategy 1: Try to get the data-testid or article id
            post_id = post_element.get_attribute('data-testid')
            
            # Strategy 2: If no testid, try getting a link to the post
            if not post_id:
                try:
                    link_elements = post_element.find_elements(By.CSS_SELECTOR, 'a[href*="/status/"]')
                    for link_element in link_elements:
                        href = link_element.get_attribute('href')
                        if href:
                            # Extract status ID from the URL
                            match = re.search(r'/status/(\d+)', href)
                            if match:
                                post_id = match.group(1)
                                break
                except:
                    pass
            
            # Strategy 3: If still no ID, generate a pseudo-ID from post content
            if not post_id:
                try:
                    # Try to get text content
                    text_elements = post_element.find_elements(By.CSS_SELECTOR, self.post_text_selector)
                    text = text_elements[0].text[:50] if text_elements else ""  # First 50 chars
                    
                    # Try to get timestamp
                    time_elements = post_element.find_elements(By.CSS_SELECTOR, self.post_timestamp_selector)
                    time_str = time_elements[0].get_attribute('datetime') if time_elements else ""
                    
                    # Combine for a pseudo-ID
                    if text or time_str:
                        text_hash = hash(text) % 10000 if text else random.randint(1000, 9999)
                        time_hash = hash(time_str) % 10000 if time_str else int(time.time() % 10000)
                        post_id = f"post_{text_hash}_{time_hash}"
                    else:
                        # Last resort - use element attributes
                        class_attr = post_element.get_attribute('class') or ""
                        attrs_hash = hash(class_attr) % 10000
                        post_id = f"post_{attrs_hash}_{int(time.time() * 1000)}"
                except:
                    # Final fallback - just use timestamp
                    post_id = f"post_{int(time.time() * 1000)}"
            
            return post_id
            
        except Exception as e:
            logger.logger.debug(f"Failed to extract post ID: {str(e)}")
            return f"unknown_post_{int(time.time() * 1000)}"
    
    def _extract_post_url(self, post_element) -> Optional[str]:
        """
        Extract the URL to the post with improved detection
        
        Args:
            post_element: The WebElement for the post
            
        Returns:
            String URL to the post or None if extraction failed
        """
        try:
            # Strategy 1: Look for status links - try multiple approaches
            link_selectors = [
                'a[href*="/status/"]',  # Standard status links
                'a[role="link"][href*="/"]',  # Any link that might lead to a tweet
                'div[role="link"][tabindex="0"]'  # Clickable div that might be a link
            ]
            
            for selector in link_selectors:
                link_elements = post_element.find_elements(By.CSS_SELECTOR, selector)
                
                for link in link_elements:
                    href = link.get_attribute('href')
                    
                    # Check if this is a direct link to a tweet status
                    if href and '/status/' in href and '/photo/' not in href and '/video/' not in href:
                        return href
            
            # Strategy 2: Try to construct URL from author and status ID
            try:
                # Get author handle
                author_elements = post_element.find_elements(By.CSS_SELECTOR, self.post_author_selector)
                if author_elements:
                    handle_elements = author_elements[0].find_elements(By.CSS_SELECTOR, 'span')
                    author_handle = None
                    
                    for span in handle_elements:
                        text = span.text
                        if text.startswith('@'):
                            author_handle = text[1:]  # Remove @ prefix
                            break
                    
                    # Get post ID from any status link
                    post_id = None
                    all_links = post_element.find_elements(By.TAG_NAME, 'a')
                    for link in all_links:
                        href = link.get_attribute('href')
                        if href and '/status/' in href:
                            match = re.search(r'/status/(\d+)', href)
                            if match:
                                post_id = match.group(1)
                                break
                    
                    # Construct URL if we have both handle and ID
                    if author_handle and post_id:
                        return f"https://twitter.com/{author_handle}/status/{post_id}"
            except:
                pass
            
            return None
            
        except Exception as e:
            logger.logger.debug(f"Failed to extract post URL: {str(e)}")
            return None
    
    def _extract_author_info(self, post_element) -> Tuple[str, str, str]:
        """
        Extract author name, handle and profile link with enhanced reliability
        
        Args:
            post_element: The WebElement for the post
            
        Returns:
            Tuple of (author_name, author_handle, profile_link)
        """
        try:
            # Strategy 1: Try standard approach first
            try:
                author_elements = post_element.find_elements(By.CSS_SELECTOR, self.post_author_selector)
                
                if author_elements:
                    author_element = author_elements[0]
                    
                    # Extract author name
                    try:
                        name_elements = author_element.find_elements(By.CSS_SELECTOR, 'span')
                        author_name = name_elements[0].text if name_elements else "Unknown"
                    except:
                        author_name = "Unknown"
                        
                    # Extract author handle
                    author_handle = "@unknown"
                    try:
                        # Try to find the handle specifically - often the second span
                        handle_elements = author_element.find_elements(By.CSS_SELECTOR, 'span')
                        for span in handle_elements:
                            text = span.text
                            if text and (text.startswith('@') or text.strip() != author_name.strip()):
                                author_handle = text if text.startswith('@') else f"@{text}"
                                break
                    except:
                        pass
                    
                    # Extract author profile link
                    author_link = ""
                    try:
                        link_elements = author_element.find_elements(By.TAG_NAME, 'a')
                        for link in link_elements:
                            href = link.get_attribute('href')
                            if href and '/i/web/status/' not in href and '/photo/' not in href:
                                author_link = href
                                # Make sure it's a full URL
                                if author_link.startswith('/'):
                                    author_link = f"https://twitter.com{author_link}"
                                break
                    except:
                        pass
                        
                    return author_name, author_handle, author_link
            except:
                pass
            
            # Strategy 2: Try alternative approach for author info
            try:
                # Look for any link with brief text that might be a username
                link_elements = post_element.find_elements(By.TAG_NAME, 'a')
                
                for link in link_elements:
                    href = link.get_attribute('href')
                    text = link.text
                    
                    # If this looks like a profile link and has brief text
                    if href and '/status/' not in href and text and len(text) < 20:
                        # This might be a profile link
                        author_name = text
                        author_handle = f"@{text.split()[-1]}"
                        
                        # Make sure link is absolute
                        if href.startswith('/'):
                            author_link = f"https://twitter.com{href}"
                        else:
                            author_link = href
                            
                        return author_name, author_handle, author_link
            except:
                pass
            
            # Fallback: return default values
            return "Unknown", "@unknown", ""
            
        except Exception as e:
            logger.logger.debug(f"Failed to extract author info: {str(e)}")
            return "Unknown", "@unknown", ""

    def _extract_timestamp(self, post_element) -> Tuple[str, Optional[datetime]]:
        """
        Extract timestamp text and parsed datetime with enhanced reliability
        
        Args:
            post_element: The WebElement for the post
            
        Returns:
            Tuple of (timestamp_text, parsed_datetime)
        """
        try:
            # Strategy 1: Try to find time element directly
            time_elements = post_element.find_elements(By.CSS_SELECTOR, self.post_timestamp_selector)
            
            if time_elements:
                time_element = time_elements[0]
                
                # Get the raw timestamp text
                timestamp_text = time_element.text
                
                # Try to parse the datetime attribute
                datetime_attr = time_element.get_attribute('datetime')
                
                if datetime_attr:
                    # Parse ISO format datetime
                    try:
                        # Handle different ISO formats
                        if 'Z' in datetime_attr:
                            parsed_time = datetime.fromisoformat(datetime_attr.replace('Z', '+00:00'))
                        else:
                            parsed_time = datetime.fromisoformat(datetime_attr)
                            
                        # Convert to local time
                        parsed_time = parsed_time.astimezone(None)
                        return timestamp_text, parsed_time
                    except ValueError:
                        # If ISO parsing fails, try different approach
                        logger.logger.debug(f"ISO datetime parsing failed for: {datetime_attr}")
                
                # Fallback: try to interpret the text (e.g., "2h", "5m", etc.)
                if timestamp_text:
                    parsed_time = self._parse_relative_time(timestamp_text)
                    return timestamp_text, parsed_time
            
            # Strategy 2: Try to find a span with time-like text
            try:
                # Look for spans with short text that might be timestamps
                span_elements = post_element.find_elements(By.TAG_NAME, 'span')
                
                for span in span_elements:
                    text = span.text.strip()
                    
                    # Check if this looks like a relative timestamp (e.g., "2h", "5m")
                    if (len(text) <= 5 and 
                        (any(c in text for c in ['s', 'm', 'h', 'd', 'w']) or 
                         text.lower() in ['now', 'just now'])):
                        
                        parsed_time = self._parse_relative_time(text)
                        return text, parsed_time
            except:
                pass
                
            # Fallback: use current time
            return "", datetime.now()
            
        except Exception as e:
            logger.logger.debug(f"Failed to extract timestamp: {str(e)}")
            return "", datetime.now()
    
    def _parse_relative_time(self, time_text: str) -> datetime:
        """
        Parse relative time expressions with enhanced pattern matching
        
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
        
        # Try to extract number and unit
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
        
        # Handle more complex formats like "Mar 15" or "2023-01-15"
        try:
            # Try to parse common date formats
            for fmt in ['%b %d', '%Y-%m-%d', '%d %b', '%b %d, %Y', '%B %d, %Y']:
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
    
    def _extract_engagement_metrics(self, post_element) -> Dict[str, int]:
        """
        Extract engagement metrics (replies, reposts, likes) with improved detection
        
        Args:
            post_element: The WebElement for the post
            
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
            # Strategy 1: Try data-testid approach
            try:
                # Find all metric elements
                metric_elements = post_element.find_elements(By.CSS_SELECTOR, self.post_metrics_selector)
                
                for element in metric_elements:
                    try:
                        test_id = element.get_attribute('data-testid') or ""
                        aria_label = element.get_attribute('aria-label') or ""
                        
                        if not aria_label:
                            # Try getting text content instead
                            aria_label = element.text
                        
                        # Extract the number from the aria-label
                        number_match = re.search(r'(\d+(?:,\d+)*)', aria_label)
                        count_str = number_match.group(1) if number_match else "0"
                        
                        # Remove commas and convert to int
                        count = int(count_str.replace(',', ''))
                        
                        # Determine metric type
                        if 'reply' in test_id or 'repl' in aria_label.lower():
                            metrics['replies'] = count
                        elif 'retweet' in test_id or 'post' in aria_label.lower():
                            metrics['reposts'] = count
                        elif 'like' in test_id or 'like' in aria_label.lower():
                            metrics['likes'] = count
                        elif 'view' in test_id or 'view' in aria_label.lower():
                            metrics['views'] = count
                    except:
                        continue
            except:
                pass
                
            # Strategy 2: Try looking for specific icons with numbers
            if sum(metrics.values()) == 0:
                try:
                    # Get all spans that might contain numbers
                    span_elements = post_element.find_elements(By.TAG_NAME, 'span')
                    
                    for span in span_elements:
                        text = span.text.strip()
                        
                        # Check if span contains a number
                        if re.match(r'^\d+(?:\.\d+)?[KMB]?$', text):
                            # Convert K, M, B to actual numbers
                            value = self._parse_metric_value(text)
                            
                            try:
                                # Check icon/svg near this span
                                parent = self.browser.driver.execute_script(
                                    "return arguments[0].parentNode;", span
                                )
                                
                                parent_html = parent.get_attribute('innerHTML').lower()
                                
                                if 'reply' in parent_html or 'comment' in parent_html:
                                    metrics['replies'] = value
                                elif 'retweet' in parent_html or 'repost' in parent_html:
                                    metrics['reposts'] = value
                                elif 'like' in parent_html or 'heart' in parent_html:
                                    metrics['likes'] = value
                                elif 'view' in parent_html or 'seen' in parent_html:
                                    metrics['views'] = value
                            except:
                                continue
                except:
                    pass
                    
            return metrics
            
        except Exception as e:
            logger.logger.debug(f"Failed to extract engagement metrics: {str(e)}")
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
        Calculate a weighted engagement score
        
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
    
    def _check_for_media(self, post_element) -> bool:
        """
        Check if post contains media (images, videos, links)
        
        Args:
            post_element: The WebElement for the post
            
        Returns:
            Boolean indicating if media was detected
        """
        try:
            # Check for images
            images = post_element.find_elements(By.CSS_SELECTOR, 'img[src*="pbs.twimg.com"]')
            if images:
                return True
                
            # Check for videos
            videos = post_element.find_elements(By.CSS_SELECTOR, 'video')
            if videos:
                return True
                
            # Check for video divs
            video_divs = post_element.find_elements(By.CSS_SELECTOR, 'div[data-testid="videoPlayer"]')
            if video_divs:
                return True
                
            # Check for card links
            cards = post_element.find_elements(By.CSS_SELECTOR, '[data-testid="card.wrapper"]')
            if cards:
                return True
                
            # Check for embedded article links
            article_links = post_element.find_elements(By.CSS_SELECTOR, 'a[role="link"][href*="http"]')
            if article_links:
                return True
                
            return False
            
        except Exception:
            return False

    def _analyze_tweet_content(self, text: str) -> Dict[str, Any]:
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
            # Extract hashtags
            hashtags = re.findall(r'#(\w+)', text)
            
            # Extract mentions
            mentions = re.findall(r'@(\w+)', text)
            
            # Check if tweet contains a question
            has_question = '?' in text
            
            # Do basic topic detection
            topics = []
            
            # Financial terms
            financial_terms = ['market', 'stock', 'invest', 'chart', 'trade', 'buy', 'sell', 'profit', 'loss']
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
                
            # Do simple sentiment analysis
            sentiment = 'neutral'
            
            positive_terms = ['bull', 'up', 'gain', 'profit', 'moon', 'pump', 'win', 'good', 'great', 'excellent']
            negative_terms = ['bear', 'down', 'loss', 'crash', 'dump', 'bad', 'terrible', 'worry', 'fear']
            
            positive_count = sum(1 for term in positive_terms if term in text.lower())
            negative_count = sum(1 for term in negative_terms if term in text.lower())
            
            if positive_count > negative_count:
                sentiment = 'positive'
            elif negative_count > positive_count:
                sentiment = 'negative'
            
            return {
                'hashtags': hashtags,
                'mentions': mentions,
                'topics': topics,
                'has_question': has_question,
                'sentiment': sentiment
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
    
    def _scroll_down(self) -> None:
        """
        Scroll down to load more posts with improved smoothness
        """
        try:
            # Smooth scrolling to avoid detection
            self.browser.driver.execute_script("""
                window.scrollBy({
                    top: 800,
                    left: 0,
                    behavior: 'smooth'
                });
            """)
            
            # Small random delay to make scrolling more natural
            time.sleep(0.5 + random.random() * 0.5)
            
        except Exception as e:
            logger.logger.warning(f"Failed to scroll timeline: {str(e)}")
            
            # Fallback to simple scrolling
            try:
                self.browser.driver.execute_script("window.scrollBy(0, 800);")
            except:
                pass
    
    def get_market_related_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        
        # Enhanced set of market-related keywords
        market_keywords = [
            # Crypto specific
            "bitcoin", "btc", "eth", "ethereum", "sol", "solana", "bnb", "binance", "xrp", "ripple",
            "avax", "avalanche", "doge", "dogecoin", "shib", "ada", "cardano", "dot", "polkadot",
            "blockchain", "defi", "nft", "crypto", "token", "altcoin", "coin", "web3", "wallet",
            "exchange", "hodl", "mining", "staking", "smart contract", "airdrop",
            
            # General finance
            "market", "stock", "trade", "trading", "price", "chart", "analysis", "technical",
            "fundamental", "invest", "investment", "fund", "capital", "bull", "bear", "rally",
            "crash", "correction", "resistance", "support", "volume", "liquidity", "volatility",
            "trend", "breakout", "sell", "buy", "long", "short", "profit", "loss", "roi", "yield",
            
            # Finance symbols
            "$", "€", "£", "¥"
        ]
        
        # Advanced detection with context awareness
        market_related = []
        
        for post in posts:
            post_text = post.get('text', '').lower()
            
            # Skip posts with no text
            if not post_text:
                continue
                
            # Track matched keywords for logging
            matched_keywords = []
            
            # Basic keyword matching
            for keyword in market_keywords:
                if keyword in post_text:
                    matched_keywords.append(keyword)
            
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
                    break
                    
            # Also check hashtags if available
            hashtags = post.get('topics', {}).get('hashtags', [])
            for tag in hashtags:
                tag = tag.lower()
                if any(keyword in tag for keyword in market_keywords):
                    matched_keywords.append(f"hashtag:{tag}")
            
            # Mark as market related if we found any matches
            if matched_keywords:
                # Mark as market related
                post['market_related'] = True
                post['market_keywords'] = matched_keywords
                market_related.append(post)
                
        logger.logger.info(f"Found {len(market_related)} market-related posts out of {len(posts)}")
        
        # Log some example keywords for debugging
        if market_related:
            examples = [p.get('market_keywords', [])[:3] for p in market_related[:3]]
            logger.logger.debug(f"Example keywords matched: {examples}")
        
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
    
    def prioritize_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort posts by engagement level, relevance, recency with improved ranking
        
        Args:
            posts: List of post data dictionaries
            
        Returns:
            Sorted list of posts
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
                hours_ago = (datetime.now() - timestamp).total_seconds() / 3600
                
                # More recent posts get higher scores (inverse relationship)
                recency_score = max(0, 30 - (hours_ago * 2.5))  # Lose 2.5 points per hour, max 30
                score += recency_score
            
            # Factor 3: Content relevance (0-20 points)
            # More crypto/finance keywords = higher relevance
            keywords = post.get('market_keywords', [])
            relevance_score = min(20, len(keywords) * 5)  # 5 points per keyword, max 20
            score += relevance_score
            
            # Factor 4: Has question (bonus 10 points)
            # Questions are good opportunities for helpful replies
            if post.get('topics', {}).get('has_question', False):
                score += 10
                
            # Factor 5: Media content (bonus 5 points)
            # Posts with media often get more engagement
            if post.get('has_media', False):
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
                   
