#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Any, Tuple
import time
import re
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
        
        # Post selectors
        self.timeline_post_selector = '[data-testid="tweet"]'
        self.post_text_selector = '[data-testid="tweetText"]'
        self.post_author_selector = '[data-testid="User-Name"]'
        self.post_timestamp_selector = 'time'
        self.post_metrics_selector = '[data-testid="reply"], [data-testid="retweet"], [data-testid="like"]'
        
        logger.logger.info("Timeline scraper initialized")
        
    def navigate_to_home_timeline(self) -> bool:
        """
        Navigate to home timeline
        
        Returns:
            bool: True if navigation was successful, False otherwise
        """
        try:
            self.browser.driver.get("https://twitter.com/home")
            
            # Wait for timeline to load
            WebDriverWait(self.browser.driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, self.timeline_post_selector))
            )
            
            time.sleep(2)  # Allow for dynamic content to load
            logger.logger.info("Successfully navigated to home timeline")
            return True
            
        except Exception as e:
            logger.log_error("Timeline Navigation", str(e))
            return False
    
    def scrape_timeline(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Scrape X timeline and return post data
        
        Args:
            count: Number of posts to return
            
        Returns:
            List of post data dictionaries
        """
        if not self.navigate_to_home_timeline():
            logger.logger.error("Failed to navigate to timeline, cannot scrape posts")
            return []
        
        retry_count = 0
        posts_data = []
        
        while retry_count < self.max_retries and len(posts_data) < count:
            try:
                # Get current posts
                post_elements = self.browser.driver.find_elements(By.CSS_SELECTOR, self.timeline_post_selector)
                
                # Check if we found any posts
                if not post_elements:
                    logger.logger.warning("No posts found on timeline")
                    time.sleep(2)
                    retry_count += 1
                    continue
                
                logger.logger.debug(f"Found {len(post_elements)} post elements")
                
                # Process each post
                new_posts_found = False
                
                for post_element in post_elements:
                    # Check if we've reached our target count
                    if len(posts_data) >= count:
                        break
                    
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
                            logger.logger.debug(f"Extracted post from {post_data.get('author_handle', 'unknown')}")
                    
                    except StaleElementReferenceException:
                        # Element reference is no longer valid, skip this post
                        continue
                        
                    except Exception as e:
                        logger.logger.warning(f"Failed to process post: {str(e)}")
                        continue
                
                # Break if we've found enough posts
                if len(posts_data) >= count:
                    break
                    
                # If no new posts were found, scroll down
                if not new_posts_found or len(posts_data) < count:
                    self._scroll_down()
                    time.sleep(self.scroll_pause_time)
                    
                # Increment retry count if we're still not finding enough posts
                if not new_posts_found:
                    retry_count += 1
                    
            except Exception as e:
                logger.log_error("Timeline Scraping", str(e))
                retry_count += 1
                time.sleep(2)
        
        logger.logger.info(f"Scraped {len(posts_data)} posts from timeline")
        
        # Trim to requested count
        return posts_data[:count]
    
    def extract_post_data(self, post_element) -> Optional[Dict[str, Any]]:
        """
        Extract relevant data from a post element
        
        Args:
            post_element: Selenium WebElement for the post
            
        Returns:
            Dictionary containing post data or None if extraction failed
        """
        try:
            # Ensure the post is in the viewport
            self.browser.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", post_element)
            time.sleep(0.5)
            
            # Extract post ID
            post_id = self._extract_post_id(post_element)
            
            # Extract post URL
            post_url = self._extract_post_url(post_element)
            
            # Extract the author information
            author_name, author_handle, author_link = self._extract_author_info(post_element)
            
            # Extract post content
            try:
                text_element = post_element.find_element(By.CSS_SELECTOR, self.post_text_selector)
                post_text = text_element.text
            except NoSuchElementException:
                post_text = ""
                
            # Extract timestamp
            timestamp, parsed_time = self._extract_timestamp(post_element)
            
            # Extract engagement metrics
            metrics = self._extract_engagement_metrics(post_element)
            
            # Calculate a simple engagement score
            engagement_score = self._calculate_engagement_score(metrics)
            
            # Check if post has images, videos, or links
            has_media = self._check_for_media(post_element)
            
            # Assemble post data
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
                'scraped_at': datetime.now()
            }
            
            return post_data
            
        except Exception as e:
            logger.log_error("Post Data Extraction", str(e))
            return None
    
    def get_market_related_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter posts to only include those related to markets
        
        Args:
            posts: List of post data dictionaries
            
        Returns:
            Filtered list of market-related posts
        """
        market_keywords = [
            "bitcoin", "crypto", "btc", "eth", "ethereum", "market", "stock", 
            "trading", "bull", "bear", "price", "rally", "crash", "investment",
            "token", "$", "coin", "blockchain", "defi", "nft", "altcoin",
            "binance", "coinbase", "exchange", "wallet", "hodl", "chart"
        ]
        
        market_related = []
        
        for post in posts:
            post_text = post.get('text', '').lower()
            
            # Check if any keyword appears in the post text
            if any(keyword in post_text for keyword in market_keywords):
                # Mark as market related
                post['market_related'] = True
                post['market_keywords'] = [k for k in market_keywords if k in post_text]
                market_related.append(post)
                
        logger.logger.debug(f"Found {len(market_related)} market-related posts out of {len(posts)}")
        return market_related
    
    def _extract_post_id(self, post_element) -> str:
        """Extract unique identifier for post"""
        try:
            # Try to get the data-testid or article id
            post_id = post_element.get_attribute('data-testid')
            
            if not post_id:
                # Try getting a link to the post
                link_element = post_element.find_element(By.CSS_SELECTOR, 'a[href*="/status/"]')
                href = link_element.get_attribute('href')
                if href:
                    # Extract status ID from the URL
                    match = re.search(r'/status/(\d+)', href)
                    if match:
                        post_id = match.group(1)
            
            if not post_id:
                # Generate a pseudo-ID from post content
                try:
                    text_element = post_element.find_element(By.CSS_SELECTOR, self.post_text_selector)
                    text = text_element.text[:50]  # First 50 chars
                    time_element = post_element.find_element(By.CSS_SELECTOR, self.post_timestamp_selector)
                    time_str = time_element.get_attribute('datetime')
                    post_id = f"{text}_{time_str}"
                except:
                    # Last resort - use a random ID with timestamp
                    post_id = f"post_{int(time.time() * 1000)}"
            
            return post_id
            
        except Exception as e:
            logger.logger.warning(f"Failed to extract post ID: {str(e)}")
            return f"unknown_post_{int(time.time() * 1000)}"
    
    def _extract_post_url(self, post_element) -> Optional[str]:
        """Extract the URL to the post"""
        try:
            # Look for the status link
            link_elements = post_element.find_elements(By.CSS_SELECTOR, 'a[href*="/status/"]')
            for link in link_elements:
                href = link.get_attribute('href')
                # Make sure this is a direct link to a tweet status
                if '/status/' in href and not '/photo/' in href:
                    return href
            
            return None
            
        except Exception as e:
            logger.logger.warning(f"Failed to extract post URL: {str(e)}")
            return None
    
    def _extract_author_info(self, post_element) -> Tuple[str, str, str]:
        """Extract author name, handle and profile link"""
        try:
            author_element = post_element.find_element(By.CSS_SELECTOR, self.post_author_selector)
            
            # Extract author name
            try:
                name_element = author_element.find_element(By.CSS_SELECTOR, 'span')
                author_name = name_element.text
            except:
                author_name = "Unknown"
                
            # Extract author handle
            try:
                handle_elements = author_element.find_elements(By.CSS_SELECTOR, 'span')
                if len(handle_elements) > 1:
                    author_handle = handle_elements[1].text
                    # Clean up if the handle starts with @
                    if author_handle.startswith('@'):
                        author_handle = author_handle
                    else:
                        author_handle = f"@{author_handle}"
                else:
                    author_handle = "@unknown"
            except:
                author_handle = "@unknown"
            
            # Extract author profile link
            try:
                link_element = author_element.find_element(By.XPATH, './/*[starts-with(@href, "/")]')
                author_link = f"https://twitter.com{link_element.get_attribute('href')}"
            except:
                author_link = ""
                
            return author_name, author_handle, author_link
            
        except Exception as e:
            logger.logger.warning(f"Failed to extract author info: {str(e)}")
            return "Unknown", "@unknown", ""
    
    def _extract_timestamp(self, post_element) -> Tuple[str, Optional[datetime]]:
        """Extract timestamp text and parsed datetime"""
        try:
            time_element = post_element.find_element(By.CSS_SELECTOR, self.post_timestamp_selector)
            
            # Get the raw timestamp text
            timestamp_text = time_element.text
            
            # Try to parse the datetime attribute
            datetime_attr = time_element.get_attribute('datetime')
            
            if datetime_attr:
                parsed_time = datetime.fromisoformat(datetime_attr.replace('Z', '+00:00'))
                # Convert to local time
                parsed_time = parsed_time.astimezone(None)
            else:
                # Fallback: try to interpret the text
                parsed_time = self._parse_relative_time(timestamp_text)
                
            return timestamp_text, parsed_time
            
        except Exception as e:
            logger.logger.warning(f"Failed to extract timestamp: {str(e)}")
            return "", None
    
    def _parse_relative_time(self, time_text: str) -> Optional[datetime]:
        """Parse relative time expressions like '5m', '2h', etc."""
        now = datetime.now()
        
        # Match common time formats
        if 'm' in time_text and time_text[:-1].isdigit():
            # Minutes
            minutes = int(time_text[:-1])
            return now - timedelta(minutes=minutes)
            
        elif 'h' in time_text and time_text[:-1].isdigit():
            # Hours
            hours = int(time_text[:-1])
            return now - timedelta(hours=hours)
            
        elif 'd' in time_text and time_text[:-1].isdigit():
            # Days
            days = int(time_text[:-1])
            return now - timedelta(days=days)
            
        return now  # Default to current time if parsing fails
    
    def _extract_engagement_metrics(self, post_element) -> Dict[str, int]:
        """Extract engagement metrics (replies, reposts, likes)"""
        metrics = {
            'replies': 0,
            'reposts': 0, 
            'likes': 0
        }
        
        try:
            # Find all metric elements
            metric_elements = post_element.find_elements(By.CSS_SELECTOR, self.post_metrics_selector)
            
            for element in metric_elements:
                try:
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
                    continue
                    
            return metrics
            
        except Exception as e:
            logger.logger.warning(f"Failed to extract engagement metrics: {str(e)}")
            return metrics
    
    def _calculate_engagement_score(self, metrics: Dict[str, int]) -> float:
        """Calculate a weighted engagement score"""
        # Weights: replies have highest value, then reposts, then likes
        reply_weight = 3.0
        repost_weight = 2.0
        like_weight = 1.0
        
        score = (
            metrics.get('replies', 0) * reply_weight +
            metrics.get('reposts', 0) * repost_weight +
            metrics.get('likes', 0) * like_weight
        )
        
        return score
    
    def _check_for_media(self, post_element) -> bool:
        """Check if post contains media (images, videos, links)"""
        try:
            # Check for images
            images = post_element.find_elements(By.CSS_SELECTOR, 'img[src*="pbs.twimg.com"]')
            if images:
                return True
                
            # Check for videos
            videos = post_element.find_elements(By.CSS_SELECTOR, 'video')
            if videos:
                return True
                
            # Check for card links
            cards = post_element.find_elements(By.CSS_SELECTOR, '[data-testid="card.wrapper"]')
            if cards:
                return True
                
            return False
            
        except Exception:
            return False
    
    def _scroll_down(self) -> None:
        """Scroll down to load more posts"""
        try:
            self.browser.driver.execute_script("window.scrollBy(0, 1000);")
        except Exception as e:
            logger.logger.warning(f"Failed to scroll timeline: {str(e)}")
    
    def prioritize_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort posts by engagement level, follower count, recency
        
        Args:
            posts: List of post data dictionaries
            
        Returns:
            Sorted list of posts
        """
        # Sort by engagement score (highest first)
        sorted_posts = sorted(posts, key=lambda x: x.get('engagement_score', 0), reverse=True)
        
        # TODO: Add follower count to sorting once we have a method to get that
        
        # Further sort by recency for posts with similar engagement
        # Group posts by engagement tiers
        engagement_tiers = []
        current_tier = []
        last_score = None
        
        for post in sorted_posts:
            score = post.get('engagement_score', 0)
            
            # If this is the first post or the score is significantly different from the last one
            if last_score is None or abs(score - last_score) > 10:
                if current_tier:
                    # Sort the current tier by recency and add to result
                    current_tier = sorted(current_tier, key=lambda x: x.get('timestamp', datetime.min), reverse=True)
                    engagement_tiers.extend(current_tier)
                    current_tier = []
                
                # Start a new tier
                current_tier = [post]
                last_score = score
            else:
                # Add to current tier
                current_tier.append(post)
                last_score = score
        
        # Add the last tier
        if current_tier:
            current_tier = sorted(current_tier, key=lambda x: x.get('timestamp', datetime.min), reverse=True)
            engagement_tiers.extend(current_tier)
        
        return engagement_tiers
    
    def filter_already_replied_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out posts we've already replied to
        
        Args:
            posts: List of post data dictionaries
            
        Returns:
            Filtered list of posts
        """
        if not self.db:
            return posts
            
        filtered_posts = []
        
        for post in posts:
            post_id = post.get('post_id')
            post_url = post.get('post_url')
            
            # Check if we've already replied to this post
            already_replied = self.db.check_if_post_replied(post_id, post_url)
            
            if not already_replied:
                filtered_posts.append(post)
                
        logger.logger.debug(f"Filtered out {len(posts) - len(filtered_posts)} already replied posts")
        return filtered_posts
