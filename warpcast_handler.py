#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Any, Union, Tuple
import time
import json
import requests
import logging
from datetime import datetime, timedelta
import os

from utils.logger import logger

class WarpcastHandler:
    """
    Handles posting to Warpcast (Farcaster) platform using their API
    Built to mirror the functionality of the Twitter posting mechanisms
    """
    
    def __init__(self, config):
        """
        Initialize the Warpcast handler with configuration and API credentials
        """
        self.config = config
        
        # Load API credentials from config/environment
        self.api_key = os.getenv('WARPCAST_API_KEY', '')
        self.api_secret = os.getenv('WARPCAST_API_SECRET', '')
        self.client_id = os.getenv('WARPCAST_CLIENT_ID', '')
        self.client_secret = os.getenv('WARPCAST_CLIENT_SECRET', '')
        self.access_token = os.getenv('WARPCAST_ACCESS_TOKEN', '')
        self.refresh_token = os.getenv('WARPCAST_REFRESH_TOKEN', '')
        
        # Warpcast API endpoints
        self.base_url = "https://api.warpcast.com/v2"
        self.auth_url = f"{self.base_url}/auth"
        self.cast_url = f"{self.base_url}/casts"
        self.user_url = f"{self.base_url}/me"
        
        # API rate limit settings
        self.rate_limit_remaining = 100
        self.rate_limit_reset = 0
        self.min_request_interval = 1  # seconds between requests
        self.last_request_time = 0
        
        # Request retry settings
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
        # Track posts to avoid duplicates
        self.recent_posts = []
        self.max_cached_posts = 50
        
        # Verify credentials on init if available
        self.is_authorized = False
        if self.access_token:
            self.is_authorized = self._verify_credentials()
            
        logger.logger.info(f"WarpcastHandler initialized. Authorized: {self.is_authorized}")

    def _verify_credentials(self) -> bool:
        """
        Verify that the provided API credentials are valid
        Returns True if credentials are valid, False otherwise
        """
        try:
            # Check if we have the necessary credentials
            if not self.access_token:
                logger.logger.warning("No Warpcast access token provided")
                return False
                
            # Try to get user profile
            response = self._make_authenticated_request(
                method="GET",
                endpoint=self.user_url
            )
            
            if response and response.status_code == 200:
                user_data = response.json()
                if 'result' in user_data and 'user' in user_data['result']:
                    username = user_data['result']['user'].get('username', 'unknown')
                    logger.logger.info(f"Successfully authenticated with Warpcast as @{username}")
                    return True
            
            logger.logger.warning("Failed to verify Warpcast credentials")
            return False
            
        except Exception as e:
            logger.log_error("Warpcast Credential Verification", str(e))
            return False

    def _enforce_rate_limit(self):
        """
        Enforce rate limiting to avoid API throttling
        """
        current_time = time.time()
        
        # If we know we're near the limit, wait until reset
        if self.rate_limit_remaining < 5 and self.rate_limit_reset > current_time:
            sleep_time = self.rate_limit_reset - current_time + 1
            logger.logger.debug(f"Rate limit nearly reached, waiting {sleep_time:.1f}s")
            time.sleep(sleep_time)
            
        # Maintain minimum interval between requests
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()

    def _update_rate_limit_info(self, response: requests.Response):
        """
        Update rate limit information from response headers
        """
        try:
            if 'X-RateLimit-Remaining' in response.headers:
                self.rate_limit_remaining = int(response.headers['X-RateLimit-Remaining'])
                
            if 'X-RateLimit-Reset' in response.headers:
                self.rate_limit_reset = int(response.headers['X-RateLimit-Reset'])
                
            logger.logger.debug(f"Warpcast rate limit: {self.rate_limit_remaining} requests remaining")
            
        except Exception as e:
            logger.logger.debug(f"Error parsing rate limit headers: {str(e)}")

    def _refresh_access_token(self) -> bool:
        """
        Refresh the access token using the refresh token
        Returns True if successful, False otherwise
        """
        try:
            if not self.refresh_token or not self.client_id or not self.client_secret:
                logger.logger.warning("Missing credentials for token refresh")
                return False
                
            data = {
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'grant_type': 'refresh_token',
                'refresh_token': self.refresh_token
            }
            
            response = requests.post(
                f"{self.auth_url}/token",
                data=json.dumps(data),
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                token_data = response.json()
                
                # Update tokens
                self.access_token = token_data.get('access_token', '')
                new_refresh_token = token_data.get('refresh_token')
                
                if new_refresh_token:
                    self.refresh_token = new_refresh_token
                    
                # Update expiration
                expires_in = token_data.get('expires_in', 3600)
                
                logger.logger.info(f"Warpcast token refreshed, expires in {expires_in}s")
                return True
            else:
                logger.logger.warning(f"Failed to refresh token: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.log_error("Warpcast Token Refresh", str(e))
            return False

    def _make_authenticated_request(self, method: str, endpoint: str, data: Dict = None, files: Dict = None, 
                                 params: Dict = None, retry_count: int = 0) -> Optional[requests.Response]:
        """
        Make an authenticated request to the Warpcast API with retry logic
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint to call
            data: Request data (for POST/PUT)
            files: Files to upload
            params: URL parameters
            retry_count: Current retry count
            
        Returns:
            Response object or None if failed
        """
        # Enforce rate limiting
        self._enforce_rate_limit()
        
        # Check if we have a valid token
        if not self.access_token:
            logger.logger.warning("No access token available for Warpcast API request")
            return None
            
        # Prepare headers
        headers = {
            'Authorization': f"Bearer {self.access_token}",
            'Accept': 'application/json'
        }
        
        # Add content type if we're sending data
        if data and not files:
            headers['Content-Type'] = 'application/json'
            
        try:
            # Make the request
            if method.upper() == 'GET':
                response = requests.get(endpoint, headers=headers, params=params)
            elif method.upper() == 'POST':
                if files:
                    # Multipart form data for file uploads
                    response = requests.post(endpoint, headers=headers, data=data, files=files)
                else:
                    # JSON data
                    response = requests.post(endpoint, headers=headers, data=json.dumps(data) if data else None)
            elif method.upper() == 'PUT':
                response = requests.put(endpoint, headers=headers, data=json.dumps(data) if data else None)
            elif method.upper() == 'DELETE':
                response = requests.delete(endpoint, headers=headers)
            else:
                logger.logger.warning(f"Unsupported HTTP method: {method}")
                return None
                
            # Update rate limit info
            self._update_rate_limit_info(response)
            
            # Check for authentication errors
            if response.status_code == 401:
                logger.logger.warning("Authentication failed, attempting to refresh token")
                
                # Try to refresh the token and retry
                if self._refresh_access_token() and retry_count < self.max_retries:
                    return self._make_authenticated_request(
                        method=method,
                        endpoint=endpoint,
                        data=data,
                        files=files,
                        params=params,
                        retry_count=retry_count + 1
                    )
                else:
                    logger.logger.error("Failed to refresh token, authentication failed")
                    return None
                    
            # Check for rate limiting
            if response.status_code == 429:
                if retry_count < self.max_retries:
                    # Extract retry-after header if available
                    retry_after = int(response.headers.get('Retry-After', self.retry_delay))
                    
                    logger.logger.warning(f"Rate limited by Warpcast API, retrying in {retry_after}s")
                    time.sleep(retry_after)
                    
                    return self._make_authenticated_request(
                        method=method,
                        endpoint=endpoint,
                        data=data,
                        files=files,
                        params=params,
                        retry_count=retry_count + 1
                    )
                else:
                    logger.logger.error(f"Exceeded max retries for rate limiting")
                    return None
                    
            # Check for server errors
            if response.status_code >= 500:
                if retry_count < self.max_retries:
                    # Use exponential backoff for server errors
                    backoff = self.retry_delay * (2 ** retry_count)
                    
                    logger.logger.warning(f"Warpcast server error: {response.status_code}, retrying in {backoff}s")
                    time.sleep(backoff)
                    
                    return self._make_authenticated_request(
                        method=method,
                        endpoint=endpoint,
                        data=data,
                        files=files,
                        params=params,
                        retry_count=retry_count + 1
                    )
                else:
                    logger.logger.error(f"Exceeded max retries for server error: {response.status_code}")
                    return None
                    
            return response
            
        except requests.exceptions.RequestException as e:
            if retry_count < self.max_retries:
                # Use exponential backoff for network errors
                backoff = self.retry_delay * (2 ** retry_count)
                
                logger.logger.warning(f"Network error: {str(e)}, retrying in {backoff}s")
                time.sleep(backoff)
                
                return self._make_authenticated_request(
                    method=method,
                    endpoint=endpoint,
                    data=data,
                    files=files,
                    params=params,
                    retry_count=retry_count + 1
                )
            else:
                logger.log_error("Warpcast API Request", f"Network error after max retries: {str(e)}")
                return None
                
        except Exception as e:
            logger.log_error("Warpcast API Request", str(e))
            return None

    def post_cast(self, text: str, reply_to: str = None, embed_url: str = None) -> Tuple[bool, Optional[str]]:
        """
        Post a new cast (Warpcast equivalent of a tweet)
        
        Args:
            text: The text content to post (max 320 chars for Warpcast)
            reply_to: Optional cast ID to reply to
            embed_url: Optional URL to embed in the cast
            
        Returns:
            Tuple of (success, cast_id)
        """
        try:
            # Check if we're authorized
            if not self.is_authorized:
                if not self._verify_credentials():
                    logger.logger.error("Not authorized with Warpcast, cannot post")
                    return False, None
                    
            # Validate text length
            if len(text) > 320:
                # Truncate with preference to complete sentences
                last_period = text[:320].rfind('.')
                last_question = text[:320].rfind('?')
                last_exclamation = text[:320].rfind('!')
                
                # Find the last sentence end
                last_sentence_end = max(last_period, last_question, last_exclamation)
                
                if last_sentence_end > 240:  # If we can get a substantial cast with complete sentence
                    text = text[:last_sentence_end+1]
                else:
                    # Find last space to avoid cutting words
                    last_space = text[:317].rfind(' ')
                    if last_space > 0:
                        text = text[:last_space] + "..."
                    else:
                        # Hard truncate as last resort
                        text = text[:317] + "..."
                        
                logger.logger.info(f"Truncated cast text to {len(text)} characters")
                
            # Check if this is a duplicate of a recent post
            if self._is_duplicate(text):
                logger.logger.warning("Skipping duplicate Warpcast post")
                return False, None
                
            # Prepare cast data
            data = {
                "text": text
            }
            
            # Add reply parent if provided
            if reply_to:
                data["parent"] = {"hash": reply_to}
                
            # Add embedded URL if provided
            if embed_url:
                data["embeds"] = [{"url": embed_url}]
                
            # Make the API request
            response = self._make_authenticated_request(
                method="POST",
                endpoint=f"{self.cast_url}",
                data=data
            )
            
            if response and response.status_code in (200, 201):
                result = response.json()
                
                if 'result' in result and 'cast' in result['result']:
                    cast_hash = result['result']['cast'].get('hash', '')
                    
                    # Add to recent posts cache
                    self._add_to_recent_posts(text)
                    
                    logger.logger.info(f"Successfully posted to Warpcast, cast hash: {cast_hash}")
                    return True, cast_hash
                    
            error_msg = f"Failed to post to Warpcast: {response.status_code}" if response else "Failed to post to Warpcast"
            if response:
                error_msg += f" - {response.text}"
                
            logger.logger.error(error_msg)
            return False, None
            
        except Exception as e:
            logger.log_error("Warpcast Post", str(e))
            return False, None

    def post_cast_with_image(self, text: str, image_path: str, reply_to: str = None) -> Tuple[bool, Optional[str]]:
        """
        Post a new cast with an attached image
        
        Args:
            text: The text content to post
            image_path: Path to image file to upload
            reply_to: Optional cast ID to reply to
            
        Returns:
            Tuple of (success, cast_id)
        """
        try:
            # Check if we're authorized
            if not self.is_authorized:
                if not self._verify_credentials():
                    logger.logger.error("Not authorized with Warpcast, cannot post")
                    return False, None
                    
            # Validate text length (Warpcast limit is 320 chars)
            if len(text) > 320:
                text = text[:317] + "..."
                logger.logger.info(f"Truncated cast text to {len(text)} characters")
                
            # Validate image file
            if not os.path.exists(image_path):
                logger.logger.error(f"Image file not found: {image_path}")
                return False, None
                
            # First, upload the image
            image_url = self._upload_image(image_path)
            if not image_url:
                logger.logger.error("Failed to upload image, proceeding with text-only cast")
                return self.post_cast(text, reply_to)
                
            # Prepare cast data
            data = {
                "text": text,
                "embeds": [{"image": {"url": image_url}}]
            }
            
            # Add reply parent if provided
            if reply_to:
                data["parent"] = {"hash": reply_to}
                
            # Make the API request
            response = self._make_authenticated_request(
                method="POST",
                endpoint=f"{self.cast_url}",
                data=data
            )
            
            if response and response.status_code in (200, 201):
                result = response.json()
                
                if 'result' in result and 'cast' in result['result']:
                    cast_hash = result['result']['cast'].get('hash', '')
                    
                    # Add to recent posts cache
                    self._add_to_recent_posts(text)
                    
                    logger.logger.info(f"Successfully posted to Warpcast with image, cast hash: {cast_hash}")
                    return True, cast_hash
                    
            error_msg = f"Failed to post to Warpcast with image: {response.status_code}" if response else "Failed to post to Warpcast with image"
            if response:
                error_msg += f" - {response.text}"
                
            logger.logger.error(error_msg)
            return False, None
            
        except Exception as e:
            logger.log_error("Warpcast Post With Image", str(e))
            return False, None

    def _upload_image(self, image_path: str) -> Optional[str]:
        """
        Upload an image to Warpcast
        
        Args:
            image_path: Path to the image file
            
        Returns:
            URL of the uploaded image or None if failed
        """
        try:
            # Create upload request
            response = self._make_authenticated_request(
                method="POST",
                endpoint=f"{self.base_url}/uploads",
                data={"content_type": "image/jpeg"}  # Adjust content type as needed
            )
            
            if not response or response.status_code != 200:
                logger.logger.error(f"Failed to initiate image upload: {response.status_code if response else 'No response'}")
                return None
                
            upload_data = response.json()
            
            if 'result' not in upload_data or 'uploadUrl' not in upload_data['result']:
                logger.logger.error("Invalid upload response format")
                return None
                
            # Get the upload URL and fields
            upload_url = upload_data['result']['uploadUrl']
            
            # Open the image file
            with open(image_path, 'rb') as image_file:
                # Upload directly to the provided URL
                upload_response = requests.put(
                    upload_url,
                    data=image_file.read(),
                    headers={'Content-Type': 'image/jpeg'}  # Adjust content type as needed
                )
                
            if upload_response.status_code not in (200, 201, 204):
                logger.logger.error(f"Failed to upload image: {upload_response.status_code}")
                return None
                
            # Return the URL for embedding
            return upload_data['result']['url']
            
        except Exception as e:
            logger.log_error("Warpcast Image Upload", str(e))
            return None

    def get_user_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the authenticated user
        
        Returns:
            User data dictionary or None if failed
        """
        try:
            response = self._make_authenticated_request(
                method="GET",
                endpoint=self.user_url
            )
            
            if not response or response.status_code != 200:
                return None
                
            user_data = response.json()
            
            if 'result' not in user_data or 'user' not in user_data['result']:
                return None
                
            return user_data['result']['user']
            
        except Exception as e:
            logger.log_error("Warpcast User Info", str(e))
            return None

    def get_user_feed(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the authenticated user's feed
        
        Args:
            limit: Maximum number of casts to retrieve
            
        Returns:
            List of cast data dictionaries
        """
        try:
            response = self._make_authenticated_request(
                method="GET",
                endpoint=f"{self.base_url}/feed",
                params={"limit": str(limit)}
            )
            
            if not response or response.status_code != 200:
                return []
                
            feed_data = response.json()
            
            if 'result' not in feed_data or 'casts' not in feed_data['result']:
                return []
                
            return feed_data['result']['casts']
            
        except Exception as e:
            logger.log_error("Warpcast Feed Retrieval", str(e))
            return []

    def get_cast_by_hash(self, cast_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific cast by its hash
        
        Args:
            cast_hash: The unique hash of the cast
            
        Returns:
            Cast data dictionary or None if not found
        """
        try:
            response = self._make_authenticated_request(
                method="GET",
                endpoint=f"{self.cast_url}/{cast_hash}"
            )
            
            if not response or response.status_code != 200:
                return None
                
            cast_data = response.json()
            
            if 'result' not in cast_data or 'cast' not in cast_data['result']:
                return None
                
            return cast_data['result']['cast']
            
        except Exception as e:
            logger.log_error("Warpcast Cast Retrieval", str(e))
            return None

    def delete_cast(self, cast_hash: str) -> bool:
        """
        Delete a cast
        
        Args:
            cast_hash: The unique hash of the cast to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = self._make_authenticated_request(
                method="DELETE",
                endpoint=f"{self.cast_url}/{cast_hash}"
            )
            
            return response is not None and response.status_code == 200
            
        except Exception as e:
            logger.log_error("Warpcast Cast Deletion", str(e))
            return False

    def _is_duplicate(self, text: str) -> bool:
        """
        Check if a post text is a duplicate of a recent post
        
        Args:
            text: The text to check
            
        Returns:
            True if it's a duplicate, False otherwise
        """
        return text in self.recent_posts

    def _add_to_recent_posts(self, text: str) -> None:
        """
        Add a post text to the recent posts cache
        
        Args:
            text: The text to add
        """
        if text not in self.recent_posts:
            self.recent_posts.append(text)
            
            # Keep cache at a reasonable size
            if len(self.recent_posts) > self.max_cached_posts:
                self.recent_posts.pop(0)  # Remove oldest entry
