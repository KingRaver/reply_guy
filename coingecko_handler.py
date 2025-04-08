#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import requests
from typing import Dict, List, Optional, Any, Union, Set
from datetime import datetime, timedelta
import json
from utils.logger import logger

class CoinGeckoHandler:
    """
    Enhanced CoinGecko API handler with caching, rate limiting, and fallback strategies
    Optimized for handling multiple tokens
    """
    def __init__(self, base_url: str, cache_duration: int = 60) -> None:
        """
        Initialize the CoinGecko handler
        
        Args:
            base_url: The base URL for the CoinGecko API
            cache_duration: Cache duration in seconds
        """
        self.base_url = base_url
        self.cache_duration = cache_duration
        self.cache = {}
        self.last_request_time = 0
        self.min_request_interval = 1.5  # Minimum 1.5 seconds between requests
        self.daily_requests = 0
        self.daily_requests_reset = datetime.now()
        self.failed_requests = 0
        self.active_retries = 0
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
        self.token_id_cache = {}  # Cache for token ID lookups
        
        logger.logger.info("CoinGecko handler initialized")
    
    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """Generate a unique cache key for the request"""
        param_str = json.dumps(params, sort_keys=True)
        return f"{endpoint}:{param_str}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if cache_key not in self.cache:
            return False
        
        cache_entry = self.cache[cache_key]
        cache_time = cache_entry['timestamp']
        current_time = time.time()
        
        return (current_time - cache_time) < self.cache_duration
    
    def _get_from_cache(self, cache_key: str) -> Any:
        """Get data from cache if available and valid"""
        if self._is_cache_valid(cache_key):
            logger.logger.debug(f"Cache hit for {cache_key}")
            return self.cache[cache_key]['data']
        return None
    
    def _add_to_cache(self, cache_key: str, data: Any) -> None:
        """Add data to cache"""
        self.cache[cache_key] = {
            'timestamp': time.time(),
            'data': data
        }
        logger.logger.debug(f"Added to cache: {cache_key}")
    
    def _clean_cache(self) -> None:
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if (current_time - entry['timestamp']) >= self.cache_duration
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")
    
    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting to avoid API restrictions"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            logger.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        # Reset daily request count if a day has passed
        if (datetime.now() - self.daily_requests_reset).total_seconds() >= 86400:
            self.daily_requests = 0
            self.daily_requests_reset = datetime.now()
            logger.logger.info("Daily request counter reset")
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Any]:
        """Make a request to the CoinGecko API with retries and error handling"""
        if params is None:
            params = {}
        
        url = f"{self.base_url}/{endpoint}"
        self._enforce_rate_limit()
        
        self.last_request_time = time.time()
        self.daily_requests += 1
        
        try:
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'application/json'
            }
            
            logger.logger.debug(f"Making API request to {endpoint} with params {params}")
            response = requests.get(url, params=params, headers=headers, timeout=30)
            
            if response.status_code == 200:
                logger.logger.debug(f"API request successful: {endpoint}")
                return response.json()
            elif response.status_code == 429:
                self.failed_requests += 1
                logger.logger.warning(f"API rate limit exceeded: {response.status_code}")
                time.sleep(self.retry_delay * 2)  # Longer delay for rate limits
                return None
            else:
                self.failed_requests += 1
                logger.logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            self.failed_requests += 1
            logger.logger.error(f"Request exception: {str(e)}")
            return None
        except Exception as e:
            self.failed_requests += 1
            logger.logger.error(f"Unexpected error in API request: {str(e)}")
            return None
    
    def get_with_cache(self, endpoint: str, params: Dict = None) -> Optional[Any]:
        """Get data from API with caching"""
        if params is None:
            params = {}
        
        cache_key = self._get_cache_key(endpoint, params)
        
        # Try to get from cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Not in cache, make API request
        retry_count = 0
        while retry_count < self.max_retries:
            data = self._make_request(endpoint, params)
            if data is not None:
                self._add_to_cache(cache_key, data)
                return data
            
            retry_count += 1
            if retry_count < self.max_retries:
                logger.logger.warning(f"Retrying API request ({retry_count}/{self.max_retries})")
                time.sleep(self.retry_delay * retry_count)
        
        logger.logger.error(f"Failed to get data after {self.max_retries} retries")
        return None
    
    def get_market_data(self, params: Dict = None) -> Optional[List[Dict[str, Any]]]:
        """
        Get cryptocurrency market data from CoinGecko
        
        Args:
            params: Query parameters for the API
            
        Returns:
            List of market data entries
        """
        endpoint = "coins/markets"
        
        # Set default params for all tracked tokens
        if params is None:
            params = {
                "vs_currency": "usd",
                "ids": "bitcoin,ethereum,solana,ripple,binancecoin,avalanche-2,polkadot,uniswap,near,aave,matic-network,filecoin,kaito",
                "order": "market_cap_desc",
                "per_page": 100,
                "page": 1,
                "sparkline": True,
                "price_change_percentage": "1h,24h,7d"
            }
        
        return self.get_with_cache(endpoint, params)
    
    def get_market_data_batched(self, token_ids: List[str], batch_size: int = 50) -> Optional[List[Dict[str, Any]]]:
        """
        Get market data for many tokens in batches to avoid URL length limitations
        
        Args:
            token_ids: List of CoinGecko token IDs
            batch_size: Maximum number of tokens per request
            
        Returns:
            Combined list of market data entries
        """
        if not token_ids:
            return []
        
        all_data = []
        for i in range(0, len(token_ids), batch_size):
            batch = token_ids[i:i+batch_size]
            batch_ids = ','.join(batch)
            
            params = {
                "vs_currency": "usd",
                "ids": batch_ids,
                "order": "market_cap_desc",
                "per_page": len(batch),
                "page": 1,
                "sparkline": True,
                "price_change_percentage": "1h,24h,7d"
            }
            
            batch_data = self.get_market_data(params)
            if batch_data:
                all_data.extend(batch_data)
            else:
                logger.logger.error(f"Failed to get data for batch {i//batch_size + 1}")
        
        return all_data if all_data else None
    
    def get_coin_detail(self, coin_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed data for a specific coin
        
        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin', 'ethereum')
            
        Returns:
            Detailed coin data
        """
        endpoint = f"coins/{coin_id}"
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "false",
            "developer_data": "false",
            "sparkline": "true"
        }
        
        return self.get_with_cache(endpoint, params)

    def get_coin_ohlc(self, coin_id: str, days: int = 1) -> Optional[List[List[float]]]:
        """
        Get OHLC data for a specific coin
        
        Args:
            coin_id: CoinGecko coin ID
            days: Number of days (1, 7, 14, 30, 90, 180, 365)
            
        Returns:
            OHLC data as list of [timestamp, open, high, low, close]
        """
        # Valid days values: 1, 7, 14, 30, 90, 180, 365
        if days not in [1, 7, 14, 30, 90, 180, 365]:
            days = 1
            
        endpoint = f"coins/{coin_id}/ohlc"
        params = {
            "vs_currency": "usd",
            "days": days
        }
        
        return self.get_with_cache(endpoint, params)
    
    def find_token_id(self, token_symbol: str) -> Optional[str]:
        """
        Find the exact CoinGecko ID for a token by symbol
        
        Args:
            token_symbol: Token symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            CoinGecko ID for token or None if not found
        """
        # Common token mappings for quick lookups
        common_mappings = {
            "btc": "bitcoin",
            "eth": "ethereum",
            "sol": "solana",
            "xrp": "ripple",
            "bnb": "binancecoin",
            "avax": "avalanche-2",
            "dot": "polkadot",
            "uni": "uniswap",
            "near": "near",
            "aave": "aave",
            "fil": "filecoin",
            "matic": "matic-network",
            "pol": "matic-network",
            "kaito": "kaito"
        }
        
        # Check in-memory cache first
        token_symbol_lower = token_symbol.lower()
        
        # Check common mappings
        if token_symbol_lower in common_mappings:
            return common_mappings[token_symbol_lower]
            
        # Check cache
        if token_symbol_lower in self.token_id_cache:
            return self.token_id_cache[token_symbol_lower]
        
        endpoint = "coins/list"
        coins_list = self.get_with_cache(endpoint)
        
        if not coins_list:
            return None
        
        # First try exact match on symbol
        for coin in coins_list:
            if coin.get('symbol', '').lower() == token_symbol_lower:
                logger.logger.info(f"Found {token_symbol} with ID: {coin['id']}")
                # Add to cache
                self.token_id_cache[token_symbol_lower] = coin['id']
                return coin['id']
        
        # If not found, try partial name match
        for coin in coins_list:
            if token_symbol_lower in coin.get('name', '').lower():
                logger.logger.info(f"Found possible {token_symbol} match with ID: {coin['id']} (name: {coin['name']})")
                # Add to cache
                self.token_id_cache[token_symbol_lower] = coin['id']
                return coin['id']
        
        logger.logger.error(f"Could not find {token_symbol} in CoinGecko coin list")
        return None
    
    def get_multiple_tokens_by_symbol(self, symbols: List[str]) -> Dict[str, str]:
        """
        Get CoinGecko IDs for multiple token symbols
        
        Args:
            symbols: List of token symbols
            
        Returns:
            Dictionary mapping symbols to CoinGecko IDs
        """
        # Common token mappings for quick lookups
        common_mappings = {
            "btc": "bitcoin",
            "eth": "ethereum",
            "sol": "solana",
            "xrp": "ripple",
            "bnb": "binancecoin",
            "avax": "avalanche-2",
            "dot": "polkadot",
            "uni": "uniswap",
            "near": "near",
            "aave": "aave",
            "fil": "filecoin",
            "matic": "matic-network",
            "pol": "matic-network",
            "kaito": "kaito"
        }
        
        # Initialize result with common mappings
        result = {}
        symbols_to_fetch = []
        
        for symbol in symbols:
            symbol_lower = symbol.lower()
            
            # Check common mappings first
            if symbol_lower in common_mappings:
                result[symbol] = common_mappings[symbol_lower]
                continue
                
            # Check cache next
            if symbol_lower in self.token_id_cache:
                result[symbol] = self.token_id_cache[symbol_lower]
                continue
                
            # Need to fetch from API
            symbols_to_fetch.append(symbol)
        
        if not symbols_to_fetch:
            return result
        
        # Fetch coins list once for all missing symbols
        endpoint = "coins/list"
        coins_list = self.get_with_cache(endpoint)
        
        if not coins_list:
            return result
        
        # Create lookup dictionary from coins list
        symbol_to_id = {}
        for coin in coins_list:
            coin_symbol = coin.get('symbol', '').lower()
            if coin_symbol and coin_symbol not in symbol_to_id:
                symbol_to_id[coin_symbol] = coin['id']
        
        # Assign found IDs to results and update cache
        for symbol in symbols_to_fetch:
            symbol_lower = symbol.lower()
            if symbol_lower in symbol_to_id:
                result[symbol] = symbol_to_id[symbol_lower]
                self.token_id_cache[symbol_lower] = symbol_to_id[symbol_lower]
                logger.logger.debug(f"Found {symbol} with ID: {symbol_to_id[symbol_lower]}")
            else:
                logger.logger.warning(f"Could not find ID for {symbol}")
                result[symbol] = None
        
        return result
    
    def get_trending_tokens(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get trending tokens from CoinGecko
        
        Returns:
            List of trending tokens data
        """
        endpoint = "search/trending"
        result = self.get_with_cache(endpoint)
        
        if result and 'coins' in result:
            return result['coins']
        return None
    
    def check_token_exists(self, token_id: str) -> bool:
        """
        Check if a token ID exists in CoinGecko
        
        Args:
            token_id: CoinGecko coin ID to check
            
        Returns:
            True if token exists, False otherwise
        """
        endpoint = f"coins/{token_id}"
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "false",
            "community_data": "false",
            "developer_data": "false",
            "sparkline": "false"
        }
        
        try:
            # Use a direct request with minimal data to check existence
            url = f"{self.base_url}/{endpoint}"
            self._enforce_rate_limit()
            
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'application/json'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            self.last_request_time = time.time()
            self.daily_requests += 1
            
            return response.status_code == 200
            
        except Exception as e:
            logger.logger.error(f"Error checking if {token_id} exists: {str(e)}")
            return False
    
    def get_request_stats(self) -> Dict[str, int]:
        """
        Get API request statistics
        
        Returns:
            Dictionary with request stats
        """
        self._clean_cache()
        return {
            'daily_requests': self.daily_requests,
            'failed_requests': self.failed_requests,
            'cache_size': len(self.cache),
            'token_id_cache_size': len(self.token_id_cache)
        }
        
    def optimize_for_multiple_tokens(self, tokens: List[str]) -> bool:
        """
        Optimize handler for a specific list of tokens by pre-caching their IDs
        
        Args:
            tokens: List of token symbols to optimize for
            
        Returns:
            True if optimization was successful, False otherwise
        """
        try:
            # Pre-fetch and cache token IDs
            token_ids = self.get_multiple_tokens_by_symbol(tokens)
            
            # Pre-fetch market data
            valid_ids = [id for id in token_ids.values() if id]
            if valid_ids:
                self.get_market_data_batched(valid_ids)
                logger.logger.info(f"Pre-cached data for {len(valid_ids)} tokens")
                return True
            
            return False
        except Exception as e:
            logger.logger.error(f"Error optimizing for multiple tokens: {str(e)}")
            return False
