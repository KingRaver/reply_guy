#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple, Set, Optional, Any
import re
import json
import time
from datetime import datetime
import string
from collections import Counter

from utils.logger import logger

class ContentAnalyzer:
    """
    Analyzer for detecting and extracting market-related content from posts
    """
    
    def __init__(self, config=None, db=None):
        """
        Initialize the content analyzer
        
        Args:
            config: Configuration instance containing settings (optional)
            db: Database instance for storing analysis data (optional)
        """
        self.config = config
        self.db = db
        
        # Initialize keyword sets for market content detection
        self._initialize_keywords()
        
        # Track trending topics
        self.trending_topics = Counter()
        self.trending_tokens = Counter()
        self.trending_reset_time = datetime.now()
        self.trending_reset_hours = 24
        
        logger.logger.info("Content analyzer initialized")
    
    def _initialize_keywords(self) -> None:
        """Initialize keyword dictionaries for different categories"""
        
        # Core crypto keywords
        self.crypto_keywords = {
            'bitcoin', 'btc', 'ethereum', 'eth', 'blockchain', 'crypto', 'token',
            'altcoin', 'defi', 'nft', 'mining', 'wallet', 'exchange', 'hodl',
            'stake', 'staking', 'coin', 'binance', 'coinbase', 'ledger', 'trezor',
            'solana', 'sol', 'ripple', 'xrp', 'cardano', 'ada', 'polkadot', 'dot',
            'dogecoin', 'doge', 'shiba', 'bnb', 'avax', 'terra', 'luna', 'link',
            'chainlink', 'matic', 'polygon', 'uniswap', 'sushiswap', 'pancakeswap',
            'aave', 'compound', 'yearn', 'maker', 'dai', 'usdt', 'usdc', 'busd',
            'tether', 'stablecoin', 'memecoin', 'yield', 'farming', 'liquidity'
        }
        
        # Market and trading keywords
        self.market_keywords = {
            'market', 'trading', 'investment', 'investor', 'bull', 'bear',
            'bullish', 'bearish', 'long', 'short', 'leverage', 'margin',
            'futures', 'option', 'swap', 'etf', 'fund', 'portfolio', 'asset',
            'stock', 'bond', 'equity', 'security', 'commodity', 'forex', 'fiat',
            'chart', 'analysis', 'indicator', 'pattern', 'support', 'resistance',
            'breakout', 'consolidation', 'volatility', 'volume', 'liquidity',
            'rally', 'crash', 'correction', 'dip', 'pump', 'dump', 'moon', 'fud',
            'trend', 'downtrend', 'uptrend', 'sideways', 'squeeze', 'capitulation',
            'profit', 'loss', 'roi', 'apy', 'apr', 'yield', 'dividend', 'return',
            'price', 'valuation', 'market cap', 'mcap', 'supply', 'demand', 'buy',
            'sell', 'order', 'entry', 'exit', 'target', 'stop loss', 'take profit'
        }
        
        # Economic and financial keywords
        self.economic_keywords = {
            'economy', 'inflation', 'deflation', 'recession', 'gdp', 'fed',
            'federal reserve', 'interest rate', 'monetary policy', 'fiscal policy',
            'treasury', 'bank', 'banking', 'central bank', 'debt', 'credit',
            'loan', 'mortgage', 'regulation', 'deregulation', 'sec', 'cftc',
            'tax', 'taxation', 'stimulus', 'bailout', 'currency', 'dollar',
            'euro', 'yen', 'yuan', 'pound', 'forex', 'exchange rate', 'liquidity',
            'financial', 'finance', 'fintech', 'payment', 'transaction', 'settlement'
        }
        
        # Cryptocurrency symbols/tickers
        self.crypto_symbols = {
            'btc', 'eth', 'sol', 'xrp', 'ada', 'dot', 'doge', 'shib', 'bnb', 
            'avax', 'matic', 'link', 'ltc', 'etc', 'bch', 'uni', 'atom', 
            'xlm', 'algo', 'near', 'ftm', 'one', 'hbar', 'vet', 'fil', 'theta',
            'egld', 'icp', 'xtz', 'eos', 'axs', 'sand', 'mana', 'gala', 'enj',
            'ape', 'aave', 'comp', 'sushi', 'cake', 'crv', 'mkr', 'snx', 'yfi'
        }
        
        # Common finance symbols and hashtags
        self.finance_symbols = {
            'spy', 'qqq', 'dia', 'iwm', 'vix', 'tlt', 'uso', 'gld', 'slv',
            'financialmarkets', 'investing', 'trading', 'wallstreetbets', 'fintwit',
            'stockmarket', 'investing101', 'tradingview', 'nyse', 'nasdaq'
        }
        
        # Categories of market topics for classification
        self.market_topic_categories = {
            'bull_market': ['bull', 'bullish', 'rally', 'pump', 'moon', 'mooning', 'to the moon', 'ath', 'all time high'],
            'bear_market': ['bear', 'bearish', 'crash', 'dump', 'dumping', 'correction', 'dip', 'sell off', 'selling', 'capitulation'],
            'trading': ['trade', 'trading', 'buy', 'sell', 'long', 'short', 'position', 'entry', 'exit', 'leverage', 'margin', 'futures'],
            'analysis': ['chart', 'analysis', 'ta', 'technical', 'indicator', 'pattern', 'support', 'resistance', 'trend', 'level'],
            'fundamental': ['adoption', 'utility', 'use case', 'fundamentals', 'project', 'team', 'roadmap', 'development', 'partnership'],
            'macro': ['fed', 'interest rate', 'inflation', 'deflation', 'economy', 'recession', 'regulation', 'policy', 'government', 'ban'],
            'sentiment': ['fud', 'fear', 'optimism', 'pessimism', 'sentiment', 'confidence', 'uncertainty', 'doubt', 'euphoria', 'panic'],
            'defi': ['defi', 'yield', 'farming', 'staking', 'liquidity', 'pool', 'swap', 'amm', 'lending', 'borrowing', 'governance'],
            'nft': ['nft', 'collectible', 'art', 'metaverse', 'game', 'gaming', 'play to earn', 'p2e', 'virtual', 'land', 'avatar']
        }
        
        # Common market questions
        self.market_question_patterns = [
            r'(what|how|why) (is|are|do|does) .*(bitcoin|crypto|market|price|rally|crash|dump|pump)',
            r'(should|could|would) .*(buy|sell|trade|invest|hold|stake)',
            r'(when|how|where) .*(invest|buy|sell|enter|exit|trade|stake)',
            r'(is|are) .*(bitcoin|crypto|market|price|rally|crash) .*(good|bad|over|starting|ending)',
            r'(what\'s|what is) (happening|going on) (with|in) .*(market|crypto|bitcoin|price)'
        ]
        
        # Sentiment words
        self.bullish_words = {
            'bullish', 'buy', 'long', 'moon', 'mooning', 'rally', 'pump', 'pumping',
            'uptrend', 'breakout', 'buy the dip', 'btd', 'hodl', 'holding', 'accumulate',
            'undervalued', 'support', 'strong', 'growth', 'profit', 'gain', 'winner',
            'higher high', 'higher low', 'bottom', 'bottoming', 'reversal', 'bounce'
        }
        
        self.bearish_words = {
            'bearish', 'sell', 'short', 'dump', 'dumping', 'crash', 'bear', 'correction',
            'downtrend', 'breakdown', 'sell the rally', 'falling', 'plummeting', 'weak',
            'overvalued', 'resistance', 'weak', 'decline', 'loss', 'loser', 'selling',
            'lower high', 'lower low', 'top', 'topping', 'head and shoulders', 'double top'
        }
        
        # Patterns for token amount formatting
        self.token_value_patterns = [
            r'\$([0-9,]+(\.[0-9]+)?)(K|M|B)?',  # $10K, $1.5M, $2B
            r'([0-9,]+(\.[0-9]+)?)\s*(BTC|ETH|SOL|XRP)',  # 0.5 BTC, 2.3 ETH
            r'([0-9,]+(\.[0-9]+)?)\s*(sats|gwei)',  # 10000 sats, 50 gwei
        ]
        
        # Combined keywords for faster checking
        self.all_market_keywords = self.crypto_keywords.union(
            self.market_keywords,
            self.economic_keywords,
            self.crypto_symbols
        )
        
        # Update symbol detection with $ prefix
        self.symbol_patterns = {f'${sym}' for sym in self.crypto_symbols}
        self.symbol_patterns.update({f'${sym}' for sym in self.finance_symbols})
        self.symbol_patterns.update(self.crypto_symbols)
        self.symbol_patterns.update(self.finance_symbols)
    
    def is_market_related(self, post_text: str) -> bool:
        """
        Determine if a post is related to markets/crypto
        
        Args:
            post_text: Text content of the post
            
        Returns:
            True if the post is market-related, False otherwise
        """
        if not post_text:
            return False
            
        post_text_lower = post_text.lower()
        
        # Check for cryptocurrency symbols with $ prefix
        for symbol in self.crypto_symbols:
            if f'${symbol}' in post_text_lower:
                return True
        
        # Check for words
        words = self._tokenize_text(post_text_lower)
        
        # If any market keyword is present, it's market-related
        if any(word in self.all_market_keywords for word in words):
            return True
            
        # Check for money values with currency symbols
        if re.search(r'\$\d+(\.\d+)?', post_text):
            # If there's a money value, check if any market words are nearby
            return any(word in self.all_market_keywords for word in words)
            
        # Check for percentage changes
        if re.search(r'\d+(\.\d+)?%', post_text):
            # If there's a percentage, check if any market words are nearby
            return any(word in self.all_market_keywords for word in words)
            
        return False
    
    def get_market_relevance_score(self, post_text: str) -> float:
        """
        Calculate a market relevance score from 0 to 1
        
        Args:
            post_text: Text content of the post
            
        Returns:
            Score from 0 (not relevant) to 1 (highly relevant)
        """
        if not post_text:
            return 0.0
            
        post_text_lower = post_text.lower()
        words = self._tokenize_text(post_text_lower)
        
        # Count market-related keywords
        keyword_count = sum(1 for word in words if word in self.all_market_keywords)
        
        # Count crypto/finance symbols with $ prefix
        symbol_count = sum(1 for symbol in self.crypto_symbols if f'${symbol}' in post_text_lower)
        
        # Find token values
        token_values = sum(1 for pattern in self.token_value_patterns if re.search(pattern, post_text))
        
        # Calculate a score based on the counts
        total_words = max(1, len(words))  # Avoid division by zero
        
        # Base score from keyword density
        keyword_score = min(1.0, (keyword_count + symbol_count + token_values) / (total_words * 0.3))
        
        # Boost score if specific patterns are found
        patterns_boost = 0.0
        
        # Check for specific market patterns
        if re.search(r'(bull|bear)\s*(market|run|trend)', post_text_lower):
            patterns_boost += 0.2
            
        if re.search(r'(buy|sell)\s*(the)\s*(dip|rally|pump)', post_text_lower):
            patterns_boost += 0.2
            
        if re.search(r'(to the moon|mooning|pumping|dumping)', post_text_lower):
            patterns_boost += 0.2
            
        if re.search(r'(btc|eth|sol)\s*(dominance|season)', post_text_lower):
            patterns_boost += 0.3
            
        # Price charts or analysis mention
        if re.search(r'(price|chart|technical)\s*(analysis|pattern|level)', post_text_lower):
            patterns_boost += 0.25
            
        # Final score with boost, capped at 1.0
        final_score = min(1.0, keyword_score + patterns_boost)
        
        return final_score
    
    def extract_market_topics(self, post_text: str) -> Dict[str, float]:
        """
        Extract market topics and their relevance from post text
        
        Args:
            post_text: Text content of the post
            
        Returns:
            Dictionary mapping topics to relevance scores (0-1)
        """
        if not post_text or not self.is_market_related(post_text):
            return {}
            
        post_text_lower = post_text.lower()
        topics = {}
        
        # Check each topic category
        for category, keywords in self.market_topic_categories.items():
            # Count how many keywords from this category appear in the text
            matches = sum(1 for keyword in keywords if keyword in post_text_lower)
            if matches > 0:
                # Calculate a relevance score based on number of matches
                # More matches = higher relevance
                relevance = min(1.0, matches / (len(keywords) * 0.3))
                topics[category] = relevance
                
                # Update trending topics counter
                self.trending_topics[category] += 1
        
        return topics
    
    def extract_mentioned_tokens(self, post_text: str) -> List[str]:
        """
        Extract mentioned cryptocurrency tokens from post
        
        Args:
            post_text: Text content of the post
            
        Returns:
            List of token symbols mentioned in the post
        """
        if not post_text:
            return []
            
        post_text_lower = post_text.lower()
        mentioned_tokens = []
        
        # Look for $ prefixed symbols first (stronger signal)
        for symbol in self.crypto_symbols:
            dollar_symbol = f'${symbol}'
            if dollar_symbol in post_text_lower:
                mentioned_tokens.append(symbol)
                
                # Update trending tokens counter
                self.trending_tokens[symbol] += 1
        
        # Then look for direct mentions of symbols (case-insensitive word boundary match)
        words = self._tokenize_text(post_text_lower)
        for symbol in self.crypto_symbols:
            if symbol in words:
                if symbol not in mentioned_tokens:  # Avoid duplicates
                    mentioned_tokens.append(symbol)
                    
                    # Update trending tokens counter
                    self.trending_tokens[symbol] += 1
        
        return mentioned_tokens
    
    def analyze_market_sentiment(self, post_text: str) -> Dict[str, Any]:
        """
        Analyze market sentiment (bullish/bearish) of a post
        
        Args:
            post_text: Text content of the post
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not post_text:
            return {'sentiment': 'neutral', 'score': 0.0}
            
        post_text_lower = post_text.lower()
        words = self._tokenize_text(post_text_lower)
        
        # Count bullish and bearish words
        bullish_count = sum(1 for word in words if word in self.bullish_words)
        bearish_count = sum(1 for word in words if word in self.bearish_words)
        
        # Check for negation
        negation_words = {'not', 'no', 'never', 'none', 'nothing', 'neither', 'nor', 'without'}
        negation_count = sum(1 for word in words if word in negation_words)
        
        # Simple detection of negation proximity to sentiment words
        # This is a very basic approach
        for i, word in enumerate(words):
            # Check for nearby negation of bullish words
            if word in self.bullish_words:
                # Check if a negation word appears right before the bullish word
                if i > 0 and words[i-1] in negation_words:
                    bullish_count -= 2  # Reduce bullish count (or even make negative)
                    bearish_count += 1  # Negated bullish can be bearish
                    
            # Check for nearby negation of bearish words
            if word in self.bearish_words:
                # Check if a negation word appears right before the bearish word
                if i > 0 and words[i-1] in negation_words:
                    bearish_count -= 2  # Reduce bearish count
                    bullish_count += 1  # Negated bearish can be bullish
        
        # Calculate a sentiment score (-1 to 1 scale)
        total_sentiment_words = bullish_count + bearish_count
        
        if total_sentiment_words == 0:
            sentiment_score = 0.0  # Neutral
        else:
            sentiment_score = (bullish_count - bearish_count) / total_sentiment_words
        
        # Determine sentiment label
        if sentiment_score > 0.3:
            sentiment = 'bullish'
        elif sentiment_score < -0.3:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
            
        # Extract price prediction if present
        price_prediction = self._extract_price_prediction(post_text)
        
        return {
            'sentiment': sentiment,
            'score': sentiment_score,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'prediction': price_prediction
        }
    
    def is_market_question(self, post_text: str) -> bool:
        """
        Determine if post contains a market-related question
        
        Args:
            post_text: Text content of the post
            
        Returns:
            True if post contains a market question, False otherwise
        """
        if not post_text:
            return False
            
        # Check if the post is market-related first
        if not self.is_market_related(post_text):
            return False
            
        post_text_lower = post_text.lower()
        
        # Check if post ends with a question mark
        if post_text.strip().endswith('?'):
            return True
            
        # Check for question patterns
        for pattern in self.market_question_patterns:
            if re.search(pattern, post_text_lower):
                return True
                
        # Check for common question words combined with market keywords
        question_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which', 'whom', 'whose'}
        words = self._tokenize_text(post_text_lower)
        
        if any(word in question_words for word in words) and any(word in self.all_market_keywords for word in words):
            return True
            
        return False
    
    def _extract_price_prediction(self, post_text: str) -> Optional[Dict[str, Any]]:
        """
        Extract price predictions from post text
        
        Args:
            post_text: Text content of the post
            
        Returns:
            Dictionary with prediction details or None if no prediction found
        """
        if not post_text:
            return None
            
        post_text_lower = post_text.lower()
        
        # Look for price targets with currency symbols
        price_targets = re.findall(r'\$(\d+[,\d]*(\.\d+)?)(K|M|B)?', post_text)
        if not price_targets:
            return None
            
        # Look for token mentions
        tokens = self.extract_mentioned_tokens(post_text)
        token = tokens[0] if tokens else None
        
        # Look for timeframes
        timeframe_patterns = {
            'short_term': r'(today|tonight|tomorrow|this week)',
            'mid_term': r'(this month|next month|soon|coming weeks)',
            'long_term': r'(this year|next year|long term|eventually)'
        }
        
        timeframe = None
        for tf, pattern in timeframe_patterns.items():
            if re.search(pattern, post_text_lower):
                timeframe = tf
                break
        
        # Format any K, M, B suffixes
        value = price_targets[0][0].replace(',', '')
        suffix = price_targets[0][2]
        
        if suffix == 'K':
            value = float(value) * 1_000
        elif suffix == 'M':
            value = float(value) * 1_000_000
        elif suffix == 'B':
            value = float(value) * 1_000_000_000
        else:
            value = float(value)
        
        return {
            'token': token,
            'value': value,
            'timeframe': timeframe
        }
    
    def get_trending_market_topics(self, min_count: int = 3, max_topics: int = 5) -> List[Tuple[str, int]]:
        """
        Get currently trending market topics based on analyzed posts
        
        Args:
            min_count: Minimum mention count to be considered trending
            max_topics: Maximum number of topics to return
            
        Returns:
            List of (topic, count) tuples sorted by count
        """
        # Reset trending counter if it's been too long
        if (datetime.now() - self.trending_reset_time).total_seconds() > (self.trending_reset_hours * 3600):
            self.trending_topics = Counter()
            self.trending_tokens = Counter()
            self.trending_reset_time = datetime.now()
            
        # Get topics with at least min_count mentions
        trending = [(topic, count) for topic, count in self.trending_topics.most_common()
                   if count >= min_count]
        
        return trending[:max_topics]
    
    def get_trending_tokens(self, min_count: int = 2, max_tokens: int = 5) -> List[Tuple[str, int]]:
        """
        Get currently trending tokens based on analyzed posts
        
        Args:
            min_count: Minimum mention count to be considered trending
            max_tokens: Maximum number of tokens to return
            
        Returns:
            List of (token, count) tuples sorted by count
        """
        # Reset trending counter if it's been too long
        if (datetime.now() - self.trending_reset_time).total_seconds() > (self.trending_reset_hours * 3600):
            self.trending_topics = Counter()
            self.trending_tokens = Counter()
            self.trending_reset_time = datetime.now()
            
        # Get tokens with at least min_count mentions
        trending = [(token, count) for token, count in self.trending_tokens.most_common()
                   if count >= min_count]
        
        return trending[:max_tokens]
    
    def analyze_post(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive market-related analysis on a post
        
        Args:
            post: Post data dictionary with at least a 'text' field
            
        Returns:
            Dictionary with analysis results
        """
        post_text = post.get('text', '')
        if not post_text:
            return {'is_market_related': False}
            
        # Check if market-related
        is_market_related = self.is_market_related(post_text)
        
        if not is_market_related:
            return {'is_market_related': False}
            
        # Get detailed market relevance score
        market_relevance = self.get_market_relevance_score(post_text)
        
        # Extract market topics
        topics = self.extract_market_topics(post_text)
        
        # Extract mentioned tokens
        tokens = self.extract_mentioned_tokens(post_text)
        
        # Analyze sentiment
        sentiment = self.analyze_market_sentiment(post_text)
        
        # Check if it's a question
        is_question = self.is_market_question(post_text)
        
        # Combine all analyses
        analysis = {
            'is_market_related': is_market_related,
            'market_relevance': market_relevance,
            'topics': topics,
            'mentioned_tokens': tokens,
            'sentiment': sentiment,
            'is_question': is_question,
            'analyzed_at': datetime.now()
        }
        
        # Store analysis in DB if available
        if self.db:
            self._store_analysis(post, analysis)
            
        return analysis
    
    def _store_analysis(self, post: Dict[str, Any], analysis: Dict[str, Any]) -> None:
        """
        Store analysis results in database
        
        Args:
            post: Original post data
            analysis: Analysis results
        """
        if not self.db:
            return
            
        try:
            self.db.store_content_analysis(
                post_id=post.get('post_id', 'unknown'),
                post_url=post.get('post_url', ''),
                post_author=post.get('author_handle', ''),
                post_text=post.get('text', ''),
                analysis_data=analysis
            )
        except Exception as e:
            logger.log_error("Content Analysis Storage", str(e))
    
    def analyze_multiple_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze multiple posts and return results
        
        Args:
            posts: List of post data dictionaries
            
        Returns:
            List of posts with analysis results added
        """
        analyzed_posts = []
        
        for post in posts:
            analysis = self.analyze_post(post)
            
            # Add analysis results to post data
            post_with_analysis = {**post, 'analysis': analysis}
            analyzed_posts.append(post_with_analysis)
            
        return analyzed_posts
    
    def find_market_related_posts(self, posts: List[Dict[str, Any]], min_relevance: float = 0.3) -> List[Dict[str, Any]]:
        """
        Find market-related posts from a list
        
        Args:
            posts: List of post data dictionaries
            min_relevance: Minimum market relevance score (0-1)
            
        Returns:
            List of market-related posts with analysis
        """
        market_posts = []
        
        for post in posts:
            analysis = self.analyze_post(post)
            
            # Check if it meets the minimum relevance criteria
            if analysis.get('is_market_related', False) and analysis.get('market_relevance', 0) >= min_relevance:
                post_with_analysis = {**post, 'analysis': analysis}
                market_posts.append(post_with_analysis)
                
        return market_posts
    
    def find_posts_mentioning_token(self, posts: List[Dict[str, Any]], token: str) -> List[Dict[str, Any]]:
        """
        Find posts mentioning a specific token
        
        Args:
            posts: List of post data dictionaries
            token: Token symbol to look for (e.g., 'btc')
            
        Returns:
            List of posts mentioning the token with analysis
        """
        token = token.lower()
        matching_posts = []
        
        for post in posts:
            post_text = post.get('text', '').lower()
            
            # Check for direct mentions or with $ prefix
            if token in self._tokenize_text(post_text) or f'${token}' in post_text:
                analysis = self.analyze_post(post)
                post_with_analysis = {**post, 'analysis': analysis}
                matching_posts.append(post_with_analysis)
                
        return matching_posts
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Split text into tokens/words
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Remove punctuation except for $ (important for crypto symbols)
        punctuation = string.punctuation.replace('$', '')
        translator = str.maketrans('', '', punctuation)
        text = text.translate(translator)
        
        # Split into words
        words = text.split()
        
        # Remove $ from beginning of words when tokenizing
        words = [word[1:] if word.startswith('$') and len(word) > 1 else word for word in words]
        
        return words
