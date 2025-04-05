#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple, Set, Optional, Any
import re
import json
import time
from datetime import datetime
import string
from collections import Counter
import statistics
import random

from utils.logger import logger

class ContentAnalyzer:
    """Enhanced content analyzer focused on social engagement and replies"""
    
    def __init__(self, config=None, db=None):
        """Initialize the content analyzer with expanded detection capabilities"""
        self.config = config
        self.db = db
        
        # Initialize keyword sets optimized for conversation detection
        self._initialize_keywords()
        
        # Engagement and conversation metrics
        self.engagement_weights = {
            'reply_worthy': 2.0,      # Post deserves a reply
            'question': 1.5,          # Post contains a question
            'opinion': 1.2,           # Post expresses an opinion
            'market_related': 1.0,    # Post discusses markets/crypto
            'technical': 0.8,         # Post contains technical analysis
            'casual': 1.3,            # Post uses casual/conversational language
            'trending': 1.4           # Post discusses trending topics
        }
        
        # Conversation state tracking
        self.conversation_history = {}
        self.max_history_size = 1000
        self.conversation_expiry = 3600  # 1 hour in seconds
        
        # Reply opportunity scoring thresholds
        self.reply_thresholds = {
            'high_value': 0.8,    # Definitely reply
            'medium_value': 0.6,  # Consider replying
            'low_value': 0.4      # Reply if not busy
        }
        
        # Tone analysis weights
        self.tone_weights = {
            'enthusiastic': 1.2,
            'skeptical': 1.1,
            'curious': 1.3,
            'informative': 1.0,
            'humorous': 1.1,
            'negative': 0.7
        }
        
        # Initialize tracking for trending topics and tokens
        self.trending = {
            'topics': Counter(),
            'tokens': Counter(),
            'phrases': Counter(),
            'sentiment': Counter()
        }
        self.trending_reset_time = datetime.now()
        self.trending_reset_hours = 4
        
        # Tracking for conversation state
        self.active_discussions = {}
        self.discussion_expiry = 1800  # 30 minutes
        
        # Recent mentions tracking
        self.recent_mentions = []
        self.max_recent_mentions = 100
        
        logger.logger.info("Enhanced content analyzer initialized")
    
    def _initialize_keywords(self) -> None:
        """Initialize expanded keyword sets for better conversation detection"""
        
        # Core crypto terms (including casual variants)
        self.crypto_terms = {
            # Traditional terms
            'bitcoin', 'btc', 'ethereum', 'eth', 'blockchain', 'crypto', 'token',
            'altcoin', 'defi', 'nft', 'mining', 'wallet', 'exchange', 'hodl',
            
            # Casual/Slang terms
            'moon', 'mooning', 'wen', 'ser', 'lfg', 'wagmi', 'ngmi', 'fud',
            'alpha', 'beta', 'degen', 'aped', 'aping', 'rekt', 'bagholder',
            'paper hands', 'diamond hands', 'dyor', 'fomo', 'ponzi', 'shill',
            
            # Common abbreviations
            'gm', 'gn', 'idk', 'tbh', 'imo', 'imho', 'nfa', 'dca', 'ta',
            
            # Sentiment indicators
            'bullish', 'bearish', 'neutral', 'optimistic', 'concerned',
            'worried', 'excited', 'cautious', 'confident', 'uncertain'
        }
        
        # Conversation starters/hooks
        self.conversation_hooks = {
            # Questions
            'what', 'how', 'why', 'when', 'where', 'who', 'thoughts',
            'anyone', 'somebody', 'thinking', 'reckon', 'believe',
            
            # Opinions
            'think', 'feel', 'seems', 'looks', 'appears', 'might',
            'maybe', 'perhaps', 'probably', 'definitely', 'surely',
            
            # Engagement
            'agree', 'disagree', 'right', 'wrong', 'true', 'false',
            'good', 'bad', 'better', 'worse', 'best', 'worst',
            
            # Community
            'team', 'community', 'fam', 'friends', 'people', 'everyone',
            'anybody', 'someone', 'guys', 'folks', 'fren', 'anon'
        }
        
        # Trading/Market discussion (casual)
        self.market_casual = {
            # Actions
            'buy', 'bought', 'buying', 'sell', 'sold', 'selling', 'hold',
            'holding', 'trade', 'trading', 'stake', 'staking', 'farm',
            'farming', 'mine', 'mining',
            
            # Status
            'pump', 'dump', 'moon', 'dip', 'correction', 'crash', 'rally',
            'boom', 'bust', 'rip', 'fire', 'heat', 'cold', 'hot', 'dead',
            
            # Feelings
            'hope', 'wish', 'want', 'need', 'fear', 'greed', 'fomo',
            'regret', 'happy', 'sad', 'angry', 'excited', 'worried',
            
            # Time
            'soon', 'now', 'later', 'never', 'always', 'today', 'tomorrow',
            'yesterday', 'week', 'month', 'year', 'eventually'
        }
        
        # Technical Discussion (keeping for analysis)
        self.technical_terms = {
            # Analysis
            'resistance', 'support', 'trend', 'pattern', 'formation',
            'indicator', 'signal', 'volume', 'momentum', 'volatility',
            
            # Metrics
            'marketcap', 'supply', 'circulation', 'liquidity', 'depth',
            'orderbook', 'spread', 'premium', 'discount', 'ratio',
            
            # Chart Patterns
            'triangle', 'wedge', 'channel', 'fibonacci', 'retrace',
            'breakout', 'breakdown', 'consolidation', 'accumulation'
        }
        
        # Meme-specific language
        self.meme_phrases = {
            # Classic memes
            'to the moon', 'wen lambo', 'this is the way', 'sir this is a wendys',
            'ngmi', 'wagmi', 'few understand', 'probably nothing', 'gm',
            
            # Variations
            'wen moon', 'ser', 'fren', 'anon', 'based', 'chad', 'wojak',
            'feels good man', 'pepe', 'ape', 'degen', 'alpha', 'beta',
            
            # Emoji substitutes (text)
            ':rocket:', ':moon:', ':gem:', ':diamond:', ':hands:', ':fire:',
            ':chart:', ':stonks:', ':up:', ':down:', ':bear:', ':bull:'
        }
        
        # Opinion indicators
        self.opinion_markers = {
            # Direct markers
            'i think', 'imo', 'imho', 'in my opinion', 'personally',
            'my thoughts', 'i believe', 'i feel', 'i reckon',
            
            # Indirect markers
            'seems like', 'looks like', 'appears to be', 'probably',
            'might be', 'could be', 'maybe', 'perhaps', 'possibly',
            
            # Strong opinions
            'definitely', 'surely', 'certainly', 'absolutely', 'no doubt',
            'must be', 'has to be', 'clearly', 'obviously', 'without question'
        }

        # Question patterns (expanded for natural language)
        self.question_patterns = [
            # Direct questions about price/market
            r'(what|how|why|when).*(price|market|going|expect)',
            r'(should|could|would).*(buy|sell|trade|hold)',
            r'(is|are|will).*(good|bad|worth|pump|dump)',
            
            # Opinion seeking
            r'(what).*(think|thought|opinion|take)',
            r'(any|your|others).*(thoughts|views|ideas)',
            r'(who|anyone).*(buying|selling|trading)',
            
            # Technical questions
            r'(what|where).*(support|resistance|trend)',
            r'(how|what).*(analysis|pattern|signal)',
            
            # General engagement
            r'(right|wrong|true|false)\?',
            r'(agree|disagree)\?',
            r'(good|bad|better|worse)\?'
        ]
        
        # Combine all terms for quick checking
        self.all_terms = self.crypto_terms.union(
            self.conversation_hooks,
            self.market_casual,
            self.technical_terms
        )

    def analyze_post(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced post analysis optimized for conversation and engagement"""
        try:
            post_text = post.get('text', '')
            if not post_text:
                return {'reply_worthy': False}

            # Basic content features
            features = {
                'has_question': self._detect_questions(post_text),
                'has_opinion': self._detect_opinions(post_text),
                'crypto_terms': self._extract_crypto_terms(post_text),
                'meme_phrases': self._detect_meme_phrases(post_text),
                'conversation_hooks': self._detect_conversation_hooks(post_text),
                'tone': self._analyze_tone(post_text),
                'sentiment': self._analyze_sentiment(post_text)
            }

            # Calculate engagement scores
            engagement_scores = {
                'question_score': 1.5 if features['has_question'] else 1.0,
                'opinion_score': 1.2 if features['has_opinion'] else 1.0,
                'crypto_relevance': len(features['crypto_terms']) * 0.15,
                'meme_score': len(features['meme_phrases']) * 0.1,
                'hook_score': len(features['conversation_hooks']) * 0.2,
                'tone_multiplier': self.tone_weights.get(features['tone'], 1.0)
            }

            # Conversation state features
            state_features = self._analyze_conversation_state(post)
            
            # Calculate initial reply score
            base_reply_score = sum(engagement_scores.values()) * state_features['activity_multiplier']
            
            # Apply contextual modifiers
            context_score = self._calculate_context_score(post, features)
            final_reply_score = base_reply_score * context_score
            
            # Determine reply worthiness
            reply_worthy = final_reply_score >= self.reply_thresholds['medium_value']
            high_value = final_reply_score >= self.reply_thresholds['high_value']

            # Build response focus
            response_focus = self._determine_response_focus(features, engagement_scores)

            # Update trending tracking
            self._update_trending(features)

            # Compile analysis results
            analysis = {
                'reply_worthy': reply_worthy,
                'high_value': high_value,
                'reply_score': final_reply_score,
                'features': features,
                'engagement_scores': engagement_scores,
                'state': state_features,
                'response_focus': response_focus,
                'analyzed_at': datetime.now()
            }

            # Store analysis if database available
            if self.db:
                self._store_analysis(post['post_id'], analysis)

            return analysis

        except Exception as e:
            logger.log_error("Post Analysis", str(e))
            return {'reply_worthy': False, 'error': str(e)}

    def _detect_questions(self, text: str) -> Dict[str, Any]:
        """Enhanced question detection with type classification"""
        questions = {
            'direct': False,          # Direct questions needing answers
            'rhetorical': False,      # Rhetorical questions
            'opinion_seeking': False, # Seeking thoughts/opinions
            'technical': False,       # Technical/analytical questions
            'count': 0               # Total questions found
        }
        
        text_lower = text.lower()
        
        # Direct question detection
        if '?' in text:
            questions['count'] += 1
            
            # Check for opinion seeking
            if any(phrase in text_lower for phrase in [
                'what do you think', 'thoughts on', 'what about',
                'agree', 'opinion on', 'what are your'
            ]):
                questions['opinion_seeking'] = True
                
            # Check for technical questions
            elif any(term in text_lower for term in self.technical_terms):
                questions['technical'] = True
                
            # Check for rhetorical questions
            elif any(phrase in text_lower for phrase in [
                'right?', 'amirite', 'isn\'t it', 'don\'t you',
                'you know what', 'guess what', 'who else'
            ]):
                questions['rhetorical'] = True
            
            # Must be a direct question
            else:
                questions['direct'] = True

        # Question-like patterns without question marks
        for pattern in self.question_patterns:
            if re.search(pattern, text_lower):
                questions['count'] += 1
                
                # Classify based on pattern
                if 'think' in pattern or 'opinion' in pattern:
                    questions['opinion_seeking'] = True
                elif 'analysis' in pattern or 'pattern' in pattern:
                    questions['technical'] = True
                else:
                    questions['direct'] = True

        return questions

    def _detect_opinions(self, text: str) -> Dict[str, Any]:
        """Detect and classify opinion expressions"""
        opinions = {
            'has_opinion': False,
            'strength': 'neutral',  # weak, moderate, strong
            'type': None,          # prediction, analysis, reaction
            'markers': []
        }
        
        text_lower = text.lower()
        
        # Check for opinion markers
        for marker in self.opinion_markers:
            if marker in text_lower:
                opinions['has_opinion'] = True
                opinions['markers'].append(marker)
        
        # Determine opinion strength
        strong_indicators = ['definitely', 'absolutely', 'certainly', 'no doubt']
        moderate_indicators = ['probably', 'likely', 'seems', 'appears']
        weak_indicators = ['maybe', 'might', 'could', 'possibly']
        
        if any(indicator in text_lower for indicator in strong_indicators):
            opinions['strength'] = 'strong'
        elif any(indicator in text_lower for indicator in moderate_indicators):
            opinions['strength'] = 'moderate'
        elif any(indicator in text_lower for indicator in weak_indicators):
            opinions['strength'] = 'weak'
        
        # Classify opinion type
        if any(term in text_lower for term in ['will', 'going to', 'expect', 'predict']):
            opinions['type'] = 'prediction'
        elif any(term in text_lower for term in ['think', 'believe', 'feel']):
            opinions['type'] = 'reaction'
        elif any(term in text_lower for term in self.technical_terms):
            opinions['type'] = 'analysis'
        
        return opinions

    def _extract_crypto_terms(self, text: str) -> List[str]:
        """Extract cryptocurrency-related terms from text"""
        text_lower = text.lower()
        words = set(text_lower.split())
        
        # Extract terms from different categories
        found_terms = {
            'crypto': [term for term in self.crypto_terms if term in words],
            'trading': [term for term in self.market_casual if term in words],
            'technical': [term for term in self.technical_terms if term in words]
        }
        
        # Check for dollar amounts and percentages
        dollar_amounts = re.findall(r'\$\d+(?:,\d+)*(?:\.\d+)?[KkMmBb]?', text)
        percentages = re.findall(r'-?\d+(?:\.\d+)?%', text)
        
        if dollar_amounts:
            found_terms['amounts'] = dollar_amounts
        if percentages:
            found_terms['percentages'] = percentages
            
        # Look for token symbols (with and without $)
        symbols = re.findall(r'\$?[A-Z]{2,5}\b', text)
        if symbols:
            found_terms['symbols'] = symbols
            
        return found_terms

    def _detect_meme_phrases(self, text: str) -> List[str]:
        """Detect meme phrases and cultural references"""
        text_lower = text.lower()
        found_memes = []
        
        # Check for exact meme phrases
        for meme in self.meme_phrases:
            if meme in text_lower:
                found_memes.append(meme)
        
        # Check for meme variations
        variations = {
            'moon': ['mooning', 'moonish', 'moonboi', 'moonshot'],
            'wen': ['wen moon', 'wen lambo', 'wen pump'],
            'ser': ['ser pls', 'yes ser', 'ok ser'],
            'fren': ['frens', 'fren group', 'crypto frens'],
            'wagmi': ['wagmi fam', 'pure wagmi', 'mega wagmi']
        }
        
        for base, vars in variations.items():
            if any(var in text_lower for var in vars):
                found_memes.append(base)
        
        # Check for emoji patterns
        emoji_patterns = [
            r'🚀+',      # Rocket spam
            r'💎\s?🙌',  # Diamond hands
            r'🌕|🌖|🌗|🌘', # Moon phases
            r'🐻|🐂',     # Bear/Bull
            r'📈|📉',     # Charts
            r'🔥|💯'      # Fire/100
        ]
        
        for pattern in emoji_patterns:
            if re.search(pattern, text):
                found_memes.append(f"emoji_{pattern}")
                
        return list(set(found_memes))  # Remove duplicates

    def _analyze_conversation_state(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze conversation context and state"""
        post_id = post.get('post_id', '')
        author_id = post.get('author_id', '')
        
        state = {
            'activity_multiplier': 1.0,
            'is_thread': False,
            'needs_quick_reply': False,
            'conversation_age': None,
            'previous_interactions': 0
        }
        
        try:
            # Check if post is part of ongoing thread
            if post.get('parent_id'):
                state['is_thread'] = True
                # Get thread history
                thread_history = self._get_thread_history(post)
                
                if thread_history:
                    # Calculate thread age
                    oldest_post = min(p['timestamp'] for p in thread_history)
                    state['conversation_age'] = (datetime.now() - oldest_post).total_seconds()
                    
                    # Count our previous interactions
                    state['previous_interactions'] = sum(
                        1 for p in thread_history 
                        if p.get('is_our_reply', False)
                    )
                    
                    # Check if quick reply needed (recent active discussion)
                    if state['conversation_age'] < 1800:  # 30 minutes
                        state['needs_quick_reply'] = True
                        state['activity_multiplier'] = 1.2
            
            # Check author interaction history
            author_history = self._get_author_history(author_id)
            if author_history:
                # Adjust activity multiplier based on previous interactions
                recent_interactions = sum(
                    1 for p in author_history
                    if (datetime.now() - p['timestamp']).total_seconds() < 86400  # 24 hours
                )
                state['activity_multiplier'] *= (1 + (recent_interactions * 0.1))
            
            # Update conversation tracking
            self._update_conversation_tracking(post)
            
            return state
            
        except Exception as e:
            logger.log_error("Conversation State Analysis", str(e))
            return state

    def _get_thread_history(self, post: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get conversation thread history"""
        try:
            thread_id = post.get('thread_id') or post.get('parent_id')
            if not thread_id or not self.db:
                return []
                
            # Query thread history from database
            history = self.db.get_thread_history(thread_id)
            
            # Sort by timestamp
            if history:
                history.sort(key=lambda x: x.get('timestamp', datetime.min))
                
            return history
            
        except Exception as e:
            logger.log_error("Thread History Retrieval", str(e))
            return []

    def _get_author_history(self, author_id: str) -> List[Dict[str, Any]]:
        """Get author interaction history"""
        try:
            if not author_id or not self.db:
                return []
                
            # Get recent author posts and our replies
            history = self.db.get_author_interaction_history(author_id, hours=24)
            
            return history
            
        except Exception as e:
            logger.log_error("Author History Retrieval", str(e))
            return []

    def _update_conversation_tracking(self, post: Dict[str, Any]) -> None:
        """Update active conversation tracking"""
        try:
            now = datetime.now()
            post_id = post.get('post_id', '')
            thread_id = post.get('thread_id') or post.get('parent_id')
            
            # Clean expired conversations
            expired = []
            for conv_id, data in self.active_discussions.items():
                if (now - data['last_update']).total_seconds() > self.discussion_expiry:
                    expired.append(conv_id)
            
            for conv_id in expired:
                del self.active_discussions[conv_id]
            
            # Update or add conversation
            if thread_id:
                if thread_id in self.active_discussions:
                    self.active_discussions[thread_id].update({
                        'last_update': now,
                        'post_count': self.active_discussions[thread_id]['post_count'] + 1,
                        'latest_post_id': post_id
                    })
                else:
                    self.active_discussions[thread_id] = {
                        'start_time': now,
                        'last_update': now,
                        'post_count': 1,
                        'latest_post_id': post_id,
                        'our_replies': 0
                    }
                    
        except Exception as e:
            logger.log_error("Conversation Tracking Update", str(e))

    def _calculate_context_score(self, post: Dict[str, Any], features: Dict[str, Any]) -> float:
        """Calculate contextual relevance score"""
        try:
            score = 1.0  # Base score
            
            # Time of day adjustment
            hour = datetime.now().hour
            if 8 <= hour <= 23:  # Active hours
                score *= 1.1
            elif 0 <= hour <= 7:  # Less active hours
                score *= 0.9
                
            # Content quality factors
            if features['has_question']:
                score *= 1.2
            if features['has_opinion']:
                score *= 1.1
            if len(features['crypto_terms']) > 2:
                score *= 1.15
            if features['tone'] in ['enthusiastic', 'curious']:
                score *= 1.1
                
            # Conversation state factors
            if post.get('thread_id'):
                # Part of a thread
                if post.get('reply_count', 0) > 0:
                    # Active discussion
                    score *= 1.2
                if post.get('is_thread_starter', False):
                    # Original post in thread
                    score *= 1.1
                    
            # Author factors
            if post.get('author_follower_count', 0) > 1000:
                score *= 1.1
            if post.get('author_is_verified', False):
                score *= 1.05
                
            # Trending topic bonus
            mentioned_topics = set(features.get('crypto_terms', []))
            trending_topics = set(k for k, v in self.trending['topics'].most_common(5))
            if mentioned_topics & trending_topics:
                score *= 1.2
                
            return min(2.0, score)  # Cap at 2.0x
            
        except Exception as e:
            logger.log_error("Context Score Calculation", str(e))
            return 1.0

    def _determine_response_focus(self, features: Dict[str, Any], 
                                engagement_scores: Dict[str, float]) -> Dict[str, Any]:
        """Determine optimal response focus and approach"""
        try:
            focus = {
                'primary_type': None,     # Main response type
                'secondary_type': None,   # Secondary response type
                'key_points': [],         # Key points to address
                'style': 'neutral',       # Response style
                'length': 'medium'        # Response length
            }
            
            # Determine primary response type
            if features['has_question']:
                focus['primary_type'] = 'answer'
                if features.get('questions', {}).get('technical', False):
                    focus['secondary_type'] = 'technical'
                elif features.get('questions', {}).get('opinion_seeking', False):
                    focus['secondary_type'] = 'opinion'
            elif features['has_opinion']:
                focus['primary_type'] = 'engage'
                if features.get('opinions', {}).get('type') == 'prediction':
                    focus['secondary_type'] = 'prediction'
                elif features.get('opinions', {}).get('type') == 'analysis':
                    focus['secondary_type'] = 'analysis'
            elif len(features['crypto_terms']) > 0:
                focus['primary_type'] = 'inform'
                
            # Determine style based on tone and content
            if features['tone'] in ['enthusiastic', 'humorous']:
                focus['style'] = 'casual'
            elif features['tone'] in ['skeptical', 'negative']:
                focus['style'] = 'balanced'
            elif features.get('meme_phrases', []):
                focus['style'] = 'playful'
                
            # Determine length based on complexity
            if features.get('questions', {}).get('technical', False):
                focus['length'] = 'long'
            elif len(features['crypto_terms']) > 3:
                focus['length'] = 'medium'
            else:
                focus['length'] = 'short'
                
            # Identify key points to address
            if focus['primary_type'] == 'answer':
                focus['key_points'] = self._extract_question_topics(features)
            elif focus['primary_type'] == 'engage':
                focus['key_points'] = self._extract_opinion_points(features)
            else:
                focus['key_points'] = list(features['crypto_terms'])[:3]
                
            return focus
            
        except Exception as e:
            logger.log_error("Response Focus Determination", str(e))
            return {'primary_type': 'neutral', 'style': 'neutral', 'length': 'medium'}

    def _update_trending(self, features: Dict[str, Any]) -> None:
        """Update trending topics and engagement tracking"""
        try:
            now = datetime.now()
            
            # Reset trending data if expired
            if (now - self.trending_reset_time).total_seconds() > (self.trending_reset_hours * 3600):
                self.trending = {
                    'topics': Counter(),
                    'tokens': Counter(),
                    'phrases': Counter(),
                    'sentiment': Counter()
                }
                self.trending_reset_time = now
            
            # Update topic tracking
            for term in features.get('crypto_terms', []):
                self.trending['topics'][term] += 1
                
            # Update token tracking
            for token in features.get('crypto_terms', {}).get('symbols', []):
                self.trending['tokens'][token] += 1
                
            # Update phrase tracking
            for phrase in features.get('meme_phrases', []):
                self.trending['phrases'][phrase] += 1
                
            # Update sentiment tracking
            sentiment = features.get('sentiment', 'neutral')
            self.trending['sentiment'][sentiment] += 1
            
            # Trim to keep only most frequent items
            for category in self.trending:
                self.trending[category] = Counter(
                    dict(self.trending[category].most_common(50))
                )
                
        except Exception as e:
            logger.log_error("Trending Update", str(e))

    def _extract_question_topics(self, features: Dict[str, Any]) -> List[str]:
        """Extract main topics from questions"""
        topics = []
        try:
            text = features.get('text', '').lower()
            
            # Extract topics based on question type
            if features.get('questions', {}).get('technical', False):
                # Look for technical terms
                topics.extend([
                    term for term in self.technical_terms
                    if term in text
                ])
                
            elif features.get('questions', {}).get('opinion_seeking', False):
                # Look for opinion subjects
                topics.extend([
                    term for term in self.market_casual
                    if term in text
                ])
                
            # Add any mentioned crypto terms
            crypto_terms = features.get('crypto_terms', {})
            if 'symbols' in crypto_terms:
                topics.extend(crypto_terms['symbols'])
            
            # Identify price-related topics
            price_patterns = [
                r'\$\d+(?:,\d+)*(?:\.\d+)?[KkMmBb]?',
                r'-?\d+(?:\.\d+)?%',
                r'(price|value|worth)'
            ]
            
            for pattern in price_patterns:
                if re.search(pattern, text):
                    topics.append('price_discussion')
                    break
                    
            # Remove duplicates while preserving order
            return list(dict.fromkeys(topics))
            
        except Exception as e:
            logger.log_error("Question Topic Extraction", str(e))
            return topics

    def _extract_opinion_points(self, features: Dict[str, Any]) -> List[str]:
        """Extract main points from opinions"""
        points = []
        try:
            text = features.get('text', '').lower()
            
            # Get opinion type and strength
            opinion_type = features.get('opinions', {}).get('type')
            opinion_strength = features.get('opinions', {}).get('strength', 'neutral')
            
            if opinion_type == 'prediction':
                # Look for timeframes
                timeframes = [
                    'today', 'tomorrow', 'week', 'month', 'year',
                    'soon', 'later', 'eventually'
                ]
                for timeframe in timeframes:
                    if timeframe in text:
                        points.append(f"timeframe_{timeframe}")
                        
                # Look for price targets
                price_matches = re.findall(
                    r'\$\d+(?:,\d+)*(?:\.\d+)?[KkMmBb]?', 
                    text
                )
                if price_matches:
                    points.append('price_target')
                    
            elif opinion_type == 'analysis':
                # Extract technical points
                for term in self.technical_terms:
                    if term in text:
                        points.append(term)
                        
            # Add sentiment if strong
            if opinion_strength in ['strong', 'moderate']:
                sentiment = features.get('sentiment')
                if sentiment:
                    points.append(f"sentiment_{sentiment}")
                    
            # Add any specific crypto terms
            crypto_terms = features.get('crypto_terms', {})
            if 'symbols' in crypto_terms:
                points.extend(crypto_terms['symbols'])
                
            # Remove duplicates while preserving order
            return list(dict.fromkeys(points))
            
        except Exception as e:
            logger.log_error("Opinion Points Extraction", str(e))
            return points

    def get_trending_stats(self) -> Dict[str, Any]:
        """Get current trending statistics"""
        try:
            stats = {
                'topics': dict(self.trending['topics'].most_common(10)),
                'tokens': dict(self.trending['tokens'].most_common(5)),
                'phrases': dict(self.trending['phrases'].most_common(5)),
                'sentiment': dict(self.trending['sentiment'].most_common(3)),
                'last_reset': self.trending_reset_time,
                'hours_tracked': self.trending_reset_hours
            }
            
            # Calculate sentiment ratio
            total_sentiment = sum(self.trending['sentiment'].values())
            if total_sentiment > 0:
                stats['sentiment_ratio'] = {
                    sentiment: count / total_sentiment
                    for sentiment, count in self.trending['sentiment'].items()
                }
            
            # Get related topics
            stats['related_topics'] = self._find_related_topics()
            
            return stats
            
        except Exception as e:
            logger.log_error("Trending Stats Retrieval", str(e))
            return {}

    def _analyze_tone(self, text: str) -> str:
        """Analyze the conversational tone of text"""
        try:
            text_lower = text.lower()
            
            # Tone markers
            tone_indicators = {
                'enthusiastic': [
                    '!', '!!', 'wow', 'amazing', 'great', 'awesome',
                    'incredible', 'bullish', 'moon', 'gem', 'excited'
                ],
                'skeptical': [
                    'hmm', 'idk', 'not sure', 'doubt', 'suspicious',
                    'bearish', 'careful', 'warning', 'scam'
                ],
                'curious': [
                    '?', 'what if', 'wonder', 'curious', 'interesting',
                    'thoughts', 'opinion', 'anyone else'
                ],
                'informative': [
                    'fyi', 'note', 'remember', 'important', 'key',
                    'consider', 'analysis', 'research'
                ],
                'humorous': [
                    'lol', 'lmao', 'rofl', '😂', '🤣', 'kek',
                    'wen lambo', 'sir this is a'
                ],
                'negative': [
                    'bad', 'terrible', 'worst', 'hate', 'stupid',
                    'waste', 'avoid', 'stay away'
                ]
            }
            
            # Count tone indicators
            tone_counts = Counter()
            
            for tone, indicators in tone_indicators.items():
                count = sum(1 for indicator in indicators if indicator in text_lower)
                # Extra weight for multiple punctuation
                if tone == 'enthusiastic':
                    count += len(re.findall(r'!{2,}', text))
                elif tone == 'curious':
                    count += len(re.findall(r'\?{2,}', text))
                tone_counts[tone] = count
            
            # Get predominant tone
            if tone_counts:
                predominant_tone = tone_counts.most_common(1)[0][0]
            else:
                predominant_tone = 'neutral'
            
            return predominant_tone
            
        except Exception as e:
            logger.log_error("Tone Analysis", str(e))
            return 'neutral'

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Enhanced sentiment analysis for crypto discussions"""
        try:
            text_lower = text.lower()
            words = text_lower.split()
            
            # Initialize sentiment tracking
            sentiment = {
                'label': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'aspects': {}
            }
            
            # Sentiment word lists
            bullish_words = {
                'strong': ['moon', 'lambo', 'hodl', 'gem', 'bullish'],
                'medium': ['up', 'high', 'good', 'nice', 'great'],
                'weak': ['hope', 'maybe', 'possible', 'potential']
            }
            
            bearish_words = {
                'strong': ['dump', 'crash', 'rekt', 'scam', 'bearish'],
                'medium': ['down', 'low', 'bad', 'poor', 'weak'],
                'weak': ['risky', 'careful', 'cautious', 'worried']
            }
            
            # Count sentiment words
            bull_counts = {
                'strong': sum(1 for word in bullish_words['strong'] if word in words),
                'medium': sum(1 for word in bullish_words['medium'] if word in words),
                'weak': sum(1 for word in bullish_words['weak'] if word in words)
            }
            
            bear_counts = {
                'strong': sum(1 for word in bearish_words['strong'] if word in words),
                'medium': sum(1 for word in bearish_words['medium'] if word in words),
                'weak': sum(1 for word in bearish_words['weak'] if word in words)
            }
            
            # Calculate weighted sentiment score
            bull_score = (
                bull_counts['strong'] * 1.0 +
                bull_counts['medium'] * 0.6 +
                bull_counts['weak'] * 0.3
            )
            
            bear_score = (
                bear_counts['strong'] * 1.0 +
                bear_counts['medium'] * 0.6 +
                bear_counts['weak'] * 0.3
            )
            
            # Calculate net sentiment
            net_score = bull_score - bear_score
            
            # Determine confidence based on word counts
            total_sentiment_words = sum(bull_counts.values()) + sum(bear_counts.values())
            if total_sentiment_words > 0:
                confidence = min(1.0, total_sentiment_words / 5)  # Cap at 1.0
            else:
                confidence = 0.0
            
            # Assign sentiment label and score
            if net_score > 0.5:
                sentiment['label'] = 'bullish'
                sentiment['score'] = min(1.0, net_score / 3)
            elif net_score < -0.5:
                sentiment['label'] = 'bearish'
                sentiment['score'] = max(-1.0, net_score / 3)
            else:
                sentiment['label'] = 'neutral'
                sentiment['score'] = net_score
            
            sentiment['confidence'] = confidence
            
            return sentiment
            
        except Exception as e:
            logger.log_error("Sentiment Analysis", str(e))
            return {'label': 'neutral', 'score': 0.0, 'confidence': 0.0}

    def _store_analysis(self, post_id: str, analysis: Dict[str, Any]) -> None:
        """Store analysis results in database"""
        try:
            if not self.db or not post_id:
                return
                
            # Prepare data for storage
            storage_data = {
                'post_id': post_id,
                'reply_worthy': analysis['reply_worthy'],
                'reply_score': analysis['reply_score'],
                'features': json.dumps(analysis['features']),
                'engagement_scores': json.dumps(analysis['engagement_scores']),
                'response_focus': json.dumps(analysis['response_focus']),
                'timestamp': datetime.now()
            }
            
            # Store in database
            self.db.store_content_analysis(**storage_data)
            
        except Exception as e:
            logger.log_error("Analysis Storage", str(e))

    def _find_related_topics(self) -> Dict[str, List[str]]:
        """Find related topics based on co-occurrence"""
        try:
            related = {}
            
            # Get top topics
            top_topics = self.trending['topics'].most_common(10)
            
            for topic, _ in top_topics:
                # Find topics that frequently appear together
                co_occurrences = Counter()
                
                # Query database for co-occurring topics
                if self.db:
                    recent_posts = self.db.get_recent_posts_with_topic(topic, hours=24)
                    
                    for post in recent_posts:
                        post_topics = set(post.get('topics', []))
                        post_topics.discard(topic)  # Remove the current topic
                        for other_topic in post_topics:
                            co_occurrences[other_topic] += 1
                
                # Get top 3 related topics
                related[topic] = [
                    topic for topic, count in co_occurrences.most_common(3)
                ]
            
            return related
            
        except Exception as e:
            logger.log_error("Related Topics Analysis", str(e))
            return {}            
