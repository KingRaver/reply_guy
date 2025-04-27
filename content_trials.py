#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple, Set, Optional, Any
import re
import json
import time
from datetime import datetime, timezone
import string
from collections import Counter
import statistics
import random

from utils.logger import logger
from datetime_utils import ensure_naive_datetimes, strip_timezone, safe_datetime_diff

class ContentAnalyzer:
    """Enhanced content analyzer focused on social engagement and replies"""
    
    def __init__(self, config=None, db=None):
        """Initialize the content analyzer with expanded detection capabilities"""
        self.config = config
        self.db = db
        self.llm_provider = None  # Will be set if needed
        
        # Initialize keyword sets optimized for conversation detection
        self._initialize_keywords()
        
        # Initialize tech-related keywords
        self._initialize_tech_keywords()
        
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
        self.trending_reset_time = strip_timezone(datetime.now())
        self.trending_reset_hours = 4
        
        # Tracking for conversation state
        self.active_discussions = {}
        self.discussion_expiry = 1800  # 30 minutes
        
        # Recent mentions tracking
        self.recent_mentions = []
        self.max_recent_mentions = 100
        
        # Market-related keywords for detecting market posts
        self.market_keywords = {
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

    def _initialize_tech_keywords(self) -> None:
        """Initialize tech-related keyword sets for content analysis"""
        
        # AI specific terms
        self.ai_terms = {
            # General AI terms
            'ai', 'artificial intelligence', 'machine learning', 'ml', 'deep learning', 'neural network',
            'llm', 'large language model', 'transformer', 'gpt', 'llama', 'claude', 'bard', 'gemini',
            'chatbot', 'chatgpt', 'generative ai', 'genai', 'diffusion model', 'stable diffusion',
            'dalle', 'midjourney', 'agi', 'artificial general intelligence', 'multimodal',
            
            # Technical AI terms
            'attention mechanism', 'fine-tuning', 'prompt engineering', 'inference', 'training',
            'computer vision', 'nlp', 'natural language processing', 'reinforcement learning',
            'unsupervised learning', 'supervised learning', 'text-to-image', 'text-to-video',
            'embedding', 'vector database', 'parameter', 'billion parameters', 'trillion parameters',
            'transfer learning', 'foundation model', 'hallucination', 'rag', 'retrieval augmented',
            
            # AI companies and labs
            'openai', 'anthropic', 'google deepmind', 'meta ai', 'microsoft ai', 'nvidia',
            'hugging face', 'stability ai', 'cohere', 'databricks', 'inflection ai',
            
            # AI ethics and concerns
            'ai safety', 'ai alignment', 'ai regulation', 'ai bias', 'ai transparency', 
            'ai ethics', 'responsible ai', 'ai governance', 'superintelligence'
        }
        
        # Quantum computing terms
        self.quantum_terms = {
            # General quantum terms
            'quantum computing', 'quantum computer', 'quantum processor', 'quantum supremacy',
            'quantum advantage', 'quantum bit', 'qubit', 'quantum gate', 'quantum circuit',
            'quantum algorithm', 'quantum annealing', 'quantum simulation',
            
            # Technical quantum terms
            'superposition', 'entanglement', 'quantum decoherence', 'quantum error correction',
            'quantum volume', 'quantum logic gate', 'quantum fourier transform', 'shor\'s algorithm',
            'grover\'s algorithm', 'quantum phase estimation', 'quantum variational circuit',
            'bloch sphere', 'quantum state', 'quantum teleportation', 'quantum key distribution',
            
            # Quantum companies
            'ibm quantum', 'google quantum', 'rigetti', 'd-wave', 'ionq', 'quantum computing inc',
            'xanadu', 'pasqal', 'quantum brilliance', 'qaski', 'zapata computing',
            
            # Quantum applications
            'quantum cryptography', 'quantum encryption', 'quantum random', 'quantum machine learning',
            'quantum finance', 'quantum chemistry', 'quantum optimization', 'quantum internet',
            'post-quantum', 'quantum resistant', 'quantum secure'
        }
        
        # Blockchain technology terms (beyond trading)
        self.blockchain_tech_terms = {
            # Technical blockchain terms
            'consensus mechanism', 'proof of stake', 'proof of work', 'delegated proof of stake',
            'byzantine fault tolerance', 'merkle tree', 'hash function', 'public key cryptography',
            'zero-knowledge proof', 'zk-rollup', 'optimistic rollup', 'sidechains', 'state channels',
            'sharding', 'layer 2', 'interoperability', 'cross-chain', 'on-chain', 'off-chain',
            
            # Advanced blockchain concepts
            'smart contract', 'decentralized autonomous organization', 'dao', 'decentralized identity',
            'did', 'verifiable credential', 'self-sovereign identity', 'trustless', 'permissionless',
            'composability', 'tokenomics', 'token engineering', 'decentralized oracle',
            'blockchain oracle', 'state machine', 'virtual machine', 'evm',
            
            # Blockchain privacy
            'zero-knowledge', 'zk-snark', 'zk-stark', 'ring signature', 'confidential transaction',
            'privacy coin', 'private transaction', 'homomorphic encryption',
            
            # Blockchain infrastructure
            'validator', 'node operator', 'staking pool', 'consensus node', 'light client',
            'full node', 'archival node', 'indexer', 'block explorer', 'smart contract audit'
        }
        
        # Advanced computing terms
        self.advanced_computing_terms = {
            # Hardware advances
            'asic', 'fpga', 'tpu', 'gpu computing', 'cuda', 'neuromorphic computing',
            'in-memory computing', 'photonic computing', 'optical computing', 'dna computing',
            'biological computing', 'molecular computing', 'quantum dot', 'spintronics',
            
            # Computing paradigms
            'edge computing', 'fog computing', 'grid computing', 'serverless computing',
            'decentralized computing', 'distributed computing', 'cloud native', 'high performance computing',
            'hpc', 'exascale computing', 'supercomputer', 'parallel computing',
            
            # Advanced database concepts
            'newSQL', 'graph database', 'time-series database', 'distributed database',
            'decentralized storage', 'content-addressable storage', 'ipfs', 'filecoin',
            
            # Networking advances
            'web3', 'web 3.0', 'decentralized web', 'mesh network', '5g', '6g',
            'space internet', 'starlink', 'low-earth orbit', 'leo satellite',
            'decentralized dns', 'ens', 'handshake protocol', 'low-power wide-area network',
            'lpwan', 'lorawan', 'software-defined networking', 'network function virtualization'
        }
        
        # Add tech terms to the existing initialization method - should be called from __init__
        # after self._initialize_keywords() is called
        self.all_tech_terms = self.ai_terms.union(
            self.quantum_terms,
            self.blockchain_tech_terms,
            self.advanced_computing_terms
        )
        
        # Technology keywords for market/tech integration
        self.tech_keywords = {
            # AI specific
            'ai': [
                'artificial intelligence', 'machine learning', 'neural network', 'deep learning',
                'llm', 'gpt', 'chatgpt', 'large language model', 'generative ai', 'transformer',
                'computer vision', 'nlp', 'agi', 'superintelligence', 'multimodal ai',
                'fine-tuning', 'vector database', 'prompt engineering', 'reinforcement learning'
            ],
            
            # Quantum computing
            'quantum': [
                'quantum computing', 'quantum computer', 'qubit', 'quantum supremacy',
                'quantum advantage', 'quantum algorithm', 'quantum encryption', 'quantum security',
                'quantum resistant', 'post-quantum', 'quantum cryptography', 'quantum key',
                'quantum annealing', 'quantum error correction', 'quantum simulator'
            ],
            
            # Advanced computing categories
            'computing': [
                'supercomputer', 'exascale', 'high-performance computing', 'edge computing',
                'neuromorphic', 'optical computing', 'asic', 'fpga', 'tpu', 'gpu', 'accelerator',
                'in-memory computing', 'dna computing', 'biological computing', 'molecular computing',
                'quantum dot', 'spintronics', 'reversible computing'
            ]
        }

    @ensure_naive_datetimes
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
                'sentiment': self._analyze_sentiment(post_text),
                'text': post_text  # Include original text for reference
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
                'analyzed_at': strip_timezone(datetime.now())
            }

            # Store analysis if database available
            if self.db:
                self._store_analysis(post['post_id'], analysis)

            return analysis

        except Exception as e:
            logger.log_error("Post Analysis", str(e))
            return {'reply_worthy': False, 'error': str(e)}
        
    def detect_tech_topics(self, text: str) -> Dict[str, Any]:
        """
        Detect tech-related topics in the post text with categorical classification
        
        Args:
            text: Post text
                
        Returns:
            Dictionary of detected tech topics with categories and confidence
        """
        if not hasattr(self, 'ai_terms'):
            self._initialize_tech_keywords()
            
        topics = {
            'has_tech_content': False,
            'categories': {
                'ai': {'detected': False, 'terms': [], 'confidence': 0.0},
                'quantum': {'detected': False, 'terms': [], 'confidence': 0.0},
                'blockchain_tech': {'detected': False, 'terms': [], 'confidence': 0.0},
                'advanced_computing': {'detected': False, 'terms': [], 'confidence': 0.0}
            },
            'integration': {
                'crypto_tech_integration': False,
                'integration_topics': []
            },
            'educational_value': 0.0  # 0.0-1.0 scale of educational value
        }
        
        text_lower = text.lower()
        words = set(text_lower.split())
        
        # Check for AI terms
        ai_matches = [term for term in self.ai_terms if term in text_lower]
        if ai_matches:
            topics['categories']['ai']['detected'] = True
            topics['categories']['ai']['terms'] = ai_matches[:5]  # Limit to top 5
            topics['categories']['ai']['confidence'] = min(1.0, len(ai_matches) * 0.2)
            topics['has_tech_content'] = True
        
        # Check for quantum terms
        quantum_matches = [term for term in self.quantum_terms if term in text_lower]
        if quantum_matches:
            topics['categories']['quantum']['detected'] = True
            topics['categories']['quantum']['terms'] = quantum_matches[:5]
            topics['categories']['quantum']['confidence'] = min(1.0, len(quantum_matches) * 0.25)
            topics['has_tech_content'] = True
        
        # Check for blockchain tech terms (beyond trading)
        blockchain_tech_matches = [term for term in self.blockchain_tech_terms if term in text_lower]
        if blockchain_tech_matches:
            topics['categories']['blockchain_tech']['detected'] = True
            topics['categories']['blockchain_tech']['terms'] = blockchain_tech_matches[:5]
            topics['categories']['blockchain_tech']['confidence'] = min(1.0, len(blockchain_tech_matches) * 0.2)
            topics['has_tech_content'] = True
        
        # Check for advanced computing terms
        computing_matches = [term for term in self.advanced_computing_terms if term in text_lower]
        if computing_matches:
            topics['categories']['advanced_computing']['detected'] = True
            topics['categories']['advanced_computing']['terms'] = computing_matches[:5]
            topics['categories']['advanced_computing']['confidence'] = min(1.0, len(computing_matches) * 0.25)
            topics['has_tech_content'] = True
        
        # Detect crypto and tech integration
        crypto_terms = self._extract_crypto_terms(text)
        has_crypto = len(crypto_terms) > 0
        
        if has_crypto and topics['has_tech_content']:
            topics['integration']['crypto_tech_integration'] = True
            
            # Identify integration topics
            integration_patterns = [
                (r'(crypto|blockchain).*?(ai|artificial intelligence|machine learning)', 'crypto_ai'),
                (r'(ai|artificial intelligence|machine learning).*?(crypto|blockchain)', 'ai_crypto'),
                (r'(quantum).*?(crypto|blockchain|bitcoin|security)', 'quantum_crypto'),
                (r'(crypto|blockchain).*?(quantum)', 'crypto_quantum'),
                (r'(ai|artificial intelligence|machine learning).*?(trading|prediction|analysis)', 'ai_trading'),
                (r'(crypto|blockchain).*?(computing|hardware|chip|processor)', 'crypto_hardware'),
                (r'(web3|decentralized).*?(ai|machine learning|model)', 'decentralized_ai')
            ]
            
            for pattern, topic in integration_patterns:
                if re.search(pattern, text_lower):
                    topics['integration']['integration_topics'].append(topic)
        
        # Calculate educational value
        # Higher value for specific technical terms, explanations, and fewer hype words
        if topics['has_tech_content']:
            # Base education value from term count
            term_count = len(ai_matches) + len(quantum_matches) + len(blockchain_tech_matches) + len(computing_matches)
            base_value = min(0.6, term_count * 0.05)
            
            # Bonus for explanatory language
            explanation_indicators = ['explained', 'how it works', 'understand', 'explanation', 'tutorial', 
                                     'guide', 'introduction', 'meaning', 'defined', 'basics']
            has_explanation = any(indicator in text_lower for indicator in explanation_indicators)
            explanation_bonus = 0.2 if has_explanation else 0.0
            
            # Penalty for hype language
            hype_indicators = ['moon', 'pump', 'revolutionary', 'game-changer', 'disrupt', 'massive', 
                              'huge', 'incredible', 'amazing', 'insane', 'unbelievable']
            hype_count = sum(1 for indicator in hype_indicators if indicator in text_lower)
            hype_penalty = min(0.3, hype_count * 0.05)
            
            # Calculate final educational value
            topics['educational_value'] = min(1.0, max(0.0, base_value + explanation_bonus - hype_penalty))
        
        return topics

    def find_tech_related_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter posts to only include those related to technology topics
        
        Args:
            posts: List of post data dictionaries
                
        Returns:
            Filtered list of tech-related posts
        """
        if not posts:
            logger.logger.warning("No posts provided to filter for tech relevance")
            return []
        
        if not hasattr(self, 'tech_keywords'):
            self._initialize_tech_keywords()
            
        logger.logger.info(f"Filtering {len(posts)} posts for tech relevance")
        
        # Flatten the keywords for efficient matching
        all_tech_keywords = []
        for category in self.tech_keywords.values():
            all_tech_keywords.extend(category)
        
        # Store matched posts with their relevance score
        tech_related = []
        
        for post in posts:
            post_text = post.get('text', '').lower() if post.get('text') else ''
            
            # Skip posts with no text
            if not post_text:
                continue
                
            # Track matched keywords for logging
            matched_keywords = []
            keyword_categories = set()
            
            # Basic keyword matching with categorization
            for category, keywords in self.tech_keywords.items():
                for keyword in keywords:
                    if keyword in post_text:
                        matched_keywords.append(keyword)
                        keyword_categories.add(category)
            
            # Advanced tech pattern matching
            tech_patterns = [
                r'artificial intelligence|machine learning|neural network|deep learning',
                r'quantum comput(ing|er|ers)|qubit|quantum algorithm',
                r'blockchain technology|web3|decentralized|distributed ledger',
                r'supercomputer|neuromorphic|exascale|high-performance comput',
                r'edge comput(ing)|cloud comput(ing)|serverless|distributed system',
                r'large language model|transformer|llm|gpt|bert|t5',
                r'computer vision|nlp|natural language process',
                r'generative ai|diffusion model|gan|text-to-image',
                r'smart contract|zero-knowledge proof|zk-rollup|optimistic rollup',
                r'self-sovereign identity|verifiable credential|decentralized identifier'
            ]
            
            for pattern in tech_patterns:
                matches = re.findall(pattern, post_text, re.IGNORECASE)
                if matches:
                    for match in matches:
                        matched_keywords.append(match if isinstance(match, str) else match[0])
                        # Try to determine category
                        if any(ai_term in match.lower() for ai_term in ['ai', 'intelligence', 'neural', 'language model', 'gpt']):
                            keyword_categories.add('ai')
                        elif 'quantum' in match.lower():
                            keyword_categories.add('quantum')
                        elif any(compute_term in match.lower() for compute_term in ['compute', 'server', 'cloud', 'edge']):
                            keyword_categories.add('computing')
                        # Skip categorization if can't determine clearly
            
            # Check for educational content patterns
            educational_patterns = [
                r'explain(s|ed|ing)?\s(how|what|why)',
                r'introduction to\s[a-z\s]+(ai|computing|blockchain|quantum)',
                r'(beginners?|complete)\sguide\sto',
                r'understand(ing)?\s[a-z\s]+(ai|computing|blockchain|quantum)',
                r'tutorial on\s[a-z\s]+(ai|computing|blockchain|quantum)'
            ]
            
            educational_content = False
            for pattern in educational_patterns:
                if re.search(pattern, post_text, re.IGNORECASE):
                    educational_content = True
                    break
            
            # Calculate relevance score based on matches
            relevance_score = 0
            
            # Base score from number of matches
            relevance_score += min(10, len(matched_keywords))
            
            # Bonus for matching multiple categories
            relevance_score += len(keyword_categories) * 3
            
            # Bonus for educational content
            if educational_content:
                relevance_score += 5
            
            # Mark as tech related if we found any matches and score is high enough
            if matched_keywords and relevance_score >= 3:
                # Add relevant fields to the post
                post['tech_related'] = True
                post['tech_keywords'] = matched_keywords
                post['tech_categories'] = list(keyword_categories)
                post['tech_relevance_score'] = relevance_score
                post['tech_educational'] = educational_content
                tech_related.append(post)
        
        logger.logger.info(f"Found {len(tech_related)} tech-related posts out of {len(posts)}")
        
        # Log some example keywords for debugging
        if tech_related:
            examples = [p.get('tech_keywords', [])[:3] for p in tech_related[:3]]
            logger.logger.debug(f"Example tech keywords matched: {examples}")
        
        # Sort by relevance score (highest first)
        tech_related.sort(key=lambda x: x.get('tech_relevance_score', 0), reverse=True)
        
        return tech_related

    def analyze_tech_post(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a tech-related post for specific tech insights and educational value
        
        Args:
            post: Post data dictionary with text content
            
        Returns:
            Dictionary with tech analysis results
        """
        post_text = post.get('text', '')
        if not post_text:
            return {'has_tech_content': False}
        
        # Get general tech topic detection
        tech_topics = self.detect_tech_topics(post_text)
        
        # Enhanced analysis for educational assessment
        analysis = {
            'has_tech_content': tech_topics['has_tech_content'],
            'categories': tech_topics['categories'],
            'educational': {
                'is_educational': tech_topics['educational_value'] >= 0.4,
                'educational_value': tech_topics['educational_value'],
                'educational_type': 'none',  # Will be updated below
                'complexity_level': 0.0,     # 0.0-1.0 scale, higher = more complex
                'audience_level': 'general'  # general, intermediate, technical
            },
            'tech_sentiment': self._analyze_tech_sentiment(post_text),
            'key_points': self._extract_tech_key_points(post_text),
            'crypto_integration': tech_topics['integration'],
            'reply_worthy': False  # Will be determined based on analysis
        }
        
        # Determine educational type if content is educational
        if analysis['educational']['is_educational']:
            # Check for different types of educational content
            if re.search(r'(how|what|why)[^.?!]{0,30}(works|function|operate)', post_text, re.IGNORECASE):
                analysis['educational']['educational_type'] = 'explanatory'
            elif re.search(r'history|evolution|development|timeline', post_text, re.IGNORECASE):
                analysis['educational']['educational_type'] = 'historical'
            elif re.search(r'future|prediction|forecast|upcoming|next', post_text, re.IGNORECASE):
                analysis['educational']['educational_type'] = 'predictive'
            elif re.search(r'comparison|versus|vs\.|differ|contrast', post_text, re.IGNORECASE):
                analysis['educational']['educational_type'] = 'comparative'
            elif re.search(r'tutorial|step[- ]by[- ]step|guide|instruction', post_text, re.IGNORECASE):
                analysis['educational']['educational_type'] = 'tutorial'
            elif re.search(r'overview|introduction|beginner|basics', post_text, re.IGNORECASE):
                analysis['educational']['educational_type'] = 'introductory'
            else:
                analysis['educational']['educational_type'] = 'informational'
        
            # Analyze complexity level
            technical_terms_count = self._count_technical_terms(post_text)
            sentence_complexity = self._analyze_sentence_complexity(post_text)
            
            # Calculate complexity score (0.0-1.0)
            complexity_score = min(1.0, (technical_terms_count * 0.05) + (sentence_complexity * 0.5))
            analysis['educational']['complexity_level'] = complexity_score
            
            # Determine audience level
            if complexity_score < 0.3:
                analysis['educational']['audience_level'] = 'general'
            elif complexity_score < 0.7:
                analysis['educational']['audience_level'] = 'intermediate'
            else:
                analysis['educational']['audience_level'] = 'technical'
        
        # Determine if post is worthy of engagement or reply
        # We want to encourage engagement with educational content
        if analysis['has_tech_content']:
            reply_score = 0.0
            
            # Educational content gets high priority
            if analysis['educational']['is_educational']:
                reply_score += 0.5
                
                # Boost for tutorial or explanatory content
                if analysis['educational']['educational_type'] in ['tutorial', 'explanatory', 'comparative']:
                    reply_score += 0.2
                    
                # Boost for appropriate complexity (not too simple, not too complex)
                complexity = analysis['educational']['complexity_level'] 
                if 0.3 <= complexity <= 0.7:
                    reply_score += 0.2
            
            # Boost for crypto integration discussions
            if analysis['crypto_integration']['crypto_tech_integration']:
                reply_score += 0.3
                
            # Boost for positive sentiment (we want to encourage positive tech discussion)
            if analysis['tech_sentiment']['label'] in ['positive', 'excited', 'curious']:
                reply_score += 0.1
                
            # Boost for questions or discussion starters
            if '?' in post_text or re.search(r'(thoughts|opinions|discuss|what do you think)', post_text, re.IGNORECASE):
                reply_score += 0.2
            
            # High-value tech categories get a boost
            for category, data in analysis['categories'].items():
                if data['detected'] and data['confidence'] > 0.5:
                    if category in ['ai', 'quantum']:  # Prioritize cutting-edge tech
                        reply_score += 0.1
                        break
            
            # Determine reply-worthiness
            analysis['reply_worthy'] = reply_score >= 0.6
            analysis['reply_score'] = reply_score
        
        return analysis

    def _analyze_tech_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment specifically in tech-related content
        
        Args:
            text: Post text content
            
        Returns:
            Dictionary with tech sentiment analysis
        """
        # Tech-specific sentiment words
        positive_tech = [
            'innovative', 'breakthrough', 'advancing', 'revolutionary', 'progress',
            'promising', 'efficient', 'powerful', 'improved', 'faster', 'optimized',
            'scalable', 'elegant', 'robust', 'cutting-edge', 'remarkable', 'state-of-the-art'
        ]
        
        negative_tech = [
            'flawed', 'limited', 'overhyped', 'dangerous', 'concerning', 'problematic',
            'disappointing', 'risky', 'failing', 'regression', 'vulnerability', 'setback',
            'inefficient', 'unstable', 'unreliable', 'stagnant', 'insecure'
        ]
        
        excitement_words = [
            'exciting', 'amazing', 'mind-blowing', 'incredible', 'fascinating',
            'groundbreaking', 'impressive', 'stunning', 'game-changing', 'transformative'
        ]
        
        concern_words = [
            'worried', 'concerned', 'alarming', 'threatened', 'fear', 'cautious',
            'skeptical', 'suspicious', 'doubtful', 'wary', 'uncertain', 'hesitant'
        ]
        
        curiosity_words = [
            'curious', 'interested', 'intrigued', 'wondering', 'exploring',
            'investigating', 'learning', 'studying', 'researching', 'discovering'
        ]
        
        text_lower = text.lower()
        
        # Count sentiment words
        pos_count = sum(1 for word in positive_tech if word in text_lower)
        neg_count = sum(1 for word in negative_tech if word in text_lower)
        excitement_count = sum(1 for word in excitement_words if word in text_lower)
        concern_count = sum(1 for word in concern_words if word in text_lower)
        curiosity_count = sum(1 for word in curiosity_words if word in text_lower)
        
        # Analyze negations
        negations = ['not', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'don\'t', 'doesn\'t', 'didn\'t', 'no', 'never']
        
        # Check for negated positive terms (not innovative = negative)
        for negation in negations:
            for word in positive_tech:
                pattern = f"{negation}\\s+\\w+\\s+{word}|{negation}\\s+{word}"
                matches = re.findall(pattern, text_lower)
                pos_count -= len(matches)
                neg_count += len(matches)
        
        # Check for negated negative terms (not flawed = positive)
        for negation in negations:
            for word in negative_tech:
                pattern = f"{negation}\\s+\\w+\\s+{word}|{negation}\\s+{word}"
                matches = re.findall(pattern, text_lower)
                neg_count -= len(matches)
                pos_count += len(matches)
        
        # Determine dominant sentiment
        sentiment_scores = {
            'positive': pos_count,
            'negative': neg_count,
            'excited': excitement_count,
            'concerned': concern_count,
            'curious': curiosity_count,
            'neutral': 1  # Default value to avoid division by zero
        }
        
        # Adjust for post size - normalize by word count to avoid long post bias
        word_count = len(text_lower.split())
        if word_count > 20:
            # Scale factors for longer posts
            for key in sentiment_scores:
                if key != 'neutral':  # Don't scale neutral
                    sentiment_scores[key] = sentiment_scores[key] * (20 / word_count) ** 0.5
        
        # Get label with highest score
        dominant_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])[0]
        
        # Calculate overall positive/negative balance
        if pos_count + neg_count > 0:
            positivity = pos_count / (pos_count + neg_count)
        else:
            positivity = 0.5  # Neutral if no sentiment words
        
        # Build sentiment result
        sentiment = {
            'label': dominant_sentiment,
            'positivity': positivity,
            'scores': {
                'positive': pos_count,
                'negative': neg_count,
                'excited': excitement_count,
                'concerned': concern_count,
                'curious': curiosity_count
            }
        }
        
        return sentiment

    def _extract_tech_key_points(self, text: str) -> List[str]:
        """
        Extract key technical points from a tech-focused post
        
        Args:
            text: Post text content
            
        Returns:
            List of extracted key points
        """
        key_points = []
        
        # Identify potential key points based on language patterns
        # Look for definition-like statements
        definition_matches = re.findall(r'([A-Z][^.!?]*?(?:is|are|refers to|means|defined as)[^.!?]*\.)', text)
        for match in definition_matches:
            if len(match.split()) >= 5:  # Ensure it's substantial
                key_points.append(('definition', match.strip()))
        
        # Look for contrast/comparison statements
        comparison_matches = re.findall(r'([^.!?]*?(?:unlike|compared to|versus|in contrast to)[^.!?]*\.)', text)
        for match in comparison_matches:
            if len(match.split()) >= 5:
                key_points.append(('comparison', match.strip()))
        
        # Look for example/use case statements
        example_matches = re.findall(r'([^.!?]*?(?:for example|for instance|use case|application|one way|can be used)[^.!?]*\.)', text)
        for match in example_matches:
            if len(match.split()) >= 5:
                key_points.append(('example', match.strip()))
        
        # Look for technical limitations or challenges
        limitation_matches = re.findall(r'([^.!?]*?(?:limitation|challenge|problem|issue|barrier|obstacle|difficulty)[^.!?]*\.)', text)
        for match in limitation_matches:
            if len(match.split()) >= 5:
                key_points.append(('limitation', match.strip()))
        
        # Look for future/prediction statements
        future_matches = re.findall(r'([^.!?]*?(?:in the future|will|could|might|predict|expect|anticipate)[^.!?]*\.)', text)
        for match in future_matches:
            if len(match.split()) >= 5 and any(future_word in match.lower() for future_word in ['future', 'soon', 'next', 'coming', 'years', 'decade']):
                key_points.append(('prediction', match.strip()))
        
        # Limit to top key points (at most 5)
        formatted_points = []
        for point_type, point_text in key_points[:5]:
            # Simplify text slightly for readability
            simplified = re.sub(r'\s+', ' ', point_text).strip()
            formatted_points.append(simplified)
        
        return formatted_points

    def _count_technical_terms(self, text: str) -> int:
        """Count technical terminology occurrences in text"""
        if not hasattr(self, 'all_tech_terms'):
            self._initialize_tech_keywords()
            
        text_lower = text.lower()
        
        # Count technical terms from our term collections
        term_count = sum(1 for term in self.all_tech_terms if term in text_lower)
        
        # Add specific technical patterns
        technical_patterns = [
            r'\d+\s*(bit|byte|kb|mb|gb|tb)',
            r'\d+\s*(nm|nanometer)',
            r'\d+\s*(ghz|mhz)',
            r'\d+\s*(core|thread)',
            r'\d+\s*(qubit|quantum bit)',
            r'\d+\s*(tesla|tflop)',
            r'\d+\s*(parameter|weight)',
            r'([A-Z]+\d*\-\d+)',  # Technical model numbers
            r'(v\d+\.\d+(?:\.\d+)?)',  # Version numbers
            r'([a-zA-Z]+\-\d+(?:\-[a-zA-Z0-9]+)?)'  # Technical identifiers
        ]
        
        for pattern in technical_patterns:
            term_count += len(re.findall(pattern, text))
        
        return term_count

    def _analyze_sentence_complexity(self, text: str) -> float:
        """
        Analyze the complexity of sentence structure in the text
        
        Returns:
            Complexity score between 0.0 and 1.0
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        if not sentences:
            return 0.0
        
        # Analyze sentence length
        word_counts = [len(s.split()) for s in sentences if s.strip()]
        
        if not word_counts:
            return 0.0
        
        avg_words_per_sentence = sum(word_counts) / len(word_counts)
        
        # Count complex sentence markers
        complex_markers = [
            'therefore', 'however', 'additionally', 'consequently',
            'furthermore', 'moreover', 'nevertheless', 'alternatively',
            'meanwhile', 'subsequently', 'conversely', 'specifically',
            'whereas', 'accordingly', 'despite', 'although', 'notwithstanding',
            'given that', 'provided that', 'insofar as', 'such that'
        ]
        
        marker_count = sum(1 for marker in complex_markers if marker in text.lower())
        
        # Count technical conjunctions
        technical_conjunctions = [
            'in other words', 'to be precise', 'in particular',
            'for instance', 'namely', 'specifically', 'that is to say',
            'to illustrate', 'in this case', 'in contrast',
            'on the contrary', 'on the other hand', 'by comparison',
            'by extension', 'in principle', 'in theory', 'in practice'
        ]
        
        tech_conj_count = sum(1 for conj in technical_conjunctions if conj in text.lower())
        
        # Calculate complexity score
        # - Longer sentences increase complexity
        length_factor = min(1.0, avg_words_per_sentence / 25)
        
        # - More complex markers increase complexity
        marker_factor = min(1.0, marker_count / 5)
        
        # - More technical conjunctions increase complexity
        conj_factor = min(1.0, tech_conj_count / 3)
        
        # Weighted combination
        complexity = (length_factor * 0.4) + (marker_factor * 0.3) + (conj_factor * 0.3)
        
        return complexity

    def find_tech_educational_content(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter posts to find specifically educational tech content
        
        Args:
            posts: List of post data dictionaries
                
        Returns:
            Filtered list of educational tech posts
        """
        # First, find tech-related posts
        tech_posts = self.find_tech_related_posts(posts)
        
        if not tech_posts:
            return []
        
        # Then filter for educational content
        educational_posts = []
        
        for post in tech_posts:
            # Get comprehensive tech analysis
            analysis = self.analyze_tech_post(post)
            
            # Store the analysis with the post
            post['tech_analysis'] = analysis
            
            # Check if it's educational content
            if analysis['educational']['is_educational']:
                educational_posts.append(post)
        
        # Sort by educational value (highest first)
        educational_posts.sort(
            key=lambda x: x['tech_analysis']['educational']['educational_value'], 
            reverse=True
        )
        
        return educational_posts

    def generate_tech_reply(self, post: Dict[str, Any], llm_provider=None) -> str:
        """
        Generate an educational tech-focused reply to a post
        
        Args:
            post: Post data dictionary
            llm_provider: Optional LLM provider
                
        Returns:
            Educational tech reply
        """
        # Use provided LLM provider or instance variable
        provider = llm_provider or self.llm_provider
        
        if not provider:
            logger.logger.error("No LLM provider available for tech reply generation")
            return "I'd love to discuss this tech topic but need more info. What specifically interests you about it?"
            
        # Perform tech analysis if not already present
        tech_analysis = post.get('tech_analysis')
        if not tech_analysis:
            tech_analysis = self.analyze_tech_post(post)
        
        # Extract key details for the prompt
        post_text = post.get('text', '')
        author_name = post.get('author_name', 'User')
        
        # Determine focus areas based on detected categories
        focus_categories = []
        for category, data in tech_analysis['categories'].items():
            if data['detected'] and data['confidence'] > 0.3:
                focus_categories.append(category)
        
        # Determine appropriate response type
        response_type = "informative"  # Default
        
        if '?' in post_text:
            response_type = "question_answer"
        elif tech_analysis['tech_sentiment']['label'] in ['curious', 'concerned']:
            response_type = "explanatory"
        elif tech_analysis['tech_sentiment']['label'] == 'excited':
            response_type = "build_on_excitement"
        
        # Check for integration with crypto
        crypto_integration = tech_analysis['crypto_integration']['crypto_tech_integration']
        
        # Determine appropriate educational level based on the post
        audience_level = tech_analysis['educational']['audience_level']
        
        # Build the prompt for tech-focused educational reply
        prompt = f"""Generate an educational reply to this technology-related post. Your reply should be informative, accurate, and engaging.

POST FROM {author_name}: "{post_text}"

TECH ANALYSIS:
- Main tech categories: {', '.join(focus_categories) if focus_categories else 'general tech'}
- Post sentiment: {tech_analysis['tech_sentiment']['label']}
- Educational value: {tech_analysis['educational']['educational_value']:.1f}/1.0
- Audience level: {audience_level}
- Crypto integration: {"Yes" if crypto_integration else "No"}

INSTRUCTIONS:
1. Response type should be: {response_type}
2. Technical accuracy level: Target {audience_level} audience
3. Keep the reply concise (1-3 paragraphs) and conversational
4. Include at least one specific, accurate fact or insight about the technology
5. If mentioning crypto, ensure the connection to the tech topic is clear and meaningful
6. End with a subtle follow-up question that encourages further discussion
7. Write in a friendly, collaborative tone that encourages learning

Your reply (about 240 characters):
"""

        # Generate reply using the LLM provider
        reply = provider.generate_text(prompt=prompt, max_tokens=350)
        
        if not reply:
            # Fallback reply if generation fails
            return "That's an interesting tech point! I'd like to explore this topic further. What specific aspect interests you most?"
        
        # Ensure reply is within Twitter's character limit
        if len(reply) > 240:
            # Find a good truncation point
            last_sentence = max(reply.rfind('. '), reply.rfind('? '), reply.rfind('! '))
            if last_sentence > 160:
                reply = reply[:last_sentence+1].strip()
            else:
                # If we can't find a good sentence break, truncate at a word boundary
                last_space = reply[:240].rfind(' ')
                if last_space > 0:
                    reply = reply[:last_space].strip()
                else:
                    reply = reply[:237] + "..."
        
        return reply

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
        
        # Flatten the keywords for efficient matching
        all_keywords = []
        for category in self.market_keywords.values():
            all_keywords.extend(category)
        
        # Store matched posts with their relevance score
        market_related = []
        
        for post in posts:
            post_text = post.get('text', '').lower() if post.get('text') else ''
            
            # Skip posts with no text
            if not post_text:
                continue
                
            # Track matched keywords for logging
            matched_keywords = []
            keyword_categories = set()
            
            # Basic keyword matching with categorization
            for category, keywords in self.market_keywords.items():
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
                        for cat, keywords in self.market_keywords.items():
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
        
    @ensure_naive_datetimes
    def filter_already_replied_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out posts we've already replied to. Handles missing database methods gracefully.
        
        Args:
            posts: List of post data dictionaries
            
        Returns:
            Filtered list of posts
        """
        if not posts:
            logger.logger.warning("No posts provided to filter for replies")
            return []
            
        logger.logger.info(f"Filtering {len(posts)} posts for ones we haven't replied to yet")
        
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
                # Try using the database method if it exists
                if self.db and hasattr(self.db, 'check_if_post_replied'):
                    already_replied = self.db.check_if_post_replied(post_id, post_url)
                else:
                    # Fallback method if database function doesn't exist
                    already_replied = self._check_if_post_replied_fallback(post_id, post_url)
            except Exception as e:
                logger.logger.warning(f"Error checking if post was replied to: {str(e)}")
                # Fall back to checking locally
                already_replied = False
            
            if not already_replied:
                filtered_posts.append(post)
        
        logger.logger.info(f"Filtered out {len(posts) - len(filtered_posts)} already replied posts")
        return filtered_posts
    
    def _check_if_post_replied_fallback(self, post_id: str, post_url: str) -> bool:
        """
        Fallback method to check if a post has been replied to when database method is unavailable
        
        Args:
            post_id: The ID of the post
            post_url: The URL of the post
            
        Returns:
            True if we've replied to this post, False otherwise
        """
        # Without proper database support, we assume not replied
        # This is a placeholder for a proper implementation
        
        # In a real implementation, we could check:
        # 1. Local memory cache of replied posts
        # 2. Simple file-based storage of previously replied posts
        # 3. Scrape the post page to see if our account has replied
        
        return False
    
    @ensure_naive_datetimes
    def prioritize_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort posts by engagement level, relevance, recency
        
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
        now = strip_timezone(datetime.now())
        
        for post in posts:
            try:
                # Base score starts at 0
                score = 0
                
                # Factor 1: Engagement score (0-50 points)
                engagement = min(100, post.get('engagement_score', 0))
                score += engagement * 0.5  # Up to 50 points
                
                # Factor 2: Recency (0-30 points)
                timestamp = post.get('timestamp')
                if timestamp:
                    # Calculate hours since posted
                    hours_ago = safe_datetime_diff(now, timestamp) / 3600
                    
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
                    
                # Store score with post
                post['priority_score'] = score
                scored_posts.append(post)
            
            except Exception as e:
                # If there's an error scoring this post, still include it with a default score
                logger.logger.warning(f"Error scoring post: {e}")
                post['priority_score'] = 0
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

    def _detect_conversation_hooks(self, text: str) -> List[str]:
        """Detect conversation hooks and engagement cues"""
        text_lower = text.lower()
        words = text_lower.split()
        found_hooks = []
        
        # Check for individual hook words
        for hook in self.conversation_hooks:
            if hook in words:
                found_hooks.append(hook)
                
        # Check for specific patterns
        conversation_patterns = [
            r'what (do|about|if) you',
            r'anyone (else|here)',
            r'do you (think|feel)',
            r'(agree|disagree)'
        ]
        
        for pattern in conversation_patterns:
            if re.search(pattern, text_lower):
                found_hooks.append(pattern)
                
        return found_hooks

    @ensure_naive_datetimes
    def _analyze_conversation_state(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze conversation context and state with safe datetime handling"""
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
            current_time = strip_timezone(datetime.now())
            
            # Check if post is part of ongoing thread
            if post.get('parent_id'):
                state['is_thread'] = True
                # Get thread history
                thread_history = self._get_thread_history(post)
                
                if thread_history:
                    # Process timestamps
                    thread_timestamps = []
                    for p in thread_history:
                        ts = p.get('timestamp')
                        if ts:
                            thread_timestamps.append(ts)
                    
                    if thread_timestamps:
                        # Calculate thread age
                        oldest_post = min(thread_timestamps)
                        state['conversation_age'] = safe_datetime_diff(current_time, oldest_post)
                        
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
                # Calculate recent interactions
                recent_interactions = 0
                for p in author_history:
                    ts = p.get('timestamp')
                    if ts:
                        time_diff = safe_datetime_diff(current_time, ts)
                        if time_diff < 86400:  # 24 hours
                            recent_interactions += 1
                
                state['activity_multiplier'] *= (1 + (recent_interactions * 0.1))
            
            # Update conversation tracking
            self._update_conversation_tracking(post)
            
            return state
            
        except Exception as e:
            logger.log_error("Conversation State Analysis", str(e))
            return state

    @ensure_naive_datetimes
    def _get_thread_history(self, post: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get conversation thread history with safe datetime handling"""
        try:
            thread_id = post.get('thread_id') or post.get('parent_id')
            if not thread_id or not self.db:
                return []
                
            # Query thread history from database
            history = self.db.get_thread_history(thread_id) or []
            
            # Return with timezone-aware datetimes handled
            return history
            
        except Exception as e:
            logger.log_error("Thread History Retrieval", str(e))
            return []

    @ensure_naive_datetimes
    def _get_author_history(self, author_id: str) -> List[Dict[str, Any]]:
        """Get author interaction history with safe datetime handling"""
        try:
            if not author_id or not self.db:
                return []
                
            # Get recent author posts and our replies
            history = self.db.get_author_interaction_history(author_id, hours=24) or []
            
            # Return with timezone-aware datetimes handled
            return history
            
        except Exception as e:
            logger.log_error("Author History Retrieval", str(e))
            return []

    @ensure_naive_datetimes
    def _update_conversation_tracking(self, post: Dict[str, Any]) -> None:
        """Update active conversation tracking with safe datetime handling"""
        try:
            now = strip_timezone(datetime.now())
            post_id = post.get('post_id', '')
            thread_id = post.get('thread_id') or post.get('parent_id')
            
            # Clean expired conversations
            expired = []
            for conv_id, data in self.active_discussions.items():
                last_update = data['last_update']
                if safe_datetime_diff(now, last_update) > self.discussion_expiry:
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

    @ensure_naive_datetimes
    def _update_trending(self, features: Dict[str, Any]) -> None:
        """Update trending topics and engagement tracking with safe datetime handling"""
        try:
            now = strip_timezone(datetime.now())
        
            # Reset trending data if expired
            if safe_datetime_diff(now, self.trending_reset_time) > (self.trending_reset_hours * 3600):
                self.trending = {
                    'topics': Counter(),
                    'tokens': Counter(),
                    'phrases': Counter(),
                    'sentiment': Counter()
                }
                self.trending_reset_time = now
        
            # Update topic tracking - handle crypto_terms properly
            if 'crypto_terms' in features:
                crypto_terms = features['crypto_terms']
            
                # Handle different crypto_terms structures
                if isinstance(crypto_terms, dict):
                    # Flatten dictionary into hashable items
                    for category, terms in crypto_terms.items():
                        if isinstance(terms, list):
                            for term in terms:
                                # Ensure term is hashable
                                if isinstance(term, str):
                                    self.trending['topics'][term] += 1
                                elif isinstance(term, (list, dict)):
                                    # Convert non-hashable types to string representation
                                    self.trending['topics'][str(term)] += 1
                elif isinstance(crypto_terms, list):
                    # Process list of terms
                    for term in crypto_terms:
                        # Ensure term is hashable
                        if isinstance(term, str):
                            self.trending['topics'][term] += 1
                        elif isinstance(term, (list, dict)):
                            # Convert non-hashable types to string representation
                            self.trending['topics'][str(term)] += 1
        
            # Update token tracking safely
            if 'crypto_terms' in features and isinstance(features['crypto_terms'], dict) and 'symbols' in features['crypto_terms']:
                for token in features['crypto_terms']['symbols']:
                    # Ensure token is hashable
                    if isinstance(token, str):
                        self.trending['tokens'][token] += 1
                    else:
                        self.trending['tokens'][str(token)] += 1
                    
            # Update phrase tracking
            for phrase in features.get('meme_phrases', []):
                # Ensure phrase is hashable
                if isinstance(phrase, str):
                    self.trending['phrases'][phrase] += 1
                else:
                    self.trending['phrases'][str(phrase)] += 1
            
            # Update sentiment tracking
            sentiment = features.get('sentiment', {})
            if isinstance(sentiment, dict) and 'label' in sentiment:
                # Extract the sentiment label
                sentiment_label = sentiment['label']
                if isinstance(sentiment_label, str):
                    self.trending['sentiment'][sentiment_label] += 1
                else:
                    self.trending['sentiment'][str(sentiment_label)] += 1
            elif isinstance(sentiment, str):
                # Sometimes sentiment might be directly a string
                self.trending['sentiment'][sentiment] += 1
            
            # Trim to keep only most frequent items
            for category in self.trending:
                self.trending[category] = Counter(
                    dict(self.trending[category].most_common(50))
                )
                
        except Exception as e:
            logger.log_error("Trending Update", str(e))

    @ensure_naive_datetimes
    def _calculate_context_score(self, post: Dict[str, Any], features: Dict[str, Any]) -> float:
        """Calculate contextual relevance score with safe datetime handling"""
        try:
            score = 1.0  # Base score
            
            # Time of day adjustment
            hour = strip_timezone(datetime.now()).hour
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

    @ensure_naive_datetimes
    def _store_analysis(self, post_id: str, analysis: Dict[str, Any]) -> None:
        """Store analysis results in database with safe datetime handling"""
        try:
            if not self.db or not post_id:
                return
                
            # Prepare data for storage
            analysis_copy = analysis.copy()
            
            # Convert analyzed_at to string to ensure it's serializable
            if 'analyzed_at' in analysis_copy:
                analysis_copy['analyzed_at'] = str(analysis_copy['analyzed_at'])
            
            # Prepare data for storage
            storage_data = {
                'post_id': post_id,
                'reply_worthy': analysis_copy['reply_worthy'],
                'reply_score': analysis_copy['reply_score'],
                'features': json.dumps(analysis_copy['features']),
                'engagement_scores': json.dumps(analysis_copy['engagement_scores']),
                'response_focus': json.dumps(analysis_copy['response_focus']),
                'timestamp': strip_timezone(datetime.now())
            }
            
            # Check if store_content_analysis method exists
            if hasattr(self.db, 'store_content_analysis'):
                self.db.store_content_analysis(**storage_data)
            else:
                logger.logger.warning("Database doesn't have store_content_analysis method, analysis not stored")
            
        except Exception as e:
            logger.log_error("Analysis Storage", str(e))

    @ensure_naive_datetimes
    def _find_related_topics(self) -> Dict[str, List[str]]:
        """Find related topics based on co-occurrence with safe error handling"""
        try:
            related = {}
            
            # Get top topics
            top_topics = self.trending['topics'].most_common(10)
            
            for topic, _ in top_topics:
                # Find topics that frequently appear together
                co_occurrences = Counter()
                
                # Query database for co-occurring topics if available
                if self.db and hasattr(self.db, 'get_recent_posts_with_topic'):
                    try:
                        recent_posts = self.db.get_recent_posts_with_topic(topic, hours=24)
                        
                        for post in recent_posts:
                            post_topics = set(post.get('topics', []))
                            post_topics.discard(topic)  # Remove the current topic
                            for other_topic in post_topics:
                                co_occurrences[other_topic] += 1
                    except Exception as db_err:
                        logger.logger.warning(f"Error retrieving posts for topic {topic}: {db_err}")
                
                # Get top 3 related topics
                related[topic] = [
                    topic for topic, count in co_occurrences.most_common(3)
                ]
            
            return related
            
        except Exception as e:
            logger.log_error("Related Topics Analysis", str(e))
            return {}
            
    @ensure_naive_datetimes
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

    def generate_response_with_llm(self, post: Dict[str, Any], market_data: Dict[str, Any] = None, llm_provider=None) -> str:
        """
        Generate a response to a post using the connected LLM provider
        
        Args:
            post: Post data dictionary
            market_data: Current market data (optional)
            llm_provider: LLM provider instance (optional)
        
        Returns:
            Generated response text
        """
        try:
            # Use provided LLM provider or instance variable
            provider = llm_provider or self.llm_provider
            
            if not provider:
                logger.logger.error("No LLM provider available for response generation")
                return "No LLM provider available to generate response"
                
            # Analyze the post
            analysis = self.analyze_post(post)
            
            # Extract relevant features for response generation
            post_text = post.get('text', '')
            author_name = post.get('author_name', 'User')
            
            # Format market data if available
            market_context = ""
            if market_data:
                # Extract relevant token data based on post content
                relevant_tokens = self._extract_relevant_tokens(post_text, market_data)
                
                # Format token data
                for token, data in relevant_tokens.items():
                    market_context += f"\n{token}: ${data.get('current_price', 0):.4f}, "
                    market_context += f"{data.get('price_change_percentage_24h', 0):.2f}% (24h)"
            
            # Build response focus
            response_focus = analysis.get('response_focus', {})
            primary_type = response_focus.get('primary_type', 'inform')
            response_style = response_focus.get('style', 'neutral')
            response_length = response_focus.get('length', 'medium')
            
            # Generate prompt for the LLM
            prompt = f"""Generate a reply to the following social media post about cryptocurrency. 
Post from {author_name}: "{post_text}"

Response Focus: {primary_type}
Style: {response_style}
Length: {response_length}

{market_context}

Key points to address:
{', '.join(response_focus.get('key_points', []))}

Your response should be conversational, helpful and tailored to the post. Do not include quotation marks or formatting markers in your reply."""

            # Generate response using the LLM provider
            response = provider.generate_text(prompt=prompt, max_tokens=300, temperature=0.7)
            
            return response or "Sorry, I couldn't generate a response at this time."
            
        except Exception as e:
            logger.log_error("Response Generation", str(e))
            return "Failed to generate response due to an error"
    
    def _extract_relevant_tokens(self, text: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract tokens mentioned in text and get their market data"""
        relevant_tokens = {}
        
        # Extract potential token symbols from text
        # Look for standard token formats (BTC, ETH, etc.)
        symbols = re.findall(r'\$?[A-Z]{2,5}\b', text)
        
        # Also look for token names
        token_names = ['bitcoin', 'ethereum', 'ripple', 'solana', 'avalanche', 'chainlink']
        found_names = [name for name in token_names if name.lower() in text.lower()]
        
        # Map common names to symbols
        name_to_symbol = {
            'bitcoin': 'BTC',
            'ethereum': 'ETH',
            'ripple': 'XRP',
            'solana': 'SOL',
            'avalanche': 'AVAX',
            'chainlink': 'LINK'
        }
        
        # Add symbols from token names
        for name in found_names:
            if name.lower() in name_to_symbol:
                symbols.append(name_to_symbol[name.lower()])
        
        # Get market data for found symbols
        for symbol in set(symbols):  # Remove duplicates
            # Remove $ prefix if present
            if symbol.startswith('$'):
                symbol = symbol[1:]
                
            # Check if symbol exists in market data
            if symbol in market_data:
                relevant_tokens[symbol] = market_data[symbol]
            
        # If no specific tokens found, include BTC as reference
        if not relevant_tokens and 'BTC' in market_data:
            relevant_tokens['BTC'] = market_data['BTC']
            
        return relevant_tokens
                                        
