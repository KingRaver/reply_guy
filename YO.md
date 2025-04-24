# 🚀 Tokenetics Market Oracle

> *Advanced cryptocurrency analytics engine leveraging AI, technical indicators, and social media engagement for precision market insights*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.2-orange)](https://www.tensorflow.org/)
[![Anthropic Claude](https://img.shields.io/badge/LLM-Anthropic%20Claude-9cf)](https://www.anthropic.com/)
[![CoinGecko API](https://img.shields.io/badge/API-CoinGecko-brightgreen)](https://www.coingecko.com/en/api)
[![SQLAlchemy](https://img.shields.io/badge/ORM-SQLAlchemy-red)](https://www.sqlalchemy.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-blueviolet)](https://scikit-learn.org/)

## 📊 System Overview

Tokenetics Market Oracle is an enterprise-grade cryptocurrency analysis system that combines real-time market data, machine learning, and natural language processing to generate high-precision market insights across multiple timeframes. The system operates on a modular architecture:

![Architecture Overview]([https://raw.githubusercontent.com/username/repository/main/docs/architecture_diagram.png](https://github.com/KingRaver/reply_guy/blob/main/architecture.txt))


### Core Components Interaction

```
Market Data ──► Technical Analysis ──► Statistical Models ──► ML Predictions
     │                  │                      │                    │
     │                  │                      │                    │
     └──────────────────┴──────────────────────┴────────────► Unified Prediction ──► Social Media Engagement
                                                                     │
                                                                     ▼
                                                          Continuous Feedback & Improvement
```

## ✨ Features

### 🔍 Multi-Timeframe Analysis
- **1-hour forecasts** for short-term traders
- **24-hour predictions** for day traders
- **7-day outlooks** for swing traders
- Automatic rotation between timeframes based on market activity

### 📊 Advanced Market Analytics
- Comprehensive technical indicators suite (RSI, MACD, Bollinger Bands, Stochastic Oscillator)
- Smart money movement detection with Z-score thresholds and volume cluster identification
- Multi-dimensional volume analysis with hourly and daily profiling
- Market correlation mapping with sector rotation detection
- Price-volume divergence detection with quantitative scoring
- Adaptive trend detection across different timeframes
- Ichimoku Cloud analysis for longer timeframes
- Advanced Directional Index (ADX) for trend strength evaluation 
- Pivot point calculations with multiple methodologies (Standard, Fibonacci, Woodie)
- Dynamic volatility assessment and market comparison metrics

### 🤖 AI-Powered Insights
- Natural language market analysis via modular LLM architecture
- Advanced prediction engine with sophisticated confidence intervals
- Sentiment detection and classification
- Momentum scoring system
- Multi-model prediction with resilient fallback systems
- Technical indicator fusion for comprehensive market analysis

### 💬 Social Media Integration
- Sophisticated X/Twitter engagement system with natural language conversation capabilities
- Advanced content analyzer for identifying high-engagement crypto discussions
- Smart reply prioritization based on relevance, engagement metrics, and conversation state
- Context-aware response generation with sentiment and tone matching
- Trending topic detection and dynamic adaptation
- Cross-timeframe engagement strategy with memory and streak tracking
- Duplicate content prevention with enhanced time-based detection
- Performance tracking and weekly automated summaries

### 📈 Prediction Tracking & Accountability
- Historical performance tracking
- Accuracy reporting by token and timeframe
- Smart evaluation of expired predictions
- Weekly performance summaries

## 📊 Example Predictions

| Timeframe | Token | Prediction | Confidence | Actual Outcome | Accuracy |
|-----------|-------|------------|------------|----------------|----------|
| 1 hour    | BTC   | $65,432    | 73%        | $65,380        | 99.92%   |
| 24 hour   | ETH   | $3,219     | 68%        | $3,126         | 97.11%   |
| 7 day     | SOL   | $147.32    | 62%        | $151.85        | 96.98%   |

*Sample prediction data - actual results will vary*

## 🧩 Natural Language Processing Capabilities

The Tokenetics Market Oracle includes advanced NLP for engagement and sentiment analysis:

- **Sentiment Detection**: Multi-dimensional sentiment analysis with confidence scoring
- **Tone Analysis**: Identification of conversational tone for better engagement
- **Opinion Detection**: Classification of opinion types, strength, and directionality
- **Meme & Cultural Reference Detection**: Recognition of crypto community slang and memes
- **Question Classification**: Differentiation between rhetorical, technical, and opinion-seeking questions
- **Conversation Hook Identification**: Detection of engagement opportunities
- **Contextual Relevance Scoring**: Evaluation of content against current market conditions
- **Memetic Response Generation**: Dynamic creation of engaging, culturally relevant responses
- **Trending Topic Detection**: Identification of popular discussion topics in the crypto space

## 🛠️ Technical Architecture

### Core Components

![Component Diagram](https://raw.githubusercontent.com/username/repository/main/docs/component_diagram.png)

*Note: Create and add a component diagram to your repository*

- **Market Data Handler**: Fetches and processes cryptocurrency market data
- **Technical Indicators Engine**: Calculates 15+ technical indicators across timeframes
- **Smart Money Detector**: Identifies institutional activity patterns
- **Natural Language Processor**: Analyzes social content for engagement opportunities
- **Content Analyzer**: Advanced crypto discussion relevance and sentiment detection
- **Analysis Engine**: Performs multi-dimensional market analysis
- **Prediction Engine**: Generates price predictions with confidence intervals and fallback models
- **Social Media Manager**: Handles posting and interaction logic for Twitter/X
- **Timeline Scraper**: Intelligent content discovery and prioritization
- **Reply Handler**: Advanced engagement optimization
- **Memory System**: Tracks conversation state and prediction performance
- **Database Manager**: Stores historical data and prediction performance for tracking

### Key Technologies

- **Python 3.8+**: Core programming language
- **Modular LLM Architecture**: Supporting multiple language models (currently using Anthropic Claude)
- **Selenium**: Web automation for social media interaction
- **CoinGecko API**: Real-time crypto market data
- **SQLAlchemy**: ORM for database operations
- **SQLite**: Local database for tracking and analytics
- **TensorFlow**: Deep learning neural networks for forecasting
- **Scikit-learn**: Machine learning for prediction models
- **Statsmodels**: Statistical time series analysis
- **Pandas/NumPy**: Data manipulation and numerical computing

## 🧠 AI and Machine Learning Models

The Tokenetics Market Oracle leverages multiple forecasting methodologies to ensure robust predictions:

### Statistical Models
- **ARIMA**: Time series forecasting with automatic parameter optimization
- **Holt-Winters**: Exponential smoothing for data with trend and seasonality
- **Weighted Average Forecasts**: Volume-weighted and linearly weighted predictions
- **Moving Average**: Adaptive window sizes based on timeframe

### Machine Learning Models
- **LSTM Neural Networks**: Deep learning for complex pattern recognition
- **Random Forest Regression**: Ensemble learning with feature importance analysis
- **Linear Regression**: Baseline model with coefficient analysis
- **Feature Engineering**: Automated creation of technical indicators as model features

### Ensemble Approach
- **Model Weighting**: Dynamic weighting based on historical performance
- **Confidence Intervals**: Statistical and ML-based confidence calculations
- **Fallback Chain**: Multi-stage fallback system for resilient predictions
- **Prediction Validation**: Comprehensive validation against historical patterns

## 🧠 Resilient Architecture

The Market Oracle is designed with multi-stage fallbacks and resilience features:

- **Enhanced Timeline Scraping**: Multiple detection strategies to ensure reliable data collection
- **Multi-Model Prediction**: Fallback prediction models when primary models have insufficient data
- **Robust Error Handling**: Comprehensive logging and exception management
- **Session Persistence**: Automatic recovery from temporary disconnections
- **Time-Based Duplicate Prevention**: Advanced algorithms to avoid repetitive content
- **Adaptive Error Recovery**: Cooldown mechanisms to prevent repeated failures
- **Graceful Degradation**: Maintains functionality even when components fail
- **Conversation State Tracking**: Memory of user interactions and engagement history
- **Engagement Pattern Recognition**: Identifies optimal conversation opportunities

## 📊 Modular LLM Architecture

The system features a flexible, provider-agnostic language model integration:

- **Provider Abstraction**: Easy switching between different LLM providers
- **Model Configuration**: Customizable model selection (currently using Claude)
- **Context Optimization**: Tailored prompts for prediction and engagement scenarios
- **Performance Tracking**: Monitoring of LLM response quality and generation time
- **FOMO Enhancement**: Smart psychological targeting for engagement optimization
- **Prompt Engineering**: Carefully crafted prompts with technical analysis context
- **Response Parsing**: Robust JSON parsing with fallback mechanisms
- **Streak Tracking**: Performance tracking for credibility enhancement
- **Conversation Focus Detection**: Identifies key topics and conversation hooks
- **Engagement Optimization**: Automatic tone and sentiment matching
- **Memetic Response Generation**: Dynamic meme phrase selection based on market conditions
- **Multi-Timeframe Awareness**: Content adaptation based on prediction timeframe

## 📋 Requirements

- Python 3.8+
- LLM API access (currently configured for Anthropic)
- CoinGecko API access
- Twitter/X account credentials
- ChromeDriver for Selenium

## 🚀 Getting Started

1. Clone the repository
   ```bash
   git clone https://github.com/kingraver/reply_guy.git
   cd reply_guy
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
   *This will install all the necessary dependencies listed in the [requirements.txt](requirements.txt) file*

3. Configure your credentials in `.env`
   ```
   LLM_API_KEY=your_llm_api_key
   TWITTER_USERNAME=your_twitter_username
   TWITTER_PASSWORD=your_twitter_password
   CHROME_DRIVER_PATH=/path/to/chromedriver
   ```

4. Run the bot
   ```bash
   python src/bot.py
   ```

## 🔧 Configuration Options

The system is highly configurable to suit different analysis needs:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `BASE_INTERVAL` | Time between regular checks (seconds) | 3600 |
| `REPLY_CHECK_INTERVAL` | Time between social media reply checks (minutes) | 5 |
| `VOLUME_TREND_THRESHOLD` | Threshold for volume trend detection | 15.0 |
| `SMART_MONEY_ZSCORE_THRESHOLD` | Z-score threshold for smart money detection | 2.0 |
| `CORRELATION_THRESHOLD` | Threshold for correlation significance | 0.75 |
| `TIMEFRAME_FREQUENCIES` | Posting frequency for different timeframes | Varies |

## 📚 Advanced Usage

### Custom Token Tracking

Add or modify tracked tokens in the configuration:

```python
self.target_chains = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'SOL': 'solana',
    # Add your tokens here
}
```

### Custom Trigger Thresholds

Adjust sensitivity for market events:

```python
self.timeframe_thresholds = {
    "1h": {
        "price_change": 3.0,    # 3% price change for 1h predictions
        "volume_change": 8.0,   # 8% volume change
        "confidence": 70,       # Minimum confidence percentage
    },
    # Customize other timeframes
}
```

## 🔮 Future Roadmap

- **Real-time Analytical Charts**: Enhanced visualization of prediction performance and market trends
- **Smart Trading Automation**: Integration with Coinbase/Base trading tools for automated trading strategies
- **Community Management**: Advanced engagement features for building crypto communities
- **On-chain Data Integration**: Incorporate wallet movements and smart contract activity
- **Extended Token Coverage**: Support for additional cryptocurrencies beyond current selection
- **Multi-platform Support**: Expand social media engagement beyond Twitter/X
- **Advanced Prompt Engineering**: Further optimization of LLM interactions for specific market scenarios
- **Adaptive Learning System**: Improvement of prediction accuracy based on historical outcomes
- **Cross-Platform Analytics**: Integration of sentiment data from multiple social platforms
- **Institutional Signal Detection**: Enhanced algorithms for detecting large player movements

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📧 Contact

For questions or support, please reach out to: vividvisions.ai@gmail.com

---

⭐ Star this repository if you find it useful! ⭐
