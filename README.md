# 🚀 Crypto Analysis Agent

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Anthropic Claude](https://img.shields.io/badge/AI-Anthropic%20Claude-9cf)](https://www.anthropic.com/)
[![CoinGecko API](https://img.shields.io/badge/API-CoinGecko-brightgreen)](https://www.coingecko.com/en/api)

An advanced cryptocurrency analysis agent that leverages AI and market data to provide intelligent market insights, predictions, and social media engagement. The agent combines real-time market data with Anthropic's Claude AI to generate high-quality analysis across multiple timeframes.

## ✨ Features

### 🔍 Multi-Timeframe Analysis
- **1-hour forecasts** for short-term traders
- **24-hour predictions** for day traders
- **7-day outlooks** for swing traders
- Automatic rotation between timeframes based on market activity

### 📊 Advanced Market Analytics
- Smart money movement detection
- Volume pattern analysis
- Market correlation mapping
- Price-volume divergence detection
- Capital rotation identification
- Comparative market performance

### 🤖 AI-Powered Insights
- Natural language market analysis via Claude API
- Smart prediction engine with confidence intervals
- Sentiment detection and classification
- Momentum scoring system

### 💬 Social Media Integration
- Automated posting of market insights
- Intelligent reply system to engage with other users
- Duplicate content prevention
- Performance tracking and weekly summaries

### 📈 Prediction Tracking & Accountability
- Historical performance tracking
- Accuracy reporting by token and timeframe
- Smart evaluation of expired predictions
- Weekly performance summaries

## 🛠️ Technical Architecture

![Architecture Overview]((https://github.com/KingRaver/reply_guy/blob/main/architecture.txt))

### Core Components

- **Market Data Handler**: Fetches and processes cryptocurrency market data
- **Analysis Engine**: Performs technical analysis on multiple timeframes
- **Prediction Engine**: Generates price predictions with confidence intervals
- **Social Media Manager**: Handles posting and interaction logic
- **Database Manager**: Stores historical data and prediction performance

### Key Technologies

- **Python 3.8+**: Core programming language
- **Anthropic Claude API**: AI-powered analysis generation
- **Selenium**: Web automation for social media interaction
- **CoinGecko API**: Real-time crypto market data
- **SQLite**: Local database for tracking and analytics

## 📋 Requirements

- Python 3.8+
- Anthropic API key
- CoinGecko API access
- Twitter account credentials
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

3. Configure your credentials in `config.py`
   ```python
   # Example configuration
   CLAUDE_API_KEY = "your_anthropic_api_key"
   TWITTER_USERNAME = "your_twitter_username"
   TWITTER_PASSWORD = "your_twitter_password"
   ```

4. Run the bot
   ```bash
   python reply_guy.py
   ```

## 🔧 Configuration Options

The agent is highly configurable to suit different analysis needs:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `BASE_INTERVAL` | Time between regular checks (seconds) | 3600 |
| `VOLUME_WINDOW_MINUTES` | Window for volume analysis | 60 |
| `VOLUME_TREND_THRESHOLD` | Threshold for volume trend detection | 15.0 |
| `SMART_MONEY_ZSCORE_THRESHOLD` | Z-score threshold for smart money detection | 2.0 |
| `CORRELATION_THRESHOLD` | Threshold for correlation significance | 0.75 |
| `TWEET_CONSTRAINTS` | Character limits for Twitter posts | Varies |

## 📊 Performance Metrics

The agent tracks its prediction performance across timeframes and tokens:

```
== 1 HOUR PREDICTIONS ==
Overall Accuracy: 68.7% (123/179)

Top Performers:
#SOL: 73.4% (32 predictions)
#ETH: 70.2% (41 predictions)
#BTC: 69.5% (52 predictions)

== 24 HOUR PREDICTIONS ==
Overall Accuracy: 65.3% (81/124)
...
```

## 📚 Advanced Usage

### Custom Token Tracking

Add or modify tracked tokens in the initialization:

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

- **On-chain data integration**: Incorporate wallet movements and smart contract activity
- **Sentiment analysis**: Monitor social media sentiment across platforms
- **NFT market integration**: Track NFT market data alongside cryptocurrencies
- **DeFi protocol analytics**: Monitor DeFi protocols for yield opportunities
- **Multi-platform support**: Expand to additional social media platforms

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/crypto-analysis-bot/issues).

## 📧 Contact

For questions or support, please reach out to: vividvisions.ai@gmail.com

---

⭐ Star this repository if you find it useful! ⭐
