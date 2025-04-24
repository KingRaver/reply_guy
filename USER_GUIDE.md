# Tokenetics Market Oracle - User Guide

This guide provides detailed instructions on using the Tokenetics Market Oracle system to generate cryptocurrency market insights, predictions, and social media engagement.

## Table of Contents

- [System Overview](#system-overview)
- [Getting Started](#getting-started)
- [Basic Operations](#basic-operations)
- [Multi-Timeframe Analysis](#multi-timeframe-analysis)
- [Social Media Engagement](#social-media-engagement)
- [Understanding Predictions](#understanding-predictions)
- [Analyzing Results](#analyzing-results)
- [Configuration Options](#configuration-options)
- [Monitoring and Logging](#monitoring-and-logging)
- [Performance Tracking](#performance-tracking)
- [Advanced Features](#advanced-features)
- [Best Practices](#best-practices)
- [FAQ](#faq)

## System Overview

Tokenetics Market Oracle is an integrated system that:

1. **Collects market data** from multiple sources
2. **Analyzes price patterns** using technical indicators
3. **Generates predictions** across multiple timeframes (1h, 24h, 7d)
4. **Engages with social media** by posting analyses and replying to relevant discussions
5. **Tracks prediction accuracy** for continuous improvement

## Getting Started

### First Run

After completing the installation steps in [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md), follow these steps for your first run:

1. Verify your configuration in the `.env` file
2. Run the system in debug mode for the first time:
   ```bash
   python bot.py --debug
   ```
3. Monitor the console output for any errors or warnings
4. Check the logs directory for detailed operational information

### Initial Configuration

The system begins with default settings, but you may want to customize it based on your specific needs:

- **Target tokens:** By default, the system tracks BTC, ETH, SOL, and other major cryptocurrencies
- **Timeframes:** The system operates across 1-hour, 24-hour, and 7-day timeframes
- **Posting schedule:** Initial configuration posts hourly analyses with timeframe rotation

## Basic Operations

### Starting the System

```bash
# Standard operation
python bot.py

# Debug mode with verbose output
python bot.py --debug

# Specify custom config file
python bot.py --config my_custom_config.py
```

### Stopping the System

The system can be stopped safely by:

- Pressing `Ctrl+C` in the console
- Sending a SIGTERM signal if running as a service

The system performs a graceful shutdown, completing current operations and saving state before exiting.

### Monitoring Operation

While running, the system provides:

- Console output with key operations
- Log files in the `logs/` directory
- Database records of activities and predictions

## Multi-Timeframe Analysis

The system performs analysis across three key timeframes, each with different characteristics:

### 1-Hour Analysis (Short Term)

- **Purpose:** Immediate market movements for short-term traders
- **Features:** Focuses on price momentum, volume spikes, and short-term patterns
- **Indicators:** Emphasizes RSI, MACD, and Bollinger Bands
- **Update Frequency:** Hourly checks with triggered updates on significant movements

### 24-Hour Analysis (Daily)

- **Purpose:** Day trading guidance and medium-term price targets
- **Features:** Incorporates market correlation, sector performance, and daily patterns
- **Indicators:** Adds ADX and Ichimoku Cloud components
- **Update Frequency:** Every 6 hours with price/volume change triggers

### 7-Day Analysis (Weekly)

- **Purpose:** Swing trading opportunities and trend confirmation
- **Features:** Includes sector rotation detection, macro trend analysis
- **Indicators:** Emphasizes pivot points, key support/resistance levels
- **Update Frequency:** Daily updates with broader market context

### Interpreting Timeframe Rotation

The system automatically rotates between timeframes based on:

- Scheduled intervals
- Market volatility triggers
- Significant price/volume changes
- Smart money movement detection

This rotation ensures users receive the most relevant analysis for current market conditions.

## Social Media Engagement

### Types of Posts

The system creates several types of social media content:

1. **Market Analysis:** Technical and fundamental insights on specific tokens
2. **Price Predictions:** Forecasts with confidence intervals and rationales
3. **Correlation Reports:** Market-wide correlation matrices showing relationships between tokens
4. **Weekly Summaries:** Performance reviews and prediction accuracy reports
5. **Replies:** Context-aware responses to market-related discussions

### Reply Functionality

The system intelligently engages with market-related discussions by:

1. **Finding relevant posts** about cryptocurrencies and market analysis
2. **Analyzing content** for sentiment, questions, and engagement opportunities
3. **Generating contextual replies** using market data and prediction insights
4. **Prioritizing engagement** based on relevance and potential impact

### How Reply Targeting Works

The system prioritizes replies to:

- Posts about tokens it's actively tracking
- Technical analysis discussions
- Questions about price movements
- High-engagement market conversations
- Posts from consistent community members

## Understanding Predictions

### Prediction Components

Each prediction includes:

- **Target Price:** Exact price forecast for the specified timeframe
- **Confidence Level:** Percentage indicating prediction certainty
- **Price Range:** Lower and upper bounds forming a confidence interval
- **Percent Change:** Expected percentage movement from current price
- **Sentiment:** BULLISH, BEARISH, or NEUTRAL classification
- **Rationale:** Brief explanation of key factors influencing the prediction
- **Key Factors:** Specific indicators or events supporting the prediction

### Example Prediction

```
#BTC 24HR PREDICTION:

BULLISH ALERT
Target: $65,432.75 (+2.31%)
Range: $64,567.80 - $66,245.30
Confidence: 73%

Multiple indicators showing accumulation patterns with bullish MACD crossover and strong volume profile.

Accuracy: 78.5% on 47 predictions
```

### Confidence Interpretation

The confidence percentage reflects:

- **Historical model accuracy** for the specific token and timeframe
- **Technical indicator agreement** across multiple analysis methods
- **Market volatility** (lower confidence in highly volatile conditions)
- **Data quality** and availability for the analysis

Generally:
- **>75%:** High confidence prediction
- **65-75%:** Moderate confidence
- **<65%:** Lower confidence requiring additional verification

## Analyzing Results

### Accuracy Tracking

The system automatically evaluates predictions when they expire:

1. The actual price at the end of the timeframe is recorded
2. Prediction is marked as correct if the actual price falls within the predicted range
3. Accuracy percentage is calculated as proximity to the exact predicted price
4. Performance statistics are updated for the token and timeframe

### Performance Metrics

The system tracks several performance metrics:

- **Accuracy Rate:** Percentage of correct predictions overall
- **Token-Specific Accuracy:** Performance broken down by individual tokens
- **Timeframe Performance:** Comparative accuracy across different timeframes
- **Deviation Analysis:** Average deviation between predicted and actual prices
- **Volume-Price Correlation:** Relationship between trading volume and price movement

### Viewing Performance Data

Performance data is available through:

1. **Weekly Summary Posts:** Automated social media posts with accuracy statistics
2. **Database Queries:** Direct access to the SQLite database tables
3. **Log Files:** Detailed prediction and evaluation logging

## Configuration Options

### Token Configuration

To modify the tokens being tracked:

1. Edit the `target_chains` dictionary in `bot.py`:
   ```python
   self.target_chains = {
       'BTC': 'bitcoin',
       'ETH': 'ethereum',
       'SOL': 'solana',
       # Add your custom tokens
       'CUSTOM': 'custom-token-id',
   }
   ```
   
2. Use the correct CoinGecko API identifier as the value for each token symbol

### Timeframe Settings

Customize timeframe analysis and posting frequency:

```python
# In bot.py
self.timeframe_posting_frequency = {
    "1h": 1,    # Every hour
    "24h": 6,   # Every 6 hours
    "7d": 24    # Once per day
}
```

### Sensitivity Adjustment

Fine-tune the system's triggers for generating new analyses:

```python
# In bot.py
self.timeframe_thresholds = {
    "1h": {
        "price_change": 3.0,    # 3% price change for 1h predictions
        "volume_change": 8.0,   # 8% volume change
        "confidence": 70,       # Minimum confidence percentage
        "fomo_factor": 1.0      # FOMO enhancement factor
    },
    # Customize other timeframes
}
```

### Social Engagement Settings

Modify reply behavior and engagement parameters:

```python
# Reply frequency and limits
self.reply_check_interval = 5   # Check every 5 minutes
self.max_replies_per_cycle = 10  # Maximum 10 replies per cycle
self.reply_cooldown = 5         # 5 minute cooldown between reply cycles
```

## Monitoring and Logging

### Log Files

The system creates several log files:

- **win.log:** Main system log with operational information
- **analysis/market_analysis.log:** Detailed market analysis records
- **claude.log:** LLM integration activity
- **coingecko.log:** API interactions and market data retrieval

### Understanding Log Messages

Log entries follow this format:
```
[TIMESTAMP] [LEVEL] [COMPONENT] - Message
```

Log levels include:
- **DEBUG:** Detailed debugging information
- **INFO:** General operational information
- **WARNING:** Minor issues that don't affect core functionality
- **ERROR:** Significant problems affecting system operation
- **CRITICAL:** Severe errors that may cause system failure

### Common Log Messages

- `Starting check for posts to reply to`: System is scanning for reply opportunities
- `Successfully posted 24h prediction for BTC`: A prediction post was created
- `Token vs Market Analysis - BTC (24h) error`: Problem with specific analysis component
- `Evaluated 5 expired 24h predictions`: Prediction evaluation completed

## Performance Tracking

### Accuracy Statistics

The system maintains comprehensive accuracy statistics in the database:

- `prediction_performance` table tracks accuracy by token and timeframe
- `prediction_outcomes` records individual prediction results
- `timeframe_metrics` compares performance across different timeframes

### Streak Tracking

The system also tracks prediction streaks:

- Consecutive correct predictions
- Maximum streak history
- Accuracy rate over time

This information is used to enhance credibility in social media engagement.

## Advanced Features

### Smart Money Detection

The system identifies potential institutional activity through:

- **Volume anomalies:** Unusual trading volumes compared to historical averages
- **Price-volume divergence:** Price movements that don't align with volume patterns
- **Z-score thresholds:** Statistical detection of outlier market behavior
- **Volume clusters:** Sequences of elevated volume indicating accumulation

### Market Correlation Analysis

The system generates correlation matrices showing relationships between different tokens:

- Positive correlations indicating tokens moving together
- Negative correlations showing inverse relationships
- Sector-specific correlations (Layer 1s, DeFi, etc.)
- Correlation changes over time

### FOMO Enhancement

The system includes a "FOMO Enhancement" feature that:

- Adjusts prediction ranges to create more engaging content
- Emphasizes potential significant price movements
- Tailors content to create psychological engagement
- Maintains statistical validity while optimizing for social engagement

## Best Practices

### Optimal Usage

For best results:

1. **Run continuously:** The system performs better with uninterrupted market data
2. **Regular maintenance:** Check logs weekly and clear old database records
3. **Performance monitoring:** Review accuracy statistics monthly to identify trends
4. **Configuration tuning:** Adjust thresholds based on market volatility

### Resource Management

The system is designed to run efficiently, but consider:

- **Memory usage:** Typically 300-500MB depending on activity
- **CPU utilization:** Spikes during analysis cycles, low during idle periods
- **Disk space:** Log rotation to prevent excessive storage usage
- **Network bandwidth:** Regular API calls require stable internet connection

### Security Considerations

- **API keys:** Rotate API keys periodically and use restricted permissions
- **Credentials:** Use a dedicated social media account for the bot
- **Database backup:** Regularly backup the SQLite database file

## FAQ

### How accurate are the predictions?

Prediction accuracy varies by token and timeframe. Typically:
- 1-hour predictions: 70-80% accuracy
- 24-hour predictions: 65-75% accuracy
- 7-day predictions: 60-70% accuracy

Accuracy is higher during stable market conditions and lower during high volatility.

### Why does the system post more about certain tokens?

The system prioritizes posts based on:
- Market activity and significant price/volume changes
- Technical indicator signals
- Historical prediction accuracy
- Community engagement patterns

### Can I add custom tokens?

Yes, you can add any token supported by the CoinGecko API by updating the `target_chains` dictionary in `bot.py`. Ensure you use the correct CoinGecko API identifier for the token.

### How does the system avoid duplicate content?

The system uses sophisticated duplicate detection:
- Time-based thresholds for similar content
- Content similarity scoring
- Rotating focus on different tokens and timeframes
- Memory of recent posts and analyses

### What happens if an API is unavailable?

The system includes robust fallback mechanisms:
- CoinGecko API issues: Uses cached market data and reduces update frequency
- LLM API issues: Falls back to template-based content generation
- Twitter/X connectivity: Queues posts for later submission

### How do I interpret the technical indicators?

The system uses standard technical analysis interpretations:
- **RSI:** Values >70 indicate overbought, <30 indicate oversold
- **MACD:** Signal line crossovers indicate potential trend changes
- **Bollinger Bands:** Price touching upper/lower bands suggests potential reversals
- **Stochastic Oscillator:** Values >80 indicate overbought, <20 indicate oversold

---

If you have additional questions not covered in this guide, please reach out to vividvisions.ai@gmail.com.
