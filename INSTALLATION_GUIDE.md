# Tokenetics Market Oracle - Installation Guide

This guide provides detailed instructions for setting up and configuring the Tokenetics Market Oracle system. Follow these steps to get your crypto analysis system up and running.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Step 1: Clone the Repository](#step-1-clone-the-repository)
- [Step 2: Set Up Python Environment](#step-2-set-up-python-environment)
- [Step 3: Install Dependencies](#step-3-install-dependencies)
- [Step 4: Configure API Keys and Credentials](#step-4-configure-api-keys-and-credentials)
- [Step 5: Set Up ChromeDriver](#step-5-set-up-chromedriver)
- [Step 6: Database Setup](#step-6-database-setup)
- [Step 7: Testing Your Installation](#step-7-testing-your-installation)
- [Step 8: Running the System](#step-8-running-the-system)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

## Prerequisites

Before beginning installation, ensure you have the following:

- Python 3.8 or higher installed
- pip package manager
- Git
- Access to required API services:
  - Anthropic Claude API account
  - CoinGecko API access
  - Twitter/X account
- Chrome browser installed (for ChromeDriver)

## Step 1: Clone the Repository

```bash
git clone https://github.com/kingraver/reply_guy.git
cd reply_guy
```

## Step 2: Set Up Python Environment

It's recommended to use a virtual environment to avoid dependency conflicts.

### Using venv (Python's built-in virtual environment)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Using Conda (Alternative)

```bash
# Create conda environment
conda create -n tokenetics python=3.8
conda activate tokenetics
```

## Step 3: Install Dependencies

Install all required packages listed in requirements.txt:

```bash
pip install -r requirements.txt
```

This will install:
- Core dependencies (requests, numpy, pandas, etc.)
- Browser automation tools (selenium, webdriver-manager)
- Data processing libraries (matplotlib, scikit-learn, statsmodels)
- Machine learning frameworks (TensorFlow)
- Database tools (SQLAlchemy)
- Utility packages

If you encounter errors with TensorFlow installation, please refer to the [Troubleshooting](#troubleshooting) section.

## Step 4: Configure API Keys and Credentials

Create a `.env` file in the root directory with your API keys and credentials:

```
# LLM Provider
LLM_API_KEY=your_anthropic_api_key

# Twitter/X Credentials
TWITTER_USERNAME=your_twitter_username
TWITTER_PASSWORD=your_twitter_password

# CoinGecko API (leave blank for free tier)
COINGECKO_API_KEY=your_coingecko_api_key

# ChromeDriver
CHROME_DRIVER_PATH=/path/to/chromedriver

# Other Configuration
BASE_INTERVAL=3600
REPLY_CHECK_INTERVAL=5
DEBUG_MODE=False
```

### API Key Acquisition Guide:

1. **Anthropic Claude API Key**:
   - Visit [Anthropic's website](https://www.anthropic.com/)
   - Create an account and navigate to the API section
   - Generate an API key and copy it to your `.env` file

2. **CoinGecko API**:
   - Free tier available without authentication
   - For higher rate limits, visit [CoinGecko Pro](https://www.coingecko.com/en/api/pricing)
   - Sign up and obtain your API key

3. **Twitter/X Credentials**:
   - Use a dedicated account for the bot
   - Ensure your account has completed all verification processes to avoid login issues

## Step 5: Set Up ChromeDriver

ChromeDriver is required for browser automation to interact with Twitter/X.

### Automatic Installation (Recommended)

The `webdriver-manager` package included in requirements.txt can automatically download and manage the ChromeDriver for you. In this case, you can set:

```
CHROME_DRIVER_PATH=AUTO
```

### Manual Installation

If you prefer manual installation:

1. Download the appropriate ChromeDriver version for your Chrome browser:
   - Visit [ChromeDriver Downloads](https://sites.google.com/chromium.org/driver/)
   - Select the version that matches your Chrome browser
   
2. Extract the downloaded file and note the path to the chromedriver executable
   - On Windows: typically `C:\path\to\chromedriver.exe`
   - On macOS/Linux: typically `/path/to/chromedriver`

3. Update your `.env` file with the path:
   ```
   CHROME_DRIVER_PATH=/path/to/chromedriver
   ```

4. Ensure the ChromeDriver executable has proper permissions:
   ```bash
   # On macOS/Linux
   chmod +x /path/to/chromedriver
   ```

## Step 6: Database Setup

The system uses a SQLite database to store market data and prediction history. The database file will be created automatically at first run, but you can also set it up manually:

```bash
# Create data directory if it doesn't exist
mkdir -p data

# Initialize empty database (optional)
touch data/crypto_history.db
```

For database backups and maintenance, you can use:

```bash
# Backup database
cp data/crypto_history.db data/backup/crypto_history.db.bak

# Restore from backup
cp data/backup/crypto_history.db.bak data/crypto_history.db
```

## Step 7: Testing Your Installation

Before running the full system, you can test individual components:

```bash
# Test database connection
python -c "from database import Database; db = Database('data/crypto_history.db'); print('Database connection successful')"

# Test CoinGecko API connection
python -c "from coingecko_handler import CoinGeckoHandler; handler = CoinGeckoHandler(); data = handler.get_market_data({'ids': 'bitcoin'}); print(f'CoinGecko API test - received data: {bool(data)}')"

# Test Anthropic API connection (replace with your actual API key)
python -c "import anthropic; client = anthropic.Client(api_key='your_anthropic_api_key'); response = client.messages.create(model='claude-3-5-sonnet-20240620', max_tokens=10, messages=[{'role': 'user', 'content': 'Say hello'}]); print('Anthropic API test - received response')"
```

## Step 8: Running the System

Once everything is set up, you can run the system:

```bash
# Start the bot
python bot.py

# Or for more verbose output
python bot.py --debug
```

The system will:
1. Initialize the database connection
2. Connect to the CoinGecko API for market data
3. Set up the browser for Twitter/X interaction
4. Begin the analysis and prediction cycle

## Troubleshooting

### TensorFlow Installation Issues

If you encounter issues with TensorFlow installation:

```bash
# For CPU-only version
pip install tensorflow-cpu

# Or for specific version
pip install tensorflow==2.16.2
```

On Apple Silicon (M1/M2) Macs:
```bash
# Install TensorFlow for macOS with Apple Metal support
pip install tensorflow-macos
pip install tensorflow-metal  # For GPU acceleration
```

### ChromeDriver Problems

If you encounter issues with ChromeDriver:

1. Verify Chrome browser version:
   ```bash
   # On Windows
   reg query "HKEY_CURRENT_USER\Software\Google\Chrome\BLBeacon" /v version
   
   # On macOS
   /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --version
   
   # On Linux
   google-chrome --version
   ```

2. Download the exact matching ChromeDriver version
3. Try running with `--no-sandbox` option by adding this to your code:
   ```python
   from selenium.webdriver.chrome.options import Options
   chrome_options = Options()
   chrome_options.add_argument('--no-sandbox')
   ```

### Browser Automation Issues

If the browser automation fails:

1. Try running in headful mode (visible browser) by modifying browser.py:
   ```python
   options.headless = False
   ```

2. Ensure your Twitter/X account doesn't have pending verification:
   - Login manually to check for verification prompts
   - Complete any required verification steps

3. Check for CAPTCHA challenges:
   - If Twitter/X frequently shows CAPTCHAs, log in manually occasionally

### API Rate Limits

- **CoinGecko**: Free tier has limited requests per minute
  - Add delays between requests
  - Consider upgrading to Pro
  
- **Anthropic Claude**: Check your usage and tier limits
  - Implement rate limiting in your code
  - Optimize prompt size to reduce token usage

## Advanced Configuration

### Customizing Analysis Settings

Edit the configuration in `config.py` to adjust:

```python
# Example adjustments
VOLUME_TREND_THRESHOLD = 15.0
SMART_MONEY_ZSCORE_THRESHOLD = 2.0
CORRELATION_THRESHOLD = 0.75

# Timeframe settings
TIMEFRAME_FREQUENCIES = {
    "1h": 1,    # Every hour
    "24h": 6,   # Every 6 hours
    "7d": 24    # Once per day
}
```

### Adding Custom Tokens

To track additional tokens, modify the `target_chains` dictionary in `bot.py`:

```python
self.target_chains = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'SOL': 'solana',
    # Add your custom tokens
    'CUSTOM': 'custom-token-id',
}
```

The key is the symbol displayed in analysis, and the value is the CoinGecko API identifier.

### Running as a Service

For 24/7 operation, consider setting up the bot as a service:

#### Using Systemd (Linux)

Create a service file at `/etc/systemd/system/tokenetics.service`:

```
[Unit]
Description=Tokenetics Market Oracle
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/reply_guy
ExecStart=/path/to/python /path/to/reply_guy/bot.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl enable tokenetics.service
sudo systemctl start tokenetics.service
sudo systemctl status tokenetics.service
```

#### Using PM2 (Cross-platform)

Install PM2:
```bash
npm install pm2 -g
```

Start the bot with PM2:
```bash
pm2 start bot.py --name tokenetics --interpreter python
pm2 save
pm2 startup
```

---

If you encounter any issues not covered in this guide, please check the project's GitHub issues or contact support at vividvisions.ai@gmail.com.
