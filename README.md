# 📈 Smart Stock Analyzer

Hey there! 👋 Welcome to my stock analysis project. This is a humble attempt to combine various aspects of stock market analysis into one tool. It's not perfect, but it tries its best to help make informed trading decisions.

## 🌟 What Can It Do?

- 📰 Analyzes news sentiment using multiple models (FinBERT, TextBlob, VADER)
- 📊 Tracks technical indicators (RSI, MACD, Bollinger Bands, etc.)
- 🤖 Uses machine learning to predict price movements
- 📱 Monitors social media sentiment from Reddit and Twitter
- 💼 Performs fundamental analysis of companies
- ⚡ Sends real-time alerts via Telegram and email
- 📉 Manages trading risk automatically
- 🤝 Can execute trades through Alpaca (if you want)

## 🚀 Getting Started

1. Clone this repo
2. Install requirements: `pip install -r requirements.txt`
3. Copy `config.yaml.example` to `config.yaml` and fill in your API keys
4. Run it: `python scripts/stock_news.py`

## ⚙️ Configuration

You'll need some API keys to make everything work:
- News API (for news fetching)
- Twitter API (for social sentiment)
- Reddit API (for social sentiment)
- Telegram Bot Token (for alerts)
- Alpaca API (for trading - optional)

## 📝 Note

This is a work in progress and should not be used as your only source for trading decisions. Always do your own research and understand the risks involved in trading. The tool is meant to assist, not replace human judgment.

## 🤝 Contributing

Found a bug? Have an idea? Feel free to open an issue or submit a PR. I'm always looking to learn and improve!

## 📜 License

MIT License - Feel free to use it however you like!
