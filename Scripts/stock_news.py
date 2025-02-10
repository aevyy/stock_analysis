import os
import torch
import pandas as pd
import yfinance as yf
from textblob import TextBlob
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from transformers import BertTokenizer, BertForSequenceClassification
from dotenv import load_dotenv
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import json
import asyncio
from scripts.visualization import StockVisualizer
from scripts.alerts import AlertSystem
from scripts.ml_predictor import PricePredictor
from scripts.social_sentiment import SocialMediaAnalyzer
from scripts.risk_manager import RiskManager
from scripts.broker_interface import BrokerInterface
from scripts.fundamental_analysis import FundamentalAnalyzer
from scripts.sector_analysis import SectorAnalyzer
from scripts.options_analysis import OptionsAnalyzer

load_dotenv()
api_key = os.getenv('API_KEY')

class StockNewsAnalyzer:
    def __init__(self, api_key, configs):
        self.api_key = api_key
        self.newsapi = NewsApiClient(api_key=api_key)
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.finbert_esg = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg')
        self.finbert_esg_tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-esg')
        self.visualizer = StockVisualizer()
        self.alert_system = AlertSystem(configs['email'], configs['telegram'])
        self.predictor = PricePredictor()
        self.social_analyzer = SocialMediaAnalyzer(configs['reddit'], configs['twitter'])
        self.risk_manager = RiskManager()
        self.broker = BrokerInterface(
            configs['alpaca']['api_key'],
            configs['alpaca']['api_secret']
        )
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.sector_analyzer = SectorAnalyzer()
        self.options_analyzer = OptionsAnalyzer()

    def fetch_news(self, company="NVIDIA"):
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=30)
        articles = self.newsapi.get_everything(
            q=company,
            language='en',
            sort_by='relevancy',
            from_param=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d')
        ).get('articles', [])

        if not articles:
            raise ValueError(f"No news articles found for {company}")

        headlines, dates = [], []
        for article in articles:
            pub_date = article['publishedAt'][:10]
            headlines.append(article['title'])
            dates.append(pub_date)

        news_df = pd.DataFrame({'date': dates, 'headline': headlines})
        news_df['date'] = pd.to_datetime(news_df['date'])
        return news_df

    def analyze_sentiment_finbert(self, text):
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            return probs[0][1].item()
        except Exception as e:
            print(f"Error in FinBERT analysis: {e}")
            return 0.5

    def analyze_esg_sentiment(self, text):
        inputs = self.finbert_esg_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.finbert_esg(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs[0].tolist()

    def add_sentiment(self, df):
        df['sentiment_textblob'] = df['headline'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        df['sentiment_finbert'] = df['headline'].apply(self.analyze_sentiment_finbert)
        df['vader_sentiment'] = df['headline'].apply(lambda x: self.vader_analyzer.polarity_scores(str(x))['compound'])
        
        esg_scores = df['headline'].apply(self.analyze_esg_sentiment)
        df['environmental_score'] = [score[0] for score in esg_scores]
        df['social_score'] = [score[1] for score in esg_scores]
        df['governance_score'] = [score[2] for score in esg_scores]
        
        df['sentiment_consensus'] = df[['sentiment_textblob', 'sentiment_finbert', 'vader_sentiment']].mean(axis=1)
        return df

    def fetch_stock_prices(self, ticker, start_date, end_date):
        """Fetch stock prices with proper buffer for technical indicators"""
        buffer_days = 50
        start_date = pd.to_datetime(start_date) - pd.Timedelta(days=buffer_days)
        end_date = pd.to_datetime(end_date)

        try:
            # Download data with buffer
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                raise ValueError(f"No stock data retrieved for {ticker}")

            # Calculate technical indicators
            data['MA5'] = data['Close'].rolling(window=5).mean()
            data['MA20'] = data['Close'].rolling(window=20).mean()

            # Calculate RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            data['RSI'] = 100 - (100 / (1 + rs))

            # Add more technical indicators
            # MACD
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp1 - exp2
            data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            data['BB_middle'] = data['Close'].rolling(window=20).mean()
            data['BB_upper'] = data['BB_middle'] + 2 * data['Close'].rolling(window=20).std()
            data['BB_lower'] = data['BB_middle'] - 2 * data['Close'].rolling(window=20).std()
            
            # Stochastic Oscillator
            low_min = data['Low'].rolling(window=14).min()
            high_max = data['High'].rolling(window=14).max()
            data['%K'] = (data['Close'] - low_min) * 100 / (high_max - low_min)
            data['%D'] = data['%K'].rolling(window=3).mean()

            # Remove buffer period
            actual_start = pd.to_datetime(start_date) + pd.Timedelta(days=buffer_days)
            data = data[data.index >= actual_start]

            # Ensure data is properly flattened when creating DataFrame
            stock_df = pd.DataFrame({
                'date': data.index.values,  # Explicitly get values
                'Close': data['Close'].values.flatten(),  # Flatten arrays
                'Volume': data['Volume'].values.flatten(),
                'MA5': data['MA5'].values.flatten(),
                'MA20': data['MA20'].values.flatten(),
                'RSI': data['RSI'].values.flatten(),
                'MACD': data['MACD'].values.flatten(),
                'Signal_Line': data['Signal_Line'].values.flatten(),
                'BB_upper': data['BB_upper'].values.flatten(),
                'BB_lower': data['BB_lower'].values.flatten(),
                'Stoch_K': data['%K'].values.flatten(),
                'Stoch_D': data['%D'].values.flatten(),
            })

            return stock_df

        except Exception as e:
            print(f"Debug info - Data shape: {data.shape if 'data' in locals() else 'No data'}")
            raise ValueError(f"Error fetching stock data: {e}")

    @staticmethod
    def calculate_rsi(prices, periods=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=periods).mean()
        avg_loss = loss.rolling(window=periods).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def process_data(self, company="NVIDIA", ticker="NVDA"):
        print("Starting data collection and processing...")

        # Use current working directory as script reference
        script_dir = os.getcwd()
        project_dir = os.path.dirname(script_dir)
        data_dir = os.path.join(project_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)

        try:
            news_df = self.fetch_news(company)
            print(f"Retrieved {len(news_df)} news articles")
            news_df = self.add_sentiment(news_df)

            start_date = news_df['date'].min()
            end_date = news_df['date'].max()
            stock_df = self.fetch_stock_prices(ticker, start_date, end_date)
            print(f"Retrieved stock data with {len(stock_df)} entries")

            if stock_df.empty:
                raise ValueError("No stock data retrieved")

            stock_df['date'] = pd.to_datetime(stock_df['date']).dt.date
            news_df['date'] = pd.to_datetime(news_df['date']).dt.date
            merged_df = pd.merge(news_df, stock_df, on='date', how='inner')
            if merged_df.empty:
                raise ValueError("No overlapping dates between news and stock data")

            merged_df = merged_df.sort_values('date')
            merged_df['price_change'] = merged_df['Close'].pct_change() * 100

            # Add social media sentiment
            social_sentiment = self.social_analyzer.get_reddit_sentiment(ticker)
            social_sentiment = social_sentiment.append(self.social_analyzer.get_twitter_sentiment(ticker))
            
            # Train price prediction model
            self.predictor.train(merged_df)
            next_day_prediction = self.predictor.predict_next_day(merged_df)
            
            # Check for alerts
            alerts = self.alert_system.check_alerts(merged_df, signals)
            for alert in alerts:
                asyncio.run(self.alert_system.send_telegram_alert(alert))
            
            # Generate visualizations
            dashboard = self.visualizer.create_dashboard(merged_df, signals, portfolio)
            dashboard.write_html(os.path.join(data_dir, 'dashboard.html'))
            
            # Perform risk analysis
            risk_metrics = self.risk_manager.check_risk_limits(portfolio)
            
            # Add fundamental analysis
            fundamental_analysis = self.fundamental_analyzer.analyze_company(ticker)
            
            # Update trading signals with fundamental data
            signals['fundamental_signal'] = (fundamental_analysis['fundamental_score'] >= 3).astype(int)
            signals['combined_signal'] = (
                signals['combined_signal'] & 
                signals['fundamental_signal']
            )
            
            # Execute trades if automated trading is enabled
            if self.configs.get('automated_trading', False):
                self.broker.execute_signals(
                    {ticker: {'action': 'buy' if signals['combined_signal'].iloc[-1] else 'sell',
                             'price': merged_df['Close'].iloc[-1],
                             'volatility': merged_df['Close'].pct_change().std()}},
                    self.risk_manager
                )
            
            # Add sector analysis
            sector_analysis = self.sector_analyzer.analyze_sector_performance()
            sector_recommendations = self.sector_analyzer.get_sector_recommendations()
            
            # Add options analysis
            options_sentiment = self.options_analyzer.analyze_options_sentiment(ticker)
            
            # Update trading signals with new data
            signals['sector_signal'] = (ticker in sector_recommendations['top_sectors']).astype(int)
            signals['options_signal'] = (options_sentiment['sentiment'] == "Bullish").astype(int)
            
            # Update combined signal
            signals['combined_signal'] = (
                signals['combined_signal'] & 
                (signals['sector_signal'] | signals['options_signal'])
            )
            
            output_path = os.path.join(data_dir, 'processed_stock_data.csv')
            merged_df.to_csv(output_path, index=False)
            print(f"Processed data saved to {output_path}")
            return merged_df, signals, portfolio, risk_metrics, fundamental_analysis, {
                'sector_analysis': sector_analysis,
                'options_sentiment': options_sentiment
            }
        except Exception as e:
            raise Exception(f"Data processing failed: {e}")

    def generate_trading_signals(self, df):
        signals = pd.DataFrame(index=df.index)
        
        # Technical signals
        signals['ma_crossover'] = (df['MA5'] > df['MA20']).astype(int)
        signals['rsi_oversold'] = (df['RSI'] < 30).astype(int)
        signals['rsi_overbought'] = (df['RSI'] > 70).astype(int)
        signals['macd_crossover'] = (df['MACD'] > df['Signal_Line']).astype(int)
        
        # Sentiment signals
        signals['sentiment_signal'] = (df['sentiment_consensus'] > 0.2).astype(int)
        signals['esg_signal'] = ((df['environmental_score'] + df['social_score'] + df['governance_score'])/3 > 0.5).astype(int)
        
        # Combined signal
        signals['combined_signal'] = (
            signals['ma_crossover'] + 
            signals['sentiment_signal'] + 
            signals['macd_crossover'] + 
            signals['esg_signal']
        ) >= 3  # At least 3 positive signals
        
        return signals

    def backtest_strategy(self, df, signals, initial_capital=100000):
        portfolio = pd.DataFrame(index=df.index)
        portfolio['holdings'] = 0
        portfolio['cash'] = initial_capital
        portfolio['position'] = 0
        
        for i in range(1, len(df)):
            if signals['combined_signal'][i] and portfolio['position'][i-1] == 0:
                # Buy signal
                shares = portfolio['cash'][i-1] // df['Close'][i]
                portfolio['holdings'][i] = shares * df['Close'][i]
                portfolio['cash'][i] = portfolio['cash'][i-1] - shares * df['Close'][i]
                portfolio['position'][i] = 1
            elif not signals['combined_signal'][i] and portfolio['position'][i-1] == 1:
                # Sell signal
                portfolio['cash'][i] = portfolio['cash'][i-1] + portfolio['holdings'][i-1]
                portfolio['holdings'][i] = 0
                portfolio['position'][i] = 0
            else:
                # Hold position
                portfolio['holdings'][i] = portfolio['holdings'][i-1]
                portfolio['cash'][i] = portfolio['cash'][i-1]
                portfolio['position'][i] = portfolio['position'][i-1]
        
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()
        
        return portfolio

    def generate_report(self, df, signals, portfolio):
        report = {
            'sentiment_analysis': {
                'average_sentiment': df['sentiment_consensus'].mean(),
                'sentiment_volatility': df['sentiment_consensus'].std(),
                'esg_scores': {
                    'environmental': df['environmental_score'].mean(),
                    'social': df['social_score'].mean(),
                    'governance': df['governance_score'].mean()
                }
            },
            'technical_analysis': {
                'average_rsi': df['RSI'].mean(),
                'overbought_periods': (df['RSI'] > 70).sum(),
                'oversold_periods': (df['RSI'] < 30).sum(),
                'macd_crossovers': signals['macd_crossover'].diff().abs().sum() // 2
            },
            'trading_performance': {
                'total_return': ((portfolio['total'].iloc[-1] - portfolio['total'].iloc[0]) / 
                                portfolio['total'].iloc[0]) * 100,
                'sharpe_ratio': portfolio['returns'].mean() / portfolio['returns'].std() * np.sqrt(252),
                'max_drawdown': (portfolio['total'] / portfolio['total'].cummax() - 1).min() * 100,
                'win_rate': (portfolio['returns'] > 0).mean() * 100
            }
        }
        return report

def main():
    try:
        analyzer = StockNewsAnalyzer(api_key, configs)
        merged_df, signals, portfolio, risk_metrics, fundamental_analysis, sector_data = analyzer.process_data()
        
        # Generate trading signals
        signals = analyzer.generate_trading_signals(merged_df)
        
        # Backtest strategy
        portfolio = analyzer.backtest_strategy(merged_df, signals)
        
        # Generate and save report
        report = analyzer.generate_report(merged_df, signals, portfolio)
        
        # Save results
        data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data')
        with open(os.path.join(data_dir, 'analysis_report.json'), 'w') as f:
            json.dump(report, f, indent=4)
            
        # Save portfolio performance
        portfolio.to_csv(os.path.join(data_dir, 'portfolio_performance.csv'))
        
        print("Analysis completed successfully!")
        return merged_df, signals, portfolio, risk_metrics, fundamental_analysis, sector_data
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    print('Starting Script...')
    main()
