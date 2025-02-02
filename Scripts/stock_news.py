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

load_dotenv()
api_key = os.getenv('API_KEY')

class StockNewsAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.newsapi = NewsApiClient(api_key=api_key)
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

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

    def add_sentiment(self, df):
        df['sentiment_textblob'] = df['headline'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        df['sentiment_finbert'] = df['headline'].apply(self.analyze_sentiment_finbert)
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
                'RSI': data['RSI'].values.flatten()
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

            output_path = os.path.join(data_dir, 'processed_stock_data.csv')
            merged_df.to_csv(output_path, index=False)
            print(f"Processed data saved to {output_path}")
            return merged_df
        except Exception as e:
            raise Exception(f"Data processing failed: {e}")

def main():
    try:
        data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data')
        csv_path = os.path.join(data_dir, 'processed_stock_data.csv')
        if os.path.exists(csv_path):
            os.remove(csv_path)

        analyzer = StockNewsAnalyzer(api_key)
        merged_df = analyzer.process_data()
        print("Data processing completed successfully!")
        return merged_df
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    print('Starting Script...')
    main()
