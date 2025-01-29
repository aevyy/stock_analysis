# Libraries
import pandas as pd
import yfinance as yf
from textblob import TextBlob
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import os
from transformers import BertTokenizer, BertForSequenceClassification
import torch

class StockNewsAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.newsapi = NewsApiClient(api_key=api_key)
        
        # Initialize FinBERT
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
    
    def fetch_news(self, company="NVIDIA"):
        """Fetch news from VALID historical dates"""
        # Use actual historical date range
        end_date = datetime.now() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=30)     # Last month
        
        articles = self.newsapi.get_everything(
            q=company,
            language='en',
            sort_by='relevancy',
            from_param=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d')
        )['articles']
        
        if not articles:
            raise ValueError(f"No news articles found for {company}")
        
        headlines = []
        dates = []
        for article in articles:
            pub_date = article['publishedAt'][:10]
            headlines.append(article['title'])
            dates.append(pub_date)
        
        news_df = pd.DataFrame({'date': dates, 'headline': headlines})
        news_df['date'] = pd.to_datetime(news_df['date'])
        return news_df
    
    def analyze_sentiment_finbert(self, text):
        """Financial sentiment analysis using FinBERT"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            return probs[0][1].item()  # Positive probability
        except Exception as e:
            print(f"Error in FinBERT analysis: {str(e)}")
            return 0.5
    
    def add_sentiment(self, df):
        """Add both sentiment scores"""
        df['sentiment_textblob'] = df['headline'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        df['sentiment_finbert'] = df['headline'].apply(self.analyze_sentiment_finbert)
        return df
    
    def fetch_stock_prices(self, ticker, start_date, end_date):
        """Proper technical indicator calculation"""
        calculation_window = max(20, 14) + 5
        start_date = pd.to_datetime(start_date) - pd.Timedelta(days=calculation_window)
        end_date = pd.to_datetime(end_date) + pd.Timedelta(days=5)

        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            
            # Calculate indicators
            data['MA5'] = data['Close'].rolling(window=5, min_periods=1).mean()
            data['MA20'] = data['Close'].rolling(window=20, min_periods=1).mean()
            data['RSI'] = self.calculate_rsi(data['Close'])

            # Filter to original date range
            mask = (data.index >= start_date + pd.Timedelta(days=calculation_window)) \
                & (data.index <= end_date - pd.Timedelta(days=5))
            filtered_data = data.loc[mask]

            # Convert all columns to 1D arrays explicitly
            stock_df = pd.DataFrame({
                'date': filtered_data.index,
                'Close': filtered_data['Close'].values.ravel(),  # Ensure 1D
                'Volume': filtered_data['Volume'].values.ravel(),
                'MA5': filtered_data['MA5'].values.ravel(),
                'MA20': filtered_data['MA20'].values.ravel(),
                'RSI': filtered_data['RSI'].values.ravel()
            })

            return stock_df.dropna()
        
        except Exception as e:
            raise ValueError(f"Stock data error: {str(e)}")
    
    @staticmethod
    def calculate_rsi(prices, periods=14):
        """Improved RSI calculation with epsilon"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling(window=periods).mean()
        avg_loss = loss.rolling(window=periods).mean()

        rs = avg_gain / (avg_loss + 1e-10)  # Preventing division by zero
        return 100 - (100 / (1 + rs))
    
    def process_data(self, company="NVIDIA", ticker="NVDA"):
        """Main processing function to collect and combine all data"""
        print("Starting data collection and processing...")
        
        # Create data directory if it doesn't exist
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        data_dir = os.path.join(project_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Fetch and process data
        news_df = self.fetch_news(company)
        print(f"Retrieved {len(news_df)} news articles")
        
        news_df = self.add_sentiment(news_df)
        print("Sentiment analysis completed")
        
        start_date = news_df['date'].min()
        end_date = news_df['date'].max()
        stock_df = self.fetch_stock_prices(ticker, start_date, end_date)
        print(f"Retrieved stock data with {len(stock_df)} entries")
        
        # Merge data
        merged_df = pd.merge(news_df, stock_df, on='date', how='inner')
        if merged_df.empty:
            raise ValueError("No overlapping dates between news and stock data")
        
        required_columns = ['Volume', 'MA5', 'MA20', 'RSI']
        if not all(col in merged_df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in merged_df.columns]
            raise ValueError(f"Missing critical columns: {missing}")
        
        # Calculate price changes
        merged_df = merged_df.sort_values('date')
        merged_df['price_change'] = merged_df['Close'].pct_change() * 100
        
        # Save processed data
        output_path = os.path.join(data_dir, 'processed_stock_data.csv')
        merged_df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
        
        return merged_df

def main():
    try:
        # Clearing previous outputs
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        if os.path.exists(os.path.join(data_dir, 'processed_stock_data.csv')):
            os.remove(os.path.join(data_dir, 'processed_stock_data.csv'))
            
        api_key = "57a0a93396044df7ae44e6d72f084cfa"
        analyzer = StockNewsAnalyzer(api_key)
        merged_df = analyzer.process_data()
        print("Data processing completed successfully!")
        return merged_df
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()