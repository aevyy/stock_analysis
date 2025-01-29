# Libraries
import pandas as pd
import yfinance as yf
from textblob import TextBlob
from newsapi import NewsApiClient
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# --- Fetching News ---
def fetch_news(api_key, company="NVIDIA"):
    newsapi = NewsApiClient(api_key=api_key)
    
    # Calculate date range (last 30 days since NewsAPI free tier limitation)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    articles = newsapi.get_everything(
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
        pub_date = article['publishedAt'][:10]  # Extract YYYY-MM-DD
        headlines.append(article['title'])
        dates.append(pub_date)
    
    news_df = pd.DataFrame({'date': dates, 'headline': headlines})
    news_df['date'] = pd.to_datetime(news_df['date'])
    return news_df

# --- Analyzing Sentiments ---
def analyze_sentiment_finbert(text):
    """Financial sentiment analysis using FinBERT"""
    try:
        tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs[0][1].item()  # Positive probability
    except Exception as e:
        print(f"Error in FinBERT analysis: {str(e)}")
        return 0.5  # Fallback neutral value

def add_sentiment(df):
    """Add both sentiment scores"""
    df['sentiment_textblob'] = df['headline'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['sentiment_finbert'] = df['headline'].apply(analyze_sentiment_finbert)
    return df

# --- Fetching Stock Prices ---
def fetch_stock_prices(ticker, start_date, end_date):
    # Add buffer days to ensure we get data
    start_date = pd.to_datetime(start_date) - pd.Timedelta(days=5)
    end_date = pd.to_datetime(end_date) + pd.Timedelta(days=5)
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No stock data found for {ticker}")
        
        # Create DataFrame with date and closing price
        stock_df = pd.DataFrame()
        stock_df['date'] = data.index
        stock_df['Close'] = data['Close'].values
        
        stock_df['date'] = pd.to_datetime(stock_df['date'])
        return stock_df
    
    except Exception as e:
        raise ValueError(f"Error fetching stock data for {ticker}: {str(e)}")

# --- Combining Data and Analyzing ---
def main():
    # Configure logging
    print("Starting analysis...")
    
    try:
        # Set up file paths
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get Scripts directory
        project_dir = os.path.dirname(script_dir)  # Get parent directory
        data_dir = os.path.join(project_dir, 'data')  # Get data directory path
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print("Created data directory")
        
        # Fetch news data
        api_key = "57a0a93396044df7ae44e6d72f084cfa"
        news_df = fetch_news(api_key=api_key, company="NVIDIA")
        print(f"Retrieved {len(news_df)} news articles")
        
        # Add sentiment analysis
        news_df = add_sentiment(news_df)
        print("Sentiment analysis completed")
        
        # Get date range from news data
        start_date = news_df['date'].min()
        end_date = news_df['date'].max()
        print(f"Analyzing period from {start_date.date()} to {end_date.date()}")
        
        # Fetch stock prices
        stock_df = fetch_stock_prices('NVDA', start_date, end_date)
        print(f"Retrieved stock data with {len(stock_df)} entries")
        
        # Merge data
        merged_df = pd.merge(news_df, stock_df, on='date', how='inner')
        if merged_df.empty:
            raise ValueError("No overlapping dates between news and stock data")
        
        print(f"Successfully merged data with {len(merged_df)} matching entries")
        
        # Sort by date and calculate price changes
        merged_df = merged_df.sort_values('date')
        merged_df['price_change'] = merged_df['Close'].pct_change() * 100
        
        # Calculate correlations
        corr_textblob = merged_df['sentiment_textblob'].corr(merged_df['price_change'])
        corr_finbert = merged_df['sentiment_finbert'].corr(merged_df['price_change'])
        print(f"\nCorrelations:")
        print(f"TextBlob: {corr_textblob:.2f}")
        print(f"FinBERT: {corr_finbert:.2f}")
        
        # Plotting (using TextBlob for visualization to maintain original behavior)
        plt.figure(figsize=(12, 8))
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Plot sentiment bars
        sentiment_bars = ax1.bar(merged_df['date'], merged_df['sentiment_textblob'], 
                               alpha=0.3, color='blue', label='TextBlob Sentiment')
        ax1.set_ylabel('Sentiment Score', color='blue', fontsize=10)
        
        # Plot price change line
        price_line = ax2.plot(merged_df['date'], merged_df['price_change'], 
                            'r-', label='Price Change (%)')
        ax2.set_ylabel('Price Change (%)', color='red', fontsize=10)
        
        # Adjust title and layout
        plt.subplots_adjust(top=0.85)
        plt.suptitle('NVIDIA Stock Price Change vs News Sentiment', 
                    y=0.98, fontsize=14, fontweight='bold')
        
        # Customize axes
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        plt.xticks(rotation=45)
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, 
                 loc='upper left', 
                 bbox_to_anchor=(0.01, 0.99),
                 fontsize=10)
        
        # Add correlation info
        correlation_text = f'Correlations:\nTextBlob: {corr_textblob:.2f}\nFinBERT: {corr_finbert:.2f}'
        plt.figtext(0.02, 0.02, correlation_text, fontsize=10)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save results
        output_csv = os.path.join(data_dir, 'nvidia_news_analysis.csv')
        output_plot = os.path.join(data_dir, 'sentiment_vs_price.png')
        
        merged_df.to_csv(output_csv, index=False)
        print(f"\nData saved to {output_csv}")
        
        plt.savefig(output_plot, bbox_inches='tight', dpi=300)
        print(f"Plot saved as {output_plot}")
        
        plt.show()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()