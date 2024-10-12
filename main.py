import requests
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Alpha Vantage API settings
alpha_vantage_api_key = 'ALPHA_VANTAGE_API_KEY'  # Replace with your Alpha Vantage API key
symbol = 'AAPL'  # Example stock symbol, you can change this as needed

# Load FinBERT model
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

# Fetch news from Alpha Vantage
def fetch_alpha_vantage_news(symbol):
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={alpha_vantage_api_key}'
    response = requests.get(url)
    data = response.json()
    if 'feed' in data:
        return pd.DataFrame(data['feed'])
    else:
        print("Error fetching news from Alpha Vantage:", data)
        return pd.DataFrame()

# Classify news as forward-looking or not
def classify_news(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    return "forward-looking" if probabilities[0][1].item() > 0.5 else "not forward-looking"

# Process and classify each piece of news
def process_news(news_df):
    # Ensure correct columns are used
    news_df['classification'] = news_df['summary'].apply(classify_news)
    return news_df[['title', 'summary', 'classification']]

# Fetch and process news from Alpha Vantage
alpha_vantage_news_df = fetch_alpha_vantage_news(symbol)
if not alpha_vantage_news_df.empty:
    alpha_vantage_news_df = process_news(alpha_vantage_news_df)
    print("Alpha Vantage News:")
    print(alpha_vantage_news_df)
else:
    print("No news data available.")
