import os
import requests
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Alpha Vantage API settings
alpha_vantage_api_key = 'YOUR_ALPHA_VANTAGE_API_KEY'  # Replace with your Alpha Vantage API key
symbol = 'AAPL'  # Example stock symbol, you can change this as needed
raw_csv_filename = 'raw_alpha_vantage_news.csv'  # CSV file to fall back on if API fails

# Load FinBERT model
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')


# Fetch news from Alpha Vantage and save raw data as CSV
def fetch_alpha_vantage_news(symbol):
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&limit=400&apikey={alpha_vantage_api_key}'
    try:
        response = requests.get(url)
        data = response.json()
        if 'feed' in data:
            df = pd.DataFrame(data['feed'])

            # Save the raw news data to a CSV file
            df.to_csv(raw_csv_filename, index=False)
            print(f"Raw Alpha Vantage News saved to {raw_csv_filename}")
            return df
        else:
            print("Error fetching news from Alpha Vantage:", data)
            return None
    except Exception as e:
        print(f"Failed to fetch data from Alpha Vantage API: {e}")
        return None


# Load news from existing CSV file if API is not available
def load_news_from_csv():
    if os.path.exists(raw_csv_filename):
        print(f"Loading news from local file: {raw_csv_filename}")
        return pd.read_csv(raw_csv_filename)
    else:
        print(f"No local file found: {raw_csv_filename}")
        return pd.DataFrame()  # Return an empty DataFrame if no file is available


# Classify news as forward-looking or not using sigmoid
def classify_news(text):
    max_length = 512
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    outputs = model(**inputs)
    sigmoid = torch.nn.functional.sigmoid(outputs.logits)
    return "forward-looking" if sigmoid[0][1].item() > 0.5 else "not forward-looking"


# Process and classify each piece of news, adding classification to the data
def process_news_and_add_classification(news_df):
    # Add a new column for classification
    news_df['classification'] = news_df['summary'].apply(classify_news)

    # Select only the required columns
    news_df = news_df[['time_published', 'title', 'classification']]

    # Save back to the same CSV file with the classification
    classified_csv_filename = 'alpha_vantage_news_with_classification.csv'
    news_df.to_csv(classified_csv_filename, index=False)
    print(f"Updated news with classification saved to {classified_csv_filename}")
    return news_df


# Main logic to fetch or load news and classify
def main():
    # Try fetching news from Alpha Vantage API
    news_df = fetch_alpha_vantage_news(symbol)

    # If the API fails, load from the local CSV file
    if news_df is None or news_df.empty:
        news_df = load_news_from_csv()

    # If we have news data, classify it
    if not news_df.empty:
        process_news_and_add_classification(news_df)
    else:
        print("No news data available for processing.")


# Run the main function
if __name__ == "__main__":
    main()
