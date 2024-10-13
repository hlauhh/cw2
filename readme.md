# Alpha Vantage News Classification Script

This project retrieves financial news for a given stock ticker using the **Alpha Vantage News Sentiment API** and classifies each news article as either *forward-looking* or *not forward-looking*. The classification is done using a pre-trained **FinBERT** model for sentiment analysis. The project includes fallback functionality to read news from a local CSV file in case the API is unavailable.

## Features

1. **News Fetching**: The script fetches up to **400 news articles** for a specific stock ticker (e.g., AAPL) using Alpha Vantage's API.
2. **News Classification**: Each article is classified as either forward-looking or not, based on the article's summary using a **FinBERT** model.
3. **CSV Output**: The output file contains `time_published`, `title`, and `classification` fields.
4. **API Fallback**: If the API fails, the script loads news from a local CSV file and continues processing.

## How FinBERT Classifies News

The FinBERT model is designed for financial sentiment analysis. For this project, it classifies whether news articles are forward-looking, meaning they speculate or predict future company or market performance.

The model tokenizes the text of each news article and processes it through a neural network. The **sigmoid function** is applied to the output logits to produce probabilities between 0 and 1 for each class. If the probability of the "forward-looking" label exceeds 0.5, the article is classified as forward-looking; otherwise, it is considered not forward-looking.

## Requirements

- Python 3.x
- Transformers (Hugging Face)
- pandas
- torch
- requests

## Setup

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Replace the placeholder API key (`YOUR_ALPHA_VANTAGE_API_KEY`) with your own Alpha Vantage API key.

## Usage

Run the main script to fetch news for the given stock ticker and classify them:
```bash
python main.py
```

The script will output a CSV file containing the time of publication, article title, and classification.
