# Alpha Vantage News Classification Script

This project retrieves financial news for a specified stock ticker using the **Alpha Vantage News Sentiment API** and classifies each article as either *forward-looking* or *not forward-looking*. The classification process leverages a pre-trained **FinBERT** model, designed specifically for financial sentiment analysis. In case of API issues, the script falls back to processing news from a local CSV file.

## Features

1. **News Fetching**: The script retrieves up to 400 news articles for a stock ticker (e.g., AAPL) using Alpha Vantage's API.
2. **News Classification**: Using the **FinBERT** model, each news article is classified as either forward-looking or not, based on the article's summary.
3. **CSV Output**: The output file contains `time_published`, `title`, and `classification` fields.
4. **Fallback to CSV**: If the API fails, the script loads news from a local CSV file and continues with classification.

## How FinBERT Classifies News

The FinBERT model is trained on financial texts and excels at understanding financial sentiment. For this project, we use it to classify whether news articles are forward-looking, meaning they speculate or make predictions about future company or market performance.

To classify each news article, FinBERT tokenizes the text and processes it through a neural network. We apply the **sigmoid function** to the model’s output logits. The **sigmoid** function is suitable for binary classification tasks, producing probabilities between 0 and 1 for each class. If the probability of the "forward-looking" label exceeds 0.5, the article is classified as forward-looking; otherwise, it is considered not forward-looking.

## Requirements

- Python 3.x
- Transformers (Hugging Face)
- pandas
- torch
- requests

## Usage

```bash
python main.py
```
