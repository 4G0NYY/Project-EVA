#OBSOLETE! DON'T USE THIS VERSION!


import yfinance as yf
from newspaper import Article
import get_predictions
import cnn
import gan
import train_gan
import get_stock_data
import train_cnn
import train_xgb_boost
import plot_confusion_matrix
import walletchecker
import cryptotransactions

# Define the ticker symbol for the stock
ticker = "AAPL"

# Get the stock data
stock_data = yf.Ticker(ticker).info

# Initialize variables to keep track of shares and cash
shares_owned = 0
cash = balance

# Continuously check for news articles
while True:
    # Get the latest news articles
    article = Article("https://www.example.com/news%22)
    article.download()
    article.parse()
    article_text = article.text

    # Use a pre-trained natural language processing model to identify relevant information in the article
    predictions = model.predict(article_text)

    # Check if the article is relevant to the stock
    if predictions["ticker"] == ticker:
        # Check if the article is positive or negative
        if predictions["sentiment"] == "positive":
            # Check if we have enough cash to buy shares
            current_price = stock_data["regularMarketPrice"]
            if cash > current_price:
                # Buy shares
                shares_owned = cash // current_price
                cash -= shares_owned * current_price
                print(f"Bought {shares_owned} shares of {ticker} at {current_price}")
        elif predictions["sentiment"] == "negative":
            # Check if we own shares
            if shares_owned > 0:
                # Sell shares
                cash += shares_owned * current_price
                shares_owned = 0
                print(f"Sold {shares_owned} shares of {ticker} at {current_price}")
    time.sleep(600)  # wait for 10 minutes