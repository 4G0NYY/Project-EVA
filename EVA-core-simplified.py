import yfinance as yf
from newspaper import Article
import requests
import json

#I will most certainly not update this first implementation, since it is absolutly obsolete and I'd have to put in way more time than I'd ever like to

# API endpoint for getting wallet balance
url = 'https://api.crypto-wallet.com/v1/balance'

# Wallet address
address = '0x1234567890abcdef'

# Make GET request to API
response = requests.get(url, params={'address': address})

# Parse response as JSON
data = json.loads(response.text)

# Check for errors in response
if 'error' in data:
    print(f'Error: {data["error"]}')
else:
    # Print out balance
    balance = data['balance']
    print(f'Balance: {balance}')

# API endpoint for making transactions
url = 'https://api.crypto-wallet.com/v1/transactions'

# Wallet address
address = '0x1234567890abcdef'

# Recipient address
recipient =  

# Amount of crypto to send
amount = 

# Create payload for POST request
payload = {'from': address, 'to': recipient, 'amount': amount}

# Make POST request to API
response = requests.post(url, json=payload)

# Parse response as JSON
data = json.loads(response.text)

# Check for errors in response
if 'error' in data:
    print(f'Error: {data["error"]}')
else:
    # Print out transaction information
    tx_id = data['transaction_id']
    print(f'Transaction ID: {tx_id}')



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