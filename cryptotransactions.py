import requests
import json


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
