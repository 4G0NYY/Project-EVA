import requests
import json

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

#most certainly obsolete too, but oh well