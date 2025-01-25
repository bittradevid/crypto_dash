import requests

url = "https://api.binance.com/api/v3/exchangeInfo"
response = requests.get(url)
data = response.json()

# Extraer todos los pares de símbolos
symbols = [symbol['symbol'] for symbol in data['symbols']]
print(symbols)