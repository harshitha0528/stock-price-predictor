from stock_data import get_stock_data

df = get_stock_data("AAPL")  # You can try "GOOGL", "TSLA", etc.
print(df.head())
