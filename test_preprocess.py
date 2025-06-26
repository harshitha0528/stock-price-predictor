from stock_data import get_stock_data, preprocess, visualize_data

df = get_stock_data("AAPL")
df = preprocess(df)
visualize_data(df)

print(df.head())

