from stock_data import get_stock_data, preprocess
from ml_model import train_model
import matplotlib.pyplot as plt

# Step 1: Get and preprocess data
df = get_stock_data("AAPL")
df = preprocess(df)

# Step 2: Train model and get predictions
model, X_test, y_test, predictions,metrics = train_model(df)

# Step 3: Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual", color='blue')
plt.plot(predictions, label="Predicted", color='orange')
plt.title("Stock Price Prediction")
plt.xlabel("Index")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()
print("Model Accuracy Metrics:")
print(f"MAE:  {metrics['mae']:.2f}")
print(f"MSE:  {metrics['mse']:.2f}")
print(f"RMSE: {metrics['rmse']:.2f}")
print(f"R²:   {metrics['r2']:.2f}")
