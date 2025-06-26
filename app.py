import streamlit as st
from stock_data import get_stock_data, preprocess
from ml_model import train_model
import pandas as pd
import matplotlib.pyplot as plt

# Streamlit App
st.set_page_config(page_title="Stock Price Predictor", layout="centered")
st.title("📈 Stock Price Predictor")

# User Input
ticker = st.text_input("Enter stock symbol", "AAPL")

if st.button("Predict"):
    with st.spinner("Fetching and predicting..."):
        # Step 1: Load and preprocess data
        df = get_stock_data(ticker)
        df = preprocess(df)

        # Step 2: Train model and get predictions
        model, X_test, y_test, predictions, metrics = train_model(df)

        # Step 3: Display Data
        st.subheader("📉 Actual vs Predicted Prices")
        result_df = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": predictions
        })
        st.line_chart(result_df)

        # Step 4: Show last prediction
        last_actual = y_test.values[-1]
        last_predicted = predictions[-1]
        st.metric("Last Actual Price", f"${last_actual:.2f}")
        st.metric("Last Predicted Price", f"${last_predicted:.2f}")

        # Step 5: Show Accuracy Metrics
        st.subheader("📊 Model Accuracy")
        st.write(f"**Mean Absolute Error (MAE):** {metrics['mae']:.2f}")
        st.write(f"**Mean Squared Error (MSE):** {metrics['mse']:.2f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {metrics['rmse']:.2f}")
        st.write(f"**R² Score:** {metrics['r2']:.2f}")

        st.success("Prediction Complete!")
