import streamlit as st
import joblib
import numpy as np
from keras.models import load_model
from datetime import datetime
import pandas as pd

# Load the scaler and models
scaler = joblib.load('scaler.pkl')
lstm_model = load_model('lstm_model.h5')
gru_model = load_model('gru_model.h5')

# Preprocess input and prepare data for prediction
def preprocess_input(date_str, seq_len, scaler, dataset):
    input_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    
    # Filter dataset up to the given date
    dataset['Date_only'] = dataset.index.date  # Extract date part from index
    filtered_data = dataset[dataset['Date_only'] <= input_date]
    
    if len(filtered_data) < seq_len:
        raise ValueError("Not enough data to create a sequence for prediction.")
    
    # Take the last 'SEQ_LEN' days
    input_data = filtered_data['Close'].values[-seq_len:]
    
    # Scale the input data
    scaled_input = scaler.transform(input_data.reshape(-1, 1))
    
    # Reshape for model input
    return scaled_input.reshape(1, seq_len, 1)

# Predict the price using a model
def predict_price(date_str, model, scaler, seq_len, dataset):
    input_data = preprocess_input(date_str, seq_len, scaler, dataset)
    predicted_price = model.predict(input_data)
    return scaler.inverse_transform(predicted_price)[0][0]

# Streamlit UI
st.title("Cryptocurrency Price Prediction")

# Input for the date (you can customize the format)
date_input = st.date_input("Enter Date for Prediction", datetime.today())

# SEQ_LEN constant (You can adjust this based on your dataset and model)
SEQ_LEN = 50  # For example, using 60 days to predict

# Load the dataset
dataset = pd.read_csv("dataset.csv", index_col="Date", parse_dates=True)

# Predict with LSTM
if st.button("Predict with LSTM"):
    try:
        predicted_price_lstm = predict_price(str(date_input), lstm_model, scaler, SEQ_LEN, dataset)
        st.write(f"Predicted price (LSTM) on {date_input}: {predicted_price_lstm:.2f}")
    except ValueError as e:
        st.error(str(e))

# Predict with GRU
if st.button("Predict with GRU"):
    try:
        predicted_price_gru = predict_price(str(date_input), gru_model, scaler, SEQ_LEN, dataset)
        st.write(f"Predicted price (GRU) on {date_input}: {predicted_price_gru:.2f}")
    except ValueError as e:
        st.error(str(e))
