import pandas as pd
import numpy as np
import requests
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go

# Function to fetch real-time Bitcoin data from Binance API
def fetch_real_time_data():
    url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=1000"
    response = requests.get(url)
    data = response.json()
    print (data)
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
    df = df[["timestamp", "close"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    print(df)
    return df

# Preprocessing function
def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0,1))
    df_scaled = scaler.fit_transform(df)
    return df_scaled, scaler

# Create sequences for LSTM model
def create_sequences(data, seq_length=30, predict_steps=15):
    X, y = [], []
    for i in range(len(data) - seq_length - predict_steps):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length: i + seq_length + predict_steps, 0])
    return np.array(X), np.array(y)

# Build LSTM model
def build_model(input_shape, predict_steps):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(predict_steps)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train and evaluate model
def train_model(X, y, predict_steps):
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    X_train, y_train = X[:-100], y[:-100]
    X_test, y_test = X[-100:], y[-100:]

    model = build_model((X.shape[1], 1), predict_steps)
    history = model.fit(X_train, y_train, batch_size=16, epochs=20, validation_data=(X_test, y_test))
    
    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    accuracy = 100 - (val_loss * 100)
    return model, accuracy

# Predict function
def predict_next_minutes(model, scaler, predict_steps):
    real_time_data = fetch_real_time_data()
    scaled_data, _ = preprocess_data(real_time_data)
    last_sequence = scaled_data[-30:]
    last_sequence = np.reshape(last_sequence, (1, last_sequence.shape[0], 1))
    predicted_scaled = model.predict(last_sequence)[0]
    predicted_prices = scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).flatten()
    return predicted_prices

# Main pipeline
data = fetch_real_time_data()
data_scaled, scaler = preprocess_data(data)

# Training for 15 min, 30 min, and 1 hour predictions
X_15, y_15 = create_sequences(data_scaled, predict_steps=15)
X_30, y_30 = create_sequences(data_scaled, predict_steps=30)
X_60, y_60 = create_sequences(data_scaled, predict_steps=60)

model_15, accuracy_15 = train_model(X_15, y_15, predict_steps=15)
model_30, accuracy_30 = train_model(X_30, y_30, predict_steps=30)
model_60, accuracy_60 = train_model(X_60, y_60, predict_steps=60)

# Streamlit app
st.markdown("""<h1 style='text-align: center; color: #4CAF50;'>Bitcoin Trend Prediction</h1>""", unsafe_allow_html=True)
st.write("Predicting the price for the next 15, 30, and 60 minutes.")

def display_prediction(model, scaler, predict_steps, accuracy, label):
    predicted_prices = predict_next_minutes(model, scaler, predict_steps)
    future_timestamps = pd.date_range(start=pd.Timestamp.now(), periods=predict_steps, freq='min')
    df_pred = pd.DataFrame({'Time': future_timestamps, 'Predicted Price': predicted_prices})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_pred['Time'], y=df_pred['Predicted Price'], mode='lines+markers', name=label))
    fig.update_layout(title=f'{label} Prediction', xaxis_title='Time', yaxis_title='Price (USD)')
    st.plotly_chart(fig)
    st.write(f"{label} Model Accuracy: {accuracy:.2f}%")

# Display buttons in horizontal layout with equal size and styled colors
col1, col2, col3 = st.columns(3)

if col1.button('Predict 15 Min', key='15min'):
    display_prediction(model_15, scaler, 15, accuracy_15, "15 Min")

if col2.button('Predict 30 Min', key='30min'):
    display_prediction(model_30, scaler, 30, accuracy_30, "30 Min")

if col3.button('Predict 1 Hour', key='1hour'):
    display_prediction(model_60, scaler, 60, accuracy_60, "1 Hour")
