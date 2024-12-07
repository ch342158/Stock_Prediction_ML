# Using the Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import yfinance as yf
from datetime import datetime, timedelta
import sys
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
from sklearn.metrics import r2_score
import ta
sys.stdout.reconfigure(encoding='utf-8')

def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError("No data found for the given ticker and date range.")
    return data

def preprocess_data(data):
    data = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

def build_optimized_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))  # Increased Dropout
    model.add(BatchNormalization())  # Added Batch Normalization
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.3))  # Increased Dropout
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_optimized_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return model

def make_predictions(model, X_train, X_test, scaler):
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    return train_predict, test_predict

def simulate_future_predictions_with_volatility(model, scaled_data, scaler, future_days, historical_data, time_step=60, scale=0.1):
    """
    Simulates future stock price predictions and adds volatility.

    Args:
        model (Sequential): Trained LSTM model.
        scaled_data (np.array): Scaled historical stock data.
        scaler (MinMaxScaler): Scaler used to transform the data.
        future_days (int): Number of business days to predict.
        historical_data (pd.Series): Original historical stock prices.
        time_step (int): Number of time steps for prediction (default 60).
        scale (float): Scale factor for volatility (default 0.1).

    Returns:
        pd.DataFrame: Future predictions with added volatility.
    """
    future_dates = pd.date_range(start=datetime.now().date() + timedelta(days=1), periods=future_days, freq='B')
    last_60_days = scaled_data[-time_step:]
    last_60_days_scaled = last_60_days.reshape(1, last_60_days.shape[0], 1)

    future_predictions = []
    for _ in range(future_days):
        prediction = model.predict(last_60_days_scaled)
        future_prediction = scaler.inverse_transform(prediction)
        future_predictions.append(future_prediction[0, 0])

        prediction_reshaped = scaler.transform([[future_prediction[0, 0]]]).reshape(1, 1, 1)
        last_60_days_scaled = np.append(last_60_days_scaled[:, 1:, :], prediction_reshaped, axis=1)

    # Add volatility to the predictions
    future_predictions_with_volatility = add_volatility(
        np.array(future_predictions), historical_data, scale=scale
    )

    future_df = pd.DataFrame(future_predictions_with_volatility, index=future_dates, columns=['Future Predicted Stock Price'])
    return future_df



def add_volatility(predictions, historical_data, scale=0.1):
    """
    Adds random noise proportional to historical volatility to the predictions.

    Args:
        predictions (np.array): Predicted stock prices.
        historical_data (pd.Series): Historical stock prices.
        scale (float): Scale factor for the noise (default 0.1).

    Returns:
        np.array: Predictions with added volatility.
    """
    volatility = np.std(historical_data[-60:])  # Calculate recent volatility
    noise = np.random.normal(0, scale * volatility, size=len(predictions))
    predictions_with_volatility = predictions + noise
    return predictions_with_volatility


def plot_combined_results(data, train_predict, test_predict, time_step):
    plt.figure(figsize=(14, 5))
    plt.plot(data.index, data['Close'], label='Actual Stock Price', color='blue')
    train_dates = data.index[time_step:time_step + len(train_predict)]
    plt.plot(train_dates, train_predict, label='Train Predicted Stock Price', color='green')
    test_dates = data.index[time_step + len(train_predict) + 1:time_step + len(train_predict) + 1 + len(test_predict)]
    plt.plot(test_dates, test_predict, label='Test Predicted Stock Price', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def plot_future_predictions(data, future_df):
    plt.figure(figsize=(14, 5))
    plt.plot(data.index, data['Close'], label='Actual Stock Price')
    plt.plot(future_df.index, future_df['Future Predicted Stock Price'], label='Future Predicted Stock Price', color='red')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def run_predictions():
    ticker = ticker_entry.get()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()
    try:
        future_days = int(future_days_entry.get())
    except ValueError:
        messagebox.showerror("Input Error", "Future days must be an integer.")
        return

    try:
        data = download_data(ticker, start_date, end_date)
    except Exception as e:
        messagebox.showerror("Data Download Error", f"Error downloading data: {e}")
        return

    scaled_data, scaler = preprocess_data(data)

    X, y = create_dataset(scaled_data, time_step=60)
    if X.shape[0] == 0:
        messagebox.showerror("Data Error", "Insufficient data to create training datasets.")
        return
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = build_optimized_model((X_train.shape[1], 1))
    model = train_optimized_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32)

    train_predict, test_predict = make_predictions(model, X_train, X_test, scaler)
    test_predict_actual = evaluate_performance(model, X_test, y_test, scaler)

    plot_combined_results(data, train_predict, test_predict, 60)

    # Use the updated function to add volatility to predictions
    future_df = simulate_future_predictions_with_volatility(model, scaled_data, scaler, future_days, data['Close'], time_step=60)

    plot_future_predictions(data, future_df)

    print("\nFuture Predicted Stock Prices for the Next {} Business Days:".format(future_days))
    print(future_df)


def evaluate_performance(model, X_test, y_test, scaler):
    test_predict = model.predict(X_test)
    test_predict = scaler.inverse_transform(test_predict)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    r2 = r2_score(y_test_actual, test_predict)
    print(f"R² Score on Test Data: {r2:.4f}")

    return test_predict
root = tk.Tk()
root.title("Stock Price Prediction")


ticker_label = ttk.Label(root, text="Ticker Symbol:")
ticker_label.grid(column=0, row=0, padx=10, pady=10, sticky=tk.W)
ticker_entry = ttk.Entry(root)
ticker_entry.grid(column=1, row=0, padx=10, pady=10)

start_date_label = ttk.Label(root, text="Start Date (YYYY-MM-DD):")
start_date_label.grid(column=0, row=1, padx=10, pady=10, sticky=tk.W)
start_date_entry = ttk.Entry(root)
start_date_entry.grid(column=1, row=1, padx=10, pady=10)

end_date_label = ttk.Label(root, text="End Date (YYYY-MM-DD):")
end_date_label.grid(column=0, row=2, padx=10, pady=10, sticky=tk.W)
end_date_entry = ttk.Entry(root)
end_date_entry.grid(column=1, row=2, padx=10, pady=10)

future_days_label = ttk.Label(root, text="Future Days (Business Days):")
future_days_label.grid(column=0, row=3, padx=10, pady=10, sticky=tk.W)
future_days_entry = ttk.Entry(root)
future_days_entry.grid(column=1, row=3, padx=10, pady=10)

run_button = ttk.Button(root, text="Run Predictions", command=run_predictions)
run_button.grid(column=0, row=4, columnspan=2, padx=10, pady=10)

root.mainloop()