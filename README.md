# Stock Price Prediction

This project uses Long Short-Term Memory (LSTM) networks to predict stock prices. The model is trained on historical stock data and can make predictions for future stock prices based on the provided data.

## Features

- Downloads historical stock data using `yfinance`.
- Preprocesses the data using `MinMaxScaler` from `scikit-learn`.
- Builds and trains an LSTM model using `Keras` (part of `TensorFlow`).
- Makes predictions on both training and test datasets.
- Simulates future stock price predictions for a specified number of business days.
- Provides a graphical user interface (GUI) using `tkinter` for easy interaction.


**Input Parameters:**

- Ticker Symbol: Enter the stock ticker symbol (e.g., AAPL for Apple Inc.).
- Start Date: Enter the start date of the historical data in the format YYYY-MM-DD.
- End Date: Enter the end date of the historical data in the format YYYY-MM-DD.
- Future Days: Enter the number of business days for which you want to predict future stock prices.
- Run Predictions
- Click the "Run Predictions" button to start the prediction process.


### The application will:
- Download the historical stock data.
- Preprocess the data.
- Build and train the LSTM model.
- Make predictions on the training and test datasets.
- Simulate future stock price predictions.
- Display the results in plots and print the future predictions in the console.


The application will generate plots showing the actual stock prices, training and test predictions, and future predicted stock prices. It will also print the future predictions in the console.

