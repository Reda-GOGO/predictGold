import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Load and preprocess the data
data = pd.read_csv('gold_data.csv', parse_dates=['Date'])
data.sort_values('Date', inplace=True)
data.set_index('Date', inplace=True)

# Select only the required columns and ensure proper type conversion
df = data[['Price', 'Open', 'High', 'Low']].astype(float)

# Scale the features to the range [0, 1]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# 2. Create sequences from the data


def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


window_size = 5  # Number of days to look back
X, y = create_sequences(scaled_data, window_size)

# Split the data into training and test sets (e.g., 80% training)
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# 3. Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='tanh', input_shape=(window_size, 4)))
model.add(Dense(4))  # Output layer for Price, Open, High, Low
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16,
          validation_data=(X_test, y_test))

# 4. Forecast future days using a recursive (rolling window) approach
future_days = 5  # Number of future days to predict
last_window = scaled_data[-window_size:]  # Get the last available window
predictions = []
current_window = last_window.copy()

for i in range(future_days):
    # Predict the next day
    pred = model.predict(np.expand_dims(current_window, axis=0))
    predictions.append(pred[0])
    # Slide the window to include the new prediction
    current_window = np.vstack((current_window[1:], pred))

predictions = np.array(predictions)
# Inverse transform the scaled predictions to original scale
predictions_inverse = scaler.inverse_transform(predictions)

# Create a date range for the predictions
last_date = df.index[-1]
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1), periods=future_days)

# Build the output DataFrame
forecast_df = pd.DataFrame(predictions_inverse, columns=[
                           'Price', 'Open', 'High', 'Low'], index=future_dates)

print(forecast_df)


# # Regular prediction
# python gold_predictor.py
#
# # Add new data and predict
# python gold_predictor.py --new-data 1850.50 1845.75 1855.25 1840.80
#
# # Force retraining
# python gold_predictor.py --force-retrain
