import numpy as np
import pandas as pd
import tensorflow as tf
import os
import joblib
import json
import sys
import argparse
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, InputLayer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.metrics import RootMeanSquaredError
from scipy.stats import linregress

# Suppress TensorFlow warnings and errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Configuration
CONFIG = {
    "window_size": 10,
    "prediction_days": 5,
    "retrain_interval": 7,  # days between retraining
    "min_retrain_samples": 10,  # minimum new samples before retraining
    "model_dir": "model_versions",
    "data_file": "gold_data.csv",
    "scaler_file": "gold_scaler.save",
    "model_version": "v1",
    "features": ['Price', 'Open', 'High', 'Low', 'Returns', 'Volatility', 'MA_5', 'MA_10', 'Price_Change']
}


def initialize_directories():
    """Create necessary directories"""
    os.makedirs(CONFIG["model_dir"], exist_ok=True)
    os.makedirs("data_history", exist_ok=True)


def save_config():
    """Save configuration to file"""
    with open(os.path.join(CONFIG["model_dir"], "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)


def load_config():
    """Load configuration from file if exists"""
    config_path = os.path.join(CONFIG["model_dir"], "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return CONFIG


# Update global config if exists
CONFIG = load_config()
initialize_directories()
save_config()

# 1. Enhanced Data Processing with Feature Engineering


def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'])
    data.sort_values('Date', inplace=True)
    data.set_index('Date', inplace=True)

    # Create technical features
    # Ensure we have base columns
    df = data.reindex(columns=CONFIG["features"][:4])
    for col in CONFIG["features"][4:]:
        if col == 'Returns':
            df['Returns'] = df['Price'].pct_change()
        elif col == 'Volatility':
            df['Volatility'] = (df['High'] - df['Low']) / df['Open']
        elif col == 'MA_5':
            df['MA_5'] = df['Price'].rolling(5).mean()
        elif col == 'MA_10':
            df['MA_10'] = df['Price'].rolling(10).mean()
        elif col == 'Price_Change':
            df['Price_Change'] = df['Price'].diff()

    # Drop NaN values from feature creation
    df = df.dropna()

    return df


def scale_data(df, scaler=None):
    """Scale data with existing scaler or create new one"""
    if scaler is None:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
    else:
        scaled_data = scaler.transform(df)
    return scaled_data, scaler

# 2. Sequence Creation for Single-Day Prediction


def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])  # Predict next day only
    return np.array(X), np.array(y)

# 3. Robust LSTM Model Architecture


def build_lstm_model(input_shape, output_size):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))  # Explicit input layer
    model.add(Bidirectional(
        LSTM(128,
             return_sequences=True,
             kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)
             )
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(
        96,
        return_sequences=True,
        kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(
        64,
        kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)
    ))
    model.add(Dropout(0.3))
    model.add(Dense(48, activation='sigmoid',
              kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
    model.add(Dense(output_size))
    return model

# 4. Custom Loss Function


def high_low_loss(y_true, y_pred):
    # Calculate weighted errors
    high_error = tf.reduce_mean(tf.square(y_true[:, 2] - y_pred[:, 2])) * 4.0
    low_error = tf.reduce_mean(tf.square(y_true[:, 3] - y_pred[:, 3])) * 4.0
    price_error = tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0])) * 1.5
    open_error = tf.reduce_mean(tf.square(y_true[:, 1] - y_pred[:, 1])) * 1.0
    return high_error + low_error + price_error + open_error

# 5. Advanced Market Analysis Agent


class MarketAnalysisAgent:
    def __init__(self):
        self.market_state = "neutral"

    def analyze_market(self, recent_data):
        """Analyze market conditions using statistical features"""
        # Find volatility index
        volatility_idx = CONFIG["features"].index(
            'Volatility') if 'Volatility' in CONFIG["features"] else 4

        # Calculate recent volatility (last 3 days)
        recent_volatility = np.mean(recent_data[:, volatility_idx])

        # Calculate trend direction using linear regression
        price_idx = CONFIG["features"].index('Price')
        recent_prices = recent_data[:, price_idx]
        slope, _, _, _, _ = linregress(
            np.arange(len(recent_prices)), recent_prices)

        # Update market state
        if recent_volatility > 0.04:
            self.market_state = "volatile"
        elif abs(slope) > 0.005:
            self.market_state = "trending"
        else:
            self.market_state = "neutral"

        return self.market_state


def train_model(df, retrain=False):
    """Train or retrain the model"""
    # Scale data
    scaler_path = os.path.join(CONFIG["model_dir"], CONFIG["scaler_file"])

    if retrain and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        scaled_data, scaler = scale_data(df, scaler)
    else:
        scaled_data, scaler = scale_data(df)
        joblib.dump(scaler, scaler_path)

    # Create sequences
    X, y = create_sequences(scaled_data, CONFIG["window_size"])

    # Train-test split
    split = int(0.85 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Build and compile model
    model = build_lstm_model(
        (CONFIG["window_size"], scaled_data.shape[1]), scaled_data.shape[1])
    model.compile(
        optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),
        loss=high_low_loss,
        metrics=[RootMeanSquaredError()]
    )

    # Callbacks - using new .keras format
    model_version = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30,
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.25,
                          patience=15, min_lr=1e-6),
        ModelCheckpoint(
            os.path.join(CONFIG["model_dir"], model_version),
            save_best_only=True,
        )
    ]

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # Update latest model version
    CONFIG["model_version"] = model_version
    save_config()

    return model, scaler


def load_latest_model():
    """Load the latest trained model and scaler"""
    model_path = os.path.join(CONFIG["model_dir"], CONFIG["model_version"])
    scaler_path = os.path.join(CONFIG["model_dir"], CONFIG["scaler_file"])

    # Try to load .keras format first
    if not os.path.exists(model_path) and model_path.endswith('.keras'):
        # Check for .h5 format for backward compatibility
        h5_path = model_path.replace('.keras', '.h5')
        if os.path.exists(h5_path):
            model_path = h5_path
            # Update config to new format for next time
            CONFIG["model_version"] = CONFIG["model_version"].replace(
                '.h5', '.keras')
            save_config()

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None

    # Load model with custom objects
    try:
        model = load_model(
            model_path,
            custom_objects={'high_low_loss': high_low_loss},
            compile=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_next_days(model, scaler, df, days=5):
    """Predict next days using the model"""
    # Get scaled data
    scaled_data, _ = scale_data(df, scaler)

    # Get last window
    last_window = scaled_data[-CONFIG["window_size"]:]
    predictions = []
    current_window = last_window.copy()
    agent = MarketAnalysisAgent()

    for day in range(days):
        # Analyze current market conditions
        market_state = agent.analyze_market(current_window[-5:])

        # Predict next day
        pred = model.predict(np.expand_dims(
            current_window, axis=0), verbose=0)[0]

        # Apply market-state-specific adjustments
        price_idx = CONFIG["features"].index('Price')
        open_idx = CONFIG["features"].index('Open')
        high_idx = CONFIG["features"].index('High')
        low_idx = CONFIG["features"].index('Low')

        price_mid = (pred[price_idx] + pred[open_idx]) / 2
        if market_state == "volatile":
            pred[high_idx] = price_mid + (pred[high_idx] - price_mid) * 1.2
            pred[low_idx] = price_mid - (price_mid - pred[low_idx]) * 1.2
        elif market_state == "trending":
            trend_direction = 1 if pred[price_idx] > current_window[-1,
                                                                    price_idx] else -1
            pred[high_idx] += 0.015 * trend_direction
            pred[low_idx] += 0.015 * trend_direction
        else:
            pred[high_idx] = price_mid + (pred[high_idx] - price_mid) * 0.8
            pred[low_idx] = price_mid - (price_mid - pred[low_idx]) * 0.8

        predictions.append(pred)
        current_window = np.vstack((current_window[1:], pred))

    # Inverse transform predictions
    predictions = np.array(predictions)
    predictions_inverse = scaler.inverse_transform(predictions)

    # Create DataFrame with all features
    predictions_df = pd.DataFrame(
        predictions_inverse,
        columns=CONFIG["features"],
        index=[df.index[-1] + timedelta(days=i+1) for i in range(days)]
    )

    # Add market state to output
    predictions_df['Market_State'] = agent.market_state

    return predictions_df[['Price', 'Open', 'High', 'Low', 'Market_State']]


def add_new_data(new_data):
    """Add new data to the dataset and archive old data"""
    # Archive current data
    archive_name = f"data_history/gold_data_{
        datetime.now().strftime('%Y%m%d')}.csv"
    if os.path.exists(CONFIG["data_file"]):
        os.rename(CONFIG["data_file"], archive_name)

    # Load existing data
    if os.path.exists(archive_name):
        existing_data = pd.read_csv(archive_name)
    else:
        existing_data = pd.DataFrame(columns=['Date'] + CONFIG["features"])

    # Add new data
    new_data_df = pd.DataFrame(
        [new_data], columns=['Date'] + CONFIG["features"])
    updated_data = pd.concat([existing_data, new_data_df], ignore_index=True)

    # Save updated data
    updated_data.to_csv(CONFIG["data_file"], index=False)
    return updated_data


def should_retrain():
    """Check if we should retrain the model"""
    # Check if model exists
    model_path = os.path.join(CONFIG["model_dir"], CONFIG["model_version"])
    if not os.path.exists(model_path):
        return True

    # Check if model version ends with .h5 (old format)
    if CONFIG["model_version"].endswith('.h5'):
        return True

    try:
        # Extract date from model filename
        model_date_str = CONFIG["model_version"].split('_')[1].split('.')[0]
        last_train_date = datetime.strptime(model_date_str, '%Y%m%d')
    except:
        return True

    # Check retrain interval
    if (datetime.now() - last_train_date).days >= CONFIG["retrain_interval"]:
        return True

    # Check if enough new data
    archive_files = [f for f in os.listdir(
        "data_history") if f.startswith("gold_data_")]
    new_samples = 0
    for f in archive_files:
        try:
            file_date = datetime.strptime(f[9:17], '%Y%m%d')
            if file_date > last_train_date:
                df = pd.read_csv(os.path.join("data_history", f))
                new_samples += df.shape[0]
        except:
            continue

    return new_samples >= CONFIG["min_retrain_samples"]


def main():
    parser = argparse.ArgumentParser(description='Gold Price Predictor')
    parser.add_argument('--new-data', nargs=4, metavar=('PRICE', 'OPEN', 'HIGH', 'LOW'),
                        help='Add new data point for today')
    parser.add_argument('--force-retrain', action='store_true',
                        help='Force model retraining')
    args = parser.parse_args()

    # Add new data if provided
    if args.new_data:
        today = datetime.now().strftime('%Y-%m-%d')
        new_record = [today] + [float(x) for x in args.new_data]
        updated_df = add_new_data(new_record)
        print(f"Added new data for {today}")

    # Load data
    try:
        df = load_and_preprocess_data(CONFIG["data_file"])
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Check if we should retrain
    retrain = args.force_retrain or should_retrain()

    # Load or train model
    model, scaler = load_latest_model()
    if model is None or retrain:
        print("Training new model..." if model is None else "Retraining model with new data...")
        try:
            model, scaler = train_model(df, retrain=True)
        except Exception as e:
            print(f"Error during training: {e}")
            sys.exit(1)
    else:
        print("Using existing model")

    # Make predictions
    try:
        predictions = predict_next_days(
            model, scaler, df, CONFIG["prediction_days"])
        print("\nGold Price Forecast:")
        print(predictions)

        # Save predictions
        predictions.to_csv('gold_forecast.csv')
        print("Forecast saved to gold_forecast.csv")
    except Exception as e:
        print(f"Error during forecasting: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


# # Regular prediction
# python gold_predictor.py
#
# # Add new data and predict
# python gold_predictor.py --new-data 1850.50 1845.75 1855.25 1840.80
#
# # Force retraining
# python gold_predictor.py --force-retrain
