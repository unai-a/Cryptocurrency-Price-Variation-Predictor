import requests
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import optuna
import pickle
from datetime import datetime, timedelta

# Binance API endpoint
BASE_URL = "https://api.binance.com/api/v3"

# Symbols
ACH_SYMBOL = "ACHUSDT"
BTC_SYMBOL = "BTCUSDT"

# Function to fetch cryptocurrency data from Binance
def fetch_cryptocurrency_data(symbol, interval, limit):
    url = f"{BASE_URL}/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                     'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                     'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


# Function to calculate actual minimum variation
def calculate_actual_min_variation(df):
    return ((df['low'].shift(1).astype(float) - df['high'].rolling(window=24).max().shift(1).astype(float)) / df['high'].rolling(window=24).max().shift(1).astype(float) * 100).iloc[-1]


# Function to calculate actual maximum variation
def calculate_actual_max_variation(df):
    return ((df['high'].shift(1).astype(float) - df['low'].rolling(window=24).min().shift(1).astype(float)) / df['low'].rolling(window=24).min().shift(1).astype(float) * 100).iloc[-1]


# Function to prepare data for training
def prepare_data(df, btc_daily_min_variation, btc_daily_max_variation):
    X = df[['open', 'high', 'low', 'close', 'volume']].copy()  # Make a copy to avoid SettingWithCopyWarning
    X['btc_daily_min_variation'] = btc_daily_min_variation
    X['btc_daily_max_variation'] = btc_daily_max_variation
    y_min = ((df['low'].shift(-1).astype(float) - df['high'].rolling(window=24).min().shift(-1).astype(float)) / df['high'].rolling(window=24).min().shift(-1).astype(float) * 100).dropna()
    y_max = ((df['high'].shift(-1).astype(float) - df['low'].rolling(window=24).max().shift(-1).astype(float)) / df['low'].rolling(window=24).max().shift(-1).astype(float) * 100).dropna()
    # Adjust length to match
    X = X.tail(len(y_min))
    return X, y_min, y_max


# Function to train model
def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    study = optuna.create_study(direction='minimize')

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20)
        }
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_error = mean_squared_error(y_val, val_pred)
        return val_error

    study.optimize(objective, n_trials=50)
    best_params = study.best_params
    model = GradientBoostingRegressor(**best_params)
    model.fit(X, y)
    return model


# Fetch cryptocurrency data for ACHUSDT and BTCUSDT
ach_df = fetch_cryptocurrency_data(ACH_SYMBOL, '1h', 1000)
btc_df = fetch_cryptocurrency_data(BTC_SYMBOL, '1d', 2)

# Calculate actual variations for ACHUSDT
ach_actual_min_variation = calculate_actual_min_variation(ach_df)
ach_actual_max_variation = calculate_actual_max_variation(ach_df)

# Convert relevant columns of BTC data to numeric
btc_df['low'] = pd.to_numeric(btc_df['low'])
btc_df['high'] = pd.to_numeric(btc_df['high'])

# Calculate BTC daily variations
btc_daily_min_variation = (
            (btc_df['low'].iloc[-1] - btc_df['high'].iloc[-2]) / btc_df['high'].iloc[-2] * 100)
btc_daily_max_variation = (
            (btc_df['high'].iloc[-1] - btc_df['low'].iloc[-2]) / btc_df['low'].iloc[-2] * 100)

# Prepare data for training ACHUSDT model
X, y_min, y_max = prepare_data(ach_df, btc_daily_min_variation, btc_daily_max_variation)

# Train models
model_min = train_model(X, y_min)
model_max = train_model(X, y_max)

# Save models
with open('model_min.pkl', 'wb') as f:
    pickle.dump(model_min, f)

with open('model_max.pkl', 'wb') as f:
    pickle.dump(model_max, f)

# Predict variations for the next 24 hours
last_data = ach_df.tail(1).copy()  # Make a copy to avoid SettingWithCopyWarning

# Calculate BTC daily variations
btc_daily_min_variation = (
            (btc_df['low'].iloc[-1].astype(float) - btc_df['high'].iloc[-2].astype(float)) / btc_df['high'].iloc[
        -2].astype(float) * 100)
btc_daily_max_variation = (
            (btc_df['high'].iloc[-1].astype(float) - btc_df['low'].iloc[-2].astype(float)) / btc_df['low'].iloc[
        -2].astype(float) * 100)

# Ensure the column names are correct
print(last_data.columns)

# Add BTC daily variations to the DataFrame
last_data['btc_daily_min_variation'] = btc_daily_min_variation
last_data['btc_daily_max_variation'] = btc_daily_max_variation

# Predict variations
predicted_min_variation = -model_min.predict(last_data[['open', 'high', 'low', 'close', 'volume', 'btc_daily_min_variation', 'btc_daily_max_variation']])[0]
predicted_max_variation = abs(model_max.predict(last_data[['open', 'high', 'low', 'close', 'volume', 'btc_daily_min_variation', 'btc_daily_max_variation']])[0])

print("Current Price:", last_data['close'].values[0])
print("Price 24 hours ago:", ach_df['close'].iloc[-25])
print("Volume:", last_data['volume'].values[0])
print("Close Time:", last_data['close_time'].values[0])
print("Quote Asset Volume:", last_data['quote_asset_volume'].values[0])
print("Number of Trades:", last_data['number_of_trades'].values[0])
print("Taker Buy Base Asset Volume:", last_data['taker_buy_base_asset_volume'].values[0])
print("Bitcoin Daily Minimum Variation:", btc_daily_min_variation)
print("Bitcoin Daily Maximum Variation:", btc_daily_max_variation)
print("Actual Minimum Variation (24h):", ach_actual_min_variation)
print("Actual Maximum Variation (24h):", ach_actual_max_variation)
print("Predicted Minimum Variation in the next 24 hours:", predicted_min_variation)
print("Predicted Maximum Variation in the next 24 hours:", predicted_max_variation)
