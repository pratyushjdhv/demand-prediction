import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

df=pd.read_csv("grocery_chain_data.csv")
df['transaction_date'] = pd.to_datetime(df['transaction_date'])

# print(df.head())

# 1. just taking imp fields/features
daily_df = df.groupby(['transaction_date', 'product_name']).agg({'quantity': 'sum','unit_price': 'mean'}).reset_index()

# preprocessing to ensure all dates for each product are present
all_dates = pd.date_range(start=daily_df['transaction_date'].min(), end=daily_df['transaction_date'].max(), freq='D')
unique_products = daily_df['product_name'].unique()
mux = pd.MultiIndex.from_product([all_dates, unique_products], names=['transaction_date', 'product_name'])
daily_df = daily_df.set_index(['transaction_date', 'product_name']).reindex(mux)
daily_df['quantity'] = daily_df['quantity'].fillna(0)
daily_df['unit_price'] = daily_df.groupby(level='product_name')['unit_price'].ffill().bfill()
daily_df = daily_df.reset_index()

daily_df.sort_values(['product_name', 'transaction_date'], inplace=True)

# past year lookback
daily_df['last_year_sales'] = daily_df.groupby('product_name')['quantity'].shift(364)

# past few weeks lok back
daily_df['recent_trend'] = daily_df.groupby('product_name')['quantity'].transform(lambda x: x.rolling(14).mean())


# looking into future with last yearss sales
indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=7)
daily_df['forward_7d_avg'] = daily_df.groupby('product_name')['quantity'].transform(lambda x: x.rolling(window=indexer).mean())

daily_df['last_year_next_7d_avg'] = daily_df.groupby('product_name')['forward_7d_avg'].shift(364)

daily_df.drop(columns=['forward_7d_avg'], inplace=True)

model_data = daily_df.dropna().copy()

# print("Features Created: 'quantity', 'last_year_sales', 'recent_trend', 'last_year_next_7d_avg', 'unit_price'")

# prepping the lstm model here

target_product = 'Tomatoes'
product_data = model_data[model_data['product_name'] == target_product].copy()

features = ['quantity', 'last_year_sales', 'recent_trend', 'last_year_next_7d_avg', 'unit_price']
dataset = product_data[features].values

# Scale data (0 to 1) for the lstm to work
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

X, y = [], []
look_back = 14
pred_ahead = 7

for i in range(len(scaled_data) - look_back - pred_ahead + 1):
    X.append(scaled_data[i:(i + look_back)])
    y.append(scaled_data[(i + look_back):(i + look_back + pred_ahead), 0])

X, y = np.array(X), np.array(y)

# training my baby!!!

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(look_back, len(features))))
model.add(Dense(pred_ahead, activation='relu'))
model.compile(optimizer='adam', loss='mse')

# Training happening here! 

print("Training Model...")
model.fit(X, y, epochs=50, verbose=0) 
print("Training Complete!")

# Making predictions for the next 7 days
last_sequence = scaled_data[-look_back:]
last_sequence = last_sequence.reshape(1, look_back, len(features))

predicted_scaled = model.predict(last_sequence)

# Inverse transform to get actual quantity predictions
dummy_matrix = np.zeros((pred_ahead, len(features)))
dummy_matrix[:, 0] = predicted_scaled[0]
predicted_final = scaler.inverse_transform(dummy_matrix)[:, 0]

# for safety purposes i converted the predictions to int using ceiling func, its better to have a bit extra than none
predicted_safety = np.ceil(predicted_final)
predicted_safety = np.maximum(0, predicted_safety)

# VALUESSSS
forecast_dates = pd.date_range(start=product_data['transaction_date'].max() + pd.Timedelta(days=1), periods=7)

forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'Raw_Model_Output': predicted_final,    # The exact float output from the LSTM
    'Safety_Stock_Order': predicted_safety  # The final order quantity (OBV WITH SAFETY)
})

# exporting json here !

final_json = {
    "product": target_product,
    "prediction_horizon": pred_ahead,
    "forecast": []
}

for i in range(len(forecast_df)):
    date_str = forecast_df['Date'].iloc[i].strftime('%Y-%m-%d')
    raw_value = float(forecast_df['Raw_Model_Output'].iloc[i])
    safe_value = int(forecast_df['Safety_Stock_Order'].iloc[i])
    
    final_json["forecast"].append({
        "date": date_str,
        "model_output": raw_value,
        "recommended_order": safe_value
    })

with open("preds.json", "w") as f:
    json.dump(final_json, f, indent=4)