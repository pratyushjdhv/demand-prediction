from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

app = Flask(__name__)

# --- DATA PREPARATION ---q
df = pd.read_csv("grocery_chain_data.csv")
df['transaction_date'] = pd.to_datetime(df['transaction_date'])

# Aggregate to daily level
daily_df = df.groupby(['transaction_date', 'product_name']).agg({'quantity': 'sum', 'unit_price': 'mean'}).reset_index()

# Ensure all dates exist for all products
all_dates = pd.date_range(start=daily_df['transaction_date'].min(), end=daily_df['transaction_date'].max(), freq='D')
unique_products = daily_df['product_name'].unique()
mux = pd.MultiIndex.from_product([all_dates, unique_products], names=['transaction_date', 'product_name'])

daily_df = daily_df.set_index(['transaction_date', 'product_name']).reindex(mux)
daily_df['quantity'] = daily_df['quantity'].fillna(0)
daily_df['unit_price'] = daily_df.groupby(level='product_name')['unit_price'].ffill().bfill()
daily_df = daily_df.reset_index()
daily_df.sort_values(['product_name', 'transaction_date'], inplace=True)

# Feature Engineering
daily_df['last_year_sales'] = daily_df.groupby('product_name')['quantity'].shift(364)
daily_df['recent_trend'] = daily_df.groupby('product_name')['quantity'].transform(lambda x: x.rolling(14).mean())

indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=7)
daily_df['forward_7d_avg'] = daily_df.groupby('product_name')['quantity'].transform(lambda x: x.rolling(window=indexer).mean())
daily_df['last_year_next_7d_avg'] = daily_df.groupby('product_name')['forward_7d_avg'].shift(364)
daily_df.drop(columns=['forward_7d_avg'], inplace=True)

model_data_global = daily_df.dropna().copy()
print("Data Ready.")

# --- ROUTES ---

@app.route('/')
def home():
    # This serves the HTML file from the 'templates' folder
    return render_template('index.html')

@app.route('/get_products', methods=['GET'])
def get_products():
    """Return list of products for the dropdown"""
    products = unique_products.tolist()
    return jsonify(products)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    target_product = data.get('product')
    
    if target_product not in unique_products:
        return jsonify({"error": "Product not found"}), 404

    # 1. Filter data for the chosen product
    product_data = model_data_global[model_data_global['product_name'] == target_product].copy()
    
    features = ['quantity', 'last_year_sales', 'recent_trend', 'last_year_next_7d_avg', 'unit_price']
    dataset = product_data[features].values

    # 2. Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # 3. Prepare sequences
    X, y = [], []
    look_back = 14
    pred_ahead = 7

    for i in range(len(scaled_data) - look_back - pred_ahead + 1):
        X.append(scaled_data[i:(i + look_back)])
        y.append(scaled_data[(i + look_back):(i + look_back + pred_ahead), 0])

    X, y = np.array(X), np.array(y)

    # 4. Train the model (Training on the fly for the dashboard demo)
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(look_back, len(features))))
    model.add(Dense(pred_ahead, activation='relu'))
    model.compile(optimizer='adam', loss='mse')
    
    model.fit(X, y, epochs=30, verbose=0) 

    # 5. Make Prediction
    last_sequence = scaled_data[-look_back:]
    last_sequence = last_sequence.reshape(1, look_back, len(features))
    predicted_scaled = model.predict(last_sequence)

    # 6. Convert back to normal numbers
    dummy_matrix = np.zeros((pred_ahead, len(features)))
    dummy_matrix[:, 0] = predicted_scaled[0]
    predicted_final = scaler.inverse_transform(dummy_matrix)[:, 0]

    # 7. Apply Safety Logic (Ceiling)
    predicted_safety = np.ceil(predicted_final)
    predicted_safety = np.maximum(0, predicted_safety)

    # 8. Get Dates
    last_date = product_data['transaction_date'].max()
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)

    # 9. Format Response
    response_data = []
    for i in range(7):
        response_data.append({
            "date": forecast_dates[i].strftime('%Y-%m-%d'),
            "raw_value": float(round(predicted_final[i], 2)),
            "safe_order": int(predicted_safety[i])
        })

    return jsonify({
        "product": target_product,
        "forecast": response_data
    })

if __name__ == '__main__':
    app.run(debug=True)