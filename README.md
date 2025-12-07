# Grocery Demand Prediction System

A machine learning-powered demand forecasting system for grocery chains, using LSTM neural networks to predict product demand and optimize inventory management.

## Overview

This project provides a web-based dashboard that predicts 7-day demand for grocery products using historical sales data. The system employs LSTM (Long Short-Term Memory) networks to analyze time-series patterns and generate safety stock recommendations.

## Features

- **LSTM-Based Forecasting**: Utilizes deep learning to predict product demand
- **Interactive Dashboard**: Vue.js-powered interface for real-time predictions
- **Safety Stock Calculation**: Automatically applies ceiling function for conservative inventory planning
- **Visual Analytics**: Chart.js integration for forecast visualization
- **Multi-Product Support**: Select from various grocery products for individual forecasts
- **Feature Engineering**: Incorporates year-over-year trends and recent sales patterns

## Tech Stack

### Backend
- **Flask**: Web framework for API endpoints
- **TensorFlow/Keras**: LSTM model implementation
- **Pandas & NumPy**: Data processing and manipulation
- **Scikit-learn**: Data normalization and preprocessing

### Frontend
- **Vue.js 3**: Reactive UI framework
- **Chart.js**: Data visualization
- **HTML5/CSS3**: Interface design

## Project Structure

```
.
├── app.py                      # Flask web application with API endpoints
├── prediction.py               # Standalone prediction script
├── grocery_chain_data.csv      # Historical sales data
├── requirements.txt            # Python dependencies
├── preds.json                  # Sample prediction output
└── templates/
    └── index.html              # Dashboard interface
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/pratyushjdhv/demand-prediction.git
cd demand-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Application

Start the Flask server:
```bash
python app.py
```

Access the dashboard at `http://localhost:5000`

### Using the Dashboard

1. Select a product from the dropdown menu
2. Click "Generate Forecast"
3. View the 7-day prediction with:
   - Raw model output (exact LSTM predictions)
   - Safety stock recommendations (ceiling values)
   - Total 7-day requirement
   - Visual chart comparing raw vs. safety stock predictions

### Standalone Prediction Script

Run predictions for a specific product (default: Tomatoes):
```bash
python prediction.py
```

Output will be saved to `preds.json`

## Model Details

### Features Used
- **Quantity**: Current daily sales
- **Last Year Sales**: Sales from 364 days ago
- **Recent Trend**: 14-day rolling average
- **Last Year Next 7D Avg**: Historical 7-day forward average from last year
- **Unit Price**: Average price per unit

### Model Architecture
- **Input**: 14-day lookback window (5 features)
- **LSTM Layer**: 50 units with ReLU activation
- **Output Layer**: Dense layer predicting 7 days ahead
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)

### Training
- **Lookback Period**: 14 days
- **Prediction Horizon**: 7 days
- **Epochs**: 30 (web app) / 50 (standalone script)
- **Scaling**: MinMaxScaler (0-1 normalization)

## API Endpoints

### `GET /`
Returns the main dashboard interface

### `GET /get_products`
Returns list of available products
```json
["Tomatoes", "Lettuce", "Milk", ...]
```

### `POST /predict`
Generates 7-day forecast for selected product

**Request:**
```json
{
  "product": "Tomatoes"
}
```

**Response:**
```json
{
  "product": "Tomatoes",
  "forecast": [
    {
      "date": "2025-12-08",
      "raw_value": 45.32,
      "safe_order": 46
    },
    ...
  ]
}
```

## Data Preprocessing

1. **Date Continuity**: Ensures all dates present for each product
2. **Missing Value Handling**: Zero-fills missing quantities, forward/backward fills prices
3. **Feature Engineering**: Creates lag features and rolling statistics
4. **Normalization**: Scales features to [0,1] range for LSTM training

## Safety Stock Logic

The system applies conservative ordering recommendations:
- Uses `np.ceil()` to round up predictions
- Ensures non-negative values with `np.maximum(0, ...)`
- Prioritizes avoiding stockouts over minimizing excess inventory


## License

This project is open source and available under the MIT License.

## Author

**Pratyush Jadhav**
- GitHub: [@pratyushjdhv](https://github.com/pratyushjdhv)

## Acknowledgments

- Historical grocery chain sales data used for demonstration purposes
- Built with modern ML and web development best practices
