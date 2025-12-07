# Grocery Demand Prediction System

<<<<<<< HEAD
## What We Built

A web-based demand forecasting dashboard for grocery chains that predicts 7-day product demand using LSTM neural networks. The system provides:
- Interactive product selection and real-time forecast generation
- Visual charts comparing raw predictions with safety stock recommendations
- API endpoints for integration with inventory management systems
- Safety stock calculations using conservative ceiling functions to avoid stockouts

## Tech Stack

**Backend:**
- Python 3.x
- Flask (Web framework & API)
- TensorFlow/Keras (LSTM model)
- Pandas & NumPy (Data processing)
- Scikit-learn (Data normalization)

**Frontend:**
- Vue.js 3 (Reactive UI)
- Chart.js (Data visualization)
- HTML5/CSS3

**AI help**
- Gemini and Perplexity were of great help in order of guidance whenever i felt lost and great resourse for further action plan analysis.

## Assumptions Made

### Feature Selection
Only five features are used for prediction:
- `quantity` (current daily sales)
- `last_year_sales` (sales from 364 days ago)
- `recent_trend` (14-day rolling average)
- `last_year_next_7d_avg` (historical 7-day forward average from last year)
- `unit_price` (average price per unit)

**Assumption:** These five features capture all necessary predictive signals. Other columns (customer_id, store_name, aisle, discount_amount, etc.) are assumed to be non-predictive noise or too sparse to offer reliable benefit for general product forecasting.

### Date Handling & Missing Data
The preprocessing fills in missing dates and sets missing quantities to zero.

**Assumption:** If a product has no transaction recorded on a specific day, its true sales quantity for that day was zero (not missing data, but actual zero sales).

### Price Handling
Missing prices are forward-filled and backward-filled within each product.

**Assumption:** Prices remain relatively stable over short periods, and filling gaps with nearby prices provides reasonable estimates.

### Model Selection
LSTM was chosen as the primary model architecture.

**Assumption:** LSTM's ability to maintain "memory" of past events makes it superior for this time-series problem compared to simpler models that treat each day independently.

## How Our Model Works (Plain English)

### Why LSTM?
We initially tried traditional models like Linear Regression and Random Forest, but they failed to produce accurate results. The reason? These models treat each day as an independent data point and cannot learn time-based patterns like seasonality, day-of-week effects, or trends.

**LSTM (Long Short-Term Memory)** is a type of neural network specifically designed for time-series data. It maintains a "memory" of past events, allowing it to recognize patterns that unfold over time—exactly what we need for sales forecasting.

### The Prediction Process

1. **Look at the Past 14 Days**: The model examines the last 2 weeks of data for a product, including sales quantities, last year's sales, recent trends, and prices.

2. **Learn Patterns**: During training, the LSTM learns relationships like:
   - "If sales were high last year at this time, they'll likely be high again"
   - "If there's an upward trend over the past 2 weeks, it might continue"
   - "Certain days of the week tend to have higher/lower sales"

3. **Predict the Next 7 Days**: Using its learned patterns and the most recent 14 days of data, the model forecasts sales for each of the next 7 days.

4. **Apply Safety Logic**: The raw predictions (like 45.32 units) are rounded up to whole numbers (46 units) to ensure you never under-order. This creates a safety buffer against stockouts.

### Training Process
- The model is normalized (scaled to 0-1 range) to help it learn faster
- It trains for 30-50 epochs (learning cycles) to optimize its predictions
- Uses the last 14 days to predict the next 7 days
- Minimizes prediction errors using Mean Squared Error (MSE)

### Output
For each product, you get:
- **Raw Model Output**: The exact prediction (e.g., 45.32 units)
- **Safety Stock Recommendation**: The rounded-up order quantity (e.g., 46 units)
- **7-Day Total**: Sum of all safety stock recommendations for the week

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web app
python app.py

# Access dashboard at http://localhost:5000
```

## Author

**Pratyush Jadhav** - [@pratyushjdhv](https://github.com/pratyushjdhv)
=======
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
>>>>>>> 9e7b600a80930a8dcca2a51a73669614f7f9a452
