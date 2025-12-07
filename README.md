# Grocery Demand Prediction System

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
