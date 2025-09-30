# Stock-Price-Prediction-ARIMA
This project demonstrates **time series forecasting** of stock prices using the **ARIMA model** in Python.   It includes steps for data preprocessing, stationarity testing, model fitting, forecasting, and evaluation.
## 🚀 Features
- Load and visualize stock price dataset
- Decompose time series into trend and seasonality
- Perform **ADF test** for stationarity
- Apply **differencing** if not stationary
- Plot **ACF** and **PACF** for ARIMA order selection
- Train ARIMA model and forecast future stock prices
- Evaluate predictions with **Mean Squared Error (MSE)**
- Generate next-period forecasts

---

## 📂 Project Structure
Stock-Price-Prediction-ARIMA/
│-- stock_data.csv # Sample dataset (Date, Close)
│-- arima_forecast.ipynb # Jupyter Notebook with full code
│-- README.md # Project documentation

yaml
Copy code

---

## 📊 Sample Dataset
The dataset (`stock_data.csv`) contains two columns:
```csv
Date,Close
2024-01-01,150.23
2024-01-02,151.10
2024-01-03,152.35
...
⚙️ Installation & Usage
1️⃣ Clone the Repository
bash
Copy code
git clone https://github.com/your-username/Stock-Price-Prediction-ARIMA.git
cd Stock-Price-Prediction-ARIMA
2️⃣ Install Dependencies
bash
Copy code
pip install pandas numpy matplotlib statsmodels scikit-learn
3️⃣ Run Jupyter Notebook
bash
Copy code
jupyter notebook arima_forecast.ipynb
📌 Example Output
Time series decomposition plots

ACF and PACF plots for ARIMA order selection

Forecast vs Actual comparison

10-day future prediction

📜 License
This project is licensed under the MIT License.
Feel free to use and modify for your own learning and projects.
