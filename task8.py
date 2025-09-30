# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# 2. Load Data (replace 'stock_data.csv' with your filename)
df = pd.read_csv('stock_data.csv', parse_dates=['Date'], index_col='Date')
ts = df['Close']

# 3. Plot original series
ts.plot(figsize=(10, 4), title='Stock Price')
plt.show()

# 4. Decompose trend & seasonality
decomp = seasonal_decompose(ts, model="additive")
decomp.plot()
plt.show()

# 5. Check stationarity (ADF Test)
result = adfuller(ts.dropna())
print("ADF Statistic:", result[0])
print("p-value:", result[1])

# 6. Differencing if not stationary
if result[1] > 0.05:
    ts_diff = ts.diff().dropna()
else:
    ts_diff = ts

# 7. Plot ACF and PACF
fig, ax = plt.subplots(2, 1, figsize=(10,8))
plot_acf(ts_diff, ax=ax[0])
plot_pacf(ts_diff, ax=ax[1])
plt.show()

# 8. Train/Test Split
train_size = int(len(ts) * 0.8)
train, test = ts[:train_size], ts[train_size:]

# 9. Fit ARIMA (change order based on ACF/PACF; here is (1,1,1) as example)
model = ARIMA(train, order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())

# 10. Forecast
forecast = model_fit.forecast(steps=len(test))
plt.figure(figsize=(10,4))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, forecast, label='Forecast')
plt.title('ARIMA Forecast vs Actual')
plt.legend()
plt.show()

# 11. Evaluate
mse = mean_squared_error(test, forecast)
print(f"Test MSE: {mse:.3f}")

# 12. Final model & next period forecast
full_model = ARIMA(ts, order=(1,1,1)).fit()
future = full_model.forecast(steps=10)
print("Next 10 periods forecast:", future)
