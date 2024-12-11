# Import necessary libraries
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Generate sample data (replace with your own data)
df = pd.DataFrame({
    'ds': pd.date_range(start='2022-01-01', periods=30),
    'y': [2, 9,9,6,4,2,9,8,7,1,2,5,9,0,2,1,0,6,8,1,1,1,1,8,2,3,5,3,2,3]})
# Create Prophet model
model = Prophet()

# Fit model to data
model.fit(df)

# Make future dataframe for prediction
future = model.make_future_dataframe(periods=30)

# Predict
forecast = model.predict(future)

# Plot forecast
fig = model.plot(forecast)
plt.show()

# Plot components
fig = model.plot_components(forecast)
plt.show()

# Evaluate model precision (last 10 values)
actual = df['y'].tail(10).values
predicted = forecast['yhat'].tail(10).values

mse = mean_squared_error(actual, predicted)
rmse = mse ** 0.5
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Predict next value
next_date = pd.to_datetime('2024-12-01')  # replace with desired date
next_df = pd.DataFrame({'ds': [next_date]})
next_value = model.predict(next_df)
print(f"Predicted next value: {next_value['yhat'].values[0]:.2f}")