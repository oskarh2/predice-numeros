import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt

# Sample data (replace with your time series data)
data = np.array([2,	9,	9,	6,	4,	2,	9, 8, 5, 9, 0, 2, 1, 0, 6, 8, 1])

# Preprocess data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data.reshape(-1, 1))

# Create the dataset with sequence length (e.g., 3 steps back to predict next step)
def create_dataset(data, sequence_length=3):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

# Prepare the dataset
sequence_length = 3
X, y = create_dataset(data, sequence_length)

# Reshape input to be [samples, time steps, features] for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build the LSTM model using InputLayer
model = Sequential()
model.add(InputLayer(input_shape=(sequence_length, 1)))  # Define input shape here
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=200, verbose=0)

# Function to predict the next N values based on the last observed sequence
def predict_next_values(model, last_sequence, n_steps, scaler):
    predictions = [2,	9,	9,	6,	4,	2,	9,	8]
    current_sequence = last_sequence.reshape((1, sequence_length, 1))
    
    for _ in range(n_steps):
        predicted_value = model.predict(current_sequence)
        predictions.append(predicted_value[0][0])
        
        # Reshape the predicted value to (1, 1, 1) to match the 3D shape of current_sequence
        predicted_value_reshaped = predicted_value.reshape((1, 1, 1))
        
        # Update the sequence with the predicted value
        current_sequence = np.append(current_sequence[:, 1:, :], predicted_value_reshaped, axis=1)
    
    # Transform predictions back to original scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

# Get the last sequence in the data to start predicting future values
last_sequence = data[-sequence_length:]

# Number of future steps to predict
n_future_steps = 5
predicted_future_values = predict_next_values(model, last_sequence, n_future_steps, scaler)

print("Predicted next values:", predicted_future_values)

# Calculate precision indicators (MAPE and R² score)
# For MAPE, we need actual future values to compare, so we can use the predictions for comparison.
# Since we don't have actual future values, we'll calculate MAPE on the predicted vs the last few actual values

# Use the last few actual values for comparison (this is just an example)
actual_values_for_comparison = scaler.inverse_transform(data[-n_future_steps:].reshape(-1, 1)).flatten()

# Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(actual_values_for_comparison, predicted_future_values)

# R-squared (R²) score
r2 = r2_score(actual_values_for_comparison, predicted_future_values)

print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
print(f"R-squared (R²) score: {r2:.4f}")

# Plotting the actual data and predicted future values
actual_data = scaler.inverse_transform(data)

plt.plot(range(len(actual_data)), actual_data, label="Actual Data")
plt.plot(range(len(actual_data), len(actual_data) + n_future_steps), predicted_future_values, label="Predicted Future Data", linestyle="--")
plt.legend()
plt.show()

# Print precision indicators
