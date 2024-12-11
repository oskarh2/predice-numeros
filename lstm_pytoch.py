import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

# Initialize training data
training_set = np.array([2, 9, 9, 6, 4, 2, 9, 8, 5, 9, 0, 2, 1, 0, 6, 8, 1]).reshape(-1, 1)
seq_length = 3
#plt.plot(training_set, label='Data')
#plt.show()

# Function to create sequences for training
def scaling_window(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        _x = data[i:(i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y)

def prepare_data(training_set):
    sc = MinMaxScaler(feature_range=(0, 1))
    training_data = sc.fit_transform(training_set)    
    x, y = scaling_window(training_data, seq_length)
    train_size = int(len(y) * 0.8)
    test_size = len(y) - train_size
    dataX = Variable(torch.Tensor(x))
    dataY = Variable(torch.Tensor(y))
    return sc, train_size, test_size, dataX, dataY

sc, train_size, test_size, dataX, dataY = prepare_data(training_set)

# Define LSTM model
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        return out

# Training and predicting with LSTM
def lstm_exec(dataX, dataY, future_steps=1):
    num_epochs = 2000
    learning_rate = 0.01
    input_size = 1
    hidden_size = 2
    num_layers = 1
    num_classes = 1
    
    # Initialize model, loss function and optimizer
    lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    
    # Reshape input to match LSTM requirements
    dataX = dataX.view(-1, seq_length, input_size)
    
    # Training loop
    for epoch in range(num_epochs):
        outputs = lstm(dataX)
        optimizer.zero_grad()
        loss = criterion(outputs, dataY)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.5f}")
    
    # Evaluate and plot predictions
    lstm.eval()
    train_predict = lstm(dataX)
    data_predict = train_predict.data.numpy()
    dataY_plot = dataY.data.numpy()
    data_predict = sc.inverse_transform(data_predict)
    dataY_plot = sc.inverse_transform(dataY_plot)
    
    # Predict future values
    last_sequence = dataX[-1].detach().numpy()  # Start with the last known sequence
    future_predictions = []
    
    for _ in range(future_steps):
        with torch.no_grad():
            last_sequence_tensor = torch.Tensor(last_sequence).view(1, seq_length, input_size)
            next_value = lstm(last_sequence_tensor)
            next_value = next_value.data.numpy()
            future_predictions.append(next_value[0][0])            
            # Update last_sequence to contain the new predicted value
            last_sequence = np.append(last_sequence, next_value)[1:]  # Shift left
    
    # Inverse transform predictions to original scale
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = sc.inverse_transform(future_predictions)
    
    # Plot predictions and actual values
    plt.axvline(x=train_size, c='r', linestyle='--', label='Train/Test Split')
    plt.plot(dataY_plot, label="Actual")
    plt.plot(data_predict, label="Predicted")
    
    # Plot future predictions
    plt.plot(range(len(dataY_plot), len(dataY_plot) + future_steps), future_predictions, label="Future Predictions", color="orange")
    plt.legend()
    plt.suptitle('Time-Series Prediction with Future Values')
    plt.show()

lstm_exec(dataX, dataY, future_steps=1)
