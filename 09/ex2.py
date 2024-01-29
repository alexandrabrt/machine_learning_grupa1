import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

symbol = 'AAPL'
start_date = '2010-01-01'
end_date = '2022-01-01'

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

data = pd.read_csv(f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1=0&period2=9999999999&interval=1d&events=history'")

prices = data['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

def create_sequences(data, seq_length):
    sequences = []
    targets = []

    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i+ seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)


sequence_length = 10
X, y = create_sequences(prices_scaled, sequence_length)

split_ratio = 0.8
split_index = int(split_ratio * len(X))

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2)

loss = model.evaluate(X_test, y_test, verbose=2)
print(f"Loss on test set: {loss}")

predictions = model.predict(X_test)

predictions_actual = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual Prices')
plt.plot(predictions_actual, label='Predicted Prices')
plt.title('Predicted vs Actual Stock Pries')
plt.xlabel('Days')
plt.ylabel('Stock price')
plt.legend()
plt.show()
