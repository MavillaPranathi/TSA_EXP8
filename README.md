# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 
### Developed by : M.Pranathi
### Register no : 212222240064

### AIM:
To implement Moving Average Model and Exponential smoothing Using supermarketsales dataset.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df=pd.read_csv('supermarketsales.csv')


# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Display shape and first 20 rows (or the available data if fewer rows)
print("Dataset shape:", df.shape)
print("First rows of dataset:\n", df.head(20))

# Plot the original data (Total)
plt.figure(figsize=(10, 6))
plt.plot(df['Total'], label='Original Data', marker='o')
plt.title('Original Time Series Data (Total)')
plt.ylabel('Total')
plt.xlabel('Date')
plt.legend()
plt.show()

# Moving Average with window size 5 and 10
rolling_mean_5 = df['Total'].rolling(window=5).mean()
rolling_mean_10 = df['Total'].rolling(window=10).mean()

# Plot original data and rolling means (5 and 10)
plt.figure(figsize=(10, 6))
plt.plot(df['Total'], label='Original Data', marker='o')
plt.plot(rolling_mean_5, label='Rolling Mean (Window=5)', marker='x')
plt.plot(rolling_mean_10, label='Rolling Mean (Window=10)', marker='^')
plt.title('Original Data vs Rolling Means')
plt.ylabel('Total')
plt.xlabel('Date')
plt.legend()
plt.show()

# Perform Exponential Smoothing
exp_smoothing = SimpleExpSmoothing(df['Total']).fit(smoothing_level=0.2, optimized=False)
exp_smoothed = exp_smoothing.fittedvalues

# Plot Original Data and Exponential Smoothing
plt.figure(figsize=(10, 6))
plt.plot(df['Total'], label='Original Data', marker='o')
plt.plot(exp_smoothed, label='Exponential Smoothing', marker='s')
plt.title('Original Data vs Exponential Smoothing')
plt.ylabel('Total')
plt.xlabel('Date')
plt.legend()
plt.show()

# Plot ACF and PACF
plt.figure(figsize=(10, 6))
plt.subplot(121)
plot_acf(df['Total'], lags=10, ax=plt.gca())
plt.title('Autocorrelation Function (ACF)')
plt.subplot(122)
plot_pacf(df['Total'], lags=10, ax=plt.gca())
plt.title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

# Generate Predictions using Exponential Smoothing (Predict next 3 values)
prediction_steps = 3
forecast = exp_smoothing.forecast(steps=prediction_steps)

# Plot original data and predictions
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Total'], label='Original Data', marker='o')
plt.plot(pd.date_range(start=df.index[-1], periods=prediction_steps + 1, freq='D')[1:], forecast, label='Predictions', marker='x')
plt.title('Original Data vs Predictions (Exponential Smoothing)')
plt.ylabel('Total')
plt.xlabel('Date')
plt.legend()
plt.show()
```

### OUTPUT:

![image](https://github.com/user-attachments/assets/65e75ec0-683f-4835-a354-8b6f202b082d)

![image](https://github.com/user-attachments/assets/b06061de-e7ec-4d09-a9a0-815265247ddc)

![image](https://github.com/user-attachments/assets/4df0a643-80af-482f-b2eb-842a39227cff)

![image](https://github.com/user-attachments/assets/91a3e011-a4e6-4ebc-8097-1efe2135a193)


![image](https://github.com/user-attachments/assets/dd424e0b-5736-4281-a8a0-91d35d3ae529)



### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using supermarketsales dataset.
