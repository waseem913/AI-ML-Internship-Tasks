# Imported all needed libraries
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Select the company (Apple)
apple = yf.Ticker("AAPL")

# Fetch 1-year of Apple data
Data = apple.history(period="1y")

# Convert to DataFrame
df = pd.DataFrame(Data)

# Select features
X = df[['Open', 'High', 'Low', 'Volume']]

# Target variable
y = df[['Close']]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict values
predictions = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, predictions)
print("Predictions:\n", predictions)
print("\nMean Squared Error:", mse)

# Reset indexes for plotting
y_test = y_test.reset_index(drop=True)

# Plotting Actual vs Predicted
plt.plot(y_test.values, label="Actual Values", color='blue')
plt.plot(predictions, label="Predicted Values", color='red')
plt.title("Actual vs Predicted Closing Prices (AAPL)")
plt.xlabel("Days")
plt.ylabel("Close Price")
plt.legend()
plt.show()
