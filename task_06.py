<<<<<<< HEAD
# Imported all required libraries for data handling, model training, evaluation, and plotting
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# Load House Price Prediction Dataset
data = pd.read_csv("House Price Prediction Dataset.csv")  

# Convert CSV data into a DataFrame
df = pd.DataFrame(data)

# Encode all categorical (object) columns into numeric using one-hot encoding
df = pd.get_dummies(df, drop_first=True)


# Separate features (X) and target variable (y)
X = df.drop('Price', axis=1)
y = df['Price']

# Split data into training and testing sets (25% test size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create Linear Regression model
model = LinearRegression()

# Train the model on training data
model.fit(X_train, y_train)

# Predict house prices for test data
predictions = model.predict(X_test)

# Calculate Mean Squared Error (lower = better performance)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Calculate Root Mean Squared Error for better interpretability
rmse = np.sqrt(mse)
print("RMSE:", rmse)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, predictions)
print("MAE:", mae)

# Plot Actual vs Predicted values for visual comparison
plt.plot(y_test.values, label="Actual Values")
plt.plot(predictions, label="Predicted Values")
plt.title("Actual vs Predicted House Prices")
plt.xlabel("Record Index")
plt.ylabel("Price")
plt.legend()
plt.show()
=======
# Imported all required libraries for data handling, model training, evaluation, and plotting
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# Load House Price Prediction Dataset
data = pd.read_csv("House Price Prediction Dataset.csv")  

# Convert CSV data into a DataFrame
df = pd.DataFrame(data)

# Encode all categorical (object) columns into numeric using one-hot encoding
df = pd.get_dummies(df, drop_first=True)


# Separate features (X) and target variable (y)
X = df.drop('Price', axis=1)
y = df['Price']

# Split data into training and testing sets (25% test size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create Linear Regression model
model = LinearRegression()

# Train the model on training data
model.fit(X_train, y_train)

# Predict house prices for test data
predictions = model.predict(X_test)

# Calculate Mean Squared Error (lower = better performance)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Calculate Root Mean Squared Error for better interpretability
rmse = np.sqrt(mse)
print("RMSE:", rmse)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, predictions)
print("MAE:", mae)

# Plot Actual vs Predicted values for visual comparison
plt.plot(y_test.values, label="Actual Values")
plt.plot(predictions, label="Predicted Values")
plt.title("Actual vs Predicted House Prices")
plt.xlabel("Record Index")
plt.ylabel("Price")
plt.legend()
plt.show()
>>>>>>> f45f1a62ece1e1f34b6fa93c3056c562f6c7a797
