# Imported all needed libraries
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Select the company (Apple)
apple = yf.Ticker("AAPL")

#Fetches 1-year of Apple data
Data = apple.history(period="1y")
#Converts data into pandas DataFrame
df= pd.DataFrame(Data)

#Uses Open, High, Low, Volume
X=df[['Open', 'High','Low','Volume'  ]]
## Target variable (to predict)
y=df[['Close']]
X_train,X_test,y_train,y_test=train_test_split(X,y)

# LinearRegression
model = LinearRegression()

# Fits model to training data
model.fit(X_train, y_train)

#Predicts closing price
predictions = model.predict(X_test)

# Measures model performance
mse = mean_squared_error(y_test, predictions)
print("Predictions:", predictions)
print("Mean Squared Error:", mse)

#Plots Actual vs Predicted
plt.plot(y_test.values,label="Actual Values",color='blue')
plt.plot(predictions,label="predicted Values",color='red')
plt.title("Actual vs Predicted Closing Prices (AAPL)")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()




