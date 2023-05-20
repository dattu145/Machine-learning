import pandas as pd
import numpy as np

# Loading dataset :
dataset = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")

# Extracting dependent and independent variables
dataset = dataset.loc[:,["lstat","medv"]]

# Visualising dataset in form of scatter plot
x = dataset["lstat"]
y = dataset["medv"]
#plt.scatter(x,y)
#plt.title("Boston Housing")
#plt.xlabel("lstat")
#plt.ylabel("medv")
#plt.show()

# Preparing data

x = pd.DataFrame(x)
y = pd.DataFrame(y)

# splitting data to train and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=14)

# fitting the train data into linear regression to make a model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# testing the model
y_pred = regressor.predict(x_test)


# visualizing the test set results
#plt.scatter(x_test, y_test, color='red')
#plt.plot(x_test, y_pred, color='blue', linewidth=3)
#plt.title('Boston Housing Test set')
#plt.xlabel('lstat')
#plt.ylabel('medv')
#plt.show()

# Testing Using new data

new_data = np.array([[5.0], [10.0], [15.0]])
y_new = regressor.predict(new_data)
print(y_new)

# Finding the accuracy of my model

accuracy = regressor.score(x_test,y_test)
print("Accuracy : ", accuracy)


# calculate MAE and MSE

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)

