import pandas as pd
import numpy as np

# Loading the dataset:
data = pd.read_csv("/storage/emulated/0/my python files/Datasets/car_data.csv")
dataset = pd.DataFrame(data)

# Dropping unwanted columns:
columns_to_drop = ["User ID"]
columns_to_drop = [col for col in columns_to_drop if col in dataset.columns]
dataset = dataset.drop(columns=columns_to_drop)

# Cleaning data (removing duplicates):
dataset.drop_duplicates(inplace=True)
print(dataset.info())

# Converting Gender to one-hot encoded columns:
gender_encoded = pd.get_dummies(dataset["Gender"], prefix="Gender")
dataset = pd.concat([dataset, gender_encoded], axis=1)
dataset = dataset.drop(columns=["Gender"])
print(dataset.info())

# Training the data:
x = dataset.loc[:, ["Gender_Female", "Gender_Male", "Age", "AnnualSalary"]]
y = dataset.loc[:, ["Purchased"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=10)

# Fitting the data into logistic regression model:
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(x_train, y_train)

# Testing the data:
y_pred = regressor.predict(x_test)

# Finding the Accuracy:
print("Accuracy:", regressor.score(x_test, y_test))

# Confusion Matrix:
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(matrix)

# Predicting values:
print("Predicted label for [0, 40, 27000]:", regressor.predict([[0, 1, 40, 27000]]))
