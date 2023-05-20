
# Importing required modules :

import numpy as np
import pandas as pd

# Loading the dataset : 

dataset = pd.read_csv("https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv")
print(dataset.info())

# Cleaning the dataset :

dataset.drop_duplicates(inplace = True)
print(dataset.info())

# Splitting the data into training and testing sets :

x = pd.DataFrame(dataset[["sepal_length","sepal_width","petal_length","petal_width"]])
y = pd.DataFrame(dataset["species"])

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=20,random_state =111)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Building the model :
	
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(x_train,y_train)

# Predicting the model :
	
y_pred = regressor.predict(x_test)

# Finding the Accuracy of my model :
	
accuracy = int(regressor.score(x_test,y_test))*100
print("Accuracy : ",accuracy,"%")

# Predicting new data :
	
print(regressor.predict([[3.5,2.9,5.2,5.3]]))