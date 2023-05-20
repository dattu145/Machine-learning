import pandas as pd
import matplotlib.pyplot as plt

# Loading a dataset :

dataset = pd.read_csv("/storage/emulated/0/my python files/Datasets/Salary_dataset.csv")

# Extracting required columns :

dataset = dataset.loc[:,["YearsExperience","Salary"]]
# cleaning data for better accuracy of my model :

dataset["Salary"] = dataset["Salary"].astype("int")
print(dataset.info())

# Data visualization :
	
x= dataset["YearsExperience"]
y = dataset["Salary"]
plt.scatter(x,y)
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Demo Test")
plt.show()

# Data Preparation :
	
x = pd.DataFrame(x)
y = pd.DataFrame(y)

# Train and Test dataset
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state = 14)

# Fitting the data into linear regression model

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# Testing the data
y_pred = regressor.predict(x_test)

# Visualizing test tested data

plt.scatter(x_test,y_test, c="r")
plt.plot(x_test,y_pred,c="purple",linewidth=2)
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Demo Test")
plt.show()

# Checking accuracy :
	
accuracy = regressor.score(x_test,y_test)
print("Accuracy : ",int(accuracy*100),"%")

# Predicting using new data :

print(dataset)
print(regressor.predict([[14]]))
