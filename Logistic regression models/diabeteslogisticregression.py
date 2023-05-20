import pandas as pd
import numpy as np

####		Loading the dataset to perform logistic regression :
	
data = pd.read_csv("/storage/emulated/0/my python files/Datasets/diabetes-dataset.csv")
dataset = pd.DataFrame(data)

# columns : ['Pregnancies', 'Glucose', 'BloodPressure','Insulin', 'BMI', 'Age', 'Outcome']

dataset = dataset.drop(columns=["SkinThickness","DiabetesPedigreeFunction"])
#dataset = dataset.drop_duplicates()

#  Training the data

from sklearn.model_selection import train_test_split

x = pd.DataFrame(dataset.loc[:,["Pregnancies","Glucose","BloodPressure","Insulin","BMI","Age"]])
y = pd.DataFrame(dataset.loc[:,"Outcome"])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.10,random_state=10)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# 		Fitting the data :
	
#print(x_train,"\n",x_test,"\n",y_train,"\n",y_test)

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(x_train,y_train)

# Predicting the model :
	
y_pred = regressor.predict(x_test)

# Comparing predicted values with actual values to check the efficiency of the model using confusion_matrix

from sklearn.metrics import confusion_matrix

topredict = confusion_matrix(y_pred,y_test)
print(topredict)

# Checking Accuracy :
	
accuracy = int(regressor.score(x_test,y_test)*100)
print("Accuracy : ",accuracy,"%")

# Predicting outcome using new data :

print(regressor.predict([[3,150,45,15,10,45]]))

