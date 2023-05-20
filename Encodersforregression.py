import pandas as pd
import numpy as np

dataset = pd.read_csv("/storage/emulated/0/my python files/Datasets/car_data.csv")
dataset = dataset.loc[:50,["Gender","Age"]]


####		Changing the string to numeric using LabelEncoder  :


from sklearn.preprocessing import LabelEncoder

encode = LabelEncoder()

dataset["Gender"] = encode.fit_transform(dataset["Gender"])

print("Using LabelEncoder : \n\n",dataset)


#### 	 Changing the string to numeric using One-hot encoder of Pandas:


gender_encoded = pd.get_dummies(dataset["Gender"], prefix="Gender")
dataset = pd.concat([dataset, gender_encoded], axis=1)
dataset = dataset.drop(columns=["Gender"])
print("Using One-hot encoder of pandas : \n\n",dataset)


####   	Changing the string to numeric using OneHotEncoder of sklearn :
	

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Example dataset
data = pd.DataFrame({'Color': ['Red', 'Blue', 'Green', 'Red', 'Green'], 'Value': [1, 2, 3, 4, 5]})

# Extract the categorical feature from the dataset
#categories = data[['Color']]

# Create an instance of the OneHotEncoder class
encoder = OneHotEncoder()

# Fit and transform the categorical data using the encoder
encoded_data = encoder.fit_transform(data[["Color"]])

# Convert the encoded data to a DataFrame
encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(['Color']))

# Concatenate the encoded DataFrame with the original dataset
merged_data = pd.concat([data, encoded_df], axis=1)

# Print the merged dataset
print("Using One-hot encoder of sklearn :\n\n",merged_data)




