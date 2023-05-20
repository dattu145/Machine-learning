import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#############           Line plot

####     fig = plt.figure(figsize = (10,2))

"""
x1 = np.arange(0,15,2)
y1= (2*x1) + 3
x2 = np.arange(0,10,3)
y2= (5*x2) + 4

plt.subplot(2,1,1)
plt.plot(x1,y1,linewidth = 2,color = "g",linestyle = "dashed",marker = "o")
plt.title("Demo graph")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend(["dots"],loc = "lower right")
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(x2,y2,linewidth = 2,color = "b",linestyle = "solid",marker = "o")
plt.title("Demo graph2")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend(["dots2"],loc = "upper left")
plt.show()
"""

# Eg : Predict the temperature!!
"""
temp = np.array([29,28,32,33,30,29,31,32,35,31,29,33,34,33,36,32,33,35,37,35])
days = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
plt.xticks(days)
plt.yticks(temp)
plt.xlabel("days")
plt.ylabel("temperature in degrees")
plt.title("Temperature Measure")
plt.plot(days,temp,marker="^",markersize = 12,markerfacecolor = "red",markeredgecolor = "black",markeredgewidth = 2)
plt.grid(True)
plt.legend(["Temp"], loc="upper left")
#Saving figure
plt.savefig("line plot.png")
plt.show()
"""

##########                Histogram

"""
x1 = np.array([12,12,3,56,45,34,28,3,78,90,47,17,95])
plt.hist(x1,bins=50)
plt.show()
"""

##########               Bar Graph

# remember when extracting keys and values from a dict don't forget to make that a list or error will occur in case of 'barh''
# here is an example graph using "barh", you can change that into a bar by changing method from "barh" to "bar'' below.

"""
data = {"apples":30,"mangoes":23,"guava":37,"lemons":45,"grapes":20}
names = list(data.keys())
values = list(data.values())
plt.barh(names, values, color='pink', edgecolor='black', linewidth=1, label='Data',alpha = 0.1)

# Add labels and a title
plt.xlabel('Category')
plt.ylabel('Quantity')
plt.title('Bar Plot')

# Add a legend
plt.legend()

# Show the plot
plt.show()
"""

##########          Scatter plot

"""
data = pd.read_excel("sample.xlsx")
dataset = pd.DataFrame(data)
dataset.rename(columns={"Height (inches)":"height","Weight (lbs)":"weight","Age (years)":"age"},inplace=True)
dataset[["height","weight","age"]] = dataset[["height","weight","age"]].astype("int")

height = dataset["height"]
weight = dataset["weight"]
age = dataset["age"]

plt.scatter(height,weight,label="weight",color="pink",marker = "^",s = 150,alpha = 0.8,edgecolors="black",linewidths=2)
plt.scatter(height,age,label="age",c="purple")
plt.xlabel("height (inches)")
plt.ylabel("Weight and Age")
plt.grid(True)
plt.xticks(height)
y_all = np.concatenate([weight,age])
plt.yticks(y_all)
plt.legend(loc="best")
plt.show()
"""

############             Box plot
"""
import pandas as pd
import matplotlib.pyplot as plt

# Create a Pandas DataFrame with weight and height data
data = {'weight': [150, 180, 165, 155, 140, 175, 185, 160, 170, 145],
        'height': [170, 175, 160, 180, 165, 170, 185, 155, 175, 165]}
df = pd.DataFrame(data)

# Create the box plot
plt.boxplot([df["weight"],df["height"]],showmeans=False,showfliers=True)

# Set the axis labels and title
plt.xticks([1,2],['Weight', 'Height'])
plt.grid(True)
plt.ylabel('Measurement')
plt.title('Box plot of weight and height')
#plt.yticks(data["weight"])
# Show the plot
plt.show()
"""