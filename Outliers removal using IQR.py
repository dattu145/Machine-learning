import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import linear_model

df = {
    'height' : [4.5,4.2,5,5.3,6,5.8,6.2,6.3,1,1.2,10,17,12]
}

df = pd.DataFrame(df)

fig = plt.figure(figsize=(10,4))
plt.boxplot(df)
plt.xticks([1],["height"])
plt.grid(True)
plt.yticks(df.height)
plt.show()

Q1 = df.height.quantile(0.25)
Q3 = df.height.quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR

df[(df.height<lower_limit)|(df.height>upper_limit)]

df = df[(df.height>lower_limit)&(df.height<upper_limit)]

plt.boxplot(df)
plt.show()
