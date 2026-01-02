from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
data=pd.read_csv("cancer.csv")
x=data.iloc[:,1:].values
y=data.iloc[:,-1].values
OH=OneHotEncoder()
LE=LabelEncoder()
x_transform=OH.fit_transform(x)
y_transform=LE.fit_transform(y)
print(x_transform[:5,:])
print(y[:5])


