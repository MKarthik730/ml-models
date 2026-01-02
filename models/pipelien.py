from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
dataR=pd.read_csv("capping_flooring_practice_multi.csv")
datax=dataR[["price", "rating"]].values
dataY=dataR["quantity_sold"]
trainx,testx,trainy,testy=train_test_split(datax,dataY,test_size=0.3,random_state=42)
steps01=[("scale", StandardScaler()),("linerar R", LinearRegression())]
pipe01=Pipeline(steps01)
pipe01.fit(trainx,trainy)
ypredict=pipe01.predict(testx)
mse=mean_squared_error(testy,ypredict)
mae=mean_absolute_error(testy,ypredict)
print("mse=",mse)
print("mae",mae)
x=float(input("enter price:"))
y=float(input("enter rating: "))
z=pipe01.predict([[x,y]])
print("predicted value: ", z[0])
