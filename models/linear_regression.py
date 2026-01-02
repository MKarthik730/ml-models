import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
data=pd.read_csv("capping_flooring_practice_multi.csv")
x=data[["price", "rating"]].values
y=data["quantity_sold"]
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_train_predict=model.predict(x_train)
y_test_predict=model.predict(x_test)
#mse=mean_squared_error(y_test,y_test_predict)
#mae=mean_absolute_error(y_test,y_train_predict)
#plt.plot(x,y,color="blue",label="original")
#plt.plot(x_test,y_test_predict,color="red",label="predicted")
price=float(input("enter the values "))
rating=float(input("enter the rating "))
predict=model.predict([[price,rating]])
print(predict)


#plt.show()
