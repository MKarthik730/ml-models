import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("capping_flooring_practice_multi.csv")

plt.xlabel("price")
plt.ylabel("quantity_sold")
#plt.scatter(x,y)
#plt.show()
x1 = data["price"].quantile(0.25)
y1=data["price"].quantile(0.75)
iqr1=y1-x1
lowp=x1-1.5*iqr1
highp=y1+1.5*iqr1
x2 = data["quantity_sold"].quantile(0.25)
y2=data["quantity_sold"].quantile(0.75)
iqr2=y2-x2
lowq=x2-1.5*iqr2
highq=y2+1.5*iqr2
print("low=", lowp,"high=",highp)
data["price"]=data["price"].clip(lower=lowp, upper=highp)
data["quantity_sold"]=data["quantity_sold"].clip(lower=lowq,upper=highq)
x=data["price"]
y=data["quantity_sold"]
plt.scatter(x,y)
plt.show()