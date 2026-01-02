import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
data=pd.read_csv("cancer.csv")
encode=OneHotEncoder()
temp=data
x=temp.drop(columns=['Class'],errors="ignore")
x_str=x.select_dtypes(include=["object","string"]).iloc[:,:-1]
x_str=encode.fit_transform(x_str)
y=data.iloc[:,-1]
L_encode=LabelEncoder()
y=L_encode.fit_transform(y)
x_train,x_test,y_train,y_test=train_test_split(x_str,y,test_size=0.3,random_state=43)
model=LogisticRegression()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
score=accuracy_score(y_test,y_predict)
print(score)
