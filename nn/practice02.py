import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score,KFold
data=pd.read_csv("cancer.csv")
def cross_val(model,x,y,cv):
    score=cross_val_score(
        model,x,y,cv=cv
    )
    arr=np.array([score])
    return arr.mean()
cv=KFold(n_splits=5,shuffle=True,random_state=42)
encode=OneHotEncoder()
temp=data
x=temp.drop(columns=['Class'],errors="ignore")
x_str=x.select_dtypes(include=["object","string"]).iloc[:,:-1]
x_str=encode.fit_transform(x_str)
y=data.iloc[:,-1]
L_encode=LabelEncoder()
y=L_encode.fit_transform(y)
x_train,x_test,y_train,y_test=train_test_split(x_str,y,test_size=0.3,random_state=43)
model_1=LogisticRegression()
model_2=RandomForestClassifier()
score_1=cross_val(model_1,x_train,y_train,cv)
score_2=cross_val(model_2,x_train,y_train,cv)
print(score_1)
print(score_2)

