from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import joblib
data=pd.read_csv("dataset.csv")
data["thalach"] = data["thalach"].astype(float)
data["chol"] = data["chol"].astype(float)
o_data=data.drop(['target'],axis=1)

mean=o_data["thalach"].mean()
o_data.loc[(o_data['thalach']>180) | (o_data['thalach']<120), 'thalach']=mean
chol_mean=o_data['chol'].mean()
o_data.loc[o_data['chol']>350,'chol']=chol_mean
x=o_data
y=data["target"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
model=RandomForestClassifier()
model.fit(x_train,y_train)
joblib.dump(model,'heart_model.joblib')