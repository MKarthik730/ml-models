import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.model_selection import train_test_split
data=pd.read_csv("titanic.csv")
x=data.drop(["Survived","names"],axis=1,errors="ignore")
y=data["Survived"]
model=RandomForestClassifier(n_estimators=20)
X_encoded = pd.get_dummies(x)
x_train,x_test,y_train,y_test=train_test_split(X_encoded,y, test_size=0.3, random_state=42)

model.fit(x_train,y_train)
y_predict=model.predict(x_test)
score=accuracy_score(y_test,y_predict)
print("accuracy score=",score)
print(classification_report(y_test,y_predict))