import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV ,KFold
from sklearn.pipeline import Pipeline
import joblib

data = pd.read_csv("dataset.csv")
if "target" in data.columns:
    data = data.drop(["target"], axis=1)
if "sex_str" in data.columns:
    data = data.drop(["sex_str"], axis=1)

data["thalach"] = data["thalach"].astype(float)
data["chol"] = data["chol"].astype(float)

mean=data["thalach"].mean()
data.loc[(data['thalach']>180) | (data['thalach']<120), 'thalach']=mean
chol_mean=data['chol'].mean()
data.loc[data['chol']>350,'chol']=chol_mean

x=data.drop(["target_str"],axis=1)
y=data["target_str"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)
model=RandomForestClassifier(random_state=42)
params={
    "model__n_estimators":[50,100,200],
    "model__max_depth":[10,20,30],
    "model__min_samples_split":[2,5,10],
    "model__min_samples_leaf":[1,2,4]
}
cat_col=x.select_dtypes(include=["object"]).columns
num=x.select_dtypes(exclude=["object"]).columns
preprocessor=ColumnTransformer(
    transformers=[
        ("num","passthrough",num),
        ("cat",OneHotEncoder(handle_unknown="ignore"),cat_col)
    ]
)
pipeline=Pipeline(steps=[
    ("processor",preprocessor),
    ("model",model)
])
grid=GridSearchCV(
    estimator=pipeline,
    param_grid=params,
    cv=KFold(n_splits=5,shuffle=True,random_state=42),
    scoring="accuracy",
    n_jobs=-1

)
grid.fit(x_train,y_train)
joblib.dump(grid,"model.joblib")

