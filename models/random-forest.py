<<<<<<< HEAD:models/random-forest.py
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,KFold,cross_val_score
from sklearn.compose import ColumnTransformer,make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np
data=pd.read_csv("capping_flooring_practice_multi.csv")
igris=ColumnTransformer(
    [
        ('num', StandardScaler(),make_column_selector(dtype_include="number")),
        ('cat', OneHotEncoder(),make_column_selector(dtype_include="object"))
    ],
    remainder='passthrough'
)
pipe=Pipeline(
   steps= [
        ("igris", igris),
        ('model', RandomForestClassifier())
    ]
)
param_grid = {
    'model__n_estimators': [50, 100],
    'model__max_depth': [None, 10, 20]
}

x=data.iloc[:,:-1]
y=data.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=43)
inner=KFold(n_splits=5,shuffle=True,random_state=42)
outer=KFold(n_splits=5,shuffle=True,random_state=42)
grid_s=GridSearchCV(estimator=pipe,param_grid=param_grid,cv=inner,scoring='accuracy')
n_scores=cross_val_score(grid_s,x_train,y_train,cv=outer,scoring="accuracy")
print(f'Nested CV Accuracy: {np.mean(n_scores):.4f} ± {np.std(n_scores):.4f}')
pipe.fit(x,y)
y_pred=pipe.predict(x_test)
=======
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,KFold,cross_val_score
from sklearn.compose import ColumnTransformer,make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np
data=pd.read_csv("capping_flooring_practice_multi.csv")
igris=ColumnTransformer(
    [
        ('num', StandardScaler(),make_column_selector(dtype_include="number")),
        ('cat', OneHotEncoder(),make_column_selector(dtype_include="object"))
    ],
    remainder='passthrough'
)
pipe=Pipeline(
   steps= [
        ("igris", igris),
        ('model', RandomForestClassifier())
    ]
)
param_grid = {
    'model__n_estimators': [50, 100],
    'model__max_depth': [None, 10, 20]
}

x=data.iloc[:,:-1]
y=data.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=43)
inner=KFold(n_splits=5,shuffle=True,random_state=42)
outer=KFold(n_splits=5,shuffle=True,random_state=42)
grid_s=GridSearchCV(estimator=pipe,param_grid=param_grid,cv=inner,scoring='accuracy')
n_scores=cross_val_score(grid_s,x_train,y_train,cv=outer,scoring="accuracy")
print(f'Nested CV Accuracy: {np.mean(n_scores):.4f} ± {np.std(n_scores):.4f}')
pipe.fit(x,y)
y_pred=pipe.predict(x_test)
>>>>>>> 1f3c67b (changed):nn/pipeline.py
print("score=", mean_absolute_error(y_test,y_pred))