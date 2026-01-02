from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer,make_column_selector
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.metrics import mean_absolute_error
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
        ('model', LinearRegression())
    ]
)
 
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=43)
pipe.fit(x,y)
y_pred=pipe.predict(x_test)
print("score=", mean_absolute_error(y_test,y_pred))