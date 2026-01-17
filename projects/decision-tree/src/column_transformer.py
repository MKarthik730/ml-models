import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
 
preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  
        )
    
    
model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])
    

#using make_columnn_selector

from sklearn.compose import make_column_selector
    
preprocessor3 = ColumnTransformer([
        ('num', StandardScaler(), 
         make_column_selector(dtype_include=np.number)),
        ('cat', OneHotEncoder(sparse_output=False), 
         make_column_selector(dtype_include=object))
    ])
 #selecting only relevant features
from sklearn.feature_selection import SelectKBest, f_classif

pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif, k=10)),
        ('classifier', LogisticRegression())
    ])