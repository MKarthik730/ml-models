import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
# Pattern 1: Different scalers for different numeric columns
preprocessor1 = ColumnTransformer([
        ('standard', StandardScaler(), ['age', 'credit_score']),
        ('minmax', MinMaxScaler(), ['income']),
        ('cat', OneHotEncoder(sparse_output=False), ['city', 'education']),
        ('binary', 'passthrough', ['owns_home'])  # leave as-is
    ])
    
    # Pattern 2: Using make_column_transformer 
preprocessor2 = make_column_transformer(
        (StandardScaler(), ['age', 'credit_score']),
        (MinMaxScaler(), ['income']),
        (OneHotEncoder(sparse_output=False), ['city', 'education']),
        remainder='passthrough'  
    )
    
    # Pattern 3: Using column selectors
from sklearn.compose import make_column_selector
    
preprocessor3 = ColumnTransformer([
        ('num', StandardScaler(), 
         make_column_selector(dtype_include=np.number)),
        ('cat', OneHotEncoder(sparse_output=False), 
         make_column_selector(dtype_include=object))
    ])
    
print("Testing different preprocessor patterns...")
for i, preprocessor in enumerate([preprocessor1, preprocessor2, preprocessor3], 1):
        X_transformed = preprocessor.fit_transform(X)
        print(f"Preprocessor {i} output shape: {X_transformed.shape}")
    
    