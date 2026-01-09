from pydantic import BaseModel
class UserCreate(BaseModel):
   age: int
   sex: int
   cp: int
   trestbps: int
   chol: float        
   fbs: int
   restecg: int
   thalach: float
   exang: int
   oldpeak: float
   slope: int
   ca: int
   thal: int
   cp_type:str