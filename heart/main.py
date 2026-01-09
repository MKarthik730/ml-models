from fastapi import FastAPI,Depends
from fastapi import HTTPException,status
from heart.data import UserCreate
from database import SessionLocal,engine
import heart.databasemodels
from heart.databasemodels import Users
from sqlalchemy.orm import Session
app=FastAPI()
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from text import OPEN_API_KEY 
import pandas as pd 

llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0.6,
    api_key=OPEN_API_KEY,
)
prompt = PromptTemplate(
    input_variables=["input"],
    template=(
        "You are an experienced cardiologist assistant at a hospital.\n"
        "ONLY answer questions related to:\n"
        "- heart disease\n"
        "- risk factors (blood pressure, cholesterol, diabetes, obesity, smoking, etc.)\n"
        "- heart-related symptoms (chest pain, shortness of breath, palpitations, etc.)\n"
        "- tests (ECG, Echo, TMT, angiography, blood tests)\n"
        "- treatments and lifestyle advice for heart health.\n\n"
        "If the question is not about heart/heart health, say briefly:\n"
        "\"I can only help with heart-related questions. Please ask something about heart health.\"\n\n"
        "Give a clear explanation in 2–4 short paragraphs, using simple language.\n\n"
        "Question: {input}"
    ),
)



chain = prompt | llm

model=joblib.load('model.joblib')
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

heart.databasemodels.Base.metadata.create_all(bind=engine)

def get_db():
    db_instance = SessionLocal()
    try:
        yield db_instance        
    finally:
        db_instance.close()  
@app.get("/")
def login():
    return "hello"

@app.get("/users")
def get_all_users(db: Session = Depends(get_db)):
    return db.query(Users).all()


@app.post("/users")
def add_user(usr: UserCreate, db: Session = Depends(get_db)):
    new_user = heart.databasemodels.Users(
        age=usr.age,
        sex=usr.sex,
        cp=usr.cp,
        trestbps=usr.trestbps,
        chol=usr.chol,
        fbs=usr.fbs,
        restecg=usr.restecg,
        thalach=usr.thalach,
        exang=usr.exang,
        oldpeak=usr.oldpeak,
        slope=usr.slope,
        ca=usr.ca,
        thal=usr.thal,
        cp_type=usr.cp_type
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user
@app.delete('/users/{id}',status_code=status.HTTP_204_NO_CONTENT)
def delete_user(id:int, db: Session=Depends(get_db)):
    d_user=db.query(Users).filter(Users.id==id).first()
    if d_user:
        db.delete(d_user)
        db.commit()
        return
    else:
        raise HTTPException(status_code=404, detail="not found")
   
@app.get("/users/search")
def search_user(name:str, db:Session=Depends(get_db)):
    usr_v=db.query(Users).filter(Users.name==name).first()
    if not usr_v:
        raise HTTPException(status_code=404,detail="user not found")
    else:
        return usr_v
    
    return None


@app.post("/predict")
def predict_heart_disease(data: UserCreate):
    row = {
        "age": data.age,
        "sex": data.sex,
        "cp": data.cp,
        "trestbps": data.trestbps,
        "chol": data.chol,
        "fbs": data.fbs,
        "restecg": data.restecg,
        "thalach": data.thalach,
        "exang": data.exang,
        "oldpeak": data.oldpeak,
        "slope": data.slope,
        "ca": data.ca,
        "thal": data.thal,
        "cp_type": data.cp_type,
    }
    features_df = pd.DataFrame([row])

    pred = model.predict(features_df)[0]          
    proba = None
    try:
       
        proba = float(model.predict_proba(features_df)[0][
            list(model.classes_).index("disease")
        ])
    except Exception:
        pass

   
    pred_flag = 1 if pred == "disease" else 0

    return {
        "prediction": pred_flag,    
        "label": pred,              
        "probability": proba,
    }
class GPTRequest(BaseModel):
    input: str

@app.post("/gpt")
def gpt_endpoint(req: GPTRequest):
    try:
        answer = chain.invoke({"input": req.input})
        return {"answer": str(answer)}
    except Exception as e:
        return {"error": str(e)}
