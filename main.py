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
model=joblib.load('heart_model.joblib')
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
        ca=usr.ca,
        thal=usr.thal
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
    features = [[
        data.age,
        data.sex,
        data.cp,
        data.trestbps,
        data.chol,
        data.fbs,
        data.restecg,
        data.thalach,
        data.exang,
        data.oldpeak,
        data.slope,
        data.ca,
        data.thal,
    ]]

    pred = model.predict(features)[0]         
    proba = None
    try:
        proba = float(model.predict_proba(features)[0][1])
    except Exception:
        pass

    return {
        "prediction": int(pred),
        "probability": proba,
    }


