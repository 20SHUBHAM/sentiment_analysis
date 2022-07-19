from typing import Optional
from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()

vector = load("vectors.joblib")
model = load("model.joblib")

#class get_review(BaseModel):
    


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/prediction")
def get_prediction(text :str):
    text = [text]
    vec = vector.transform(text)
    prediction = model.predict(vec)
    prediction = int(prediction)
    if prediction >0:
        prediction="positive"
    else:
        prediction = "negative"

    return {"sentence":text,"prediction":prediction}
