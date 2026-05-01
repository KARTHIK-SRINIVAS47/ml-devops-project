from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from textblob import TextBlob
from prometheus_client import Counter, generate_latest
import logging

app = FastAPI()
VERSION = "v1"

REQUEST_COUNT = Counter('request_count', 'Total API Requests')

logging.basicConfig(level=logging.INFO)

class TextInput(BaseModel):
    text: str

def analyze(text: str):
    if not text or text.strip() == "":
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    polarity = TextBlob(text).sentiment.polarity
    return "Positive" if polarity > 0 else "Negative"

@app.get("/")
def home():
    return {"message": "ML API is running"}

@app.post("/predict")
def predict_post(input: TextInput):
    REQUEST_COUNT.inc()
    logging.info(f"POST Input: {input.text}")
    return {"sentiment": analyze(input.text)}

@app.get("/predict")
def predict_get(text: str):
    REQUEST_COUNT.inc()
    logging.info(f"GET Input: {text}")
    return {"sentiment": analyze(text)}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.get("/health")
def health():
    return {"status": "OK"}

@app.get("/version")
def version():
    return {"version": VERSION}