from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = AutoTokenizer.from_pretrained("model")
model = AutoModelForSequenceClassification.from_pretrained("model")

class InputText(BaseModel):
    text: str
    context: str = ""

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

@app.post("/predict")
def predict(data: InputText):
    sentiment = get_sentiment(data.text)
    
    combined = f"Context: {data.context} Text: {data.text} Emotion: {sentiment}"
    
    inputs = tokenizer(combined, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs).item()
    
    return {
        "sarcasm": bool(pred),
        "confidence": float(probs[0][pred])
    }