from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from fastapi.middleware.cors import CORSMiddleware

# ✅ Initialize app
app = FastAPI()

# ✅ Enable CORS (VERY IMPORTANT for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load model from Hugging Face
MODEL_NAME = "cardiffnlp/twitter-roberta-base-irony"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# ✅ Request format
class InputText(BaseModel):
    text: str
    context: str = ""

# ✅ Root route (optional)
@app.get("/")
def root():
    return {"message": "Sarcasm Detection API is running"}

# ✅ Prediction route
@app.post("/predict")
def predict(data: InputText):
    text = data.text

    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits → probabilities
    probs = F.softmax(outputs.logits, dim=1)

    confidence, predicted_class = torch.max(probs, dim=1)

    confidence = confidence.item()
    predicted_class = predicted_class.item()

    # 🔥 Confidence threshold fix
    if confidence < 0.6:
        sarcasm = False
    else:
        sarcasm = True if predicted_class == 1 else False

    return {
        "sarcasm": sarcasm,
        "confidence": round(confidence, 3)
    }