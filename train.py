from datasets import load_dataset

# Load sarcasm dataset
dataset = load_dataset("tweet_eval", "irony")

print(dataset)
from textblob import TextBlob
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

def preprocess(example):
    text = example["text"]
    
    sentiment = get_sentiment(text)
    
    # No context in dataset → keep empty for now
    combined = f"Text: {text} Emotion: {sentiment}"
    
    return tokenizer(combined, truncation=True, padding="max_length")

dataset = dataset.map(preprocess)
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    num_train_epochs=2,
    logging_dir="./logs",
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"]
)

trainer.train()
model.save_pretrained("model")
tokenizer.save_pretrained("model")