"""
presetup.py - One-time script to download DistilBERT model for offline use.
Run this script ONCE with internet access before running the main app.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
LOCAL_DIR = "./models/distilbert-base-uncased-finetuned-sst-2-english"

print(f"Downloading model '{MODEL_NAME}' to '{LOCAL_DIR}' ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

tokenizer.save_pretrained(LOCAL_DIR)
model.save_pretrained(LOCAL_DIR)
print("Download complete. Model saved locally.")