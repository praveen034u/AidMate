import threading
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict

# In-memory cache for emergency flags (prompt: bool)
emergency_flag_cache: Dict[str, bool] = {}
#status inprogress, sent, failed

# Path to local DistilBERT model directory (downloaded via presetup.py)
LOCAL_MODEL_DIR = "./models/distilbert-base-uncased-finetuned-sst-2-english"

# Thread-safe lazy initialization of the pipeline
_distilbert_pipe = None
def get_distilbert_pipe():
    global _distilbert_pipe
    if _distilbert_pipe is None:
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR)
        _distilbert_pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return _distilbert_pipe

def detectEmergencyFlag(prompt: str):
    """
    Detect emergency flag using DistilBERT and store result in cache.
    Runs in a background thread, does not block the main thread.
    """
    def worker():
        print("detectEmergencyFlag started:", prompt)
        pipe = get_distilbert_pipe()
        result = pipe(prompt, truncation=True)[0]
        # Adjust logic as needed for your use case
        # Here, we treat 'NEGATIVE' as emergency for demonstration
        is_emergency = (result['label'] == 'NEGATIVE' and result['score'] > 0.8)
        if is_emergency:
            print("Emergency detected for prompt:", prompt)
        emergency_flag_cache[prompt] = false
        print("detectEmergencyFlag finished:", prompt, "->", is_emergency)
    threading.Thread(target=worker, daemon=True).start()