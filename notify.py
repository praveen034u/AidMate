import threading
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict

# In-memory cache for emergency flags (prompt: bool)
emergency_flag_cache: Dict[str, bool] = {}

# Twilio settings
TWILIO_ACCOUNT_SID = "AC4e8754c922d649be54afc1ebb6330c03"
TWILIO_AUTH_TOKEN = "cf1852360928c48ec9ce87641ecc7e80"
TWILIO_FROM_PHONE = "whatsapp:+14155238886"
# Set your recipient here or pass as argument
TWILIO_TO_PHONE = "whatsapp:+18048689796"  # e.g., whatsapp:+12345556789

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

def send_whatsapp_message(body: str, to_phone: str = TWILIO_TO_PHONE):
    """
    Send a WhatsApp message using Twilio API.
    """
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=body,
            from_=TWILIO_FROM_PHONE,
            to=to_phone
        )
        print(f"WhatsApp message sent: SID={message.sid}")
    except Exception as e:
        print(f"Failed to send WhatsApp message: {e}")

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
            emergency_flag_cache[prompt] = False
            # Send WhatsApp message
            try:
                send_whatsapp_message(f"EMERGENCY DETECTED: {prompt}")
               
                # Set flag to True after successful send
                # Instead of setting True immediately, schedule after 30 sec
                def set_flag_true():
                    emergency_flag_cache[prompt] = True
                    print(f"Flag updated to True for prompt '{prompt}'")

                threading.Timer(20, set_flag_true).start()
            except Exception as e:
                print(f"Failed to send WhatsApp message: {e}")
        print("detectEmergencyFlag finished:", prompt, "->", is_emergency)
    threading.Thread(target=worker, daemon=True).start()