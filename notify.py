import threading
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any, Mapping, Optional
from twilio.rest import Client

# In-memory cache for emergency flags (prompt: bool)
emergency_flag_cache: Dict[str, bool] = {}

# Twilio settings
TWILIO_ACCOUNT_SID = "AC4e8754c922d649be54afc1ebb6330c03"
TWILIO_AUTH_TOKEN = "261023ec01ed921f0628aeaf829238fd"
TWILIO_FROM_PHONE = "whatsapp:+14155238886"

# Path to local DistilBERT model directory (downloaded via presetup.py)
LOCAL_MODEL_DIR = "./models/distilbert-base-uncased-finetuned-sst-2-english"

# Thread-safe lazy initialization of the pipeline
_distilbert_pipe = None
_distilbert_lock = threading.Lock()

def get_distilbert_pipe():
    global _distilbert_pipe
    if _distilbert_pipe is None:
        with _distilbert_lock:
            if _distilbert_pipe is None:  # double-checked locking
                tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
                model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR)
                _distilbert_pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return _distilbert_pipe

def send_whatsapp_message(body: str, to_phone: str):
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

def _sanitize_to_whatsapp(contact_no: Optional[str]) -> Optional[str]:
    """
    Clean spaces, ensure it’s prefixed with 'whatsapp:' and has a leading '+'.
    Returns None if invalid.
    """
    if not contact_no or not contact_no.strip():
        return None
    raw = contact_no.strip().replace(" ", "")
    if not raw.startswith("+") and not raw.startswith("whatsapp:"):
        raw = "+" + raw
    if not raw.startswith("whatsapp:"):
        raw = "whatsapp:" + raw
        print(f"ready for whatsup number{raw}")
    return raw

# -------------------- NEW/UPDATED LOCATION HELPERS --------------------

def _try_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None

def _extract_location(loc: Any) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Extract (latitude, longitude, accuracy) from either:
      - Pydantic model with attributes: latitude, longitude, accuracy
      - dict-like with keys: latitude/longitude/accuracy or lat/lon/accuracy
    Returns (lat, lon, acc) or (None, None, None) if unavailable.
    """
    if loc is None:
        return (None, None, None)

    # Attribute access (Pydantic model or any object)
    lat = getattr(loc, "latitude", None)
    lon = getattr(loc, "longitude", None)
    acc = getattr(loc, "accuracy", None)

    # If attributes not there, try dict-style access
    if lat is None and isinstance(loc, Mapping):
        lat = loc.get("latitude", loc.get("lat"))
    if lon is None and isinstance(loc, Mapping):
        lon = loc.get("longitude", loc.get("lon"))
    if acc is None and isinstance(loc, Mapping):
        acc = loc.get("accuracy")

    # Coerce to floats where possible
    lat_f = _try_float(lat)
    lon_f = _try_float(lon)
    acc_f = _try_float(acc)

    return (lat_f, lon_f, acc_f)

def _format_location(loc: Any) -> str:
    """
    Format a location string from object/dict:
      "LAT, LON (±ACCURACY m) | https://maps.google.com/?q=LAT,LON"
    Returns 'N/A' if no usable coordinates.
    """
    lat, lon, acc = _extract_location(loc)
    if lat is None or lon is None:
        return "N/A"

    lat_s = f"{lat:.6f}"
    lon_s = f"{lon:.6f}"
    acc_s = f"±{acc:.1f} m" if acc is not None else "±unknown"
    return f"{lat_s}, {lon_s} ({acc_s}) | https://maps.google.com/?q={lat_s},{lon_s}"

# ---------------------------------------------------------------------

def detectEmergencyFlag(prompt: str, contactNo: str, location: Any):
    """
    Detect emergency flag using DistilBERT and store result in cache.
    Runs in a background thread, does not block the main thread.
    Includes `location` in the WhatsApp message body.
    """
    def worker():
        print("detectEmergencyFlag started:", prompt)
        pipe = get_distilbert_pipe()
        result = pipe(prompt, truncation=True)[0]
        # Demo logic: treat NEGATIVE as emergency
        is_emergency = (result['label'] == 'NEGATIVE' and result['score'] > 0.8)

        # Validate/sanitize phone
        to_phone = _sanitize_to_whatsapp(contactNo)
        if not to_phone:
            print("contactNo is null or empty/invalid")
            return
        print(f"WhatsApp dest: {to_phone}")

        # Build message with location (handles both model & dict)
        loc_text = _format_location(location)
        msg_body = (
            "EMERGENCY DETECTED 🚨\n"
            f"Message: {prompt}\n"
            f"Location: {loc_text}"
        )

        if is_emergency:
            print("Emergency detected for prompt:", prompt)
            emergency_flag_cache[prompt] = False
            try:
                send_whatsapp_message(msg_body, to_phone)

                # Set flag to True after delay (20s)
                def set_flag_true():
                    emergency_flag_cache[prompt] = True
                    print(f"Flag updated to True for prompt '{prompt}'")

                threading.Timer(20, set_flag_true).start()
            except Exception as e:
                print(f"Failed to send WhatsApp message: {e}")
        else:
            print("No emergency detected for prompt:", prompt)

        print("detectEmergencyFlag finished:", prompt, "->", is_emergency)

    threading.Thread(target=worker, daemon=True).start()
