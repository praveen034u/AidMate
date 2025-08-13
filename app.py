# app.py  (minimal, no-RAG)
import os, uuid, subprocess
from pathlib import Path
from typing import Optional
import requests

from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# -------------------------
# Config
# -------------------------
WORKDIR = Path(r"C:\SourceCode\AidMate")
STATIC_DIR = WORKDIR / "static"
AUDIO_DIR  = STATIC_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Piper paths (from your setup scripts)
PIPER_EXE = WORKDIR / "piper" / "piper.exe"
PIPER_MODEL = WORKDIR / "piper" / "en_US-amy-low.onnx"
PIPER_CFG   = WORKDIR / "piper" / "en_US-amy-low.onnx.json"

# Ollama host:port (honors env var set by your launcher)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
OLLAMA_GEN_URL = f"http://{OLLAMA_HOST}/api/generate"

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="AidMate Minimal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class AskBody(BaseModel):
    prompt: str
    model: str = "llama3.1:8b"  # change if you prefer a smaller model
    system: Optional[str] = (
        "You are AidMate, an offline assistant. "
        "Be concise and step-by-step. End with: Not a medical professional."
    )

def ensure_piper_ok():
    for p in (PIPER_EXE, PIPER_MODEL, PIPER_CFG):
        if not p.exists():
            raise RuntimeError(f"Missing Piper file: {p}")

def call_ollama(prompt: str, model: str) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(OLLAMA_GEN_URL, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()

@app.post("/ask")
def ask(body: AskBody):
    sys_prefix = (body.system + "\n\n") if body.system else ""
    reply = call_ollama(sys_prefix + body.prompt, body.model)
    return {"answer": reply}

@app.post("/tts")
def tts(text: str = Form(...)):
    ensure_piper_ok()
    fname = f"{uuid.uuid4().hex}.wav"
    out_path = AUDIO_DIR / fname

    p = subprocess.Popen(
        [str(PIPER_EXE), "-m", str(PIPER_MODEL), "-c", str(PIPER_CFG), "-f", str(out_path)],
        stdin=subprocess.PIPE
    )
    p.communicate(input=text.encode("utf-8"))
    p.wait()

    if not out_path.exists():
        return {"error": "Piper failed to synthesize audio."}
    return {"audio_url": f"/static/audio/{fname}"}
