# app.py  (minimal, no-RAG)  -- cleaned TTS + improved prompt
import os, uuid, subprocess, re
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
# FIXED: removed the extra "piper" segment
PIPER_EXE  = WORKDIR / "piper/piper" / "piper.exe"
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
    # Prompt guidance: H2 heading, numbered steps; avoid inline emphasis markers
    system: Optional[str] = (
        "You are AidMate, an offline assistant.\n"
        "- Start with a single H2 markdown heading (## Heading) naming the action or section.\n"
        "- Then provide concise, step-by-step numbered instructions (1., 2., 3.).\n"
        "- Do NOT use bold/italics/emphasis markers like **, *, or _ in the body.\n"
        "- Avoid decorative symbols; keep content clean and readable.\n"
        "- End with: Not a medical professional."
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

def clean_markdown_for_tts(text: str) -> str:
    """
    Strip markdown artifacts so TTS doesn't read symbols, but preserve
    a single heading and list structure.
    """
    t = text.replace("\r\n", "\n").strip()

    # Remove code fences/blocks
    t = re.sub(r"```.*?```", "", t, flags=re.S)

    lines_out = []
    for raw in t.split("\n"):
        line = raw.strip()

        if not line:
            continue

        # Keep markdown headings: turn '#', '##'... into plain heading text
        if line.startswith("#"):
            # Remove leading hashes and extra spaces
            heading = re.sub(r"^#+\s*", "", line)
            # Also remove any emphasis marks that might be inside the heading
            heading = re.sub(r"(\*{1,3}|_{1,3})(.+?)\1", r"\2", heading)
            lines_out.append(heading)
            continue

        # Collapse standalone bold heading like **Stay Safe** into plain text
        m = re.fullmatch(r"\*{1,3}\s*(.+?)\s*\*{1,3}", line)
        if m:
            lines_out.append(m.group(1))
            continue

        # Remove leading bullets (-, *) but keep the content
        line = re.sub(r"^\s*[-*]\s+", "", line)

        # Convert "1. Step" to "1. Step" (kept) but ensure consistent space
        line = re.sub(r"^\s*(\d+)\.\s*", r"\1. ", line)

        # Strip emphasis markers (**bold**, *em*, _em_)
        line = re.sub(r"(\*{1,3}|_{1,3})(.+?)\1", r"\2", line)

        # Convert links [text](url) -> text
        line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)

        # Inline code `code` -> code
        line = re.sub(r"`([^`]+)`", r"\1", line)

        if line:
            lines_out.append(line)

    # Join with newlines to keep steps separated for natural pauses
    cleaned = "\n".join(lines_out)

    # Squeeze multiple blank lines (just in case)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    return cleaned

@app.post("/ask")
def ask(body: AskBody):
    sys_prefix = (body.system + "\n\n") if body.system else ""
    reply = call_ollama(sys_prefix + body.prompt, body.model)
    return {"answer": reply}

@app.post("/tts")
def tts(text: str = Form(...)):
    ensure_piper_ok()
    # Clean markdown so the voice doesn't say asterisks or symbols
    safe_text = clean_markdown_for_tts(text)

    fname = f"{uuid.uuid4().hex}.wav"
    out_path = AUDIO_DIR / fname

    p = subprocess.Popen(
        [str(PIPER_EXE), "-m", str(PIPER_MODEL), "-c", str(PIPER_CFG), "-f", str(out_path)],
        stdin=subprocess.PIPE
    )
    p.communicate(input=safe_text.encode("utf-8"))
    p.wait()

    if not out_path.exists():
        return {"error": "Piper failed to synthesize audio."}
    return {"audio_url": f"/static/audio/{fname}"}
