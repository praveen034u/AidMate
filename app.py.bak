# app.py  (voice-powered, minimal no-RAG)
import os, uuid, subprocess, re, shutil, tempfile
from pathlib import Path
from typing import Optional, Dict, List
import requests

from fastapi import FastAPI, Form, UploadFile, File, Header
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

# Piper (TTS)
PIPER_EXE   = WORKDIR / "piper/piper" / "piper.exe"
PIPER_MODEL = WORKDIR / "piper" / "en_US-amy-low.onnx"
PIPER_CFG   = WORKDIR / "piper" / "en_US-amy-low.onnx.json"

# ---- Whisper.cpp (STT) ----
WHISPER_DIR   = WORKDIR / "whisper"
WHISPER_EXE_NEW = WHISPER_DIR / "whisper-whisper.exe"   # new naming in whisper.cpp
WHISPER_EXE_OLD = WHISPER_DIR / "whisper.exe"           # legacy binary (deprecated)
WHISPER_MODEL   = WHISPER_DIR / "ggml-base.en.bin"      # change to ggml-small.en.bin for better accuracy
FFMPEG_BIN      = "ffmpeg"                              # must be on PATH

# Ollama
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
OLLAMA_GEN_URL = f"http://{OLLAMA_HOST}/api/generate"

# -------------------------
# Runtime checks
# -------------------------
def must_exist(p: Path):
    if not p.exists():
        raise RuntimeError(f"Missing required file: {p}")

def ensure_piper_ok():
    for p in (PIPER_EXE, PIPER_MODEL, PIPER_CFG):
        must_exist(p)

def resolve_whisper_exe() -> Path:
    """Prefer whisper-whisper.exe; if missing, use whisper.exe without failing."""
    if WHISPER_EXE_NEW.exists():
        return WHISPER_EXE_NEW
    elif WHISPER_EXE_OLD.exists():
        # Optionally log or print that you're falling back
        print("Warning: using deprecated whisper.exe (will still work)")
        return WHISPER_EXE_OLD
    else:
        raise RuntimeError(
            f"Whisper binary not found. Expected either:\n"
            f" • {WHISPER_EXE_NEW}\n"
            f" • {WHISPER_EXE_OLD}"
        )

def _discover_whisper_exe() -> Path:
  # Prefer new, then old, then try other likely exe names in the folder
  if WHISPER_EXE_NEW.exists(): return WHISPER_EXE_NEW
  if WHISPER_EXE_OLD.exists(): return WHISPER_EXE_OLD
  # Search for any exe that looks like whisper, or a generic main.exe
  cands = list(WHISPER_DIR.glob("*whisper*.exe")) + list(WHISPER_DIR.glob("main.exe"))
  if cands: return cands[0]
  raise RuntimeError(
      f"Whisper binary not found. Expected one of:\n"
      f"  {WHISPER_EXE_NEW}\n  {WHISPER_EXE_OLD}\n  {WHISPER_DIR}\\*whisper*.exe\n  {WHISPER_DIR}\\main.exe"
  )

def ensure_whisper_ok() -> Path:
  exe = _discover_whisper_exe()
  if not WHISPER_MODEL.exists():
    raise RuntimeError(f"Missing Whisper model: {WHISPER_MODEL}")
  if not shutil.which(FFMPEG_BIN):
    raise RuntimeError("ffmpeg not found in PATH")
  return exe

def _guess_ext(content_type: str) -> str:
  ct = (content_type or "").lower()
  if "webm" in ct: return ".webm"
  if "ogg"  in ct: return ".ogg"
  if "wav"  in ct or "x-wav" in ct: return ".wav"
  if "mpeg" in ct or "mp3" in ct: return ".mp3"
  if "mp4"  in ct or "m4a" in ct or "aac" in ct: return ".m4a"
  return ".bin"

def _strip_whisper_warning(s: str) -> str:
  # Remove the deprecation block if present; keep only meaningful text
  if not s: return ""
  lines = []
  for line in s.splitlines():
    if "binary 'whisper.exe' is deprecated" in line.lower(): 
      continue
    if "Please use 'whisper-whisper.exe' instead" in line:
      continue
    if "examples/deprecation-warning" in line:
      continue
    lines.append(line)
  return "\n".join(lines).strip()
# -------------------------
# Conversation memory
# -------------------------
# In-memory store: { session_id: [("user", text), ("assistant", text), ...] }
CONV_MEM: Dict[str, List[tuple]] = {}
MAX_TURNS = 8  # last 8 exchanges (~16 messages)

def push_turn(session_id: str, role: str, text: str):
    if not session_id:
        return
    hist = CONV_MEM.setdefault(session_id, [])
    hist.append((role, text))
    # cap memory
    if len(hist) > MAX_TURNS * 2:
        CONV_MEM[session_id] = hist[-MAX_TURNS*2:]

def render_history(session_id: str) -> str:
    """Turn memory into a short transcript for the model."""
    hist = CONV_MEM.get(session_id, [])
    lines = []
    for role, msg in hist[-MAX_TURNS*2:]:
        prefix = "User" if role == "user" else "Assistant"
        lines.append(f"{prefix}: {msg}")
    return "\n".join(lines)

# -------------------------
# Markdown cleaner for TTS
# -------------------------
def clean_markdown_for_tts(text: str) -> str:
    t = text.replace("\r\n", "\n").strip()
    t = re.sub(r"```.*?```", "", t, flags=re.S)

    lines_out = []
    for raw in t.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            heading = re.sub(r"^#+\s*", "", line)
            heading = re.sub(r"(\*{1,3}|_{1,3})(.+?)\1", r"\2", heading)
            lines_out.append(heading)
            continue
        m = re.fullmatch(r"\*{1,3}\s*(.+?)\s*\*{1,3}", line)
        if m:
            lines_out.append(m.group(1))
            continue
        line = re.sub(r"^\s*[-*]\s+", "", line)
        line = re.sub(r"^\s*(\d+)\.\s*", r"\1. ", line)
        line = re.sub(r"(\*{1,3}|_{1,3})(.+?)\1", r"\2", line)   # bold/italics
        line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)     # links
        line = re.sub(r"`([^`]+)`", r"\1", line)                 # inline code
        if line:
            lines_out.append(line)
    cleaned = "\n".join(lines_out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned

# -------------------------
# Model I/O
# -------------------------
class AskBody(BaseModel):
    prompt: str
    model: str = "gpt-oss:20b"
    #  model: str = "llama3.1:8b"
    system: Optional[str] = (
        "You are AidMate, an offline assistant.\n"
        "- Start with a single H2 markdown heading (## Heading) naming the action/section.\n"
        "- Then provide concise, step-by-step numbered instructions (1., 2., 3.).\n"
        "- Do NOT use bold/italics/emphasis markers like **, *, or _ in the body.\n"
        "- Avoid decorative symbols; keep content clean and readable.\n"
        "- End with: Not a medical professional."
    )

def call_ollama(full_prompt: str, model: str) -> str:
    payload = {"model": model, "prompt": full_prompt, "stream": False}
    r = requests.post(OLLAMA_GEN_URL, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="AidMate Voice API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.post("/ask")
def ask(body: AskBody, x_session_id: Optional[str] = Header(default="")):
    # stitch short conversational context
    history = render_history(x_session_id)
    sys_prefix = (body.system + "\n\n") if body.system else ""
    convo = (history + "\n\n") if history else ""
    reply = call_ollama(sys_prefix + convo + "User: " + body.prompt + "\nAssistant:", body.model)

    # update memory
    push_turn(x_session_id, "user", body.prompt)
    push_turn(x_session_id, "assistant", reply)
    return {"answer": reply, "session": x_session_id or ""}

@app.post("/tts")
def tts(text: str = Form(...)):
    ensure_piper_ok()
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

@app.post("/stt")
def stt(file: UploadFile = File(...), lang: str = Form("en")):
    """
    Accepts audio (webm/ogg/wav/m4a/mp3), converts to 16k mono WAV via ffmpeg,
    runs Whisper.cpp, and returns text.
    Robust to different whisper builds/output behaviors.
    """
    try:
        whisper_exe = ensure_whisper_ok()   # verifies model + ffmpeg, resolves exe path
    except Exception as e:
        return {"error": "setup", "detail": str(e)}

    # read uploaded bytes
    try:
        raw = file.file.read()
    finally:
        try:
            file.file.close()
        except Exception:
            pass

    import tempfile, glob
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        ext = _guess_ext(file.content_type)
        in_path  = td_path / f"input{ext}"
        wav_path = td_path / "audio.wav"

        in_path.write_bytes(raw)

        # 1) ffmpeg -> 16kHz mono wav
        conv = subprocess.run(
            [
                FFMPEG_BIN, "-y",
                "-hide_banner", "-loglevel", "error",
                "-i", str(in_path),
                "-ac", "1", "-ar", "16000",
                str(wav_path),
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=td
        )
        if conv.returncode != 0 or not wav_path.exists():
            return {
                "error": "ffmpeg conversion failed",
                "detail": (conv.stderr or conv.stdout).strip()[:2000],
                "content_type": file.content_type,
                "saved_as": str(in_path),
            }

        def run_and_collect(cmd):
            """Run whisper with given args in temp cwd; return (rc, out, err, txt_paths)."""
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=td
            )
            # collect any .txt produced in cwd
            txt_paths = [Path(p) for p in glob.glob(str(td_path / "*.txt"))]
            return proc.returncode, proc.stdout, proc.stderr, txt_paths

        # Attempt A: classic '-otxt' (default output: audio.wav.txt in cwd)
        cmd_a = [str(whisper_exe), "-m", str(WHISPER_MODEL), "-f", str(wav_path), "-l", lang, "-otxt"]
        rc_a, out_a, err_a, txts_a = run_and_collect(cmd_a)

        def pick_text(txt_paths):
            for p in txt_paths:
                try:
                    txt = p.read_text(encoding="utf-8", errors="ignore").strip()
                    if txt:
                        return txt
                except Exception:
                    pass
            return ""

        text = pick_text(txts_a)

        # Attempt B: explicit output prefix '-of out'
        if not text:
            out_prefix = td_path / "out"
            cmd_b = [str(whisper_exe), "-m", str(WHISPER_MODEL), "-f", str(wav_path), "-l", lang, "-otxt", "-of", str(out_prefix)]
            rc_b, out_b, err_b, txts_b = run_and_collect(cmd_b)
            text = pick_text(txts_b)
        else:
            rc_b = 0; out_b = ""; err_b = ""; txts_b = []

        # If still no text, return helpful diagnostics (trim noisy deprecation lines)
        def clean_diag(s: str) -> str:
            if not s: return ""
            bad = [
                "binary 'whisper.exe' is deprecated",
                "Please use 'whisper-whisper.exe' instead",
                "examples/deprecation-warning"
            ]
            keep = []
            for line in s.splitlines():
                if any(bad_frag.lower() in line.lower() for bad_frag in bad):
                    continue
                keep.append(line)
            return "\n".join(keep).strip()

        if text:
            return {"text": text}

        # Nothing usable produced — show stderr/stdout from both attempts
        diag = (clean_diag(err_a or out_a) + "\n" + clean_diag(err_b or out_b)).strip()
        return {"error": "whisper failed", "detail": diag[:2000]}