# app.py  (voice-powered, optimized for CPU, no-RAG by default unless index exists)

import os, uuid, subprocess, re, shutil, tempfile
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import base64
import mimetypes
import requests
from notify import detectEmergencyFlag, emergency_flag_cache

from fastapi import FastAPI, Form, UploadFile, File, Header, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
# ---- LangChain / FAISS
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# Import upload routers
from upload import router as upload_router

# -------------------------
# Config
# -------------------------
UPLOAD_DIR = "uploads"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".csv"}  # Keep tight for speed; add doc/docx later if you wire Unstructured
WORKDIR = Path(r"C:\SourceCode\AidMate")
STATIC_DIR = WORKDIR / "static"
AUDIO_DIR  = STATIC_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Piper (TTS)
PIPER_EXE   = WORKDIR / "piper/piper" / "piper.exe"
PIPER_MODEL = WORKDIR / "piper" / "en_US-amy-low.onnx"
PIPER_CFG   = WORKDIR / "piper" / "en_US-amy-low.onnx.json"

# Whisper.cpp (STT)
WHISPER_DIR     = WORKDIR / "whisper"
WHISPER_EXE_NEW = WHISPER_DIR / "whisper-cli.exe"   # new naming in whisper.cpp
WHISPER_EXE_OLD = WHISPER_DIR / "main.exe"          # legacy binary (deprecated)
WHISPER_MODEL   = WHISPER_DIR / "ggml-base.en.bin"  # small and fast on CPU
FFMPEG_BIN      = "ffmpeg"                          # must be on PATH

# Ollama / Model choices (lightweight CPU)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
OLLAMA_GEN_URL = f"http://{OLLAMA_HOST}/api/generate"
# Lightweight embedding model
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "all-minilm:latest")
# Lightweight chat model we’ll alias to when code asks for "gpt-oss"
OLLAMA_CHAT_MODEL_DEFAULT = os.environ.get("OLLAMA_CHAT_MODEL", "phi3:mini")

# Vector DB
VECTOR_DB_ROOT = Path("./vectordb")
VECTOR_DB_ROOT.mkdir(parents=True, exist_ok=True)
INDEX_PATH = VECTOR_DB_ROOT  # single persistent location

# ------- No Answer indicators when VectorDb lacks relevance ------------
no_answer_indicators = [
    "not in the database", "no information", "cannot find", "i don't know",
    "no relevant information", "not mentioned", "not available in the context",
    "i may be able to assist you better", "not a medical professional"
]

# -------------------------
# Runtime checks
# -------------------------
def must_exist(p: Path):
    if not p.exists():
        raise RuntimeError(f"Missing required file: {p}")

def ensure_piper_ok():
    for p in (PIPER_EXE, PIPER_MODEL, PIPER_CFG):
        must_exist(p)

def _discover_whisper_exe() -> Path:
    if WHISPER_EXE_NEW.exists(): return WHISPER_EXE_NEW
    if WHISPER_EXE_OLD.exists(): return WHISPER_EXE_OLD
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

# -------------------------
# Conversation memory
# -------------------------
CONV_MEM: Dict[str, List[Tuple[str, str]]] = {}
MAX_TURNS = 8  # last 8 exchanges (~16 messages)

def push_turn(session_id: str, role: str, text: str):
    if not session_id:
        return
    hist = CONV_MEM.setdefault(session_id, [])
    hist.append((role, text))
    if len(hist) > MAX_TURNS * 2:
        CONV_MEM[session_id] = hist[-MAX_TURNS*2:]

def render_history(session_id: str) -> str:
    hist = CONV_MEM.get(session_id, [])
    lines = []
    for role, msg in hist[-MAX_TURNS*2:]:
        prefix = "User" if role == "user" else "Assistant"
        lines.append(f"{prefix}: {msg}")
    return "\n".join(lines)

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
# Model aliasing (keep external name "gpt-oss")
# -------------------------
MODEL_ALIASES = {
    "gpt-oss":        OLLAMA_CHAT_MODEL_DEFAULT,
    "gpt-oss:20b":    OLLAMA_CHAT_MODEL_DEFAULT,
    "gpt-oss:latest": OLLAMA_CHAT_MODEL_DEFAULT,
}

def resolve_chat_model(requested: str) -> str:
    # If caller passes an Ollama name directly, use it; else map "gpt-oss*" -> lightweight alias
    return MODEL_ALIASES.get((requested or "").lower(), requested or OLLAMA_CHAT_MODEL_DEFAULT)

# -------------------------
# Global singletons (avoid reloading each request)
# -------------------------
_EMBEDDINGS: Optional[OllamaEmbeddings] = None
_VECTORSTORE: Optional[FAISS] = None

def get_embeddings() -> OllamaEmbeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        # Keep dimensions tiny and CPU friendly
        _EMBEDDINGS = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
    return _EMBEDDINGS

def load_vectorstore_if_exists(crisis: str) -> Optional[FAISS]:
    global _VECTORSTORE
    if _VECTORSTORE is not None:
        return _VECTORSTORE

    # Build path as Path object
    vectordbPath = Path(INDEX_PATH) / crisis  

    if vectordbPath.exists() and any(vectordbPath.glob("*")):
        try:
            emb = get_embeddings()
            _VECTORSTORE = FAISS.load_local(
                str(vectordbPath),
                emb,
                allow_dangerous_deserialization=True
            )
            return _VECTORSTORE
        except Exception as e:
            print(f"[RAG] Failed to load vectorstore: {e}")
            return None
    return None

def reset_vectorstore_cache(crisis: str):
    global _VECTORSTORE
    _VECTORSTORE = None
    load_vectorstore_if_exists(crisis)

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="AidMate Voice API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class Location(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0, description="WGS84 latitude in degrees")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="WGS84 longitude in degrees")
    accuracy: Optional[float] = Field(
        None, ge=0.0, description="Horizontal accuracy radius in meters"
    )

    # (Optional) accept strings like "41.8781" and coerce to float
    @field_validator("latitude", "longitude", "accuracy", mode="before")
    @classmethod
    def _coerce_str_to_float(cls, v):
        if v is None:
            return v
        if isinstance(v, (int, float)):
            return float(v)
        # allow numeric strings
        return float(str(v).strip())
    
class AskBody(BaseModel):
    prompt: str
    crisis: Optional[str] = None
    model: str = "gpt-oss"
    system: Optional[str] = (
        "You are AidMate, an offline assistant.\n"
        "- Start with a single H2 markdown heading (## Heading) naming the action/section.\n"
        "- Then provide concise, step-by-step numbered instructions (1., 2., 3.).\n"
        "- Do NOT use bold/italics/emphasis markers like **, *, or _ in the body.\n"
        "- Avoid decorative symbols; keep content clean and readable.\n"
        "- If you are unsure or the information is not available, say \"I don't know\"."
    )
    # Updated: strongly-typed location object
    location: Optional[Location] = None
    contactNo: Optional[str] = None

@app.get("/")
async def root():
    return {"emergency_flag_cache": emergency_flag_cache}

@app.get("/health")
def health():
    return {
        "ok": True,
        "embed_model": OLLAMA_EMBED_MODEL,
        "chat_model_alias": OLLAMA_CHAT_MODEL_DEFAULT,
        "index_exists": any(INDEX_PATH.glob("*"))
    }

def _loc_to_meta(location) -> str:
    """
    Build a neutral metadata line from a Location model (v2) or dict fallback.
    """
    if not location:
        return ""
    # Try attributes (Location model), fall back to dict-like access
    lat = getattr(location, "latitude", None)
    lon = getattr(location, "longitude", None)
    acc = getattr(location, "accuracy", None)

    if lat is None or lon is None:
        # dict-style fallback if a plain dict was sent
        try:
            lat = location.get("latitude", location.get("lat"))
            lon = location.get("longitude", location.get("lon"))
            acc = location.get("accuracy")
        except Exception:
            return ""

    if lat is None or lon is None:
        return ""

    meta = "[User Location]\n"
    meta += f"Latitude: {lat}, Longitude: {lon}"
    if acc is not None:
        meta += f", Accuracy≈{acc}m"
    return meta + "\n"


@app.post("/ask")
def ask(body: AskBody, x_session_id: Optional[str] = Header(default="")):
    print("Ask:", body.prompt)
    print("Session:", x_session_id or "(new)")
    print("Crisis:", body.crisis)

    # Robust location logging for both Pydantic v2 and v1 (or dict fallback)
    if body.location:
        try:
            loc_log = body.location.model_dump()   # Pydantic v2
        except AttributeError:
            try:
                loc_log = body.location.dict()     # Pydantic v1
            except AttributeError:
                loc_log = body.location            # last resort (already a dict/other)
        print("Location:", loc_log)

    if body.contactNo:
        print("ContactNo:", body.contactNo)

    # Background flag detection (pass the Location model; your worker can accept it)
    # If your detectEmergencyFlag expects a dict, change the next line to:
    # detectEmergencyFlag(body.prompt, body.contactNo, getattr(body.location, "model_dump", lambda: body.location)())
    detectEmergencyFlag(body.prompt, body.contactNo, body.location)

    # Assemble conversation
    history = render_history(x_session_id)
    sys_prefix = (body.system + "\n\n") if body.system else ""
    convo = (history + "\n\n") if history else ""
    requested_model = resolve_chat_model(body.model)

    # Neutral metadata header (uses the Location model)
    meta = ""
    meta += _loc_to_meta(body.location)
    if body.contactNo:
        meta += "[Contact]\n" + str(body.contactNo).strip() + "\n"

    # Prefer RAG if an index exists for the selected crisis
    vector = load_vectorstore_if_exists(body.crisis)
    if vector is not None:
        reply = call_rag(body.prompt, requested_model, vector)
    else:
        reply = call_ollama(sys_prefix + meta + convo + "User: " + body.prompt + "\nAssistant:", requested_model)

    if any(indicator in (reply or "").lower() for indicator in no_answer_indicators):
        reply = call_ollama(sys_prefix + meta + convo + "User: " + body.prompt + "\nAssistant:", requested_model)

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
    try:
        whisper_exe = ensure_whisper_ok()
    except Exception as e:
        return {"error": "setup", "detail": str(e)}

    try:
        raw = file.file.read()
    finally:
        try:
            file.file.close()
        except Exception:
            pass

    if not raw:
        return {"error": "no-audio", "detail": "Uploaded file is empty (0 bytes)."}

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        ext = _guess_ext(file.content_type)
        in_path  = td_path / f"input{ext}"
        wav_path = td_path / "audio.wav"
        in_path.write_bytes(raw)

        try:
            conv = subprocess.run(
                [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error", "-i", str(in_path), "-ac", "1", "-ar", "16000", str(wav_path)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=td, timeout=90
            )
        except subprocess.TimeoutExpired:
            return {"error": "ffmpeg timeout", "detail": "ffmpeg exceeded 90s limit."}

        if conv.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size == 0:
            return {"error": "ffmpeg conversion failed", "detail": (conv.stderr or conv.stdout or "ffmpeg returned non-zero without output.")[:2000]}

        out_txt = td_path / "out.txt"
        cmd = [str(whisper_exe), "-m", str(WHISPER_MODEL), "-f", str(wav_path), "-l", lang, "-otxt", "-of", "out"]

        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=td, timeout=300)
        except subprocess.TimeoutExpired:
            return {"error": "whisper timeout", "detail": "whisper exceeded 300s limit."}

        text = ""
        if out_txt.exists():
            try:
                text = out_txt.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                text = ""

        if not text:
            text = (proc.stdout or "").strip()

        if text:
            return {"text": text}

        diag = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip()
        return {"error": "whisper failed", "detail": diag[:2000]}

# -------------------------
# Upload endpoints (file & JSON) + indexing
# -------------------------
os.makedirs(UPLOAD_DIR, exist_ok=True)

def validate_file(file: UploadFile) -> bool:
    file_extension = Path(file.filename).suffix.lower()
    return file_extension in ALLOWED_EXTENSIONS

def generate_unique_filename(original_filename: str) -> str:
    file_extension = Path(original_filename).suffix
    unique_id = str(uuid.uuid4())
    return f"{unique_id}{file_extension}"

def list_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def load_documents_from_files() -> List[Document]:
    files = list_files(UPLOAD_DIR)
    print(f"[Index] Found {len(files)} files.")
    files = [Path(UPLOAD_DIR) / f for f in files]
    docs: List[Document] = []
    for file in files:
        ext = file.suffix.lower()
        try:
            if ext == ".txt":
                loader = TextLoader(str(file), encoding="utf-8")
            elif ext == ".csv":
                loader = CSVLoader(str(file))
            elif ext == ".pdf":
                loader = PyPDFLoader(str(file))
            else:
                print(f"[Index] Skipping unsupported file type: {file}")
                continue
            docs.extend(loader.load())
        except Exception as e:
            print(f"[Index] Failed to load {file}: {e}")
    return docs

def create_vector_db_from_files() -> bool:
    """Create FAISS vector DB once per upload batch (CPU-friendly)."""
    embeddings = get_embeddings()
    docs = load_documents_from_files()

    if not docs:
        raise ValueError("No documents loaded. Make sure files are valid.")

    print(f"[Index] Loaded {len(docs)} docs. Creating embeddings with {OLLAMA_EMBED_MODEL} ...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    os.makedirs(INDEX_PATH, exist_ok=True)
    vectorstore.save_local(str(INDEX_PATH))
    print(f"[Index] Vector DB saved at {INDEX_PATH}")
    reset_vectorstore_cache()
    return True


# -------------------------
# RAG + LLM calls (CPU-lean)
# -------------------------
def make_cpu_llm(model_name: str) -> OllamaLLM:
    # Conservative settings for CPU boxes; tune as needed
    return OllamaLLM(
        model=model_name,
        temperature=0.2,
        num_ctx=2048,
        # use available threads but avoid oversubscription
        num_thread=min(os.cpu_count() or 4, 6),
        top_p=0.9
    )

def setup_rag_chain(model_name: str, vector_store: FAISS):
    llm = make_cpu_llm(model_name)
    # MMR for diversity; small k for speed
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 8})

    template = (
        "You are AidMate, an offline assistant that uses the provided context to answer the question.\n"
        "- Use only the information provided in the context to answer the question.\n"
        "- If the answer is not in the context, say \"I don't know\" and end with: Not a medical professional.\n"
        "- Start with a single H2 markdown heading (## Heading) naming the action/section.\n"
        "- Then provide concise, step-by-step numbered instructions (1., 2., 3.).\n"
        "- Do NOT use bold/italics/emphasis markers like **, *, or _ in the body.\n"
        "- Avoid decorative symbols; keep content clean and readable.\n\n"
        "Question: {question}\n"
        "Context:\n{context}\n"
    )
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def call_rag(user_prompt: str, model_name: str, vector_store: FAISS) -> str:
    print(f"Rag is called with model name {model_name}")
    rag_chain = setup_rag_chain(model_name, vector_store)
    print(f"user_prompt:{user_prompt}")
    return (rag_chain.invoke(user_prompt) or "").strip()

def call_ollama(full_prompt: str, model_name: str) -> str:
    print(f"full_prompt used for ollama:{full_prompt}")
    # Raw REST for maximal control + portability
    payload = {
        "model": model_name,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 2048,
            "top_p": 0.9,
            "num_thread": min(os.cpu_count() or 4, 6)
        }
    }
    r = requests.post(OLLAMA_GEN_URL, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()

# Register routers
app.include_router(upload_router, prefix="/upload", tags=["Upload"])
