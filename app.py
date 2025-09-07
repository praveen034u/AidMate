# app.py  (voice-powered, minimal no-RAG)

import os
import uuid
import subprocess
import re
import shutil
import tempfile
import mimetypes
from pathlib import Path
from typing import Optional, Dict, List

import requests
from fastapi import FastAPI, Form, UploadFile, File, Header, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -------------------------
# Config
# -------------------------
UPLOAD_DIR = "uploads"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".doc", ".docx"}  # loaders below support txt/csv/pdf; others will be skipped gracefully
WORKDIR = Path(r"C:\SourceCode\AidMate")
STATIC_DIR = WORKDIR / "static"
AUDIO_DIR = STATIC_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Piper (TTS)
PIPER_EXE = WORKDIR / "piper/piper" / "piper.exe"
PIPER_MODEL = WORKDIR / "piper" / "en_US-amy-low.onnx"
PIPER_CFG = WORKDIR / "piper" / "en_US-amy-low.onnx.json"

# ---- Whisper.cpp (STT) ----
WHISPER_DIR = WORKDIR / "whisper"
WHISPER_EXE_NEW = WHISPER_DIR / "whisper-cli.exe"
WHISPER_EXE_OLD = WHISPER_DIR / "main.exe"
WHISPER_MODEL = WHISPER_DIR / "ggml-base.en.bin"
FFMPEG_BIN = "ffmpeg"

# Ollama
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
OLLAMA_GEN_URL = f"http://{OLLAMA_HOST}/api/generate"
# Kept for completeness; embeddings now use HuggingFace for CPU performance
OLLAMA_EMBED_MODEL = "nomic-embed-text:latest"

# Vector DB
VECTOR_DB_ROOT = Path("./vectordb")
VECTOR_DB_ROOT.mkdir(parents=True, exist_ok=True)

# -------No Answer Indicators when VectorDb does not contain relevant information------------
no_answer_indicators = [
    "not in the database",
    "no information",
    "cannot find",
    "i don't know",
    "no relevant information",
    "not mentioned",
    "not available in the context",
    "i may be able to assist you better",
    "not a medical professional",
]

# -------------------------
# Performance Caching
# -------------------------
_VECTORSTORE = None
_EMBEDDINGS = None
_CHAIN_CACHE: Dict[tuple, object] = {}

def get_embeddings():
    """Fast CPU embedding model."""
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDINGS

def get_vectorstore():
    """Load FAISS if present; otherwise initialize an empty store."""
    global _VECTORSTORE
    if _VECTORSTORE is None:
        embeddings = get_embeddings()
        try:
            _VECTORSTORE = FAISS.load_local(
                VECTOR_DB_ROOT, embeddings, allow_dangerous_deserialization=True
            )
        except Exception:
            _VECTORSTORE = FAISS.from_documents([], embeddings)
    return _VECTORSTORE

def invalidate_vector_cache():
    global _VECTORSTORE, _CHAIN_CACHE
    _VECTORSTORE = None
    _CHAIN_CACHE.clear()

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
    if WHISPER_EXE_NEW.exists():
        return WHISPER_EXE_NEW
    elif WHISPER_EXE_OLD.exists():
        print("Warning: using deprecated whisper.exe (will still work)")
        return WHISPER_EXE_OLD
    else:
        raise RuntimeError(
            "Whisper binary not found. Expected either:\n"
            f"  {WHISPER_EXE_NEW}\n"
            f"  {WHISPER_EXE_OLD}"
        )

def _discover_whisper_exe() -> Path:
    # Prefer new, then old, then try other likely exe names in the folder
    if WHISPER_EXE_NEW.exists():
        return WHISPER_EXE_NEW
    if WHISPER_EXE_OLD.exists():
        return WHISPER_EXE_OLD

    # Search for any exe that looks like whisper, or a generic main.exe
    cands = list(WHISPER_DIR.glob("*whisper*.exe")) + list(WHISPER_DIR.glob("main.exe"))
    if cands:
        return cands[0]
    raise RuntimeError(
        "Whisper binary not found. Expected one of:\n"
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
    if "webm" in ct:
        return ".webm"
    if "ogg" in ct:
        return ".ogg"
    if "wav" in ct or "x-wav" in ct:
        return ".wav"
    if "mpeg" in ct or "mp3" in ct:
        return ".mp3"
    if "mp4" in ct or "m4a" in ct or "aac" in ct:
        return ".m4a"
    return ".bin"

def _strip_whisper_warning(s: str) -> str:
    if not s:
        return ""
    lines = []
    for line in s.splitlines():
        ll = line.lower()
        if "binary 'whisper.exe' is deprecated" in ll:
            continue
        if "please use 'whisper-whisper.exe' instead" in ll:
            continue
        if "examples/deprecation-warning" in ll:
            continue
        lines.append(line)
    return "\n".join(lines).strip()

# -------------------------
# Conversation memory
# -------------------------
CONV_MEM: Dict[str, List[tuple]] = {}
MAX_TURNS = 8

def push_turn(session_id: str, role: str, text: str):
    if not session_id:
        return
    hist = CONV_MEM.setdefault(session_id, [])
    hist.append((role, text))
    if len(hist) > MAX_TURNS * 2:
        CONV_MEM[session_id] = hist[-MAX_TURNS * 2 :]

def render_history(session_id: str) -> str:
    """Turn memory into a short transcript for the model."""
    hist = CONV_MEM.get(session_id, [])
    lines = []
    for role, msg in hist[-MAX_TURNS * 2 :]:
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
        line = re.sub(r"(\*{1,3}|_{1,3})(.+?)\1", r"\2", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
        line = re.sub(r"`([^`]+)`", r"\1", line)
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
    model: str = "gpt-oss:latest"  # Option A alias (point this to a small model in Ollama)
    # model: str = "gpt-oss:20b"
    system: Optional[str] = (
        "You are AidMate, an offline assistant.\n"
        "- Start with a single H2 markdown heading (## Heading) naming the action/section.\n"
        "- Then provide concise, step-by-step numbered instructions (1., 2., 3.).\n"
        "- Do NOT use bold/italics/emphasis markers like **, *, or _ in the body.\n"
        "- Avoid decorative symbols; keep content clean and readable.\n"
        "- End with: Not a medical professional."
    )

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="AidMate Voice API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.post("/ask")
def ask(body: AskBody, x_session_id: Optional[str] = Header(default="")):
    print("Ask:", body.prompt)
    print("Session:", x_session_id or "(new)")

    history = render_history(x_session_id)
    sys_prefix = (body.system + "\n\n") if body.system else ""
    convo = (history + "\n\n") if history else ""

    reply = call_Vector_ollama(body.prompt, body.model)
    print("Reply:", reply)

    if any(indicator in reply.lower() for indicator in no_answer_indicators):
        # If the answer indicates no information found, fallback to plain LLM response
        print("No relevant information found in the context. Falling back to plain LLM response.")
        reply = call_ollama(sys_prefix + convo + "User: " + body.prompt + "\nAssistant:", body.model)
        print("Fallback Reply:", reply)

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
        stdin=subprocess.PIPE,
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

    # 1) Read uploaded bytes
    try:
        raw = file.file.read()
    finally:
        try:
            file.file.close()
        except Exception:
            pass

    if not raw or len(raw) == 0:
        return {"error": "no-audio", "detail": "Uploaded file is empty (0 bytes)."}

    import glob
    import json

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        ext = _guess_ext(file.content_type)
        in_path = td_path / f"input{ext}"
        wav_path = td_path / "audio.wav"

        in_path.write_bytes(raw)

        # 2) ffmpeg -> 16kHz mono wav
        try:
            conv = subprocess.run(
                [
                    FFMPEG_BIN,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(in_path),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    str(wav_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=td,
                timeout=90,
            )
        except subprocess.TimeoutExpired:
            return {"error": "ffmpeg timeout", "detail": "ffmpeg exceeded 90s limit."}

        if conv.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size == 0:
            return {
                "error": "ffmpeg conversion failed",
                "detail": (conv.stderr or conv.stdout or "ffmpeg returned non-zero without output.")[:2000],
                "content_type": file.content_type,
            }

        # 3) Run Whisper.cpp with explicit output prefix
        out_txt = td_path / "out.txt"
        cmd = [
            str(whisper_exe),
            "-m",
            str(WHISPER_MODEL),
            "-f",
            str(wav_path),
            "-l",
            lang,
            "-otxt",
            "-of",
            "out",
        ]

        try:
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=td, timeout=300
            )
        except subprocess.TimeoutExpired:
            return {"error": "whisper timeout", "detail": "whisper exceeded 300s limit."}

        # 4) Prefer file output; if missing, fall back to stdout
        text = ""
        if out_txt.exists():
            try:
                text = out_txt.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                text = ""

        if not text:
            # Some builds print text to stdout if file writing fails
            text = (proc.stdout or "").strip()

        if text:
            return {"text": text}

        # 5) Return raw diagnostics (don’t over-clean; just trim length)
        diag = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip()
        return {"error": "whisper failed", "detail": diag[:2000]}

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

def validate_file(file: UploadFile) -> bool:
    """Validate file extension and size"""
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        return False
    return True

def generate_unique_filename(original_filename: str) -> str:
    file_extension = Path(original_filename).suffix
    unique_id = str(uuid.uuid4())
    return f"{unique_id}{file_extension}"

def generate_unique_foldername():
    unique_id = str(uuid.uuid4())
    return f"{unique_id}"

@app.post("/upload/file")
def upload_file(file: UploadFile = File(...)):
    try:
        if not validate_file(file):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}",
            )
        unique_filename = generate_unique_filename(file.filename)
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        mime_type = mimetypes.guess_type(file.filename)[0]
        if create_vector_db_from_files():
            print("Vector DB created/updated successfully.")
            invalidate_vector_cache()  # Invalidate cache after DB update
            return JSONResponse(
                status_code=status.HTTP_201_CREATED,
                content={
                    "message": "File uploaded successfully",
                    "filename": unique_filename,
                    "mime_type": mime_type,
                    "file_path": file_path,
                },
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to upload file: {str(e)}"
        )

def list_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def load_documents_from_files():
    files = list_files(UPLOAD_DIR)
    print(f"Found {len(files)} files.")
    files = [Path(UPLOAD_DIR) / f for f in files]
    docs: List[Document] = []
    for file in files:
        ext = file.suffix.lower()
        if ext == ".txt":
            loader = TextLoader(str(file), encoding="utf-8")
        elif ext == ".csv":
            loader = CSVLoader(str(file))
        elif ext == ".pdf":
            loader = PyPDFLoader(str(file))
        else:
            print(f"Skipping unsupported file type: {file}")
            continue
        docs.extend(loader.load())
    return docs

def create_vector_db_from_files():
    embeddings = get_embeddings()
    docs = load_documents_from_files()
    if not docs:
        raise ValueError("No documents loaded. Make sure files are valid.")
    print(f"Loaded {len(docs)} documents. Splitting + creating embeddings...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    print(f"Chunked into {len(chunks)} segments.")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    save_path = VECTOR_DB_ROOT
    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(str(save_path))
    print(f"Vector DB saved at {save_path}")
    return True

@app.get("/")
async def root():
    return {"message": "Hi, I am live"}

def setup_rag_chain(model, vector_store, humanMessage: str):
    llm = OllamaLLM(model=model, temperature=0.3)
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 2, "fetch_k": 6, "lambda_mult": 0.5},
    )

    def _combine_context(docs):
        max_chars = 2000
        out, used = [], 0
        for d in docs:
            t = (d.page_content or "")
            if used + len(t) > max_chars:
                break
            out.append(t)
            used += len(t)
        return "\n\n".join(out).strip()

    template = (
        "You are AidMate. Use only the provided context.\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION:\n{question}\n\n"
        "Answer briefly with steps (1-3). End with: Not a medical professional."
    )
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"docs": retriever, "question": RunnablePassthrough()}
        | (lambda x: {"context": _combine_context(x["docs"]), "question": x["question"]})
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def call_Vector_ollama(full_prompt: str, model: str) -> str:
    vector_store = get_vectorstore()
    key = (model,)
    if key not in _CHAIN_CACHE:
        _CHAIN_CACHE[key] = setup_rag_chain(model, vector_store, full_prompt)
    rag_chain = _CHAIN_CACHE[key]
    response = rag_chain.invoke(full_prompt)
    return response

def call_ollama(full_prompt: str, model: str) -> str:
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "num_predict": 256,
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.15,
            "num_ctx": 2048,
        },
    }
    r = requests.post(OLLAMA_GEN_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()
# -------------------------