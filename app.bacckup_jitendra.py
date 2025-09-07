# app.py  (voice-powered, minimal no-RAG)


import os, uuid, subprocess, re, shutil, tempfile
from pathlib import Path
from typing import Optional, Dict, List
import requests
from fastapi import FastAPI, Form, UploadFile, File, Header , HTTPException, status
from fastapi.responses import JSONResponse
import mimetypes
import base64
import shutil
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough



# -------------------------
# Config
# -------------------------
UPLOAD_DIR = "uploads"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".doc", ".docx"}
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
WHISPER_EXE_NEW = WHISPER_DIR / "whisper-cli.exe"   # new naming in whisper.cpp
WHISPER_EXE_OLD = WHISPER_DIR / "main.exe"           # legacy binary (deprecated)
WHISPER_MODEL   = WHISPER_DIR / "ggml-base.en.bin"      # change to ggml-small.en.bin for better accuracy
FFMPEG_BIN      = "ffmpeg"                              # must be on PATH

# Ollama
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
OLLAMA_GEN_URL = f"http://{OLLAMA_HOST}/api/generate"
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
                    "not a medical professional"
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
            f" â€¢ {WHISPER_EXE_NEW}\n"
            f" â€¢ {WHISPER_EXE_OLD}"
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
    #model: str = "gpt-oss:20b"
    model: str = "llama3.1:8b"
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
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.post("/ask")
def ask(body: AskBody, x_session_id: Optional[str] = Header(default="")):
    # stitch short conversational context
    print("Ask:", body.prompt)
    print("Session:", x_session_id or "(new)")

    history = render_history(x_session_id)
    sys_prefix = (body.system + "\n\n") if body.system else ""
    convo = (history + "\n\n") if history else ""

    # Call the LLM + VectorDb  with the user prompt
    reply = call_Vector_ollama(body.prompt, body.model)
    print("Reply:", reply)

    if any(indicator in reply.lower() for indicator in no_answer_indicators):
            # If the answer indicates no information found, fallback to plain LLM response
            print("No relevant information found in the context. Falling back to plain LLM response.")
            reply = call_ollama(sys_prefix + convo + "User: " + body.prompt + "\nAssistant:", body.model)
            print("Fallback Reply:", reply)

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
    runs Whisper.cpp, and returns text. Emits useful diagnostics when it fails.
    """
    # 0) Tooling sanity
    try:
        whisper_exe = ensure_whisper_ok()   # verifies model + ffmpeg, resolves exe path
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

    import tempfile, glob, json
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        ext = _guess_ext(file.content_type)
        in_path  = td_path / f"input{ext}"
        wav_path = td_path / "audio.wav"

        in_path.write_bytes(raw)

        # 2) ffmpeg -> 16kHz mono wav
        try:
            conv = subprocess.run(
                [
                    FFMPEG_BIN, "-y",
                    "-hide_banner", "-loglevel", "error",
                    "-i", str(in_path),
                    "-ac", "1", "-ar", "16000",
                    str(wav_path),
                ],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=td, timeout=90
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
        cmd = [str(whisper_exe),
               "-m", str(WHISPER_MODEL),
               "-f", str(wav_path),
               "-l", lang,
               "-otxt", "-of", "out"]

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
    # Check file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        return False

    return True

def generate_unique_filename(original_filename: str) -> str:
    """Generate unique filename to avoid conflicts"""
    file_extension = Path(original_filename).suffix
    unique_id = str(uuid.uuid4())
    return f"{unique_id}{file_extension}"

def generate_unique_foldername():
    """Generate unique folder name to avoid conflicts"""  
    unique_id = str(uuid.uuid4())
    return f"{unique_id}"

@app.post("/upload/file")
def upload_file(file: UploadFile =  File(...)):
    """Upload a single file"""
    try:
        # Validate file
        if not validate_file(file):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
       
        # Generate unique filename
        unique_filename = generate_unique_filename(file.filename)
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        # Save file to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer) 
        
        # Get file info
        #file_size = len(content)
        mime_type = mimetypes.guess_type(file.filename)[0]

        # create or update vector DB
        if create_vector_db_from_files():
            print("Vector DB created/updated successfully.")
            # for filename in os.listdir(UPLOAD_DIR):
            #     file_path = os.path.join(UPLOAD_DIR, filename)
            #     if os.path.isfile(file_path): 
            #         print("File deleted successfully.") # ensures it's a file
            #         os.remove(file_path)                
               
            return JSONResponse(
                        status_code=status.HTTP_201_CREATED,
                            content={
                            "message": "File uploaded successfully",
                            "filename": unique_filename,               
                            "mime_type": mime_type,
                            "file_path": file_path
                        }
                    )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        )
    
def list_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
def load_documents_from_files():
    """Load documents from multiple file formats and return as LangChain Documents."""   
    files = list_files(UPLOAD_DIR)
    print(f"Found {len(files)} files.")
    files = [Path(UPLOAD_DIR) / f for f in files]   
    docs = []
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
    """Create and save a FAISS vector DB from multiple input files."""
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
    docs = load_documents_from_files()

    if not docs:
        raise ValueError("No documents loaded. Make sure files are valid.")

    print(f"Loaded {len(docs)} documents. Creating embeddings...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    save_path = VECTOR_DB_ROOT  # unique folder
    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(str(save_path))
    print(f"Vector DB saved at {save_path}")
    return True

@app.get("/")
async def root():
    return {"message": "Hi, I am live"}


 # 3. Set up the retrieval chain
def setup_rag_chain(model, vector_store, humanMessage: str):
    llm = OllamaLLM(model=model, temperature=0.1)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  

    template = """
    You are AidMate, an offline assistant that uses the provided context to answer the question {question}.
    - Use only the information provided in the context to answer the question.
    - Answer the question based only on the following context: {context}
    - If the answer is not contained within the context, say "I don't know" and end with: Not a medical professional.
    - If the question is not related to the context, politely inform them that you are tuned to only answer questions related to the context.
    - Use a neutral, professional tone.
    - If the context contains multiple relevant pieces of information, synthesize them into a coherent answer.
    - Start with a single H2 markdown heading (## Heading) naming the action/section.
    - Then provide concise, step-by-step numbered instructions (1., 2., 3.).
    - Do NOT use bold/italics/emphasis markers like **, *, or _ in the body.
    - Avoid decorative symbols; keep content clean and readable."""

    prompt = ChatPromptTemplate.from_template(template)

    # 1) Map user question into {question}
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    
    return chain

def call_Vector_ollama(full_prompt: str, model: str) -> str:
        vector_store_path = VECTOR_DB_ROOT
        print(f"Loading vector store from {vector_store_path}...")
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
        print("Loading FAISS vector store...")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        print("Setting up RAG chain...")
        rag_chain = setup_rag_chain(model, vector_store, full_prompt)
        print("Invoking RAG chain...")
        print("Full prompt:", full_prompt)
        response = rag_chain.invoke(full_prompt)
        print("RAG chain response:", response)
        return response

def call_ollama(full_prompt: str, model: str) -> str:
    payload = {"model": model, "prompt": full_prompt, "stream": False}
    r = requests.post(OLLAMA_GEN_URL, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()





# -------- JSON Upload Support (non-breaking) --------
class UploadJsonBody(BaseModel):
    filename: str
    # Either provide raw text (for .txt) or base64 content (for any file)
    text: Optional[str] = None
    content_base64: Optional[str] = None
    mime_type: Optional[str] = None  # optional hint; used only for response

def _validate_json_upload(filename: str, byte_len: int) -> None:
    ext = Path(filename).suffix.lower() or ".txt"
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    if byte_len > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Max {MAX_FILE_SIZE} bytes"
        )

@app.post("/upload/file-json")
def upload_file_json(body: UploadJsonBody):
    """
    Upload a file using JSON payload.
    - Provide either `text` (for .txt files) OR `content_base64` (for binary/text).
    - `filename` must include an allowed extension.
    Behavior mirrors /upload/file: saves to disk and refreshes the vector DB.
    """
    if not body.filename:
        raise HTTPException(status_code=400, detail="filename is required")

    ext = Path(body.filename).suffix.lower() or ".txt"
    unique_filename = generate_unique_filename(body.filename)
    file_path = Path(UPLOAD_DIR) / unique_filename

    # Build bytes
    if body.text is not None and body.content_base64 is not None:
        raise HTTPException(status_code=400, detail="Provide either text or content_base64, not both.")
    if body.text is None and body.content_base64 is None:
        raise HTTPException(status_code=400, detail="Provide text or content_base64.")

    if body.text is not None:
        # Only permit .txt if sending raw text
        if ext != ".txt":
            raise HTTPException(status_code=400, detail="When using `text`, filename must have .txt extension.")
        data = body.text.encode("utf-8")
    else:
        # content_base64 path
        try:
            data = base64.b64decode(body.content_base64 or "", validate=True)
        except Exception:
            raise HTTPException(status_code=400, detail="content_base64 is not valid base64.")
        if not data:
            raise HTTPException(status_code=400, detail="Decoded content is empty.")

    # Validate type/size
    _validate_json_upload(body.filename, len(data))

    # Save
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(data)

    # Update vector DB (same as /upload/file)
    if create_vector_db_from_files():
        mime_type = body.mime_type or mimetypes.guess_type(body.filename)[0]
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "message": "File uploaded successfully (JSON)",
                "filename": unique_filename,
                "original_filename": body.filename,
                "mime_type": mime_type,
                "file_path": str(file_path),
                "bytes": len(data),
            },
        )
    # If we got here, index wasn't created for some reason
    raise HTTPException(status_code=500, detail="Vector DB update failed after upload.")


