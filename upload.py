# upload.py
import os, uuid, shutil, mimetypes, base64, json
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from app import Path as AppPath  # reuse paths if needed

UPLOAD_DIR = "uploads"
INDEX_PATH = Path("./vectordb")
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".csv"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

router = APIRouter()
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Helpers ---
def validate_file(file: UploadFile) -> bool:
    return Path(file.filename).suffix.lower() in ALLOWED_EXTENSIONS

def generate_unique_filename(original_filename: str) -> str:
    return f"{uuid.uuid4()}{Path(original_filename).suffix}"

def list_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def load_documents_from_files() -> List[Document]:
    files = [Path(UPLOAD_DIR) / f for f in list_files(UPLOAD_DIR)]
    docs: List[Document] = []
    for file in files:
        try:
            if file.suffix == ".txt":
                loader = TextLoader(str(file), encoding="utf-8")
            elif file.suffix == ".csv":
                loader = CSVLoader(str(file))
            elif file.suffix == ".pdf":
                loader = PyPDFLoader(str(file))
            else:
                continue
            docs.extend(loader.load())
        except Exception as e:
            print(f"[Index] Failed to load {file}: {e}")
    return docs

def create_vector_db_from_files():
    from app import get_embeddings, reset_vectorstore_cache, OLLAMA_EMBED_MODEL
    docs = load_documents_from_files()
    if not docs:
        raise ValueError("No documents loaded.")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    os.makedirs(INDEX_PATH, exist_ok=True)
    vectorstore.save_local(str(INDEX_PATH))
    reset_vectorstore_cache()
    return True

# --- Models ---
class AlertJsonBody(BaseModel):
    id: str
    crisis: str
    source: str
    issued_at: str
    expires_at: Optional[str] = None
    region: List[str]
    lat: float
    lon: float
    severity: str
    language: str
    url: str
    text: str

# --- Routes ---
@router.post("/file")
def upload_file(file: UploadFile = File(...)):
    if not validate_file(file):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not allowed. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    unique_filename = generate_unique_filename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    mime_type = mimetypes.guess_type(file.filename)[0]
    if create_vector_db_from_files():
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "message": "File uploaded successfully",
                "filename": unique_filename,
                "mime_type": mime_type,
                "file_path": file_path
            }
        )

@router.post("/json")
def upload_file_json(body: AlertJsonBody):
    unique_filename = generate_unique_filename("alert.json")
    file_path = Path(UPLOAD_DIR) / unique_filename
    file_path.write_text(json.dumps(body.dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    docs = [Document(page_content=body.text, metadata=body.dict())]
    from app import get_embeddings, reset_vectorstore_cache
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(str(INDEX_PATH))
    reset_vectorstore_cache()

    return JSONResponse(
        status_code=201,
        content={
            "message": "Alert uploaded successfully",
            "filename": unique_filename,
            "file_path": str(file_path),
            "id": body.id,
            "crisis": body.crisis,
        },
    )
