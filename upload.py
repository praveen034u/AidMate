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
BASE_DIR = Path(__file__).resolve().parent

router = APIRouter()
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Helpers ---
def validate_file(file: UploadFile) -> bool:
    return Path(file.filename).suffix.lower() in ALLOWED_EXTENSIONS

def generate_unique_filename(original_filename: str) -> str:
    return f"{uuid.uuid4()}{Path(original_filename).suffix}"

# def list_files(directory):
#     return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def list_files(documentPath: str) -> List[Path]:
    """Return only valid files from a directory, ignoring dirs & hidden files."""
    path = Path(documentPath)
    if not path.exists() or not path.is_dir():
        print(f"[Index] Path does not exist or is not a directory: {documentPath}")
        return []

    return [
        f for f in path.rglob("*")
        if f.is_file() and not f.name.startswith(".")  # keep Path objects only
    ]

def load_documents_from_files(documenPath) -> List[Document]:
    files = list_files(documenPath)
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

def create_vector_db_from_files(documenPath,vectorPath,crisis):
    print("Creating vector DB from files... ")
    from app import get_embeddings, reset_vectorstore_cache, OLLAMA_EMBED_MODEL
    print("Using documenPath:", documenPath)
    docs = load_documents_from_files(documenPath)
    if not docs:
        raise ValueError("No documents loaded.")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    os.makedirs(vectorPath, exist_ok=True)
    vectorstore.save_local(str(vectorPath))
    reset_vectorstore_cache(crisis)
    return True

# --- Models ---
class AlertJsonBody(BaseModel):
    id: str
    crisis: str
    source: str
    issued_at: str
    expires_at: Optional[str] = None
    region: List[str]
    lat: Optional[float] = None
    lon: Optional[float] = None
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
    directoryPath = os.path.join(UPLOAD_DIR,"files")
    os.makedirs(directoryPath, exist_ok=True)

    file_path = os.path.join(directoryPath, unique_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    mime_type = mimetypes.guess_type(file.filename)[0]
    vectorDbPath = os.path.join(INDEX_PATH, "files")   
    os.makedirs(vectorDbPath, exist_ok=True)
    print("Vector DB directory ensured:", vectorDbPath)
    if create_vector_db_from_files(directoryPath,vectorDbPath,"files"):
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
    print("Received JSON body:", body)  
    crisis = body.crisis
    print("Crisis:", crisis)
    text = body.text
    print("Text:", text)

    if not crisis or not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required fields 'crisis' or 'text'."
        )
    file_name = "data.txt"  

    print("Creating directory for crisis:", crisis)

    print("Base directory:", BASE_DIR)

    directoryPath = os.path.join(BASE_DIR, UPLOAD_DIR, "json", crisis)
    print("Directory path:", directoryPath)

    os.makedirs(directoryPath, exist_ok=True)
    print("Directory created/ensured:", directoryPath)

    file_path = os.path.join(directoryPath, file_name)
    print("File path:", file_path)

    # ✅ Now you can safely create the file
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("")   # create empty file
        print("File created:", file_path)
    else:
        print("File already exists:", file_path)


    with open(file_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    print("Text appended to file.")
   
    # Update vector DB
    vectorDbPath = os.path.join(INDEX_PATH, crisis)   
    os.makedirs(vectorDbPath, exist_ok=True)
    print("Vector DB directory ensured:", vectorDbPath)
    create_vector_db_from_files(directoryPath,vectorDbPath,crisis)
    print("Vector DB updated.")
 

    return JSONResponse(
        status_code=201,
        content={
            "message": "json data  uploaded successfully",
            "filename": file_name,
            "file_path": str(file_path),
            "id": body.id,
            "crisis": body.crisis,
        },
    )
