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
from pydantic import BaseModel

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
# (config and helpers unchanged) ...
# -------------------------

app = FastAPI(title="AidMate Voice API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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
    # NEW: optional fields from the UI
    location: Optional[Dict[str, float]] = None
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

@app.post("/ask")
def ask(body: AskBody, x_session_id: Optional[str] = Header(default="")):
    print("Ask:", body.prompt)
    print("Session:", x_session_id or "(new)")
    print("Crisis:", body.crisis)
    if body.location:  # log for server visibility
        print("Location:", body.location)
    if body.contactNo:
        print("ContactNo:", body.contactNo)

    # background flag detection
    detectEmergencyFlag(body.prompt)

    # assemble conversation
    history = render_history(x_session_id)
    sys_prefix = (body.system + "\n\n") if body.system else ""
    convo = (history + "\n\n") if history else ""
    requested_model = resolve_chat_model(body.model)

    # NEW: neutral metadata header (consumed by LLM or downstream)
    meta = ""
    if body.location and isinstance(body.location, dict):
        lat = body.location.get("lat")
        lon = body.location.get("lon")
        acc = body.location.get("accuracy")
        if lat is not None and lon is not None:
            meta += "[User Location]\n"
            meta += f"Latitude: {lat}, Longitude: {lon}"
            if acc is not None:
                meta += f", Accuracy≈{acc}m"
            meta += "\n"
    if body.contactNo:
        meta += "[Contact]\n" + str(body.contactNo).strip() + "\n"

    # prefer RAG if an index exists for the selected crisis
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

# /tts, /stt, uploads, RAG helpers remain unchanged from your file...
app.include_router(upload_router, prefix="/upload", tags=["Upload"])
