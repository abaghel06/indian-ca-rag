import shutil
import tempfile
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from ingest import ingest_pdf, ingest_urls
from retriever import retrieve
from chain import answer

app = FastAPI(title="Indian CA RAG API")


# --- Request/Response Models ---

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    sources: List[str]

class IngestURLRequest(BaseModel):
    urls: List[str]

class IngestResponse(BaseModel):
    message: str
    chunks_added: int


# --- Endpoints ---

@app.post("/ask", response_model=QuestionResponse)
def ask(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    docs = retrieve(req.question)
    result = answer(req.question, docs)
    return result


@app.post("/ingest/pdf", response_model=IngestResponse)
def ingest_pdf_endpoint(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    chunks = ingest_pdf(tmp_path)
    return {"message": f"Ingested {file.filename}", "chunks_added": chunks}


@app.post("/ingest/urls", response_model=IngestResponse)
def ingest_urls_endpoint(req: IngestURLRequest):
    if not req.urls:
        raise HTTPException(status_code=400, detail="No URLs provided.")
    chunks = ingest_urls(req.urls)
    return {"message": f"Ingested {len(req.urls)} URL(s)", "chunks_added": chunks}


@app.get("/health")
def health():
    return {"status": "ok"}
