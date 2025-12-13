import time
import os
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

from src.database import init_db, get_session, Document, Interaction
from src.ml import get_model
from src.search import search_documents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    strategy: str = "log"

class SearchResult(BaseModel):
    id: int
    content: str
    category: Optional[str] = None
    score: float
    original_score: float
    feedback_score: int

class SearchResponse(BaseModel):
    results: List[SearchResult]
    execution_time_ms: float

class FeedbackRequest(BaseModel):
    document_id: int
    query: str
    score_delta: int = 1

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting app...")
    init_db()
    get_model()
    yield
    logger.info("Stopping app...")

app = FastAPI(title="Adaptive Search Engine", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/v1/search", response_model=SearchResponse)
def search_api(req: SearchRequest, session: Session = Depends(get_session)):
    start = time.time()
    results = search_documents(session, req.query, req.limit, req.strategy)
    elapsed = (time.time() - start) * 1000
    return {"results": results, "execution_time_ms": round(elapsed, 2)}

@app.post("/api/v1/feedback")
def feedback_api(req: FeedbackRequest, session: Session = Depends(get_session)):
    if not session.get(Document, req.document_id):
        raise HTTPException(404, "Document not found")
    inter = Interaction(document_id=req.document_id, query_text=req.query, score_delta=req.score_delta)
    session.add(inter)
    session.commit()
    return {"status": "ok", "new_score_delta": req.score_delta}
