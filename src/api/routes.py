from fastapi import APIRouter, Depends
from sqlmodel import Session
from src.db.session import get_session
from src.db.models import Interaction
from src.services.search_service import search_documents
from pydantic import BaseModel

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    limit: int = 10

class FeedbackRequest(BaseModel):
    document_id: int
    query: str
    score_delta: int

@router.post("/search")
def search(req: SearchRequest, session: Session = Depends(get_session)):
    docs = search_documents(session, req.query, req.limit)
    return {"results": docs}

@router.post("/feedback")
def feedback(req: FeedbackRequest, session: Session = Depends(get_session)):
    inter = Interaction(
        document_id=req.document_id, 
        query_text=req.query, 
        score_delta=req.score_delta
    )
    session.add(inter)
    session.commit()
    return {"status": "recorded"}
