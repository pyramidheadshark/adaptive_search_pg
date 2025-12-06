from fastapi import APIRouter, Depends
from sqlmodel import Session
from src.db.session import get_session
from src.services.search_service import search_documents
from pydantic import BaseModel

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    limit: int = 10

@router.post("/search")
def search(req: SearchRequest, session: Session = Depends(get_session)):
    docs = search_documents(session, req.query, req.limit)
    return {"results": docs}
