from sqlmodel import Session, select
from src.db.models import Document
from src.services.ml_service import get_embedding

def search_documents(session: Session, query: str, limit: int = 10):
    vector = get_embedding(query)
    
    # L2 distance (postgres <-> operator)
    stmt = select(Document).order_by(Document.embedding.l2_distance(vector)).limit(limit)
    results = session.exec(stmt).all()
    return results
