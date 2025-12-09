from sqlalchemy import text
from sqlmodel import Session, select, func
from src.db.models import Document, Interaction
from src.services.ml_service import get_embedding
import math

def search_documents(session: Session, query: str, limit: int = 10):
    vector = get_embedding(query)
    
    stmt = select(Document, Document.embedding.cosine_distance(vector).label("dist")).order_by(text("dist ASC")).limit(limit * 2)
    candidates = session.exec(stmt).all()
    
    if not candidates:
        return []
        
    doc_ids = [doc.id for doc, _ in candidates]
    fb_stmt = select(Interaction.document_id, func.sum(Interaction.score_delta)).where(Interaction.document_id.in_(doc_ids)).group_by(Interaction.document_id)
    feedback_map = {row[0]: row[1] for row in session.exec(fb_stmt).all()}
    
    ranked = []
    for doc, dist in candidates:
        sim = 1 - dist
        fb = feedback_map.get(doc.id, 0)
        
        # Log-Decay formula
        safe_fb = max(0, fb)
        score = sim * (1 + 0.5 * math.log(1 + safe_fb))
        
        ranked.append({"id": doc.id, "content": doc.content, "score": score})
        
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:limit]
