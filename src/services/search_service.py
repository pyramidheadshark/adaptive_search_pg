from sqlalchemy import text
from sqlmodel import Session, select, func
from src.db.models import Document, Interaction
from src.services.ml_service import get_embedding
import math

def calculate_boost(strategy, sim, fb):
    safe_fb = max(0, fb)
    if strategy == "linear":
        return sim + (0.002 * safe_fb) # Reduced from 0.01
    elif strategy == "sigmoid":
        k = 20
        boost = safe_fb / (safe_fb + k)
        return sim * (1 + 0.5 * boost)
    else: 
        return sim * (1 + 0.1 * math.log(1 + safe_fb)) # Reduced from 0.5

def search_documents(session: Session, query: str, limit: int = 10, strategy: str = "log"):
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
        score = calculate_boost(strategy, sim, fb)
        ranked.append({"id": doc.id, "content": doc.content, "score": score, "feedback": fb})
        
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:limit]
