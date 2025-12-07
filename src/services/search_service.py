from sqlmodel import Session, select, func
from src.db.models import Document, Interaction
from src.services.ml_service import get_embedding

def search_documents(session: Session, query: str, limit: int = 10):
    vector = get_embedding(query)
    
    # 1. Vector Search
    stmt = select(Document, Document.embedding.cosine_distance(vector).label("dist")).order_by(text("dist")).limit(limit * 2)
    candidates = session.exec(stmt).all()
    
    # 2. Get Feedback
    doc_ids = [doc.id for doc, _ in candidates]
    fb_stmt = select(Interaction.document_id, func.sum(Interaction.score_delta)).where(Interaction.document_id.in_(doc_ids)).group_by(Interaction.document_id)
    feedback_map = {row[0]: row[1] for row in session.exec(fb_stmt).all()}
    
    # 3. Re-rank (Linear)
    ranked = []
    for doc, dist in candidates:
        sim = 1 - dist
        fb = feedback_map.get(doc.id, 0)
        score = sim + (0.01 * fb)
        ranked.append({"doc": doc, "score": score})
        
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return [x["doc"] for x in ranked[:limit]]
