import math
from typing import List, Dict, Any
from sqlmodel import Session, select, text, func
from src.database import Document, Interaction
from src.ml import get_model

def calculate_boost(strategy: str, similarity: float, feedback: int) -> float:
    safe_feedback = max(0, feedback)
    
    if strategy == "log":
        return similarity * (1 + 0.05 * math.log(1 + safe_feedback))
        
    elif strategy == "linear":
        return similarity + (0.001 * safe_feedback)
        
    elif strategy == "sigmoid":
        k = 50
        boost = safe_feedback / (safe_feedback + k)
        return similarity * (1 + 0.5 * boost)
        
    return similarity

def search_documents(session: Session, query: str, limit: int = 10, strategy: str = "log") -> List[Dict[str, Any]]:
    model = get_model()
    query_vector = model.encode(query).tolist()
    
    candidates_limit = max(50, limit * 2)
    stmt = select(
        Document, 
        (Document.embedding.cosine_distance(query_vector)).label("distance")
    ).order_by(text("distance ASC")).limit(candidates_limit)
    
    results = session.exec(stmt).all()
    if not results:
        return []

    doc_ids = [doc.id for doc, _ in results]
    
    feedback_stmt = select(
        Interaction.document_id,
        func.sum(Interaction.score_delta).label("total_score")
    ).where(Interaction.document_id.in_(doc_ids)).group_by(Interaction.document_id)
    
    feedback_map = {row.document_id: row.total_score for row in session.exec(feedback_stmt).all()}

    processed_results = []
    for doc, distance in results:
        similarity = 1 - distance
        feedback = feedback_map.get(doc.id, 0)
        
        final_score = calculate_boost(strategy, similarity, feedback)
        
        processed_results.append({
            "id": doc.id,
            "content": doc.content,
            "category": doc.category,
            "score": final_score,
            "original_score": similarity,
            "feedback_score": feedback
        })
        
    processed_results.sort(key=lambda x: x["score"], reverse=True)
    
    return processed_results[:limit]
