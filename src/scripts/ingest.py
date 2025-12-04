from src.db.session import SessionLocal
from src.db.models import Document
from src.services.ml_service import get_embedding

def ingest():
    db = SessionLocal()
    texts = ["hello world", "machine learning is cool", "databases are fun"]
    
    for t in texts:
        vec = get_embedding(t)
        doc = Document(content=t, embedding=vec)
        db.add(doc)
    
    db.commit()
    print("Done")

if __name__ == "__main__":
    ingest()
