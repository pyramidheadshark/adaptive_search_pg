import logging
from datasets import load_dataset
from sqlmodel import Session
from src.db.session import engine
from src.db.models import Document
from src.services.ml_service import get_embedding

logging.basicConfig(level=logging.INFO)

def load_data():
    logging.info("Downloading dataset...")
    dataset = load_dataset("BeIR/nfcorpus", "corpus", split="corpus")
    
    with Session(engine) as session:
        # Taking first 1000 for testing
        for item in dataset.select(range(1000)):
            text = item['text']
            title = item['title']
            full_text = f"{title} {text}"
            
            emb = get_embedding(full_text)
            doc = Document(content=full_text, embedding=emb)
            session.add(doc)
        
        session.commit()
    logging.info("Data loaded")

if __name__ == "__main__":
    load_data()
