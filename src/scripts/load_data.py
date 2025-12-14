import logging
import sys
import os
from typing import List
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sqlmodel import Session, select, text
from tqdm import tqdm
from huggingface_hub import login
from src.database import Document, engine, init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_db_connection():
    try:
        with Session(engine) as session:
            session.exec(text("SELECT 1"))
        logger.info("Database connection successful.")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        sys.exit(1)

def load_nfcorpus():
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        logger.info("Authenticating with Hugging Face...")
        login(token=hf_token.strip())
    
    logger.info("Loading NFCorpus dataset...")
    dataset = load_dataset("BeIR/nfcorpus", "corpus", split="corpus")
    logger.info("Loading Model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    batch_size = 64
    buffer = []
    
    logger.info("Starting ingestion...")
    with Session(engine) as session:
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i : i + batch_size]
            combined = [f"{t} {txt}".strip() for t, txt in zip(batch["title"], batch["text"])]
            embeddings = model.encode(combined)
            
            for j, txt in enumerate(combined):
                doc = Document(content=txt, category="nutrition", embedding=embeddings[j].tolist())
                buffer.append(doc)
            
            session.add_all(buffer)
            session.commit()
            buffer.clear()

if __name__ == "__main__":
    init_db()
    check_db_connection()
    load_nfcorpus()
