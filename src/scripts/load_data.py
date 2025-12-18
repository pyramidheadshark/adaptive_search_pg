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
    else:
        logger.warning("No HF_TOKEN found. You might hit rate limits (429).")

    with Session(engine) as session:
        try:
            statement = select(Document).limit(1)
            result = session.exec(statement).first()
            if result:
                logger.info("Database already contains data. Skipping load.")
                return
        except Exception as e:
            logger.warning(f"Error checking data presence: {e}")

    logger.info("Loading NFCorpus dataset from HuggingFace...")
    dataset = load_dataset("BeIR/nfcorpus", "corpus", split="corpus")
        
    logger.info(f"Loaded {len(dataset)} documents.")

    logger.info("Loading SentenceTransformer model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    batch_size = 64
    documents_buffer: List[Document] = []
    
    logger.info("Starting encoding and ingestion...")
    
    with Session(engine) as session:
        for i in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
            batch = dataset[i : i + batch_size]
            
            ids = batch["_id"]
            texts = batch["text"]
            titles = batch["title"]
            
            combined_texts = [f"{t} {txt}".strip() for t, txt in zip(titles, texts)]
            
            embeddings = model.encode(combined_texts)
            
            for j, text_content in enumerate(combined_texts):
                doc = Document(
                    content=text_content,
                    category="nutrition",
                    embedding=embeddings[j].tolist()
                )
                documents_buffer.append(doc)
            
            session.add_all(documents_buffer)
            session.commit()
            documents_buffer.clear()
            
    logger.info("Data loading completed successfully!")

if __name__ == "__main__":
    init_db()
    check_db_connection()
    load_nfcorpus()
