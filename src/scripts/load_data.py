import logging
import sys
import os
from typing import List

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sqlmodel import Session, select, text
from tqdm import tqdm
from huggingface_hub import login

from src.database import Document, User, engine, init_db
from src.security import get_password_hash

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

def create_default_user():
    with Session(engine) as session:
        user = session.exec(select(User).where(User.username == "admin")).first()
        if not user:
            logger.info("Creating default admin user...")
            secure_hash = get_password_hash("admin123") 
            
            admin = User(
                username="admin", 
                password_hash=secure_hash,
                role="admin"
            )
            session.add(admin)
            session.commit()

def load_nfcorpus():
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        logger.info("Authenticating with Hugging Face...")
        login(token=hf_token.strip())
    
    create_default_user()

    with Session(engine) as session:
        try:
            statement = select(Document).limit(1)
            result = session.exec(statement).first()
            if result:
                logger.info("Database already contains data. Skipping load.")
                return
        except Exception as e:
            logger.warning(f"Error checking data: {e}")

    logger.info("Loading NFCorpus dataset...")
    dataset = load_dataset("BeIR/nfcorpus", "corpus", split="corpus")
    
    logger.info("Loading SentenceTransformer model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    batch_size = 64
    documents_buffer: List[Document] = []
    
    logger.info("Starting encoding...")
    
    with Session(engine) as session:
        for i in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
            batch = dataset[i : i + batch_size]
            titles = batch["title"]
            texts = batch["text"]
            
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
            
    logger.info("Data loading completed!")

if __name__ == "__main__":
    init_db()
    check_db_connection()
    load_nfcorpus()
