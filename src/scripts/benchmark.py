import logging
from datasets import load_dataset
from src.services.search_service import search_documents
from src.db.session import engine
from sqlmodel import Session

def run_benchmark():
    queries = load_dataset("BeIR/nfcorpus", "queries", split="queries")
    
    with Session(engine) as session:
        for q in queries.select(range(10)):
            print(f"Testing query: {q['text']}")
            res = search_documents(session, q['text'])
            print(res)

if __name__ == "__main__":
    run_benchmark()
