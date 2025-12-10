import logging
import json
from datasets import load_dataset
from sqlmodel import Session
from src.db.session import engine
from src.services.search_service import search_documents

logging.basicConfig(level=logging.INFO)

def calculate_mrr(rank):
    return 1.0 / rank if rank > 0 else 0.0

def run_benchmark():
    queries = load_dataset("BeIR/nfcorpus", "queries", split="queries")
    
    results = []
    with Session(engine) as session:
        # Taking a small subset for quick testing
        for q in queries.select(range(20)):
            text = q['text']
            res = search_documents(session, text, limit=20)
            
            # Simple check if any result was returned
            top_id = res[0]['id'] if res else None
            results.append({"query": text, "top_1": top_id})
            
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    run_benchmark()
