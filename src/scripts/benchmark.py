import json
import logging
import sys
import random
from typing import List, Dict
import pandas as pd
from datasets import load_dataset
from sqlmodel import Session, select, delete, func
from tqdm import tqdm

from src.database import engine, init_db, Interaction, Document, User
from src.search import search_documents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")

def get_default_user_id():
    with Session(engine) as session:
        user = session.exec(select(User)).first()
        if not user:
            logger.error("No users found! Run 'make load' first.")
            sys.exit(1)
        return user.id

def check_db_data():
    with Session(engine) as session:
        count = session.exec(select(func.count(Document.id))).one()
        if count == 0:
            logger.error("Database is empty! Run 'make load' first.")
            sys.exit(1)

def reset_feedback():
    with Session(engine) as session:
        session.exec(delete(Interaction))
        session.commit()

def get_valid_queries(limit: int = None):
    queries_dataset = load_dataset("BeIR/nfcorpus", "queries", split="queries")
    qrels_dataset = load_dataset("BeIR/nfcorpus-qrels", split="test")
    
    df_qrels = pd.DataFrame(qrels_dataset)
    df_relevant = df_qrels[df_qrels['score'] >= 1].copy()
    df_relevant['query-id'] = df_relevant['query-id'].astype(str)
    valid_qids = set(df_relevant['query-id'].unique())
    
    valid_queries_data = []
    for q in queries_dataset:
        qid_str = str(q['_id'])
        if qid_str in valid_qids:
            valid_queries_data.append((qid_str, q['text']))
            
    if limit:
        return valid_queries_data[:limit]
    return valid_queries_data

def get_rank(results, target_id):
    for i, doc in enumerate(results):
        if doc["id"] == target_id:
            return i + 1
    return 21 

def run_detailed_experiment(
    exp_name: str,
    strategies: List[str],
    queries: List[tuple],
    max_clicks: int,
    noise_prob: float,
    checkpoints: List[int],
    user_id: int
) -> List[Dict]:
    
    raw_data = []

    for strategy in strategies:
        reset_feedback()
        
        with Session(engine) as session:
            for q_id, q_text in tqdm(queries, desc=f"[{exp_name}] {strategy}"):
                results_init = search_documents(session, q_text, limit=20, strategy=strategy)
                if len(results_init) < 5:
                    continue
                
                target_doc = results_init[4]
                distractor_doc = results_init[0]
                if distractor_doc["id"] == target_doc["id"]:
                    if len(results_init) > 1: distractor_doc = results_init[1]

                if 0 in checkpoints:
                    raw_data.append({
                        "experiment": exp_name,
                        "strategy": strategy,
                        "query_id": q_id,
                        "clicks": 0,
                        "target_rank": 5,
                        "distractor_rank": 1,
                        "is_noisy": False
                    })

                current_clicks = 0
                for i in range(1, max_clicks + 1):
                    is_noise = random.random() < noise_prob
                    click_target_id = distractor_doc["id"] if is_noise else target_doc["id"]
                    
                    inter = Interaction(
                        user_id=user_id,
                        document_id=click_target_id,
                        query_text=q_text,
                        score_delta=1
                    )
                    session.add(inter)
                    session.commit()
                    current_clicks += 1

                    if current_clicks in checkpoints:
                        results_current = search_documents(session, q_text, limit=20, strategy=strategy)
                        t_rank = get_rank(results_current, target_doc["id"])
                        d_rank = get_rank(results_current, distractor_doc["id"])
                        
                        raw_data.append({
                            "experiment": exp_name,
                            "strategy": strategy,
                            "query_id": q_id,
                            "clicks": current_clicks,
                            "target_rank": t_rank,
                            "distractor_rank": d_rank,
                            "is_noisy": is_noise
                        })

    return raw_data

def main():
    init_db()
    check_db_data()
    user_id = get_default_user_id()
    
    strategies = ["log", "linear", "sigmoid"]
    full_dataset = get_valid_queries(limit=None)
    subset = full_dataset[:40]

    all_data = []

    logger.info("Starting Exp 1: Efficiency...")
    data1 = run_detailed_experiment(
        "Efficiency", strategies, full_dataset, 
        max_clicks=5, noise_prob=0.0, checkpoints=[0, 5], user_id=user_id
    )
    all_data.extend(data1)

    logger.info("Starting Exp 2: Noise...")
    data2 = run_detailed_experiment(
        "Noise", strategies, subset, 
        max_clicks=5, noise_prob=0.2, checkpoints=[0, 5], user_id=user_id
    )
    all_data.extend(data2)

    logger.info("Starting Exp 3: Saturation...")
    data3 = run_detailed_experiment(
        "Saturation", strategies, subset, 
        max_clicks=20, noise_prob=0.0, 
        checkpoints=[0, 1, 3, 5, 10, 15, 20], user_id=user_id
    )
    all_data.extend(data3)

    output_path = "data/benchmark_raw_data.json"
    with open(output_path, "w") as f:
        json.dump(all_data, f, indent=2)
    
    logger.info(f"Saved raw data to {output_path}")

if __name__ == "__main__":
    main()
