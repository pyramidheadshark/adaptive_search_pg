import pandas as pd
from datasets import load_dataset
import sys

def analyze_nfcorpus():
    try:
        corpus = load_dataset("BeIR/nfcorpus", "corpus", split="corpus")
        queries = load_dataset("BeIR/nfcorpus", "queries", split="queries")
        qrels = load_dataset("BeIR/nfcorpus-qrels", split="test")
    except Exception as e:
        sys.exit(1)

    df_corpus = pd.DataFrame(corpus)
    df_queries = pd.DataFrame(queries)
    df_qrels = pd.DataFrame(qrels)

    print(f"--- Corpus Stats ---")
    print(f"Documents: {len(df_corpus)}")
    print(f"Avg words: {df_corpus['text'].apply(lambda x: len(x.split())).mean():.1f}")

    print(f"\n--- Queries Stats ---")
    print(f"Total queries: {len(df_queries)}")

    print(f"\n--- Relevance Stats (Qrels) ---")
    print(f"Total links: {len(df_qrels)}")
    counts = df_qrels.groupby('query-id').size()
    print(f"Queries with answers: {len(counts)}")
    print(f"Avg docs per query: {counts.mean():.2f}")
    print(f"Max docs per query: {counts.max()}")

    print(f"\n--- Score Distribution ---")
    print(df_qrels['score'].value_counts().sort_index())

    queries_with_multi_scores = df_qrels.groupby('query-id')['score'].nunique()
    multi_score_count = (queries_with_multi_scores > 1).sum()
    print(f"\nQueries with multi-level relevance: {multi_score_count}")

if __name__ == "__main__":
    analyze_nfcorpus()
