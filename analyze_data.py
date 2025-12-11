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

    print(f"--- Relevance Stats ---")
    print(f"Total links: {len(df_qrels)}")

if __name__ == "__main__":
    analyze_nfcorpus()
