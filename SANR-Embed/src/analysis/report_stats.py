import pandas as pd
import os

def report():
    gold_path = "SANR-Embed/data/processed/gold_standard.csv"
    train_path = "SANR-Embed/data/processed/train.csv"
    test_path = "SANR-Embed/data/processed/test.csv"
    emb_path = "SANR-Embed/data/processed/embeddings.csv"
    
    print("=== SANR-Embed Phase 1 Report ===\n")
    
    # Gold Standard
    if os.path.exists(gold_path):
        gold = pd.read_csv(gold_path)
        print(f"Gold Standard: {len(gold)} records")
        print("Columns:", list(gold.columns))
        print(f"Primary Labels: {gold['label_primary'].nunique()} unique classes")
    else:
        print("Gold Standard NOT FOUND")
        
    # Splits
    if os.path.exists(train_path) and os.path.exists(test_path):
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        print(f"\nTrain/Test Split: {len(train)} / {len(test)}")
    else:
        print("\nTrain/Test splits NOT FOUND")
        
    # Embeddings
    if os.path.exists(emb_path):
        emb = pd.read_csv(emb_path)
        print(f"\nAligned Embeddings: {len(emb)} vectors")
        
        if os.path.exists(gold_path):
            coverage = len(emb) / len(gold) * 100
            print(f"Coverage: {coverage:.1f}% of Gold Standard has embeddings")
            
            # Check drift/comparison note
            print("\nComparison Note:")
            print("Original 'sentence_embeddings.csv' IDs did NOT match 'Labeled Data.json' IDs.")
            print("Alignment was performed using text content matching.")
            print(f"Matched {len(emb)} records successfully.")
    else:
        print("\nAligned Embeddings NOT FOUND")

if __name__ == "__main__":
    report()







