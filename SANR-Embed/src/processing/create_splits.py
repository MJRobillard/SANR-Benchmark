import pandas as pd
from sklearn.model_selection import train_test_split
import os

def create_splits():
    gold_path = "SANR-Embed/data/processed/gold_standard.csv"
    output_dir = "SANR-Embed/data/processed"
    
    if not os.path.exists(gold_path):
        print(f"Gold standard file not found at {gold_path}")
        return

    print(f"Reading {gold_path}...")
    df = pd.read_csv(gold_path)
    
    # Simple stratified split based on primary label if possible
    # Filter out classes with only 1 sample for stratification
    class_counts = df['label_primary'].value_counts()
    valid_classes = class_counts[class_counts > 1].index
    
    df_strat = df[df['label_primary'].isin(valid_classes)]
    df_rest = df[~df['label_primary'].isin(valid_classes)]
    
    print(f"Splitting {len(df_strat)} stratified records and {len(df_rest)} others...")
    
    train, test = train_test_split(
        df_strat, 
        test_size=0.2, 
        random_state=42, 
        stratify=df_strat['label_primary']
    )
    
    # Add the rest to train (or handle otherwise, but adding to train is safest for rare classes)
    train = pd.concat([train, df_rest])
    
    print(f"Train size: {len(train)}")
    print(f"Test size: {len(test)}")
    
    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print("Saved train.csv and test.csv")

if __name__ == "__main__":
    create_splits()







