import pandas as pd
import os

def verify_splits():
    train_path = "SANR-Embed/data/processed/train.csv"
    test_path = "SANR-Embed/data/processed/test.csv"
    
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print("Files not found.")
        return

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    print(f"Train Total: {len(train)}")
    print(f"Test Total: {len(test)}")
    
    # Get counts
    train_counts = train['label_primary'].value_counts()
    test_counts = test['label_primary'].value_counts()
    
    all_classes = set(train_counts.index) | set(test_counts.index)
    
    print("\nClass Distribution (Train vs Test):")
    print(f"{'Class':<20} | {'Train':<5} | {'Test':<5} | {'Total':<5}")
    print("-" * 45)
    
    for cls in sorted(list(all_classes)):
        tr = train_counts.get(cls, 0)
        te = test_counts.get(cls, 0)
        total = tr + te
        print(f"{str(cls):<20} | {tr:<5} | {te:<5} | {total:<5}")

if __name__ == "__main__":
    verify_splits()







