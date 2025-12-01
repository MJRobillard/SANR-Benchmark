import pandas as pd
import os
import sys
from sklearn.metrics import classification_report, f1_score

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from models import get_model

def evaluate(model, df, text_col, label_col, lang_name):
    print(f"\n" + "="*40)
    print(f"Evaluating on {lang_name} ({text_col})...")
    print("="*40)
    
    # Filter out rows where text or label is missing
    df_clean = df.dropna(subset=[text_col, label_col])
    print(f"Samples after dropping NaNs: {len(df_clean)} (Original: {len(df)})")
    
    texts = df_clean[text_col].astype(str).tolist()
    y_true = df_clean[label_col].tolist()
    
    label_set = sorted(list(set(y_true)))
    
    preds = []
    # Classify each text
    for i, text in enumerate(texts):
        if i % 100 == 0:
            print(f"Processing {i}/{len(texts)}...", end='\r')
        try:
            p = model.classify(text, label_set)
            preds.append(p)
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            preds.append("error")
    print(f"Processing {len(texts)}/{len(texts)} - Done.")

    f1 = f1_score(y_true, preds, average='macro', zero_division=0)
    print(f"\nMacro F1 for {lang_name}: {f1:.4f}")
    
    # Print full report
    print("\nClassification Report:")
    print(classification_report(y_true, preds, zero_division=0))

def main():
    # Path to the google translated test file
    data_path = os.path.join("SANR-Embed", "data", "processed", "test_google_translated.csv")
    
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    model_name = "classification_model"
    try:
        print(f"Loading model: {model_name}...")
        model = get_model(model_name)
    except ValueError as e:
        print(f"Error: {e}")
        print("Make sure the model files are in SANR-Embed/src/models/Classification/Classification")
        return

    # Evaluate English
    evaluate(model, df, "text_english", "label_primary", "English")
    
    # Evaluate Chinese
    evaluate(model, df, "text_chinese", "label_primary", "Chinese")

if __name__ == "__main__":
    main()







