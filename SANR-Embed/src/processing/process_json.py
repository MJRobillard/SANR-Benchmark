import json
import pandas as pd
import re

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_dataset(json_path, output_path):
    print(f"Reading {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = []
    for item in data['records']:
        # Extract core fields
        record = {
            'id': item.get('id'),
            'year': item.get('year'),
            'notary': item.get('author_name'),
            'rollo': item.get('rollo_number'),
            'image_num': item.get('image_number'),
            'text_original': normalize_text(item.get('content')),
        }
        
        # Extract Labels
        # Primary label is the first in the list, or 'Unknown'
        classes = item.get('class_label', [])
        record['label_primary'] = classes[0] if classes else None
        
        # Extended label as a string for easier CSV handling
        extended = item.get('extended_class_label', [])
        record['label_extended'] = "; ".join(extended) if extended else None
        
        # Wikidata URIs
        uris = item.get('wikidata_concept', [])
        record['wikidata_uri'] = "; ".join(uris) if uris else None

        records.append(record)

    df = pd.DataFrame(records)
    
    # Filter out empty text or empty labels
    initial_count = len(df)
    df = df[df['text_original'].str.len() > 0]
    df = df[df['label_primary'].notna()]
    
    print(f"Processed {len(df)} records (filtered from {initial_count}).")
    print("\nClass Distribution (Top 20):")
    print(df['label_primary'].value_counts().head(20))
    
    print(f"\nSaving to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    # Ensure directory exists
    import os
    os.makedirs("SANR-Embed/data/processed", exist_ok=True)
    
    process_dataset(
        json_path="SpanishNotaryCollection/dataset/Labeled Data.json",
        output_path="SANR-Embed/data/processed/gold_standard.csv"
    )







