import pandas as pd
import re
import numpy as np
import os

def normalize(text):
    if not isinstance(text, str): return ""
    return re.sub(r'\s+', '', text.lower())

def parse_embedding(emb_str):
    # Convert numpy string format '[ -0.1  0.2 ... ]' to list
    try:
        # Remove brackets
        content = emb_str.strip('[]')
        # Split by whitespace
        values = content.split()
        # Convert to floats
        return [float(v) for v in values]
    except:
        return []

def align_embeddings():
    gold_path = "SANR-Embed/data/processed/gold_standard.csv"
    emb_path = "SpanishNotaryCollection/model/sentence_embeddings.csv"
    output_path = "SANR-Embed/data/processed/embeddings.csv"
    
    print("Loading files for alignment...")
    gold = pd.read_csv(gold_path)
    emb = pd.read_csv(emb_path)
    
    # Create mapping key
    gold['norm_text'] = gold['text_original'].apply(normalize)
    emb['norm_text'] = emb['Sentence'].apply(normalize)
    
    # Drop duplicates in embeddings to avoid explosion (we saw 2 duplicates)
    emb = emb.drop_duplicates(subset=['norm_text'])
    
    # Merge
    merged = pd.merge(gold[['id', 'norm_text']], emb[['norm_text', 'Embedding']], on='norm_text', how='inner')
    
    print(f"Matched {len(merged)} embeddings out of {len(gold)} gold records.")
    
    # Clean up embedding format
    # The source format is likely numpy string. Let's clean it.
    # Actually, let's just keep it simple: ID, embedding_vector (as json string)
    import json
    
    def clean_vector(s):
        vec = parse_embedding(s)
        return json.dumps(vec)
        
    merged['vector'] = merged['Embedding'].apply(clean_vector)
    
    final_df = merged[['id', 'vector']]
    final_df.to_csv(output_path, index=False)
    print(f"Saved aligned embeddings to {output_path}")

if __name__ == "__main__":
    # Create directory if needed
    os.makedirs("SANR-Embed/src/embeddings", exist_ok=True) # Just ensuring the dir exists for the script
    align_embeddings()







