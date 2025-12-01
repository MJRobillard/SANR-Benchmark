from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
from typing import List, Optional

app = FastAPI(title="SANR-Embed API", version="0.1.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "gold_standard.csv")

# Cache for data
_DATA_CACHE = None

def get_data():
    global _DATA_CACHE
    if _DATA_CACHE is None:
        if not os.path.exists(DATA_PATH):
            return pd.DataFrame()
        _DATA_CACHE = pd.read_csv(DATA_PATH)
        # Fill NaN values
        _DATA_CACHE = _DATA_CACHE.fillna("")
    return _DATA_CACHE

@app.get("/")
async def root():
    return {"message": "Welcome to SANR-Embed API", "status": "running"}

@app.get("/api/documents")
async def get_documents(limit: int = 100, offset: int = 0):
    df = get_data()
    if df.empty:
        return []
    
    # Slice and convert to list of dicts
    records = df.iloc[offset : offset + limit].to_dict(orient="records")
    return records

@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    df = get_data()
    if df.empty:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Ensure ID column is treated as string for comparison
    df['id'] = df['id'].astype(str)
    record = df[df['id'] == str(doc_id)]
    
    if record.empty:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return record.iloc[0].to_dict()

@app.get("/api/stats")
async def get_stats():
    df = get_data()
    if df.empty:
        return {"total": 0}
    
    return {
        "total_documents": len(df),
        "years": df['year'].unique().tolist(),
        "classes": df['label_primary'].unique().tolist(),
        "notaries": df['notary'].unique().tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)






