from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
from pathlib import Path
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Reuse the loading and search functions from app.py
def load_indexes_and_metadata(embeddings_dir, processed_dir):
    indexes = {}
    metadatas = {}
    
    for faiss_file in Path(embeddings_dir).glob("*_index.faiss"):
        base = faiss_file.stem.replace('_index', '')
        json_file = Path(processed_dir) / f"{base}.json"
        
        if not json_file.exists():
            continue
            
        try:
            # Load FAISS index
            index = faiss.read_index(str(faiss_file))
            indexes[base] = index
            
            # Load metadata
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            metadatas[base] = data['chunks']
        except Exception as e:
            print(f"Error loading {faiss_file} or {json_file}: {str(e)}")
            
    return indexes, metadatas

def search(query, indexes, metadatas, model, top_k=10):
    query_emb = model.encode([query])
    results = []
    for base, index in indexes.items():
        D, I = index.search(query_emb.astype(np.float32), top_k)
        for dist, idx in zip(D[0], I[0]):
            if idx < len(metadatas[base]):
                chunk = metadatas[base][idx]
                similarity = 1 / (1 + dist)
                results.append({
                    'source_file': str(chunk['source_file']),
                    'chunk_id': str(chunk['chunk_id']),
                    'text': str(chunk['text']),
                    'score': float(dist),
                    'similarity': float(similarity)
                })
    # Sort by score (lower is better for L2)
    results = sorted(results, key=lambda x: x['score'])[:top_k]
    return results

# Initialize FastAPI app
app = FastAPI(title="Vector Search API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and indexes at startup
model = None
indexes = None
metadatas = None

@app.on_event("startup")
async def startup_event():
    global model, indexes, metadatas
    # Initialize model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load indexes and metadata
    base_dir = Path(__file__).parent.parent
    embeddings_dir = base_dir / "data/embeddings"
    processed_dir = base_dir / "data/processed"
    
    indexes, metadatas = load_indexes_and_metadata(embeddings_dir, processed_dir)
    if not indexes:
        raise Exception("No FAISS indexes found. Please run the embedding script first.")

# Request and response models
class SearchRequest(BaseModel):
    query: str

class SearchResult(BaseModel):
    source_file: str
    chunk_id: str
    text: str
    score: float
    similarity: float

class SearchResponse(BaseModel):
    results: List[SearchResult]

@app.post("/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    if not model or not indexes:
        raise HTTPException(status_code=500, detail="Search system not properly initialized")
    
    results = search(
        request.query,
        indexes,
        metadatas,
        model,
        top_k=10  # Fixed to return top 10 results
    )
    
    return SearchResponse(results=results)

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 