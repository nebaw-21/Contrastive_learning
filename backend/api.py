"""
FastAPI backend for news article semantic similarity search
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Optional
import os
import pickle

app = FastAPI(title="News Article Semantic Similarity API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
index = None
articles = None
embeddings = None


class SearchRequest(BaseModel):
    article: str
    top_k: int = 5


class SearchResponse(BaseModel):
    similar_articles: List[str]
    scores: List[float]
    indices: List[int]


def load_model(model_path: str = "models/news_contrastive_model"):
    """Load the trained model."""
    global model
    if os.path.exists(model_path):
        print(f"Loading fine-tuned model from {model_path}")
        model = SentenceTransformer(model_path)
    else:
        print("Fine-tuned model not found, using baseline model")
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model


def load_index(index_path: str = "models/faiss_index.bin", 
               articles_path: str = "models/articles.pkl"):
    """Load FAISS index and articles."""
    global index, articles
    
    if os.path.exists(index_path) and os.path.exists(articles_path):
        print(f"Loading FAISS index from {index_path}")
        index = faiss.read_index(index_path)
        
        with open(articles_path, 'rb') as f:
            articles = pickle.load(f)
        
        print(f"Loaded {len(articles)} articles")
    else:
        print("Index not found. Please run the indexing script first.")
        index = None
        articles = None
    
    return index, articles


@app.on_event("startup")
async def startup_event():
    """Initialize model and index on startup."""
    global model, index, articles
    model = load_model()
    index, articles = load_index()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "News Article Semantic Similarity API",
        "endpoints": {
            "/search": "POST - Search for similar articles",
            "/health": "GET - Health check",
            "/encode": "POST - Encode a single article"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "index_loaded": index is not None,
        "num_articles": len(articles) if articles else 0
    }


@app.post("/encode", response_model=dict)
async def encode_article(request: SearchRequest):
    """Encode a single article to embedding."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    embedding = model.encode([request.article])[0]
    return {
        "embedding": embedding.tolist(),
        "dimension": len(embedding)
    }


@app.post("/search", response_model=SearchResponse)
async def search_articles(request: SearchRequest):
    """
    Search for similar articles.
    
    Args:
        request: SearchRequest with article text and top_k
        
    Returns:
        SearchResponse with similar articles, scores, and indices
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if index is None or articles is None:
        raise HTTPException(status_code=500, detail="Index not loaded. Please run indexing first.")
    
    # Encode query article
    query_embedding = model.encode([request.article])
    query_embedding = np.array(query_embedding).astype('float32')
    
    # Search in FAISS index
    k = min(request.top_k, len(articles))
    distances, indices = index.search(query_embedding, k)
    
    # Convert distances to similarity scores (L2 distance -> similarity)
    # For cosine similarity, we'd use dot product instead
    scores = (1.0 / (1.0 + distances[0])).tolist()  # Convert distance to similarity
    
    # Get similar articles
    similar_articles = [articles[idx] for idx in indices[0]]
    
    return SearchResponse(
        similar_articles=similar_articles,
        scores=scores,
        indices=indices[0].tolist()
    )


@app.post("/search_batch", response_model=dict)
async def search_batch(articles: List[str], top_k: int = 5):
    """
    Search for similar articles for multiple queries.
    
    Args:
        articles: List of article texts
        top_k: Number of results per query
        
    Returns:
        Dictionary with results for each query
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if index is None or articles is None:
        raise HTTPException(status_code=500, detail="Index not loaded")
    
    # Encode all queries
    query_embeddings = model.encode(articles)
    query_embeddings = np.array(query_embeddings).astype('float32')
    
    # Search for each query
    k = min(top_k, len(articles))
    distances, indices = index.search(query_embeddings, k)
    
    results = []
    for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
        scores = (1.0 / (1.0 + dist_row)).tolist()
        similar_articles = [articles[idx] for idx in idx_row]
        results.append({
            "query": articles[i],
            "similar_articles": similar_articles,
            "scores": scores,
            "indices": idx_row.tolist()
        })
    
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

