"""
Script to create FAISS index from articles
"""
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss
import numpy as np
import pickle
import os
from tqdm import tqdm
import sys

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import preprocess_dataset, get_text_and_labels

def create_index(model_path: str = "models/news_contrastive_model",
                 output_index_path: str = "models/faiss_index.bin",
                 output_articles_path: str = "models/articles.pkl",
                 dataset_name: str = "ag_news",
                 max_articles: int = 10000):
    """
    Create FAISS index from articles.
    
    Args:
        model_path: Path to trained model
        output_index_path: Path to save FAISS index
        output_articles_path: Path to save articles list
        dataset_name: Name of dataset to use
        max_articles: Maximum number of articles to index
    """
    print("Loading model...")
    if os.path.exists(model_path):
        model = SentenceTransformer(model_path)
    else:
        print("Fine-tuned model not found, using baseline")
        model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    dataset = preprocess_dataset(dataset)
    
    # Get articles using the data_loader utility
    articles, _ = get_text_and_labels(dataset['train'], max_samples=max_articles)
    
    print(f"Encoding {len(articles)} articles...")
    embeddings = model.encode(articles, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    print(f"Creating FAISS index with dimension {dimension}...")
    
    # Use L2 distance index
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save index
    os.makedirs(os.path.dirname(output_index_path), exist_ok=True)
    faiss.write_index(index, output_index_path)
    print(f"Index saved to {output_index_path}")
    
    # Save articles
    with open(output_articles_path, 'wb') as f:
        pickle.dump(articles, f)
    print(f"Articles saved to {output_articles_path}")
    
    print(f"\nIndex created successfully!")
    print(f"  - Articles: {len(articles)}")
    print(f"  - Dimension: {dimension}")
    print(f"  - Index type: L2 distance")


if __name__ == "__main__":
    create_index()

