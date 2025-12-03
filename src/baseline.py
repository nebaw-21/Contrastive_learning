"""
Baseline evaluation using pre-trained models without fine-tuning
"""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple
import umap
import matplotlib.pyplot as plt


class BaselineEvaluator:
    """Evaluate baseline performance using pre-trained embeddings."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize baseline evaluator.
        
        Args:
            model_name: Name of pre-trained SentenceTransformer model
        """
        print(f"Loading baseline model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.texts = None
    
    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        print(f"Encoding {len(texts)} texts...")
        self.texts = texts
        self.embeddings = self.model.encode(texts, show_progress_bar=show_progress)
        return self.embeddings
    
    def compute_similarity(self, query_idx: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Compute cosine similarity and return top-K most similar articles.
        
        Args:
            query_idx: Index of query article
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        if self.embeddings is None:
            raise ValueError("Must encode texts first")
        
        query_emb = self.embeddings[query_idx:query_idx+1]
        similarities = cosine_similarity(query_emb, self.embeddings)[0]
        
        # Exclude the query itself
        similarities[query_idx] = -1
        
        # Get top-K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(idx, float(similarities[idx])) for idx in top_indices]
        
        return results
    
    def visualize_embeddings(self, labels: List[int] = None, n_samples: int = 1000, 
                            save_path: str = None, title: str = "Baseline Embeddings"):
        """
        Visualize embeddings using UMAP dimensionality reduction.
        
        Args:
            labels: Optional labels for coloring points
            n_samples: Number of samples to visualize (for large datasets)
            save_path: Path to save the plot
            title: Plot title
        """
        if self.embeddings is None:
            raise ValueError("Must encode texts first")
        
        embeddings_to_plot = self.embeddings
        labels_to_plot = labels
        
        if len(self.embeddings) > n_samples:
            indices = np.random.choice(len(self.embeddings), n_samples, replace=False)
            embeddings_to_plot = self.embeddings[indices]
            if labels:
                labels_to_plot = [labels[i] for i in indices]
        
        print("Reducing dimensions with UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings_to_plot)
        
        plt.figure(figsize=(12, 8))
        
        if labels_to_plot:
            scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels_to_plot, 
                                cmap='tab10', alpha=0.6, s=20)
            plt.colorbar(scatter, label='Topic Label')
        else:
            plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=20)
        
        plt.title(title)
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_loader import load_news_dataset, preprocess_dataset, get_text_and_labels
    
    # Test baseline evaluation
    dataset = load_news_dataset("ag_news")
    dataset = preprocess_dataset(dataset)
    
    texts, labels = get_text_and_labels(dataset['test'], max_samples=500)
    
    evaluator = BaselineEvaluator()
    embeddings = evaluator.encode(texts)
    
    # Test similarity search
    results = evaluator.compute_similarity(0, top_k=5)
    print(f"\nTop 5 similar articles to query:")
    for idx, score in results:
        print(f"  [{idx}] (score: {score:.4f}): {texts[idx][:80]}...")
    
    # Visualize
    evaluator.visualize_embeddings(labels, save_path="baseline_embeddings.png")

