"""
Evaluation metrics and visualization for contrastive learning
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import umap
from collections import defaultdict


class Evaluator:
    """Evaluate model performance on retrieval tasks."""
    
    def __init__(self, model, texts: List[str], labels: List[int] = None):
        """
        Initialize evaluator.
        
        Args:
            model: SentenceTransformer model
            texts: List of article texts
            labels: Optional list of labels for ground truth
        """
        self.model = model
        self.texts = texts
        self.labels = labels
        self.embeddings = None
        self._encode()
    
    def _encode(self):
        """Encode all texts."""
        print("Encoding texts for evaluation...")
        self.embeddings = self.model.encode(self.texts, show_progress_bar=True)
    
    def recall_at_k(self, query_idx: int, k: int = 10, 
                   same_label_only: bool = True) -> float:
        """
        Compute Recall@K for a query.
        
        Args:
            query_idx: Index of query article
            k: Number of top results to consider
            same_label_only: Whether to only consider articles with same label as relevant
            
        Returns:
            Recall@K score
        """
        if self.labels is None:
            raise ValueError("Labels required for Recall@K")
        
        query_label = self.labels[query_idx]
        query_emb = self.embeddings[query_idx:query_idx+1]
        
        # Compute similarities
        similarities = cosine_similarity(query_emb, self.embeddings)[0]
        similarities[query_idx] = -1  # Exclude query itself
        
        # Get top-K indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        # Count relevant items (same label)
        relevant_count = sum(1 for idx in top_k_indices if self.labels[idx] == query_label)
        
        # Total relevant items (excluding query)
        total_relevant = sum(1 for i, label in enumerate(self.labels) 
                           if label == query_label and i != query_idx)
        
        if total_relevant == 0:
            return 0.0
        
        return relevant_count / total_relevant
    
    def mean_reciprocal_rank(self, query_idx: int, same_label_only: bool = True) -> float:
        """
        Compute Mean Reciprocal Rank (MRR) for a query.
        
        Args:
            query_idx: Index of query article
            same_label_only: Whether to only consider articles with same label as relevant
            
        Returns:
            MRR score
        """
        if self.labels is None:
            raise ValueError("Labels required for MRR")
        
        query_label = self.labels[query_idx]
        query_emb = self.embeddings[query_idx:query_idx+1]
        
        # Compute similarities
        similarities = cosine_similarity(query_emb, self.embeddings)[0]
        similarities[query_idx] = -1  # Exclude query itself
        
        # Get ranked indices
        ranked_indices = np.argsort(similarities)[::-1]
        
        # Find rank of first relevant item
        for rank, idx in enumerate(ranked_indices, start=1):
            if self.labels[idx] == query_label:
                return 1.0 / rank
        
        return 0.0
    
    def evaluate_all(self, k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
        """
        Evaluate on all queries.
        
        Args:
            k_values: List of K values for Recall@K
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("Evaluating on all queries...")
        
        results = {}
        
        # Compute Recall@K for each k
        for k in k_values:
            recalls = []
            for i in range(len(self.texts)):
                try:
                    recall = self.recall_at_k(i, k=k)
                    recalls.append(recall)
                except:
                    continue
            
            avg_recall = np.mean(recalls) if recalls else 0.0
            results[f'Recall@{k}'] = avg_recall
            print(f"Recall@{k}: {avg_recall:.4f}")
        
        # Compute MRR
        mrrs = []
        for i in range(len(self.texts)):
            try:
                mrr = self.mean_reciprocal_rank(i)
                mrrs.append(mrr)
            except:
                continue
        
        avg_mrr = np.mean(mrrs) if mrrs else 0.0
        results['MRR'] = avg_mrr
        print(f"MRR: {avg_mrr:.4f}")
        
        return results
    
    def visualize_embeddings(self, labels: List[int] = None, 
                           n_samples: int = 1000,
                           save_path: str = None,
                           title: str = "Embeddings Visualization"):
        """
        Visualize embeddings using UMAP.
        
        Args:
            labels: Optional labels for coloring
            n_samples: Number of samples to visualize
            save_path: Path to save plot
            title: Plot title
        """
        if labels is None:
            labels = self.labels
        
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
            scatter = plt.scatter(reduced[:, 0], reduced[:, 1], 
                                c=labels_to_plot, cmap='tab10', 
                                alpha=0.6, s=20)
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
    
    def compare_with_baseline(self, baseline_embeddings: np.ndarray,
                             k_values: List[int] = [1, 5, 10]) -> Dict[str, Dict[str, float]]:
        """
        Compare fine-tuned model with baseline.
        
        Args:
            baseline_embeddings: Baseline embeddings
            k_values: List of K values for Recall@K
            
        Returns:
            Dictionary comparing baseline and fine-tuned metrics
        """
        if self.labels is None:
            raise ValueError("Labels required for comparison")
        
        print("Comparing with baseline...")
        
        # Save current embeddings
        current_embeddings = self.embeddings.copy()
        
        # Evaluate baseline
        self.embeddings = baseline_embeddings
        baseline_results = self.evaluate_all(k_values)
        
        # Evaluate fine-tuned
        self.embeddings = current_embeddings
        finetuned_results = self.evaluate_all(k_values)
        
        # Compute improvements
        improvements = {}
        for metric in baseline_results:
            improvement = finetuned_results[metric] - baseline_results[metric]
            improvements[metric] = improvement
        
        comparison = {
            'baseline': baseline_results,
            'fine_tuned': finetuned_results,
            'improvement': improvements
        }
        
        print("\nComparison Results:")
        print("=" * 50)
        for metric in baseline_results:
            print(f"{metric}:")
            print(f"  Baseline:   {baseline_results[metric]:.4f}")
            print(f"  Fine-tuned: {finetuned_results[metric]:.4f}")
            print(f"  Improvement: {improvements[metric]:+.4f}")
        
        return comparison


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sentence_transformers import SentenceTransformer
    from src.data_loader import load_news_dataset, preprocess_dataset, get_text_and_labels
    
    # Test evaluation
    dataset = load_news_dataset("ag_news")
    dataset = preprocess_dataset(dataset)
    
    texts, labels = get_text_and_labels(dataset['test'], max_samples=500)
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    evaluator = Evaluator(model, texts, labels)
    
    results = evaluator.evaluate_all(k_values=[1, 5, 10])
    evaluator.visualize_embeddings(save_path="evaluation_embeddings.png")

