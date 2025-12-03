"""
Hard negative mining for contrastive learning
"""
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple, Set
import re


class HardNegativeMiner:
    """Mine hard negative examples for contrastive learning."""
    
    def __init__(self, model: SentenceTransformer = None):
        """
        Initialize hard negative miner.
        
        Args:
            model: Optional pre-trained SentenceTransformer for semantic search
        """
        self.model = model
        self.bm25 = None
        self.corpus = None
        self.embeddings = None
    
    def build_bm25_index(self, corpus: List[str]):
        """
        Build BM25 index for lexical hard negative mining.
        
        Args:
            corpus: List of documents to index
        """
        print("Building BM25 index...")
        self.corpus = corpus
        # Tokenize documents
        tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"Indexed {len(corpus)} documents")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Convert to lowercase and split on whitespace/punctuation
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def mine_bm25_hard_negatives(self, query: str, n: int = 10, 
                                 exclude_indices: Set[int] = None) -> List[int]:
        """
        Mine hard negatives using BM25 (lexical similarity).
        
        Args:
            query: Query text
            n: Number of hard negatives to return
            exclude_indices: Set of indices to exclude from results
            
        Returns:
            List of document indices
        """
        if self.bm25 is None:
            raise ValueError("Must build BM25 index first")
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-N indices
        top_indices = np.argsort(scores)[::-1]
        
        # Filter out excluded indices
        if exclude_indices:
            top_indices = [idx for idx in top_indices if idx not in exclude_indices]
        
        return top_indices[:n].tolist()
    
    def build_semantic_index(self, corpus: List[str], model: SentenceTransformer = None):
        """
        Build semantic index using embeddings.
        
        Args:
            corpus: List of documents
            model: SentenceTransformer model (uses self.model if None)
        """
        if model is None:
            model = self.model
        if model is None:
            raise ValueError("Must provide a model")
        
        print("Building semantic index...")
        self.corpus = corpus
        self.embeddings = model.encode(corpus, show_progress_bar=True)
        print(f"Indexed {len(corpus)} documents")
    
    def mine_semantic_hard_negatives(self, query: str, n: int = 10,
                                    exclude_indices: Set[int] = None,
                                    model: SentenceTransformer = None) -> List[int]:
        """
        Mine hard negatives using semantic similarity.
        
        Args:
            query: Query text
            n: Number of hard negatives to return
            exclude_indices: Set of indices to exclude
            model: SentenceTransformer model (uses self.model if None)
            
        Returns:
            List of document indices
        """
        if self.embeddings is None:
            if model is None:
                model = self.model
            if model is None:
                raise ValueError("Must build semantic index or provide model")
            self.build_semantic_index(self.corpus, model)
        
        if model is None:
            model = self.model
        
        # Encode query
        query_emb = model.encode([query])
        
        # Compute similarities
        similarities = cosine_similarity(query_emb, self.embeddings)[0]
        
        # Get top-N indices (high similarity but different meaning = hard negative)
        top_indices = np.argsort(similarities)[::-1]
        
        # Filter out excluded indices
        if exclude_indices:
            top_indices = [idx for idx in top_indices if idx not in exclude_indices]
        
        return top_indices[:n].tolist()
    
    def mine_hard_negatives_for_triplets(self, triplets: List[Tuple[str, str, str]],
                                        method: str = 'semantic',
                                        n_hard_negatives: int = 1) -> List[Tuple[str, str, str, str]]:
        """
        Add hard negatives to existing triplets.
        
        Args:
            triplets: List of (anchor, positive, negative) triplets
            method: 'bm25' or 'semantic'
            n_hard_negatives: Number of hard negatives to add per triplet
            
        Returns:
            List of (anchor, positive, negative, hard_negative) tuples
        """
        if method == 'bm25' and self.bm25 is None:
            raise ValueError("Must build BM25 index first")
        if method == 'semantic' and self.embeddings is None and self.model is None:
            raise ValueError("Must build semantic index or provide model")
        
        enhanced_triplets = []
        
        for anchor, positive, negative in triplets:
            # Find hard negatives for anchor
            if method == 'bm25':
                hard_neg_indices = self.mine_bm25_hard_negatives(anchor, n=n_hard_negatives)
                hard_negatives = [self.corpus[idx] for idx in hard_neg_indices]
            else:  # semantic
                hard_neg_indices = self.mine_semantic_hard_negatives(anchor, n=n_hard_negatives)
                hard_negatives = [self.corpus[idx] for idx in hard_neg_indices]
            
            # Add each hard negative as a separate triplet
            for hard_neg in hard_negatives:
                enhanced_triplets.append((anchor, positive, negative, hard_neg))
        
        return enhanced_triplets


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_loader import load_news_dataset, preprocess_dataset, get_text_and_labels
    from sentence_transformers import SentenceTransformer
    
    # Test hard negative mining
    dataset = load_news_dataset("ag_news")
    dataset = preprocess_dataset(dataset)
    
    texts, labels = get_text_and_labels(dataset['train'], max_samples=1000)
    
    # Test BM25
    miner = HardNegativeMiner()
    miner.build_bm25_index(texts)
    
    query = "breaking news in politics"
    hard_negs = miner.mine_bm25_hard_negatives(query, n=5)
    print(f"\nBM25 Hard negatives for '{query}':")
    for idx in hard_negs:
        print(f"  [{idx}]: {texts[idx][:80]}...")
    
    # Test semantic
    model = SentenceTransformer('all-MiniLM-L6-v2')
    miner.model = model
    miner.build_semantic_index(texts, model)
    
    hard_negs_sem = miner.mine_semantic_hard_negatives(query, n=5)
    print(f"\nSemantic Hard negatives for '{query}':")
    for idx in hard_negs_sem:
        print(f"  [{idx}]: {texts[idx][:80]}...")

