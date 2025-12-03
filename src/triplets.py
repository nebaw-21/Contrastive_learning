"""
Triplet generation for contrastive learning
"""
import random
from typing import List, Tuple, Dict, Any
from collections import defaultdict


def create_triplets_from_dataset(dataset_split, max_triplets: int = None) -> List[Tuple[str, str, str]]:
    """
    Create anchor-positive-negative triplets from dataset.
    
    Args:
        dataset_split: Dataset split (must have 'text' and 'label' fields)
        max_triplets: Maximum number of triplets to generate (None for all)
        
    Returns:
        List of (anchor, positive, negative) triplets
    """
    print("Creating triplets...")
    
    # Group articles by topic/label
    topic_to_articles = defaultdict(list)
    articles = []
    
    for item in dataset_split:
        text = item.get('text') or item.get('description', '')
        label = item.get('label')
        
        if text and label is not None:
            topic_to_articles[label].append(text)
            articles.append((text, label))
    
    print(f"Found {len(topic_to_articles)} topics")
    print(f"Total articles: {len(articles)}")
    
    # Generate triplets
    triplets = []
    available_labels = list(topic_to_articles.keys())
    
    for anchor_text, anchor_label in articles:
        # Find positive (same topic, different article)
        positives = [t for t in topic_to_articles[anchor_label] if t != anchor_text]
        if not positives:
            continue
        
        positive = random.choice(positives)
        
        # Find negative (different topic)
        negative_labels = [l for l in available_labels if l != anchor_label]
        if not negative_labels:
            continue
        
        negative_label = random.choice(negative_labels)
        negative = random.choice(topic_to_articles[negative_label])
        
        triplets.append((anchor_text, positive, negative))
        
        if max_triplets and len(triplets) >= max_triplets:
            break
    
    print(f"Generated {len(triplets)} triplets")
    return triplets


def create_triplets_from_texts_and_labels(texts: List[str], labels: List[int], 
                                         max_triplets: int = None) -> List[Tuple[str, str, str]]:
    """
    Create triplets from separate lists of texts and labels.
    
    Args:
        texts: List of article texts
        labels: List of corresponding labels
        max_triplets: Maximum number of triplets to generate
        
    Returns:
        List of (anchor, positive, negative) triplets
    """
    # Group by label
    topic_to_articles = defaultdict(list)
    for text, label in zip(texts, labels):
        topic_to_articles[label].append(text)
    
    # Generate triplets
    triplets = []
    available_labels = list(topic_to_articles.keys())
    
    for i, (anchor_text, anchor_label) in enumerate(zip(texts, labels)):
        # Positive: same label, different text
        positives = [t for t in topic_to_articles[anchor_label] if t != anchor_text]
        if not positives:
            continue
        
        positive = random.choice(positives)
        
        # Negative: different label
        negative_labels = [l for l in available_labels if l != anchor_label]
        if not negative_labels:
            continue
        
        negative_label = random.choice(negative_labels)
        negative = random.choice(topic_to_articles[negative_label])
        
        triplets.append((anchor_text, positive, negative))
        
        if max_triplets and len(triplets) >= max_triplets:
            break
    
    return triplets


def split_triplets(triplets: List[Tuple[str, str, str]], train_ratio: float = 0.8) -> Tuple[List, List]:
    """
    Split triplets into train and validation sets.
    
    Args:
        triplets: List of triplets
        train_ratio: Ratio of training data
        
    Returns:
        Tuple of (train_triplets, val_triplets)
    """
    random.shuffle(triplets)
    split_idx = int(len(triplets) * train_ratio)
    return triplets[:split_idx], triplets[split_idx:]


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_loader import load_news_dataset, preprocess_dataset
    
    # Test triplet generation
    dataset = load_news_dataset("ag_news")
    dataset = preprocess_dataset(dataset)
    
    triplets = create_triplets_from_dataset(dataset['train'], max_triplets=100)
    print(f"\nSample triplet:")
    print(f"Anchor: {triplets[0][0][:100]}...")
    print(f"Positive: {triplets[0][1][:100]}...")
    print(f"Negative: {triplets[0][2][:100]}...")

