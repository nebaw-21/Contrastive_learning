"""
Dataset loading and preprocessing module
"""
from datasets import load_dataset
from typing import Dict, List, Any
import pandas as pd


def load_news_dataset(dataset_name: str = "ag_news") -> Dict[str, Any]:
    """
    Load a news dataset from HuggingFace datasets.
    
    Args:
        dataset_name: Name of the dataset (default: "ag_news")
        
    Returns:
        Dictionary containing train and test splits
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    print(f"Dataset loaded. Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")
    return dataset


def preprocess_text(text: str, lowercase: bool = True, remove_stopwords: bool = False) -> str:
    """
    Preprocess text data.
    
    Args:
        text: Input text
        lowercase: Whether to convert to lowercase
        remove_stopwords: Whether to remove stopwords (optional, not implemented by default)
        
    Returns:
        Preprocessed text
    """
    if lowercase:
        text = text.lower().strip()
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    return text


def preprocess_dataset(dataset: Dict[str, Any], lowercase: bool = True) -> Dict[str, Any]:
    """
    Preprocess entire dataset.
    
    Args:
        dataset: Dataset dictionary
        lowercase: Whether to convert to lowercase
        
    Returns:
        Preprocessed dataset
    """
    print("Preprocessing dataset...")
    
    def preprocess_batch(examples):
        if 'text' in examples:
            examples['text'] = [preprocess_text(t, lowercase=lowercase) for t in examples['text']]
        elif 'description' in examples:
            examples['text'] = [preprocess_text(t, lowercase=lowercase) for t in examples['description']]
        return examples
    
    dataset = dataset.map(preprocess_batch, batched=True)
    print("Preprocessing complete.")
    return dataset


def get_text_and_labels(dataset_split, max_samples: int = None):
    """
    Extract texts and labels from dataset split.
    
    Args:
        dataset_split: Dataset split (train/test)
        max_samples: Maximum number of samples to extract (None for all)
        
    Returns:
        Tuple of (texts, labels)
    """
    texts = []
    labels = []
    
    if max_samples:
        dataset_split = dataset_split.select(range(min(max_samples, len(dataset_split))))
    
    for item in dataset_split:
        if 'text' in item:
            texts.append(item['text'])
        elif 'description' in item:
            texts.append(item['description'])
        else:
            continue
            
        if 'label' in item:
            labels.append(item['label'])
        else:
            labels.append(None)
    
    return texts, labels


if __name__ == "__main__":
    # Test the data loader
    dataset = load_news_dataset("ag_news")
    dataset = preprocess_dataset(dataset)
    
    print("\nSample from train set:")
    print(dataset['train'][0])

