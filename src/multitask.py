"""
Multi-task learning extension: contrastive learning + topic classification
"""
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import InputExample, losses


class MultiTaskDataset(Dataset):
    """Dataset for multi-task learning."""
    
    def __init__(self, triplets: List[Tuple[str, str, str]], labels: List[int]):
        """
        Initialize dataset.
        
        Args:
            triplets: List of (anchor, positive, negative) triplets
            labels: List of labels for anchors
        """
        self.triplets = triplets
        self.labels = labels
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]
        label = self.labels[idx]
        return {
            'anchor': anchor,
            'positive': positive,
            'negative': negative,
            'label': label
        }


class MultiTaskModel(nn.Module):
    """Multi-task model combining contrastive learning and classification."""
    
    def __init__(self, base_model: SentenceTransformer, num_classes: int):
        """
        Initialize multi-task model.
        
        Args:
            base_model: Base SentenceTransformer model
            num_classes: Number of topic classes
        """
        super().__init__()
        self.base_model = base_model
        embedding_dim = base_model.get_sentence_embedding_dimension()
        self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, texts: List[str]):
        """
        Forward pass.
        
        Args:
            texts: List of input texts
            
        Returns:
            Tuple of (embeddings, logits)
        """
        embeddings = self.base_model.encode(texts, convert_to_tensor=True)
        logits = self.classifier(embeddings)
        return embeddings, logits
    
    def encode(self, texts: List[str], show_progress: bool = True):
        """Encode texts to embeddings."""
        return self.base_model.encode(texts, show_progress_bar=show_progress)


class MultiTaskTrainer:
    """Trainer for multi-task learning."""
    
    def __init__(self, base_model_name: str = 'all-MiniLM-L6-v2', num_classes: int = 4):
        """
        Initialize multi-task trainer.
        
        Args:
            base_model_name: Name of base SentenceTransformer model
            num_classes: Number of topic classes
        """
        print(f"Initializing multi-task model with {num_classes} classes")
        base_model = SentenceTransformer(base_model_name)
        self.model = MultiTaskModel(base_model, num_classes)
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train(self, train_dataloader: DataLoader,
              num_epochs: int = 2,
              contrastive_weight: float = 0.7,
              classification_weight: float = 0.3,
              learning_rate: float = 2e-5,
              output_path: str = 'models/multitask_model'):
        """
        Train multi-task model.
        
        Args:
            train_dataloader: DataLoader with training examples
            num_epochs: Number of training epochs
            contrastive_weight: Weight for contrastive loss
            classification_weight: Weight for classification loss
            learning_rate: Learning rate
            output_path: Path to save model
        """
        print(f"\nStarting multi-task training...")
        print(f"Contrastive weight: {contrastive_weight}, Classification weight: {classification_weight}")
        
        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Loss functions
        contrastive_loss_fn = nn.TripletMarginLoss(margin=1.0)
        classification_loss_fn = nn.CrossEntropyLoss()
        
        self.model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            total_contrastive_loss = 0
            total_classification_loss = 0
            
            for batch in train_dataloader:
                anchors = batch['anchor']
                positives = batch['positive']
                negatives = batch['negative']
                labels = torch.tensor(batch['label']).to(self.device)
                
                # Get embeddings
                anchor_emb = self.model.base_model.encode(anchors, convert_to_tensor=True)
                positive_emb = self.model.base_model.encode(positives, convert_to_tensor=True)
                negative_emb = self.model.base_model.encode(negatives, convert_to_tensor=True)
                
                # Contrastive loss (triplet loss)
                contrastive_loss = contrastive_loss_fn(anchor_emb, positive_emb, negative_emb)
                
                # Classification loss
                logits = self.model.classifier(anchor_emb)
                classification_loss = classification_loss_fn(logits, labels)
                
                # Combined loss
                loss = contrastive_weight * contrastive_loss + classification_weight * classification_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_contrastive_loss += contrastive_loss.item()
                total_classification_loss += classification_loss.item()
            
            avg_loss = total_loss / len(train_dataloader)
            avg_contrastive = total_contrastive_loss / len(train_dataloader)
            avg_classification = total_classification_loss / len(train_dataloader)
            
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Total Loss: {avg_loss:.4f}")
            print(f"  Contrastive Loss: {avg_contrastive:.4f}")
            print(f"  Classification Loss: {avg_classification:.4f}")
        
        # Save model
        torch.save(self.model.state_dict(), f"{output_path}_state_dict.pt")
        self.model.base_model.save(output_path)
        print(f"\nModel saved to {output_path}")


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_loader import load_news_dataset, preprocess_dataset
    from src.triplets import create_triplets_from_dataset
    
    # Test multi-task training
    dataset = load_news_dataset("ag_news")
    dataset = preprocess_dataset(dataset)
    
    # Create triplets with labels
    triplets = create_triplets_from_dataset(dataset['train'], max_triplets=1000)
    
    # Get labels for anchors (assuming first element in each triplet is anchor)
    # In practice, you'd need to track labels during triplet creation
    labels = [0] * len(triplets)  # Placeholder - would need actual labels
    
    dataset_obj = MultiTaskDataset(triplets, labels)
    dataloader = DataLoader(dataset_obj, batch_size=16, shuffle=True)
    
    trainer = MultiTaskTrainer(num_classes=4)
    trainer.train(dataloader, num_epochs=2)

