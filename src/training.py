"""
Contrastive learning training module
"""
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional
import os
import torch

# Set PyTorch CUDA memory allocation configuration to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class ContrastiveTrainer:
    """Train a contrastive learning model for semantic similarity."""
    
    def __init__(self, base_model_name: str = 'all-MiniLM-L6-v2', device: Optional[str] = None):
        """
        Initialize trainer.
        
        Args:
            base_model_name: Name of pre-trained SentenceTransformer model
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"[GPU] GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"[GPU] CUDA version: {torch.version.cuda}")
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"[GPU] Total GPU memory: {total_memory:.2f} GB")
                
                # Set GPU memory allocation limit to 2GB
                max_memory_gb = 2.0
                torch.cuda.set_per_process_memory_fraction(max_memory_gb / total_memory, 0)
                torch.cuda.empty_cache()  # Clear cache before setting limit
                print(f"[GPU] Memory allocation limit set to: {max_memory_gb} GB")
            else:
                device = 'cpu'
                print("[WARNING] GPU not available, using CPU")
        else:
            if device == 'cuda' and not torch.cuda.is_available():
                print("[WARNING] CUDA requested but not available, falling back to CPU")
                device = 'cpu'
            elif device == 'cuda':
                # Set GPU memory allocation limit to 2GB
                max_memory_gb = 2.0
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                torch.cuda.set_per_process_memory_fraction(max_memory_gb / total_memory, 0)
                torch.cuda.empty_cache()
                print(f"[GPU] Memory allocation limit set to: {max_memory_gb} GB")
        
        self.device = device
        print(f"Initializing model: {base_model_name} on {device.upper()}")
        self.model = SentenceTransformer(base_model_name, device=device)
        self.base_model_name = base_model_name
    
    def prepare_dataloader(self, triplets: List[Tuple[str, str, str]], 
                          batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """
        Prepare DataLoader from triplets.
        
        Args:
            triplets: List of (anchor, positive, negative) triplets
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader
        """
        # Convert triplets to InputExample format for TripletLoss
        train_examples = [
            InputExample(texts=[anchor, positive, negative])
            for anchor, positive, negative in triplets
        ]
        
        train_dataloader = DataLoader(train_examples, shuffle=shuffle, batch_size=batch_size)
        return train_dataloader
    
    def train(self, train_dataloader: DataLoader,
              loss_type: str = 'triplet',
              num_epochs: int = 2,
              warmup_steps: Optional[int] = None,
              output_path: str = 'models/news_contrastive_model',
              save_model: bool = True):
        """
        Train the model with contrastive learning.
        
        Args:
            train_dataloader: DataLoader with training examples
            loss_type: Type of loss ('triplet', 'cosine', 'contrastive')
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps (auto-calculated if None)
            output_path: Path to save the trained model
            save_model: Whether to save the model after training
        """
        print(f"\nStarting training with {loss_type} loss...")
        print(f"Number of epochs: {num_epochs}")
        print(f"Batch size: {train_dataloader.batch_size}")
        print(f"Device: {self.device.upper()}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        
        # Select loss function
        if loss_type == 'triplet':
            train_loss = losses.TripletLoss(model=self.model)
        elif loss_type == 'cosine':
            train_loss = losses.CosineSimilarityLoss(model=self.model)
        elif loss_type == 'contrastive':
            train_loss = losses.ContrastiveLoss(model=self.model)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Calculate warmup steps if not provided
        if warmup_steps is None:
            warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
        
        print(f"Warmup steps: {warmup_steps}")
        
        # Clear GPU cache before training
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all operations are complete
        
        # Train the model with error handling for OOM
        try:
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=num_epochs,
                warmup_steps=warmup_steps,
                output_path=output_path if save_model else None,
                show_progress_bar=True
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n[ERROR] CUDA out of memory!")
            print(f"[SOLUTION] Try reducing batch size. Current batch size: {train_dataloader.batch_size}")
            print(f"[SOLUTION] Suggested batch sizes: 4, 8, or 16")
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            raise
        
        # Clear GPU cache after training
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            final_allocated = torch.cuda.memory_allocated(0) / 1e9
            final_reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"\n[GPU] Final Memory - Allocated: {final_allocated:.2f} GB, Reserved: {final_reserved:.2f} GB")
        
        if save_model:
            print(f"\nModel saved to {output_path}")
    
    def train_with_infonce(self, train_dataloader: DataLoader,
                          num_epochs: int = 2,
                          temperature: float = 0.05,
                          warmup_steps: Optional[int] = None,
                          output_path: str = 'models/news_contrastive_model',
                          save_model: bool = True):
        """
        Train with InfoNCE loss (MultipleNegativesRankingLoss).
        
        Args:
            train_dataloader: DataLoader with training examples
            num_epochs: Number of training epochs
            temperature: Temperature parameter for InfoNCE
            warmup_steps: Number of warmup steps
            output_path: Path to save the trained model
            save_model: Whether to save the model after training
        """
        print(f"\nStarting training with InfoNCE loss (temperature={temperature})...")
        print(f"Number of epochs: {num_epochs}")
        print(f"Device: {self.device.upper()}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        # InfoNCE is implemented as MultipleNegativesRankingLoss in sentence-transformers
        train_loss = losses.MultipleNegativesRankingLoss(model=self.model, 
                                                         scale=1.0/temperature)
        
        if warmup_steps is None:
            warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
        
        print(f"Warmup steps: {warmup_steps}")
        
        # Clear GPU cache before training
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all operations are complete
        
        # Train with error handling for OOM
        try:
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=num_epochs,
                warmup_steps=warmup_steps,
                output_path=output_path if save_model else None,
                show_progress_bar=True
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n[ERROR] CUDA out of memory!")
            print(f"[SOLUTION] Try reducing batch size. Current batch size: {train_dataloader.batch_size}")
            print(f"[SOLUTION] Suggested batch sizes: 4, 8, or 16")
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            raise
        
        # Clear GPU cache after training
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            final_allocated = torch.cuda.memory_allocated(0) / 1e9
            final_reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"\n[GPU] Final Memory - Allocated: {final_allocated:.2f} GB, Reserved: {final_reserved:.2f} GB")
        
        if save_model:
            print(f"\nModel saved to {output_path}")
    
    def load_model(self, model_path: str):
        """Load a saved model."""
        print(f"Loading model from {model_path}")
        self.model = SentenceTransformer(model_path)
    
    def encode(self, texts: List[str], show_progress: bool = True):
        """Encode texts to embeddings."""
        return self.model.encode(texts, show_progress_bar=show_progress)


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_loader import load_news_dataset, preprocess_dataset
    from src.triplets import create_triplets_from_dataset
    
    # Test training
    print("Loading dataset...")
    dataset = load_news_dataset("ag_news")
    dataset = preprocess_dataset(dataset)
    
    print("Creating triplets...")
    triplets = create_triplets_from_dataset(dataset['train'], max_triplets=2000)
    
    print("Initializing trainer...")
    trainer = ContrastiveTrainer()
    
    print("Preparing dataloader...")
    train_dataloader = trainer.prepare_dataloader(triplets, batch_size=32)
    
    print("Training model...")
    trainer.train(train_dataloader, loss_type='triplet', num_epochs=2)

