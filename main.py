"""
Main training script for News Article Semantic Similarity using Contrastive Learning
"""
import argparse
import os
import torch

# Set PyTorch CUDA memory allocation configuration to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from src.data_loader import load_news_dataset, preprocess_dataset
from src.triplets import create_triplets_from_dataset, split_triplets
from src.baseline import BaselineEvaluator
from src.training import ContrastiveTrainer
from src.evaluation import Evaluator
from src.hard_negatives import HardNegativeMiner
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser(description="Train contrastive learning model for news article similarity")
    parser.add_argument("--dataset", type=str, default="ag_news", help="Dataset name")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Base model name")
    parser.add_argument("--max_triplets", type=int, default=10000, help="Maximum number of triplets")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (auto-adjusted for GPU memory if not specified)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--loss", type=str, default="triplet", choices=["triplet", "infonce", "cosine"],
                       help="Loss function type")
    parser.add_argument("--temperature", type=float, default=0.05, help="Temperature for InfoNCE")
    parser.add_argument("--output_dir", type=str, default="models/news_contrastive_model", 
                       help="Output directory for model")
    parser.add_argument("--use_hard_negatives", action="store_true", help="Use hard negative mining")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip baseline evaluation")
    parser.add_argument("--skip_training", action="store_true", help="Skip training (evaluation only)")
    parser.add_argument("--baseline_samples", type=int, default=15000, help="Number of samples for baseline evaluation")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    
    print("=" * 60)
    print("News Article Semantic Similarity - Contrastive Learning")
    print("=" * 60)
    
    # Check GPU availability and set memory limit
    if torch.cuda.is_available():
        print(f"\n[GPU] GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"[GPU] CUDA Version: {torch.version.cuda}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] Total GPU Memory: {total_memory:.2f} GB")
        
        # Set GPU memory allocation limit to 2GB
        max_memory_gb = 2.0
        torch.cuda.set_per_process_memory_fraction(max_memory_gb / total_memory, 0)
        torch.cuda.empty_cache()  # Clear cache
        print(f"[GPU] Memory allocation limit set to: {max_memory_gb} GB")
        print(f"[GPU] Using expandable memory segments to reduce fragmentation")
        
        device = 'cuda'
        
        # Auto-adjust batch size for small GPUs
        if args.batch_size is None:
            if total_memory < 3.0:  # Less than 3GB GPU (like MX450 with 2GB)
                args.batch_size = 8
                print(f"[GPU] Auto-adjusted batch size to {args.batch_size} for small GPU ({total_memory:.2f} GB)")
            elif total_memory < 6.0:  # Less than 6GB GPU
                args.batch_size = 16
                print(f"[GPU] Auto-adjusted batch size to {args.batch_size} for medium GPU ({total_memory:.2f} GB)")
            else:
                args.batch_size = 32
                print(f"[GPU] Using default batch size: {args.batch_size}")
    else:
        print("\n[WARNING] No GPU detected - training will use CPU (slower)")
        device = 'cpu'
        if args.batch_size is None:
            args.batch_size = 32
    
    # 1. Load and preprocess dataset
    print("\n[1/6] Loading dataset...")
    dataset = load_news_dataset(args.dataset)
    dataset = preprocess_dataset(dataset)
    
    # 2. Baseline evaluation
    if not args.skip_baseline:
        print("\n[2/6] Baseline evaluation...")
        from src.data_loader import get_text_and_labels
        
        # Limit to test set size if requested samples exceed it
        max_test_samples = len(dataset['test'])
        baseline_samples = min(args.baseline_samples, max_test_samples)
        print(f"Using {baseline_samples} samples for baseline evaluation (test set has {max_test_samples} samples)")
        
        test_texts, test_labels = get_text_and_labels(dataset['test'], max_samples=baseline_samples)
        baseline_evaluator = BaselineEvaluator(model_name=args.model)
        baseline_embeddings = baseline_evaluator.encode(test_texts)
        
        # For visualization, use a subset to avoid memory issues
        viz_samples = min(2000, len(test_texts))
        print(f"Visualizing {viz_samples} samples (subset for memory efficiency)")
        baseline_evaluator.visualize_embeddings(
            labels=test_labels[:viz_samples] if len(test_labels) > viz_samples else test_labels,
            n_samples=viz_samples,
            save_path="baseline_embeddings.png",
            title="Baseline Embeddings (Pre-trained)"
        )
        print("Baseline evaluation complete.")
    else:
        baseline_embeddings = None
        test_texts, test_labels = None, None
    
    if args.skip_training:
        print("\nSkipping training (--skip_training flag set)")
        return
    
    # 3. Create triplets
    print("\n[3/6] Creating triplets...")
    triplets = create_triplets_from_dataset(dataset['train'], max_triplets=args.max_triplets)
    train_triplets, val_triplets = split_triplets(triplets, train_ratio=0.9)
    print(f"Created {len(triplets)} triplets ({len(train_triplets)} train, {len(val_triplets)} val)")
    
    # 4. Hard negative mining (optional)
    if args.use_hard_negatives:
        print("\n[4/6] Hard negative mining...")
        from src.data_loader import get_text_and_labels
        train_texts, _ = get_text_and_labels(dataset['train'], max_samples=5000)
        
        miner = HardNegativeMiner()
        miner.build_bm25_index(train_texts)
        print("Hard negative mining complete.")
    else:
        print("\n[4/6] Skipping hard negative mining")
    
    # 5. Training
    print("\n[5/6] Training model...")
    trainer = ContrastiveTrainer(base_model_name=args.model, device=device)
    train_dataloader = trainer.prepare_dataloader(
        train_triplets,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    if args.loss == "infonce":
        trainer.train_with_infonce(
            train_dataloader,
            num_epochs=args.epochs,
            temperature=args.temperature,
            output_path=args.output_dir,
            save_model=True
        )
    else:
        trainer.train(
            train_dataloader,
            loss_type=args.loss,
            num_epochs=args.epochs,
            output_path=args.output_dir,
            save_model=True
        )
    
    # 6. Evaluation
    print("\n[6/6] Evaluating fine-tuned model...")
    if test_texts and test_labels:
        evaluator = Evaluator(trainer.model, test_texts, test_labels)
        finetuned_results = evaluator.evaluate_all(k_values=[1, 5, 10])
        
        evaluator.visualize_embeddings(
            labels=test_labels,
            save_path="finetuned_embeddings.png",
            title="Fine-tuned Embeddings"
        )
        
        # Compare with baseline
        if baseline_embeddings is not None:
            print("\nComparing with baseline...")
            comparison = evaluator.compare_with_baseline(baseline_embeddings, k_values=[1, 5, 10])
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

