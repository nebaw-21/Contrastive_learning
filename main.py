"""
Main training script for News Article Semantic Similarity using Contrastive Learning
"""
import argparse
from src.data_loader import load_news_dataset, preprocess_dataset
from src.triplets import create_triplets_from_dataset, split_triplets
from src.baseline import BaselineEvaluator
from src.training import ContrastiveTrainer
from src.evaluation import Evaluator
from src.hard_negatives import HardNegativeMiner
from sentence_transformers import SentenceTransformer
import os


def main():
    parser = argparse.ArgumentParser(description="Train contrastive learning model for news article similarity")
    parser.add_argument("--dataset", type=str, default="ag_news", help="Dataset name")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Base model name")
    parser.add_argument("--max_triplets", type=int, default=10000, help="Maximum number of triplets")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--loss", type=str, default="triplet", choices=["triplet", "infonce", "cosine"],
                       help="Loss function type")
    parser.add_argument("--temperature", type=float, default=0.05, help="Temperature for InfoNCE")
    parser.add_argument("--output_dir", type=str, default="models/news_contrastive_model", 
                       help="Output directory for model")
    parser.add_argument("--use_hard_negatives", action="store_true", help="Use hard negative mining")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip baseline evaluation")
    parser.add_argument("--skip_training", action="store_true", help="Skip training (evaluation only)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    
    print("=" * 60)
    print("News Article Semantic Similarity - Contrastive Learning")
    print("=" * 60)
    
    # 1. Load and preprocess dataset
    print("\n[1/6] Loading dataset...")
    dataset = load_news_dataset(args.dataset)
    dataset = preprocess_dataset(dataset)
    
    # 2. Baseline evaluation
    if not args.skip_baseline:
        print("\n[2/6] Baseline evaluation...")
        from src.data_loader import get_text_and_labels
        
        test_texts, test_labels = get_text_and_labels(dataset['test'], max_samples=1000)
        baseline_evaluator = BaselineEvaluator(model_name=args.model)
        baseline_embeddings = baseline_evaluator.encode(test_texts)
        baseline_evaluator.visualize_embeddings(
            labels=test_labels,
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
    trainer = ContrastiveTrainer(base_model_name=args.model)
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

