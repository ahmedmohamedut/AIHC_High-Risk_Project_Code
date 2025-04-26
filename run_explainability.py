import torch
import argparse
import os

from model import BrainTumorCNN
from dataset import get_data_loaders
from explainability import analyze_misclassified
from utils import set_seed, load_model

def parse_args():
    parser = argparse.ArgumentParser(description='Brain Tumor Classification Explainability')
    parser.add_argument('--data_dir', type=str, default='brain-tumor-mri-dataset',
                        help='Directory containing the dataset')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_samples', type=int, default=10, 
                        help='Number of misclassified samples to analyze')
    parser.add_argument('--save_dir', type=str, default='explanations',
                        help='Directory to save explanation visualizations')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define label mapping
    LABEL_MAP = {
        'notumor': 0,
        'glioma': 1,
        'meningioma': 2,
        'pituitary': 3
    }
    # Class names in the correct order
    class_names = list(LABEL_MAP.keys())
    
    # Get test data loader (we only need the test loader for explainability analysis)
    _, _, test_loader = get_data_loaders(
        base_directory=args.data_dir,
        label_map=LABEL_MAP,
        batch_size=args.batch_size,
        random_state=args.seed,
        num_workers=args.num_workers
    )
    
    # Initialize model
    model = BrainTumorCNN(num_classes=len(LABEL_MAP)).to(device)
    
    # Load model checkpoint
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    model, checkpoint = load_model(model, args.checkpoint_path, device=device)
    print(f"Model loaded (epoch {checkpoint['epoch']}, val_acc: {checkpoint['val_acc']:.2f}%)")
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Run explainability analysis on misclassified samples
    print(f"\n{'='*20} Running Explainability Analysis {'='*20}")
    analyze_misclassified(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        num_samples=args.num_samples,
        save_dir=args.save_dir
    )
    
    print(f"\n{'='*20} Analysis Complete {'='*20}")
    print(f"Explanations saved to: {args.save_dir}")

if __name__ == "__main__":
    main()
