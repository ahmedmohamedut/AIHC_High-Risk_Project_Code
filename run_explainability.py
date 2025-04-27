import os
import argparse
import pandas as pd
import torch
from model import BrainTumorCNN
from utils import load_model
from explainability import ExplainabilityAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Explainability Visualizations for Misclassified Images')
    parser.add_argument('--misclassified_csv', type=str, required=True, help='Path to CSV file with misclassified images')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='explainability_results', help='Directory to save visualization results')
    parser.add_argument('--methods', type=str, default='gradcam++,eigencam,ablationcam', help='Comma-separated list of explainability methods to use')
    parser.add_argument('--image_size', type=int, default=224, help='Image size for model input')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of images to process (for debugging)')
    parser.add_argument('--target_layer', type=str, default=None, help='Name of the target layer to use for explainability (default: auto-detect)')
    return parser.parse_args()


def get_target_layer(model, layer_name):
    """
    Get a specific layer from the model by name.

    Args:
        model: PyTorch model
        layer_name: Name of the layer to get

    Returns:
        The requested layer or None if not found
    """
    if layer_name is None:
        return None

    # Try to get the layer directly from the model
    if hasattr(model, layer_name):
        return getattr(model, layer_name)

    # Otherwise, search through named modules
    for name, module in model.named_modules():
        if name == layer_name:
            return module

    print(f"WARNING: Layer '{layer_name}' not found in model. Available layers:")
    for name, _ in model.named_modules():
        if name:  # Skip empty string (the model itself)
            print(f" - {name}")

    return None


def main():
    # Parse command line arguments
    args = parse_args()

    # Check if the CSV file exists
    if not os.path.exists(args.misclassified_csv):
        print(f"Error: CSV file {args.misclassified_csv} not found.")
        return

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

    # Initialize model
    model = BrainTumorCNN(num_classes=len(LABEL_MAP)).to(device)

    # Load model weights
    model, checkpoint = load_model(model, args.checkpoint_path, device=device)
    model.eval()  # Set model to evaluation mode

    print(f"Loaded model from checkpoint: {args.checkpoint_path}")
    print(f"Model was trained for {checkpoint['epoch'] + 1} epochs")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}, Validation accuracy: {checkpoint['val_acc']:.2f}%")

    # Load misclassified images data
    misclassified_df = pd.read_csv(args.misclassified_csv)
    print(f"Loaded {len(misclassified_df)} misclassified images from {args.misclassified_csv}")

    # Limit number of images if specified
    if args.limit and args.limit < len(misclassified_df):
        misclassified_df = misclassified_df.head(args.limit)
        print(f"Limited to first {args.limit} images")

    # Get target layer if specified
    target_layer = get_target_layer(model, args.target_layer)

    # Create explainability analyzer
    analyzer = ExplainabilityAnalyzer(model, device, target_layer)

    # Parse methods list
    methods = [m.strip() for m in args.methods.split(',')]

    # Check for valid methods
    valid_methods = ["gradcam++", "eigencam", "ablationcam"]
    for method in methods:
        if method.lower() not in valid_methods:
            print(f"Warning: Method '{method}' is not recognized. Valid methods are: {', '.join(valid_methods)}")

    # Filter to only valid methods
    methods = [m for m in methods if m.lower() in valid_methods]

    if not methods:
        print("No valid explainability methods specified. Exiting.")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Apply each explainability method
    for method in methods:
        print(f"\nApplying {method} explainability method...")
        analyzer.explain_batch(
            misclassified_df=misclassified_df,
            method=method,
            output_dir=args.output_dir,
            image_size=(args.image_size, args.image_size),
            label_map=LABEL_MAP
        )

    print(f"\nExplainability analysis complete. Results saved to {args.output_dir}")
    print("\nVisualization guide:")
    print("  - Left: Original image")
    print("  - Middle: CAM overlay showing areas the model focused on")
    print("  - Right: Raw activation map")


if __name__ == "__main__":
    main()