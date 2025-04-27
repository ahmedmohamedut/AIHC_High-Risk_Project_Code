import torch
import pandas as pd
import argparse
import os
from model import BrainTumorCNN
from dataset import get_data_loaders
from utils import load_model

def parse_args():
    parser = argparse.ArgumentParser(description='Generate Brain Tumor Misclassification Report')
    parser.add_argument('--data_dir', type=str, default='brain-tumor-mri-dataset', help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_csv', type=str, default='misclassified_images.csv', help='Output CSV file path')
    parser.add_argument('--image_size', type=int, default=224, help='Image size for model input')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    return parser.parse_args()

def generate_misclassification_report(model, test_loader, device, label_map, output_path):
    """
    Generate a report of misclassified images during testing.
    
    Args:
        model: The neural network model
        test_loader: DataLoader for test data
        device: Device to evaluate on (cuda/cpu)
        label_map: Dictionary mapping class indices to class names
        output_path: Path to save the CSV report
    """
    # Create inverse label map (index to class name)
    inv_label_map = {v: k for k, v in label_map.items()}
    
    model.eval()
    misclassified = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Find misclassified samples in this batch
            for i, (true, pred) in enumerate(zip(labels, predicted)):
                if true.item() != pred.item():
                    # Calculate the global index in the dataset
                    global_idx = batch_idx * test_loader.batch_size + i
                    
                    # Ensure we don't go out of bounds
                    if global_idx < len(test_loader.dataset):
                        # Get the original file path from test_loader's dataset
                        file_path = test_loader.dataset.dataframe.iloc[global_idx]['file_path']
                        true_label = true.item()
                        pred_label = pred.item()
                        
                        misclassified.append({
                            'file_path': file_path,
                            'true_class': inv_label_map[true_label],
                            'predicted_class': inv_label_map[pred_label]
                        })
    
    # Create DataFrame and save to CSV
    if misclassified:
        df = pd.DataFrame(misclassified)
        df.to_csv(output_path, index=False)
        print(f"Misclassification report saved to {output_path}")
        print(f"Total misclassified images: {len(misclassified)}")
        
        # Calculate per-class misclassification statistics
        print("\nMisclassification statistics by class:")
        for class_name in label_map.keys():
            class_total = len([m for m in misclassified if m['true_class'] == class_name])
            if class_total > 0:
                # Count where this class was misclassified as other classes
                misclass_counts = {}
                for m in misclassified:
                    if m['true_class'] == class_name:
                        pred = m['predicted_class']
                        if pred not in misclass_counts:
                            misclass_counts[pred] = 0
                        misclass_counts[pred] += 1
                
                print(f"  {class_name}: {class_total} misclassified as:")
                for pred_class, count in misclass_counts.items():
                    print(f"    - {pred_class}: {count} ({count/class_total*100:.1f}%)")
    else:
        print("No misclassified images found.")

def main():
    # Parse command line arguments
    args = parse_args()
    
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
    
    # Get only test loader - we don't need train or val for this task
    _, _, test_loader = get_data_loaders(
        base_directory=args.data_dir,
        label_map=LABEL_MAP,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        num_workers=args.num_workers
    )
    
    # Initialize and load model
    model = BrainTumorCNN(num_classes=len(LABEL_MAP)).to(device)
    model, checkpoint = load_model(model, args.checkpoint_path, device=device)
    
    print(f"Loaded model from checkpoint: {args.checkpoint_path}")
    print(f"Model was trained for {checkpoint['epoch'] + 1} epochs")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}, Validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Generate misclassification report
    generate_misclassification_report(
        model=model,
        test_loader=test_loader,
        device=device,
        label_map=LABEL_MAP,
        output_path=args.output_csv
    )

if __name__ == "__main__":
    main()
