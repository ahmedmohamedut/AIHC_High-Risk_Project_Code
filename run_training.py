import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from datetime import datetime

# Import our modules
from model import BrainTumorCNN
from dataset import get_data_loaders
from trainer import train_model, evaluate_model
from utils import set_seed, log_model_graph, plot_confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser(description='Brain Tumor Classification')
    parser.add_argument('--data_dir', type=str, default='brain-tumor-mri-dataset', help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0007, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--image_size', type=int, default=224, help='Image size for model input')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)

    # Set default to float32
    torch.set_default_tensor_type(torch.FloatTensor)

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
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        base_directory=args.data_dir,
        label_map=LABEL_MAP,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        random_state=args.seed,
        num_workers=args.num_workers
    )
    
    # Initialize model
    model = BrainTumorCNN(num_classes=len(LABEL_MAP)).to(device)
    
    # Log model architecture to TensorBoard
    log_model_graph(model, input_size=(1, 3, args.image_size, args.image_size), device=device)
    
    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3
    )
    
    # Create unique run name based on timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"brain_tumor_cnn_{timestamp}"
    checkpoint_dir = f"{args.checkpoint_dir}/{run_name}"
    
    # Train the model
    print(f"\n{'='*20} Training Started {'='*20}")
    print(f"Run name: {run_name}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"{'='*54}\n")
    
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        checkpoint_dir=checkpoint_dir,
        patience=args.patience,
        label_map=LABEL_MAP
    )
    
    # Evaluate the model on test set
    test_loss, test_acc, confusion_mat = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        label_map=LABEL_MAP
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        cm=confusion_mat,
        class_names=list(LABEL_MAP.keys()),
        save_path=f"{checkpoint_dir}/confusion_matrix.png"
    )
    
    print(f"\n{'='*20} Training Complete {'='*20}")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Model checkpoints saved to: {checkpoint_dir}")
    print(f"View TensorBoard logs with: tensorboard --logdir=runs")
    print(f"{'='*54}\n")

if __name__ == "__main__":
    main()
