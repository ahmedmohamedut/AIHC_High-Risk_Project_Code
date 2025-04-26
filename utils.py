import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot a confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def log_model_graph(model, input_size=(1, 3, 224, 224), device='cuda'):
    """
    Log model architecture to TensorBoard.
    
    Args:
        model: The neural network model
        input_size: Input tensor size
        device: Device to run on
    """
    writer = SummaryWriter('runs/model_architecture')
    dummy_input = torch.randn(input_size).to(device)
    writer.add_graph(model, dummy_input)
    writer.close()
    print("Model architecture logged to TensorBoard")

def save_model(model, optimizer, epoch, val_loss, val_acc, path):
    """
    Save model checkpoint.
    
    Args:
        model: The neural network model
        optimizer: The optimizer
        epoch: Current epoch
        val_loss: Validation loss
        val_acc: Validation accuracy
        path: Path to save the checkpoint
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc
    }, path)
    print(f"Model saved to {path}")

def load_model(model, path, optimizer=None, device='cuda'):
    """
    Load model checkpoint.
    
    Args:
        model: The neural network model
        path: Path to the checkpoint
        optimizer: The optimizer (optional)
        device: Device to load to
    
    Returns:
        model, optimizer (if provided), checkpoint data
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer, checkpoint
    
    return model, checkpoint
