import os
import time
from datetime import datetime
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.tensorboard import SummaryWriter

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                device, num_epochs=10, checkpoint_dir='checkpoints', patience=7, label_map=None):
    """
    Train the model with TensorBoard logging and early stopping.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer for updating weights
        scheduler: Learning rate scheduler
        device: Device to train on (cuda/cpu)
        num_epochs: Maximum number of training epochs
        checkpoint_dir: Directory to save model checkpoints
        patience: Number of epochs to wait before early stopping
        label_map: Dictionary mapping class indices to class names
    
    Returns:
        Trained model
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set up TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/brain_tumor_cnn_{timestamp}')
    
    # Inverse label map for TensorBoard logging
    if label_map:
        inv_label_map = {v: k for k, v in label_map.items()}
    else:
        inv_label_map = {i: str(i) for i in range(len(train_loader.dataset.dataframe['label'].unique()))}
    
    # Track best performance for early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        
        #-------- Training Phase --------#
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Store predictions and labels for metrics
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            
            # Log batch loss
            writer.add_scalar('Batch/Loss', loss.item(), epoch * len(train_loader) + batch_idx)
            
        train_loss /= len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        
        #-------- Validation Phase --------#
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Store predictions and labels for metrics
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        
        # Calculate metrics
        train_cm = confusion_matrix(all_train_labels, all_train_preds)
        val_cm = confusion_matrix(all_val_labels, all_val_preds)
        
        # Generate classification reports
        train_report = classification_report(all_train_labels, all_train_preds, output_dict=True)
        val_report = classification_report(all_val_labels, all_val_preds, output_dict=True)
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        
        # Log per-class metrics
        num_classes = len(np.unique(all_train_labels + all_val_labels))
        for class_idx in range(num_classes):
            class_name = inv_label_map[class_idx]
            if str(class_idx) in train_report and str(class_idx) in val_report:
                writer.add_scalar(f'Precision/Train/{class_name}', train_report[str(class_idx)]['precision'], epoch)
                writer.add_scalar(f'Recall/Train/{class_name}', train_report[str(class_idx)]['recall'], epoch)
                writer.add_scalar(f'F1-score/Train/{class_name}', train_report[str(class_idx)]['f1-score'], epoch)
                
                writer.add_scalar(f'Precision/Val/{class_name}', val_report[str(class_idx)]['precision'], epoch)
                writer.add_scalar(f'Recall/Val/{class_name}', val_report[str(class_idx)]['recall'], epoch)
                writer.add_scalar(f'F1-score/Val/{class_name}', val_report[str(class_idx)]['f1-score'], epoch)
        
        # Step the learning rate scheduler
        if scheduler:
            scheduler.step(val_loss)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Print statistics
        print(f'Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 60)
        
        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    writer.close()
    return model

def evaluate_model(model, test_loader, criterion, device, label_map=None):
    """
    Evaluate the model on the test set and display detailed metrics.
    
    Args:
        model: The neural network model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to evaluate on (cuda/cpu)
        label_map: Dictionary mapping class indices to class names
    
    Returns:
        test_loss, test_accuracy, confusion_matrix
    """
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss /= len(test_loader)
    test_acc = 100.0 * accuracy_score(all_labels, all_preds)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Get class names for report if label_map is provided
    target_names = list(label_map.keys()) if label_map else None
    
    # Print detailed evaluation metrics
    print("\n===== Model Evaluation =====")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    return test_loss, test_acc, cm
