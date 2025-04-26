import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split

class ImageDataset(Dataset):
    """Custom dataset for brain tumor MRI images."""
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, 1]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label

def create_dataset(base_directory, split_type, label_map):
    """Create a DataFrame from image files in the given directory."""
    categories = os.listdir(os.path.join(base_directory, split_type))
    data_list = []
    
    for category in categories:
        category_path = os.path.join(base_directory, split_type, category)
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)
            # Ensure we're only adding image files
            if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                data_list.append([file_path, category])
    
    df = pd.DataFrame(data_list, columns=['file_path', 'label'])
    df['label'] = df['label'].map(label_map)
    return df

def get_data_transforms(image_size=(224, 224)):
    """Get data transforms for training and testing."""
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=(0.8, 1.2)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform

def get_data_loaders(base_directory, label_map, batch_size=32, image_size=(224, 224), random_state=42, num_workers=4):
    """Prepare and return DataLoaders for training, validation, and testing."""
    # Get data transforms
    train_transform, test_transform = get_data_transforms(image_size)
    
    # Create DataFrames
    train_df = create_dataset(base_directory, 'Training', label_map)
    test_df = create_dataset(base_directory, 'Testing', label_map)
    
    # Split test data into validation and test sets
    test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=random_state)
    
    # Reset indices
    test_df = test_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    # Print dataset statistics
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Testing samples: {len(test_df)}")
    
    # Class distribution
    print("\nClass distribution:")
    for split_name, df in [("Training", train_df), ("Validation", val_df), ("Testing", test_df)]:
        class_counts = df['label'].value_counts().sort_index()
        print(f"{split_name}: {class_counts.to_dict()}")
    
    # Create datasets
    train_dataset = ImageDataset(train_df, transform=train_transform)
    val_dataset = ImageDataset(val_df, transform=test_transform)
    test_dataset = ImageDataset(test_df, transform=test_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
