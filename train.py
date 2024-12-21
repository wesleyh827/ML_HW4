import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from PIL import Image
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

def preprocess_image(image):
    # Remove noise
    image = cv2.fastNlMeansDenoising(image)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    
    # Normalize
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    return image

class EmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        
        self.emotion_map = {
            'Angry': 0,
            'Disgust': 1,
            'Fear': 2,
            'Happy': 3,
            'Neutral': 4,
            'Sad': 5,
            'Surprise': 6
        }
        
        self.image_paths = []
        self.labels = []
        
        if is_train:
            for emotion_folder in os.listdir(root_dir):
                if emotion_folder in self.emotion_map:
                    emotion_path = os.path.join(root_dir, emotion_folder)
                    label = self.emotion_map[emotion_folder]
                    
                    for img_path in os.listdir(emotion_path):
                        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.image_paths.append(os.path.join(emotion_path, img_path))
                            self.labels.append(label)
        else:
            for img_name in os.listdir(root_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root_dir, img_name))
                    self.labels.append(0)  # Dummy label for test data
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if self.is_train:
            image = preprocess_image(image)
        
        if self.transform:
            if isinstance(self.transform, A.Compose):
                augmented = self.transform(image=image)
                image = augmented['image']
            else:
                image = Image.fromarray(image)
                image = self.transform(image)
        
        if self.is_train:
            return image, self.labels[idx]
        else:
            return image, os.path.basename(img_path)

class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Second block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 7)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc='Training')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Use mixed precision training
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
        
        train_loss = running_loss/len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        val_pbar = tqdm(val_loader, desc='Validation')
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({'acc': 100. * val_correct / val_total})
        
        val_acc = 100. * val_correct / val_total
        
        # Update learning rate scheduler
        scheduler.step(val_acc)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Training Accuracy: {train_acc:.2f}%')
        print(f'Validation Accuracy: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
            print(f'Saved new best model with validation accuracy: {val_acc:.2f}%')
    
    return model, history

def create_submission(model, test_loader, device, output_file='submission.csv'):
    model.eval()
    predictions = []
    filenames = []
    
    test_pbar = tqdm(test_loader, desc='Creating submission')
    with torch.no_grad():
        for images, image_names in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            predictions.extend(predicted.cpu().numpy())
            clean_names = [os.path.splitext(name)[0] for name in image_names]
            filenames.extend(clean_names)
    
    submission_df = pd.DataFrame({
        'filename': filenames,
        'label': predictions
    })
    submission_df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")

def get_sampler(dataset):
    # 只使用訓練集部分的標籤
    labels = [dataset.labels[i] for i in dataset.indices]
    class_counts = torch.bincount(torch.tensor(labels))
    weights = 1.0 / class_counts.float()
    sample_weights = weights[labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms with Albumentations
    train_transform = A.Compose([
        A.Resize(48, 48),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.RandomGamma(p=1),
            A.ColorJitter(p=1)
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(p=1),
            A.MultiplicativeNoise(p=1)
        ], p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.OpticalDistortion(p=1),
            A.GridDistortion(p=1),
            A.ElasticTransform(p=1)
        ], p=0.3),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(48, 48),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])
    
    # Create datasets
    full_dataset = EmotionDataset(
        root_dir="../data/Images/train",
        transform=train_transform,
        is_train=True
    )
    
    test_dataset = EmotionDataset(
        root_dir="../data/Images/test",
        transform=val_transform,
        is_train=False
    )
    
    # Create train/val indices
    train_size = int(0.9 * len(full_dataset))
    indices = torch.randperm(len(full_dataset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create weighted sampler for training
    train_labels = [full_dataset.labels[i] for i in train_indices]
    class_counts = torch.bincount(torch.tensor(train_labels))
    weights = 1.0 / class_counts.float()
    sample_weights = weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(train_indices))
    
    # Create data loaders
    train_loader = DataLoader(
        full_dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        full_dataset,
        batch_size=32,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Number of training images: {len(train_indices)}")
    print(f"Number of validation images: {len(val_indices)}")
    print(f"Number of test images: {len(test_dataset)}")
    
    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {total_params}")
   
    # Initialize model
    model = EmotionCNN().to(device)
    
    count_parameters(model)
    
    
    # Loss function and optimizer
    criterion = LabelSmoothingLoss(classes=7, smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Train model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=50,
        device=device
    )
    
    # Load best model and create submission
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    create_submission(model, test_loader, device)

if __name__ == '__main__':
    main()