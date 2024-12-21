import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
import cv2

def basic_preprocess(image):
    # 與train.py中完全相同的基礎預處理
    image = cv2.fastNlMeansDenoising(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return image

class EmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        for img_name in os.listdir(root_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(root_dir, img_name))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # 應用基礎預處理
        image = basic_preprocess(image)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
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

def create_submission(model, test_loader, device, output_file='submission.csv'):
    model.eval()
    predictions = []
    filenames = []
    
    with torch.no_grad():
        for images, image_names in tqdm(test_loader, desc='Creating submission'):
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transform for test set
    test_transform = A.Compose([
        A.Resize(48, 48),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])
    
    # Create test dataset and loader
    test_dataset = EmotionDataset(
        root_dir="../data/Images/test",
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Load model
    model = EmotionCNN().to(device)
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Create submission
    create_submission(model, test_loader, device)

if __name__ == '__main__':
    main()