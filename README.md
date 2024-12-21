# Facial Emotion Recognition

A deep learning project for facial emotion recognition using PyTorch, implementing a custom CNN architecture to classify seven different emotions.

## Project Structure
```
.
├── data/
│   ├── Images/
│   │   ├── train/
│   │   │   ├── Angry/
│   │   │   ├── Disgust/
│   │   │   ├── Fear/
│   │   │   ├── Happy/
│   │   │   ├── Neutral/
│   │   │   ├── Sad/
│   │   │   └── Surprise/
│   │   └── test/
├── train.py
├── inference.py
└── README.md
```

## Environment Setup

1. Create a new Python environment:
```bash
# Using conda
conda create -n emotion python=3.11.7
conda activate emotion
```

2. Install required packages:
```bash
pip install torch torchvision
pip install opencv-python
pip install albumentations
pip install pandas
pip install tqdm
pip install scikit-learn
pip install matplotlib
pip install seaborn
```

## Data Preparation

1. Place the dataset in the following structure:
```
data/Images/train/
    ├── Angry/
    │   └── *.jpg
    ├── Disgust/
    │   └── *.jpg
    └── ...

data/Images/test/
    └── *.jpg
```

2. Ensure all images are grayscale and of size 48x48 pixels.

## Training

To train the model:

```bash
python train.py
```

Key training parameters (can be modified in train.py):
- Learning rate: 0.001
- Batch size: 32
- Number of epochs: 50
- Early stopping patience: 5

The training process will:
- Save the best model based on validation accuracy
- Generate learning curves
- Log training progress
- Create checkpoints

## Inference

To run inference and generate submissions:

```bash
python inference.py
```

This will:
- Load the best saved model
- Process test images
- Generate a submission.csv file

## Model Architecture

The project uses a custom CNN with:
- 3 convolutional blocks
- Batch normalization
- Dropout layers
- Label smoothing loss
- AdamW optimizer

## File Descriptions

- `train.py`: Main training script with data augmentation and model definition
- `inference.py`: Inference script for generating predictions
- `best_model.pth`: Saved model weights (generated after training)

## Output Files

Training will generate:
- `best_model.pth`: Best model weights
- `training.log`: Training logs
- `learning_curves.png`: Training visualization

Inference will generate:
- `submission.csv`: Predictions for test set

## Performance

Expected performance metrics:
- Training accuracy: ~75%
- Validation accuracy: ~74%

## Requirements

- Python 3.9+
- PyTorch 1.9+
- CUDA 11.3+ (for GPU training)
- 8GB+ RAM
- GPU with 6GB+ VRAM (recommended)

## Notes

- Training uses mixed precision for better performance
- Data augmentation is crucial for better generalization
- Model checkpoints are saved every 5 epochs
- Early stopping is implemented with 5 epochs patience

