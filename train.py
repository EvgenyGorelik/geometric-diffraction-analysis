"""
This script trains a ResNet model on a dataset using PyTorch.
Functions:
    set_seed(seed):
        Sets the random seed for reproducibility across various libraries.
    main():
        Parses command-line arguments, sets up data loaders, model, loss function, and optimizer.
        Trains the model and evaluates it on a validation set, saving the best model.
Command-line arguments:
    --root_dir (str): Root directory of the dataset. Default is 'data/set1_processed'.
    --batch_size (int): Batch size for training. Default is 32.
    --learning_rate (float): Learning rate for the optimizer. Default is 0.001.
    --num_epochs (int): Number of epochs to train the model. Default is 7.
    --model_path (str): Path to save the trained model. Default is 'models/diff_fibonacci_model.pth'.
    --use_class_weights (bool): Use class weights for training. Default is False.
    --seed (int): Random seed for reproducibility. Default is 42.
"""

import torch
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from utils.data_utils import create_dataloader
import os
import argparse
import random
import numpy as np



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description='Train a ResNet model on a dataset')
    parser.add_argument('--root_dir', type=str, default='data/set1_processed', help='Root directory of the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=7, help='Number of epochs to train the model')
    parser.add_argument('--model_path', type=str, default='models/diff_fibonacci_model.pth', help='Path to save the trained model')
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    root_dir = args.root_dir
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    model_path = args.model_path
    use_class_weights = args.use_class_weights

    print('Setting random seed...')
    set_seed(args.seed)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create dataloaders
    print('Creating dataloaders...')
    train_loader = create_dataloader(root_dir=os.path.join(root_dir, 'train'), batch_size=batch_size, transform=transform, use_class_weights=use_class_weights)
    val_loader = create_dataloader(root_dir=os.path.join(root_dir, 'val'), batch_size=batch_size, transform=transform, use_class_weights=use_class_weights)

    # Model, loss function, and optimizer
    print('Creating model...')
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, train_loader.dataset.num_classes) 

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_score = 0.0

    print('Starting training...')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

        # Validation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        score = correct / total
        print(f'Validation Accuracy: {100 * score}%')
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), model_path)
            print(f'Model saved to {model_path}')


    print('Training complete')


if __name__ == '__main__':
    main()
