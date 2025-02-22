
"""
run_inference.py

This script performs image classification inference using a pre-trained ResNet-50 model.

Usage:
    python run_inference.py <model_path> <image_path>

Arguments:
    model_path (str): Path to the model weights file.
    image_path (str): Path to the image file or directory containing images.

Description:
    The script loads a pre-trained ResNet-50 model, modifies the final fully connected layer to match the number of classes (2 in this case), and loads the saved weights from the specified file. It then preprocesses the input image, runs inference, and prints the predicted class along with the confidence score.

Functions:
    None

Dependencies:
    - torch
    - torchvision
    - PIL
    - argparse
    - matplotlib
    - os
    - json
    - tqdm

Example:
    python run_inference.py /path/to/model_weights.pth /path/to/image.jpg --visualize
    python run_inference.py /path/to/model_weights.pth /path/to/image_directory --output results.json
"""
import torch
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm


CLASSES = ['2DZone', '3DLaueIntersections']

if __name__ == '__main__':
    # Argument parser for file paths
    parser = ArgumentParser(description='Inference script for image classification')
    parser.add_argument('model_path', type=str, help='Path to the model weights file')
    parser.add_argument('image_path', type=str, help='Path to the image file or directory containing images')
    parser.add_argument('--visualize', action='store_true', help='Display the image with predicted class')
    parser.add_argument('--output', type=str, help='Output file for results', default='results.json')
    args = parser.parse_args()
    # Load the saved weights
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # CIFAR-10 has 10 classes
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Define the image transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if os.path.isfile(args.image_path):
        # Load and preprocess the image
        img = Image.open(args.image_path)
        img_t = preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0)

        # Run inference
        with torch.no_grad():
            out = model(batch_t)

        # Get the predicted class
        confidence, index = torch.softmax(out, dim=1).max(dim=1)
        predicted_class = CLASSES[index.item()]

        print(f'The predicted class is: {predicted_class}\nWith confidence: {confidence.item()}')

        results = [
            {
                'image_path': args.image_path,
                'predicted_class': predicted_class,
                'confidence': confidence.item()
            }
        ]        
        if args.visualize:
            plt.imshow(img)
            plt.title(f'Predicted class: {predicted_class}')
            plt.show()
        
    elif os.path.isdir(args.image_path):
        results = []
        print('Running inference on images in the directory...')
        for file in tqdm(os.listdir(args.image_path)):
            if file.endswith('.jpg'):
                img = Image.open(os.path.join(args.image_path, file))
                img_t = preprocess(img)
                batch_t = torch.unsqueeze(img_t, 0)

                # Run inference
                with torch.no_grad():
                    out = model(batch_t)

                # Get the predicted class
                confidence, index = torch.softmax(out, dim=1).max(dim=1)
                predicted_class = CLASSES[index.item()]

                results.append({
                    'image_path': os.path.join(args.image_path, file),
                    'predicted_class': predicted_class,
                    'confidence': confidence.item()
                })
    else:
        print('Invalid image path. Please provide a valid image file or directory.')
        exit()
    
    print(f'Writing results to {args.output}')
    if os.path.dirname(args.output) != '':
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
