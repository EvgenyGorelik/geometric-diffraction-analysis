
import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch

'''
folder structure:
    data
        - validataion
            - img00050004.jpg
            - img00050005.jpg
            - img00050007.jpg
            - img00050009.jpg
            - ...
        - training
            - img0000000.jpg
            - img0000001.jpg
            - img0000002.jpg
            - img0000003.jpg
            - img0000004.jpg
            - ...
        - labels.json

        
labels.json:
    [
        {
            "file_path": "img00050004.jpg",
            "file_class": 0
        },
        {
            "file_path": "img00050005.jpg",
            "file_class": 1
        },
        {
            "file_path": "img00050006.jpg",
            "file_class": 1
        },
        {
            "file_path": "img00050007.jpg",
            "file_class": 0
        },
        {
            "file_path": "img00000000.jpg",
            "file_class": 0
        },
        {
            "file_path": "img00000001.jpg",
            "file_class": 1
        }
    ]

'''

class DIFFFibonacciDataset(Dataset):
    def __init__(self, root_dir, label_file : str = 'labels.json', transform=None):
        self.root_dir = root_dir
        self.file_dir = os.path.join(root_dir, 'files')
        self.data_list = os.listdir(self.file_dir)
        self.transform = transform
        self.label_file = os.path.join(root_dir, label_file)
        self.labels = self.load_labels()
        self.classes  = ['2DZone', '3DLaueIntersections']
        self.file_class_map = {i: self.classes[i] for i in range(len(self.classes))}
        self.num_classes = len(self.file_class_map)

    def load_labels(self):
        with open(self.label_file, 'r') as f:
            labels = json.load(f)
        labels = {label['file_path']: label['file_class'] for label in labels}
        return labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.file_dir, self.data_list[idx]) 
        image = Image.open(img_name).convert('RGB')
        label = self.file_class_map[self.labels[self.data_list[idx]]]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_class_weights(dataset):
    class_counts = [0, 0]
    for _, label in dataset:
        class_counts[label] += 1
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    return class_weights

def get_sampler(dataset, class_weights):
    sample_weights = [0] * len(dataset)
    for idx, (_, label) in enumerate(dataset):
        sample_weights[idx] = class_weights[label]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

def create_dataloader(root_dir, batch_size=32, transform=None, use_class_weights=False):
    assert os.path.exists(root_dir), f"Directory {root_dir} does not exist"
    # Create datasets
    dataset = DIFFFibonacciDataset(root_dir=root_dir, transform=transform)

    if use_class_weights:
        # Calculate class weights
        class_weights = get_class_weights(dataset)
        
        # Create samplers
        sampler = get_sampler(dataset, class_weights)
    else:
        sampler = None
    
    # Create dataloaders with samplers
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from torchvision import transforms

    data_loader = create_dataloader(root_dir='data', batch_size=4, 
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))

    # Get the first batch of images and labels
    images, labels = next(iter(data_loader))

    # Select the first image
    image = images[0]

    # Convert the tensor to a numpy array and transpose the dimensions
    image = image.numpy().transpose((1, 2, 0))

    # Unnormalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    # Plot the image
    plt.imshow(image)
    plt.title(f'Label: {data_loader.dataset.file_class_map[labels[0].item()]}')
    plt.show()