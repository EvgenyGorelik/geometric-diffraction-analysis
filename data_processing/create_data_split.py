"""
This script creates a data split for training and validation from a given dataset.
The data is split based on the given ratio, and the split data and labels are saved in the output folder.
The datastructure of the input folder is as follows:
    data_path
    ├── label1
    │   ├──
    │   ├──
    │   └──
    ├── label2
    │   ├──
    │   ├──
    │   └──
    └── label3

    
The output folder will have the following structure:
    output_folder
    ├── train
    │   ├── files
    │   │
    │   └── labels.json
    └── val
        ├── files
        │
        └── labels.json
Usage:
    python create_data_split.py --data_path <data_path> --split_ratio <split_ratio> --labels <label1> <label2> ... --output_folder <output_folder>
    
Functions:
    create_split(data_path: str, split_ratio: float, labels: list, output_folder: str):
        Creates a split of the data into training and validation sets, and saves the split data and labels.
Arguments:
    --data_path (str): Path to the data directory. Default is '../data/set1'.
    --split_ratio (float): Ratio of training data to validation data. Default is 0.8.
    --labels (list of str): List of labels to consider for the split. Default is ['2DZone', '3DLaueIntersections'].
    --output_folder (str): Output folder for the split data. Default is '../data/set1_processed'.
Example usage:
    python create_data_split.py --data_path ../data/set1 --split_ratio 0.8 --labels 2DZone 3DLaueIntersections --output_folder ../data/set1_processed
"""

import os
import json
import shutil
import random
import argparse




def create_split(data_path: str = 'data', split_ratio: float = 0.8, labels: list = ['2DZone', '3DLaueIntersections'], output_folder: str = 'output'):
    train_folder = os.path.join(output_folder, 'train')
    train_file_folder = os.path.join(train_folder, 'files')
    val_folder = os.path.join(output_folder, 'val')
    val_file_folder = os.path.join(val_folder, 'files')
    os.makedirs(train_file_folder, exist_ok=True)
    os.makedirs(val_file_folder, exist_ok=True)
    
    label_train_data = []
    label_val_data = []
    for label in labels:
        assert os.path.exists(os.path.join(data_path, label)), f"Label {label} does not exist in {data_path}"
        for filename in os.listdir(os.path.join(data_path, label)):
            if filename.endswith('.jpg'):
                rand_num = random.random()
                if rand_num < split_ratio:
                    shutil.copy(os.path.join(data_path, label, filename), os.path.join(train_file_folder, filename))
                    label_train_data.append({"file_path": os.path.join(filename), "file_class": label})
                else:
                    shutil.copy(os.path.join(data_path, label, filename), os.path.join(val_file_folder, filename))
                    label_val_data.append({"file_path": os.path.join(filename), "file_class": label})

    with open(os.path.join(train_folder, 'labels.json'), 'w+') as f:
        json.dump(label_train_data, f, indent=4)

    with open(os.path.join(val_folder, 'labels.json'), 'w+') as f:
        json.dump(label_val_data, f, indent=4)

    with open(os.path.join(output_folder, 'classes.json'), 'w+') as f:
        json.dump(labels, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create data split for training and validation.')
    parser.add_argument('--data_path', type=str, default='../data/set1', help='Path to the data directory')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Ratio of training data to validation data')
    parser.add_argument('--labels', type=str, nargs='+', default=['2DZone', '3DLaueIntersections'], help='List of labels')
    parser.add_argument('--output_folder', type=str, default='../data/set1_processed', help='Output folder for the split data')

    args = parser.parse_args()

    create_split(data_path=args.data_path, split_ratio=args.split_ratio, labels=args.labels, output_folder=args.output_folder)