import shutil
import os
import glob
from argparse import ArgumentParser

if __name__ == '__main__':

    args_parser = ArgumentParser()
    args_parser.add_argument('--output_path', type=str, default='data/segmentation', help='Path to prepared data directory')
    args_parser.add_argument('--data_path', type=str, default='data/volume_500', help='Path to raw data directory')
    args = args_parser.parse_args()

    zonal_labels = []

    # create train and val directories if they don't exist
    train_img_dir = os.path.join(args.output_path, 'train', 'images')
    train_mask_dir = os.path.join(args.output_path, 'train', 'masks')
    val_img_dir = os.path.join(args.output_path, 'val', 'images')
    val_mask_dir = os.path.join(args.output_path, 'val', 'masks')

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)

    directories = sorted([e for e in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, e))])
    img_directories = list()
    mask_directories = list()
    for d in directories:
        if '_zonal_seg' in d:
            mask_directories.append(os.path.join(args.data_path, d))
            with open(os.path.join(args.data_path, f'{d}.txt'), 'r') as f:
                zonal_labels += [line.strip().split('/')[-1] for line in f.readlines()]
        else:
            img_directories.append(os.path.join(args.data_path, d))

    for img_dir, mask_dir in zip(img_directories, mask_directories):
        img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.jpg')))

        # split 80% train, 20% val
        split_idx = int(0.8 * len(img_files))
        train_img_files = img_files[:split_idx]
        train_mask_files = mask_files[:split_idx]
        val_img_files = img_files[split_idx:]
        val_mask_files = mask_files[split_idx:]

        # copy files to their respective directories
        for f in train_img_files:
            shutil.copy(f, train_img_dir)
        for f in train_mask_files:
            shutil.copy(f, train_mask_dir)
        for f in val_img_files:
            shutil.copy(f, val_img_dir)
        for f in val_mask_files:
            shutil.copy(f, val_mask_dir)

    # save zonal labels to a text file
    with open(os.path.join(args.output_path, 'zonal_labels.txt'), 'w') as f:
        for label in zonal_labels:
            f.write(f"{label}\n")