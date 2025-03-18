import cv2
import numpy as np
import os
import random
import argparse

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def overlay_images(img1, img2):
    # Ensure both images are binary
    _, img1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
    _, img2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
    
    # Resize images to the same size if necessary
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Overlay images
    overlay = cv2.bitwise_and(img1, img2)
    return overlay

def main(folder, output_folder):
    images = load_images_from_folder(folder)
    
    if len(images) < 2:
        print("Not enough images to overlay.")
        return
    
    os.makedirs(output_folder, exist_ok=True)

    for i in range(len(images)):
        img1 = images[i]
        img2 = random.choice(images)
        while img2 is img1:
            img2 = random.choice(images)
        
        overlay = overlay_images(img1, img2)
        cv2.imwrite(os.path.join(output_folder,f'overlay_{i}.jpg'), cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Overlay images from a specified folder.')
    parser.add_argument('folder', type=str, help='Path to the folder containing images')
    parser.add_argument('output_folder', type=str, help='Path to the folder to save overlay images')
    args = parser.parse_args()
    main(args.folder, args.output_folder)
