import os
from PIL import Image
import argparse
import numpy as np
import cv2
from scipy import ndimage
import json
from tqdm import tqdm

from scipy.ndimage import maximum_filter

def non_max_suppression(img, size):
    # Apply maximum filter
    max_img = maximum_filter(img, size=size)
    
    # Suppress non-maximum values
    suppressed_img = (img == max_img) * img
    
    return suppressed_img

def convert_img(img: np.ndarray, img_size: tuple, dot_size: tuple, threshold: float, nms_size: int):

    non_max_suppression(img, nms_size)
    canvas = np.ones_like(img)
    canvas[img > threshold*(img.max() - img.min())] = 0

    # Upscale canvas to the size of synth_img
    upscaled_canvas = cv2.resize(canvas, (img_size[1], img_size[0])).astype(canvas.dtype)
    binary_img = upscaled_canvas < 1

    # Label the connected components in the binary image
    labeled_img, num_features = ndimage.label(binary_img)

    # Get the centers of all features
    centers = ndimage.center_of_mass(binary_img, labeled_img, range(1, num_features + 1))

    canvas_extend = np.ones_like(upscaled_canvas)
    X, Y = np.meshgrid(np.arange(upscaled_canvas.shape[1]),np.arange(upscaled_canvas.shape[0]))
    for center in centers:
        if np.linalg.norm(center - np.array(upscaled_canvas.shape)/2) < 10:
            canvas_extend[((X - center[1])**2 + (Y - center[0])**2) < dot_size[1]**2] = 0
        else:        
            canvas_extend[((X - center[1])**2 + (Y - center[0])**2) < dot_size[0]**2] = 0
    canvas_rgb = np.stack((canvas_extend,)*3, axis=-1) * 255
    return canvas_rgb.astype(np.int16)

def main(args):
    input_folder = args.input_folder
    config_file = args.config_file
    with open(config_file, 'r') as fp:
        config = json.load(fp)
    
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith('.tif'):
            img = np.asarray(Image.open(os.path.join(input_folder, filename)))
            img_converted = convert_img(
                    img,
                    (config['img_width'], config['img_height']),
                    (config['dot_radius_median'], config['dot_radius_max']),
                    args.threshold,
                    args.nms_size
                )
            output_filename = os.path.splitext(filename)[0] + '.png'
            cv2.imwrite(os.path.join(output_folder, output_filename), img_converted)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert .tif images to .png format.')
    parser.add_argument('input_folder', type=str, help='Path to the input folder containing .tif images')
    parser.add_argument('config_file', type=str, help='Config path')
    parser.add_argument('--threshold', type=float, help='(img_max-img_min)*threshold', default=0.5)
    parser.add_argument('--nms_size', type=int, help='non max suppression filtersize', default=10)
    parser.add_argument('--output_folder', type=str, help='Path to the output folder to save .png images', default='data')
    args = parser.parse_args()

    main(args)