import argparse
import numpy as np
from PIL import Image
from scipy import ndimage
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get metadata from synthetic image.')
    parser.add_argument('img_path', type=str, help='Path to image')
    parser.add_argument('--threshold', type=float, help='(img_max-img_min)*threshold', default=0.5)
    parser.add_argument('--output', type=str, help='Path to result', default='img_stats.json')
    
    args = parser.parse_args()
    synth_img = np.array(Image.open(args.img_path))

    # Convert the image to grayscale if it's not already
    gray_synth_img = np.mean(synth_img, axis=2)

    # Threshold the image to create a binary image
    binary_synth_img = gray_synth_img < args.threshold*(np.max(gray_synth_img) - np.min(gray_synth_img)) 

    # Label the connected components in the binary image
    labeled_img, num_features = ndimage.label(binary_synth_img)

    # Measure the properties of the labeled regions
    sizes = ndimage.sum(binary_synth_img, labeled_img, range(num_features + 1))

    # Find the diameter of the largest dot
    dot_diameters = np.sqrt(sizes / np.pi) * 2
    dot_radius = np.array([np.median(dot_diameters), np.max(dot_diameters)]) / 2

    print(f'max: {np.max(synth_img)}\nmin: {np.min(synth_img)}\nnum_dots: {num_features}\nmedian_dot_radius: {dot_radius[0]}\nmax_dot_radius: {dot_radius[1]}')
    results = {
        'img_width': int(synth_img.shape[0]),
        'img_height': int(synth_img.shape[1]),
        'img_max': float(np.max(synth_img)),
        'img_min': float(np.min(synth_img)),
        'num_dots': int(num_features),
        'dot_radius_median': float(dot_radius[0]),
        'dot_radius_max': float(dot_radius[1])
    }
    with open(args.output, 'w+') as fp:
        json.dump(results, fp)