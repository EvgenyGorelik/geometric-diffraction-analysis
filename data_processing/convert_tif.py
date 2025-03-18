import os
from PIL import Image
import argparse


def main(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):
            img = Image.open(os.path.join(input_folder, filename))
            output_filename = os.path.splitext(filename)[0] + '.png'
            img.save(os.path.join(output_folder, output_filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert .tif images to .png format.')
    parser.add_argument('input_folder', type=str, help='Path to the input folder containing .tif images')
    parser.add_argument('output_folder', type=str, help='Path to the output folder to save .png images')
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    main(input_folder, output_folder)