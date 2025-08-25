import cv2
import matplotlib.pyplot as plt
import numpy as np

from glob import glob
from tqdm import tqdm
import os

from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.interpolate import interp1d

from argparse import ArgumentParser


def radial_profile(data, center=None, bins=100, reduce=np.median):
    """
    Compute radial profile of 2D data by integrating pixel values radially.

    Parameters:
        data : 2D numpy array
            Input image data (e.g., Gaussian bell).
        center : tuple or None
            (x, y) coordinates of the center. If None, uses image center.
        bins : int
            Number of radial bins.

    Returns:
        bin_centers : 1D numpy array
            Radius values (bin centers).
        radial_mean : 1D numpy array
            Radially integrated intensity (mean or sum per bin).
    """
    y, x = np.indices(data.shape)

    if center is None:
        center = (x.max() / 2, y.max() / 2)

    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)

    # Flatten arrays
    r = r.ravel()
    values = data.ravel()

    # Bin the radii
    bin_edges = np.linspace(0, r.max(), bins + 1)
    bin_indices = np.digitize(r, bin_edges)

    radial_sum = np.zeros(bins)
    radial_count = np.zeros(bins)
    radial_median = np.zeros(bins)

    for i in range(1, bins + 1):
        mask = bin_indices == i
        radial_sum[i-1] = values[mask].sum()
        radial_count[i-1] = mask.sum()
        radial_median[i-1] = reduce(values[mask])


    # Avoid division by zero
    radial_mean = radial_sum / np.maximum(radial_count, 1)

    # Calculate bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return bin_centers, radial_median

def create_rotational_image_from_profile(r, profile, shape, center):
    """
    Create a 2D rotationally symmetric image from a 1D radial profile.

    Parameters:
        r : 1D numpy array
            Radial distances where profile is defined (monotonically increasing).
        profile : 1D numpy array
            Profile values at corresponding radial distances.
        shape : tuple of ints (height, width)
            Size of the output 2D image.
        center : tuple of floats (x_center, y_center)
            Coordinates of the rotational center in the output image.

    Returns:
        img : 2D numpy array
            Generated image with intensities following the radial profile.
    """

    height, width = shape
    y, x = np.indices((height, width))
    x = x - center[0]
    y = y - center[1]
    
    # Calculate the radial distance of each pixel from the center
    dist = np.sqrt(x**2 + y**2)

    # Interpolate the profile to find value at each pixel radius
    interp_func = interp1d(r, profile, bounds_error=False, fill_value=0)

    img = interp_func(dist)

    return img

def nms_filter(image, size=3, threshold=0):
    """
    Apply Non-Maximum Suppression (NMS) to a 2D image.

    Parameters:
        image : 2D numpy array
            Input image.
        size : int
            Neighborhood size for local maxima (must be odd).
        threshold : float
            Minimum value to consider as a peak.

    Returns:
        nms_mask : 2D boolean numpy array
            Mask of local maxima after NMS.
    """

    # Find local maxima
    local_max = (image == maximum_filter(image, size=size))
    # Apply threshold
    nms_mask = (local_max & (image > threshold)).astype(np.uint8)
    return nms_mask

def snap_img_to_rotational(img, rotational_img, closest_max_pos, mode='distance', rad=300, sigma=4):
    """
    Snap the original image to the rotationally symmetric image within a specified radius.

    Parameters:
        img : 2D numpy array
            Original image.
        rotational_img : 2D numpy array
            Rotationally symmetric image.
        closest_max_pos : tuple of floats (x_center, y_center)
            Coordinates of the rotational center.
        rad : int
            Radius around the center to snap.
        sigma : float
            Standard deviation for Gaussian smoothing.

    Returns:
        img_snapping : 2D numpy array
            Image with snapped intensities within the specified radius.
    """
    assert mode in ['distance', 'value'], "Unsupported mode. Use 'distance', 'value'."

    if mode == 'value':
        rotational_img_norm = (rotational_img - rotational_img.min()) / (rotational_img.max() - rotational_img.min())
        indices_y, indices_x = np.where(rotational_img_norm > 0.004)
    elif mode == 'distance':
        indices_y, indices_x = np.indices(img.shape)
        distances = np.sqrt((indices_x - closest_max_pos[0])**2 + (indices_y - closest_max_pos[1])**2)
        mask = distances < rad
        indices_y = indices_y[mask]
        indices_x = indices_x[mask]

    img_snapping = gaussian_filter(img, sigma=sigma).copy()
    img_snapping[indices_y, indices_x] = rotational_img[indices_y, indices_x]

    return img_snapping

def calculate_center_coordinate(img, sigma=10, search_size=40, neighborhood_size=20):
    filtered_image = gaussian_filter(img, sigma=sigma)
    center_ind = filtered_image.argmax()
    center_coor = np.array([center_ind % img.shape[0], center_ind // img.shape[0]])

    image, x, y = filtered_image, center_coor[0], center_coor[1]
    half_size = search_size // 2

    # Define search window boundaries with clipping at image edges
    x_min = max(x - half_size, 0)
    x_max = min(x + half_size + 1, image.shape[1])
    y_min = max(y - half_size, 0)
    y_max = min(y + half_size + 1, image.shape[0])

    # Extract the search region
    search_region = image[y_min:y_max, x_min:x_max]

    # Find local maxima in the search region using a maximum filter
    local_max = (search_region == maximum_filter(search_region, size=neighborhood_size))

    # Get coordinates of all local maxima in the search region
    max_coords = np.argwhere(local_max)

    if len(max_coords) == 0:
        # No local maxima found, return the original point and its value
        return (x, y), image[y, x]

    # Convert local max coordinates to global image coordinates
    global_coords = max_coords + np.array([y_min, x_min])

    # Calculate distances to (y, x)
    distances = np.sqrt((global_coords[:, 0] - y)**2 + (global_coords[:, 1] - x)**2)

    # Find the closest maximum
    closest_index = np.argmin(distances)
    closest_max_pos = (global_coords[closest_index, 1], global_coords[closest_index, 0])  # (x, y)
    closest_max_value = image[closest_max_pos[1], closest_max_pos[0]]
    return closest_max_pos, closest_max_value



if __name__ == "__main__":

    parser = ArgumentParser(description='Extract peaks from images using rotational symmetry')
    parser.add_argument('--data_path', type=str, default='data', help='Path to the image files')
    parser.add_argument('--output_path', type=str, default='output', help='Path to save output results')
    parser.add_argument('--center_sigma', type=float, default=10, help='Sigma for Gaussian filter in center detection')
    parser.add_argument('--center_search_size', type=int, default=40, help='Search size for center coordinate')
    parser.add_argument('--center_neighborhood_size', type=int, default=20, help='Neighborhood size for local maxima')
    parser.add_argument('--snapping_sigma', type=float, default=10, help='Sigma for Gaussian filter in center detection')
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    file_list = sorted(glob(os.path.join(args.data_path, '*.tif')))
    for f in tqdm(file_list):
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED).astype(np.uint16)
        closest_max_pos, closest_max_value = calculate_center_coordinate(img, sigma=args.center_sigma, search_size=args.center_search_size, neighborhood_size=args.center_neighborhood_size)

        r, profile = radial_profile(img, center=closest_max_pos, bins=300, reduce=np.median)
        rotational_img = create_rotational_image_from_profile(r, profile, img.shape, closest_max_pos)
        img_snapping = snap_img_to_rotational(img, rotational_img, closest_max_pos, mode='distance', rad=300, sigma=args.snapping_sigma)
        filtered_img = nms_filter(img_snapping - rotational_img, size=40, threshold=10)
        peaks_y, peaks_x = np.where(filtered_img)

        
        # Draw the peaks on the image using OpenCV
        output_img = np.ones_like(img, dtype=np.uint8)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR) * 255
        for px, py in zip(peaks_x, peaks_y):
            cv2.circle(output_img, (px, py), radius=30, color=(0, 0, 0), thickness=-1)

        # Save the output image with peaks marked
        cv2.imwrite(os.path.join(args.output_path, os.path.basename(f).replace('.tif', '_peaks.png')), output_img)
