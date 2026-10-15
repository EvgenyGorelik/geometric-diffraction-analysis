"""Run from the repository root: python -m examples.reconstruct_lattice_demo."""
import numpy as np
from scipy.spatial.transform import Rotation
from utils.reconstruct_lattice import Calibration, reconstruct_images


def synthetic_batch():
    """Five nearby orientations; truth is used only to generate/validate images."""
    basis = np.array([[.051, 0, 0], [.012, .063, 0], [.011, .023, .076]])
    hkl = np.array(np.meshgrid(*[np.arange(-20, 21)] * 3)).reshape(3, -1).T
    lattice = hkl @ basis
    calibration = Calibration((1024, 1024), .002, .4)
    rotations = Rotation.from_euler('ZYX', [[0, 0, 0], [2, .01, .006],
        [4, .02, .012], [6, .03, .018], [8, .04, .024]], degrees=True).as_matrix()
    images = []
    for rotation in rotations:
        q = lattice @ rotation
        q = q[abs(np.linalg.norm(q - [0, 0, 2.5], axis=1) - 2.5) < .001]
        xy = q[:, :2] * 2.5 / (2.5 - q[:, 2, None])
        uv = xy / calibration.pixel_size + calibration.center
        image = np.full((2048, 2048), 255, dtype=np.uint8)
        for u, v in uv:
            x, y = int(round(u)), int(round(v))
            if 4 <= x < 2044 and 4 <= y < 2044:
                yy, xx = np.mgrid[y-3:y+4, x-3:x+4]
                patch = 255 * (1 - np.exp(-((xx-u)**2 + (yy-v)**2) / (2 * .8**2)))
                image[y-3:y+4, x-3:x+4] = np.minimum(image[y-3:y+4, x-3:x+4], np.rint(patch).astype(np.uint8))
        images.append(image)
    return images, [calibration] * len(images), basis


def main():
    images, calibrations, truth = synthetic_batch()
    result = reconstruct_images(images, calibrations, tolerance=.003, max_trials=6000, peak_options={'threshold': 254})
    volume = abs(np.linalg.det(result['basis']))
    expected = abs(np.linalg.det(truth))
    print('Reconstructed reciprocal basis (rows):\n', result['basis'])
    print(f'Indexed fraction: {result["inliers"].mean():.1%}')
    print(f'Reciprocal cell volume: {volume:.8g}; true: {expected:.8g}')
    print(f'RMS indexing error: {np.sqrt(np.mean(result["residuals"]**2)):.5g}')
    return result


if __name__ == '__main__':
    main()
