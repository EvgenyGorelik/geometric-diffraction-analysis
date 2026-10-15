"""Pinhole camera projection adapted from vggt/extract_camera_positions.ipynb."""
import numpy as np


def camera_matrices(image_size, wavelength, focal_length_px=None,
                    principal_point_px=None, camera_distance=None):
    """Return K and world-to-camera [R|t] for a +z camera looking at the origin.

    This is the notebook's look-at construction for the fixed +z Ewald center.
    Camera +z points toward the origin; image +y points down.
    """
    focal = float(image_size * 10 if focal_length_px is None else focal_length_px)
    center = np.asarray([image_size//2, image_size//2] if principal_point_px is None
                        else principal_point_px, dtype=float)
    distance = float(1/wavelength if camera_distance is None else camera_distance)
    if not np.isfinite(focal) or focal <= 0 or center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError('focal_length_px must be positive; principal_point_px must contain two finite values')
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError('camera_distance must be finite and positive')
    intrinsic = np.array([[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1.]])
    camera_to_world = np.eye(4)
    camera_to_world[:3, :3] = np.diag([1., -1., -1.])
    camera_to_world[:3, 3] = [0, 0, distance]
    return intrinsic, np.linalg.inv(camera_to_world)[:3]


def project_camera(points, intrinsic, extrinsic):
    """Project with K[R|t]; retain source indices, reject nonpositive depth."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    camera_points = homogeneous @ np.asarray(extrinsic).T
    depth = camera_points[:, 2]
    valid = np.isfinite(camera_points).all(axis=1) & (depth > 0)
    projected = camera_points[valid] @ np.asarray(intrinsic).T
    return projected[:, :2]/projected[:, 2, None], depth[valid], np.flatnonzero(valid)


def rasterize_camera(xy, intensities, size, spot_radius=5):
    """Draw notebook-style discs in pixel space, retaining numerical intensity.

    Center coordinates truncate to integer pixels as in the notebook. Contributions
    from overlapping discs add. Radius 0 produces a single-pixel impulse.
    """
    if not isinstance(spot_radius, int) or isinstance(spot_radius, bool) or spot_radius < 0:
        raise ValueError('spot_radius_px must be a nonnegative integer')
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    intensities = np.asarray(intensities, dtype=float)
    if intensities.shape != (len(xy),):
        raise ValueError('one intensity is required per projected point')
    image = np.zeros((size, size), dtype=np.float64)
    valid = np.isfinite(xy).all(axis=1) & (xy >= 0).all(axis=1) & (xy < size).all(axis=1)
    pixels = np.full((len(xy), 2), -1, dtype=int)
    pixels[valid] = xy[valid].astype(int)
    for j in np.flatnonzero(valid):
        x, y = pixels[j]
        if spot_radius == 0:
            image[y, x] += intensities[j]
            continue
        left, right = max(0, x-spot_radius), min(size, x+spot_radius+1)
        top, bottom = max(0, y-spot_radius), min(size, y+spot_radius+1)
        rows, cols = np.ogrid[top:bottom, left:right]
        disc = (cols-x)**2 + (rows-y)**2 < spot_radius**2
        image[top:bottom, left:right] += disc * intensities[j]
    return image, pixels, valid
