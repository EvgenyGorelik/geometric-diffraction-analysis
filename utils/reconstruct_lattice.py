"""Image-only orientation and reciprocal-lattice estimation with known calibration.

Row-vector convention: camera_q @ rotation = reference_q; hkl @ basis = q.
No orientations, spot identities, or lattice parameters are inputs.
"""
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.ndimage import center_of_mass, label
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Calibration:
    center: tuple
    pixel_size: float
    wavelength: float
    detector_distance: float = 0.0

    def lift(self, uv):
        """Invert simulation.utils.project_reflections_2D on the Ewald sphere.

        pixel_size converts pixels to the simulator's detector coordinates.
        Pixel v and detector y have the same sign. Finite excitation thickness
        becomes coordinate error; it cannot be recovered from a single image.
        """
        if self.pixel_size <= 0 or self.wavelength <= 0 or self.detector_distance < 0:
            raise ValueError('Calibration requires positive scale/wavelength and nonnegative distance')
        xy = (np.asarray(uv, float) - self.center) * self.pixel_size
        radius = 1 / self.wavelength
        rays = np.column_stack((xy, np.full(len(xy), -self.detector_distance - radius)))
        return radius * rays / np.linalg.norm(rays, axis=1)[:, None] + [0, 0, radius]


def extract_spots(image, *, dark=True, threshold=None, min_area=3, max_area=None):
    """Intensity-weighted component centroids (u,v), suitable for separated dots.

    Use a preprocessed image for experimental backgrounds. Touching spots must
    be separated upstream. Threshold is in input intensity units.
    """
    im = np.asarray(image, float)
    if im.ndim != 2 or not np.isfinite(im).all():
        raise ValueError('Expected a finite grayscale image')
    threshold = (im.min() + im.max()) / 2 if threshold is None else threshold
    mask = im < threshold if dark else im > threshold
    components, count = label(mask)
    sizes = np.bincount(components.ravel())
    ids = [i for i in range(1, count + 1) if sizes[i] >= min_area and
           (max_area is None or sizes[i] <= max_area)]
    if not ids:
        return np.empty((0, 2))
    weights = np.maximum(threshold - im if dark else im - threshold, 0)
    return np.asarray(center_of_mass(weights, components, ids))[:, ::-1]


def _rotation(a, b):
    u, _, vt = np.linalg.svd(a.T @ b)
    return u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt


def _matches(a, b, rotation, tolerance):
    distances, indices = cKDTree(b).query(a @ rotation)
    # Enforce one-to-one correspondences, keeping the closest first.
    used, pairs = set(), []
    for i in np.argsort(distances):
        j = int(indices[i])
        if distances[i] <= tolerance and j not in used:
            used.add(j)
            pairs.append((i, j))
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def match_views(a, b, *, tolerance, min_matches=4, max_trials=4000, seed=0):
    """Radial proposals -> independent vector pairs -> SO(3) -> consensus.

    Returns a rotation taking a to b, with matched indices and a score, or None.
    Short reflections are sampled and scored more strongly. Search is bounded;
    symmetric lattices can admit several equivalent orientations.
    """
    rng = np.random.default_rng(seed)
    ra, rb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    candidates = np.argwhere(abs(ra[:, None] - rb[None, :]) <= tolerance)
    if len(candidates) < 2:
        return None
    weights = 1 / np.maximum(ra[candidates[:, 0]], tolerance)
    weights /= weights.sum()
    best = None
    seen = set()
    for _ in range(max_trials):
        chosen = tuple(sorted(rng.choice(len(candidates), 2, replace=False, p=weights)))
        if chosen in seen:
            continue
        seen.add(chosen)
        ij = candidates[list(chosen)]
        ia, ib = ij[:, 0], ij[:, 1]
        if ia[0] == ia[1] or ib[0] == ib[1]:
            continue
        aa, bb = a[ia], b[ib]
        if min(np.linalg.norm(np.cross(*aa)), np.linalg.norm(np.cross(*bb))) < 0.05 * max(np.prod(ra[ia]), np.prod(rb[ib])):
            continue
        if abs(np.linalg.norm(aa[0] - aa[1]) - np.linalg.norm(bb[0] - bb[1])) > 2 * tolerance:
            continue
        rot = _rotation(aa, bb)
        pairs = _matches(a, b, rot, tolerance)
        if len(pairs) < min_matches:
            continue
        for _ in range(3):
            rot = _rotation(a[pairs[:, 0]], b[pairs[:, 1]])
            pairs = _matches(a, b, rot, tolerance)
            if len(pairs) < min_matches:
                break
        if len(pairs) < min_matches:
            continue
        residual = np.linalg.norm(a[pairs[:, 0]] @ rot - b[pairs[:, 1]], axis=1)
        score = float(np.sum((1 - (residual / tolerance)**2) / (1 + ra[pairs[:, 0]] / np.median(ra))))
        if best is None or score > best['score']:
            best = dict(rotation=rot, pairs=pairs, score=score, rms=float(np.sqrt(np.mean(residual**2))))
    return best


def fit_basis(points, *, tolerance, candidate_limit=32, min_fraction=0.9):
    """Search short observed vectors/differences, then alternate indexing and LS.

    Finite search, not a proof of primitivity. Missing reflections may leave a
    sublattice indistinguishable from the primitive lattice.
    """
    points = np.asarray(points, float)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or tolerance <= 0 or candidate_limit < 3 or not 0 < min_fraction <= 1):
        raise ValueError('Require finite (M,3) points, positive tolerance, at least 3 candidates, and fraction in (0,1]')
    singular = np.linalg.svd(points, compute_uv=False)
    if len(singular) < 3 or singular[-1] < 3 * tolerance * np.sqrt(len(points)):
        raise ValueError('Insufficient independent planes: observed points are effectively coplanar')
    short = points[np.argsort(np.linalg.norm(points, axis=1))[:80]]
    vectors = np.vstack((short, (short[:, None] - short[None, :]).reshape(-1, 3)))
    candidates = []
    for v in vectors[np.argsort(np.linalg.norm(vectors, axis=1))]:
        if np.linalg.norm(v) <= 3 * tolerance:
            continue
        if any(min(np.linalg.norm(v - w), np.linalg.norm(v + w)) < 2 * tolerance for w in candidates):
            continue
        near = vectors[np.linalg.norm(vectors - v, axis=1) < 2 * tolerance]
        opposite = vectors[np.linalg.norm(vectors + v, axis=1) < 2 * tolerance]
        candidates.append(np.mean(np.vstack((near, -opposite)), axis=0))
        if len(candidates) == candidate_limit:
            break
    best = None
    for triplet in combinations(candidates, 3):
        basis = np.asarray(triplet)
        if np.linalg.svd(basis, compute_uv=False)[-1] <= 3 * tolerance:
            continue
        for iteration in range(5):
            fractional = points @ np.linalg.inv(basis)
            indices = np.rint(fractional)
            errors = np.linalg.norm(points - indices @ basis, axis=1)
            # Initial vector errors amplify at high Miller indices. First
            # refine a plausible fractional indexing before imposing q error.
            good = (np.max(abs(fractional - indices), axis=1) <= 0.2
                    if iteration < 2 else errors <= tolerance)
            if good.mean() < min_fraction or np.linalg.matrix_rank(indices[good]) < 3:
                break
            basis = np.linalg.lstsq(indices[good], points[good], rcond=None)[0]
        indices = np.rint(points @ np.linalg.inv(basis)).astype(int)
        errors = np.linalg.norm(points - indices @ basis, axis=1)
        good = errors <= tolerance
        if good.mean() < min_fraction:
            continue
        # Prefer coverage, then the largest cell consistent with those points;
        # avoids selecting arbitrarily dense superlattices with equal support.
        key = (int(good.sum()), abs(np.linalg.det(basis)), -float(np.mean(errors[good]**2)))
        if best is None or key > best[0]:
            best = key, basis, indices, errors, good
    if best is None:
        raise ValueError('No lattice fits the required fraction; check calibration, matches, and tolerance')
    _, basis, indices, errors, good = best
    # Integer row operations shorten the reported basis without changing its
    # lattice or the already assigned reciprocal points.
    for _ in range(30):
        order = np.argsort(np.linalg.norm(basis, axis=1))
        basis, indices = basis[order], indices[:, order]
        changed = False
        for i in range(1, 3):
            for j in range(i - 1, -1, -1):
                multiple = int(np.rint(np.dot(basis[i], basis[j]) / np.dot(basis[j], basis[j])))
                if multiple:
                    basis[i] -= multiple * basis[j]
                    indices[:, j] += multiple * indices[:, i]
                    changed = True
        if not changed:
            break
    return dict(basis=basis, hkl=indices, residuals=errors, inliers=good)


def reconstruct_points(uv_points, calibrations, *, tolerance=0.003,
                       min_matches=4, max_trials=4000, seed=0,
                       candidate_limit=32, min_fraction=0.9):
    """Reconstruct using only detected coordinates and instrument calibration.

    All images must connect to image 0 through verified overlaps. Rotations are
    propagated along a maximum-score spanning tree. Disconnected batches fail
    explicitly. This initial estimator does not perform global bundle adjustment.
    """
    if tolerance <= 0 or min_matches < 3 or max_trials < 1:
        raise ValueError('Require positive tolerance/trials and at least three matches')
    if len(uv_points) < 2 or len(uv_points) != len(calibrations):
        raise ValueError('Provide at least two images and one calibration per image')
    clouds = []
    for uv, calibration in zip(uv_points, calibrations):
        uv = np.asarray(uv, float)
        if uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all():
            raise ValueError('Each coordinate array must be finite with shape (M,2)')
        q = calibration.lift(uv)
        clouds.append(q[np.linalg.norm(q, axis=1) > 3 * tolerance])
    links = []
    for i, j in combinations(range(len(clouds)), 2):
        match = match_views(clouds[i], clouds[j], tolerance=tolerance,
                            min_matches=min_matches, max_trials=max_trials, seed=seed + i * len(clouds) + j)
        if match is not None:
            links.append(dict(i=i, j=j, **match))
    rotations = {0: np.eye(3)}
    tree = []
    for _ in range(len(clouds) - 1):
        eligible = [e for e in links if (e['i'] in rotations) != (e['j'] in rotations)]
        if not eligible:
            raise ValueError(f'Disconnected overlap graph; unregistered images: {sorted(set(range(len(clouds))) - rotations.keys())}')
        e = max(eligible, key=lambda e: e['score'])
        i, j, r = e['i'], e['j'], e['rotation']
        if j in rotations:
            rotations[i] = r @ rotations[j]
        else:
            rotations[j] = r.T @ rotations[i]
        tree.append(e)
    rotations = np.stack([rotations[i] for i in range(len(clouds))])
    points = np.vstack([q @ r for q, r in zip(clouds, rotations)])
    fit = fit_basis(points, tolerance=tolerance, candidate_limit=candidate_limit, min_fraction=min_fraction)
    # Standard active column rotation Rz(azimuth) Ry(elevation) Rx(roll).
    angles = Rotation.from_matrix(rotations.transpose(0, 2, 1)).as_euler('ZYX', degrees=True)
    return dict(**fit, points=points, rotations=rotations,
                azimuth_elevation_roll=angles, links=links, tree=tree,
                image_ids=np.repeat(np.arange(len(clouds)), [len(q) for q in clouds]))


def reconstruct_images(images, calibrations, *, peak_options=None, **kwargs):
    """Entry point for a batch of grayscale arrays; see reconstruct_points."""
    uv = [extract_spots(im, **(peak_options or {})) for im in images]
    return dict(reconstruct_points(uv, calibrations, **kwargs), uv_points=uv)
