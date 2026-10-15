"""Configurable, deterministic rotation experiments with numerical detector images."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from simulation.utils.lattice_parameters_reciprocal import lattice_parameters_reciprocal
from simulation.utils.reciprocal_lattice_vectors import reciprocal_lattice_vectors
from simulation.utils.simulate_diffraction_experiment import ewald_association


def load_config(path):
    """Read JSON or simple key=value cell files (no executable expressions)."""
    path = Path(path)
    text = path.read_text()
    if text.lstrip().startswith('{'):
        config = json.loads(text)
    else:
        config = {}
        for line_number, line in enumerate(text.splitlines(), 1):
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            if '=' not in line:
                raise ValueError(f'{path}:{line_number}: expected key=value')
            key, value = map(str.strip, line.split('=', 1))
            key = 'wavelength' if key in ('lambda', 'lambda / wavelength') else key
            try:
                config[key] = json.loads(value)
            except json.JSONDecodeError:
                config[key] = value.strip('\"\'')
    if 'lambda' in config:
        config.setdefault('wavelength', config.pop('lambda'))
    return config


def reciprocal_basis(config):
    values = np.array([config[k] for k in ('a', 'b', 'c', 'alpha', 'beta', 'gamma')], dtype=float)
    if not np.isfinite(values).all() or np.any(values[:3] <= 0) or np.any(values[3:] <= 0) or np.any(values[3:] >= 180):
        raise ValueError('cell lengths must be positive; angles must lie between 0 and 180 degrees')
    ca, cb, cg = np.cos(np.deg2rad(values[3:]))
    if 1 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg <= 1e-12:
        raise ValueError('cell has zero or invalid volume')
    return np.array(reciprocal_lattice_vectors(*lattice_parameters_reciprocal(*values)))


def load_hkl(path, duplicate_policy='error'):
    """Read signed h k l I [extra columns]. Ignore 000 terminator; no symmetry expansion."""
    if duplicate_policy not in ('error', 'mean', 'sum', 'first'):
        raise ValueError('duplicate_policy must be error, mean, sum or first')
    records = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.split('#', 1)[0].split('!', 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        try:
            indices = np.array([float(v) for v in fields[:3]])
            intensity = float(fields[3])
            if len(indices) != 3 or not np.isfinite(indices).all() or np.any(indices != np.rint(indices)) or not np.isfinite(intensity):
                raise ValueError()
            key = tuple(int(v) for v in indices)
        except (ValueError, IndexError):
            raise ValueError(f'{path}:{number}: expected integer h k l and finite intensity') from None
        if key == (0, 0, 0):
            continue
        if key in records and duplicate_policy == 'error':
            raise ValueError(f'{path}:{number}: duplicate reflection {key}; select duplicate_policy explicitly')
        records.setdefault(key, []).append(intensity)
    if not records:
        raise ValueError('HKL file contains no reflections')
    reducer = {'error': lambda v: v[0], 'first': lambda v: v[0], 'mean': np.mean, 'sum': np.sum}[duplicate_policy]
    return np.array(list(records), dtype=int), np.array([reducer(v) for v in records.values()], dtype=float)


def generate_hkl(basis, resolution):
    """Enumerate the complete resolution ellipsoid, including skew cells."""
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError('resolution must be positive')
    bounds = np.ceil(np.linalg.norm(np.linalg.inv(basis), axis=0) / resolution).astype(int)
    # Slice h to keep intermediate arrays small for elongated cells.
    kl = np.array(np.meshgrid(np.arange(-bounds[1], bounds[1]+1),
                             np.arange(-bounds[2], bounds[2]+1), indexing='ij')).reshape(2, -1).T
    rows = []
    for h in range(-bounds[0], bounds[0]+1):
        indices = np.column_stack((np.full(len(kl), h), kl))
        norms = np.linalg.norm(indices @ basis, axis=1)
        rows.append(indices[(norms > 0) & (norms <= 1/resolution)])
    return np.concatenate(rows)


def scan_angles(start, stop, step):
    if not np.isfinite([start, stop, step]).all() or step == 0 or (stop-start)*step < 0:
        raise ValueError('angle step must be nonzero and point from start to stop')
    count = int(np.floor((stop-start)/step + 1e-10)) + 1
    return start + np.arange(count) * step


def project_to_detector(spots, wavelength, detector_distance):
    """Repository detector convention; return coordinates and source-row indices."""
    radius = 1 / wavelength
    denominator = spots[:, 2] - radius
    with np.errstate(divide='ignore', invalid='ignore'):
        scale = (-detector_distance-radius) / denominator
    valid = np.isfinite(scale) & (scale > 0)
    return spots[valid, :2] * scale[valid, None], np.flatnonzero(valid)


def rasterize(xy, intensities, size, edge):
    """Nearest pixel impulses; coincident spots add. No scaling or clipping."""
    image = np.zeros((size, size), dtype=np.float64)
    valid = np.isfinite(xy).all(axis=1) & (np.abs(xy) <= edge).all(axis=1)
    pixels = np.full((len(xy), 2), -1, dtype=int)
    pixels[valid, 0] = np.rint((xy[valid, 0]+edge)/(2*edge)*(size-1)).astype(int)
    pixels[valid, 1] = np.rint((edge-xy[valid, 1])/(2*edge)*(size-1)).astype(int)
    np.add.at(image, (pixels[valid, 1], pixels[valid, 0]), intensities[valid])
    return image, pixels, valid


def run_experiment(config, output, *, hkl_path=None, intensity_mode=None):
    config = dict(config)
    basis = reciprocal_basis(config)
    wavelength = float(config.get('wavelength', 0.0251))
    resolution = float(config.get('resolution', 0.85))
    mode = intensity_mode or config.get('intensity_mode', 'constant')
    hkl_path = hkl_path or config.get('hkl_file')
    if mode not in ('constant', 'hkl'):
        raise ValueError('intensity_mode must be constant or hkl')
    if mode == 'hkl' and not hkl_path:
        raise ValueError('hkl intensity mode requires an HKL file')
    if hkl_path:
        hkl, intensities = load_hkl(hkl_path, config.get('duplicate_policy', 'error'))
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError('resolution must be positive')
        keep = np.linalg.norm(hkl @ basis, axis=1) <= 1/resolution
        hkl, intensities = hkl[keep], intensities[keep]
    else:
        hkl = generate_hkl(basis, resolution)
        intensities = np.ones(len(hkl))
    if mode == 'constant':
        intensities = np.full(len(hkl), float(config.get('constant_intensity', 1000)))
    if not np.isfinite(intensities).all():
        raise ValueError('intensities must be finite')
    axis = np.asarray(config.get('rotation_axis', [0, 1, 0]), dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) == 0:
        raise ValueError('rotation_axis must be a finite nonzero 3-vector in laboratory coordinates')
    axis /= np.linalg.norm(axis)
    angles = scan_angles(float(config.get('start_deg', 0)), float(config.get('stop_deg', 180)), float(config.get('step_deg', 1)))
    initial = np.asarray(config.get('initial_euler_deg', [0, 0, 0]), dtype=float)
    if initial.shape != (3,) or not np.isfinite(initial).all():
        raise ValueError('initial_euler_deg must contain three finite xyz Euler angles')
    reflections = Rotation.from_euler('xyz', initial, degrees=True).apply(hkl @ basis)
    tolerance = float(config.get('excitation_error', 0.015))
    slope = float(config.get('excitation_error_slope', 0))
    reference_hkl = np.asarray(config.get('excitation_reference_hkl', [0, 0, 0]), dtype=float)
    if reference_hkl.shape != (3,) or not np.isfinite(reference_hkl).all():
        raise ValueError('excitation_reference_hkl must contain three finite values')
    reference_radius = float(np.linalg.norm(reference_hkl @ basis))
    ewald_association(reflections, wavelength, tolerance, slope, reference_radius)  # validate before output
    size = config.get('image_size', 1024)
    edge = float(config.get('edge', 1.5))
    distance = float(config.get('detector_distance', 0))
    if not isinstance(size, int) or isinstance(size, bool) or size < 2 or not np.isfinite(edge) or edge <= 0 or not np.isfinite(distance) or distance < 0:
        raise ValueError('image_size must be integer >=2; edge positive; detector_distance nonnegative')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('output directory must be empty to avoid mixing experiments')
    config.update(intensity_mode=mode, hkl_file=str(Path(hkl_path).resolve()) if hkl_path else None,
                  rotation_axis=axis.tolist())
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    frames = []
    for frame, angle in enumerate(angles):
        rotated = Rotation.from_rotvec(axis*np.deg2rad(angle)).apply(reflections)
        mask, errors, thresholds = ewald_association(rotated, wavelength, tolerance, slope, reference_radius)
        selected = np.flatnonzero(mask)
        xy, projected = project_to_detector(rotated[mask], wavelength, distance)
        selected = selected[projected]
        image, pixels, valid = rasterize(xy, intensities[selected], size, edge)
        stem = f'frame_{frame:05d}'
        np.save(output/f'{stem}.npy', image)
        # Float TIFF keeps high dynamic range; NPY is the authoritative float64 image.
        Image.fromarray(image.astype(np.float32)).save(output/f'{stem}.tif')
        with (output/f'{stem}.csv').open('w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(['h', 'k', 'l', 'intensity', 'x', 'y', 'column', 'row', 'sphere_error', 'threshold'])
            for j in np.flatnonzero(valid):
                idx = selected[j]
                writer.writerow([*hkl[idx], intensities[idx], *xy[j], *pixels[j], errors[idx], thresholds[idx]])
        frames.append({'frame': frame, 'angle_deg': float(angle), 'reflections_on_detector': int(valid.sum()), 'image': f'{stem}.npy'})
    (output/'frames.json').write_text(json.dumps(frames, indent=2)+'\n')
    return frames


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', '--output_folder', dest='output', default='rotation_output')
    parser.add_argument('--hkl-file')
    parser.add_argument('--intensity-mode', choices=['constant', 'hkl'])
    parser.add_argument('--axis', type=float, nargs=3)
    parser.add_argument('--start-deg', type=float)
    parser.add_argument('--stop-deg', type=float)
    parser.add_argument('--step-deg', type=float)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    for name in ('start_deg', 'stop_deg', 'step_deg'):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    if args.axis is not None:
        config['rotation_axis'] = args.axis
    if config.get('hkl_file'):
        config['hkl_file'] = str(Path(args.config).resolve().parent / config['hkl_file'])
    frames = run_experiment(config, args.output, hkl_path=args.hkl_file, intensity_mode=args.intensity_mode)
    print(f'Saved {len(frames)} frames to {args.output}')


if __name__ == '__main__':
    main()
