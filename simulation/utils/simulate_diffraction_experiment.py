import numpy as np


def ewald_association(reflections, wavelength, tolerance, tolerance_slope=0.0, reference_radius=0.0):
    """Return mask, radial sphere error and threshold, all in inverse Å.

    threshold(G) = tolerance + tolerance_slope * max(0, ||G|| - reference_radius). The slope is
    dimensionless. Reciprocal vectors follow the repository's 1/d convention.
    """
    reflections = np.asarray(reflections, dtype=float).reshape(-1, 3)
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError('wavelength must be finite and positive')
    if not np.isfinite([tolerance, tolerance_slope, reference_radius]).all() or min(tolerance, tolerance_slope, reference_radius) < 0:
        raise ValueError('Ewald tolerance and slope must be finite and nonnegative')
    if not np.isfinite(reflections).all():
        raise ValueError('reflections must be finite')
    radius = 1 / wavelength
    errors = np.abs(np.linalg.norm(reflections - [0, 0, radius], axis=1) - radius)
    thresholds = tolerance + tolerance_slope * np.maximum(0, np.linalg.norm(reflections, axis=1) - reference_radius)
    return errors < thresholds, errors, thresholds


def simulate_diffraction_experiment(hkl_list, reflections, wavelength, tolerance,
                                    tolerance_slope=0.0, *, include_sphere=True):
    """Associate reflections with the Ewald sphere; retain the legacy return tuple.

    Set include_sphere=False for scans to avoid constructing a visualization mesh.
    tolerance_slope=0 preserves the original static association criterion.
    """
    reflections = np.asarray(reflections, dtype=float).reshape(-1, 3)
    hkl_list = np.asarray(hkl_list).reshape(-1, 3)
    if len(hkl_list) != len(reflections):
        raise ValueError('hkl_list and reflections must have equal lengths')
    mask, _, _ = ewald_association(reflections, wavelength, tolerance, tolerance_slope)
    sphere = np.empty((0, 3))
    if include_sphere:
        radius = 1 / wavelength
        theta, phi = np.meshgrid(np.linspace(0, np.pi, 200), np.linspace(0, 2 * np.pi, 400))
        sphere = radius * np.column_stack((
            (np.sin(theta) * np.cos(phi)).ravel(),
            (np.sin(theta) * np.sin(phi)).ravel(), np.cos(theta).ravel()))
        sphere += [0, 0, radius]
    return sphere, reflections[mask], hkl_list[mask]
