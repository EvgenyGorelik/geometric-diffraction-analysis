import numpy as np

def fibonacci_orientations(N):
    """
    Generates N approximately equidistant orientations on a sphere using the Fibonacci lattice method.
    
    Parameters:
        N (int): Number of orientations to generate
    
    Returns:
        np.ndarray: Nx3 matrix where each row is a unit vector [x, y, z]
    """
    # Golden ratio for optimal spacing
    phi = (1 + np.sqrt(5)) / 2
    
    # Compute spherical coordinates
    theta = np.arccos(1 - 2 * (np.arange(1, N + 1) / (N + 1)))  # Elevation angle
    azimuth = 2 * np.pi * np.arange(1, N + 1) / phi             # Azimuthal angle
    
    # Convert to Cartesian coordinates (unit sphere)
    x = np.sin(theta) * np.cos(azimuth)
    y = np.sin(theta) * np.sin(azimuth)
    z = np.cos(theta)
    
    # Store as Nx3 matrix
    orientations = np.column_stack((x, y, z))
    
    return orientations
