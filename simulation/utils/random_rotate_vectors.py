import numpy as np

def random_rotate_vectors(v1, v2, v3):
    """
    Randomly rotate a set of vectors (v1, v2, v3) while preserving their mutual relationship.
    
    Parameters:
        v1, v2, v3 (array-like): Vectors to be rotated (length-3 arrays or lists)
    
    Returns:
        tuple: Rotated vectors as NumPy arrays
    """
    # Generate random Euler angles (in radians)
    theta = 2 * np.pi * np.random.rand()  # Rotation around z-axis
    phi = 2 * np.pi * np.random.rand()    # Rotation around x-axis
    psi = 2 * np.pi * np.random.rand()    # Rotation around y-axis
    
    # Rotation matrices
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta),  np.cos(theta), 0],
                   [0,              0,             1]])  # Rotation around z-axis

    Rx = np.array([[1, 0,            0],
                   [0, np.cos(phi), -np.sin(phi)],
                   [0, np.sin(phi),  np.cos(phi)]]) # Rotation around x-axis

    Ry = np.array([[np.cos(psi), 0, np.sin(psi)],
                   [0,          1, 0],
                   [-np.sin(psi), 0, np.cos(psi)]]) # Rotation around y-axis
    
    # Combined rotation matrix
    R = Rz @ Ry @ Rx  # Matrix multiplication
    
    # Rotate vectors
    v1_rot = R @ np.asarray(v1)
    v2_rot = R @ np.asarray(v2)
    v3_rot = R @ np.asarray(v3)
    
    return v1_rot, v2_rot, v3_rot