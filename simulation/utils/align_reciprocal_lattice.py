import numpy as np

def align_reciprocal_lattice(reflections, target_direction):
    """
    Rotates reciprocal lattice reflections so that a given vector aligns with the z-axis.
    
    Parameters:
        reflections (np.ndarray): Nx3 array of reciprocal lattice points
        target_direction (array-like): 1x3 vector defining the desired orientation along the z-axis
    
    Returns:
        np.ndarray: Nx3 array of rotated reflections
    """
    # Normalize target direction
    target_direction = np.array(target_direction) / np.linalg.norm(target_direction)
    
    # Define the z-axis unit vector
    z_axis = np.array([0, 0, 1])
    
    # Compute rotation axis (cross product)
    rotation_axis = np.cross(target_direction, z_axis)
    
    # Compute rotation angle (dot product)
    cos_theta = np.dot(target_direction, z_axis)
    tilt_angle = np.arccos(cos_theta)  # Angle in radians
    
    # Handle the case where the target direction is already aligned with the z-axis
    if np.linalg.norm(rotation_axis) < 1e-6:
        return reflections
    
    # Normalize rotation axis
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    ux, uy, uz = rotation_axis
    
    # Rodrigues' rotation formula
    cos_t = np.cos(tilt_angle)
    sin_t = np.sin(tilt_angle)
    one_minus_cos_t = 1 - cos_t
    
    R = np.array([
        [cos_t + ux**2 * one_minus_cos_t, ux * uy * one_minus_cos_t - uz * sin_t, ux * uz * one_minus_cos_t + uy * sin_t],
        [uy * ux * one_minus_cos_t + uz * sin_t, cos_t + uy**2 * one_minus_cos_t, uy * uz * one_minus_cos_t - ux * sin_t],
        [uz * ux * one_minus_cos_t - uy * sin_t, uz * uy * one_minus_cos_t + ux * sin_t, cos_t + uz**2 * one_minus_cos_t]
    ])
    
    # Apply rotation to reflections
    rotated_reflections = reflections @ R.T  # Matrix multiplication
    
    return rotated_reflections
