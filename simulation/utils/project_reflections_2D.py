import numpy as np

def project_reflections_2D(diffraction_spots, detector_distance, wavelength):
    """
    Projects diffraction spots onto a 2D detector plane.
    
    Parameters:
        diffraction_spots (np.ndarray): Nx3 array of diffraction spot coordinates
        detector_distance (float): Distance of the detector from the sample
        wavelength (float): Wavelength of the incident beam
    
    Returns:
        tuple: 
            - projected_x (np.ndarray): X-coordinates of projected reflections
            - projected_y (np.ndarray): Y-coordinates of projected reflections
    """
    if len(diffraction_spots) == 0:
        print("Warning: No diffraction spots to project.")
        return np.array([]), np.array([])
    
    # Define Ewald sphere radius
    R = 1 / wavelength
    
    projected_x = []
    projected_y = []
    
    for G in diffraction_spots:
        # Compute scaling factor for projection onto detector plane
        t = (-detector_distance - R) / (G[2] - R)
        
        # Check if reflection is in front of the detector
        if t > 0 and np.isfinite(t):
            projected_x.append(G[0] * t)
            projected_y.append(G[1] * t)
    
    if not projected_x:
        print("Warning: No valid projected reflections onto the detector.")
        return np.array([]), np.array([])
    
    return np.array(projected_x), np.array(projected_y)
