import numpy as np

def simulate_diffraction_experiment(hkl_list, reflections, wavelength, tolerance):
    """
    Simulates a diffraction experiment by identifying reflections that lie on the Ewald sphere.
    
    Parameters:
        hkl_list (np.ndarray): List of Miller indices [h, k, l]
        reflections (np.ndarray): List of reciprocal lattice vectors [Gx, Gy, Gz]
        wavelength (float): Wavelength of the incident beam
        tolerance (float): Tolerance for identifying diffraction spots
    
    Returns:
        tuple: 
            - ewald_sphere (np.ndarray): Points on the Ewald sphere for visualization
            - diffraction_spots (np.ndarray): Identified diffraction spots
    """
    # Compute Ewald sphere radius
    R = 1 / wavelength
    
    # Define the Ewald sphere center
    ewald_center = np.array([0, 0, R])
    
    # Generate a mesh of points for visualization (optional)
    theta, phi = np.meshgrid(np.linspace(0, np.pi, 200), np.linspace(0, 2 * np.pi, 400))
    X = R * np.sin(theta) * np.cos(phi)
    Y = R * np.sin(theta) * np.sin(phi)
    Z = R * np.cos(theta)
    
    # Store sphere points for visualization
    ewald_sphere = np.column_stack((X.ravel(), Y.ravel(), Z.ravel())) + ewald_center
    
    # Identify diffraction spots (reflections close to the Ewald sphere surface)
    diffraction_spots = []
    hkl_included = []
    for i, G in enumerate(reflections):
        distance_to_sphere = abs(np.linalg.norm(G - ewald_center) - R)
        if distance_to_sphere < tolerance:  # Allow small tolerance for numerical accuracy
            diffraction_spots.append(G)
            hkl_included.append(hkl_list[i])
    
    diffraction_spots = np.array(diffraction_spots)
    hkl_included = np.array(hkl_included)
    
    # Display the number of observed reflections
    print(f'Number of diffraction spots: {len(diffraction_spots)}')
    
    return ewald_sphere, diffraction_spots, hkl_included
