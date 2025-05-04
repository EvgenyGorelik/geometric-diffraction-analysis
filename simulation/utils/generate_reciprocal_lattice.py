import numpy as np

def generate_reciprocal_lattice(v1, v2, v3, resolution):
    """
    Generates reciprocal lattice points within a given resolution sphere.
    
    Parameters:
        v1, v2, v3 (array-like): Reciprocal lattice basis vectors (length-3 arrays or lists)
        resolution (float): Maximum resolution (Å)
    
    Returns:
        tuple: 
            - hkl_list (NumPy array): List of Miller indices [h, k, l]
            - reflections (NumPy array): List of reciprocal lattice vectors [Gx, Gy, Gz]
    """
    # Calculate reciprocal lattice parameters from vectors
    astar = np.linalg.norm(v1)
    bstar = np.linalg.norm(v2)
    cstar = np.linalg.norm(v3)
    
    # Define resolution sphere
    sphere_radius = 1 / resolution
    max_index = int(np.ceil(sphere_radius / min(astar, bstar, cstar)))  # Fix max index
    
    # Store reflections
    hkl_list = []
    reflections = []
    
    # Generate reciprocal lattice reflections
    for h in range(-max_index, max_index + 1):
        for k in range(-max_index, max_index + 1):
            for l in range(-max_index, max_index + 1):
                G = h * np.array(v1) + k * np.array(v2) + l * np.array(v3)
                G_norm = np.linalg.norm(G)
                if 0 < G_norm <= sphere_radius:
                    hkl_list.append([h, k, l])
                    reflections.append(G)
    
    # Convert lists to NumPy arrays
    hkl_list = np.array(hkl_list)
    reflections = np.array(reflections)
        
    return hkl_list, reflections
