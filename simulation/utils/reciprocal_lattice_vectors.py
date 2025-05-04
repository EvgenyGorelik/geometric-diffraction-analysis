import numpy as np

def reciprocal_lattice_vectors(astar, bstar, cstar, alphastar, betastar, gammastar):
    """
    Generates reciprocal lattice basis vectors from reciprocal lattice parameters.
    
    Parameters:
        astar, bstar, cstar (float): Reciprocal lattice parameters (Å⁻¹)
        alphastar, betastar, gammastar (float): Reciprocal lattice angles (degrees)
    
    Returns:
        tuple: Three reciprocal lattice basis vectors as 3x1 NumPy arrays
    """
    # Convert angles from degrees to radians
    alpha_rad = np.radians(alphastar)
    beta_rad = np.radians(betastar)
    gamma_rad = np.radians(gammastar)
    
    # Compute reciprocal lattice basis vectors
    v1 = np.array([astar, 0, 0])
    v2 = np.array([bstar * np.cos(gamma_rad), bstar * np.sin(gamma_rad), 0])
    
    # Calculate the components of v3
    v3x = cstar * np.cos(beta_rad)
    v3y = cstar * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad)
    v3z = np.sqrt(max(0, cstar**2 - v3x**2 - v3y**2))  # Ensure non-negative value
    v3 = np.array([v3x, v3y, v3z])
    
    return v1, v2, v3
