import numpy as np

def lattice_parameters_reciprocal(a, b, c, alpha, beta, gamma):
    """
    Compute the reciprocal lattice parameters and angles.
    
    Args:
    - a, b, c (float): Lattice parameters (Å)
    - alpha, beta, gamma (float): Angles (degrees)
    
    Returns:
    - astar, bstar, cstar (float): Reciprocal lattice parameters
    - alphastar, betastar, gammastar (float): Reciprocal angles (degrees)
    """
    # Convert angles from degrees to radians
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)
    
    # Compute the volume of the unit cell
    V = a * b * c * np.sqrt(1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 +
                            2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma))
    
    # Compute reciprocal lattice parameters
    astar = (b * c * np.sin(alpha)) / V
    bstar = (a * c * np.sin(beta)) / V
    cstar = (a * b * np.sin(gamma)) / V
    
    # Compute reciprocal angles (in radians)
    alphastar = np.degrees(np.arccos((np.cos(beta) * np.cos(gamma) - np.cos(alpha)) / 
                                     (np.sin(beta) * np.sin(gamma))))
    betastar = np.degrees(np.arccos((np.cos(alpha) * np.cos(gamma) - np.cos(beta)) / 
                                    (np.sin(alpha) * np.sin(gamma))))
    gammastar = np.degrees(np.arccos((np.cos(alpha) * np.cos(beta) - np.cos(gamma)) / 
                                      (np.sin(alpha) * np.sin(beta))))
    
    return astar, bstar, cstar, alphastar, betastar, gammastar

# Example usage:
a, b, c = 5.0, 6.0, 7.0
alpha, beta, gamma = 90.0, 90.0, 90.0

astar, bstar, cstar, alphastar, betastar, gammastar = lattice_parameters_reciprocal(a, b, c, alpha, beta, gamma)

print(f"Reciprocal lattice parameters:")
print(f"astar = {astar:.3f}")
print(f"bstar = {bstar:.3f}")
print(f"cstar = {cstar:.3f}")
print(f"alphastar = {alphastar:.3f}°")
print(f"betastar = {betastar:.3f}°")
print(f"gammastar = {gammastar:.3f}°")
