import numpy as np

def generate_unit_cell_anisotropy(volume, a_range, b_range, c_range, alpha_range, beta_range, gamma_range, tolerance, anisotropy_limit):
    """
    Generates unit cell parameters (a, b, c, alpha, beta, gamma) within specified ranges
    that satisfy a given unit cell volume and an anisotropy constraint.
    
    Args:
    - volume (float): Target unit cell volume (Å³)
    - a_range, b_range, c_range (list): [min, max] for a, b, c (Å)
    - alpha_range, beta_range, gamma_range (list): [min, max] for angles (degrees)
    - tolerance (float): Allowed deviation from target volume
    - anisotropy_limit (float): Maximum allowed anisotropy (max(a, b, c)/min(a, b, c))
    
    Returns:
    - a, b, c (float): Lattice parameters (Å)
    - alpha, beta, gamma (float): Lattice angles (degrees)
    """
    def rand_range(min_val, max_val):
        """Helper function to generate a random number in a given range."""
        return np.random.uniform(min_val, max_val)
    
    def unit_cell_volume(a, b, c, alpha, beta, gamma):
        """Compute the unit cell volume based on lattice parameters and angles."""
        # Convert angles to radians for the volume calculation
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        gamma_rad = np.radians(gamma)
        
        # Use the formula for the volume of a triclinic unit cell
        volume = (a * b * c) * np.sqrt(1 - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2 +
                                       2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad))
        return volume

    max_attempts = 10000
    found = False

    for attempt in range(max_attempts):
        # Generate lattice parameters a, b, c within the specified ranges
        a = rand_range(a_range[0], a_range[1])
        b = rand_range(b_range[0], b_range[1])
        c = rand_range(c_range[0], c_range[1])

        # Ensure anisotropy constraint is met
        anisotropy = max(a, b, c) / min(a, b, c)
        if anisotropy > anisotropy_limit:
            continue
        
        # Generate angles alpha, beta, gamma within the specified ranges
        alpha = rand_range(alpha_range[0], alpha_range[1])
        beta = rand_range(beta_range[0], beta_range[1])
        gamma = rand_range(gamma_range[0], gamma_range[1])
        
        # Compute the volume of the unit cell
        calculated_volume = unit_cell_volume(a, b, c, alpha, beta, gamma)
        
        # Check if the computed volume is close to the target volume within tolerance
        if abs(calculated_volume - volume) < tolerance:
            found = True
            break

    if not found:
        raise ValueError('Could not generate anisotropic unit cell parameters within the constraints.')

    return a, b, c, alpha, beta, gamma

# Example usage:
volume = 500
a_range = [3.5, 15]
b_range = [4, 30]
c_range = [5, 45]
alpha_range = [90, 120]
beta_range = [90, 120]
gamma_range = [90, 120]
tolerance = 100
anisotropy_limit = 1.5

a, b, c, alpha, beta, gamma = generate_unit_cell_anisotropy(volume, a_range, b_range, c_range, alpha_range, beta_range, gamma_range, tolerance, anisotropy_limit)

print(f"Generated unit cell parameters:")
print(f"a = {a:.3f} Å")
print(f"b = {b:.3f} Å")
print(f"c = {c:.3f} Å")
print(f"alpha = {alpha:.2f}°")
print(f"beta = {beta:.2f}°")
print(f"gamma = {gamma:.2f}°")
