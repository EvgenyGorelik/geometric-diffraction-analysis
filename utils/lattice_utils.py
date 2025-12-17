"""
Utility functions for creating 3D lattice vectors from crystallographic parameters.
This module provides functions to create lattice basis vectors from standard
crystallographic parameters and can be easily integrated into the existing 
geometric-diffraction-analysis codebase.
"""

import numpy as np


def lattice_vectors_from_parameters(length_a, length_b, length_c, 
                                  angle_ab, angle_bc, angle_ac):
    """
    Create 3D lattice basis vectors from crystallographic parameters.
    
    This function creates lattice vectors using standard crystallographic
    conventions. The parameter names match your request exactly.
    
    Parameters:
    -----------
    length_a, length_b, length_c : float
        Unit cell edge lengths (Å)
    angle_ab : float  
        Angle between vectors a and b in degrees (γ in standard notation)
    angle_bc : float
        Angle between vectors b and c in degrees (α in standard notation) 
    angle_ac : float
        Angle between vectors a and c in degrees (β in standard notation)
        
    Returns:
    --------
    tuple of numpy arrays
        Three 3D vectors (a, b, c) representing the lattice basis vectors
        
    Example:
    --------
    >>> length_a, length_b, length_c = 5.0, 6.0, 7.0
    >>> angle_ab, angle_bc, angle_ac = 90.0, 90.0, 90.0
    >>> a, b, c = lattice_vectors_from_parameters(length_a, length_b, length_c,
    ...                                          angle_ab, angle_bc, angle_ac)
    >>> print(f"a = {a}")
    >>> print(f"b = {b}")  
    >>> print(f"c = {c}")
    a = [5. 0. 0.]
    b = [0. 6. 0.]
    c = [0. 0. 7.]
    """
    
    # Convert angles from degrees to radians
    alpha = np.radians(angle_bc)  # angle between b and c
    beta = np.radians(angle_ac)   # angle between a and c  
    gamma = np.radians(angle_ab)  # angle between a and b
    
    # Pre-compute trigonometric values
    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)
    
    # Check if angles form a valid unit cell
    volume_factor = (1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 
                    2*cos_alpha*cos_beta*cos_gamma)
    
    if volume_factor <= 1e-10:
        raise ValueError(f"Invalid angle combination: angles cannot form a valid unit cell "
                        f"(volume factor = {volume_factor:.2e})")
    
    # Construct lattice vectors following crystallographic conventions
    
    # Vector a: along x-axis
    a = np.array([length_a, 0.0, 0.0])
    
    # Vector b: in xy-plane at angle gamma from a
    b = np.array([
        length_b * cos_gamma,
        length_b * sin_gamma, 
        0.0
    ])
    
    # Vector c: positioned to satisfy all angle constraints
    cx = length_c * cos_beta
    
    # Calculate cy using the constraint that angle(b,c) = alpha
    # From dot product: cos(alpha) = (b·c)/(|b||c|)
    if abs(sin_gamma) < 1e-10:
        raise ValueError("Vectors a and b are collinear (angle_ab ≈ 0° or 180°)")
        
    cy = (cos_alpha * length_c - cos_gamma * cos_beta * length_c) / sin_gamma
    
    # Calculate cz from the constraint |c| = length_c
    cz_squared = length_c**2 - cx**2 - cy**2
    
    if cz_squared < -1e-10:  # Allow small negative values due to floating point errors
        raise ValueError(f"Cannot construct vector c: inconsistent constraints "
                        f"(cz² = {cz_squared:.2e})")
    
    cz = np.sqrt(max(0, cz_squared))  # Ensure non-negative
    
    c = np.array([cx, cy, cz])
    
    return a, b, c


def create_unit_cell_lattice(a, b, c, n_cells=1):
    """
    Generate lattice points within a specified number of unit cells.
    
    Parameters:
    -----------
    a, b, c : numpy arrays
        Lattice basis vectors
    n_cells : int
        Number of unit cells in each direction (default=1 gives 3x3x3 grid)
        
    Returns:
    --------
    numpy array
        Array of lattice points with shape (N, 3)
    """
    points = []
    
    for i in range(-n_cells, n_cells + 1):
        for j in range(-n_cells, n_cells + 1):
            for k in range(-n_cells, n_cells + 1):
                point = i * a + j * b + k * c
                points.append(point)
    
    return np.array(points)


def verify_lattice_vectors(a, b, c, expected_lengths, expected_angles, tolerance=1e-6):
    """
    Verify that lattice vectors have the expected lengths and angles.
    
    Parameters:
    -----------
    a, b, c : numpy arrays
        Lattice basis vectors to verify
    expected_lengths : tuple
        Expected lengths (length_a, length_b, length_c)
    expected_angles : tuple
        Expected angles in degrees (angle_ab, angle_bc, angle_ac)
    tolerance : float
        Tolerance for verification (default=1e-6)
        
    Returns:
    --------
    bool
        True if all parameters are within tolerance
    dict
        Detailed verification results
    """
    
    def angle_between_vectors(v1, v2):
        """Calculate angle between vectors in degrees."""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    # Calculate actual values
    actual_lengths = (np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c))
    actual_angles = (
        angle_between_vectors(a, b),  # angle_ab (γ)
        angle_between_vectors(b, c),  # angle_bc (α)
        angle_between_vectors(a, c)   # angle_ac (β)
    )
    
    # Calculate differences
    length_diffs = tuple(abs(e - a) for e, a in zip(expected_lengths, actual_lengths))
    angle_diffs = tuple(abs(e - a) for e, a in zip(expected_angles, actual_angles))
    
    # Check if within tolerance
    lengths_ok = all(d <= tolerance for d in length_diffs)
    angles_ok = all(d <= tolerance for d in angle_diffs)
    
    results = {
        'passed': lengths_ok and angles_ok,
        'lengths': {
            'expected': expected_lengths,
            'actual': actual_lengths,
            'differences': length_diffs,
            'passed': lengths_ok
        },
        'angles': {
            'expected': expected_angles,
            'actual': actual_angles, 
            'differences': angle_diffs,
            'passed': angles_ok
        }
    }
    
    return results['passed'], results


# Integration with existing codebase functions

def get_reciprocal_lattice_from_direct(a, b, c):
    """
    Calculate reciprocal lattice vectors from direct lattice vectors.
    This can be used with your existing reciprocal lattice functions.
    
    Parameters:
    -----------
    a, b, c : numpy arrays
        Direct lattice basis vectors
        
    Returns:
    --------
    tuple of numpy arrays
        Reciprocal lattice vectors (a*, b*, c*)
    """
    # Calculate volume of unit cell
    V = np.dot(a, np.cross(b, c))
    
    if abs(V) < 1e-10:
        raise ValueError("Unit cell volume is zero - vectors are coplanar")
    
    # Calculate reciprocal vectors
    a_star = 2 * np.pi * np.cross(b, c) / V
    b_star = 2 * np.pi * np.cross(c, a) / V  
    c_star = 2 * np.pi * np.cross(a, b) / V
    
    return a_star, b_star, c_star


def get_lattice_parameters_from_vectors(a, b, c):
    """
    Extract lattice parameters from lattice vectors.
    Useful for checking or converting between representations.
    
    Parameters:
    -----------
    a, b, c : numpy arrays
        Lattice basis vectors
        
    Returns:
    --------
    tuple
        (length_a, length_b, length_c, angle_ab, angle_bc, angle_ac) in Å and degrees
    """
    def angle_between_vectors(v1, v2):
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    length_a = np.linalg.norm(a)
    length_b = np.linalg.norm(b)  
    length_c = np.linalg.norm(c)
    
    angle_ab = angle_between_vectors(a, b)  # γ
    angle_bc = angle_between_vectors(b, c)  # α
    angle_ac = angle_between_vectors(a, c)  # β
    
    return length_a, length_b, length_c, angle_ab, angle_bc, angle_ac


if __name__ == "__main__":
    # Quick test of the main function
    print("Testing lattice vector creation:")
    print("=" * 40)
    
    # Test case: monoclinic crystal
    length_a, length_b, length_c = 8.0, 9.0, 12.0
    angle_ab, angle_bc, angle_ac = 90.0, 90.0, 110.0
    
    print(f"Input: a={length_a}, b={length_b}, c={length_c} Å")
    print(f"       α={angle_bc}°, β={angle_ac}°, γ={angle_ab}°")
    
    try:
        a, b, c = lattice_vectors_from_parameters(
            length_a, length_b, length_c,
            angle_ab, angle_bc, angle_ac
        )
        
        print(f"\nLattice vectors:")
        print(f"a = [{a[0]:7.3f}, {a[1]:7.3f}, {a[2]:7.3f}]")
        print(f"b = [{b[0]:7.3f}, {b[1]:7.3f}, {b[2]:7.3f}]")
        print(f"c = [{c[0]:7.3f}, {c[1]:7.3f}, {c[2]:7.3f}]")
        
        # Verify the result
        passed, results = verify_lattice_vectors(
            a, b, c,
            (length_a, length_b, length_c),
            (angle_ab, angle_bc, angle_ac)
        )
        
        print(f"\nVerification: {'PASSED' if passed else 'FAILED'}")
        if not passed:
            print(f"Length errors: {results['lengths']['differences']}")
            print(f"Angle errors: {results['angles']['differences']}")
        
        # Test reciprocal lattice calculation
        a_star, b_star, c_star = get_reciprocal_lattice_from_direct(a, b, c)
        print(f"\nReciprocal lattice vectors:")
        print(f"a* = [{a_star[0]:7.4f}, {a_star[1]:7.4f}, {a_star[2]:7.4f}]")
        print(f"b* = [{b_star[0]:7.4f}, {b_star[1]:7.4f}, {b_star[2]:7.4f}]")
        print(f"c* = [{c_star[0]:7.4f}, {c_star[1]:7.4f}, {c_star[2]:7.4f}]")
        
    except ValueError as e:
        print(f"Error: {e}")