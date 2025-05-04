import numpy as np

def is_linear_combination(a, b, c, d, tol=1e-10):
    """
    Check if d can be expressed as a linear combination of vectors (b-a) and (c-a).
    
    Parameters:
    a, b, c, d : array-like (1D arrays of length 2 or 3)
    tol : float (tolerance for numerical precision)
    
    Returns:
    bool : True if d can be expressed as a linear combination, otherwise False
    """
    # Compute spanning vectors
    v1 = np.array(b) - np.array(a)
    v2 = np.array(c) - np.array(a)
    target = np.array(d) - np.array(a)

    # Construct matrix M with columns [v1, v2]
    M = np.column_stack((v1, v2))

    try:
        # Solve for coefficients (lambda, mu)
        coeffs = np.linalg.lstsq(M, target, rcond=None)[0]
        
        # Check if the reconstructed vector matches d-a (within tolerance)
        return np.allclose(M @ coeffs, target, atol=tol)
    except np.linalg.LinAlgError:
        return False  # If singular matrix (collinear vectors), no solution

# Example usage
a = np.array([0, 0])
b = np.array([1, 0])
c = np.array([0, 1])
d = np.array([0.5, 0.5])  # Should return True

print(is_linear_combination(a, b, c, d))