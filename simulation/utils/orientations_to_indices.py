import numpy as np

def orientations_to_indices(orientations, astar_rot, bstar_rot, cstar_rot):
    """
    Converts orientation vectors into indices relative to the rotated basis vectors.
    
    Parameters:
        orientations (np.ndarray): Nx3 matrix where each row is an orientation vector [x, y, z]
        astar_rot, bstar_rot, cstar_rot (array-like): Rotated basis vectors
    
    Returns:
        np.ndarray: Nx3 matrix where each row contains the indices [h, k, l] representing the orientation
    """
    # Construct the matrix of rotated basis vectors
    B = np.column_stack((astar_rot, bstar_rot, cstar_rot))
    
    # Solve for indices using matrix inversion
    indices = np.linalg.solve(B, orientations.T).T
    
    # Round to 4 decimal places
    indices = np.round(indices, 4)
    
    return indices
