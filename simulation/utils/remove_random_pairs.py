import numpy as np

def remove_random_pairs(x, y, fraction_limit):
    """
    Randomly removes a fraction of coordinate pairs from the given lists.
    
    Parameters:
        x (np.ndarray): Array of x coordinates.
        y (np.ndarray): Array of y coordinates.
        fraction_limit (float): Maximum fraction of coordinate pairs to remove (between 0 and 1).
    
    Returns:
        tuple:
            - x_reduced (np.ndarray): Reduced array of x coordinates.
            - y_reduced (np.ndarray): Reduced array of y coordinates.
    """
    # Randomly choose a fraction that does not exceed the limit
    fraction = np.random.rand() * fraction_limit
    
    # Number of coordinate pairs
    num_pairs = len(x)
    
    # Calculate the number of pairs to remove
    num_to_remove = round(fraction * num_pairs)
    
    # Generate random indices to remove
    remove_indices = np.random.choice(num_pairs, num_to_remove, replace=False)
    
    # Remove the selected pairs
    mask = np.ones(num_pairs, dtype=bool)
    mask[remove_indices] = False
    
    x_reduced = x[mask]
    y_reduced = y[mask]
    
    return x_reduced, y_reduced