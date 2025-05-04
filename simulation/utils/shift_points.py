import numpy as np

def shift_points(x, y, max_shift):
    """
    Apply a small random shift to the entire set of points.

    Parameters:
        x (np.array): Column vector of x coordinates.
        y (np.array): Column vector of y coordinates.
        max_shift (float): Maximum allowed shift for both x and y directions.

    Returns:
        tuple: (x_shifted, y_shifted, shift_vector)
            - x_shifted (np.array): Shifted column vector of x coordinates.
            - y_shifted (np.array): Shifted column vector of y coordinates.
            - shift_vector (tuple): (shift_x, shift_y) describing the applied shift.
    """
    # Generate random shifts within the range [-max_shift, max_shift]
    shift_x = (np.random.rand() * 2 - 1) * max_shift
    shift_y = (np.random.rand() * 2 - 1) * max_shift
    
    # Apply the shift to all coordinates
    x_shifted = x + shift_x
    y_shifted = y + shift_y
    
    # Return the shift vector as well
    shift_vector = (shift_x, shift_y)
    
    return x_shifted, y_shifted, shift_vector