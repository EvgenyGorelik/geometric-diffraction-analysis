import numpy as np

def add_noise(x, y, noise_level):
    """
    Adds a small random noise to the given coordinate pairs.
    
    Parameters:
        x (numpy array): Column vector of x coordinates.
        y (numpy array): Column vector of y coordinates.
        noise_level (float): Standard deviation of the Gaussian noise to add to each coordinate.
    
    Returns:
        x_noisy (numpy array): Noisy x coordinates.
        y_noisy (numpy array): Noisy y coordinates.
    
    Example usage:
        x_noisy, y_noisy = add_noise(x, y, 0.01)
    """
    x_noisy = x + noise_level * np.random.randn(*x.shape)
    y_noisy = y + noise_level * np.random.randn(*y.shape)
    
    return x_noisy, y_noisy
