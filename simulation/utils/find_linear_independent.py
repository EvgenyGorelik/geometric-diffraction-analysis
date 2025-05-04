import numpy as np

def find_linear_independent(hkl):

    idx_grid = np.stack(np.meshgrid(np.arange(len(hkl)), np.arange(len(hkl)), np.arange(len(hkl)))).reshape(3, -1).T

    for i, j, k in idx_grid:
        # Extract the three points
        point1 = hkl[i]
        point2 = hkl[j]
        point3 = hkl[k]

        # Form a matrix with the points as rows
        matrix = np.array([point1, point2, point3])

        # Compute the determinant
        determinant = np.linalg.det(matrix)

        # Define vectors a and b
        a = np.array(point2 - point1)  # Replace with your vector a
        b = np.array(point3 - point1)  # Replace with your vector b

        # Compute the cross product
        cross_product = np.cross(a, b)
        lin_dep = np.allclose(cross_product, 0)
        # Check if the cross product is zero
        if not lin_dep:
            return [point1, point2, point3]
    return None