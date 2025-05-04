import numpy as np
import matplotlib.pyplot as plt
import os

# Placeholder for your custom functions
from utils import (
    generate_unit_cell_anisotropy, 
    lattice_parameters_reciprocal,
    reciprocal_lattice_vectors, 
    random_rotate_vectors,
    generate_reciprocal_lattice,
    fibonacci_orientations,
    orientations_to_indices,
    align_reciprocal_lattice,
    simulate_diffraction_experiment,
    project_reflections_2D,
    remove_random_pairs,
    add_random_pairs,
    add_noise,
    shift_points,
    save_parameters,
    is_linear_combination,
    find_linear_independent
)

# if more than that percentage of reflections are linearly dependend, the pattern is considered zonal
ZONAL_SAMPLE_THRESHOLD = 0.9

# Define target volume and parameter ranges
target_volume = 500  # Å³
volume_error = 100  # Volume tolerance (Å³)
a_range = [3.5, 15]  # Å
b_range = [4, 30]  # Å
c_range = [5, 45]  # Å
alpha_range = [90, 120]  # degrees
beta_range = [90, 120]  # degrees
gamma_range = [90, 120]  # degrees

anisotropy = 1.5

# Generate unit cell parameters
a, b, c, alpha, beta, gamma = generate_unit_cell_anisotropy(
    target_volume, a_range, b_range, c_range, alpha_range, beta_range, gamma_range, volume_error, anisotropy
)

# Round the parameters
a = round(a, 3)
b = round(b, 3)
c = round(c, 3)
alpha = round(alpha, 2)
beta = round(beta, 2)
gamma = round(gamma, 2)

# Display results
print(f"Generated unit cell parameters:")
print(f"a = {a:.3f} Å")
print(f"b = {b:.3f} Å")
print(f"c = {c:.3f} Å")
print(f"alpha = {alpha:.2f}°")
print(f"beta = {beta:.2f}°")
print(f"gamma = {gamma:.2f}°")

resolution = 0.85  # Resolution in Å

# Compute reciprocal lattice parameters
astar, bstar, cstar, alphastar, betastar, gammastar = lattice_parameters_reciprocal(a, b, c, alpha, beta, gamma)
v1, v2, v3 = reciprocal_lattice_vectors(astar, bstar, cstar, alphastar, betastar, gammastar)

astar_rot, bstar_rot, cstar_rot = random_rotate_vectors(v1, v2, v3)

# Generate reciprocal lattice reflections
hkl_list, reflections = generate_reciprocal_lattice(v1, v2, v3, resolution)

N = 1000  # Number of orientations
orientations = fibonacci_orientations(N)
indices = orientations_to_indices(orientations, astar_rot, bstar_rot, cstar_rot)
# Parameters
wavelength = 0.0407  # 90 kV
R = 1 / wavelength

# Compute Ewald sphere
theta, phi = np.meshgrid(np.linspace(0, np.pi, 50), np.linspace(0, 2 * np.pi, 100))
X = R * np.sin(theta) * np.cos(phi)
Y = R * np.sin(theta) * np.sin(phi)
Z = R * np.cos(theta)

# Shift the sphere to the Ewald center
X = X + 0
Y = Y + 0
Z = Z + R

excitation_error = 0.015
detector_distance = 0  # Example detector distance
edge = 1
fraction_limit_missing = 0.03
fraction_limit_additional = 0.05
noise_level = 0.0012
max_shift = 0.01

# Create figure
F1, ax = plt.subplots(figsize=(5, 5))

zonal_list = []

for i in range(orientations.shape[0]):
    print(f"***** Pattern {i+1} out of {orientations.shape[0]}")

    rotated_reflections = align_reciprocal_lattice(reflections, orientations[i, :])
    ewald_sphere, diffraction_spots, hkl_included = simulate_diffraction_experiment(hkl_list, rotated_reflections, wavelength, excitation_error)
    
    
    # Calculate a spanning linear space from linear independent hkl combinations and calculate how many points, can be linearly represented using this spanning tree
    lin_indep_vectors = find_linear_independent(hkl_included)
    is_zonal = False
    if lin_indep_vectors is not None:
        lin_comb_sum = []
        for hkl_test in range(3, len(hkl_included)):
            lin_comb_sum.append(is_linear_combination(lin_indep_vectors[0], lin_indep_vectors[1], lin_indep_vectors[2], hkl_included[hkl_test]))
        if sum(lin_comb_sum) > ZONAL_SAMPLE_THRESHOLD * len(hkl_included):
            is_zonal = True

    projected_x, projected_y = project_reflections_2D(diffraction_spots, detector_distance, wavelength)

    x_reduced, y_reduced = remove_random_pairs(projected_x, projected_y, fraction_limit_missing)
    x_augmented, y_augmented = add_random_pairs(x_reduced, y_reduced, fraction_limit_additional)
    x_noisy, y_noisy = add_noise(x_augmented, y_augmented, noise_level)
    x_shifted, y_shifted, shift_vector = shift_points(x_noisy, y_noisy, max_shift)

    ax.scatter(x_shifted, y_shifted, 10, color='k', marker='o')
    ax.scatter(shift_vector[0], shift_vector[1], 30, color='k', marker='o')

    ax.grid(False)
    ax.axis('equal')
    ax.set_xlim([-edge, edge])
    ax.set_ylim([-edge, edge])
    ax.axis('off')

    # Save the figure
    filename = f"volume_{target_volume}/CELL_{a}_{b}_{c}_{alpha}_{beta}_{gamma}_WL_{wavelength}_ExcErr_{excitation_error}_RES_{resolution}/DIFF_{a}_{b}_{c}_{alpha}_{beta}_{gamma}_ori_{indices[i, 0]}_{indices[i, 1]}_{indices[i, 2]}.jpg"
    if is_zonal:
        zonal_list.append(filename)
    # Create the directory if it doesn't exist
    filepath = os.path.dirname(filename)
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    F1.savefig(filename)
    ax.clear()

# Save parameters to a text file
params_filename = f"volume_{target_volume}/CELL_{a}_{b}_{c}_{alpha}_{beta}_{gamma}_WL_{wavelength}_ExcErr_{excitation_error}_RES_{resolution}_params.txt"
save_parameters(params_filename, a, b, c, alpha, beta, gamma, astar, bstar, cstar, alphastar, betastar, gammastar, v1, v2, v3, target_volume, wavelength, resolution, excitation_error, noise_level, max_shift, fraction_limit_missing, fraction_limit_additional)

zonal_filename = f"volume_{target_volume}/CELL_{a}_{b}_{c}_{alpha}_{beta}_{gamma}_WL_{wavelength}_ExcErr_{excitation_error}_RES_{resolution}_2DZones.txt"
with open(zonal_filename, 'w+') as fp:
    fp.write('\n'.join(zonal_list))


print("DONE")
