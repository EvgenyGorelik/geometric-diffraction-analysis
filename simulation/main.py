import numpy as np
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser

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
    is_linear_combination
)


def main():
    # Parse command-line arguments
    parser = ArgumentParser(description="Simulate diffraction patterns and analyze zonal samples.")
    parser.add_argument("--target_volume", type=float, default=500, help="Target unit cell volume in Å³.")
    parser.add_argument("--volume_error", type=float, default=100, help="Volume tolerance in Å³.")
    parser.add_argument("--a_range", type=float, nargs=2, default=[3.5, 15], help="Range for parameter a in Å.")
    parser.add_argument("--b_range", type=float, nargs=2, default=[4, 30], help="Range for parameter b in Å.")
    parser.add_argument("--c_range", type=float, nargs=2, default=[5, 45], help="Range for parameter c in Å.")
    parser.add_argument("--alpha_range", type=float, nargs=2, default=[90, 120], help="Range for alpha in degrees.")
    parser.add_argument("--beta_range", type=float, nargs=2, default=[90, 120], help="Range for beta in degrees.")
    parser.add_argument("--gamma_range", type=float, nargs=2, default=[90, 120], help="Range for gamma in degrees.")
    parser.add_argument("--anisotropy", type=float, default=1.5, help="Anisotropy factor.")
    parser.add_argument("--resolution", type=float, default=0.85, help="Resolution in Å.")
    parser.add_argument("--wavelength", type=float, default=0.0407, help="Wavelength in Å.")
    parser.add_argument("--excitation_error", type=float, default=0.015, help="Excitation error.")
    parser.add_argument("--detector_distance", type=float, default=0, help="Detector distance.")
    parser.add_argument("--edge", type=float, default=1, help="Edge length for plotting.")
    parser.add_argument("--fraction_limit_missing", type=float, default=0.03, help="Fraction limit for missing reflections.")
    parser.add_argument("--fraction_limit_additional", type=float, default=0.05, help="Fraction limit for additional reflections.")
    parser.add_argument("--noise_level", type=float, default=0.0012, help="Noise level.")
    parser.add_argument("--max_shift", type=float, default=0.01, help="Maximum shift for reflections.")
    parser.add_argument("--N", type=int, default=1000, help="Number of orientations.")
    parser.add_argument("--zonal_sample_threshold", type=float, default=0.9, help="Threshold for zonal sample detection.")

    args = parser.parse_args()

    # Update parameters with parsed arguments
    target_volume = args.target_volume
    volume_error = args.volume_error
    a_range = args.a_range
    b_range = args.b_range
    c_range = args.c_range
    alpha_range = args.alpha_range
    beta_range = args.beta_range
    gamma_range = args.gamma_range
    anisotropy = args.anisotropy
    resolution = args.resolution
    wavelength = args.wavelength
    excitation_error = args.excitation_error
    detector_distance = args.detector_distance
    edge = args.edge
    fraction_limit_missing = args.fraction_limit_missing
    fraction_limit_additional = args.fraction_limit_additional
    noise_level = args.noise_level
    max_shift = args.max_shift
    N = args.N
    # if more than that percentage of reflections are linearly dependend, the pattern is considered zonal
    ZONAL_SAMPLE_THRESHOLD = args.zonal_sample_threshold

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


    # Compute reciprocal lattice parameters
    astar, bstar, cstar, alphastar, betastar, gammastar = lattice_parameters_reciprocal(a, b, c, alpha, beta, gamma)
    v1, v2, v3 = reciprocal_lattice_vectors(astar, bstar, cstar, alphastar, betastar, gammastar)

    astar_rot, bstar_rot, cstar_rot = random_rotate_vectors(v1, v2, v3)

    # Generate reciprocal lattice reflections
    hkl_list, reflections = generate_reciprocal_lattice(v1, v2, v3, resolution)

    orientations = fibonacci_orientations(N)
    indices = orientations_to_indices(orientations, astar_rot, bstar_rot, cstar_rot)
    # Parameters
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


    # Create figure
    F1, ax = plt.subplots(figsize=(5, 5))

    zonal_list = []

    for i in range(orientations.shape[0]):
        print(f"***** Pattern {i+1} out of {orientations.shape[0]}")

        rotated_reflections = align_reciprocal_lattice(reflections, orientations[i, :])
        ewald_sphere, diffraction_spots, hkl_included = simulate_diffraction_experiment(hkl_list, rotated_reflections, wavelength, excitation_error)
        

        # Calculate a spanning linear space from hkl points 0, 1, 2 and calculate how many points, can be linearly represented using this spanning tree
        # If the relative percentage is high enough in relation to the number of reflections, the sample is considered a 2D Zone
        # TODO: make sure to take three points which are not in itself linearly dependend
        lin_comb_sum = []
        for hkl_test in range(3, len(hkl_included)):
            lin_comb_sum.append(is_linear_combination(hkl_included[0], hkl_included[1], hkl_included[2], hkl_included[hkl_test]))
        is_zonal = False
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


if __name__ == "__main__":
    main()