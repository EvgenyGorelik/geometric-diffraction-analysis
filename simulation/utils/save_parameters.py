def save_parameters(filename, a, b, c, alpha, beta, gamma,
                    astar, bstar, cstar, alphastar, betastar, gammastar,
                    v1, v2, v3, target_volume, wavelength, resolution, excitation_error,
                    noise_level, max_shift, fraction_limit_missing, fraction_limit_additional):
    """
    Saves crystallographic parameters to a file.

    Parameters:
        filename (str): Name of the output file.
        a, b, c (float): Unit cell lengths in Å.
        alpha, beta, gamma (float): Unit cell angles in degrees.
        astar, bstar, cstar (float): Reciprocal lattice lengths in Å⁻¹.
        alphastar, betastar, gammastar (float): Reciprocal lattice angles in degrees.
        v1, v2, v3 (tuple): Reciprocal lattice vectors.
        target_volume (float): Target volume in Å³.
        wavelength (float): Wavelength in Å.
        resolution (float): Resolution in Å.
        excitation_error (float): Excitation error.
        noise_level (float): Noise level.
        max_shift (float): Maximum shift.
        fraction_limit_missing (float): Maximum fraction of missing data.
        fraction_limit_additional (float): Maximum fraction of additional data.
    """

    with open(filename, 'w') as f:
        # Write the unit cell parameters
        f.write("Unit Cell Parameters:\n")
        f.write(f"a = {a:.3f} Å\n")
        f.write(f"b = {b:.3f} Å\n")
        f.write(f"c = {c:.3f} Å\n")
        f.write(f"alpha = {alpha:.2f}°\n")
        f.write(f"beta = {beta:.2f}°\n")
        f.write(f"gamma = {gamma:.2f}°\n\n")

        # Write the reciprocal lattice parameters
        f.write("Reciprocal Lattice Parameters:\n")
        f.write(f"astar = {astar:.3f} Å⁻¹\n")
        f.write(f"bstar = {bstar:.3f} Å⁻¹\n")
        f.write(f"cstar = {cstar:.3f} Å⁻¹\n")
        f.write(f"alphastar = {alphastar:.2f}°\n")
        f.write(f"betastar = {betastar:.2f}°\n")
        f.write(f"gammastar = {gammastar:.2f}°\n\n")

        # Write the reciprocal lattice vectors
        f.write("Reciprocal Lattice Vectors:\n")
        f.write(f"v1 = [{v1[0]:.4f}, {v1[1]:.4f}, {v1[2]:.4f}]\n")
        f.write(f"v2 = [{v2[0]:.4f}, {v2[1]:.4f}, {v2[2]:.4f}]\n")
        f.write(f"v3 = [{v3[0]:.4f}, {v3[1]:.4f}, {v3[2]:.4f}]\n\n")

        # Write additional parameters
        f.write(f"Target volume: {target_volume:.0f} Å³\n")
        f.write(f"Wavelength: {wavelength:.4f} Å\n")
        f.write(f"Resolution: {resolution:.2f} Å\n")
        f.write(f"Excitation Error: {excitation_error:.4f}\n")
        f.write(f"Noise Level: {noise_level:.4f}\n")
        f.write(f"Max Shift: {max_shift:.4f}\n")
        f.write(f"Missing Fraction Limit: {fraction_limit_missing:.4f}\n")
        f.write(f"Additional Fraction Limit: {fraction_limit_additional:.4f}\n")

    print(f"Parameters saved to {filename}")
