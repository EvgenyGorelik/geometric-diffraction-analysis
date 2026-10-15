import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from utils.reconstruct_lattice import Calibration, extract_spots, match_views, fit_basis, reconstruct_points
from simulation.utils.project_reflections_2D import project_reflections_2D


class ReconstructionTests(unittest.TestCase):
    def test_projection_inverse(self):
        calibration = Calibration((250, 200), 0.001, 0.4, 2)
        uv = np.array([[300, 240], [120, 380], [250, 200.]])
        q = calibration.lift(uv)
        x, y = project_reflections_2D(q, 2, 0.4)
        np.testing.assert_allclose(np.column_stack((x, y)) / .001 + calibration.center, uv)

    def test_peaks(self):
        image = np.full((60, 80), 255.)
        image[10:13, 20:23] = 0
        image[40:43, 60:63] = 0
        np.testing.assert_allclose(extract_spots(image), [[21, 11], [61, 41]])
        self.assertEqual(extract_spots(np.ones((20, 20))).shape, (0, 2))

    def test_rotation_with_missing_and_false_spots(self):
        rng = np.random.default_rng(6)
        a = rng.normal(size=(20, 3))
        r = Rotation.from_euler('ZYX', [35, 27, -18], degrees=True).as_matrix()
        b = np.vstack((a[:15] @ r + rng.normal(scale=0.0001, size=(15, 3)), rng.normal(size=(4, 3))))
        result = match_views(a, b, tolerance=.002, max_trials=1000)
        self.assertGreaterEqual(len(result['pairs']), 15)
        np.testing.assert_allclose(result['rotation'], r, atol=.001)

    def test_basis_and_coplanar_rejection(self):
        rng = np.random.default_rng(3)
        basis = np.array([[.21, 0, 0], [.03, .27, 0], [.02, .04, .32]])
        hkl = np.array(np.meshgrid(*[np.arange(-2, 3)] * 3)).reshape(3, -1).T
        points = hkl @ basis + rng.normal(scale=.0001, size=hkl.shape)
        result = fit_basis(points, tolerance=.001)
        self.assertAlmostEqual(abs(np.linalg.det(result['basis'])), abs(np.linalg.det(basis)), places=5)
        self.assertTrue(result['inliers'].all())
        with self.assertRaisesRegex(ValueError, 'coplanar'):
            fit_basis(points[hkl[:, 2] == 0], tolerance=.001)

    def test_image_batch_end_to_end(self):
        from examples.reconstruct_lattice_demo import synthetic_batch
        from utils.reconstruct_lattice import reconstruct_images
        images, calibration, truth = synthetic_batch()
        result = reconstruct_images(images, calibration, tolerance=.003,
                                    max_trials=6000, peak_options={'threshold': 254})
        self.assertTrue(result['inliers'].all())
        self.assertLess(abs(abs(np.linalg.det(result['basis']) / np.linalg.det(truth)) - 1), .005)
        # Check lattice equivalence, not only equal cell volumes.
        transform = result['basis'] @ np.linalg.inv(truth)
        np.testing.assert_allclose(transform, np.rint(transform), atol=.01)
        self.assertAlmostEqual(abs(np.linalg.det(np.rint(transform))), 1.)
        np.testing.assert_allclose(np.linalg.norm(result['points'] - result['hkl'] @ result['basis'], axis=1),
                                   result['residuals'], atol=1e-10)

    def test_disconnected(self):
        with self.assertRaisesRegex(ValueError, 'Disconnected'):
            reconstruct_points([np.empty((0, 2))] * 2, [Calibration((0, 0), 1, 1)] * 2)


if __name__ == '__main__':
    unittest.main()
