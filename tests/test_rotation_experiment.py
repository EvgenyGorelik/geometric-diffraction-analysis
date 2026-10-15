import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from simulation.rotation_experiment import (load_config, reciprocal_basis, load_hkl,
    scan_angles, generate_hkl, rasterize, project_to_detector, run_experiment)
from simulation.utils.simulate_diffraction_experiment import ewald_association, simulate_diffraction_experiment

CELL = dict(a=10, b=10, c=10, alpha=90, beta=90, gamma=90, wavelength=.0251)


class RotationTests(unittest.TestCase):
    def test_scan_interval(self):
        np.testing.assert_allclose(scan_angles(0, 1, .3), [0, .3, .6, .9])
        np.testing.assert_allclose(scan_angles(1, 0, -.5), [1, .5, 0])
        with self.assertRaises(ValueError):
            scan_angles(0, 1, -1)

    def test_rotation_geometry(self):
        point = Rotation.from_rotvec([0, np.pi/2, 0]).apply([[1, 0, 0]])
        np.testing.assert_allclose(point, [[0, 0, -1]], atol=1e-15)
        xy, indices = project_to_detector(np.array([[1, 2, 0], [1, 0, 50]]), .0251, 0)
        np.testing.assert_allclose(xy, [[1, 2]])
        np.testing.assert_equal(indices, [0])

    def test_dynamic_and_static_association(self):
        g = np.array([[.1, 0, .01], [1, 0, .04]])
        static, _, _ = ewald_association(g, .0251, .015)
        dynamic, _, thresholds = ewald_association(g, .0251, .015, .05)
        self.assertEqual(static.tolist(), [True, False])
        self.assertEqual(dynamic.tolist(), [True, True])
        np.testing.assert_allclose(thresholds, .015+.05*np.linalg.norm(g, axis=1))
        empty = simulate_diffraction_experiment([[1, 0, 0]], [[0, 0, 1]], .0251, 0, include_sphere=False)
        self.assertEqual(empty[1].shape, (0, 3))

    def test_threshold_example_111_333(self):
        basis = reciprocal_basis(CELL)
        radius111 = np.linalg.norm(np.array([1, 1, 1]) @ basis)
        g = np.array([[1, 1, 1], [3, 3, 3]]) @ basis
        _, _, thresholds = ewald_association(g, .0251, .01, .04/(2*radius111), radius111)
        np.testing.assert_allclose(thresholds, [.01, .05])

    def test_intensities_and_collisions(self):
        xy = np.array([[0, 0], [.5, 0], [.5, 0], [2, 0]])
        image, _, valid = rasterize(xy, np.array([1000, 4.5, 2, 8]), 5, 1)
        self.assertEqual(image[2, 2], 1000)
        self.assertEqual(image[2, 3], 6.5)
        self.assertEqual(image.sum(), 1006.5)
        self.assertFalse(valid[-1])

    def test_config_hkl_and_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg = root/'cell.cfg'
            cfg.write_text('a=9.68934\nb=8.41686\nc=35.6672\nalpha=90\nbeta=89.2448\ngamma=90\nlambda / wavelength=0.0251\n')
            config = load_config(cfg)
            self.assertEqual(config['beta'], 89.2448)
            self.assertTrue(np.isfinite(reciprocal_basis(config)).all())
            file = root/'test.hkl'
            file.write_text('1 0 0 1000 0\n0 1 0 200.5 0\n0 0 0 0 0\n')
            hkl, intensity = load_hkl(file)
            np.testing.assert_equal(intensity, [1000, 200.5])
            cfg = dict(CELL, start_deg=0, stop_deg=90, step_deg=90, image_size=33, edge=1,
                       excitation_error=.01, intensity_mode='hkl')
            frames = run_experiment(cfg, root/'raw', hkl_path=file)
            image = np.load(root/'raw/frame_00000.npy')
            self.assertEqual(image.max(), 1000)
            self.assertEqual(image.sum(), 1200.5)
            self.assertEqual(len(frames), 2)
            self.assertEqual(frames[1]['reflections_on_detector'], 1)
            self.assertEqual(json.loads((root/'raw/config.json').read_text())['intensity_mode'], 'hkl')
            run_experiment(cfg, root/'constant', hkl_path=file, intensity_mode='constant')
            self.assertEqual(np.load(root/'constant/frame_00000.npy').sum(), 2000)
            with self.assertRaises(ValueError):
                run_experiment(cfg, root/'raw', hkl_path=file)
            file.write_text('1 0 0 5\n1 0 0 9\n')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                load_hkl(file)
            self.assertEqual(load_hkl(file, 'mean')[1][0], 7)

    def test_skew_cell_complete_enumeration(self):
        basis = reciprocal_basis(dict(CELL, gamma=25))
        generated = set(map(tuple, generate_hkl(basis, 2)))
        brute = { (h,k,l) for h in range(-7,8) for k in range(-7,8) for l in range(-7,8)
                 if 0 < np.linalg.norm(np.array([h,k,l]) @ basis) <= .5 }
        self.assertEqual(generated, brute)

    def test_invalid_inputs(self):
        for cfg in (dict(CELL, a=0), dict(CELL, gamma=180), dict(CELL, alpha=10, beta=10, gamma=170)):
            with self.assertRaises(ValueError):
                reciprocal_basis(cfg)
        with self.assertRaises(ValueError):
            ewald_association([], 0, .01)


if __name__ == '__main__':
    unittest.main()
