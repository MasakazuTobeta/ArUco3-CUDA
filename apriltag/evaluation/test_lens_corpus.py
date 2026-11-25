# SPDX-License-Identifier: Apache-2.0
"""Independent projection, sampling and curvature checks for the wide-angle corpus."""
import unittest

import cv2
import numpy as np

from lens_corpus import CODEBOOK, DARK, EXTERNAL_LENSES, LENSES, UNIT, WHITE, Camera, draw, grids_from_codebook, pose


class LensCorpusTest(unittest.TestCase):
    def test_forward_against_opencv_and_inverse_ray(self):
        rng = np.random.default_rng(1371)
        points = rng.uniform(-2, 2, (48, 3))
        points[:, 2] = rng.uniform(.3, 2, 48)
        zero = np.zeros(3)
        for model, fov in LENSES:
            camera = Camera(640, 360, model, fov)
            f = camera.focal
            cx, cy = camera.principal
            intrinsics = np.array([[f, 0., cx], [0., f, cy], [0., 0., 1.]])
            if model == 'equidistant':
                oracle, _ = cv2.fisheye.projectPoints(points[:, None], zero, zero, intrinsics, np.zeros(4))
            else:
                oracle, _ = cv2.projectPoints(points, zero, zero, intrinsics, np.zeros(5))
            xy = camera.project(points)
            np.testing.assert_allclose(xy, oracle.reshape(-1, 2), atol=1e-10, rtol=0)
            np.testing.assert_allclose(np.stack(camera.rays(*xy.T), axis=1),
                                       points / np.linalg.norm(points, axis=1)[:, None], atol=1e-12)
            # Include the sensor axis and corners, even where 165° horizontal gives >180° diagonal.
            pixels = np.array([camera.principal, [-.5, -.5], [639.5, 359.5], [639.5, cy]])
            directions = np.stack(camera.rays(*pixels.T), axis=1)
            np.testing.assert_allclose(camera.project(directions), pixels, atol=1e-10, rtol=0)

    def test_independent_azimuthal_models(self):
        angles = np.array([0., .25, .75, 1.25, 1.61])
        for model, fov in EXTERNAL_LENSES:
            camera = Camera(640, 360, model, fov)
            theta = np.deg2rad(fov / 2)
            # Use chord length or geometric stereographic intersection to verify
            # the radial map independently of the generator's trigonometric form.
            def radial(a):
                if model == 'equisolid':
                    return np.sqrt(2 * (1 - np.cos(a)))
                if model == 'stereographic':
                    return 2 * np.sin(a) / (1 + np.cos(a))
                return a
            rays = np.stack((np.sin(angles), 0 * angles, np.cos(angles)), axis=1)
            pixels = camera.project(rays)
            np.testing.assert_allclose(pixels[:, 0] - camera.principal[0],
                                       320 * radial(angles) / radial(theta), atol=1e-11)
            np.testing.assert_allclose(np.stack(camera.rays(*pixels.T), axis=1), rays, atol=1e-12)

    def test_curvature_and_raster(self):
        grid = grids_from_codebook(CODEBOOK)[37]
        maxima = []
        for model, fov in LENSES:
            camera = Camera(480, 270, model, fov)
            target = pose(camera, 37, 1, 5, -.31, (.007, -.012), False)
            maxima.append(max(target.label(camera)['edge_max_sag_px']))
            cell = (np.arange(8) + .5) / 8 - .5
            uv = np.stack(np.meshgrid(cell, cell), axis=-1)
            xy = np.rint(camera.project(target.world(uv))).astype(int)
            for samples in (1, 3):
                patch = np.full((270, 480), 151, np.uint8)
                full = patch.copy()
                draw(patch, camera, target, grid, samples)
                draw(full, camera, target, grid, samples, full_sensor=True)
                np.testing.assert_array_equal(patch, full)
                np.testing.assert_array_equal(patch[xy[..., 1], xy[..., 0]],
                                              np.where(grid != 0, WHITE, DARK))
                if samples == 1:
                    self.assertEqual(set(np.unique(patch)), {DARK, WHITE, 151})
                else:
                    self.assertGreater(len(np.unique(patch)), 6)
            # The maximum displacement of the true curved edge from its straight
            # corner-to-corner chords is measured by an independent homogeneous line fit.
            edges = target.edges(camera, samples=101)
            corners = camera.project(target.world(UNIT))
            homogeneous = np.column_stack([corners, np.ones(4)])
            lines = np.cross(homogeneous, np.roll(homogeneous, -1, axis=0))
            distances = np.abs((edges * lines[:, None, :2]).sum(axis=2) + lines[:, None, 2])
            measured = (distances / np.linalg.norm(lines[:, :2], axis=1)[:, None]).max()
            self.assertAlmostEqual(measured, maxima[-1], delta=.01)
        self.assertLess(maxima[0], 1e-9)
        self.assertGreater(maxima[1], .5)
        self.assertGreater(maxima[2], maxima[1] * 1.6)


if __name__ == '__main__':
    unittest.main()
