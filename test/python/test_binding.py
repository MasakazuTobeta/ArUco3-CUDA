#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Python package that need no GPU.

Aruco3CudaConfig is mirrored by hand in python/aruco3cuda/_binding.py, so the two
tests in ConfigTest that guard the mirror carry most of the weight here. They
were checked by deliberately inserting a field into the ctypes struct alone:

  - test_config_defaults_match_the_header compares all 33 defaults against
    include/aruco3cuda/config.hpp. An insertion that moves the later fields makes
    it read the wrong bytes, and it failed on three fields when tried.
  - test_every_field_is_covered compares the declared field names against the two
    tables. This is the one that catches an insertion which shifts nothing:
    putting an int where the compiler had already left alignment padding leaves
    every offset unchanged, and only the name check noticed.

Neither alone is enough, which is why both are here.
"""
import unittest

import aruco3cuda


class VersionTest(unittest.TestCase):
    def test_version_is_a_dotted_string(self):
        version = aruco3cuda.version()
        self.assertRegex(version, r"^\d+\.\d+\.\d+$")
        self.assertEqual(version, aruco3cuda.__version__)


class DictionaryTest(unittest.TestCase):
    def test_builtin_dictionary_is_listed(self):
        entries = aruco3cuda.dictionaries()
        self.assertGreaterEqual(len(entries), 1)
        names = [entry.name for entry in entries]
        self.assertIn("DICT_ARUCO_MIP_36h12", names)

    def test_metadata_matches_the_dictionary(self):
        info = aruco3cuda.find_dictionary("DICT_ARUCO_MIP_36h12")
        self.assertEqual(info.marker_size, 6)
        self.assertEqual(info.code_count, 250)
        self.assertEqual(info.max_correction_bits, 5)

    def test_unknown_dictionary_names_the_known_ones(self):
        with self.assertRaises(aruco3cuda.Error) as raised:
            aruco3cuda.find_dictionary("DICT_NOPE")
        self.assertEqual(raised.exception.status, aruco3cuda.Status.UNSUPPORTED_DICTIONARY)
        self.assertIn("DICT_ARUCO_MIP_36h12", str(raised.exception))


class MarkerBitsTest(unittest.TestCase):
    def test_shape_and_values(self):
        rows = aruco3cuda.marker_bits("DICT_ARUCO_MIP_36h12", 42)
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertEqual(len(row), 6)
            for value in row:
                self.assertIn(value, (0, 1))

    def test_rotation_one_is_the_pattern_turned_counterclockwise(self):
        # The header states that the rotation index advances counterclockwise.
        # Deriving rotation 1 from rotation 0 here checks both that claim and
        # the row-major unpacking, because getting either wrong breaks this.
        base = aruco3cuda.marker_bits("DICT_ARUCO_MIP_36h12", 42, 0)
        turned = aruco3cuda.marker_bits("DICT_ARUCO_MIP_36h12", 42, 1)
        size = len(base)
        expected = [[base[column][size - 1 - row] for column in range(size)] for row in range(size)]
        self.assertEqual(turned, expected)

    def test_four_rotations_return_to_the_start(self):
        rows = aruco3cuda.marker_bits("DICT_ARUCO_MIP_36h12", 7, 0)
        size = len(rows)
        turned = rows
        for _ in range(4):
            turned = [
                [turned[column][size - 1 - row] for column in range(size)] for row in range(size)
            ]
        self.assertEqual(turned, rows)

    def test_id_past_the_end_is_rejected(self):
        with self.assertRaises(aruco3cuda.Error) as raised:
            aruco3cuda.marker_bits("DICT_ARUCO_MIP_36h12", 250)
        self.assertEqual(raised.exception.status, aruco3cuda.Status.INVALID_ARGUMENT)

    def test_last_id_is_accepted(self):
        rows = aruco3cuda.marker_bits("DICT_ARUCO_MIP_36h12", 249)
        self.assertEqual(len(rows), 6)

    def test_rotation_out_of_range_is_rejected(self):
        with self.assertRaises(aruco3cuda.Error):
            aruco3cuda.marker_bits("DICT_ARUCO_MIP_36h12", 0, 4)


class ConfigTest(unittest.TestCase):
    # Every default in include/aruco3cuda/config.hpp. Kept in declaration order
    # so that a reader can diff the two side by side.
    kIntegerDefaults = {
        "adaptive_thresh_win_size_min_px": 3,
        "adaptive_thresh_win_size_max_px": 23,
        "adaptive_thresh_win_size_step_px": 10,
        "min_distance_to_border_px": 3,
        "marker_border_bits": 1,
        "perspective_remove_pixel_per_cell": 4,
        "corner_refine_method": 1,
        "corner_refinement_win_size_px": 5,
        "corner_refinement_max_iterations": 30,
        "use_aruco3_detection": 1,
        "min_side_length_canonical_img_px": 32,
        "max_candidates": 4096,
        "max_markers": 1024,
        "max_width_px": 3840,
        "max_height_px": 2160,
        "cuda_block_dim": 16,
    }
    kRealDefaults = {
        "adaptive_thresh_constant": 7.0,
        "min_marker_perimeter_rate": 0.03,
        "max_marker_perimeter_rate": 4.0,
        "polygonal_approx_accuracy_rate": 0.03,
        "min_corner_distance_rate": 0.05,
        "min_marker_distance_rate": 0.125,
        "min_group_distance": 0.21,
        "min_quad_inlier_ratio": 0.80,
        "min_edge_support_ratio": 2.0,
        "perspective_remove_ignored_margin_per_cell": 0.13,
        "max_erroneous_bits_in_border_rate": 0.35,
        "min_otsu_std_dev": 5.0,
        "error_correction_rate": 0.6,
        "valid_bit_threshold": 0.49,
        "relative_corner_refinement_win_size": 0.3,
        "corner_refinement_min_accuracy_px": 0.1,
        # Declared float rather than double, so it is compared loosely enough
        # for the narrowing.
        "min_marker_length_ratio_original_img": 0.05,
    }

    def test_config_defaults_match_the_header(self):
        config = aruco3cuda.Config()
        for name, expected in self.kIntegerDefaults.items():
            with self.subTest(field=name):
                self.assertEqual(getattr(config, name), expected)
        for name, expected in self.kRealDefaults.items():
            with self.subTest(field=name):
                self.assertAlmostEqual(getattr(config, name), expected, places=6)

    def test_every_field_is_covered(self):
        # Without this, a field added to the struct could be left out of the two
        # tables above and the layout check would quietly stop covering it.
        declared = {name for name, _ in aruco3cuda.Config._fields_}
        checked = set(self.kIntegerDefaults) | set(self.kRealDefaults)
        self.assertEqual(declared, checked)

    def test_overrides_are_applied(self):
        config = aruco3cuda.Config(max_width_px=1280, max_height_px=720)
        self.assertEqual(config.max_width_px, 1280)
        self.assertEqual(config.max_height_px, 720)
        self.assertEqual(config.max_markers, 1024)

    def test_unknown_field_is_rejected(self):
        with self.assertRaises(AttributeError):
            aruco3cuda.Config(no_such_field=1)

    def test_defaults_validate(self):
        aruco3cuda.Config().validate()

    def test_validate_names_the_offending_field(self):
        config = aruco3cuda.Config(adaptive_thresh_win_size_min_px=2)
        with self.assertRaises(aruco3cuda.Error) as raised:
            config.validate()
        self.assertEqual(raised.exception.status, aruco3cuda.Status.INVALID_CONFIG)
        self.assertIn("adaptive_thresh_win_size_min_px", str(raised.exception))

    def test_corner_refine_method_out_of_range_is_rejected(self):
        # The C++ type makes this unrepresentable, so the check has to happen at
        # the C boundary. Nothing else would catch it.
        config = aruco3cuda.Config(corner_refine_method=7)
        with self.assertRaises(aruco3cuda.Error) as raised:
            config.validate()
        self.assertIn("corner_refine_method", str(raised.exception))


class StatusTest(unittest.TestCase):
    def test_names_come_from_the_library(self):
        from aruco3cuda import _status_name

        self.assertEqual(_status_name(aruco3cuda.Status.OK), "kOk")
        self.assertEqual(_status_name(aruco3cuda.Status.CUDA_ERROR), "kCudaError")
        self.assertEqual(_status_name(aruco3cuda.Status.INTERNAL_ERROR), "kInternalError")


class DetectorLifetimeTest(unittest.TestCase):
    # Creating a detector calls no CUDA API, so these run without a device.

    def test_download_before_initialize_is_rejected(self):
        with aruco3cuda.Detector() as detector:
            with self.assertRaises(aruco3cuda.Error) as raised:
                detector.download()
            self.assertEqual(raised.exception.status, aruco3cuda.Status.NOT_INITIALIZED)

    def test_close_is_idempotent(self):
        detector = aruco3cuda.Detector()
        detector.close()
        detector.close()

    def test_host_buffer_is_rejected_with_guidance(self):
        with aruco3cuda.Detector() as detector:
            with self.assertRaises(TypeError) as raised:
                detector.detect(b"\x00" * 16)
            self.assertIn("DeviceImage.from_host", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
