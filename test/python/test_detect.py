#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Python package that need a GPU.

Kept apart from test_binding.py so that CI, which has no device, can run one
file and skip the other rather than having to filter inside a single file.

The marker is rendered here rather than read from a fixture, so the round trip
covers the same ground as the C++ one: a break in the bit order, the border
polarity, or the corner ordering shows up as a missing id.
"""
import unittest

import aruco3cuda

kDictionary = "DICT_ARUCO_MIP_36h12"
kWhite = 255
kBlack = 0


def render(marker_id, side_px, margin_px, border_bits=1):
    """Render a marker on a white canvas and return (side, pixels)."""
    bits = aruco3cuda.marker_bits(kDictionary, marker_id)
    marker_size = len(bits)
    cells = marker_size + 2 * border_bits
    canvas_side = side_px + 2 * margin_px
    pixels = bytearray([kWhite]) * (canvas_side * canvas_side)
    for row_px in range(side_px):
        inner_row = row_px * cells // side_px - border_bits
        base = (row_px + margin_px) * canvas_side + margin_px
        for col_px in range(side_px):
            inner_col = col_px * cells // side_px - border_bits
            value = kBlack
            if 0 <= inner_row < marker_size and 0 <= inner_col < marker_size:
                value = kWhite if bits[inner_row][inner_col] else kBlack
            pixels[base + col_px] = value
    return canvas_side, pixels


class DetectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.side, cls.pixels = render(42, 200, 40)

    def make_detector(self, **overrides):
        config = aruco3cuda.Config(
            max_width_px=self.side, max_height_px=self.side, **overrides
        )
        detector = aruco3cuda.Detector()
        detector.initialize(kDictionary, config)
        return detector, config

    def test_round_trip(self):
        detector, _ = self.make_detector()
        with detector:
            with aruco3cuda.DeviceImage.from_host(self.pixels, self.side, self.side) as image:
                with aruco3cuda.Stream() as stream:
                    detector.detect(image, stream)
                    result = detector.download(stream)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, 42)
        self.assertEqual(result.accepted_total, 1)
        self.assertFalse(result.truncated)
        # Drawn with the outer edge at 40 and side + 39, so every corner has to
        # land near one of those two.
        for x, y in result[0].corners:
            self.assertTrue(
                min(abs(x - 40.0), abs(x - 239.0)) < 1.0, f"unexpected x {x}"
            )
            self.assertTrue(
                min(abs(y - 40.0), abs(y - 239.0)) < 1.0, f"unexpected y {y}"
            )

    def test_default_stream_finds_the_same_marker(self):
        # CUDA cannot capture a graph on the default stream, so this is a
        # different dispatch path inside the detector.
        detector, _ = self.make_detector()
        with detector:
            with aruco3cuda.DeviceImage.from_host(self.pixels, self.side, self.side) as image:
                detector.detect(image, None)
                result = detector.download(None)
        self.assertEqual([entry.id for entry in result], [42])

    def test_repeating_does_not_allocate_again(self):
        detector, _ = self.make_detector()
        with detector:
            with aruco3cuda.DeviceImage.from_host(self.pixels, self.side, self.side) as image:
                with aruco3cuda.Stream() as stream:
                    for _ in range(8):
                        detector.detect(image, stream)
                        detector.download(stream)
                statistics = detector.workspace_statistics()
        self.assertEqual(statistics.allocation_count, 1)
        self.assertEqual(statistics.reallocation_count, 0)
        self.assertGreater(statistics.peak_used_bytes, 0)

    def test_cuda_array_interface_input(self):
        # DeviceImage publishes __cuda_array_interface__, so passing it through
        # that path exercises the branch a CuPy or PyTorch array would take,
        # without requiring either to be installed.
        detector, _ = self.make_detector()

        class Wrapper:
            def __init__(self, interface):
                self.__cuda_array_interface__ = interface

        with detector:
            with aruco3cuda.DeviceImage.from_host(self.pixels, self.side, self.side) as image:
                wrapper = Wrapper(dict(image.__cuda_array_interface__))
                with aruco3cuda.Stream() as stream:
                    detector.detect(wrapper, stream)
                    result = detector.download(stream)
        self.assertEqual([entry.id for entry in result], [42])

    def test_device_detections_are_addressable(self):
        detector, config = self.make_detector()
        with detector:
            with aruco3cuda.DeviceImage.from_host(self.pixels, self.side, self.side) as image:
                with aruco3cuda.Stream() as stream:
                    detector.detect(image, stream)
                    device = detector.device_detections()
                    detector.download(stream)
        self.assertEqual(device.capacity, config.max_markers)
        for name in ("ids", "rotations", "corner_x", "corner_y", "count", "accepted_total"):
            self.assertIsNotNone(getattr(device, name), name)

    def test_marker_below_the_aruco3_lower_bound_is_missed(self):
        # The most common reason a first run reports nothing. Pinned down from
        # both sides: missed with the strategy on, found with it off.
        side, pixels = render(42, 40, 400)
        config = aruco3cuda.Config(max_width_px=side, max_height_px=side)
        with aruco3cuda.Detector() as detector:
            detector.initialize(kDictionary, config)
            with aruco3cuda.DeviceImage.from_host(pixels, side, side) as image:
                detector.detect(image, None)
                result = detector.download(None)
        self.assertEqual(len(result), 0)

        with aruco3cuda.Detector() as detector:
            detector.initialize(
                kDictionary,
                aruco3cuda.Config(
                    max_width_px=side,
                    max_height_px=side,
                    use_aruco3_detection=0,
                    corner_refine_method=aruco3cuda.CornerRefine.NONE,
                ),
            )
            with aruco3cuda.DeviceImage.from_host(pixels, side, side) as image:
                detector.detect(image, None)
                result = detector.download(None)
        self.assertEqual([entry.id for entry in result], [42])

    def test_unsupported_configuration_pair_is_rejected(self):
        # ArUco3 on with refinement off would leave the corners in the
        # coordinates of the downscaled image.
        config = aruco3cuda.Config(
            max_width_px=self.side,
            max_height_px=self.side,
            corner_refine_method=aruco3cuda.CornerRefine.NONE,
        )
        with aruco3cuda.Detector() as detector:
            with self.assertRaises(aruco3cuda.Error) as raised:
                detector.initialize(kDictionary, config)
            self.assertEqual(raised.exception.status, aruco3cuda.Status.INVALID_CONFIG)


class DeviceImageTest(unittest.TestCase):
    def test_pitch_is_at_least_the_width(self):
        with aruco3cuda.DeviceImage(97, 13) as image:
            self.assertGreaterEqual(image.pitch_bytes, 97)
            self.assertNotEqual(image.pointer, 0)

    def test_wrong_byte_count_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            aruco3cuda.DeviceImage.from_host(b"\x00" * 10, 4, 4)
        self.assertIn("expected 16 bytes", str(raised.exception))

    def test_close_is_idempotent(self):
        image = aruco3cuda.DeviceImage(8, 8)
        image.close()
        image.close()

    def test_non_positive_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            aruco3cuda.DeviceImage(0, 8)


if __name__ == "__main__":
    unittest.main()
