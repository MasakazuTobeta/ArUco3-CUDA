#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pure CMake selection tests; no CUDA compiler or device required."""
import pathlib
import subprocess
import sys
import tempfile
import unittest

CMAKE, MODULE = sys.argv[1:3]
del sys.argv[1:3]


class StrategyTest(unittest.TestCase):
    def select(self, **values):
        with tempfile.TemporaryDirectory() as directory:
            script = pathlib.Path(directory) / "select.cmake"
            module = pathlib.Path(MODULE).resolve().as_posix()
            script.write_text('cmake_minimum_required(VERSION 3.24)\ninclude("' + module + '")\n'
                              'message(STATUS "VALUES=${ARUCO3CUDA_CONTOUR_STRATEGY_VALUE},'
                              '${ARUCO3CUDA_THRESHOLD_STRATEGY_VALUE},${ARUCO3CUDA_PYRAMID_STRATEGY_VALUE},'
                              '${ARUCO3CUDA_OTSU_STRATEGY_VALUE},${ARUCO3CUDA_CANDIDATE_GRID_STRATEGY_VALUE}")\n')
            return subprocess.run([CMAKE, *["-D"+k+"="+str(v) for k,v in values.items()],
                                   "-P", str(script)], text=True, capture_output=True, check=False)

    def test_defaults_and_explicit_optimized_choices(self):
        result = self.select()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("VALUES=0,0,0,0,0", result.stdout)
        result = self.select(ARUCO3CUDA_CONTOUR_STRATEGY="ORDERED_CONTOURS",
                             ARUCO3CUDA_THRESHOLD_STRATEGY="FUSED_AUTO",
                             ARUCO3CUDA_PYRAMID_STRATEGY="DEMAND",
                             ARUCO3CUDA_OTSU_STRATEGY="DENSITY_FIXED_POINT",
                             ARUCO3CUDA_CANDIDATE_GRID_STRATEGY="PERSISTENT")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("VALUES=1,1,1,2,1", result.stdout)

    def test_invalid_enums_and_integer_boundaries(self):
        for name in ["CONTOUR", "THRESHOLD", "PYRAMID", "OTSU", "CANDIDATE_GRID"]:
            self.assertNotEqual(self.select(**{"ARUCO3CUDA_"+name+"_STRATEGY":"UNKNOWN"}).returncode,0)
        for name, valid, invalid in [("ARUCO3CUDA_OTSU_MAX_OCCUPIED_BINS", [1,256], [0,257,-1,"1.5"]),
                                     ("ARUCO3CUDA_CANDIDATE_GRID_BLOCKS", [1,2147483647], [0,2147483648,-1,"1x"])]:
            for value in valid:
                result=self.select(**{name:value});self.assertEqual(result.returncode,0,result.stderr)
            for value in invalid:self.assertNotEqual(self.select(**{name:value}).returncode,0)

    def test_contour_variants_and_coefficient_selection(self):
        choices={"ARUCO3CUDA_CONTOUR_COUNT_STRATEGY":"EDGE_AGGREGATE",
                 "ARUCO3CUDA_CONTOUR_RDP_STRATEGY":"WARP",
                 "ARUCO3CUDA_CONTOUR_MATERIALIZE_STRATEGY":"WARP",
                 "ARUCO3CUDA_CONTOUR_MATERIALIZE_COMPACTION_STRATEGY":"ELIGIBLE_EDGES",
                 "ARUCO3CUDA_RESIZE_COEFFICIENT_STRATEGY":"KLEIDI_PRECOMPUTED"}
        result=self.select(**choices);self.assertEqual(result.returncode,0,result.stderr)
        for name in choices:
            self.assertNotEqual(self.select(**{name:"UNKNOWN"}).returncode,0)
        self.assertNotEqual(self.select(ARUCO3CUDA_CONTOUR_MATERIALIZE_COMPACTION_STRATEGY="ELIGIBLE_EDGES").returncode,0)
        self.assertNotEqual(self.select(ARUCO3CUDA_RESIZE_COEFFICIENT_STRATEGY="PRECOMPUTED").returncode,0)
        self.assertNotEqual(self.select(CMAKE_CUDA_FLAGS="-DARUCO3CUDA_EXPERIMENT_COOPERATIVE_RDP=1").returncode,0)
        self.assertNotEqual(self.select(CMAKE_CXX_FLAGS="-DARUCO3CUDA_PRECOMPUTED_RESIZE_COEFFICIENTS=1").returncode,0)

    def test_legacy_defines_cannot_override_maintained_choices(self):
        for flags in ["CMAKE_CUDA_FLAGS", "CMAKE_CXX_FLAGS_RELEASE"]:
            for value in ["-DARUCO3CUDA_ENABLE_FUSED_THRESHOLD=1", "-D ARUCO3CUDA_ENABLE_FUSED_THRESHOLD"]:
                result=self.select(**{flags:value})
                self.assertNotEqual(result.returncode,0)
                self.assertIn("conflicts",result.stderr)
        result=self.select(ARUCO3CUDA_THRESHOLD_STRATEGY="FUSED_AUTO",
                           CMAKE_CUDA_FLAGS="-DARUCO3CUDA_ENABLE_FUSED_THRESHOLD=1")
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertNotEqual(self.select(ARUCO3CUDA_EXPERIMENT_CONTOUR_CANDIDATES="ON").returncode,0)
        self.assertEqual(self.select(ARUCO3CUDA_EXPERIMENT_CONTOUR_CANDIDATES="ON",
                                     ARUCO3CUDA_CONTOUR_STRATEGY="ORDERED_CONTOURS").returncode,0)
        self.assertNotEqual(self.select(CMAKE_CUDA_FLAGS="-DARUCO3CUDA_OTSU_FIXED_POINT_MAX_OCCUPIED_BINS=64").returncode,0)
        self.assertNotEqual(self.select(CMAKE_CXX_FLAGS="-DARUCO3CUDA_CANDIDATE_GRID_BLOCKS=64").returncode,0)


if __name__ == "__main__":
    unittest.main()
