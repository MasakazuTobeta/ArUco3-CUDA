#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate resize profile CMake selection without a CUDA compiler or device."""
import pathlib
import subprocess
import sys
import tempfile
import unittest

CMAKE, MODULE = sys.argv[1:3]
del sys.argv[1:3]


class ResizeProfileTest(unittest.TestCase):
    def check_selection(self, profile, backend, fma="SEPARATE"):
        with tempfile.TemporaryDirectory() as directory:
            script = pathlib.Path(directory) / "select.cmake"
            module = pathlib.Path(MODULE).resolve().as_posix().replace('"', '\\"')
            script.write_text('include("' + module + '")\n'
                              'message(STATUS "SELECTED=${ARUCO3CUDA_RESIZE_PROFILE_OPENCV_ARM_HAL_VALUE},'
                              '${ARUCO3CUDA_RESIZE_ARM_SVE_BYTES_VALUE},${ARUCO3CUDA_CAROTENE_X_FMA_VALUE}")\n')
            args = [CMAKE]
            if profile is not None:
                args.append("-DARUCO3CUDA_RESIZE_PROFILE=" + profile)
            if backend is not None:
                args.append("-DARUCO3CUDA_RESIZE_ARM_BACKEND=" + backend)
            if fma is not None:
                args.append("-DARUCO3CUDA_CAROTENE_X_FMA=" + fma)
            return subprocess.run([*args, "-P", str(script)], check=False,
                                  capture_output=True, text=True)

    def test_explicit_profiles_and_descriptors(self):
        for profile, flag in [("GENERIC", 0), ("OPENCV_ARM_HAL", 1)]:
            for backend, width in [("AUTO", -1), ("NEON", 0), ("SVE128", 16),
                                   ("SVE256", 32), ("SVE512", 64)]:
                with self.subTest(profile=profile, backend=backend):
                    result = self.check_selection(profile, backend)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(f"SELECTED={flag},{width}", result.stdout)

    def test_fma_contract_is_explicit_and_independent_of_vector_layout(self):
        for backend in ["NEON", "SVE128", "SVE256", "SVE512"]:
            for fma, value in [("FUSED", 1), ("SEPARATE", 0)]:
                result = self.check_selection("OPENCV_ARM_HAL", backend, fma)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertRegex(result.stdout, rf"SELECTED=1,-?\d+,{value}(?:\n|$)")
        self.assertEqual(self.check_selection("GENERIC", "AUTO", None).returncode, 0)
        for fma in [None, "AUTO", "INVALID"]:
            result = self.check_selection("OPENCV_ARM_HAL", "AUTO", fma)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ARUCO3CUDA_CAROTENE_X_FMA", result.stderr)

    def test_unknown_profile_and_descriptor_are_errors(self):
        for profile, backend, name in [("NEON", "AUTO", "ARUCO3CUDA_RESIZE_PROFILE"),
                                        ("GENERIC", "SVE2048", "ARUCO3CUDA_RESIZE_ARM_BACKEND")]:
            result = self.check_selection(profile, backend)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(name, result.stderr)


if __name__ == "__main__":
    unittest.main()
