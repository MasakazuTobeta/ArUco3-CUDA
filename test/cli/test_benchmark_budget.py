#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Budget CLI parsing and aggregation checks; no images or CUDA work."""
import contextlib
import importlib.util
import io
import subprocess
import sys
import unittest

BENCH, AGGREGATE = sys.argv[1:3]
del sys.argv[1:3]
spec = importlib.util.spec_from_file_location("aggregate", AGGREGATE)
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)


class BudgetTest(unittest.TestCase):
    def test_invalid_budgets_fail_before_image_or_cuda_work(self):
        cases = [
            (["--max-candidates", "0"], "max_candidates"),
            (["--max-candidates", "4194305"], "max_candidates"),
            (["--max-markers", "0"], "max_markers"),
            (["--max-markers", "-1"], "max_markers"),
            (["--max-markers", "3", "--max-candidates", "2"], "exceeds max_candidates"),
            (["--max-candidates", "2", "--max-markers", "3"], "exceeds max_candidates"),
            (["--max-candidates", "16384junk"], "cannot parse"),
            (["--max-markers", "999999999999999999999"], "cannot parse"),
        ]
        for args, reason in cases:
            with self.subTest(args=args):
                result = subprocess.run([BENCH, *args, "--input", "/not-opened.png"],
                                        capture_output=True, text=True, check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(reason, result.stderr)
                self.assertNotIn("imread", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_valid_budget_options_are_recognized(self):
        result = subprocess.run([BENCH, "--max-candidates", "16384", "--max-markers",
                                 "2048", "--help"], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--max-candidates", result.stdout)
        self.assertIn("--max-markers", result.stdout)

    def test_aggregation_keeps_different_and_unknown_budgets_separate(self):
        def row(candidates):
            conditions = {"max_markers": 1024, "fxfy_effective": 1.0,
                          "use_aruco3_detection": True}
            if candidates is not None:
                conditions["max_candidates"] = candidates
            return {"route": "CPU", "memory_mode": "N/A", "conditions": conditions,
                    "width_px": 32, "height_px": 32, "detection_count": 0,
                    "image_path": "same.png", "end_to_end": {"p50_ms": 1.0}}

        def printed(records):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                aggregate.print_run_variance(aggregate.measurement_rows(records))
            return out.getvalue()

        self.assertNotIn("n=", printed([row(4096), row(16384), row(None)]))
        self.assertIn("n=2", printed([row(16384), row(16384)]))
        generic = row(16384)
        generic["conditions"].update(gpu_resize_profile="GENERIC", gpu_resize_backend="GENERIC")
        neon = row(16384)
        neon["conditions"].update(gpu_resize_profile="OPENCV_ARM_HAL", gpu_resize_backend="NEON")
        sve = row(16384)
        sve["conditions"].update(gpu_resize_profile="OPENCV_ARM_HAL", gpu_resize_backend="SVE256")
        self.assertNotIn("n=", printed([generic, neon, sve]))
        fused = row(16384)
        fused["conditions"].update(neon["conditions"], gpu_carotene_x_fma="FUSED")
        separate = row(16384)
        separate["conditions"].update(neon["conditions"], gpu_carotene_x_fma="SEPARATE")
        self.assertNotIn("n=", printed([neon, fused, separate]))
        self.assertIn("n=2", printed([fused, fused]))
        grid128 = row(16384)
        grid128["conditions"].update(fused["conditions"], gpu_strategy_identity="candidate_grid=PERSISTENT;candidate_grid_blocks=128")
        grid512 = row(16384)
        grid512["conditions"].update(fused["conditions"], gpu_strategy_identity="candidate_grid=PERSISTENT;candidate_grid_blocks=512")
        self.assertNotIn("n=", printed([fused, grid128, grid512]))
        self.assertIn("n=2", printed([grid512, grid512]))


if __name__ == "__main__":
    unittest.main()
