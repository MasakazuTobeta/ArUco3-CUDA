# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the changing-frame result format and manifest binding."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import run_sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
import score


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        image = self.root / "frame.pgm"
        image.write_bytes(b"P5\n1 1\n255\n\x00")
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({
            "dictionary": "DICT_APRILTAG_36h11",
            "scenes": [{"name": "frame", "path": image.name,
                        "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}],
        }))
        self.digest = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        self.out = self.root / "out"

    @staticmethod
    def row(cycle=0):
        return {"backend": "nvidia", "scope": "device", "cycle": cycle, "frame": 0,
                "ms": 1.0, "tags": [{"id": 17, "corners": [[3, 3], [0, 3], [0, 0], [3, 0]]}]}

    def run_rows(self, *rows, check_command=None):
        # Isolate the protocol from device timing. The binary identity is still computed.
        args = ["run_sequence", sys.executable, str(self.manifest), str(self.out),
                "--backend", "nvidia", "--samples", str(len(rows))]
        result = subprocess.CompletedProcess([], 0, stdout="\n".join(map(json.dumps, rows)))
        def execute(command, **_):
            if check_command is not None:
                check_command(command)
            return result

        with mock.patch.object(sys, "argv", args), mock.patch.object(subprocess, "run", side_effect=execute):
            run_sequence.main()

    def test_round_trip_and_wrong_manifest(self):
        self.run_rows(self.row(), self.row(1))
        path = self.out / "observations.jsonl"
        obs = score.load_observations(path, {"frame"}, self.digest)
        self.assertEqual(obs["frame"][0]["corners"], [0, 0, 3, 0, 3, 3, 0, 3])
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            score.load_observations(path, {"frame"}, "different", allow_unbound=True)

    def test_native_loader_receives_the_verified_bytes(self):
        original = self.root / "frame.pgm"
        expected = original.read_bytes()

        def replace_source(command):
            original.write_bytes(b"changed after verification")
            self.assertEqual(Path(command[-1]).read_bytes(), expected)

        self.run_rows(self.row(), check_command=replace_source)
        self.assertEqual(json.loads((self.out / "conditions.json").read_text())["command"][-1], str(original))

    def test_invalid_backend_or_results_do_not_publish_observations(self):
        for override in ({"backend": "project"}, {"backend": None}, {"ms": True},
                         {"frame": False}, {"tags": {}}, {"tags": [{"id": 587, "corners": []}]}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.run_rows(dict(self.row(), **override))
            self.assertFalse(self.out.exists())

    def test_changed_replay_records_evidence(self):
        with self.assertRaisesRegex(ValueError, "changing detection"):
            self.run_rows(self.row(), dict(self.row(1), tags=[]))
        self.assertTrue((self.out / "changed-observation.json").is_file())
        self.assertFalse((self.out / "conditions.json").exists())

    def test_legacy_run_requires_matching_identity(self):
        path = self.root / "observations.jsonl"
        path.write_text(json.dumps({"scene": "frame", "detections": []}))
        with self.assertRaisesRegex(ValueError, "lack a manifest"):
            score.load_observations(path, {"frame"}, self.digest)
        with mock.patch("sys.stderr"):
            self.assertEqual(score.load_observations(path, {"frame"}, self.digest, True), {"frame": []})
        (self.root / "conditions.json").write_text(json.dumps({"manifest_sha256": self.digest}))
        self.assertEqual(score.load_observations(path, {"frame"}, self.digest), {"frame": []})
        with self.assertRaisesRegex(ValueError, "legacy run manifest mismatch"):
            score.load_observations(path, {"frame"}, "different", allow_unbound=True)

    def test_duplicate_missing_and_mixed_rows(self):
        path = self.root / "observations.jsonl"
        row = {"scene": "frame", "detections": [], "manifest_sha256": self.digest}
        for rows in ([row, row], [row], [row, {"scene": "other", "detections": []}]):
            path.write_text("\n".join(map(json.dumps, rows)))
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                score.load_observations(path, {"frame", "other"}, self.digest, allow_unbound=True)


if __name__ == "__main__":
    unittest.main()
