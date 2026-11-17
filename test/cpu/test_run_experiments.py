#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise experiment output ownership without OpenCV or a compiled detector."""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER = ROOT / "cpu" / "run_experiments.py"
HOST_RUNNER = ROOT / "docs" / "measurements" / "2026-11-17-cpu-multihost" / "run_host.py"
HOST_AUDIT = HOST_RUNNER.with_name("qa_audit.py")


@unittest.skipUnless(hasattr(os, "sched_getaffinity"), "Runner requires Linux affinity")
class ExperimentOutputTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)
        self.output = self.root / "results" / "experiment"
        self.cpu = min(os.sched_getaffinity(0))
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        (self.root / "source.py").write_text("# fixture source\n")
        subprocess.run(["git", "add", "source.py"], cwd=self.root, check=True)
        subprocess.run(["git", "-c", "user.name=Runner test", "-c",
                        "user.email=runner-test@example.invalid", "commit", "-qm", "fixture"],
                       cwd=self.root, check=True)
        self.binary = self.root / "fake-benchmark"
        self.binary.write_text("#!" + sys.executable + "\n"
                               "import json, sys\n"
                               "print(json.dumps({'arguments': sys.argv[1:]}), flush=True)\n")
        self.binary.chmod(0o755)
        (self.root / "inputs").mkdir()

    def command(self, *extra):
        return [sys.executable, str(RUNNER), "--binary", str(self.binary),
                "--input-dir", str(self.root / "inputs"), "--output-dir", str(self.output),
                "--cpu-list", str(self.cpu), "--runs", "2", "--iterations", "1",
                "--warmup", "0", *extra]

    def run_runner(self, *extra, environment=None):
        return subprocess.run(self.command(*extra), cwd=self.root, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
                              env=environment)

    def output_snapshot(self):
        return {str(p.relative_to(self.output)): p.read_bytes()
                for p in self.output.rglob("*") if p.is_file()}

    def test_new_output_records_matching_metadata_and_both_runs(self):
        result = self.run_runner()
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((self.output / "environment.json").read_text())
        self.assertEqual(metadata["binary_sha256"], hashlib.sha256(self.binary.read_bytes()).hexdigest())
        self.assertEqual(metadata["independent_processes"], 2)
        self.assertEqual(metadata["cpu_affinity"], [self.cpu])
        for run in [1, 2]:
            row = json.loads((self.output / ("run-%d.jsonl" % run)).read_text())
            self.assertEqual(row["arguments"], metadata["command"][1:])

    def test_rerun_cannot_replace_existing_provenance(self):
        self.output.mkdir(parents=True)
        (self.output / "environment.json").write_text('{"original": true}\n')
        (self.output / "run-1.jsonl").write_text('{"original_measurement": true}\n')
        before = self.output_snapshot()
        result = self.run_runner("--background-note", "different invocation")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.output_snapshot(), before)

    def test_partial_and_empty_directories_are_not_reused(self):
        for existing in [None, "environment.json", "run-2.jsonl"]:
            with self.subTest(existing=existing):
                self.output = self.root / ("empty" if existing is None else existing + ".dir")
                self.output.mkdir()
                if existing:
                    (self.output / existing).write_text("original\n")
                before = self.output_snapshot()
                result = self.run_runner()
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.output_snapshot(), before)

    def test_symlink_to_existing_output_is_rejected_without_changes(self):
        original = self.root / "original"
        original.mkdir()
        (original / "environment.json").write_text("original\n")
        self.output.parent.mkdir()
        self.output.symlink_to(original, target_is_directory=True)
        before = self.output_snapshot()
        result = self.run_runner()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(self.output.is_symlink())
        self.assertEqual(self.output_snapshot(), before)

    def test_missing_binary_does_not_create_output_directory(self):
        result = self.run_runner("--binary", str(self.root / "missing"))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())

    def test_binary_path_selects_the_same_file_that_was_hashed(self):
        competing = self.root / "competing-path"
        competing.mkdir()
        other = competing / self.binary.name
        other.write_text("#!" + sys.executable + "\nprint('different executable')\n")
        other.chmod(0o755)
        environment = dict(os.environ, PATH=str(competing) + os.pathsep + os.environ.get("PATH", ""))
        result = self.run_runner("--binary", self.binary.name, environment=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((self.output / "environment.json").read_text())
        self.assertEqual(metadata["command"][0], str(self.binary))
        self.assertEqual(metadata["binary_sha256"], hashlib.sha256(self.binary.read_bytes()).hexdigest())
        row = json.loads((self.output / "run-1.jsonl").read_text())
        self.assertEqual(row["arguments"], metadata["command"][1:])

    def test_failed_binary_leaves_evidence_that_retry_cannot_overwrite(self):
        self.binary.write_text("#!" + sys.executable + "\n"
                               "print('partial output', flush=True)\nraise SystemExit(9)\n")
        first = self.run_runner()
        self.assertNotEqual(first.returncode, 0)
        self.assertEqual((self.output / "run-1.jsonl").read_text(), "partial output\n")
        self.assertFalse((self.output / "run-2.jsonl").exists())
        before = self.output_snapshot()
        second = self.run_runner("--background-note", "retry after failure")
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(self.output_snapshot(), before)

    def test_concurrent_invocations_have_exactly_one_output_owner(self):
        first = subprocess.Popen(self.command("--background-note", "first"), cwd=self.root,
                                 text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        second = subprocess.Popen(self.command("--background-note", "second"), cwd=self.root,
                                  text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            first.communicate(timeout=20)
            second.communicate(timeout=20)
        finally:
            for process in [first, second]:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
        self.assertEqual(sorted([first.returncode == 0, second.returncode == 0]), [False, True])
        winner = "first" if first.returncode == 0 else "second"
        metadata = json.loads((self.output / "environment.json").read_text())
        self.assertTrue(metadata["background_load"].startswith(winner + ";"))
        self.assertEqual(set(self.output_snapshot()), {"environment.json", "run-1.jsonl", "run-2.jsonl"})


@unittest.skipUnless(hasattr(os, "sched_getaffinity"), "Runner requires Linux affinity")
class HostOutputTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)
        self.output = self.root / "results"
        self.cpus = sorted(os.sched_getaffinity(0))
        self.binary = self.root / "fake-benchmark"
        self.binary.write_text("#!" + sys.executable + "\n" + '''
import argparse, hashlib, json, pathlib
p = argparse.ArgumentParser()
for name in ['input-dir', 'iterations', 'warmup', 'threads', 'aruco3', 'filter']:
    p.add_argument('--' + name, default='')
a, _ = p.parse_known_args()
print(json.dumps({'type': 'environment', 'opencv_version': '4.14.0',
                  'opencv_threads': int(a.threads), 'iterations': int(a.iterations),
                  'warmup': int(a.warmup), 'aruco3': bool(int(a.aruco3)),
                  'profile_stages': False, 'throughput_frames': 0}))
for image in sorted(pathlib.Path(a.input_dir).glob('*.png')):
    for variant, duration in [('opencv', 2.0), ('optimized', 1.0)]:
        print(json.dumps({'type': 'measurement', 'scene': image.name,
            'variant': variant, 'sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
            'width': 32, 'height': 24,
            'compatibility_discrepancy': '', 'max_corner_error_px': 0,
            'detections': 4 if a.filter else 0, 'reference_detections': 4 if a.filter else 0,
            'rejected': 0, 'reference_rejected': 0,
            'p50_ms': duration, 'p95_ms': duration, 'p99_ms': duration}))
''')
        self.binary.chmod(0o755)
        inputs = self.root / "data" / "cpu-real-background"
        controls = self.root / "data" / "cpu-experiment"
        inputs.mkdir(parents=True)
        controls.mkdir(parents=True)
        for index in range(16):
            (inputs / ("photo-%02d.png" % index)).write_bytes(bytes([index]))
        (controls / "cpu_1280x720_n4_clean.png").write_bytes(b"control")
        files = {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in self.root.glob("data/*/*.png")}
        self.manifest = {"git_head": "test-fixture", "files": files, "sources": {},
                         "inputs": {pathlib.Path(p).name: digest for p, digest in files.items()
                                    if "cpu-real-background" in p},
                         "controls": {pathlib.Path(p).name: digest for p, digest in files.items()
                                      if "cpu-experiment" in p}}
        (self.root / "snapshot.json").write_text(json.dumps(self.manifest))
        (self.root / "cpu").mkdir()
        shutil.copyfile(ROOT / "cpu" / "summarize_experiments.py",
                        self.root / "cpu" / "summarize_experiments.py")

    def run_runner(self, *extra, python_options=(), environment=None):
        four = self.cpus[:4] if len(self.cpus) >= 4 else [0, 1, 2, 3]
        return subprocess.run([sys.executable, *python_options, str(HOST_RUNNER), "--binary", str(self.binary),
                               "--output-dir", str(self.output), "--cpu-one", str(self.cpus[0]),
                               "--cpu-four", ",".join(map(str, four)), *extra],
                              cwd=self.root, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=30, env=environment)

    def output_snapshot(self):
        return {str(p.relative_to(self.output)): p.read_bytes()
                for p in self.output.rglob("*") if p.is_file()}

    def test_existing_host_results_keep_original_top_level_provenance(self):
        group = self.output / "positive-control-1-1t"
        group.mkdir(parents=True)
        (group / "run-1.jsonl").write_text("original output\n")
        (self.output / "snapshot.json").write_text('{"original_snapshot": true}\n')
        (self.output / "input-hashes.json").write_text('{"original_inputs": true}\n')
        before = self.output_snapshot()
        result = self.run_runner()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.output_snapshot(), before)

    def test_invalid_affinity_does_not_publish_any_output(self):
        for existing in [False, True]:
            with self.subTest(existing=existing):
                self.output = self.root / ("existing" if existing else "fresh")
                if existing:
                    self.output.mkdir()
                    (self.output / "snapshot.json").write_text("original snapshot\n")
                before = self.output_snapshot()
                result = self.run_runner("--cpu-four", str(self.cpus[0]))
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.output.exists(), existing)
                self.assertEqual(self.output_snapshot(), before)

    def test_fresh_host_run_completes_all_groups(self):
        if len(self.cpus) < 4:
            self.skipTest("Successful host run requires four available CPUs")
        result = self.run_runner()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((self.output / "snapshot.json").read_text()), self.manifest)
        self.assertEqual(len(list(self.output.glob("*/validation.json"))), 10)
        self.assertEqual(len(list(self.output.glob("*/summary.json"))), 4)
        completion = json.loads((self.output / "completion.json").read_text())
        self.assertEqual(completion["timing_rows"], 384)
        self.assertEqual(completion["binary_sha256"], hashlib.sha256(self.binary.read_bytes()).hexdigest())

    def test_optimized_mode_cannot_skip_output_validation(self):
        self.binary.write_text(self.binary.read_text().replace("4 if a.filter else 0", "99"))
        modes = [("-O", ["-O"], None), ("-OO", ["-OO"], None),
                 ("environment", [], dict(os.environ, PYTHONOPTIMIZE="1"))]
        for name, options, environment in modes:
            with self.subTest(mode=name):
                self.output = self.root / ("disabled-assertions-" + name)
                result = self.run_runner(python_options=options, environment=environment)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Validation requires Python assertions", result.stderr)
                self.assertFalse(self.output.exists())

    def test_audit_rejects_optimized_mode_before_reading_or_writing(self):
        audit_output = self.root / "audit-result.json"
        original = '{"preserve_previous_audit": true}\n'
        audit_output.write_text(original)
        modes = [("-O", ["-O"], None), ("-OO", ["-OO"], None),
                 ("environment", [], dict(os.environ, PYTHONOPTIMIZE="1"))]
        for name, options, environment in modes:
            with self.subTest(mode=name):
                result = subprocess.run([sys.executable, *options, str(HOST_AUDIT), "missing-host",
                                         "--output", str(audit_output)], cwd=self.root, text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        timeout=20, env=environment)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Validation requires Python assertions", result.stderr)
                self.assertEqual(audit_output.read_text(), original)


if __name__ == "__main__":
    unittest.main()
