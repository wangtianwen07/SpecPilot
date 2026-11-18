"""CLI integration tests; synthetic data and local subprocesses only."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from specpilot_tensorfold.__main__ import main
from test_evaluation import synthetic_fixture


class CliTests(unittest.TestCase):
    def invoke(self, args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(args)
        return status, json.loads(output.getvalue())

    def test_offline_evaluate_and_atomic_file(self):
        contract, records = synthetic_fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
            (root / "records.json").write_text(json.dumps({"records": records}), encoding="utf-8")
            args = ["evaluate", "--contract", str(root / "contract.json"), "--records",
                    str(root / "records.json"), "--output", str(root / "result.json")]
            code, result = self.invoke(args)
            self.assertEqual(code, 0)
            self.assertEqual(result["status"], "SYNTHETIC_PASS")
            self.assertFalse(result["promotion_allowed"])
            self.assertEqual(json.loads((root / "result.json").read_text(encoding="utf-8")), result)
            records.pop()
            (root / "records.json").write_text(json.dumps(records), encoding="utf-8")
            code, result = self.invoke(args)
            self.assertEqual(code, 2)
            self.assertEqual(result["status"], "INVALID_EVIDENCE")

    def test_measure_defaults_to_no_http(self):
        with tempfile.TemporaryDirectory() as temp:
            batch = Path(temp) / "batch.json"
            batch.write_text(json.dumps({"schema_version": 1, "base_url": "http://127.0.0.1:1/v1",
                                         "model": "fixture", "concurrency": 1,
                                         "requests": [{"body": {}, "metadata": {}}]}), encoding="utf-8")
            with patch("specpilot_tensorfold.__main__.measure_batch", side_effect=AssertionError("must not infer")):
                code, report = self.invoke(["measure", "--batch", str(batch), "--output", str(Path(temp) / "plan.json")])
            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "PLAN")
            self.assertFalse(report["inference_started"])

    def test_capture_defaults_to_no_ssh(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch("specpilot_tensorfold.configuration.subprocess.run",
                       side_effect=AssertionError("must not connect")):
                code, report = self.invoke([
                    "capture-config", "--head", "fixture-head", "--worker", "fixture-worker",
                    "--recipe", "/opt/fixture", "--container", "fixture-baseline",
                    "--lock", "/opt/control/pair.lock", "--transaction", "fixture-capture",
                    "--output", str(Path(temp) / "plan.json")])
            self.assertEqual(code, 0)
            self.assertEqual(report["action"], "capture-config")
            self.assertFalse(report["execution_enabled"])
            self.assertFalse(report["mutates_remote"])

    def test_error_replaces_old_result_with_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "out.json"
            out.write_text('{"status":"PASS"}', encoding="utf-8")
            code, report = self.invoke(["evaluate", "--contract", str(Path(temp) / "absent.json"),
                                        "--records", "none", "--output", str(out)])
            self.assertEqual(code, 2)
            self.assertEqual(report["status"], "ERROR")
            self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["status"], "ERROR")

    def test_missing_api_key_never_calls_server(self):
        with patch.dict(os.environ, {}, clear=True), patch("specpilot_tensorfold.__main__.TensorFoldClient") as client:
            code, report = self.invoke(["inspect-api", "--base-url", "http://127.0.0.1:1", "--model", "fixture",
                                        "--api-key-env", "FIXTURE_SECRET"])
            self.assertEqual(code, 2)
            client.assert_not_called()

    def test_both_entrypoints_work_without_sglang(self):
        root = Path(__file__).resolve().parents[2]
        for argv in ([sys.executable, "-m", "specpilot_tensorfold", "--help"],
                     [sys.executable, "scripts/tensorfold.py", "--help"]):
            with self.subTest(argv=argv):
                p = subprocess.run(argv, cwd=root, capture_output=True, text=True, encoding="utf-8",
                                   env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=20)
                self.assertEqual(p.returncode, 0, p.stderr)
                self.assertIn("capture-config", p.stdout)
                self.assertIn("evaluate", p.stdout)


if __name__ == "__main__":
    unittest.main()
