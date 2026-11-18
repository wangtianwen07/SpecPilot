"""Execute generated Python helpers with fake processes; never SSH/Docker/GPU.

Unlike source/AST checks, these execute the serialized definitions and their
read-only call chains, catching missing injected constants and imports.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from specpilot_tensorfold import configuration as capture
from specpilot_tensorfold import deployment
from test_configuration import evidence, request
from test_deployment import FakeRemote, configuration


def python_body(script, marker):
    return script.split(marker + "'\n", 1)[1].rsplit("\n" + marker, 1)[0]


def execute_python(source):
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(source, "<generated-remote-helper>", "exec"), {})
    return output.getvalue()


class GeneratedDriverTests(unittest.TestCase):
    def test_capture_driver_executes_serialized_config_reader_and_probe_builder(self):
        report = evidence()
        raw_env = {**report["env"], "CONTEXT": "auto", "PARALLEL": "auto", "MODEL_SOURCE": "download",
                   "WORKER_WEIGHTS": "rsync", "CONTEXT_RETRY": "1", "MEMGUARD": "0"}
        calls, queries = [], []

        def run(argv, **kwargs):
            calls.append(argv)
            if argv[:2] == ["bash", "-c"]:
                allowed = argv[5:]
                raw = b"".join(key.encode() + b"\0" + raw_env[key].encode() + b"\0"
                               for key in allowed if key in raw_env)
                return subprocess.CompletedProcess(argv, 0, raw, b"")
            if argv[:4] == ["docker", "container", "inspect", "--format"]:
                return subprocess.CompletedProcess(argv, 0, json.dumps(report["image"]), "")
            if argv[0] in ("python3", "ssh"):
                query = json.loads(kwargs["input"])
                queries.append(query)
                role = "worker" if argv[0] == "ssh" else "head"
                return subprocess.CompletedProcess(argv, 0, json.dumps(report["pair"][role]), "")
            raise AssertionError("unexpected generated subprocess: " + repr(argv))

        code = python_body(capture._capture_script(request()), "SPECPILOT_CAPTURE_DRIVER")
        with patch("subprocess.run", side_effect=run), \
                patch("subprocess.Popen", side_effect=AssertionError("real process forbidden")), \
                patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
            output = execute_python(code)
        result = json.loads(output.split("SPECPILOT_CAPTURE_RESULT=", 1)[1])
        self.assertEqual(result["env"]["CONTEXT"], "1048576")
        self.assertEqual(result["env"]["PARALLEL"], "8")
        self.assertEqual(result["env"]["WORKER_WEIGHTS"], "check")
        self.assertEqual(len(queries), 4)
        self.assertTrue(all(q["reconstructible_keys"] for q in queries))
        self.assertIn("expected_files_sha256", queries[2])
        self.assertEqual(sum(argv[:2] == ["bash", "-c"] for argv in calls), 2)

    def test_deployment_driver_executes_actual_inspect_run_and_verifier(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            cfg = deployment.DeploymentConfig.from_dict(configuration()).data
            fake = FakeRemote(configuration(), tmp)
            # Only owner_record.exists uses this path on inspect; keep that local
            # observation within the temporary test directory on Windows too.
            cfg["lock_path"] = str(Path(tmp) / "test.lock")

            def run(argv, **kwargs):
                if argv[0] not in ("python3", "ssh"):
                    raise AssertionError("inspect dispatched a non-probe command")
                query = json.loads(kwargs["input"])
                self.assertIn("reconstructible_keys", query)
                calls.append(query)
                role = "worker" if argv[0] == "ssh" else "head"
                return subprocess.CompletedProcess(argv, 0, json.dumps(fake.probe("baseline", role)), "")

            code = python_body(deployment._remote_script(cfg, "inspect"), "SPECPILOT_FROZEN_DRIVER")
            with patch("subprocess.run", side_effect=run), \
                    patch("subprocess.Popen", side_effect=AssertionError("real process forbidden")), \
                    patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")), \
                    patch("signal.SIGHUP", 1, create=True), patch("signal.signal"):
                output = execute_python(code)
        result = json.loads(output.split("SPECPILOT_DEPLOYMENT_RESULT=", 1)[1])
        self.assertEqual(result["state"], "baseline")
        self.assertEqual(result["action"], "inspect")
        self.assertEqual(len(calls), 3)  # Presence probe, then head+worker verification.
        self.assertEqual(set(result["inspection"]), {"head", "worker"})


class ExecutedProbeTests(unittest.TestCase):
    def probe(self, recipe, container, image, model_path, drafter_path):
        query = deployment.runtime_probe_request(recipe, "head", include_recipe=False)
        query.pop("api", None)  # API call chain is covered by the generated-driver test above.
        query["weights"] = {"model": str(model_path), "drafter": str(drafter_path)}
        calls = []

        def run(argv, **kwargs):
            calls.append(argv)
            if argv[:2] == ["nvidia-smi", "--query-gpu=uuid,name"]:
                stdout = "GPU-" + "1" * 36 + ", NVIDIA GB10\n"
            elif argv[:2] == ["nvidia-smi", "--query-compute-apps=pid,process_name"]:
                stdout = "111, fake-tensorfold\n"
            elif argv[:3] == ["docker", "image", "inspect"]:
                stdout = json.dumps([image])
            elif argv[:3] == ["docker", "container", "inspect"]:
                stdout = json.dumps([container])
            elif argv == ["docker", "ps", "-q"]:
                stdout = "fake-container-id\n"
            elif argv[:2] == ["docker", "top"]:
                stdout = "PID\n111\n"
            else:
                raise AssertionError("unexpected probe command: " + repr(argv))
            return subprocess.CompletedProcess(argv, 0, stdout, "")

        read_text = Path.read_text

        def safe_read_text(path, *args, **kwargs):
            if str(path).replace("\\", "/") == "/etc/machine-id":
                return "a" * 32 + "\n"
            return read_text(path, *args, **kwargs)

        with patch("subprocess.run", side_effect=run), \
                patch("subprocess.Popen", side_effect=AssertionError("real process forbidden")), \
                patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")), \
                patch.object(Path, "read_text", safe_read_text), patch("sys.stdin", io.StringIO(json.dumps(query))):
            output = execute_python(deployment._PROBE)
        result = json.loads(output)
        self.assertTrue(result["gpu_workloads"]["known"])
        self.assertEqual(result["gpu_workloads"]["unmanaged_processes"], [])
        self.assertNotIn("private-test-secret", output)
        self.assertEqual(result["weights"], {"model": "1" * 40, "drafter": "2" * 40})
        self.assertGreaterEqual(len(calls), 6)
        return result

    def test_real_probe_normalizes_isolated_paths_but_detects_semantic_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, drafter = root / "model", root / "drafter"
            for path, revision in ((model, "1" * 40), (drafter, "2" * 40)):
                path.mkdir()
                (path / ".revision").write_text(revision, encoding="utf-8")
            cfg = configuration()
            image = {"Id": "sha256:" + "4" * 64,
                     "Config": {"Env": ["PATH=/usr/bin"], "Entrypoint": ["/usr/local/bin/tensorfold-native"],
                                "Cmd": None, "WorkingDir": "/", "Labels": cfg["baseline"]["image_labels"]},
                     "RootFS": {"Layers": ["sha256:" + "9" * 64]}}

            def make(kind):
                recipe = copy.deepcopy(cfg[kind])
                directory = root / kind
                directory.mkdir()
                recipe["recipe_path"] = (directory / "recipe").as_posix()
                for key in ("CACHE_DIR", "STATE_DIR", "LOG_DIR"):
                    path = directory / key.lower()
                    path.mkdir()
                    recipe["env"][key] = path.as_posix()
                config_path = directory / "state_dir" / "model-config.json"
                config_path.write_text('{"bits":4}', encoding="utf-8")
                env = recipe["env"]
                container = {"Image": image["Id"], "Name": "/" + env["CONTAINER_NAME"],
                             "State": {"Running": True},
                             "Config": {"Env": ["HOME=" + env["CACHE_DIR"], "TF_GLM_RANK=0", "TF_GLM_WORLD=2",
                                                "PRIVATE_TOKEN=private-test-secret"],
                                        "Entrypoint": ["/usr/local/bin/tensorfold-native"],
                                        "Cmd": deployment.expected_command(env), "Labels": {"tf.recipe": "glm53-spark"}},
                             "HostConfig": {"Runtime": "nvidia", "NetworkMode": "host", "IpcMode": "host"},
                             "Mounts": [{"Source": model.as_posix(), "Destination": "/model", "RW": False},
                                        {"Source": env["CACHE_DIR"], "Destination": "/cache", "RW": True},
                                        {"Source": config_path.as_posix(), "Destination": "/model/config.json", "RW": False}]}
                return recipe, container, config_path

            baseline, baseline_co, _ = make("baseline")
            candidate, candidate_co, candidate_config = make("candidate")
            candidate_co["Mounts"].reverse()  # Docker mount listing order is not semantics.
            first = self.probe(baseline, baseline_co, image, model, drafter)
            second = self.probe(candidate, candidate_co, image, model, drafter)
            self.assertEqual(first["runtime_sha256"], second["runtime_sha256"])
            changed = copy.deepcopy(candidate_co)
            cmd = changed["Config"]["Cmd"]
            cmd[cmd.index("--context") + 1] = "8192"
            third = self.probe(candidate, changed, image, model, drafter)
            self.assertNotEqual(second["runtime_sha256"], third["runtime_sha256"])
            candidate_config.write_text('{"bits":3}', encoding="utf-8")
            fourth = self.probe(candidate, candidate_co, image, model, drafter)
            self.assertNotEqual(second["runtime_sha256"], fourth["runtime_sha256"])


if __name__ == "__main__":
    unittest.main()
