"""Read-only baseline capture tests. All SSH, Docker and GPU data is fabricated."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from specpilot_tensorfold import configuration as capture
from specpilot_tensorfold import deployment


def request():
    return {"head": {"ssh_target": "user@head.example", "ssh_options": []},
            "worker": {"ssh_target": "user@worker.example", "ssh_options": []},
            "recipe_path": "/trusted/recipe", "container_name": "frozen-base",
            "lock_path": "/trusted/pair.lock", "transaction_id": "capture-test"}


def evidence():
    env = {key: "0" for key in deployment.REQUIRED_ENV}
    env.update({"MODEL_SOURCE": "local", "MODEL_ID": "example/exl3", "MODEL_REVISION": "1" * 40,
                "DFLASH2_ID": "example/dflash2", "DFLASH2_REVISION": "2" * 40,
                "WORKER_WEIGHTS": "check", "TP": "2", "CONTEXT": "1048576", "CONTEXT_RETRY": "0",
                "PARALLEL": "8", "KV": "fp8", "KV_PAGES": "on", "DENSE": "q8", "HEAD": "q8",
                "DRAFTER": "dflash2", "DRAFTER_BITS": "4", "HOST": "127.0.0.1", "PORT": "8888",
                "NCCL_PORT": "29583", "CONTAINER_NAME": "frozen-base", "SERVED_NAME": "example-model",
                "REASONING_EFFORT": "high", "CHAT_TEMPLATE": "auto", "MEMGUARD": "1",
                "MEMORY_RESERVE_GIB": "12.5", "TF_BIN": "/usr/local/bin/tensorfold-native",
                "SERVE_EXTRA_ARGS": "", "CONFIG_OVERLAY": "auto"})
    for name in ("MODEL_DIR", "DFLASH2_DIR", "WORKER_MODEL_DIR", "WORKER_DFLASH2_DIR", "CACHE_DIR",
                 "STATE_DIR", "LOG_DIR", "WORKER_CACHE_DIR", "WORKER_STATE_DIR", "WORKER_OVERLAY"):
        env[name] = "/trusted/" + name.lower()
    files = {key: "a" * 64 for key in deployment.REQUIRED_FILES}
    files.update({".env": None, "scripts/local.sh": None, "patches/engine/0001-frozen.patch": "b" * 64,
                  "templates/chat.jinja": "c" * 64})
    labels = {"tf.patches": "a1b2c3d4e5f6", "tf.ref": "5" * 40, "tf.sm": "121", "tf.zig": "0.17.0",
              "tf.engine": "zig", "unrelated.private.label": "not-returned-in-config"}
    pair = {}
    for rank, role in enumerate(("head", "worker")):
        native_env = {field: env[key] for key, field in deployment.NATIVE_ENV_MAP.items() if key in env}
        native_env.update({"TF_GLM_RANK": str(rank), "TF_GLM_WORLD": "2", "TF_GLM_DENSE_BITS": "8",
                           "TF_GLM_HEAD_BITS": "8", "TF_KV_PAGES": "on", "NCCL_COMM_ID": "192.0.2.10:29583"})
        pair[role] = {"machine_id": ("a" if role == "head" else "b") * 32,
                      "gpu_uuid": "GPU-" + ("a" if role == "head" else "b") * 36, "gpu_name": "NVIDIA GB10",
                      "recipe_commit": "3" * 40, "files_sha256": copy.deepcopy(files),
                      "recipe_label": labels["tf.patches"], "runtime_sha256": ("c" if role == "head" else "d") * 64,
                      "image": {"id": "sha256:" + "4" * 64, "labels": copy.deepcopy(labels), "content_sha256": "e" * 64},
                      "container": {"running": True, "image_id": "sha256:" + "4" * 64,
                                    "labels": {"tf.recipe": "glm53-spark", "tf.patches": labels["tf.patches"]},
                                    "env": native_env,
                                    "cmd": deployment.expected_command(env),
                                    "mounts": [{"Source": env["MODEL_DIR" if role == "head" else "WORKER_MODEL_DIR"],
                                                "Destination": "/model", "RW": False},
                                               {"Source": env["DFLASH2_DIR" if role == "head" else "WORKER_DFLASH2_DIR"],
                                                "Destination": "/dflash2", "RW": False}]},
                      "weights": {"model": env["MODEL_REVISION"], "drafter": env["DFLASH2_REVISION"]},
                      "gpu_workloads": {"known": True, "other_containers": [], "unmanaged_processes": []},
                      "api": {"health": {"status": "ok", "live": {"connections": 0}},
                              "v1/models": {"data": [{"id": env["SERVED_NAME"]}]}}}
    return {"env": env, "image": "example-tensorfold:baseline", "pair": pair}


def result_runner(report, code=0):
    def run(argv, *, input, timeout):
        return subprocess.CompletedProcess(argv, code, "SPECPILOT_CAPTURE_RESULT=" + json.dumps(report) + "\n", "")
    return run


class CaptureContractTests(unittest.TestCase):
    def test_default_is_a_reviewable_plan_and_does_not_call_runner(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("plan attempted to execute")
        plan = capture.capture_baseline_config(**request(), runner=forbidden)
        self.assertEqual(plan["action"], "capture-config")
        self.assertFalse(plan["execution_enabled"])
        self.assertFalse(plan["mutates_remote"])
        self.assertEqual(plan["safety_overrides"], capture.SAFETY_OVERRIDES)
        self.assertIn("future starts only", plan["safety_override_scope"])
        self.assertEqual(plan["script_sha256"], hashlib.sha256(plan["stdin"].encode()).hexdigest())
        self.assertEqual(plan["argv"][-1], "bash -s")
        self.assertIn("StrictHostKeyChecking=yes", plan["argv"])
        python = plan["stdin"].split("SPECPILOT_CAPTURE_DRIVER'\n", 1)[1].rsplit("\nSPECPILOT_CAPTURE_DRIVER", 1)[0]
        ast.parse(python)
        self.assertNotIn("run_locked(", python)
        self.assertNotIn("docker build", python)
        self.assertNotIn("/v1/chat/completions", python)

    def test_execute_returns_a_valid_baseline_only_config_without_writing(self):
        report = evidence()
        with patch("subprocess.run", side_effect=AssertionError("local verifier tried running a command")):
            config = capture.capture_baseline_config(**request(), execute=True, runner=result_runner(report))
        self.assertIsNone(config["candidate"])
        self.assertEqual(config["baseline"]["env"], report["env"])
        self.assertEqual(config["baseline"]["runtime_sha256"]["head"], "c" * 64)
        self.assertEqual(config["head"]["machine_id"], "a" * 32)
        self.assertEqual(config["worker"]["gpu_uuid"], "GPU-" + "b" * 36)
        self.assertNotIn("unrelated.private.label", config["baseline"]["image_labels"])
        self.assertEqual(deployment.DeploymentConfig.from_dict(config).data, config)

    def test_rejects_other_gpu_rank_image_mount_and_revision(self):
        defects = [lambda p: p["worker"].update(gpu_name="NVIDIA H100"),
                   lambda p: p["worker"].update(machine_id=p["head"]["machine_id"]),
                   lambda p: p["worker"]["container"]["env"].update(TF_GLM_RANK="0"),
                   lambda p: p["worker"]["container"].update(running=False),
                   lambda p: p["worker"]["image"].update(content_sha256="f" * 64),
                   lambda p: p["worker"]["container"]["mounts"][0].update(RW=True),
                   lambda p: p["worker"]["weights"].update(model="9" * 40),
                   lambda p: p["worker"]["container"]["env"].update(NCCL_COMM_ID="192.0.2.20:29583")]
        for defect in defects:
            report = evidence()
            defect(report["pair"])
            with self.subTest(defect=defect), self.assertRaises((deployment.DeploymentError, ValueError)):
                capture.capture_baseline_config(**request(), execute=True, runner=result_runner(report))

    def test_never_fills_missing_required_configuration_or_hardware(self):
        for remove in (lambda r: r["env"].pop("CHAT_TEMPLATE"),
                       lambda r: r["pair"]["head"].pop("gpu_uuid"),
                       lambda r: r["pair"]["worker"].pop("runtime_sha256")):
            report = evidence()
            remove(report)
            with self.subTest(remove=remove), self.assertRaises((deployment.DeploymentError, ValueError)):
                capture.capture_baseline_config(**request(), execute=True, runner=result_runner(report))

    def test_rejects_remote_error_missing_or_ambiguous_marker(self):
        cases = [(1, "SPECPILOT_CAPTURE_RESULT={\"error\":\"not running\"}\n"),
                 (0, "ordinary log only\n"), (0, "SPECPILOT_CAPTURE_RESULT={}\nSPECPILOT_CAPTURE_RESULT={}\n")]
        for code, stdout in cases:
            def run(argv, **kwargs):
                return subprocess.CompletedProcess(argv, code, stdout, "private detail")
            with self.subTest(code=code, stdout=stdout), self.assertRaises(deployment.DeploymentError):
                capture.capture_baseline_config(**request(), execute=True, runner=run)

    def test_default_transport_uses_explicit_utf8(self):
        report = evidence()
        with patch("subprocess.run", return_value=subprocess.CompletedProcess(
                [], 0, "SPECPILOT_CAPTURE_RESULT=" + json.dumps(report), "")) as run:
            capture.capture_baseline_config(**request(), execute=True)
        self.assertEqual(run.call_args.kwargs["encoding"], "utf-8")

    def test_untrusted_request_syntax_is_validated_before_plan(self):
        for change in ({"recipe_path": "/trusted/../other"}, {"container_name": "name;bad"},
                       {"transaction_id": "with spaces"}, {"head": {"ssh_target": "-oProxyCommand=bad"}},
                       {"head": {"ssh_target": "host", "ssh_options": ["-o", "StrictHostKeyChecking=no"]}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                capture.capture_baseline_config(**{**request(), **change})


class EnvironmentReadTests(unittest.TestCase):
    def test_values_are_literal_and_only_requested_names_are_read(self):
        calls = []
        def run(argv, **kwargs):
            calls.append((argv, kwargs))
            return subprocess.CompletedProcess(argv, 0, b"SAFE\0literal $(not-run); ' quoted\0EMPTY\0\0", b"")
        values = capture._read_recipe_env("/trusted/recipe", ["SAFE", "EMPTY"], ["SAFE", "EMPTY"], run)
        self.assertEqual(values, {"SAFE": "literal $(not-run); ' quoted", "EMPTY": ""})
        self.assertNotIn("shell", calls[0][1])
        self.assertIn('source "$1/scripts/config.sh" >/dev/null', calls[0][0][2])

    def test_missing_values_and_non_allowlisted_values_are_refused(self):
        for raw in (b"SAFE\0yes\0", b"SAFE\0yes\0SECRET\0do-not-return\0", b"SAFE\0line1\nline2\0EMPTY\0\0"):
            def run(argv, **kwargs):
                return subprocess.CompletedProcess(argv, 0, raw, b"")
            with self.subTest(raw=raw), self.assertRaises(RuntimeError):
                capture._read_recipe_env("/trusted/recipe", ["SAFE", "EMPTY"], ["SAFE", "EMPTY"], run)

    def test_config_failure_does_not_expose_stderr_secrets(self):
        def run(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 1, b"", b"TOKEN=private-secret")
        with self.assertRaises(RuntimeError) as raised:
            capture._read_recipe_env("/trusted/recipe", ["SAFE"], ["SAFE"], run)
        self.assertNotIn("private-secret", str(raised.exception))

    @unittest.skipIf(os.name == "nt", "the remote recipe runs Bash on Linux; exercised in Linux/WSL tests")
    def test_real_bash_sources_fixture_without_exporting_secret_or_evaluating_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            (root / "scripts" / "config.sh").write_text(
                "SAFE='literal $(touch never-created); 中文'\nEMPTY=''\nSECRET='private'\nprintf 'ignored stdout'\n",
                encoding="utf-8")
            values = capture._read_recipe_env(str(root), ["SAFE", "EMPTY"], ["SAFE", "EMPTY"])
            self.assertEqual(values, {"SAFE": "literal $(touch never-created); 中文", "EMPTY": ""})
            self.assertFalse((root / "never-created").exists())


class CollectSnapshotTests(unittest.TestCase):
    def collect(self, *, env_update=None, mutate=None):
        report = evidence()
        raw_env = {**report["env"], "CONTEXT": "auto", "PARALLEL": "auto", "MODEL_SOURCE": "download",
                   "WORKER_WEIGHTS": "rsync", "CONTEXT_RETRY": "1", "MEMGUARD": "0"}
        raw_env.update(env_update or {})
        calls = []
        count = {"head": 0, "worker": 0}

        def run(argv, **kwargs):
            calls.append((argv, kwargs))
            if argv[:3] == ["docker", "container", "inspect"]:
                return subprocess.CompletedProcess(argv, 0, json.dumps(report["image"]), "")
            role = "worker" if argv[0] == "ssh" else "head"
            count[role] += 1
            value = copy.deepcopy(report["pair"][role])
            if mutate:
                mutate(role, count[role], value)
            return subprocess.CompletedProcess(argv, 0, json.dumps(value), "")

        with patch.object(capture, "_read_recipe_env", return_value=raw_env), patch("subprocess.run", side_effect=run):
            result = capture._collect_capture(request(), "read-only probe placeholder")
        return result, calls

    def test_auto_context_parallel_come_from_both_live_commands_and_only_safety_overrides_change(self):
        result, calls = self.collect()
        self.assertEqual(result["env"], evidence()["env"])
        self.assertEqual(len(calls), 5)  # One selected inspect and two snapshots of each rank.
        self.assertEqual(calls[0][0][:4], ["docker", "container", "inspect", "--format"])
        q = json.loads(calls[3][1]["input"])
        self.assertIn("expected_files_sha256", q)
        self.assertEqual(q["normalization_paths"]["/trusted/recipe"], "<RECIPE>")
        self.assertEqual(json.loads(calls[4][1]["input"])["weights"]["model"], "/trusted/worker_model_dir")

    def test_explicit_config_runtime_disagreement_is_not_silently_replaced(self):
        with self.assertRaisesRegex(RuntimeError, "running CONTEXT"):
            self.collect(env_update={"CONTEXT": "8192"})

    def test_two_ranks_must_agree_on_actual_parallelism(self):
        def mutate(role, count, value):
            if role == "worker":
                cmd = value["container"]["cmd"]
                cmd[cmd.index("--parallel") + 1] = "4"
        with self.assertRaisesRegex(RuntimeError, "ranks disagree"):
            self.collect(mutate=mutate)

    def test_runtime_or_inventory_changes_during_capture_are_rejected(self):
        for changed in ("runtime_sha256", "files_sha256"):
            def mutate(role, count, value):
                if role == "head" and count == 2:
                    if changed == "files_sha256":
                        value[changed]["scripts/config.sh"] = "f" * 64
                    else:
                        value[changed] = "f" * 64
            with self.subTest(changed=changed), self.assertRaisesRegex(RuntimeError, "changed during capture"):
                self.collect(mutate=mutate)


if __name__ == "__main__":
    unittest.main()
