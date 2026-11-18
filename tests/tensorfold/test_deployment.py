"""CPU-only contract tests: no SSH, Docker, model, or GPU is contacted."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from specpilot_tensorfold.deployment import (
    DeploymentConfig, DeploymentError, REQUIRED_ENV, REQUIRED_FILES,
    TensorFoldDeployment, _PROBE, _RemoteController, NATIVE_ENV_MAP, TUNING_ENV, expected_command,
)


def configuration():
    env = {name: "0" for name in REQUIRED_ENV}
    env.update({
        "MODEL_SOURCE": "local", "MODEL_ID": "example/frozen-exl3", "MODEL_REVISION": "1" * 40,
        "DFLASH2_ID": "example/frozen-drafter", "DFLASH2_REVISION": "2" * 40,
        "WORKER_WEIGHTS": "check", "TP": "2", "CONTEXT": "1048576", "CONTEXT_RETRY": "0",
        "PARALLEL": "8", "KV": "fp8", "KV_PAGES": "on", "DENSE": "q8", "HEAD": "q8",
        "DRAFTER": "dflash2", "DRAFTER_BITS": "4", "HOST": "127.0.0.1", "PORT": "8888",
        "NCCL_PORT": "29583", "CONTAINER_NAME": "frozen-base", "SERVED_NAME": "example-model",
        "REASONING_EFFORT": "high", "CHAT_TEMPLATE": "auto", "MEMGUARD": "1",
        "MEMORY_RESERVE_GIB": "12.5",
        "TF_BIN": "/usr/local/bin/tensorfold-native", "CONFIG_OVERLAY": "auto", "SERVE_EXTRA_ARGS": "",
        "MAX_TOKENS": "8192", "VISION_MAX_IMAGES": "4", "MTP_DRAFTS": "", "MTP_CONFIDENCE": "",
        "PROMPT_CACHE_GIB": "", "PREFILL_SHARE": "0.68", "PREFILL_BUDGET": "", "NCCL_CHANNELS": "2",
    })
    for name in ("MODEL_DIR", "DFLASH2_DIR", "WORKER_MODEL_DIR", "WORKER_DFLASH2_DIR", "CACHE_DIR",
                 "STATE_DIR", "LOG_DIR", "WORKER_CACHE_DIR", "WORKER_STATE_DIR", "WORKER_OVERLAY"):
        env[name] = "/frozen/" + name.lower()
    files = {path: "a" * 64 for path in REQUIRED_FILES}
    files.update({".env": None, "scripts/local.sh": None, "patches/engine/0001-frozen.patch": "b" * 64})
    recipe = {
        "recipe_path": "/frozen/recipe", "recipe_commit": "3" * 40,
        "image": "example-tensorfold:baseline", "image_ids": {"head": "sha256:" + "4" * 64, "worker": "sha256:" + "4" * 64},
        "image_labels": {"tf.patches": "a1b2c3d4e5f6", "tf.ref": "5" * 40, "tf.sm": "121", "tf.zig": "0.17.0", "tf.engine": "zig"},
        "files_sha256": files, "env": env,
        "runtime_sha256": {"head": "a" * 64, "worker": "b" * 64},
    }
    candidate = copy.deepcopy(recipe)
    candidate.update(recipe_path="/experiment/recipe", image="example-tensorfold:candidate")
    candidate["image_ids"] = {"head": "sha256:" + "6" * 64, "worker": "sha256:" + "6" * 64}
    candidate["image_labels"]["tf.patches"] = "f6e5d4c3b2a1"
    candidate["env"]["CONTAINER_NAME"] = "frozen-candidate"
    for key in ("CACHE_DIR", "STATE_DIR", "LOG_DIR", "WORKER_CACHE_DIR", "WORKER_STATE_DIR", "WORKER_OVERLAY"):
        candidate["env"][key] = "/experiment/" + key.lower()
    return {"head": {"ssh_target": "user@head.example", "machine_id": "1" * 32, "gpu_uuid": "GPU-" + "1" * 36},
            "worker": {"ssh_target": "user@worker.example", "machine_id": "2" * 32, "gpu_uuid": "GPU-" + "2" * 36},
            "baseline": recipe, "candidate": candidate, "lock_path": "/frozen/cluster.lock", "transaction_id": "cpu-test"}


class FakeRemote(_RemoteController):
    def __init__(self, config, root):
        super().__init__(DeploymentConfig.from_dict(config).data, _PROBE)
        self.owner = Path(root) / "owner.json"
        self.remote_journal = Path(root) / "journal.jsonl"
        self.mutating = True
        self.present = {"baseline": {"head": True, "worker": True}, "candidate": {"head": False, "worker": False}}
        self.calls = []
        self.fail_at = None
        self.mutate_probe = None
        self.built = False

    def probe(self, kind, role, *, include_recipe=True):
        r, e = self.cfg[kind], self.cfg[kind]["env"]
        image_id = r["image_ids"].get(role, "sha256:" + "6" * 64)
        im = {"id": image_id, "labels": r["image_labels"], "content_sha256": kind + "-content"}
        co = None
        if self.present[kind][role]:
            co = {"running": True, "image_id": image_id,
                  "labels": {"tf.recipe": "glm53-spark", "tf.patches": r["image_labels"]["tf.patches"]},
                  "env": {"TF_GLM_RANK": "0" if role == "head" else "1", "TF_GLM_WORLD": "2",
                          "TF_GLM_DENSE_BITS": "8", "TF_GLM_HEAD_BITS": "8", "TF_KV_PAGES": "on"},
                  "cmd": expected_command(e),
                  "mounts": [{"Source": e["MODEL_DIR" if role == "head" else "WORKER_MODEL_DIR"], "Destination": "/model", "RW": False},
                             {"Source": e["DFLASH2_DIR" if role == "head" else "WORKER_DFLASH2_DIR"], "Destination": "/dflash2", "RW": False}]}
            co["env"].update({v: e[k] for k, v in NATIVE_ENV_MAP.items() if k in e and e[k]})
            co["env"].update({k: e[k] for k in TUNING_ENV if k in e})
            co["env"]["NCCL_COMM_ID"] = "192.0.2.1:" + e["NCCL_PORT"]
        p = {"recipe_commit": r["recipe_commit"], "files_sha256": r["files_sha256"],
             "recipe_label": r["image_labels"]["tf.patches"], "image": im, "container": co,
             "weights": {"model": e["MODEL_REVISION"], "drafter": e["DFLASH2_REVISION"]},
             "machine_id": self.cfg[role]["machine_id"], "gpu_uuid": self.cfg[role]["gpu_uuid"], "gpu_name": "NVIDIA GB10",
             "runtime_sha256": r["runtime_sha256"][role], "unreproducible_env": [],
             "gpu_workloads": {"known": True, "other_containers": [], "unmanaged_processes": []},
             "api": {"health": {"status": "ok", "live": {"connections": 0}},
                     "v1/models": {"data": [{"id": e["SERVED_NAME"]}]}}}
        p = copy.deepcopy(p)
        if self.mutate_probe:
            self.mutate_probe(kind, role, p)
        return p

    def node_command(self, role, args, timeout):
        self.calls.append(("candidate", args[1], (role,)))
        if args[1] == "rm":
            self.present["candidate"][role] = False
        elif args[1] in {"stop", "kill"}:
            # Presence is enough for test ownership; the mock removes the stopped
            # container immediately and the subsequent rm is still exercised.
            self.present["candidate"][role] = False
        return ""

    def recipe(self, kind, script, *args, timeout=600):
        self.calls.append((kind, script, args))
        self.event("fake_recipe", kind=kind, script=script)
        if (kind, script) == self.fail_at:
            if kind == "candidate" and script == "start.sh":
                self.present[kind]["worker"] = True  # rank 1 launched before head failed
            raise DeploymentError("injected failure")
        if script == "stop.sh":
            self.present[kind] = {"head": False, "worker": False}
        elif script == "start.sh":
            other = "candidate" if kind == "baseline" else "baseline"
            if any(self.present[other].values()):
                raise AssertionError("two workloads would overlap")
            self.present[kind] = {"head": True, "worker": True}
        elif script == "scripts/prepare.sh":
            if any(any(v.values()) for v in self.present.values()):
                raise AssertionError("building while a workload is serving")
            self.built = True


class ConfigTests(unittest.TestCase):
    def test_plan_is_the_default_and_never_calls_runner(self):
        def forbidden(*a, **k):
            raise AssertionError("runner called in plan mode")
        c = TensorFoldDeployment(DeploymentConfig.from_dict(configuration()), forbidden)
        for action in ("inspect", "prepare", "activate", "restore"):
            p = getattr(c, action)()
            self.assertFalse(p["execution_enabled"])
            self.assertEqual(p["mutates_remote"], action != "inspect")
            self.assertEqual(p["argv"][-1], "bash -s")
            self.assertIn("StrictHostKeyChecking=yes", p["argv"])
            self.assertEqual(p["script_sha256"], hashlib.sha256(p["stdin"].encode()).hexdigest())
            code = p["stdin"].split("SPECPILOT_FROZEN_DRIVER'\n", 1)[1].rsplit("\nSPECPILOT_FROZEN_DRIVER", 1)[0]
            ast.parse(code)
        ast.parse(_PROBE)

    def test_candidate_optional_for_read_only_inspect(self):
        d = configuration()
        d["candidate"] = None
        c = TensorFoldDeployment(DeploymentConfig.from_dict(d))
        self.assertEqual(c.inspect()["action"], "inspect")
        with self.assertRaises(ValueError):
            c.prepare()

    def test_prepare_may_build_but_activate_requires_both_image_ids(self):
        d = configuration()
        d["candidate"]["image_ids"] = {}
        c = TensorFoldDeployment(DeploymentConfig.from_dict(d))
        self.assertEqual(c.prepare()["action"], "prepare")
        with self.assertRaisesRegex(ValueError, "freeze the prepare result"):
            c.activate()

    def test_untrusted_values_cannot_escape_generated_script(self):
        d = configuration()
        d["candidate"]["recipe_path"] = "/experiment/has 'quote'; $(touch BAD)"
        p = TensorFoldDeployment(DeploymentConfig.from_dict(d)).prepare()
        self.assertNotIn("$(touch BAD)", p["stdin"])
        self.assertNotIn("shell=True", p["stdin"])

    def test_frozen_identity_rejections(self):
        cases = [
            lambda d: d["baseline"]["image_ids"].pop("worker"),
            lambda d: d["baseline"]["env"].update(TENSORFOLD_API_KEY="secret"),
            lambda d: d["baseline"]["env"].update(CONTEXT="auto"),
            lambda d: d["candidate"]["env"].update(CONTEXT="8192"),
            lambda d: d["candidate"]["env"].update(HEAD="q4"),
            lambda d: d["candidate"]["env"].update(MEMORY_RESERVE_GIB="20"),
            lambda d: d["candidate"]["env"].update(HOST_CACHE_GIB="3"),
            lambda d: d["candidate"]["env"].update(CONTAINER_NAME="frozen-base"),
            lambda d: d["candidate"]["env"].update(CACHE_DIR=d["baseline"]["env"]["CACHE_DIR"]),
            lambda d: d["baseline"]["files_sha256"].pop(".env"),
            lambda d: d["baseline"]["files_sha256"].update({"../outside": "a" * 64}),
            lambda d: d["head"].update(ssh_options=["-o", "ProxyCommand=arbitrary-command"]),
            lambda d: d["head"].update(ssh_options=["-o", "StrictHostKeyChecking=no"]),
            lambda d: d["head"].update(machine_id=d["worker"]["machine_id"]),
            lambda d: d["head"].update(gpu_uuid=d["worker"]["gpu_uuid"]),
        ]
        for change in cases:
            with self.subTest(change=change):
                d = configuration()
                change(d)
                with self.assertRaises(ValueError):
                    DeploymentConfig.from_dict(d)

    def test_runner_contract_and_local_journal(self):
        calls = []
        def runner(argv, *, input, timeout):
            calls.append((argv, input, timeout))
            return subprocess.CompletedProcess(argv, 0, 'SPECPILOT_DEPLOYMENT_RESULT={"state":"baseline"}\n', '')
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "journal.jsonl"
            c = TensorFoldDeployment(DeploymentConfig.from_dict(configuration()), runner, execute=True, journal_path=journal)
            self.assertEqual(c.inspect()["state"], "baseline")
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(journal.read_text().splitlines()), 2)

    def test_transport_failure_marks_state_unknown(self):
        def runner(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, 1)
        c = TensorFoldDeployment(DeploymentConfig.from_dict(configuration()), runner, execute=True)
        with self.assertRaisesRegex(DeploymentError, "state is unknown"):
            c.inspect()
        self.assertEqual(c.journal[-1]["event"], "transport_failed")


class TransactionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.c = FakeRemote(configuration(), self.temp.name)

    def test_prepare_stops_before_build_and_restores(self):
        r = self.c.run_locked("prepare")
        self.assertEqual(r["state"], "baseline")
        self.assertEqual([(k, s) for k, s, _ in self.c.calls], [
            ("baseline", "stop.sh"), ("candidate", "scripts/prepare.sh"), ("baseline", "start.sh")])
        self.assertFalse(self.c.owner.exists())
        self.assertTrue(self.c.built)

    def test_prepare_failure_recovers_and_preserves_original_error(self):
        self.c.fail_at = ("candidate", "scripts/prepare.sh")
        with self.assertRaisesRegex(DeploymentError, "injected failure"):
            self.c.run_locked("prepare")
        self.assertTrue(all(self.c.present["baseline"].values()))
        self.assertFalse(self.c.owner.exists())
        self.assertIn("transaction_failed", [e["event"] for e in self.c.events])

    def test_activate_retains_reservation_restore_releases_it(self):
        r = self.c.run_locked("activate")
        self.assertEqual(r["state"], "candidate")
        self.assertTrue(self.c.owner.exists())
        self.assertFalse(any(self.c.present["baseline"].values()))
        r = self.c.run_locked("restore")
        self.assertEqual(r["state"], "baseline")
        self.assertFalse(self.c.owner.exists())
        self.assertFalse(any(self.c.present["candidate"].values()))

    def test_partial_rank_launch_failure_is_cleaned_before_baseline(self):
        self.c.fail_at = ("candidate", "start.sh")
        with self.assertRaisesRegex(DeploymentError, "injected failure"):
            self.c.run_locked("activate")
        self.assertEqual([(k, s) for k, s, _ in self.c.calls], [
            ("baseline", "stop.sh"), ("candidate", "start.sh"), ("candidate", "stop"),
            ("candidate", "logs"), ("candidate", "rm"), ("baseline", "start.sh")])
        self.assertFalse(any(self.c.present["candidate"].values()))
        self.assertTrue(all(self.c.present["baseline"].values()))

    def test_different_transaction_cannot_steal_reservation(self):
        self.c.reserve()
        d = configuration()
        d["transaction_id"] = "other"
        other = FakeRemote(d, self.temp.name)
        with self.assertRaisesRegex(DeploymentError, "reserved"):
            other.run_locked("activate")
        self.assertEqual(other.calls, [])
        self.assertEqual(json.loads(self.c.owner.read_text())["transaction_id"], "cpu-test")

    def test_same_transaction_cannot_change_frozen_config(self):
        self.c.reserve()
        d = configuration()
        d["candidate"]["env"]["TF_GLM_L2PF"] = "1"
        with self.assertRaisesRegex(DeploymentError, "different frozen config"):
            FakeRemote(d, self.temp.name).run_locked("activate")

    def test_identity_failures_do_not_stop_baseline(self):
        def change_content(kind, role, p):
            if kind == "baseline" and role == "worker":
                p["image"]["content_sha256"] = "different"
        self.c.mutate_probe = change_content
        with self.assertRaisesRegex(DeploymentError, "content differs"):
            self.c.run_locked("activate")
        self.assertEqual(self.c.calls, [])
        self.assertFalse(self.c.owner.exists())

    def test_file_inventory_change_fails_closed(self):
        def changed(kind, role, p):
            if role == "head":
                p["files_sha256"]["scripts/new-hook.sh"] = "c" * 64
        self.c.mutate_probe = changed
        with self.assertRaisesRegex(DeploymentError, "inventory changed"):
            self.c.verify("baseline", running=True)

    def test_busy_baseline_not_stopped(self):
        def busy(kind, role, p):
            p["api"]["health"]["live"]["connections"] = 1
        self.c.mutate_probe = busy
        with self.assertRaisesRegex(DeploymentError, "zero live connections"):
            self.c.run_locked("activate")
        self.assertEqual(self.c.calls, [])

    def test_runtime_context_and_precision_checked(self):
        for mutate in (lambda p: p["container"]["cmd"].__setitem__(p["container"]["cmd"].index("--context") + 1, "8192"),
                       lambda p: p["container"]["env"].update(TF_GLM_HEAD_BITS="4"),
                       lambda p: p.update(runtime_sha256="c" * 64),
                       lambda p: p["container"]["env"].update(TF_GLM_W8A8="changed"),
                       lambda p: p["container"]["cmd"].__setitem__(p["container"]["cmd"].index("--drafter-bits") + 1, "0"),
                       lambda p: p["container"]["mounts"][0].update(RW=True)):
            with self.subTest(mutate=mutate):
                self.c.mutate_probe = lambda kind, role, p: mutate(p) if kind == "baseline" else None
                with self.assertRaises(DeploymentError):
                    self.c.verify("baseline", running=True)

    def test_unknown_or_other_gpu_workloads_fail_before_stopping(self):
        for mutate in (lambda p: p["gpu_workloads"].update(known=False),
                       lambda p: p["gpu_workloads"].update(other_containers=["unrelated"]),
                       lambda p: p["gpu_workloads"].update(unmanaged_processes=[{"pid": 42}]),
                       lambda p: p.update(unreproducible_env=["TENSORFOLD_API_KEY"])):
            with self.subTest(mutate=mutate):
                self.c.mutate_probe = lambda kind, role, p: mutate(p)
                with self.assertRaises(DeploymentError):
                    self.c.run_locked("activate")
                self.assertEqual(self.c.calls, [])
                self.assertFalse(self.c.owner.exists())

    def test_physical_identity_or_mixed_pair_rejected(self):
        for mutate in (lambda p: p.update(gpu_uuid="GPU-" + "3" * 36),
                       lambda p: p.update(machine_id="3" * 32),
                       lambda p: p["container"]["env"].update(NCCL_COMM_ID="192.0.2.99:29583")):
            self.c.mutate_probe = lambda kind, role, p: mutate(p) if role == "worker" else None
            with self.assertRaises(DeploymentError):
                self.c.run_locked("activate")
            self.assertFalse(self.c.owner.exists())

    def test_recovery_does_not_read_broken_candidate_source(self):
        original = self.c.probe
        def broken(kind, role, *, include_recipe=True):
            if kind == "candidate" and include_recipe and self.c.built:
                raise DeploymentError("candidate source now unreadable")
            return original(kind, role, include_recipe=include_recipe)
        self.c.probe = broken
        with self.assertRaisesRegex(DeploymentError, "source now unreadable"):
            self.c.run_locked("prepare")
        self.assertTrue(all(self.c.present["baseline"].values()))
        self.assertFalse(self.c.owner.exists())

    def test_existing_reservation_is_kept_on_preflight_failure(self):
        self.c.reserve()
        self.c.mutate_probe = lambda kind, role, p: p["gpu_workloads"].update(known=False)
        with self.assertRaises(DeploymentError):
            self.c.run_locked("activate")
        self.assertTrue(self.c.owner.exists())

    def test_prepare_restoration_failure_does_not_restart_its_time_budget(self):
        self.c.fail_at = ("baseline", "start.sh")
        def forbidden_second_recovery(timeout):
            self.fail("the one restore budget must not be restarted after normal restoration fails")
        self.c.recover = forbidden_second_recovery
        with self.assertRaisesRegex(DeploymentError, "baseline restoration failed.*reservation retained"):
            self.c.run_locked("prepare")
        self.assertTrue(self.c.owner.exists())
        self.assertEqual(sum(k == "baseline" and s == "start.sh" for k, s, _ in self.c.calls), 1)

    def test_recovery_failure_retains_owner_and_both_errors(self):
        self.c.fail_at = ("candidate", "start.sh")
        original = self.c.recover
        def failed_recovery(timeout):
            raise DeploymentError("rollback unavailable")
        self.c.recover = failed_recovery
        with self.assertRaisesRegex(DeploymentError, "injected failure.*rollback unavailable.*retained"):
            self.c.run_locked("activate")
        self.assertTrue(self.c.owner.exists())
        self.assertEqual(self.c.events[-1]["event"], "recovery_failed")
        self.c.recover = original


if __name__ == "__main__":
    unittest.main()
