"""Read-only capture of an existing TensorFold two-Spark deployment.

No optimization objective, candidate, workload, build or restart is created.
``scripts/config.sh`` and its local includes are explicitly trusted recipe code.
Capturing does not change them: the four safety overrides below constrain a
future deployment and are not claims about how the historical start was run.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import shlex
import subprocess
from typing import Any

from . import deployment


ENV_ALLOWLIST = deployment.ENV_ALLOWLIST
REQUIRED_ENV = deployment.REQUIRED_ENV
runtime_probe_request = deployment.runtime_probe_request


SAFETY_OVERRIDES = {
    "MODEL_SOURCE": "local", "WORKER_WEIGHTS": "check", "CONTEXT_RETRY": "0", "MEMGUARD": "1",
}
_RESULT_PREFIX = "SPECPILOT_CAPTURE_RESULT="


def _read_recipe_env(recipe_path: str, allowed: list[str], required: list[str], run=None) -> dict:
    """Source only the trusted recipe, returning only allowlisted existing vars.

    NUL framing preserves literal spaces, quotes and shell metacharacters. The
    shell never evaluates a value returned by the recipe a second time. Remote
    login environment overrides are removed, so the captured files define the
    settings; effective runtime identity is checked separately against Docker.
    """
    run = run or subprocess.run
    clean = {k: os.environ[k] for k in ("HOME", "USER", "LOGNAME", "PATH", "LANG", "LC_ALL", "XDG_CACHE_HOME")
             if k in os.environ}
    script = r'''set -euo pipefail
source "$1/scripts/config.sh" >/dev/null
for _specpilot_key in "${@:2}"; do
  if [[ -v "$_specpilot_key" ]]; then
    printf '%s\0%s\0' "$_specpilot_key" "${!_specpilot_key}"
  fi
done
'''
    result = run(["bash", "-c", script, "specpilot-capture-env", recipe_path, *allowed],
                 env=clean, capture_output=True, timeout=45)
    if result.returncode:
        # stderr from trusted local.sh may still contain sensitive values.
        raise RuntimeError("trusted recipe config could not be read; inspect it directly on the head")
    raw = result.stdout
    if not isinstance(raw, bytes) or not raw.endswith(b"\0"):
        raise RuntimeError("recipe environment has invalid framing")
    pieces = raw[:-1].split(b"\0")
    if len(pieces) % 2:
        raise RuntimeError("recipe environment has invalid framing")
    values = {}
    for offset in range(0, len(pieces), 2):
        key, value = pieces[offset].decode("utf-8"), pieces[offset + 1].decode("utf-8")
        if key not in allowed or key in values:
            raise RuntimeError("recipe environment contains an unexpected or duplicate name")
        if "\n" in value or "\r" in value:
            raise RuntimeError("recipe environment values must be single-line strings")
        values[key] = value
    missing = set(required) - values.keys()
    if missing:
        raise RuntimeError("recipe config does not define required settings: " + ", ".join(sorted(missing)))
    return values


def _runtime_number(container: dict, flag: str) -> str:
    cmd = container.get("cmd")
    if not isinstance(cmd, list) or cmd.count(flag) != 1:
        raise RuntimeError("running container must contain exactly one " + flag)
    pos = cmd.index(flag) + 1
    if pos == len(cmd) or not isinstance(cmd[pos], str) or not cmd[pos].isdigit() or int(cmd[pos]) <= 0:
        raise RuntimeError("running container has no explicit positive " + flag)
    return cmd[pos]


def _collect_capture(request: dict, probe_source: str) -> dict:
    """Runs on the head; only config reads, Docker inspect and read-only probes."""
    env_before = _read_recipe_env(request["recipe_path"], sorted(ENV_ALLOWLIST), sorted(REQUIRED_ENV))
    if env_before["CONTAINER_NAME"] != request["container_name"]:
        raise RuntimeError("requested container name differs from the trusted recipe config")
    clean = {k: os.environ[k] for k in ("HOME", "USER", "LOGNAME", "PATH", "LANG", "LC_ALL", "XDG_CACHE_HOME")
             if k in os.environ}
    result = subprocess.run(
        ["docker", "container", "inspect", "--format", "{{json .Config.Image}}", request["container_name"]],
        capture_output=True, text=True, encoding="utf-8", timeout=45, env=clean)
    if result.returncode:
        raise RuntimeError("existing head container could not be inspected")
    image = json.loads(result.stdout)
    if not isinstance(image, str) or not image:
        raise RuntimeError("existing head container has no image reference")
    recipe = {"recipe_path": request["recipe_path"], "image": image, "env": dict(env_before)}

    def probe(role, expected_files=None):
        q = runtime_probe_request(recipe, role)
        if expected_files is not None and role == "head":
            q["expected_files_sha256"] = expected_files
            q["label_env"] = recipe["env"]
        argv = ["python3", "-c", probe_source]
        if role == "worker":
            node = request["worker"]
            argv = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                    *node["ssh_options"], "--", node["ssh_target"], shlex.join(argv)]
        result = subprocess.run(argv, input=json.dumps(q), capture_output=True, text=True, encoding="utf-8",
                                timeout=120, env=clean)
        if result.returncode:
            raise RuntimeError(role + " read-only capture probe failed")
        value = json.loads(result.stdout)
        if not isinstance(value, dict):
            raise RuntimeError(role + " capture probe did not return an object")
        return value

    before = {role: probe(role) for role in ("head", "worker")}
    for role, snapshot in before.items():
        co = snapshot.get("container")
        if not isinstance(co, dict) or co.get("running") is not True:
            raise RuntimeError(role + " baseline container is not running")
    for key, flag in (("CONTEXT", "--context"), ("PARALLEL", "--parallel")):
        actual = _runtime_number(before["head"]["container"], flag)
        if _runtime_number(before["worker"]["container"], flag) != actual:
            raise RuntimeError("the two running ranks disagree on " + key)
        if env_before[key] not in ("auto", actual):
            raise RuntimeError("recipe config disagrees with the running " + key)
        recipe["env"][key] = actual
    recipe["env"].update(SAFETY_OVERRIDES)
    after = {role: probe(role, before["head"]["files_sha256"]) for role in ("head", "worker")}
    env_after = _read_recipe_env(request["recipe_path"], sorted(ENV_ALLOWLIST), sorted(REQUIRED_ENV))
    if env_before != env_after:
        raise RuntimeError("recipe settings changed during capture; no config was frozen")
    for role in ("head", "worker"):
        # Health counters are allowed to change while an existing service runs.
        for key in ("machine_id", "gpu_uuid", "image", "container", "weights", "runtime_sha256"):
            if before[role].get(key) != after[role].get(key):
                raise RuntimeError(role + " serving identity changed during capture")
    for key in ("recipe_commit", "files_sha256"):
        if before["head"].get(key) != after["head"].get(key):
            raise RuntimeError("recipe inventory changed during capture")
    return {"env": recipe["env"], "image": image, "pair": after}


def _validate_request(head, worker, recipe_path, container_name, lock_path, transaction_id) -> dict:
    request = {"recipe_path": deployment._absolute(recipe_path, "recipe_path"),
               "lock_path": deployment._absolute(lock_path, "lock_path"),
               "container_name": container_name, "transaction_id": transaction_id}
    for role, node in (("head", head), ("worker", worker)):
        if not isinstance(node, dict) or set(node) - {"ssh_target", "ssh_options"}:
            raise ValueError(role + " must contain ssh_target and optional ssh_options")
        target = node.get("ssh_target")
        if not isinstance(target, str) or not deployment._TARGET.fullmatch(target):
            raise ValueError("invalid " + role + " SSH target")
        request[role] = {"ssh_target": target, "ssh_options": deployment._ssh_options(node.get("ssh_options", []))}
    if request["head"]["ssh_target"] == request["worker"]["ssh_target"]:
        raise ValueError("head and worker must have distinct SSH targets")
    if not isinstance(container_name, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", container_name):
        raise ValueError("invalid container_name")
    if not isinstance(transaction_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", transaction_id):
        raise ValueError("transaction_id must be a stable identifier of at most 80 characters")
    return request


def _capture_script(request: dict) -> str:
    # repr embeds data as Python literals; neither a path nor a value is shell-expanded.
    pieces = [
        "import json, os, shlex, subprocess", "from typing import Any",
        "ENV_ALLOWLIST=frozenset(" + repr(sorted(deployment.ENV_ALLOWLIST)) + ")",
        "REQUIRED_ENV=frozenset(" + repr(sorted(deployment.REQUIRED_ENV)) + ")",
        "RUNTIME_KEYS=frozenset(" + repr(sorted(deployment.RUNTIME_KEYS)) + ")",
        "RECONSTRUCTIBLE_KEYS=frozenset(" + repr(sorted(deployment.RECONSTRUCTIBLE_KEYS)) + ")",
        "TUNING_ENV=frozenset(" + repr(sorted(deployment.TUNING_ENV)) + ")",
        "SAFETY_OVERRIDES=" + repr(SAFETY_OVERRIDES),
        inspect.getsource(deployment.runtime_probe_request), inspect.getsource(_read_recipe_env),
        inspect.getsource(_runtime_number), inspect.getsource(_collect_capture),
        "try:", "    result=_collect_capture(" + repr(request) + ", " + repr(deployment._PROBE) + ")",
        "    print(" + repr(_RESULT_PREFIX) + "+json.dumps(result,sort_keys=True))",
        "except Exception as exc:",
        "    print(" + repr(_RESULT_PREFIX) + "+json.dumps({'error':str(exc)}))",
        "    raise SystemExit(1)",
    ]
    return "set -euo pipefail\npython3 - <<'SPECPILOT_CAPTURE_DRIVER'\n" + "\n".join(pieces) + "\nSPECPILOT_CAPTURE_DRIVER\n"


def _configuration_from_capture(request: dict, report: dict) -> dict:
    pair, env = report["pair"], report["env"]
    if set(pair) != {"head", "worker"} or not isinstance(env, dict):
        raise deployment.DeploymentError("capture is missing the two-rank evidence")
    if any(env.get(key) != value for key, value in SAFETY_OVERRIDES.items()):
        raise deployment.DeploymentError("capture did not freeze the required future-start safeguards")
    head = pair["head"]
    baseline = {"recipe_path": request["recipe_path"], "recipe_commit": head["recipe_commit"],
                "image": report["image"], "env": env, "files_sha256": head["files_sha256"],
                "image_labels": {k: v for k, v in head["image"]["labels"].items() if k.startswith("tf.")},
                "image_ids": {role: p["image"]["id"] for role, p in pair.items()},
                "runtime_sha256": {role: p["runtime_sha256"] for role, p in pair.items()}}
    config = {"baseline": baseline, "candidate": None, "lock_path": request["lock_path"],
              "transaction_id": request["transaction_id"]}
    for role, snapshot in pair.items():
        config[role] = {**request[role], "machine_id": snapshot["machine_id"], "gpu_uuid": snapshot["gpu_uuid"]}
    validated = deployment.DeploymentConfig.from_dict(config).data

    class CapturedController(deployment._RemoteController):
        def probe(self, kind, role):
            if kind != "baseline":
                raise deployment.DeploymentError("capture does not define a candidate")
            return pair[role]

    # This verifier consumes already captured metadata only. It does not inspect
    # local files at Linux paths, write ownership records or issue commands.
    verifier = CapturedController(validated, deployment._PROBE)
    verifier.verify("baseline", running=True)
    comm = [pair[role]["container"]["env"].get("NCCL_COMM_ID") for role in ("head", "worker")]
    if not all(isinstance(value, str) and value for value in comm) or comm[0] != comm[1]:
        raise deployment.DeploymentError("captured TP ranks do not share the same NCCL rendezvous")
    if comm[0].rsplit(":", 1)[-1] != env["NCCL_PORT"]:
        raise deployment.DeploymentError("captured NCCL rendezvous does not match configured port")
    return validated


def capture_baseline_config(head: dict, worker: dict, recipe_path: str, container_name: str,
                            lock_path: str, transaction_id: str, *, execute: bool = False,
                            runner=None) -> dict:
    """Plan or read an existing baseline; the returned config has candidate=null.

    ``runner(argv, *, input, timeout)`` returns a CompletedProcess. It is never
    called in default plan mode. No configuration file is written by this API.
    """
    request = _validate_request(head, worker, recipe_path, container_name, lock_path, transaction_id)
    script = _capture_script(request)
    node = request["head"]
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", *node["ssh_options"],
            "--", node["ssh_target"], "bash -s"]
    plan = {"action": "capture-config", "execution_enabled": execute, "mutates_remote": False,
            "argv": argv, "stdin": script, "timeout": 600,
            "script_sha256": hashlib.sha256(script.encode()).hexdigest(),
            "transaction_id": transaction_id, "safety_overrides": dict(SAFETY_OVERRIDES),
            "safety_override_scope": "future starts only; existing service is not modified",
            "trusted_code": "the existing recipe scripts/config.sh and its local includes"}
    if not execute:
        return plan
    if runner is None:
        def runner(argv, *, input, timeout):
            return subprocess.run(argv, input=input, text=True, encoding="utf-8",
                                  capture_output=True, timeout=timeout)
    try:
        result = runner(argv, input=script, timeout=plan["timeout"])
    except Exception as exc:
        raise deployment.DeploymentError("read-only capture transport failed") from exc
    lines = [line[len(_RESULT_PREFIX):] for line in result.stdout.splitlines() if line.startswith(_RESULT_PREFIX)]
    if len(lines) != 1:
        raise deployment.DeploymentError("missing or ambiguous read-only capture result")
    report = json.loads(lines[0])
    if not isinstance(report, dict) or result.returncode or "error" in report:
        raise deployment.DeploymentError(report.get("error", "read-only capture failed") if isinstance(report, dict)
                                         else "read-only capture result is not an object")
    try:
        return _configuration_from_capture(request, report)
    except (KeyError, TypeError, AttributeError) as exc:
        raise deployment.DeploymentError("read-only capture evidence is incomplete or invalid") from exc
