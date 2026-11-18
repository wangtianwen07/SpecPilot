"""Opt-in, transactional deployment of a frozen two-Spark TensorFold recipe.

This module does not create an optimization task. Every method is a plan until
``execute=True`` is supplied to the controller. Remote recipes are trusted code:
their complete deployment-file inventory must be frozen before execution.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import signal
import subprocess
import time
from typing import Any, Callable, Mapping


class DeploymentError(RuntimeError):
    """An identity, ownership, command, or recovery check failed."""


ENV_ALLOWLIST = frozenset("""
MODEL_SOURCE MODEL_ID MODEL_REVISION MODEL_DIR DFLASH2_ID DFLASH2_REVISION DFLASH2_DIR
WORKER_MODEL_DIR WORKER_DFLASH2_DIR WORKER_WEIGHTS WORKER_CACHE_DIR WORKER_STATE_DIR
WORKER_OVERLAY CACHE_DIR STATE_DIR LOG_DIR CONTAINER_NAME PORT HOST NCCL_PORT
SERVED_NAME TP CONTEXT CONTEXT_MAX CONTEXT_RETRY PARALLEL KV KV_PAGES DENSE HEAD
DRAFTER DRAFTER_BITS MTP_DRAFTS MTP_CONFIDENCE THINKING REASONING_EFFORT CLEAR_THINKING
MAX_TOKENS VISION VISION_MAX_IMAGES CHAT_TEMPLATE MEMORY_RESERVE_GIB HOST_CACHE_GIB
PROMPT_CACHE_GIB ABSORB_MMA W8A8 CE_EXCHANGE PREFILL_SHARE PREFILL_BUDGET
MEMGUARD MEMGUARD_FLOOR_GIB MEMGUARD_HOLD_S MEMGUARD_INTERVAL_S MIN_AVAIL_GIB
WAIT_TIMEOUT SMOKE_TIMEOUT STOP_TIMEOUT LOG_KEEP ROCE ROCE_MAX_KB NCCL_CHANNELS
FABRIC_PEER FABRIC_SELF NCCL_SOCKET_IFNAME TF_ROCE_HCA NCCL_IB_GID_INDEX
TF_GLM_DRAFT_CONFIDENCE TF_GLM_DRAFT_RULE TF_GLM_DRAFT_LAMBDA TF_GLM_DFLASH2_EDGE
TF_GLM_DFLASH2_TCONF TF_GLM_DFLASH2_Q32 TF_GLM_MM_STAGES TF_GLM_PROF
TF_GLM_DFLASH2_PROF TF_GLM_DFLASH2_TRACE TF_GLM_DFLASH2_FORCE TF_GLM_PDL_CHAIN
TF_GLM_L2PF TF_GLM_L2PF_MB TF_GLM_L2PF_CTAS TF_GLM_L2PF_PACE_GBPS
L2PF CACHE_FIRST KV_SHARD LEARN_GIB LEARN_MIN_FREE_GIB PROMPT_CACHE_MIN_GIB
EFFORT_LINE EXTRA_EOS THINK_OFF_SPLIT
SERVE_EXTRA_ARGS CONFIG_OVERLAY TF_BIN
""".split())
REQUIRED_ENV = frozenset("""
MODEL_SOURCE MODEL_ID MODEL_REVISION MODEL_DIR DFLASH2_ID DFLASH2_REVISION DFLASH2_DIR
WORKER_MODEL_DIR WORKER_DFLASH2_DIR WORKER_WEIGHTS CACHE_DIR STATE_DIR LOG_DIR
WORKER_CACHE_DIR WORKER_STATE_DIR WORKER_OVERLAY CONTAINER_NAME PORT HOST NCCL_PORT
SERVED_NAME TP CONTEXT CONTEXT_RETRY PARALLEL KV KV_PAGES DENSE HEAD DRAFTER
DRAFTER_BITS THINKING REASONING_EFFORT CLEAR_THINKING VISION CHAT_TEMPLATE
MEMORY_RESERVE_GIB HOST_CACHE_GIB ABSORB_MMA W8A8 CE_EXCHANGE MEMGUARD
SERVE_EXTRA_ARGS CONFIG_OVERLAY TF_BIN
MAX_TOKENS VISION_MAX_IMAGES MTP_DRAFTS MTP_CONFIDENCE PROMPT_CACHE_GIB
PREFILL_SHARE PREFILL_BUDGET ROCE ROCE_MAX_KB NCCL_CHANNELS
""".split())
IMMUTABLE_ENV = REQUIRED_ENV - frozenset("""
CACHE_DIR STATE_DIR LOG_DIR WORKER_CACHE_DIR WORKER_STATE_DIR WORKER_OVERLAY
CONTAINER_NAME
""".split())
REQUIRED_FILES = frozenset({
    "start.sh", "stop.sh", "scripts/config.sh", "scripts/prepare.sh", "scripts/nodes.sh",
    "docker/Dockerfile", ".dockerignore", ".env", "scripts/local.sh",
})
REQUIRED_LABELS = frozenset({"tf.patches", "tf.ref", "tf.sm", "tf.zig", "tf.engine"})
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_TARGET = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.@:-]*\Z")

# Recipe names mapped to the runtime names start.sh passes to Docker. Hashing
# covers every other environment value without printing secrets.
NATIVE_ENV_MAP = {
    "MEMORY_RESERVE_GIB": "TENSORFOLD_MEMORY_RESERVE_GIB", "CLEAR_THINKING": "TF_GLM_CLEAR_THINKING",
    "KV_PAGES": "TF_KV_PAGES", "CE_EXCHANGE": "TF_GLM_CE_EXCHANGE", "ABSORB_MMA": "TF_GLM_ABSORB_MMA",
    "W8A8": "TF_GLM_W8A8", "HOST_CACHE_GIB": "TF_GLM_HOST_CACHE_GIB", "PREFILL_SHARE": "TENSORFOLD_PREFILL_SHARE",
    "PREFILL_BUDGET": "TENSORFOLD_PREFILL_BUDGET", "ROCE_MAX_KB": "TF_ROCE_MAX_KB", "TF_BIN": "TF_BIN",
}
TUNING_ENV = frozenset(k for k in ENV_ALLOWLIST if k.startswith("TF_GLM_") and k not in {"TF_GLM_RANK", "TF_GLM_WORLD"})
RUNTIME_KEYS = frozenset(NATIVE_ENV_MAP.values()) | TUNING_ENV | frozenset({
    "TF_GLM_RANK", "TF_GLM_WORLD", "TF_GLM_DENSE_BITS", "TF_GLM_HEAD_BITS", "TF_ROCE",
    "TF_ROCE_HCA", "NCCL_COMM_ID", "NCCL_SOCKET_IFNAME", "NCCL_IB_HCA", "NCCL_IB_GID_INDEX",
    "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS", "TF_PROMPT_CACHE_FIRST", "TF_GLM_KV_SHARD",
    "TF_EXTRA_EOS", "TF_THINK_OFF_SPLIT",
    "TF_CUDA_DEVICE", "TENSORFOLD_NO_UPDATE_CHECK", "TENSORFOLD_NO_LIVE", "TF_REPO", "TF_REF",
})
RECONSTRUCTIBLE_KEYS = RUNTIME_KEYS - frozenset({"TF_PROMPT_CACHE_FIRST", "TF_GLM_KV_SHARD", "TF_EXTRA_EOS", "TF_THINK_OFF_SPLIT"})


def expected_command(env: dict[str, str]) -> list[str]:
    """Reconstruct this frozen Mia recipe's native argv without executing it."""
    cmd = ["serve", "/model"]
    for flag, key in (("--name", "SERVED_NAME"), ("--host", "HOST"), ("--port", "PORT"),
                      ("--context", "CONTEXT"), ("--parallel", "PARALLEL"),
                      ("--max-tokens", "MAX_TOKENS"), ("--kv-cache", "KV")):
        cmd += [flag, env[key]]
    cmd += ["--no-update-check", "--thinking" if env["THINKING"] == "1" else "--no-thinking"]
    if env["REASONING_EFFORT"]:
        cmd += ["--reasoning-effort", env["REASONING_EFFORT"]]
    if env["VISION"] == "1":
        cmd += ["--vision", "--vision-max-images", env["VISION_MAX_IMAGES"]]
    if env["CHAT_TEMPLATE"].startswith("/") or (env["CHAT_TEMPLATE"] == "auto" and env["VISION"] == "1"):
        cmd += ["--chat-template", "/templates/chat_template.jinja"]
    if env["DRAFTER"] == "dflash2":
        cmd += ["--drafter", "/dflash2", "--drafter-bits", env["DRAFTER_BITS"]]
    elif env["DRAFTER"] == "none":
        cmd += ["--no-drafts"]
    else:
        raise DeploymentError("unsupported drafter in frozen recipe")
    if env["DRAFTER"] != "none":
        for flag, key in (("--mtp-drafts", "MTP_DRAFTS"), ("--mtp-confidence", "MTP_CONFIDENCE")):
            if env[key]: cmd += [flag, env[key]]
    if env["PROMPT_CACHE_GIB"]:
        cmd += ["--prompt-cache-gib", env["PROMPT_CACHE_GIB"]]
    # The shell recipe uses read -r -a, which splits whitespace without shell parsing.
    return cmd + env["SERVE_EXTRA_ARGS"].split()


def runtime_probe_request(recipe: dict[str, Any], role: str, *, include_recipe: bool = True,
                          allowed_containers: list[str] | None = None) -> dict[str, Any]:
    """Shared with the read-only capture helper; no command is executed here."""
    e = recipe["env"]
    q = {"recipe": recipe if role == "head" and include_recipe else None,
         "image": recipe["image"], "container": e["CONTAINER_NAME"], "runtime_keys": sorted(RUNTIME_KEYS),
         "tuning_env": sorted(TUNING_ENV), "allowed_containers": allowed_containers or [e["CONTAINER_NAME"]],
         "reconstructible_keys": sorted(RECONSTRUCTIBLE_KEYS),
         "normalization_paths": {},
         "weights": {"model": e["MODEL_DIR" if role == "head" else "WORKER_MODEL_DIR"],
                     "drafter": e["DFLASH2_DIR" if role == "head" else "WORKER_DFLASH2_DIR"]}}
    # Longer paths first when rewriting strings. Keys, rather than values, give
    # baseline and candidate the same logical mount identity.
    keys = ("CACHE_DIR", "STATE_DIR", "LOG_DIR") if role == "head" else ("WORKER_CACHE_DIR", "WORKER_STATE_DIR", "WORKER_OVERLAY")
    q["normalization_paths"] = {e[k]: "<" + k + ">" for k in keys}
    q["normalization_paths"][recipe["recipe_path"]] = "<RECIPE>"
    if include_recipe and role == "head" and recipe.get("files_sha256"):
        q["expected_files_sha256"] = recipe["files_sha256"]
        q["label_env"] = recipe["env"]
    if role == "head":
        host = "127.0.0.1" if e["HOST"] in {"0.0.0.0", "::"} else e["HOST"]
        if ":" in host:
            host = "[" + host + "]"
        q["api"] = "http://" + host + ":" + e["PORT"]
    return q


def _absolute(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.startswith("/") or "\x00" in value or "\n" in value:
        raise ValueError(f"{name} must be an absolute Linux path")
    path = PurePosixPath(value)
    if ".." in path.parts or str(path) != value or value == "/":
        raise ValueError(f"{name} must be normalized and must not be /")
    return value


def _ssh_options(raw: Any) -> list[str]:
    if not isinstance(raw, list) or not all(isinstance(v, str) for v in raw):
        raise ValueError("ssh_options must be a list of strings")
    allowed = {"BatchMode", "ConnectTimeout", "StrictHostKeyChecking", "UserKnownHostsFile",
               "IdentityFile", "IdentitiesOnly", "ServerAliveInterval", "ServerAliveCountMax", "LogLevel"}
    i = 0
    while i < len(raw):
        if raw[i] not in {"-i", "-p", "-o"} or i + 1 == len(raw):
            raise ValueError("ssh_options supports only explicit -i, -p and approved -o pairs")
        flag, value = raw[i:i + 2]
        if not value or any(c.isspace() for c in value) or "\x00" in value:
            raise ValueError("ssh option values must not contain whitespace or NUL")
        if flag == "-p" and not (value.isdigit() and 0 < int(value) < 65536):
            raise ValueError("invalid SSH port")
        if flag == "-o":
            key, sep, val = value.partition("=")
            if not sep or key not in allowed or not val:
                raise ValueError(f"SSH option is not allowed: {key}")
            if key == "StrictHostKeyChecking" and val != "yes":
                raise ValueError("host key checking must remain enabled")
            if key == "BatchMode" and val != "yes":
                raise ValueError("SSH must use BatchMode=yes")
        i += 2
    return raw


@dataclass(frozen=True)
class DeploymentConfig:
    """JSON-compatible frozen input; candidate may be null for read-only inspect."""
    data: dict[str, Any]

    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> "DeploymentConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DeploymentConfig":
        d = json.loads(json.dumps(value))
        allowed = {"head", "worker", "baseline", "candidate", "lock_path", "transaction_id", "timeouts"}
        if set(d) - allowed:
            raise ValueError(f"unknown deployment fields: {sorted(set(d) - allowed)}")
        for role in ("head", "worker"):
            n = d.get(role)
            if not isinstance(n, dict) or set(n) - {"ssh_target", "ssh_options", "machine_id", "gpu_uuid"}:
                raise ValueError(f"{role} must contain SSH and physical hardware identity")
            if not _TARGET.fullmatch(n.get("ssh_target", "")):
                raise ValueError(f"invalid {role}.ssh_target")
            n["ssh_options"] = _ssh_options(n.get("ssh_options", []))
            if not re.fullmatch(r"[0-9a-f]{32}", n.get("machine_id", "")):
                raise ValueError(f"{role}.machine_id must be frozen from /etc/machine-id")
            if not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", n.get("gpu_uuid", "")):
                raise ValueError(f"{role}.gpu_uuid must be the physical GPU UUID")
        if d["head"]["ssh_target"] == d["worker"]["ssh_target"]:
            raise ValueError("head and worker must be distinct SSH targets")
        if d["head"]["machine_id"] == d["worker"]["machine_id"] or d["head"]["gpu_uuid"] == d["worker"]["gpu_uuid"]:
            raise ValueError("head and worker must be physically distinct")
        _absolute(d.get("lock_path"), "lock_path")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", d.get("transaction_id", "")):
            raise ValueError("transaction_id must be a stable identifier of at most 80 characters")
        for kind in ("baseline", "candidate"):
            r = d.get(kind)
            if r is None and kind == "candidate":
                d[kind] = None
                continue
            if not isinstance(r, dict):
                raise ValueError(f"{kind} is required")
            if set(r) - {"recipe_path", "recipe_commit", "image", "image_ids", "image_labels", "files_sha256", "env", "runtime_sha256"}:
                raise ValueError(f"unknown fields in {kind}")
            _absolute(r.get("recipe_path"), f"{kind}.recipe_path")
            if not _COMMIT.fullmatch(r.get("recipe_commit", "")):
                raise ValueError(f"{kind}.recipe_commit must be a full SHA")
            if not re.fullmatch(r"[a-z0-9][a-z0-9._/:@-]+", r.get("image", "")):
                raise ValueError(f"invalid {kind}.image")
            ids = r.setdefault("image_ids", {})
            if not isinstance(ids, dict) or set(ids) - {"head", "worker"}:
                raise ValueError("image_ids must map head/worker to Docker image IDs")
            if kind == "baseline" and set(ids) != {"head", "worker"}:
                raise ValueError("both baseline image IDs must be frozen")
            for image_id in ids.values():
                if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
                    raise ValueError("image IDs must be full sha256 digests")
            runtime = r.get("runtime_sha256", {})
            if not isinstance(runtime, dict) or set(runtime) != {"head", "worker"} or not all(_SHA.fullmatch(v) for v in runtime.values()):
                raise ValueError(f"{kind}.runtime_sha256 must freeze both effective runtime fingerprints")
            labels = r.get("image_labels", {})
            if not isinstance(labels, dict) or not REQUIRED_LABELS <= labels.keys():
                raise ValueError(f"{kind} is missing required image labels")
            if not all(isinstance(k, str) and k.startswith("tf.") and isinstance(v, str) for k, v in labels.items()):
                raise ValueError("image_labels must be string-valued tf.* labels")
            if labels["tf.sm"] != "121" or labels["tf.engine"] != "zig" or not _COMMIT.fullmatch(labels["tf.ref"]):
                raise ValueError("this adapter requires a pinned TensorFold Zig build for SM121")
            files = r.get("files_sha256", {})
            if not isinstance(files, dict) or not REQUIRED_FILES <= files.keys():
                raise ValueError(f"{kind}.files_sha256 must include required scripts and both optional config paths")
            if not any(p.startswith("patches/engine/") for p in files):
                raise ValueError("freeze the complete engine patch inventory")
            for path, digest in files.items():
                p = PurePosixPath(path)
                if p.is_absolute() or ".." in p.parts or str(p) != path or "\x00" in path:
                    raise ValueError("frozen file names must be normalized relative paths")
                if digest is None:
                    if path not in {".env", "scripts/local.sh"}:
                        raise ValueError("only .env and scripts/local.sh may be frozen absent")
                elif not isinstance(digest, str) or not _SHA.fullmatch(digest):
                    raise ValueError("frozen file digests must be SHA256 or an explicit absent null")
            env = r.get("env", {})
            if not isinstance(env, dict) or set(env) - ENV_ALLOWLIST or not REQUIRED_ENV <= env.keys():
                raise ValueError(f"{kind}.env has unknown names or lacks required explicit serving settings")
            if not all(isinstance(v, str) and "\x00" not in v and "\n" not in v for v in env.values()):
                raise ValueError("env values must be single-line strings")
            if (env["MODEL_SOURCE"], env["WORKER_WEIGHTS"], env["TP"], env["CONTEXT_RETRY"], env["MEMGUARD"]) != ("local", "check", "2", "0", "1"):
                raise ValueError("require local weights, check-only worker weights, TP2, no context fallback, and memguard")
            if not env["CONTEXT"].isdigit() or int(env["CONTEXT"]) <= 0:
                raise ValueError("CONTEXT must be explicit; auto fitting is forbidden")
            if not _COMMIT.fullmatch(env["MODEL_REVISION"]) or not _COMMIT.fullmatch(env["DFLASH2_REVISION"]):
                raise ValueError("model and drafter revisions must be pinned")
            for key in ("PORT", "NCCL_PORT"):
                if not env[key].isdigit() or not 0 < int(env[key]) < 65535:
                    raise ValueError(f"invalid {key}")
            if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", env["CONTAINER_NAME"]):
                raise ValueError("invalid container name")
            for key in ("MODEL_DIR", "DFLASH2_DIR", "WORKER_MODEL_DIR", "WORKER_DFLASH2_DIR", "CACHE_DIR", "STATE_DIR", "LOG_DIR", "WORKER_CACHE_DIR", "WORKER_STATE_DIR", "WORKER_OVERLAY"):
                _absolute(env[key], f"{kind}.env.{key}")
        b, c = d["baseline"], d["candidate"]
        if c:
            if b["recipe_path"] == c["recipe_path"] or b["image"] == c["image"]:
                raise ValueError("candidate needs a distinct recipe directory and image reference")
            for key in IMMUTABLE_ENV:
                if b["env"][key] != c["env"][key]:
                    raise ValueError(f"candidate changes frozen serving identity: {key}")
            for key in ("CACHE_DIR", "STATE_DIR", "LOG_DIR", "WORKER_CACHE_DIR", "WORKER_STATE_DIR", "WORKER_OVERLAY"):
                if b["env"][key] == c["env"][key]:
                    raise ValueError(f"candidate must isolate {key}")
            if b["env"]["CONTAINER_NAME"] == c["env"]["CONTAINER_NAME"]:
                raise ValueError("candidate must use a distinct container name")
            if b["image_labels"]["tf.ref"] != c["image_labels"]["tf.ref"]:
                raise ValueError("candidate must keep the frozen TensorFold engine pin")
        defaults = {"inspect": 120, "prepare": 3600, "activate": 900, "restore": 900}
        supplied = d.get("timeouts", {})
        if not isinstance(supplied, dict) or set(supplied) - defaults.keys():
            raise ValueError("unknown deployment timeout")
        defaults.update(supplied)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 1 <= v <= 86400 for v in defaults.values()):
            raise ValueError("timeouts must be finite seconds between 1 and 86400")
        d["timeouts"] = defaults
        return cls(d)


# This helper is sent to each host. It prints selected metadata, never a full
# Docker inspect (which could contain API keys in Config.Env).
_PROBE = r'''
import hashlib, json, os, pathlib, subprocess, sys, urllib.request
q=json.load(sys.stdin)
def call(args, optional=False):
    r=subprocess.run(args, capture_output=True, text=True, encoding='utf-8', timeout=45)
    if r.returncode:
        if optional: return None
        raise RuntimeError('probe command failed: '+args[0]+' '+args[1])
    return r.stdout
def one(kind, name):
    raw=call(['docker',kind,'inspect',name], optional=True)
    return json.loads(raw)[0] if raw else None
r={'machine_id':pathlib.Path('/etc/machine-id').read_text().strip()}
gpu=call(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader']).strip().splitlines()
if len(gpu)!=1: raise RuntimeError('expected exactly one physical GPU on each Spark')
r['gpu_uuid'],r['gpu_name']=[x.strip() for x in gpu[0].split(',',1)]
if q.get('recipe'):
    p=pathlib.Path(q['recipe']['recipe_path'])
    if p.resolve()!=p: raise RuntimeError('recipe path resolves through a symlink')
    r['recipe_commit']=call(['git','-C',str(p),'rev-parse','HEAD']).strip()
    inventory=set()
    for root in ('scripts','docker','patches/engine','templates'):
        for f in (p/root).rglob('*'):
            if f.is_symlink(): raise RuntimeError('symlink in frozen deployment inventory')
            if f.is_file(): inventory.add(f.relative_to(p).as_posix())
    inventory.update(('start.sh','stop.sh','.dockerignore','.env','scripts/local.sh'))
    r['files_sha256']={}
    for name in sorted(inventory):
        f=p/name
        if f.is_symlink(): raise RuntimeError('symlink in frozen deployment files')
        r['files_sha256'][name]=hashlib.sha256(f.read_bytes()).hexdigest() if f.is_file() else None
    if q.get('expected_files_sha256'):
        if r['files_sha256']!=q['expected_files_sha256']: raise RuntimeError('frozen deployment inventory changed before helper execution')
        env={k:os.environ[k] for k in ('HOME','USER','LOGNAME','PATH','LANG','LC_ALL') if k in os.environ}
        env.update(q['label_env'])
        label=subprocess.run(['bash','scripts/prepare.sh','--label'],cwd=str(p),env=env,capture_output=True,text=True,encoding='utf-8',timeout=45)
        if label.returncode:raise RuntimeError('frozen recipe label helper failed')
        r['recipe_label']=label.stdout.strip().splitlines()[-1]
im=one('image',q['image'])
if im:
    runtime=im.get('Config') or {}
    ident={k:runtime.get(k) for k in ('Env','Entrypoint','Cmd','WorkingDir')}
    ident['layers']=(im.get('RootFS') or {}).get('Layers')
    r['image']={'id':im['Id'],'labels':runtime.get('Labels') or {},
                'content_sha256':hashlib.sha256(json.dumps(ident,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
else: r['image']=None
co=one('container',q['container'])
if co:
    e={x.split('=',1)[0]:x.split('=',1)[1] for x in co['Config'].get('Env',[]) if '=' in x}
    r['container']={'image_id':co['Image'],'running':bool(co['State']['Running']),
       'labels':co['Config'].get('Labels') or {},'cmd':co['Config'].get('Cmd') or [],
       'env':{k:v for k,v in e.items() if k in q['runtime_keys']},
       'mounts':[{k:m.get(k) for k in ('Source','Destination','RW')} for m in co.get('Mounts',[])]}
    image_env={x.split('=',1)[0]:x.split('=',1)[1] for x in (im or {}).get('Config',{}).get('Env',[]) if '=' in x}
    r['unreproducible_env']=sorted(k for k,v in e.items() if k.startswith(('TF_','TENSORFOLD_','NCCL_')) and k not in q['reconstructible_keys'] and image_env.get(k)!=v)
    def normalize(v):
        if isinstance(v,str):
            for source,target in sorted(q.get('normalization_paths',{}).items(),key=lambda x:-len(x[0])):
                if v==source or v.startswith(source+'/'): v=target+v[len(source):]
            return v
        if isinstance(v,list):return [normalize(x) for x in v]
        if isinstance(v,dict):return {k:normalize(x) for k,x in v.items()}
        return v
    identity={'cmd':co['Config'].get('Cmd'), 'entrypoint':co['Config'].get('Entrypoint'),
       'env':{k:v for k,v in e.items() if k not in q.get('tuning_env',[])},
       'mounts':sorted(r['container']['mounts'],key=lambda m:m['Destination']),
       'host_config':{k:co.get('HostConfig',{}).get(k) for k in
          ('NetworkMode','IpcMode','ShmSize','Runtime','DeviceRequests','Devices','CapAdd','Ulimits','Privileged')}}
    # Small configuration/template mounts affect output semantics. Freeze their
    # contents, not only their paths; never read full checkpoint/cache payloads.
    identity['mounted_config_sha256']={}
    for mount in r['container']['mounts']:
        if mount['Destination'] in ('/model/config.json','/templates/chat_template.jinja'):
            identity['mounted_config_sha256'][mount['Destination']]=hashlib.sha256(pathlib.Path(mount['Source']).read_bytes()).hexdigest()
    r['runtime_sha256']=hashlib.sha256(json.dumps(normalize(identity),sort_keys=True,separators=(',',':')).encode()).hexdigest()
else:r['container']=None
r['gpu_workloads']={'known':True,'other_containers':[],'unmanaged_processes':[],'observation':'nvml'}
raw=call(['docker','ps','-q'],optional=True)
owned_pids=set()
if raw is None:r['gpu_workloads']['known']=False
else:
    for cid in raw.split():
        c=one('container',cid)
        if not c: r['gpu_workloads']['known']=False; continue
        name=c['Name'].lstrip('/')
        hc=c.get('HostConfig') or {}
        ce={s.split('=',1)[0]:s.split('=',1)[1] for s in c['Config'].get('Env',[]) if '=' in s}
        claims=bool(hc.get('DeviceRequests')) or hc.get('Runtime')=='nvidia' or any('nvidia' in str(x) for x in hc.get('Devices') or [])
        if claims and name not in q.get('allowed_containers',[]):r['gpu_workloads']['other_containers'].append(name)
        if name in q.get('allowed_containers',[]):
            top=call(['docker','top',cid,'-eo','pid'],optional=True)
            if top is None:r['gpu_workloads']['known']=False
            else:owned_pids.update(int(x.strip()) for x in top.splitlines() if x.strip().isdigit())
apps=call(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],optional=True)
observed=set()
nvml_ok=apps is not None and '[N/A]' not in apps and 'Not Supported' not in apps
if nvml_ok:
    for line in apps.splitlines():
        if not line.strip():continue
        pid,_,name=line.partition(',')
        if not pid.strip().isdigit():nvml_ok=False;continue
        observed.add(int(pid.strip()))
        if int(pid.strip()) not in owned_pids:r['gpu_workloads']['unmanaged_processes'].append({'pid':int(pid.strip()),'name':name.strip()})
# GB10 may return an empty, successful compute-apps list while CUDA is active.
# Empty output, or failure to observe the owned serving process, is not proof
# of idleness. Fall back to open device handles and fail closed on permissions.
if not nvml_ok or not observed or (owned_pids and not observed.intersection(owned_pids)):
    r['gpu_workloads']['observation']='proc-device-handles'
    fd_observed=set(); fd_known=True
    for proc in pathlib.Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            for fd in (proc/'fd').iterdir():
                try:target=os.readlink(fd)
                except FileNotFoundError:continue
                if target.startswith('/dev/nvidia'):
                    fd_observed.add(int(proc.name));break
        except (FileNotFoundError,ProcessLookupError):continue
        except (PermissionError,OSError):fd_known=False
    if owned_pids and not fd_observed.intersection(owned_pids):fd_known=False
    r['gpu_workloads']['known']=r['gpu_workloads']['known'] and fd_known
    for pid in sorted(fd_observed-owned_pids):
        if not any(p['pid']==pid for p in r['gpu_workloads']['unmanaged_processes']):
            r['gpu_workloads']['unmanaged_processes'].append({'pid':pid,'name':'open NVIDIA device handle'})
r['weights']={}
for label,path in q.get('weights',{}).items():
    f=pathlib.Path(path)/'.revision'
    r['weights'][label]=f.read_text().strip() if f.is_file() else None
if q.get('api'):
    r['api']={}
    for route in ('health','v1/models'):
        try:
            with urllib.request.urlopen(q['api']+'/'+route, timeout=10) as response:
                r['api'][route]=json.load(response)
        except Exception as e: r['api'][route]={'probe_error':str(e)}
print(json.dumps(r,sort_keys=True))
'''


class _RemoteController:
    """Runs on the head host; all externally invoked programs use argv lists."""
    def __init__(self, config: dict[str, Any], probe_source: str):
        self.cfg = config
        self.probe_source = probe_source
        self.events: list[dict[str, Any]] = []
        self.owner = Path(config["lock_path"] + ".owner.json")
        self.remote_journal = Path(config["lock_path"] + ".journal.jsonl")

    def event(self, event_name: str, **details: Any) -> None:
        entry = {"time": time.time(), "transaction_id": self.cfg["transaction_id"], "event": event_name, **details}
        self.events.append(entry)
        if getattr(self, "mutating", False):
            with self.remote_journal.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())

    def call(self, argv: list[str], *, stdin: str | None = None, timeout: float = 120,
             env: dict[str, str] | None = None, cwd: str | None = None) -> str:
        timeout = self.remaining(timeout)
        self.event("command", argv=argv, cwd=cwd, timeout=timeout)
        p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, encoding="utf-8", env=env, cwd=cwd, start_new_session=True)
        try:
            stdout, stderr = p.communicate(stdin, timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                stdout, stderr = p.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                stdout, stderr = p.communicate(timeout=15)
            self.event("command_timeout", stdout=stdout, stderr=stderr)
            raise DeploymentError(f"command timeout; process group terminated: {argv[0]}")
        self.event("command_result", returncode=p.returncode, stdout=stdout, stderr=stderr)
        if p.returncode:
            raise DeploymentError(f"command failed ({p.returncode}): {argv[0]}")
        return stdout

    def remaining(self, maximum: float) -> float:
        remaining = getattr(self, "deadline", float("inf")) - time.monotonic()
        if remaining <= 0:
            raise DeploymentError("remote transaction stage exhausted its total time budget")
        return min(maximum, remaining)

    def clean_env(self, recipe: dict[str, Any]) -> dict[str, str]:
        env = {k: os.environ[k] for k in ("HOME", "USER", "LOGNAME", "PATH", "LANG", "LC_ALL") if k in os.environ}
        env.update(recipe["env"])
        env.update({"IMAGE": recipe["image"], "WORKER": self.cfg["worker"]["ssh_target"],
                    "SSH_OPTS": shlex.join(["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes"] + self.cfg["worker"]["ssh_options"])})
        return env

    def probe(self, kind: str, role: str, *, include_recipe: bool = True) -> dict[str, Any]:
        r = self.cfg[kind]
        names = [v["env"]["CONTAINER_NAME"] for k, v in self.cfg.items() if k in {"baseline", "candidate"} and v]
        q = runtime_probe_request(r, role, include_recipe=include_recipe, allowed_containers=names)
        argv = ["python3", "-c", self.probe_source]
        if role == "worker":
            n = self.cfg["worker"]
            argv = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", *n["ssh_options"], "--", n["ssh_target"], shlex.join(argv)]
        # Avoid logging the long helper source. Its hash identifies the exact probe.
        p = subprocess.run(argv, input=json.dumps(q), capture_output=True, text=True, encoding="utf-8", timeout=self.remaining(90),
                           env=self.clean_env(r))
        if p.returncode:
            raise DeploymentError(f"{role} probe failed: {p.stderr.strip()}")
        result = json.loads(p.stdout)
        self.event("probe", kind=kind, role=role, result=result)
        return result

    def verify(self, kind: str, *, running: bool | None = None, image_required: bool = True) -> dict[str, Any]:
        r = self.cfg[kind]
        pair = {role: self.probe(kind, role) for role in ("head", "worker")}
        h = pair["head"]
        if h["recipe_commit"] != r["recipe_commit"] or h["files_sha256"] != r["files_sha256"]:
            raise DeploymentError(f"{kind} recipe commit or deployment file inventory changed")
        # Only execute a recipe helper after its complete source inventory has
        # matched; inspect never sources changed/unreviewed local.sh content.
        if h.get("recipe_label") != r["image_labels"]["tf.patches"]:
            raise DeploymentError(f"{kind} recipe label does not match frozen image label")
        for role, p in pair.items():
            if p.get("machine_id") != self.cfg[role]["machine_id"] or p.get("gpu_uuid") != self.cfg[role]["gpu_uuid"] or "GB10" not in p.get("gpu_name", ""):
                raise DeploymentError(f"{role} physical Spark/GPU identity changed")
            if p["weights"] != {"model": r["env"]["MODEL_REVISION"], "drafter": r["env"]["DFLASH2_REVISION"]}:
                raise DeploymentError(f"{kind} {role} weight revision markers do not match")
            im = p["image"]
            if im is None:
                if image_required:
                    raise DeploymentError(f"{kind} {role} image is missing")
            else:
                if role in r["image_ids"] and im["id"] != r["image_ids"][role]:
                    raise DeploymentError(f"{kind} {role} image ID changed")
                if any(im["labels"].get(k) != v for k, v in r["image_labels"].items()):
                    raise DeploymentError(f"{kind} {role} image labels changed")
            co = p["container"]
            if running is not None and bool(co and co["running"]) != running:
                raise DeploymentError(f"{kind} {role} running state does not match")
            if co and co["running"] and running is not None:
                if p.get("runtime_sha256") != r["runtime_sha256"][role]:
                    raise DeploymentError(f"{kind} {role} effective command/environment/mount fingerprint changed")
                if im is None or co["image_id"] != im["id"] or co["labels"].get("tf.recipe") != "glm53-spark":
                    raise DeploymentError(f"{role} serving container does not use the verified image")
                if co["labels"].get("tf.patches") != r["image_labels"]["tf.patches"]:
                    raise DeploymentError(f"{role} serving container patch identity changed")
                expected_rank = "0" if role == "head" else "1"
                if co["env"].get("TF_GLM_RANK") != expected_rank or co["env"].get("TF_GLM_WORLD") != "2":
                    raise DeploymentError(f"{role} TP rank identity changed")
                for key, field in (("DENSE", "TF_GLM_DENSE_BITS"), ("HEAD", "TF_GLM_HEAD_BITS")):
                    expected = {"bf16": "16", "q8": "8", "q6": "6", "q4": "4"}.get(r["env"][key])
                    if expected is None or co["env"].get(field) != expected:
                        raise DeploymentError(f"{role} runtime {field} changed")
                if co["env"].get("TF_KV_PAGES") != r["env"]["KV_PAGES"]:
                    raise DeploymentError(f"{role} KV page policy changed")
                for key, field in NATIVE_ENV_MAP.items():
                    if key in r["env"] and (co["env"].get(field) or "") != r["env"][key]:
                        raise DeploymentError(f"{role} effective {field} differs from explicit configuration")
                for key in TUNING_ENV:
                    if co["env"].get(key) != r["env"].get(key):
                        raise DeploymentError(f"{role} tuning variable {key} differs from explicit configuration")
                if co["cmd"] != expected_command(r["env"]):
                    raise DeploymentError(f"{role} native command cannot be reproduced by the explicit frozen configuration")
                mounts = {m["Destination"]: m for m in co["mounts"]}
                for dest, key in (("/model", "MODEL_DIR" if role == "head" else "WORKER_MODEL_DIR"), ("/dflash2", "DFLASH2_DIR" if role == "head" else "WORKER_DFLASH2_DIR")):
                    if dest not in mounts or mounts[dest]["Source"] != r["env"][key] or mounts[dest]["RW"]:
                        raise DeploymentError(f"{role} {dest} must mount the frozen weights read-only")
        images = [pair[role]["image"] for role in ("head", "worker")]
        if all(images) and images[0]["content_sha256"] != images[1]["content_sha256"]:
            raise DeploymentError("head and worker image content differs")
        if running:
            comms = [pair[role]["container"]["env"].get("NCCL_COMM_ID") for role in ("head", "worker")]
            if not comms[0] or comms[0] != comms[1] or not comms[0].endswith(":" + r["env"]["NCCL_PORT"]):
                raise DeploymentError("the two ranks do not share the frozen NCCL rendezvous port")
            api = pair["head"].get("api", {})
            if api.get("health", {}).get("status") != "ok":
                raise DeploymentError(f"{kind} API health is not ok")
            models = api.get("v1/models", {}).get("data", [])
            if not any(m.get("id") == r["env"]["SERVED_NAME"] for m in models):
                raise DeploymentError(f"{kind} API does not list the frozen served model")
        return pair

    def recipe(self, kind: str, script: str, *args: str, timeout: float = 600) -> None:
        r = self.cfg[kind]
        self.call(["bash", script, *args], cwd=r["recipe_path"], env=self.clean_env(r), timeout=timeout)

    def ensure_stopped(self, kind: str) -> None:
        for role in ("head", "worker"):
            co = self.probe(kind, role, include_recipe=False)["container"]
            if co is not None:
                raise DeploymentError(f"{kind} {role} container still exists; no new workload may start")

    def start(self, kind: str, timeout: float) -> dict[str, Any]:
        # start.sh can build if the tag is missing. Refuse that before calling it.
        self.exclusive(self.verify(kind))
        self.ensure_stopped(kind)
        self.recipe(kind, "start.sh", timeout=timeout)
        result = self.verify(kind, running=True)
        self.event("serving_verified", kind=kind)
        return result

    def stop(self, kind: str, timeout: float) -> None:
        # Verify each extant container before a recipe stop can touch it.
        pair = self.verify(kind)
        r = self.cfg[kind]
        for role, p in pair.items():
            co = p["container"]
            if co and (co["image_id"] != p["image"]["id"] or co["labels"].get("tf.recipe") != "glm53-spark" or co["labels"].get("tf.patches") != r["image_labels"]["tf.patches"]):
                raise DeploymentError(f"refusing to stop unrecognized {role} container")
        self.recipe(kind, "stop.sh", timeout=timeout)
        self.ensure_stopped(kind)

    def owner_record(self) -> dict[str, Any] | None:
        if not self.owner.exists():
            return None
        return json.loads(self.owner.read_text(encoding="utf-8"))

    def reserve(self) -> bool:
        old = self.owner_record()
        digest = hashlib.sha256(json.dumps(self.cfg, sort_keys=True).encode()).hexdigest()
        record = {"transaction_id": self.cfg["transaction_id"], "config_sha256": digest}
        if old is not None and old != record:
            raise DeploymentError("cluster is reserved by another transaction or a different frozen config")
        if old is None:
            with self.owner.open("x", encoding="utf-8") as f:
                json.dump(record, f, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
        self.event("reserved", owner=record)
        return old is None

    def release(self) -> None:
        self.owner.unlink()
        self.event("released")

    def recover(self, timeout: float) -> dict[str, Any]:
        self.deadline = time.monotonic() + timeout
        self.event("recovery_started")
        # Candidate may have failed during its rank-1 launch. Stop both if any exist.
        c = self.cfg.get("candidate")
        if c:
            present = any(self.probe("candidate", role, include_recipe=False)["container"] for role in ("head", "worker"))
            if present:
                self.cleanup_candidate(timeout)
        bpair = self.verify("baseline")
        if all(p["container"] and p["container"]["running"] for p in bpair.values()):
            result = self.verify("baseline", running=True)
        else:
            if any(p["container"] for p in bpair.values()):
                self.stop("baseline", timeout)
            result = self.start("baseline", timeout)
        self.event("recovery_verified")
        return result

    def cleanup_candidate(self, timeout: float) -> None:
        """Rollback must not execute or even read a damaged candidate recipe."""
        r = self.cfg["candidate"]
        for role in ("head", "worker"):
            p = self.probe("candidate", role, include_recipe=False)
            co = p["container"]
            if co is None:
                continue
            if p.get("machine_id") != self.cfg[role]["machine_id"] or p.get("gpu_uuid") != self.cfg[role]["gpu_uuid"]:
                raise DeploymentError("rollback hardware identity changed")
            if (co["image_id"] != r["image_ids"].get(role) or
                    co["labels"].get("tf.recipe") != "glm53-spark" or
                    co["labels"].get("tf.patches") != r["image_labels"]["tf.patches"]):
                raise DeploymentError(f"rollback refuses an unrecognized {role} candidate container")
            name = r["env"]["CONTAINER_NAME"]
            try:
                self.node_command(role, ["docker", "stop", "-t", "45", name], timeout)
            except DeploymentError as e:
                self.event("candidate_stop_failed", role=role, error=str(e))
            after = self.probe("candidate", role, include_recipe=False)["container"]
            if after and after["running"]:
                self.node_command(role, ["docker", "kill", name], timeout)
                after = self.probe("candidate", role, include_recipe=False)["container"]
            if after and after["running"]:
                raise DeploymentError(f"rollback could not stop {role}; refusing to start baseline")
            try:
                self.node_command(role, ["docker", "logs", "--timestamps", name], timeout)
            except DeploymentError as e:
                self.event("candidate_logs_unavailable", role=role, error=str(e))
            self.node_command(role, ["docker", "rm", name], timeout)
        self.ensure_stopped("candidate")

    def node_command(self, role: str, args: list[str], timeout: float) -> str:
        if role == "worker":
            n = self.cfg["worker"]
            args = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", *n["ssh_options"],
                    "--", n["ssh_target"], shlex.join(args)]
        return self.call(args, timeout=timeout)

    def exclusive(self, pair: dict[str, Any]) -> None:
        for role, p in pair.items():
            if p.get("unreproducible_env"):
                raise DeploymentError(f"{role} has unsupported external native environment names: {p['unreproducible_env']}; refusing a transition")
            workloads = p.get("gpu_workloads", {})
            if not workloads.get("known"):
                raise DeploymentError(f"{role} GPU workload observation is unavailable; refusing a transition")
            if workloads.get("other_containers") or workloads.get("unmanaged_processes"):
                raise DeploymentError(f"{role} has another GPU workload; refusing a transition")

    def run_locked(self, action: str) -> dict[str, Any]:
        timeout = self.cfg["timeouts"][action]
        self.deadline = time.monotonic() + timeout
        new_owner = self.reserve()
        if action == "restore":
            result = self.recover(timeout)
            self.release()
            return {"action": action, "state": "baseline", "inspection": result}
        touched = False
        restoration_started = False
        try:
            if self.cfg.get("candidate") is None:
                raise DeploymentError("candidate configuration is required")
            # Every preflight is read-only and must finish before stopping a rank.
            baseline = self.verify("baseline", running=True)
            self.exclusive(baseline)
            connections = baseline["head"].get("api", {}).get("health", {}).get("live", {}).get("connections")
            if type(connections) is not int or connections != 0:
                raise DeploymentError("baseline API must report zero live connections before transition")
            self.verify("candidate", image_required=action != "prepare")
            # Distinct names are required so a candidate never adopts baseline state.
            self.ensure_stopped("candidate")
            touched = True
            self.stop("baseline", timeout)
            if action == "prepare":
                self.exclusive(self.verify("candidate", image_required=False))
                self.recipe("candidate", "scripts/prepare.sh", "--image", timeout=timeout)
                result = self.verify("candidate")
                self.event("candidate_prepared", image_ids={k: v["image"]["id"] for k, v in result.items()})
                restoration_started = True
                self.deadline = time.monotonic() + self.cfg["timeouts"]["restore"]
                self.start("baseline", self.cfg["timeouts"]["restore"])
                self.release()
                return {"action": action, "state": "baseline", "candidate": result}
            result = self.start("candidate", timeout)
            return {"action": action, "state": "candidate", "reservation_retained": True, "inspection": result}
        except BaseException as original:
            self.event("transaction_failed", error=str(original))
            if touched:
                if restoration_started:
                    self.event("recovery_failed", error=str(original))
                    raise DeploymentError(f"baseline restoration failed: {original}; reservation retained") from original
                try:
                    self.recover(self.cfg["timeouts"]["restore"])
                    self.release()
                except BaseException as recovery:
                    self.event("recovery_failed", error=str(recovery), original_error=str(original))
                    raise DeploymentError(f"transaction failed: {original}; restoration failed: {recovery}; reservation retained") from original
            elif new_owner:
                self.release()
            raise

    def run(self, action: str) -> dict[str, Any]:
        if action == "inspect":
            self.mutating = False
            self.deadline = time.monotonic() + self.cfg["timeouts"]["inspect"]
            owner = self.owner_record()
            # Reservation alone is not proof that activation finished. Probe
            # presence, then verify the running recipe; never guess from a tag.
            b = self.probe("baseline", "head")["container"]
            kind = "baseline" if b and b["running"] else "candidate"
            if kind == "candidate" and not self.cfg.get("candidate"):
                raise DeploymentError("baseline is not running and no candidate is configured")
            return {"action": action, "state": kind, "owner": owner, "inspection": self.verify(kind, running=True)}
        self.mutating = True
        os.umask(0o077)
        import fcntl
        lock = Path(self.cfg["lock_path"])
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as e:
                raise DeploymentError("another cluster transition holds the lock") from e
            self.event("transition_started", action=action)
            return self.run_locked(action)


def _remote_script(config: dict[str, Any], action: str) -> str:
    payload = base64.b64encode(json.dumps(config, sort_keys=True).encode()).decode("ascii")
    code = "\n".join([
        "import base64, hashlib, json, os, shlex, signal, subprocess, sys, time",
        "from pathlib import Path", "from typing import Any",
        "class DeploymentError(RuntimeError): pass",
        f"NATIVE_ENV_MAP={NATIVE_ENV_MAP!r}", f"TUNING_ENV={TUNING_ENV!r}", f"RUNTIME_KEYS={RUNTIME_KEYS!r}",
        f"RECONSTRUCTIBLE_KEYS={RECONSTRUCTIBLE_KEYS!r}",
        inspect.getsource(runtime_probe_request),
        inspect.getsource(expected_command),
        inspect.getsource(_RemoteController),
        f"config=json.loads(base64.b64decode({payload!r}))",
        f"controller=_RemoteController(config, {_PROBE!r})",
        "def interrupted(signum, frame): raise DeploymentError('remote transaction interrupted by signal '+str(signum))",
        "signal.signal(signal.SIGTERM, interrupted)", "signal.signal(signal.SIGHUP, interrupted)",
        "try:", f"    result=controller.run({action!r})",
        "    result['events']=controller.events",
        "    print('SPECPILOT_DEPLOYMENT_RESULT='+json.dumps(result, sort_keys=True))",
        "except BaseException as e:",
        "    print('SPECPILOT_DEPLOYMENT_RESULT='+json.dumps({'error':str(e),'events':controller.events},sort_keys=True))",
        "    sys.exit(1)",
    ])
    return "set -euo pipefail\npython3 - <<'SPECPILOT_FROZEN_DRIVER'\n" + code + "\nSPECPILOT_FROZEN_DRIVER\n"


class TensorFoldDeployment:
    """Build/deploy adapters. No remote command runs unless execute=True."""
    def __init__(self, config: DeploymentConfig, runner: Callable[..., Any] | None = None,
                 *, execute: bool = False, journal_path: str | os.PathLike[str] | None = None):
        self.config = config
        self.runner = runner or self._subprocess
        self.execute = execute
        self.journal_path = Path(journal_path) if journal_path is not None else None
        self.journal: list[dict[str, Any]] = []

    @staticmethod
    def _subprocess(argv: list[str], *, input: str, timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.run(argv, input=input, text=True, encoding="utf-8", capture_output=True, timeout=timeout)

    def plan(self, action: str) -> dict[str, Any]:
        if action not in {"inspect", "prepare", "activate", "restore"}:
            raise ValueError(f"unknown deployment action: {action}")
        if action in {"prepare", "activate"} and self.config.data.get("candidate") is None:
            raise ValueError("prepare and activate require a frozen candidate")
        if action == "activate" and set(self.config.data["candidate"]["image_ids"]) != {"head", "worker"}:
            raise ValueError("activate requires both candidate image IDs; freeze the prepare result first")
        cfg = self.config.data
        n = cfg["head"]
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", *n["ssh_options"], "--", n["ssh_target"], "bash -s"]
        script = _remote_script(cfg, action)
        # The outer SSH timeout leaves time for rollback after the action timeout.
        timeout = cfg["timeouts"][action] + (0 if action == "inspect" else cfg["timeouts"]["restore"] + 240)
        return {"action": action, "execution_enabled": self.execute, "mutates_remote": action != "inspect",
                "argv": argv, "stdin": script, "timeout": timeout,
                "script_sha256": hashlib.sha256(script.encode()).hexdigest(),
                "transaction_id": cfg["transaction_id"]}

    def _record(self, entry: dict[str, Any]) -> None:
        self.journal.append(entry)
        if self.journal_path is not None:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            with self.journal_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, sort_keys=True) + "\n")

    def _action(self, action: str) -> dict[str, Any]:
        plan = self.plan(action)
        if not self.execute:
            return plan
        self._record({"time": time.time(), "event": "dispatch", "action": action,
                      "script_sha256": plan["script_sha256"], "transaction_id": plan["transaction_id"]})
        try:
            result = self.runner(plan["argv"], input=plan["stdin"], timeout=plan["timeout"])
        except BaseException as e:
            self._record({"time": time.time(), "event": "transport_failed", "error": str(e),
                          "remote_state": "unknown; inspect before retry; no automatic ownership clearing"})
            raise DeploymentError(f"SSH transport failed; remote state is unknown: {e}") from e
        entries = [line.split("=", 1)[1] for line in result.stdout.splitlines()
                   if line.startswith("SPECPILOT_DEPLOYMENT_RESULT=")]
        if len(entries) != 1:
            self._record({"event": "invalid_remote_result", "returncode": result.returncode,
                          "stdout": result.stdout, "stderr": result.stderr})
            raise DeploymentError("missing or ambiguous remote result; inspect before retry")
        report = json.loads(entries[0])
        self._record({"time": time.time(), "event": "result", "action": action, "report": report})
        if result.returncode or "error" in report:
            raise DeploymentError(report.get("error", f"remote action failed with exit {result.returncode}"))
        return report

    def inspect(self) -> dict[str, Any]:
        return self._action("inspect")

    def prepare(self) -> dict[str, Any]:
        return self._action("prepare")

    def activate(self) -> dict[str, Any]:
        return self._action("activate")

    def restore(self) -> dict[str, Any]:
        return self._action("restore")
