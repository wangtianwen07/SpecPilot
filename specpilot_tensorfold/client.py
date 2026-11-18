"""Standard-library TensorFold Native HTTP adapter (f350cd3e API).

Token evidence is taken from the engine, never reconstructed from output text.
Client timing includes queueing/network; tokenization is outside the timed call.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any

from .evaluation import token_ids_sha256


class ProtocolError(ValueError):
    """Missing, inconsistent or unsupported native protocol evidence."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProtocolError("HTTP redirects are refused; configure the actual endpoint")


def json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def native_token_sha256(ids: list[int]) -> str:
    """TensorFold tokens.zig: decimal uint32 IDs separated by commas."""
    return hashlib.sha256(",".join(str(v) for v in ids).encode("ascii")).hexdigest()


def _ids(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not value or any(
        type(v) is not int or not 0 <= v <= 0xFFFFFFFF for v in value
    ):
        raise ProtocolError(f"{label}: expected a nonempty array of uint32 token IDs")
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ProtocolError(f"{label}: invalid integer")
    return value


def _visible(delta: dict) -> bool:
    if delta.get("content") or delta.get("reasoning_content"):
        return True
    # Tool IDs, type/index and empty function headers are transport metadata.
    for call in delta.get("tool_calls") or []:
        function = call.get("function") if isinstance(call, dict) else None
        if isinstance(function, dict) and any(
            isinstance(function.get(key), str) and function[key] for key in ("name", "arguments")
        ):
            return True
    return False


class TensorFoldClient:
    def __init__(self, base_url: str, model: str, *, api_key: str | None = None,
                 timeout: float = 300, opener=None, clock=time.perf_counter_ns):
        url = urllib.parse.urlsplit(base_url)
        if (url.scheme not in ("http", "https") or not url.hostname or url.username
                or url.password or url.query or url.fragment or url.path.rstrip("/") not in ("", "/v1")):
            raise ValueError("base_url must be an http(s) origin, optionally ending in /v1; no credentials/query")
        if not isinstance(model, str) or not model:
            raise ValueError("model is required")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        self.origin = urllib.parse.urlunsplit((url.scheme, url.netloc, "", "", ""))
        self.model, self.timeout, self.clock = model, timeout, clock
        self.opener = opener or urllib.request.build_opener(_NoRedirect())
        self._headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            self._headers["Authorization"] = "Bearer " + api_key

    def _open(self, path: str, body: dict | None = None):
        payload = None if body is None else json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        req = urllib.request.Request(self.origin + path, data=payload, headers=self._headers,
                                     method="GET" if body is None else "POST")
        return self.opener.open(req, timeout=self.timeout)

    def _json(self, path: str, body: dict | None = None) -> dict:
        with self._open(path, body) as response:
            raw = response.read(64 * 1024 * 1024 + 1)
        if len(raw) > 64 * 1024 * 1024:
            raise ProtocolError("HTTP JSON response exceeded 64 MiB")
        result = json.loads(raw)
        if not isinstance(result, dict) or "error" in result:
            raise ProtocolError("HTTP response is not a successful object")
        return result

    def inspect_api(self) -> dict:
        """Read-only discovery. Does not claim token/scorer capabilities from /models."""
        models = self._json("/v1/models")
        entries = models.get("data")
        if not isinstance(entries, list):
            raise ProtocolError("/v1/models has no data array")
        available = [m.get("id") for m in entries if isinstance(m, dict)]
        if self.model not in available:
            raise ProtocolError("configured model is absent from /v1/models")
        return {"model_available": True, "model": self.model, "models": available,
                "generation_tested": False, "token_ids_tested": False,
                "teacher_forcing_scorer": "not_implemented"}

    def prepare(self, body: dict) -> tuple[dict, list[int]]:
        """Freeze request semantics and obtain the server's rendered prompt IDs."""
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
            raise ProtocolError("a nonempty messages array is required")
        if body.get("model", self.model) != self.model:
            raise ProtocolError("request model does not match configured model")
        if type(body.get("temperature")) not in (int, float) or body["temperature"] != 0:
            raise ProtocolError("this adapter's exact comparison supports explicit temperature=0 only")
        _integer(body.get("max_tokens"), "max_tokens", 1)
        if _integer(body.get("n", 1), "n", 1) != 1:
            raise ProtocolError("only n=1 is supported")
        if body.get("add_generation_prompt", True) is not True:
            raise ProtocolError("chat tokenization requires add_generation_prompt=true")
        if not isinstance(body.get("stream_options", {}), dict):
            raise ProtocolError("stream_options must be an object")
        # The measurement controls transport/evidence, not model semantics.
        body = json.loads(json.dumps(body, allow_nan=False))
        body.update(model=self.model, stream=True, return_token_ids=True)
        body["stream_options"] = {**body.get("stream_options", {}), "include_usage": True}
        tokenized = self._json("/v1/tokenize", body)
        ids = _ids(tokenized.get("tokens"), "tokenize.tokens")
        if _integer(tokenized.get("count"), "tokenize.count", 1) != len(ids):
            raise ProtocolError("tokenize count disagrees with actual IDs")
        return body, ids

    def measure(self, body: dict, metadata: dict) -> dict:
        prepared, input_ids = self.prepare(body)
        return self.measure_prepared(prepared, input_ids, metadata)

    def measure_prepared(self, body: dict, input_ids: list[int], metadata: dict) -> dict:
        """Read all SSE frames, including separate terminal metadata and usage."""
        if (not isinstance(body, dict) or body.get("model") != self.model
                or body.get("stream") is not True or body.get("return_token_ids") is not True
                or type(body.get("temperature")) not in (int, float) or body["temperature"] != 0):
            raise ProtocolError("prepared request must use configured model, greedy streaming and token IDs")
        if (not isinstance(body.get("messages"), list) or not body["messages"]
                or _integer(body.get("n", 1), "n", 1) != 1
                or body.get("add_generation_prompt", True) is not True):
            raise ProtocolError("prepared request must describe one chat with a generation prompt")
        _integer(body.get("max_tokens"), "max_tokens", 1)
        _ids(input_ids, "input_token_ids")
        required = ("scenario_id", "cluster_id", "block_id", "slot", "arm", "request_id", "artifact_id")
        if any(k not in metadata for k in required):
            raise ProtocolError("measurement metadata is incomplete")
        if metadata["arm"] not in ("baseline", "candidate") or type(metadata["slot"]) is not int or metadata["slot"] not in range(4):
            raise ProtocolError("invalid ABBA arm/slot metadata")
        if metadata["arm"] != ("baseline", "candidate", "candidate", "baseline")[metadata["slot"]]:
            raise ProtocolError("arm disagrees with ABBA slot")
        if any(not isinstance(metadata[k], str) or not metadata[k] for k in required if k != "slot"):
            raise ProtocolError("measurement identity fields must be nonempty strings")
        start = self.clock()
        first = None
        tf, usage, speculative = {}, {}, {}
        finish = None
        text_parts, reasoning_parts, tool_parts = [], [], []
        done = False
        end = None
        raw_hash = hashlib.sha256()
        total_bytes = 0
        frames = []
        with self._open("/v1/chat/completions", body) as response:
            content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
            if content_type != "text/event-stream":
                raise ProtocolError("streaming response must have text/event-stream content type")
            while True:
                line = response.readline(8 * 1024 * 1024 + 1)
                if not line:
                    break
                now = self.clock()
                if now - start > self.timeout * 1e9:
                    raise TimeoutError("stream exceeded total request timeout")
                total_bytes += len(line)
                if len(line) > 8 * 1024 * 1024 or total_bytes > 64 * 1024 * 1024:
                    raise ProtocolError("SSE response exceeded size limit")
                raw_hash.update(line)
                line = line.rstrip(b"\r\n")
                if line.startswith(b"data:"):
                    frames.append(line[5:].lstrip(b" "))
                    continue
                if line or not frames:
                    continue
                payload = b"\n".join(frames)
                frames = []
                if payload == b"[DONE]":
                    done = True
                    end = now
                    break
                event = json.loads(payload)
                if not isinstance(event, dict) or "error" in event:
                    raise ProtocolError("SSE contains an error or non-object")
                for key, target in (("tensorfold", tf), ("usage", usage), ("speculative", speculative)):
                    if event.get(key) is not None:
                        if not isinstance(event[key], dict):
                            raise ProtocolError(f"{key} metadata is not an object")
                        target.update(event[key])
                choices = event.get("choices", [])
                if not isinstance(choices, list) or len(choices) > 1:
                    raise ProtocolError("expected at most one choice per event")
                for choice in choices:
                    if not isinstance(choice, dict):
                        raise ProtocolError("choice is not an object")
                    delta = choice.get("delta", {})
                    if not isinstance(delta, dict):
                        raise ProtocolError("delta is not an object")
                    if any(delta.get(key) is not None and not isinstance(delta[key], str)
                           for key in ("content", "reasoning_content")):
                        raise ProtocolError("content and reasoning deltas must be strings or null")
                    if delta.get("tool_calls") is not None and not isinstance(delta["tool_calls"], list):
                        raise ProtocolError("tool_calls delta is not an array")
                    if first is None and _visible(delta):
                        first = now - start
                    if delta.get("content"):
                        text_parts.append(delta["content"])
                    if delta.get("reasoning_content"):
                        reasoning_parts.append(delta["reasoning_content"])
                    if delta.get("tool_calls"):
                        tool_parts.append(delta["tool_calls"])
                    if choice.get("finish_reason") is not None:
                        if finish is not None:
                            raise ProtocolError("multiple terminal choices")
                        finish = choice["finish_reason"]
        if not done or end is None or not isinstance(finish, str) or not finish:
            raise ProtocolError("incomplete SSE: missing finish reason or [DONE]")
        output_ids = _ids(tf.get("token_ids"), "tensorfold.token_ids")
        engine_sha = tf.get("token_sha256")
        if engine_sha != native_token_sha256(output_ids):
            raise ProtocolError("native full token hash is absent or disagrees with token IDs")
        completion = _integer(usage.get("completion_tokens"), "usage.completion_tokens", 1)
        prompt_count = _integer(usage.get("prompt_tokens"), "usage.prompt_tokens", 1)
        if completion != len(output_ids) or prompt_count != len(input_ids):
            raise ProtocolError("usage token counts disagree with actual input/output IDs")
        record = {k: metadata[k] for k in required}
        record.update(schema_version=1, evidence_kind="measured", status="ok", decoding="greedy",
                      input_token_ids=input_ids, output_token_ids=output_ids,
                      input_token_sha256=token_ids_sha256(input_ids),
                      output_token_sha256=token_ids_sha256(output_ids),
                      engine_token_sha256=engine_sha, completion_tokens=completion,
                      e2e_ns=end - start, ttft_ns=first,
                      ttft_kind="client_first_visible_delta", finish_reason=finish,
                      request_sha256=json_sha256(body), raw_record_sha256=raw_hash.hexdigest(),
                      model=self.model, content="".join(text_parts), reasoning_content="".join(reasoning_parts),
                      tool_call_deltas=tool_parts, native_timing=tf,
                      speculative=speculative, usage=usage)
        return record


def measure_batch(client: TensorFoldClient, requests: list[dict], concurrency: int) -> dict:
    """One explicitly supplied cluster/ABBA slot; no automatic workload or search.

    Each request is {body, metadata}. Failures remain records, never filtered out.
    Tokenization finishes before the simultaneous measured wave begins.
    """
    if type(concurrency) is not int or concurrency < 1 or len(requests) != concurrency:
        raise ValueError("one cluster must contain exactly concurrency requests")
    if not requests:
        raise ValueError("empty batch")
    shared = ("scenario_id", "cluster_id", "block_id", "slot", "arm", "artifact_id")
    identities = [tuple(r.get("metadata", {}).get(k) for k in shared) for r in requests]
    if len(set(identities)) != 1 or None in identities[0]:
        raise ValueError("a batch must share one scenario, cluster, block, slot, arm and artifact")
    request_ids = [r.get("metadata", {}).get("request_id") for r in requests]
    if None in request_ids or len(set(request_ids)) != len(request_ids):
        raise ValueError("request IDs must be present and unique")
    prepared = []
    for r in requests:
        body, ids = client.prepare(r["body"])
        prepared.append((body, ids, r["metadata"]))
    barrier = Barrier(concurrency + 1)

    def one(args):
        barrier.wait(timeout=client.timeout)
        try:
            return client.measure_prepared(*args)
        except Exception as exc:
            return {**args[2], "schema_version": 1, "evidence_kind": "measured",
                    "status": "error", "error_type": type(exc).__name__,
                    "error": str(exc)}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = [pool.submit(one, p) for p in prepared]
        start = client.clock()
        barrier.wait(timeout=client.timeout)
        records = [p.result() for p in pending]
        wall_ns = client.clock() - start
    successful = all(r["status"] == "ok" for r in records)
    return {"schema_version": 1, "evidence_kind": "measured", "records": records,
            "complete": successful, "batch_wall_ns": wall_ns,
            "batch_wall_tokens_per_second":
                sum(r["completion_tokens"] for r in records) / (wall_ns / 1e9) if successful else None,
            "protocol": "one_concurrent_wave;tokenize_excluded;timed_until_DONE;no_retries"}
