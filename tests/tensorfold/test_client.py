"""Native protocol tests: fabricated HTTP data, no model or GPU execution.

The wire layout follows TensorFold f350cd3e server/openai.zig and tokens.zig.
Fixture token IDs are deliberately arbitrary; output text cannot recover them.
"""
from __future__ import annotations

import copy
import io
import json
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from specpilot_tensorfold.client import (
    ProtocolError, TensorFoldClient, measure_batch, native_token_sha256,
)


INPUT_IDS = [11, 12]
# Thinking marker, reasoning, closing marker, visible answer, EOS.
OUTPUT_IDS = [901, 13, 902, 14, 2]


def body(text="hello"):
    return {"messages": [{"role": "user", "content": text}], "temperature": 0,
            "max_tokens": 16, "chat_template_kwargs": {"enable_thinking": True}}


def metadata(request_id="r0", **updates):
    result = {"scenario_id": "s", "cluster_id": "c", "block_id": "b", "slot": 0,
              "arm": "baseline", "request_id": request_id, "artifact_id": "image-a"}
    result.update(updates)
    return result


def tokenization(ids=None, **updates):
    ids = INPUT_IDS if ids is None else ids
    result = {"count": len(ids), "tokens": ids, "max_model_len": 1048576}
    result.update(updates)
    return result


def terminal():
    return {
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "tensorfold": {"token_ids": OUTPUT_IDS, "token_sha256": native_token_sha256(OUTPUT_IDS),
                       "token_sha": native_token_sha256(OUTPUT_IDS)[:12],
                       "time_to_first_token": 66.0, "seconds": 99.0,
                       "prefill_seconds": 77.0, "tokens_per_second": 4 / 99.0},
        "speculative": {"rounds": 2, "drafted": 6, "accepted": 3, "tokens_per_round": 2.5},
    }


def usage():
    return {"prompt_tokens": len(INPUT_IDS), "completion_tokens": len(OUTPUT_IDS),
            "total_tokens": len(INPUT_IDS) + len(OUTPUT_IDS),
            "completion_tokens_details": {"reasoning_tokens": 2},
            "prompt_tokens_details": {"cached_tokens": 0}}


def native_events(separate=True):
    last = terminal()
    result = [
        {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
        ": keepalive",
        {"choices": [{"index": 0, "delta": {"reasoning_content": "想一想"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "答案"}, "finish_reason": None}]},
        last,
    ]
    if separate:
        result.append({"choices": [], "usage": usage()})
    else:
        last["usage"] = usage()
    return result


def sse(events=None, done=True):
    events = native_events() if events is None else events
    frames = []
    for event in events:
        if isinstance(event, str):
            frames.append(event.encode() + b"\n\n")
        else:
            frames.append(b"data: " + json.dumps(event, ensure_ascii=False).encode() + b"\n\n")
    if done:
        frames.append(b"data: [DONE]\n\n")
    return b"".join(frames)


class ManualClock:
    def __init__(self):
        self.ns = 10_000_000

    def __call__(self):
        return self.ns


class Response(io.BytesIO):
    def __init__(self, raw, content_type="application/json", clock=None):
        super().__init__(raw)
        self.headers = {"Content-Type": content_type}
        self.clock = clock

    def readline(self, *args):
        result = super().readline(*args)
        if result and self.clock is not None:
            self.clock.ns += 1_000_000
        return result


class NativeOpener:
    def __init__(self, stream=None, tokens=None, clock=None):
        self.stream = sse() if stream is None else stream
        self.tokens = tokenization() if tokens is None else tokens
        self.clock = clock
        self.calls = []

    def open(self, request, timeout=None):
        payload = json.loads(request.data) if request.data else None
        self.calls.append((request.full_url, payload, timeout))
        if request.full_url.endswith("/v1/tokenize"):
            if self.clock is not None:
                self.clock.ns += 100_000_000_000  # Must remain outside measurement.
            return Response(json.dumps(self.tokens).encode())
        if request.full_url.endswith("/v1/chat/completions"):
            return Response(self.stream, "text/event-stream; charset=utf-8", self.clock)
        if request.full_url.endswith("/v1/models"):
            return Response(b'{"data":[{"id":"test-model"}]}')
        raise AssertionError(request.full_url)


def client(opener=None, clock=None, **kwargs):
    options = {"opener": opener or NativeOpener()}
    if clock is not None:
        options["clock"] = clock
    options.update(kwargs)
    return TensorFoldClient("http://127.0.0.1:1/v1", "test-model", **options)


class NativeProtocolTests(unittest.TestCase):
    def test_prepare_uses_rendered_chat_ids_and_preserves_template(self):
        opener = NativeOpener()
        original = body()
        before = copy.deepcopy(original)
        prepared, ids = client(opener).prepare(original)
        self.assertEqual(original, before)
        self.assertEqual(ids, INPUT_IDS)
        self.assertTrue(prepared["stream"])
        self.assertTrue(prepared["return_token_ids"])
        self.assertTrue(prepared["stream_options"]["include_usage"])
        self.assertEqual(opener.calls[0][0], "http://127.0.0.1:1/v1/tokenize")
        self.assertEqual(opener.calls[0][1]["chat_template_kwargs"], before["chat_template_kwargs"])

    def test_native_finish_metadata_and_separate_usage_are_combined(self):
        record = client().measure(body(), metadata())
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["output_token_ids"], OUTPUT_IDS)
        self.assertEqual(record["completion_tokens"], 5)  # Includes reasoning + markers + EOS.
        self.assertEqual(record["content"], "答案")
        self.assertEqual(record["reasoning_content"], "想一想")
        self.assertEqual(record["usage"]["completion_tokens_details"]["reasoning_tokens"], 2)
        self.assertEqual(record["engine_token_sha256"], native_token_sha256(OUTPUT_IDS))
        self.assertEqual(record["speculative"]["accepted"], 3)
        self.assertEqual(record["native_timing"]["seconds"], 99.0)

    def test_usage_also_accepted_on_finish_frame(self):
        record = client(NativeOpener(stream=sse(native_events(separate=False)))).measure(body(), metadata())
        self.assertEqual(record["completion_tokens"], len(OUTPUT_IDS))

    def test_client_clock_excludes_tokenization_role_and_keepalive(self):
        clock = ManualClock()
        record = client(NativeOpener(clock=clock), clock).measure(body(), metadata())
        # Each SSE frame takes two lines, with one millisecond per line.
        self.assertEqual(record["ttft_ns"], 6_000_000)
        self.assertEqual(record["e2e_ns"], 14_000_000)
        self.assertEqual(record["ttft_kind"], "client_first_visible_delta")
        self.assertEqual(record["native_timing"]["time_to_first_token"], 66.0)
        self.assertNotIn("tpot_ns", record)  # No subtraction of native prefill from native seconds.

    def test_token_sha_matches_native_decimal_csv_known_vector(self):
        self.assertEqual(native_token_sha256([1, 23, 456, 0xFFFFFFFF]),
                         "59ce3a8d28d46b5fe396be42f190dd3dd26a64b88007ad56d7cd3609fd42bf8c")
        self.assertNotEqual(native_token_sha256([1, 23]), native_token_sha256([12, 3]))

    def test_empty_tool_headers_do_not_count_as_visible_but_function_payload_does(self):
        for function in ({"name": "lookup"}, {"arguments": "{}"}):
            with self.subTest(function=function):
                events = native_events()
                events[1] = {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "id": "call-id", "type": "function",
                     "function": {"name": "", "arguments": ""}},
                ]}}]}
                events[2] = {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "function": function},
                ]}}]}
                clock = ManualClock()
                record = client(NativeOpener(stream=sse(events), clock=clock), clock).measure(body(), metadata())
                self.assertEqual(record["ttft_ns"], 6_000_000)
                self.assertEqual(len(record["tool_call_deltas"]), 2)

    def test_e2e_stops_at_done_before_connection_cleanup(self):
        clock = ManualClock()

        class SlowCloseResponse(Response):
            def close(self):
                if not self.closed:
                    self.clock.ns += 100_000_000_000
                super().close()

        class SlowCloseOpener(NativeOpener):
            def open(self, request, timeout=None):
                if request.full_url.endswith("/v1/chat/completions"):
                    return SlowCloseResponse(self.stream, "text/event-stream", self.clock)
                return super().open(request, timeout)

        record = client(SlowCloseOpener(clock=clock), clock).measure(body(), metadata())
        self.assertEqual(record["e2e_ns"], 14_000_000)

    def test_missing_ids_or_wrong_hash_are_rejected(self):
        for defect in ("missing_ids", "empty_ids", "missing_sha", "wrong_sha", "bool_id"):
            with self.subTest(defect=defect):
                events = native_events()
                tf = events[4]["tensorfold"]
                if defect == "missing_ids":
                    del tf["token_ids"]
                elif defect == "empty_ids":
                    tf["token_ids"] = []
                elif defect == "missing_sha":
                    del tf["token_sha256"]
                elif defect == "wrong_sha":
                    tf["token_sha256"] = "0" * 64
                else:
                    tf["token_ids"] = [True]
                with self.assertRaises(ProtocolError):
                    client(NativeOpener(stream=sse(events))).measure(body(), metadata())

    def test_usage_counts_must_agree_with_full_ids(self):
        for field, wrong in (("completion_tokens", 4), ("completion_tokens", True),
                             ("prompt_tokens", 1), ("prompt_tokens", True)):
            with self.subTest(field=field, wrong=wrong):
                events = native_events()
                events[5]["usage"][field] = wrong
                with self.assertRaises(ProtocolError):
                    client(NativeOpener(stream=sse(events))).measure(body(), metadata())

    def test_tokenize_counts_are_strict_integers(self):
        for tokens in (tokenization(count=999), tokenization([11], count=True),
                       tokenization([False]), tokenization([])):
            with self.subTest(tokens=tokens), self.assertRaises(ProtocolError):
                client(NativeOpener(tokens=tokens)).prepare(body())

    def test_truncated_stream_or_missing_terminal_is_not_success(self):
        variants = (sse(done=False), sse(native_events()[:4]), sse()[:-1])
        for stream in variants:
            with self.subTest(stream=stream[-40:]), self.assertRaises(ProtocolError):
                client(NativeOpener(stream=stream)).measure(body(), metadata())

    def test_error_is_rejected_before_or_after_finish_even_with_http_200(self):
        for position in (0, 4, 6):
            events = native_events()
            events.insert(position, {"error": {"message": "native failure", "type": "server_error"}})
            with self.subTest(position=position), self.assertRaises(ProtocolError):
                client(NativeOpener(stream=sse(events))).measure(body(), metadata())

    def test_malformed_choice_delta_and_multiple_finish_are_rejected(self):
        bad_choices = ([None], [{"delta": []}], [{"delta": {"content": 9}}],
                       [{"delta": {"reasoning_content": True}}], [{"delta": {"tool_calls": {}}}])
        for choices in bad_choices:
            events = native_events()
            events.insert(2, {"choices": choices})
            with self.subTest(choices=choices), self.assertRaises(ProtocolError):
                client(NativeOpener(stream=sse(events))).measure(body(), metadata())
        events = native_events()
        events.append(terminal())
        with self.assertRaisesRegex(ProtocolError, "multiple terminal"):
            client(NativeOpener(stream=sse(events))).measure(body(), metadata())

    def test_only_role_and_eos_has_no_visible_ttft(self):
        events = native_events()
        del events[2:4]
        final = events[2]
        final["tensorfold"]["token_ids"] = [2]
        final["tensorfold"]["token_sha256"] = native_token_sha256([2])
        events[3]["usage"]["completion_tokens"] = 1
        record = client(NativeOpener(stream=sse(events))).measure(body(), metadata())
        self.assertIsNone(record["ttft_ns"])
        self.assertEqual(record["completion_tokens"], 1)

    def test_total_timeout_catches_keepalive_progress(self):
        clock = ManualClock()
        with self.assertRaises(TimeoutError):
            client(NativeOpener(clock=clock), clock, timeout=0.005).measure(body(), metadata())

    def test_prepare_refuses_semantic_and_template_ambiguities(self):
        for update in ({"temperature": 0.5}, {"temperature": False}, {"n": True}, {"n": 2},
                       {"max_tokens": True}, {"model": "wrong"}, {"add_generation_prompt": False},
                       {"add_generation_prompt": 1}, {"stream_options": []}):
            with self.subTest(update=update), self.assertRaises(ProtocolError):
                client().prepare({**body(), **update})

    def test_measure_prepared_cannot_relabel_non_greedy_or_nonstream_request(self):
        c = client()
        prepared, ids = c.prepare(body())
        for update in ({"temperature": 0.1}, {"temperature": False}, {"stream": False},
                       {"return_token_ids": False}, {"model": "wrong"}, {"n": True},
                       {"add_generation_prompt": False}):
            with self.subTest(update=update), self.assertRaises(ProtocolError):
                c.measure_prepared({**prepared, **update}, ids, metadata())

    def test_models_inspection_does_not_claim_generation_or_scorer(self):
        result = client().inspect_api()
        self.assertTrue(result["model_available"])
        self.assertFalse(result["generation_tested"])
        self.assertFalse(result["token_ids_tested"])
        self.assertEqual(result["teacher_forcing_scorer"], "not_implemented")


@contextmanager
def native_http_server(redirect=False):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def write_reply(self, status, raw, content_type="application/json", **headers):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            seen.append((self.path, None, self.headers.get("Authorization")))
            if redirect:
                self.write_reply(302, b"", Location="/capture")
            else:
                self.write_reply(200, b'{"data":[{"id":"test-model"}]}')

        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            payload = json.loads(raw)
            seen.append((self.path, payload, self.headers.get("Authorization")))
            if self.path == "/v1/tokenize":
                self.write_reply(200, json.dumps(tokenization()).encode())
            elif self.path == "/v1/chat/completions":
                self.write_reply(200, sse(), "text/event-stream")
            else:
                self.write_reply(404, b"{}")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", seen
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


class LocalHttpTransportTests(unittest.TestCase):
    def test_real_stdlib_transport_against_fake_native_server(self):
        with native_http_server() as (url, seen):
            c = TensorFoldClient(url, "test-model", api_key="test-only")
            c.inspect_api()
            record = c.measure(body(), metadata())
        self.assertEqual(record["output_token_ids"], OUTPUT_IDS)
        self.assertGreater(record["e2e_ns"], 0)
        self.assertGreaterEqual(record["ttft_ns"], 0)
        self.assertLessEqual(record["ttft_ns"], record["e2e_ns"])
        self.assertEqual([r[0] for r in seen], ["/v1/models", "/v1/tokenize", "/v1/chat/completions"])
        self.assertTrue(all(r[2] == "Bearer test-only" for r in seen))

    def test_redirects_are_refused_before_following_destination(self):
        with native_http_server(redirect=True) as (url, seen):
            with self.assertRaisesRegex(ProtocolError, "redirects are refused"):
                TensorFoldClient(url, "test-model", api_key="test-only").inspect_api()
        self.assertEqual([r[0] for r in seen], ["/v1/models"])


class BatchTests(unittest.TestCase):
    def test_runtime_failure_stays_in_batch_and_invalidates_throughput(self):
        class MixedOpener(NativeOpener):
            def open(self, request, timeout=None):
                payload = json.loads(request.data) if request.data else None
                if (request.full_url.endswith("/v1/chat/completions")
                        and payload["messages"][0]["content"] == "fail"):
                    self.calls.append((request.full_url, payload, timeout))
                    return Response(sse([{"error": {"message": "failure"}}]), "text/event-stream")
                return super().open(request, timeout)

        opener = MixedOpener()
        batch = measure_batch(client(opener), [
            {"body": body("ok"), "metadata": metadata("ok")},
            {"body": body("fail"), "metadata": metadata("fail")},
        ], 2)
        self.assertFalse(batch["complete"])
        self.assertIsNone(batch["batch_wall_tokens_per_second"])
        self.assertEqual([r["status"] for r in batch["records"]], ["ok", "error"])
        self.assertEqual(batch["records"][1]["request_id"], "fail")
        self.assertEqual(batch["records"][1]["error_type"], "ProtocolError")
        self.assertTrue(all(url.endswith("/v1/tokenize") for url, _, _ in opener.calls[:2]))
        self.assertEqual(len(opener.calls), 4)  # No retry or silently dropped request.

    def test_successful_batch_wall_rate_includes_every_output_token(self):
        batch = measure_batch(client(), [
            {"body": body("a"), "metadata": metadata("a")},
            {"body": body("b"), "metadata": metadata("b")},
        ], 2)
        self.assertTrue(batch["complete"])
        self.assertEqual(len(batch["records"]), 2)
        self.assertAlmostEqual(batch["batch_wall_tokens_per_second"], 10 / (batch["batch_wall_ns"] / 1e9))

    def test_mixed_cluster_or_duplicate_request_is_rejected_before_network(self):
        for second in (metadata("r1", cluster_id="different"), metadata("r0")):
            opener = NativeOpener()
            with self.subTest(second=second), self.assertRaises(ValueError):
                measure_batch(client(opener), [{"body": body(), "metadata": metadata()},
                                              {"body": body(), "metadata": second}], 2)
            self.assertEqual(opener.calls, [])


if __name__ == "__main__":
    unittest.main()
