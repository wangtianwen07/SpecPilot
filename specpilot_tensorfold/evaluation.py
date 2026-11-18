"""Explicit-contract evaluation of TensorFold ABBA measurements (stdlib only).

This module neither schedules experiments nor promotes or seals a candidate.  A
PASS describes the supplied evidence, not proof of execution, isolation, or
universal model equivalence.  Callers must supply independently collected engine
token IDs and artifact identities.  Text re-tokenization is not equivalent.

Record schema (one request in one slot):
    schema_version, evidence_kind, scenario_id, cluster_id, block_id, slot,
    arm, request_id, status, artifact_id, input_token_ids, output_token_ids,
    input_token_sha256, output_token_sha256, completion_tokens,
    e2e_ns, ttft_ns, finish_reason, request_sha256, decoding, ttft_kind

Slots 0,1,2,3 declare baseline,candidate,candidate,baseline.  Each declared
cluster/block must contain every contracted request in every slot.  Repeated
blocks of the same workload cluster are aggregated before cluster bootstrap.
The evaluator checks declared order and completeness; an external runner must
actually execute that order and retain its launch/stop timestamps and logs.

Metric names are e2e, ttft, tpot; TPOT=(e2e_ns-ttft_ns)/(completion_tokens-1).
TTFT is client time to the first nonempty visible content/reasoning/tool delta,
not server time to its first generated token. TPOT is a client-visible average
interval proxy; SSE chunks can contain multiple tokens. Tokenization is outside
the timed request. These metrics must not be mixed with server-token timings.
The token count includes all engine-emitted IDs, including reasoning and EOS,
and therefore may differ from a provider's public usage.completion_tokens.
Raw provider counters and engine-specific token hashes may be retained as
additional record fields, but cannot replace the canonical fields above.

No user objective or performance threshold is defaulted here.  Synthetic
fixtures remain synthetic even when all checks pass.  teacher_forced is a
reserved, deliberately unsupported quality capability until an independent
scorer is implemented and audited.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from typing import Any


SCHEMA_VERSION = 1
METRICS = frozenset(("e2e", "ttft", "tpot"))
ARMS = ("baseline", "candidate", "candidate", "baseline")


def _integer(value: Any, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _number(value: Any, minimum: float | None = None) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value) and (minimum is None or value >= minimum)
    except OverflowError:
        return False


def _name(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def token_ids_sha256(ids: list[int]) -> str:
    """SHA256 of compact UTF-8 JSON IDs, not the engine's comma-decimal digest."""
    if not isinstance(ids, list) or not all(_integer(v) and v <= 2**32 - 1 for v in ids):
        raise ValueError("token IDs must be a list of uint32 integers, excluding bool")
    return _canonical_sha256(ids)


def validate_contract(contract: dict) -> dict:
    """Validate a fully explicit contract; return errors instead of raising.

    Contract fields: schema_version, evidence_kind, quality_mode, artifacts
    {baseline,candidate}, scenarios [{id,concurrency,clusters:[{id,request_ids,
    block_ids}]}], target {scenario_id,metric,min_speedup,min_ci_speedup}, guards
    [{scenario_id,metric,max_latency_ratio}], and bootstrap
    {resamples,confidence,seed,min_clusters}.  guards=[] is an explicit choice
    and does not establish performance protection for unlisted metrics.
    """
    errors: list[str] = []
    if not isinstance(contract, dict):
        return {"valid": False, "errors": ["contract must be an object"]}
    if not _integer(contract.get("schema_version")) or contract["schema_version"] != SCHEMA_VERSION:
        errors.append("schema_version must be 1")
    if contract.get("evidence_kind") not in ("measured", "synthetic"):
        errors.append("evidence_kind must be measured or synthetic")
    if contract.get("quality_mode") not in ("exact", "teacher_forced"):
        errors.append("quality_mode must be exact or teacher_forced")
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, dict) or not all(_name(artifacts.get(a)) for a in set(ARMS)):
        errors.append("artifacts must explicitly identify baseline and candidate")
    scenario_ids: set[str] = set()
    scenarios = contract.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        errors.append("scenarios must be a nonempty list")
        scenarios = []
    for si, scenario in enumerate(scenarios):
        label = f"scenarios[{si}]"
        if not isinstance(scenario, dict):
            errors.append(f"{label} must be an object")
            continue
        sid = scenario.get("id")
        if not _name(sid):
            errors.append(f"{label}.id must be a nonempty string")
        elif sid in scenario_ids:
            errors.append(f"duplicate scenario id: {sid}")
        else:
            scenario_ids.add(sid)
        concurrency = scenario.get("concurrency")
        if not _integer(concurrency, 1):
            errors.append(f"{label}.concurrency must be a positive integer")
        clusters = scenario.get("clusters")
        if not isinstance(clusters, list) or not clusters:
            errors.append(f"{label}.clusters must be a nonempty list")
            continue
        cluster_ids: set[str] = set()
        for ci, cluster in enumerate(clusters):
            clabel = f"{label}.clusters[{ci}]"
            if not isinstance(cluster, dict):
                errors.append(f"{clabel} must be an object")
                continue
            cid = cluster.get("id")
            if not _name(cid):
                errors.append(f"{clabel}.id must be a nonempty string")
            elif cid in cluster_ids:
                errors.append(f"duplicate cluster id in {sid}: {cid}")
            else:
                cluster_ids.add(cid)
            for key in ("request_ids", "block_ids"):
                values = cluster.get(key)
                if not isinstance(values, list) or not values or not all(_name(x) for x in values):
                    errors.append(f"{clabel}.{key} must be a nonempty list of strings")
                elif len(set(values)) != len(values):
                    errors.append(f"{clabel}.{key} contains duplicates")
            requests = cluster.get("request_ids")
            if (_integer(concurrency, 1) and isinstance(requests, list)
                    and len(requests) != concurrency):
                errors.append(f"{clabel} request count must equal concurrency for a single-wave batch")
    target = contract.get("target")
    guards = contract.get("guards")
    checks: list[tuple[str, Any]] = [("target", target)]
    if not isinstance(guards, list):
        errors.append("guards must be an explicit list (possibly empty)")
        guards = []
    checks += [(f"guards[{i}]", guard) for i, guard in enumerate(guards)]
    seen_guards: set[tuple[str, str]] = set()
    for label, check in checks:
        if not isinstance(check, dict):
            errors.append(f"{label} must be an object")
            continue
        if not _name(check.get("scenario_id")) or check["scenario_id"] not in scenario_ids:
            errors.append(f"{label}.scenario_id must identify a contracted scenario")
        if not isinstance(check.get("metric"), str) or check["metric"] not in METRICS:
            errors.append(f"{label}.metric must be e2e, ttft or tpot")
        if label == "target":
            for key in ("min_speedup", "min_ci_speedup"):
                if not _number(check.get(key), 1):
                    errors.append(f"target.{key} must explicitly be a finite ratio >= 1")
        else:
            if not _number(check.get("max_latency_ratio"), 1):
                errors.append(f"{label}.max_latency_ratio must explicitly be a finite ratio >= 1")
            if _name(check.get("scenario_id")) and isinstance(check.get("metric"), str):
                guard_id = (check["scenario_id"], check["metric"])
                if guard_id in seen_guards:
                    errors.append(f"duplicate guard: {guard_id}")
                seen_guards.add(guard_id)
    boot = contract.get("bootstrap")
    if not isinstance(boot, dict):
        errors.append("bootstrap must be explicit")
    else:
        if not _integer(boot.get("resamples"), 100):
            errors.append("bootstrap.resamples must be an integer >= 100")
        confidence = boot.get("confidence")
        if not _number(confidence) or not 0 < confidence < 1:
            errors.append("bootstrap.confidence must be between 0 and 1")
        if not _integer(boot.get("seed")):
            errors.append("bootstrap.seed must be a non-negative integer")
        if not _integer(boot.get("min_clusters"), 2):
            errors.append("bootstrap.min_clusters must be an integer >= 2")
    try:
        digest = _canonical_sha256(contract)
    except (TypeError, ValueError):
        errors.append("contract must be finite JSON data")
        digest = None
    return {"valid": not errors, "errors": errors, "contract_sha256": digest}


def _result(status: str, contract: dict, validation: dict, **kwargs: Any) -> dict:
    return {"schema_version": SCHEMA_VERSION, "status": status,
            "evidence_kind": contract.get("evidence_kind") if isinstance(contract, dict) else None,
            "contract_sha256": validation.get("contract_sha256"),
            "promotion_allowed": False, "sealed": False,
            "quality_mode": contract.get("quality_mode") if isinstance(contract, dict) else None,
            **kwargs}


def _percentile(values: list[float], q: float) -> float:
    pos = q * (len(values) - 1)
    left = math.floor(pos)
    right = math.ceil(pos)
    return values[left] + (values[right] - values[left]) * (pos - left)


def _metric(row: dict, metric: str) -> float | None:
    if metric == "e2e":
        return float(row["e2e_ns"])
    first = row["ttft_ns"]
    if first is None:
        return None
    if metric == "ttft":
        return float(first) if first > 0 else None
    count = row["completion_tokens"]
    if count <= 1 or row["e2e_ns"] <= first:
        return None
    return (row["e2e_ns"] - first) / (count - 1)


def evaluate(contract: dict, records: list[dict]) -> dict:
    """Check complete ABBA evidence and exact tokens, then bootstrap clusters.

    Status is PASS or SYNTHETIC_PASS only if every explicit check passes.
    Missing/invalid data gives INVALID_EVIDENCE; unstable baseline tokens or
    insufficient independent clusters gives INCONCLUSIVE.  Unsupported scorer
    capability gives UNSUPPORTED.  No result permits automatic promotion.
    """
    validation = validate_contract(contract)
    if not validation["valid"]:
        return _result("INVALID_CONTRACT", contract, validation, errors=validation["errors"])
    if contract["quality_mode"] == "teacher_forced":
        return _result("UNSUPPORTED", contract, validation,
                       errors=["independent teacher-forced scorer is not implemented; no quality acceptance is possible"])
    if not isinstance(records, list):
        return _result("INVALID_EVIDENCE", contract, validation, errors=["records must be a list"])

    expected: set[tuple] = set()
    for scenario in contract["scenarios"]:
        for cluster in scenario["clusters"]:
            for block in cluster["block_ids"]:
                for slot in range(4):
                    for rid in cluster["request_ids"]:
                        expected.add((scenario["id"], cluster["id"], block, slot, rid))
    indexed: dict[tuple, dict] = {}
    errors: list[str] = []
    for i, row in enumerate(records):
        prefix = f"record[{i}]"
        if not isinstance(row, dict):
            errors.append(f"{prefix} must be an object")
            continue
        fields = ("scenario_id", "cluster_id", "block_id", "request_id")
        if not all(_name(row.get(key)) for key in fields) or not _integer(row.get("slot")):
            errors.append(f"{prefix} has invalid identity fields")
            continue
        key = (row["scenario_id"], row["cluster_id"], row["block_id"], row["slot"], row["request_id"])
        if key not in expected:
            errors.append(f"{prefix} is not contracted: {key}")
            continue
        if key in indexed:
            errors.append(f"duplicate record: {key}")
            continue
        indexed[key] = row
        arm = ARMS[row["slot"]]
        if not _integer(row.get("schema_version")) or row["schema_version"] != SCHEMA_VERSION:
            errors.append(f"{prefix}.schema_version must be 1")
        if row.get("arm") != arm:
            errors.append(f"{prefix} violates ABBA slot {row['slot']} (expected {arm})")
        if row.get("artifact_id") != contract["artifacts"][arm]:
            errors.append(f"{prefix}.artifact_id differs from the frozen {arm} identity")
        if row.get("evidence_kind") != contract["evidence_kind"]:
            errors.append(f"{prefix}.evidence_kind differs from contract; synthetic cannot become measured")
        if row.get("status") != "ok":
            errors.append(f"{prefix} is not a successful request")
        if not _name(row.get("finish_reason")):
            errors.append(f"{prefix}.finish_reason is missing")
        if not isinstance(row.get("request_sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", row["request_sha256"]) is None:
            errors.append(f"{prefix}.request_sha256 must identify the complete frozen request")
        if row.get("decoding") != "greedy":
            errors.append(f"{prefix}.decoding must be greedy for exact mode")
        if row.get("ttft_kind") != "client_first_visible_delta":
            errors.append(f"{prefix}.ttft_kind must be client_first_visible_delta, not a server-token timing")
        for direction in ("input", "output"):
            ids = row.get(f"{direction}_token_ids")
            if not isinstance(ids, list) or not ids or not all(_integer(t) and t <= 2**32 - 1 for t in ids):
                errors.append(f"{prefix}.{direction}_token_ids must contain real nonempty uint32 IDs, excluding bool")
            elif row.get(f"{direction}_token_sha256") != token_ids_sha256(ids):
                errors.append(f"{prefix}.{direction}_token_sha256 does not match the IDs")
        count = row.get("completion_tokens")
        if not _integer(count, 1) or not isinstance(row.get("output_token_ids"), list) or count != len(row["output_token_ids"]):
            errors.append(f"{prefix}.completion_tokens must equal the complete output ID count")
        elapsed = row.get("e2e_ns")
        first = row.get("ttft_ns")
        if not _integer(elapsed, 1) or elapsed > 2**63 - 1:
            errors.append(f"{prefix}.e2e_ns must be a positive signed-64-bit integer")
        if "ttft_ns" not in row or (first is not None and (not _integer(first) or (_integer(elapsed, 1) and first > elapsed))):
            errors.append(f"{prefix}.ttft_ns must be null or a non-negative integer <= e2e_ns")
    missing = expected - set(indexed)
    if missing:
        errors.append(f"missing {len(missing)} contracted records; first missing: {sorted(missing)[0]}")
    if errors:
        return _result("INVALID_EVIDENCE", contract, validation, errors=errors,
                       expected_records=len(expected), observed_records=len(records))

    baseline_unstable: list[str] = []
    token_mismatches: list[str] = []
    input_mismatches: list[str] = []
    repeated_workloads: list[str] = []
    for scenario in contract["scenarios"]:
        # Identical workload batches under new cluster IDs are not independent
        # prompts.  Reject them instead of inflating the bootstrap sample count.
        fingerprints: dict[tuple, str] = {}
        request_clusters: dict[tuple, str] = {}
        for cluster in scenario["clusters"]:
            sid, cid = scenario["id"], cluster["id"]
            fingerprint = []
            for rid in cluster["request_ids"]:
                rows = [indexed[(sid, cid, block, slot, rid)]
                        for block in cluster["block_ids"] for slot in range(4)]
                reference = rows[0]
                request_fp = (reference["input_token_sha256"], reference["request_sha256"])
                fingerprint.append(request_fp)
                prior_cluster = request_clusters.get(request_fp)
                if prior_cluster is not None and prior_cluster != cid:
                    repeated_workloads.append(f"{sid}/{cid}/{rid} overlaps workload cluster {prior_cluster}")
                request_clusters[request_fp] = cid
                identity = f"{sid}/{cid}/{rid}"
                if any(r["input_token_ids"] != reference["input_token_ids"] for r in rows):
                    input_mismatches.append(identity)
                if any(r["request_sha256"] != reference["request_sha256"] for r in rows):
                    input_mismatches.append(identity + ":request_sha256")
                baseline_rows = [r for r in rows if r["arm"] == "baseline"]
                if any(r["output_token_ids"] != reference["output_token_ids"] for r in baseline_rows):
                    baseline_unstable.append(identity)
                if any(r["output_token_ids"] != reference["output_token_ids"] for r in rows if r["arm"] == "candidate"):
                    token_mismatches.append(identity)
                if any(r["finish_reason"] != reference["finish_reason"] for r in rows):
                    token_mismatches.append(identity + ":finish_reason")
            fp = tuple(sorted(fingerprint))
            if fp in fingerprints:
                repeated_workloads.append(f"{sid}/{cid} duplicates workload cluster {fingerprints[fp]}")
            fingerprints[fp] = cid
    if input_mismatches:
        return _result("INVALID_EVIDENCE", contract, validation,
                       errors=["paired inputs differ: " + x for x in input_mismatches])
    if repeated_workloads:
        return _result("INVALID_EVIDENCE", contract, validation,
                       errors=["non-independent workload clusters: " + x for x in repeated_workloads])
    if baseline_unstable:
        return _result("INCONCLUSIVE", contract, validation,
                       errors=["baseline output is not deterministic: " + x for x in baseline_unstable],
                       exact_tokens_passed=False)
    if token_mismatches:
        return _result("FAIL", contract, validation,
                       errors=["exact output mismatch: " + x for x in token_mismatches],
                       exact_tokens_passed=False)

    requested = defaultdict(set)
    requested[contract["target"]["scenario_id"]].add(contract["target"]["metric"])
    for guard in contract["guards"]:
        requested[guard["scenario_id"]].add(guard["metric"])
    stats: dict[str, dict] = {}
    boot = contract["bootstrap"]
    insufficient: list[str] = []
    unavailable: list[str] = []
    for scenario in contract["scenarios"]:
        sid = scenario["id"]
        stats[sid] = {}
        for metric in sorted(requested[sid]):
            cluster_logs = []
            block_count = 0
            for cluster in scenario["clusters"]:
                samples = []
                for block in cluster["block_ids"]:
                    block_count += 1
                    for rid in cluster["request_ids"]:
                        rows = [indexed[(sid, cluster["id"], block, slot, rid)] for slot in range(4)]
                        values = [_metric(r, metric) for r in rows]
                        if any(v is None or v <= 0 or not math.isfinite(v) for v in values):
                            unavailable.append(f"{sid}/{cluster['id']}/{block}/{rid}: {metric}")
                            continue
                        a0, b1, b2, a3 = values
                        # Averaging logs in both temporal directions cancels a
                        # linear drift in log latency over equally spaced slots.
                        samples.append(((math.log(a0) - math.log(b1))
                                        + (math.log(a3) - math.log(b2))) / 2)
                if samples:
                    cluster_logs.append(sum(samples) / len(samples))
            if len(cluster_logs) < boot["min_clusters"]:
                insufficient.append(f"{sid}/{metric}: {len(cluster_logs)} clusters < {boot['min_clusters']}")
                continue
            rng = random.Random(boot["seed"])
            n = len(cluster_logs)
            draws = sorted(sum(cluster_logs[rng.randrange(n)] for _ in range(n)) / n
                           for _ in range(boot["resamples"]))
            tail = (1 - boot["confidence"]) / 2
            point = math.exp(sum(cluster_logs) / n)
            lo, hi = math.exp(_percentile(draws, tail)), math.exp(_percentile(draws, 1 - tail))
            stats[sid][metric] = {
                "speedup": point, "speedup_ci": [lo, hi],
                "latency_ratio": 1 / point, "latency_ratio_ci": [1 / hi, 1 / lo],
                "confidence": boot["confidence"], "cluster_count": n,
                "block_count": block_count, "cluster_log_speedups": cluster_logs,
                "method": "equal_cluster_geomean_paired_ABBA_percentile_bootstrap",
                "measurement": {"e2e": "client_request_e2e",
                                "ttft": "client_first_visible_delta",
                                "tpot": "client_visible_average_interval"}[metric],
            }
    if unavailable:
        return _result("INVALID_EVIDENCE", contract, validation,
                       errors=["required metric unavailable: " + x for x in unavailable],
                       exact_tokens_passed=True)
    if insufficient:
        return _result("INCONCLUSIVE", contract, validation, errors=insufficient,
                       exact_tokens_passed=True, metrics=stats)

    target = contract["target"]
    result = stats[target["scenario_id"]][target["metric"]]
    target_pass = (result["speedup"] >= target["min_speedup"]
                   and result["speedup_ci"][0] > target["min_ci_speedup"])
    checks = [{"kind": "target", **target, "passed": target_pass,
               "comparison": "speedup >= min_speedup and CI lower > min_ci_speedup"}]
    for guard in contract["guards"]:
        result = stats[guard["scenario_id"]][guard["metric"]]
        checks.append({"kind": "guard", **guard,
                       "passed": result["latency_ratio_ci"][1] <= guard["max_latency_ratio"],
                       "comparison": "latency ratio CI upper <= max_latency_ratio"})
    passed = all(c["passed"] for c in checks)
    status = ("SYNTHETIC_PASS" if contract["evidence_kind"] == "synthetic" else "PASS") if passed else "FAIL"
    return _result(status, contract, validation, errors=[], exact_tokens_passed=True,
                   metrics=stats, checks=checks, expected_records=len(expected),
                   observed_records=len(records),
                   unassessed_scenarios=[s["id"] for s in contract["scenarios"] if not requested[s["id"]]],
                   limitations=["Only supplied evidence is evaluated; physical ABBA execution and artifact identity require independent logs.",
                                "Exact token equality on these requests is not a proof for unseen inputs.",
                                "This result is not a sealed final and does not authorize promotion."])
