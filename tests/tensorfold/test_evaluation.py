"""Synthetic-only tests of the evaluator, not measured acceleration evidence."""

from __future__ import annotations

import copy
import unittest

from specpilot_tensorfold.evaluation import evaluate, token_ids_sha256, validate_contract


def synthetic_fixture():
    """Explicit test thresholds below are not a proposed user optimization task."""
    contract = {
        "schema_version": 1, "evidence_kind": "synthetic", "quality_mode": "exact",
        "artifacts": {"baseline": "fixture-baseline", "candidate": "fixture-candidate"},
        "scenarios": [
            {"id": scenario, "concurrency": concurrency,
             "clusters": [{"id": f"batch-{i}",
                           "request_ids": [f"r{j}" for j in range(concurrency)],
                           "block_ids": ["repeat-0", "repeat-1"]}
                          for i in range(4)]}
            for scenario, concurrency in (("short", 1), ("batch", 4))
        ],
        "target": {"scenario_id": "short", "metric": "e2e", "min_speedup": 1.01,
                   "min_ci_speedup": 1.0},
        "guards": [{"scenario_id": "batch", "metric": "e2e", "max_latency_ratio": 1.05}],
        "bootstrap": {"resamples": 200, "confidence": 0.95, "seed": 17, "min_clusters": 3},
    }
    records = []
    for si, scenario in enumerate(contract["scenarios"]):
        for ci, cluster in enumerate(scenario["clusters"]):
            for block in cluster["block_ids"]:
                for slot, arm in enumerate(("baseline", "candidate", "candidate", "baseline")):
                    for ri, request_id in enumerate(cluster["request_ids"]):
                        inputs = [100 + si, 1000 + ci, 2000 + ri]
                        # Include synthetic reasoning/final/EOS IDs in the same complete list.
                        outputs = [10, 11, 12, 2]
                        elapsed = 1_000_000 if arm == "baseline" else 970_000
                        records.append({
                            "schema_version": 1, "evidence_kind": "synthetic",
                            "scenario_id": scenario["id"], "cluster_id": cluster["id"],
                            "block_id": block, "slot": slot, "arm": arm, "request_id": request_id,
                            "status": "ok", "artifact_id": contract["artifacts"][arm],
                            "input_token_ids": inputs, "output_token_ids": outputs,
                            "input_token_sha256": token_ids_sha256(inputs),
                            "request_sha256": token_ids_sha256([99] + inputs),
                            "decoding": "greedy", "ttft_kind": "client_first_visible_delta",
                            "output_token_sha256": token_ids_sha256(outputs),
                            "completion_tokens": len(outputs), "e2e_ns": elapsed,
                            "ttft_ns": elapsed // 10, "finish_reason": "stop",
                        })
    return contract, records


class EvaluationTests(unittest.TestCase):
    def test_fixture_passes_but_never_claims_measured_sealed_or_promoted(self):
        contract, records = synthetic_fixture()
        before = copy.deepcopy((contract, records))
        result = evaluate(contract, records)
        self.assertEqual(result["status"], "SYNTHETIC_PASS")
        self.assertEqual(result["evidence_kind"], "synthetic")
        self.assertFalse(result["promotion_allowed"])
        self.assertFalse(result["sealed"])
        self.assertEqual(before, (contract, records))
        self.assertAlmostEqual(result["metrics"]["short"]["e2e"]["speedup"], 1 / .97)
        self.assertEqual(result["metrics"]["batch"]["e2e"]["cluster_count"], 4)
        self.assertEqual(result["metrics"]["batch"]["e2e"]["block_count"], 8)

    def test_synthetic_cannot_be_consumed_as_measured(self):
        contract, records = synthetic_fixture()
        contract["evidence_kind"] = "measured"
        result = evaluate(contract, records)
        self.assertEqual(result["status"], "INVALID_EVIDENCE")
        self.assertIn("synthetic cannot become measured", " ".join(result["errors"]))

    def test_explicit_contract_required(self):
        contract, records = synthetic_fixture()
        for field in ("target", "guards", "bootstrap", "artifacts", "evidence_kind"):
            broken = copy.deepcopy(contract)
            del broken[field]
            with self.subTest(field=field):
                self.assertFalse(validate_contract(broken)["valid"])
                self.assertEqual(evaluate(broken, records)["status"], "INVALID_CONTRACT")

    def test_contract_rejects_invalid_thresholds_and_types(self):
        contract, _ = synthetic_fixture()
        for bad in (None, True, "1.01", float("nan"), float("inf"), .9, 10**400):
            broken = copy.deepcopy(contract)
            broken["target"]["min_speedup"] = bad
            with self.subTest(value=bad):
                self.assertFalse(validate_contract(broken)["valid"])
        self.assertFalse(validate_contract(None)["valid"])
        self.assertEqual(evaluate(None, [])["status"], "INVALID_CONTRACT")

    def test_teacher_forcing_fails_closed(self):
        contract, records = synthetic_fixture()
        contract["quality_mode"] = "teacher_forced"
        result = evaluate(contract, records)
        self.assertEqual(result["status"], "UNSUPPORTED")
        self.assertFalse(result["promotion_allowed"])

    def test_cluster_is_one_wave_not_a_closed_loop_request_stream(self):
        contract, _ = synthetic_fixture()
        cluster = contract["scenarios"][1]["clusters"][0]
        for requests in (["r0"], ["r0", "r1", "r2", "r3", "r4"]):
            cluster["request_ids"] = requests
            with self.subTest(requests=requests):
                result = validate_contract(contract)
                self.assertFalse(result["valid"])
                self.assertIn("single-wave", " ".join(result["errors"]))

    def test_missing_individual_record_or_whole_block_is_rejected(self):
        contract, records = synthetic_fixture()
        cases = [records[:-1], [r for r in records if r["block_id"] != "repeat-1"]]
        for incomplete in cases:
            with self.subTest(count=len(incomplete)):
                self.assertEqual(evaluate(contract, incomplete)["status"], "INVALID_EVIDENCE")

    def test_duplicates_failed_requests_wrong_artifacts_or_order_are_rejected(self):
        contract, records = synthetic_fixture()
        self.assertEqual(evaluate(contract, records + [records[0]])["status"], "INVALID_EVIDENCE")
        for field, value in (("status", "error"), ("arm", "candidate"), ("artifact_id", "unknown")):
            bad = copy.deepcopy(records)
            bad[0][field] = value
            with self.subTest(field=field):
                self.assertEqual(evaluate(contract, bad)["status"], "INVALID_EVIDENCE")

    def test_missing_ids_never_pass_as_none_equals_none(self):
        contract, records = synthetic_fixture()
        for row in records:
            row["output_token_ids"] = None
            row["output_token_sha256"] = None
        self.assertEqual(evaluate(contract, records)["status"], "INVALID_EVIDENCE")

    def test_wrong_hash_and_count_are_rejected(self):
        contract, records = synthetic_fixture()
        for field, value in (("input_token_sha256", "bad"), ("output_token_sha256", "bad"),
                             ("completion_tokens", 3), ("completion_tokens", True)):
            bad = copy.deepcopy(records)
            bad[0][field] = value
            with self.subTest(field=field):
                self.assertEqual(evaluate(contract, bad)["status"], "INVALID_EVIDENCE")

    def test_input_identity_includes_tokens_not_only_request_ids(self):
        contract, records = synthetic_fixture()
        records[1]["input_token_ids"] = [77, 88]
        records[1]["input_token_sha256"] = token_ids_sha256([77, 88])
        self.assertEqual(evaluate(contract, records)["status"], "INVALID_EVIDENCE")

    def test_candidate_token_change_fails_exact_quality(self):
        contract, records = synthetic_fixture()
        record = next(r for r in records if r["arm"] == "candidate")
        record["output_token_ids"] = [10, 11, 13, 2]
        record["output_token_sha256"] = token_ids_sha256(record["output_token_ids"])
        result = evaluate(contract, records)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["exact_tokens_passed"])

    def test_baseline_nondeterminism_is_inconclusive(self):
        contract, records = synthetic_fixture()
        records[3]["output_token_ids"] = [10, 11, 13, 2]
        records[3]["output_token_sha256"] = token_ids_sha256(records[3]["output_token_ids"])
        self.assertEqual(evaluate(contract, records)["status"], "INCONCLUSIVE")

    def test_guard_cannot_be_compensated_by_target_speedup(self):
        contract, records = synthetic_fixture()
        for row in records:
            if row["arm"] == "candidate":
                row["e2e_ns"] = 500_000 if row["scenario_id"] == "short" else 1_200_000
        result = evaluate(contract, records)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(result["checks"][0]["passed"])
        self.assertFalse(result["checks"][1]["passed"])
        self.assertAlmostEqual(result["metrics"]["batch"]["e2e"]["latency_ratio"], 1.2)

    def test_tpot_is_recomputed_and_missing_ttft_fails_only_if_needed(self):
        contract, records = synthetic_fixture()
        contract["target"]["metric"] = "tpot"
        result = evaluate(contract, records)
        self.assertEqual(result["status"], "SYNTHETIC_PASS")
        for row in records:
            row["ttft_ns"] = None
        self.assertEqual(evaluate(contract, records)["status"], "INVALID_EVIDENCE")
        contract["target"]["metric"] = "e2e"
        self.assertEqual(evaluate(contract, records)["status"], "SYNTHETIC_PASS")

    def test_tpot_rejects_single_token_outputs(self):
        contract, records = synthetic_fixture()
        contract["target"]["metric"] = "tpot"
        for row in records:
            row["output_token_ids"] = [2]
            row["output_token_sha256"] = token_ids_sha256([2])
            row["completion_tokens"] = 1
        self.assertEqual(evaluate(contract, records)["status"], "INVALID_EVIDENCE")

    def test_repeats_do_not_inflate_independent_cluster_count(self):
        contract, records = synthetic_fixture()
        contract["bootstrap"]["min_clusters"] = 5
        result = evaluate(contract, records)
        self.assertEqual(result["status"], "INCONCLUSIVE")
        self.assertIn("4 clusters < 5", " ".join(result["errors"]))

    def test_renaming_duplicate_workloads_does_not_create_independence(self):
        contract, records = synthetic_fixture()
        for row in records:
            if row["scenario_id"] == "short" and row["cluster_id"] == "batch-1":
                row["input_token_ids"] = [100, 1000, 2000]
                row["input_token_sha256"] = token_ids_sha256(row["input_token_ids"])
                row["request_sha256"] = token_ids_sha256([99] + row["input_token_ids"])
        result = evaluate(contract, records)
        self.assertEqual(result["status"], "INVALID_EVIDENCE")
        self.assertIn("non-independent", " ".join(result["errors"]))

    def test_abba_cancels_symmetric_time_drift_and_bootstrap_is_deterministic(self):
        contract, records = synthetic_fixture()
        multipliers = (1, 2, 4, 8)
        for row in records:
            row["e2e_ns"] *= multipliers[row["slot"]]
            row["ttft_ns"] *= multipliers[row["slot"]]
        first = evaluate(contract, records)
        second = evaluate(contract, list(reversed(records)))
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "SYNTHETIC_PASS")
        self.assertAlmostEqual(first["metrics"]["short"]["e2e"]["speedup"], 1 / .97)

    def test_shared_request_between_distinct_batches_is_not_independent(self):
        contract, records = synthetic_fixture()
        for row in records:
            if row["scenario_id"] == "batch" and row["cluster_id"] == "batch-1" and row["request_id"] == "r0":
                row["input_token_ids"] = [101, 1000, 2000]
                row["input_token_sha256"] = token_ids_sha256(row["input_token_ids"])
                row["request_sha256"] = token_ids_sha256([99] + row["input_token_ids"])
        result = evaluate(contract, records)
        self.assertEqual(result["status"], "INVALID_EVIDENCE")
        self.assertIn("overlaps workload cluster", " ".join(result["errors"]))

    def test_token_digest_rejects_non_engine_integer_ids(self):
        for bad in (None, "1,2", [True], [-1], [1.5], [2**32]):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    token_ids_sha256(bad)
        self.assertIsInstance(token_ids_sha256([0, 2**32 - 1]), str)

    def test_out_of_range_engine_token_record_is_rejected(self):
        contract, records = synthetic_fixture()
        records[0]["output_token_ids"] = [10, 11, 12, 2**32]
        result = evaluate(contract, records)
        self.assertEqual(result["status"], "INVALID_EVIDENCE")
        self.assertIn("uint32", " ".join(result["errors"]))

    def test_request_hash_sampling_or_timing_semantics_cannot_change(self):
        contract, records = synthetic_fixture()
        for field, value in (("request_sha256", "0" * 64), ("decoding", "sampling"),
                             ("ttft_kind", "server_first_token")):
            bad = copy.deepcopy(records)
            bad[1][field] = value
            with self.subTest(field=field):
                self.assertEqual(evaluate(contract, bad)["status"], "INVALID_EVIDENCE")


if __name__ == "__main__":
    unittest.main()
