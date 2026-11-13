"""All15 final bundles: explicit identities and independent raw/causal replay."""
import argparse
import json
from pathlib import Path
import numpy as np
from verify_robot_servo_response_v1 import ROOT, Audit, read, sha, AXES25
from verify_robot_servo_model_v1 import verify_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=ROOT / "output/robot_servo_response_all15_v2")
    parser.add_argument("--base", type=Path, default=ROOT / "calibration/phase3_path_validation_v3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); audit = Audit()
    ledger = read(args.bundle / "execution_complete.json")
    expected = {f"{s}/checkpoint_{i:02d}" for s in ("capsule", "box", "sphere") for i in range(5)}
    audit.check("terminal_complete15_no_failures", ledger["complete"] and not ledger["failures"] and len(ledger["contexts"]) == 15)
    audit.check("all15_exact_context_assignment", {r["context_id"] for r in ledger["contexts"]} == expected)
    definition = read(args.bundle / "analysis_definition.json")
    audit.check("author_final_source_identity", sha(definition["execution_source"]) == definition["execution_source_sha256"])
    binding = read(args.bundle / "source_binding.json")
    audit.check("author_binding_identity", sha(args.bundle / "source_binding.json") == definition["source_binding_sha256"])
    inventory = read(ROOT / "review/robot_servo_all15_inventory_v1.json")
    pinned_historical = {r["name"]: r["sha256"] for r in inventory["historical_source_snapshots"]}
    declared_historical = {r["label"]: r["sha256"] for r in binding["verified_source_files"]}
    audit.check("historical_source_set_same_as_independent65", declared_historical == pinned_historical)
    audit.check("historical_data_identity", sha(args.base / "plan.json") == binding["plan_sha256"] and sha(args.base / "execution_complete.json") == binding["execution_complete_sha256"])
    contexts = []; attempted = unavailable = 0; raw_files_rehashed = 0
    for row in ledger["contexts"]:
        context = row["context_id"]; directory = args.bundle / context
        model = read(directory / "model.json")
        audit.check(context + "/ledger_model_identity", sha(directory / "model.json") == row["model_sha256"])
        manifest = read(directory / "input_manifest.json")
        for name, record in manifest["branches"].items():
            audit.check(context + "/" + name + "/metadata_identity", sha(record["metadata_path"]) == record["metadata_sha256"])
            if record.get("available") is False:
                unavailable += 1
                audit.check(context + "/" + name + "/unavailable_not_fabricated", not (args.base / context / name / "trace.npz").exists())
            else:
                attempted += 1; raw_files_rehashed += 1
                actual = sha(record["trace_path"])
                audit.check(context + "/" + name + "/trace_identity", actual == record["trace_sha256"] == read(record["metadata_path"])["trace_sha256"])
        contexts.append(verify_context(directory, args.base, audit))
    audit.check("382_raw_plus8_unavailable_retained", attempted == 382 and unavailable == 8)
    maxima = {}
    for h in ("0.02", "0.04", "0.06", "0.1"):
        vectors = np.array([c["causal_by_horizon"][h]["max_abs"] for c in contexts])
        rmses = np.array([c["causal_by_horizon"][h]["rmse"] for c in contexts])
        maxima[h] = dict(max_empirical_abs_error25=np.max(vectors, axis=0).tolist(),
            max_context_rmse25=np.max(rmses, axis=0).tolist(),
            context_for_each_max_abs=[contexts[int(i)]["context_id"] for i in np.argmax(vectors, axis=0)],
            task_tolerance_consumption_max=[float(vectors[:, 0].max() / .015), float(vectors[:, 1].max() / .015), float(vectors[:, 24].max() / .01)])
    helper_files = (Path(__file__), Path(__file__).with_name("verify_robot_servo_model_v1.py"), Path(__file__).with_name("verify_robot_servo_response_v1.py"))
    result = dict(schema="independent_robot_servo_all15_models_v1", passed=all(c["passed"] for c in audit.checks), checks=audit.checks,
        bundle=str(args.bundle), bundle_execution_sha256=sha(args.bundle / "execution_complete.json"),
        output_axes25=AXES25, raw_files_rehashed=raw_files_rehashed, contexts=contexts, observed_error_maxima=maxima,
        model_admitted=False, pressure_safety_certified=False, new_physics=False,
        scope="all15 full micro refits and all120 mixed raw/causal arithmetic; nominal seen local paths, no fresh physical conditions or untested sequence certification",
        checker_source_sha256={str(p): sha(p) for p in helper_files}, author_source_sha256=definition["execution_source_sha256"])
    with args.output.open("x", encoding="utf-8") as stream:json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(passed=result["passed"], checks=len(audit.checks), contexts=len(contexts), raw_files_rehashed=raw_files_rehashed,
        failed=[c["name"] for c in audit.checks if not c["passed"]], new_physics=False), ensure_ascii=False))
    if not result["passed"]:raise SystemExit(1)


if __name__ == "__main__":main()
