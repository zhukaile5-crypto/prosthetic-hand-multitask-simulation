"""Software-only preflight and adversarial dispatch/prefix checks."""

import argparse
from pathlib import Path
import json
import sys
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from identification import run_v21_motion_reserved_development_v2 as runner

HISTORICAL_PASSIVE = ROOT / "review/v23_original_V21_full16_bridges_v1.json"
HISTORICAL_PASSIVE_SHA = "7d00df8e10dac18e0d5bd3ef8f3e89542fc189c0d075e3ef6bee10cf1df9812a"


class Overlay:
    """Substitute one in-memory column without changing the frozen archive."""

    def __init__(self, base, name, value):
        self.base, self.name, self.value = base, name, value

    def __getitem__(self, name):
        return self.value if name == self.name else self.base[name]


def prefix_checks(active_path, original_path):
    with np.load(active_path / "trace.npz", allow_pickle=False) as a, np.load(original_path / "trace.npz", allow_pickle=False) as b:
        baseline = runner.precommand_prefix(a, b)
        assert baseline["passed"] and baseline["first_changed_reference_s"] is None
        x = a["executed_reference21"].copy()
        x[2300, 0] += .001
        early = runner.precommand_prefix(Overlay(a, "executed_reference21", x), b)
        assert not early["passed"] and early["reason"] == "executed_command_differs_before_release"
        x = a["executed_reference21"].copy()
        x[2685, 0] += .001
        later = runner.precommand_prefix(Overlay(a, "executed_reference21", x), b)
        assert later["passed"] and later["first_changed_reference_s"] == 2685 * .005
        assert later["control_state_samples"] == 2686 and later["control_action_samples"] == 2685
        q = a["qpos"].copy()
        q[2685, 0] += .00001
        changed_state = runner.precommand_prefix(Overlay(Overlay(a, "executed_reference21", x), "qpos", q), b)
        assert not changed_state["passed"] and changed_state["checks"]["qpos"] is False
    return dict(frozen_archive_prefix_fields=len(baseline["checks"]),
                premature_command_difference_rejected=True, first_change_tick_precommit_state_checked=True,
                later_reference_detected_without_equating_future_motion=True)


def dispatch_checks(jobs):
    launched = []
    checkpoints = []
    def launch(job):
        launched.append(job["id"])
        return dict(status="evaluated", task_success=False)
    def checkpoint(rows):
        checkpoints.append(tuple(r["status"] for r in rows))
    def forbid(*args, **kwargs):
        raise AssertionError("No bridge-gated active work should be launched")
    rows, passed = runner.stage_trials(jobs, launch, lambda j:{"passed":False}, forbid, checkpoint=checkpoint)
    assert not passed and launched == [jobs[0]["id"]]
    assert len(rows) == 5 and all(r["status"] == "not_run_passive_gate_failed" for r in rows[1:])
    assert checkpoints[-1] == tuple(r["status"] for r in rows)
    launched.clear()
    rows, passed = runner.stage_trials(jobs, launch, lambda j:{"passed":True},
        lambda j:{"available":j["context"][0] != "collision", "reason":"fixture_unavailable"})
    assert passed and launched == [j["id"] for j in jobs]
    assert rows[-1]["status"] == "evaluated_comparison_unavailable" and rows[-1]["task_success"] is False
    return dict(failed_bridge_blocks_all_active_trials=True, unavailable_comparison_keeps_evaluated_task_label=True,
                ledger_updated_each_stage=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    registered = runner.read(runner.DEFAULT_REGISTRATION)
    attempted_physics = []
    def deny(*args, **kwargs):
        attempted_physics.append(True)
        raise AssertionError("Software test attempted a model or physics operation")
    with patch.object(mujoco, "MjModel", side_effect=deny), patch.object(mujoco, "MjData", side_effect=deny), \
            patch.object(mujoco, "mj_step", side_effect=deny), patch.object(mujoco, "mj_forward", side_effect=deny), \
            patch.object(runner, "worker", side_effect=deny):
        assert runner.build_registration() == registered
        archive = runner.read(runner.require_hash(HISTORICAL_PASSIVE, HISTORICAL_PASSIVE_SHA))
        case, = (r for r in archive["rows"] if tuple(r["context_key"]) == runner.PASSIVE_CONTEXT)
        left, right = Path(case["actual"]), Path(case["original"])
        full = runner.compare_saved_continuations(left, right)
        assert full["passed"]
        prefix = prefix_checks(left, right)
        dispatch = dispatch_checks(registered["jobs"])
    assert not attempted_physics
    report = dict(schema="v21_motion_reserved_development_runner_software_v2", passed=True, physics_steps=0,
                  model_instances=0, launch_attempts=0, registered=len(registered["jobs"]),
                  registration_sha256=runner.sha(runner.DEFAULT_REGISTRATION),
                  runner_sha256=runner.sha(Path(runner.__file__)), check_sha256=runner.sha(Path(__file__)),
                  archived_passive_comparison=dict(passed=True, trace_fields=len(full["trace"]["checks"]),
                      release_records=full["release"]["records"], physical_intervals=full["delivery"]["intervals"]),
                  precommand_prefix=prefix, dispatch=dispatch)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(dict(passed=True, registered=report["registered"], **prefix, **dispatch)))


if __name__ == "__main__":
    main()
