"""Bounded integration checks; no search for controller parameters."""
import json
from pathlib import Path
import time
import numpy as np


def run(out):
    from .runtime import ARMS, make_session
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    results, starts = {}, {}
    for arm in ARMS:
        began = time.perf_counter()
        session = make_session(arm)
        truth = session.plant.truth()
        starts[arm] = {k: np.asarray(truth[k]).copy() for k in ("qpos", "qvel", "normal", "skin_normal", "ctrl")}
        initial_contact = float(truth["all_hand_object"])
        if initial_contact > .02:
            raise ValueError("Initial hand-object contact is not an open, unprepared start")
        for _ in range(121):
            session.step_one_sensor_tick()
        trace = session.trace
        selected = np.asarray([r["requested_reference21"] for r in trace])
        executed = np.asarray([r["executed_reference21"] for r in trace])
        checks = dict(initial_hand_object_contact_below_0_02_N=initial_contact <= .02,
            controller_active_every_tick=all(r["controller_active"] for r in trace),
            finite_physical_state=all(np.isfinite(r["qpos"]).all() and np.isfinite(r["qvel"]).all() for r in trace),
            no_physics_warning=max(r["warning_count"] for r in trace) == 0,
            selected_equals_executed=np.array_equal(selected, executed),
            no_hidden_ctrl_clipping=session.plant.max_hidden_clip <= 1e-10,
            native_actions_changed=np.max(np.abs(executed-session.initial_reference21)) > 1e-6,
            no_lift_before_contact_gate=(session.pickup_task.lift_started is not None or
                                        max(abs(r["lift_ref"]) for r in trace) == 0.))
        session.running = False
        session.completed = False
        session.save(out / arm)
        results[arm] = dict(checks={k: bool(v) for k, v in checks.items()},
            initial_hand_object_contact_N=initial_contact,
            simulated_seconds=session.time, wall_seconds=time.perf_counter()-began,
            final_normal_N=trace[-1]["normal"].tolist(),
            maximum_reference_change=float(np.max(np.abs(executed-session.initial_reference21))))
        print(json.dumps(dict(preflight_arm=arm, **results[arm]), ensure_ascii=False), flush=True)
    same_start = {k: bool(np.array_equal(starts["baseline"][k], starts["macro"][k])) for k in starts["baseline"]}
    result = dict(schema="cold_start_preflight_v1", same_initial_state=same_start, arms=results,
                  passed=all(same_start.values()) and all(all(r["checks"].values()) for r in results.values()),
                  scope="0.605s interface and physical sanity only; not acquisition or whole-task success")
    (out / "preflight.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    from v22_joint_task_fusion.numerics import ensure_numeric_environment
    ensure_numeric_environment()
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    result = run(args.out)
    print(json.dumps(dict(passed=result["passed"])), flush=True)
    if not result["passed"]:
        raise SystemExit(1)
