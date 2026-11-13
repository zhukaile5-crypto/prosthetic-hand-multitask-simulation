"""Small three-arm held press comparison. Only --execute dispatches physics."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import numpy as np

from identification import run_held_wrist_alignment_v1 as prior
from identification import v21_held_press_v1 as task
from review.read_held_press_v1 import evaluate

frozen = prior.frozen
ROOT = frozen.ROOT
REGISTRATION = ROOT / "review/held_press_pilot_registration_v1.json"
SEEDS = prior.SEEDS
WORKERS = 3
TREATMENT = ("identification/v21_held_press_v1.py", "identification/run_held_press_v1.py",
             "review/read_held_press_v1.py")
DT = .005
REQUIRE = prior.require


def sources():
    return {**prior.sources(), **{n:ROOT/n for n in TREATMENT}}


def identity():
    return {n:frozen.sha(p) for n,p in sources().items()}


def registration():
    admission = frozen.check_admission()
    REQUIRE(admission["passed"], "Common platform admission failed")
    return dict(schema="held_press_pilot_registration_v1", scope="one nominal held capsule and one lift goal",
        jobs=[dict(id=f"{arm}__{seed}",arm=arm,seed=seed) for seed in SEEDS for arm in task.ARMS],
        target_N=task.TARGET_N, press_start_s=task.START_S, steady_window_s=[task.HOLD_START_S,task.END_S],
        common_approach=dict(knee_reference_m=task.SLOW_LIFT_REFERENCE_M, fast_m_s=task.FAST_APPROACH_M_S,
                             slow_m_s=task.SLOW_APPROACH_M_S, contact_N=task.CONTACT_N,
                             consecutive_samples=task.CONTACT_SAMPLES, high_load_backoff_N=task.HIGH_LOAD_N),
        common_load_regulation=dict(gain_m_s_per_N=task.GAIN_M_S_PER_N, max_press_m_s=task.PRESS_MAX_M_S,
                                    max_unload_m_s=task.UNLOAD_MAX_M_S, press_axis=18, receiver_delay_s=task.DELAY_S),
        expansion=dict(filter_alpha=task.FILTER_ALPHA, lead_s=task.LEAD_S, slope_span_s=task.RATE_ESTIMATE_SPAN_S,
                       slope_cap_N_s=task.RATE_CAP_N_S),
        comparison=dict(ratio_pi="legacy force-ratio grip and common proportional lift load feedback",
                        current="integrated grip and common proportional lift load feedback",
                        expanded="same integrated grip and delay-aware lift load feedback"),
        primary=["first-contact and task overshoot", "fixed-window steady signed error and RMSE",
                 "fixed-window load RMS about its own mean", "object-relative drift and contact loss", "force guard/excess"],
        environment=frozen.environment(), numerics=frozen.numerical_identity(), sources_sha256=identity(),
        foundation_admission_sha256=admission["manifest_sha256"], confirmation=False)


def software_gate():
    plain, expanded = task.PressLift(.1,False), task.PressLift(.1,True)
    def send(arm, now, value, *, age=.010, valid=True, hand=True):
        arm.observe(now=now, reference=.04, packet=SimpleNamespace(valid=valid,time_s=now,
            acquisition_time_s=now-age,normal_load_N=value), hand_valid=hand, lower=0., upper=.1)
        return arm.records[-1]
    REQUIRE(task.force_rate(1.) < 0. == task.force_rate(2.) < task.force_rate(3.),"Wrong force feedback sign")
    for k in range(8):
        t = task.START_S+k*DT
        a,b = (send(arm,t,.25*k) for arm in (plain,expanded))
        REQUIRE(abs(a["lift_rate_request_m_s"])<=.035 and abs(b["lift_rate_request_m_s"])<=.035,"Wrong lift bounds")
        if k < 3:
            REQUIRE(a["internal_lift_target_m"] == b["internal_lift_target_m"],"Approach must match")
    REQUIRE(a["admitted_contact"] and b["admitted_contact"] and a["feedback_load_N"] != b["feedback_load_N"],
            "Expected shared contact gate and distinct load estimate")
    bad=send(expanded, task.START_S+.050, 0., age=.015)
    REQUIRE(not bad["load_measurement_valid"] and bad["lift_rate_request_m_s"]==0., "Stale load must not advance")
    bad=send(plain, task.START_S+.050, 0., hand=False)
    REQUIRE(bad["lift_rate_request_m_s"]==0.,"Grip gate must block descent")
    REQUIRE(send(expanded,task.START_S+.055,4.)["lift_rate_request_m_s"]>0.,"Load overload must unload")


def trace_proof(path, session):
    t = np.arange(task.CONTROL_TICKS)*DT
    want = t[t>=task.START_S]
    rec = session.user_trajectory.records
    with np.load(path,allow_pickle=False) as d:
        REQUIRE(np.array_equal(d["time"],t) and np.allclose(d["substep_time"],np.arange(24000)*.0005,rtol=0.,atol=1e-8),
                "Incomplete common control or physics clock")
        REQUIRE(np.array_equal([x["time_s"] for x in rec],want), "Incomplete public press evidence")
        REQUIRE(np.array_equal(d["actual_motion_target"][t>=task.START_S,0],[x["internal_lift_target_m"] for x in rec])
                and np.all(d["requested_wrist_reference"]==0.),"Press role changed more than the lift target")
        if session.handoff_passed:
            REQUIRE(session.release_at is None and session.plant.release_started is None
                    and not np.any(d["command"]=="release"),"Press must remain in holding")
        return dict(records=len(rec), load_valid=sum(r["load_measurement_valid"] for r in rec),
                    press_guard_permits=sum(r["hand_contact_and_force_permission"] for r in rec),
                    receiver_contact_admitted=any(r["admitted_contact"] for r in rec),
                    warning_count=int(np.max(d["warning_count"])),
                    max_executed_lift_rate_m_s=float(np.max(abs(np.diff(d["executed_reference21"][:,18])))/DT))


def worker(job, plan, path):
    started = time.perf_counter()
    session = None
    try:
        REQUIRE(identity()==plan["sources_sha256"] and frozen.numerical_identity()==plan["numerics"],"Worker source drift")
        session = task.PressSession(arm=job["arm"],seed=job["seed"])
        REQUIRE(session.tick==session.sensors.samples==0 and session.time==0.,"Expected fresh common preparation")
        for _ in range(task.CONTROL_TICKS):
            session.step_one_sensor_tick()
        REQUIRE(session.tick==task.CONTROL_TICKS and session.time==task.END_S, "Did not reach press boundary")
        session.running = False
        session.press_finished = True
        meta = session.save(path)
        frozen.write(path/"press_control.json",session.user_trajectory.records)
        proof=trace_proof(path/"trace.npz",session)
        result=evaluate(path,target_N=task.TARGET_N,press_start_s=task.START_S,
                        hold_start_s=task.HOLD_START_S,hold_end_s=task.END_S)
        row=dict(status="evaluated",handoff_passed=meta["handoff_passed"],trace_sha256=meta["trace_sha256"],
                 controller_sha256=frozen.sha(path/"press_control.json"),proof=proof,metrics=result,
                 wall_seconds=time.perf_counter()-started)
        frozen.write(path/"result.json",row)
        return row
    except Exception as exc:
        row=dict(status="runtime_or_evaluation_failure",error=f"{type(exc).__name__}: {exc}",
                 traceback=traceback.format_exc(),time_s=None if session is None else session.time)
        if session is not None and session.trace and not path.exists():
            try:
                session.running = False
                session.press_finished = False
                session.save(path)
            except Exception as save_error:
                row["partial_save_error"]=str(save_error)
        path.mkdir(parents=True,exist_ok=True)
        frozen.write(path/"runtime_error.json",row)
        return row


def audits(out,rows):
    by={(r["job"]["seed"],r["job"]["arm"]):r for r in rows}
    evidence=[]
    for seed in SEEDS:
        def load(arm):
            row=by[seed,arm]
            REQUIRE(row["status"]=="evaluated","Missing evaluated press arm")
            path=out/"runs"/row["job"]["id"]
            frozen.require_hash(path/"trace.npz",row["trace_sha256"])
            frozen.require_hash(path/"press_control.json",row["controller_sha256"])
            return path
        pi,cur,new=(load(a) for a in task.ARMS)
        m=[frozen.read(x/"metadata.json") for x in (pi,cur,new)]
        matched=all(m[0][key]==x[key] for x in m[1:] for key in
                    ("seed","config","noise","limits","budget","model_sha256","physics_timestep","initial_position"))
        matched &= all(m[0]["held_press"]["fixture"]==x["held_press"]["fixture"] for x in m[1:])
        control,expanded=(frozen.read(x/"press_control.json") for x in (cur,new))
        first=next((b["time_s"] for a,b in zip(control,expanded,strict=True)
                    if a["internal_lift_target_m"]!=b["internal_lift_target_m"]),None)
        until=first if first is not None else task.END_S-DT
        a,b=prior.equal_prefix(pi,cur,7.5),prior.equal_prefix(cur,new,until)
        first_load = next((x for x in expanded if x["time_s"]==first),None)
        with np.load(cur/"trace.npz",allow_pickle=False) as left, np.load(new/"trace.npz",allow_pickle=False) as right:
            changed=np.flatnonzero(left["executed_reference21"][:,18]!=right["executed_reference21"][:,18])
            first_actual=None if not len(changed) else float(right["time"][changed[0]])
            REQUIRE(first_actual is None or (first is not None and first_actual>=first),"Lift changed before press treatment")
        evidence.append(dict(seed=seed,matched_configuration=matched,common_preparation=a,
                             shared_approach_through_first_treatment=b,first_control_target_difference_s=first,
                             first_executed_lift_difference_s=first_actual,first_treatment_measurement=first_load))
    return evidence


def execute(out,plan):
    out.mkdir(parents=True,exist_ok=False)
    frozen.write(out/"plan.json",plan)
    rows=[dict(job=j,status="queued") for j in plan["jobs"]]
    ledger=out/"assignment_ledger.json"
    frozen.write(ledger,rows)
    check=[]
    error=None
    pins_ok=False
    try:
        for n,p in sources().items():
            destination=out/"sources"/n
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(frozen.require_hash(p,plan["sources_sha256"][n]).read_bytes())
            frozen.require_hash(destination,plan["sources_sha256"][n])
        frozen.write(out/"source_snapshot_manifest.json",plan["sources_sha256"])
        with ProcessPoolExecutor(max_workers=WORKERS,max_tasks_per_child=1) as pool:
            pending={pool.submit(worker,r["job"],plan,out/"runs"/r["job"]["id"]):r for r in rows}
            for future in as_completed(pending):
                row=pending[future]
                try:
                    row.update(future.result())
                except Exception as exc:
                    row.update(status="dispatch_failure",error=f"{type(exc).__name__}: {exc}",
                               traceback=traceback.format_exc())
                frozen.write(ledger,rows)
                print(row["job"]["id"],row["status"],row.get("wall_seconds"),row.get("error",""),flush=True)
        check=audits(out,rows)
        pins_ok=registration()==plan
    except Exception as exc:
        error=dict(error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
    frozen.write(out/"pair_audit.json",check)
    matched=len(check)==len(SEEDS) and all(r["matched_configuration"] and r["common_preparation"]["passed"]
                                           and r["shared_approach_through_first_treatment"]["passed"] for r in check)
    assessed=sum(r["status"]=="evaluated" for r in rows)
    final=dict(execution_complete=error is None and matched and pins_ok and assessed==len(rows),
               assessed=assessed,planned=len(rows),matched_pair_audits_passed=matched,
               sources_and_environment_unchanged=pins_ok,assignment_sha256=frozen.sha(ledger),error=error)
    frozen.write(out/"execution_complete.json",final)
    print(json.dumps(final),flush=True)
    return 0 if final["execution_complete"] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register",action="store_true")
    mode.add_argument("--preflight",action="store_true")
    mode.add_argument("--execute",action="store_true")
    p.add_argument("--out")
    args=p.parse_args()
    if args.execute!=bool(args.out):
        p.error("A fresh --out below output is required only for --execute")
    software_gate()
    plan=registration()
    if args.register:
        with REGISTRATION.open("x",encoding="utf-8") as f:
            json.dump(plan,f,ensure_ascii=False,indent=2,allow_nan=False)
            f.write("\n")
        return 0
    REQUIRE(frozen.read(REGISTRATION)==plan,"Registration differs from live prerequisites")
    if args.preflight:
        print(f"PASS: {len(plan['jobs'])} jobs; {len(plan['sources_sha256'])} sources; 0 physics steps")
        return 0
    return execute(frozen.output_path(args.out),plan)


if __name__=="__main__":
    raise SystemExit(main())

