"""Fixed four-scene macro ablation. New outputs only; no gain search."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
from .runtime import CoreSession, ARMS, ROOT, source_files, identity
from .verify import preflight
from identification.run_v21_motion_reserved_development_v2 import (
    environment, numerical_identity, check_admission, evaluate_trace, evaluate_research, write)
from identification.v21_preview_approach_runtime_v1 import source_files as preserved_sources
from v21_stable_handoff.runtime import ReleaseSession
from review.read_v21_motion_reserved_active_comparison_v2 import _arm, MATCHED_FIELDS

SEED=619741
SCENES=("nominal","friction_drop","fragile_conflict","collision")
DURATION=16.
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def require(test,message):
    if not test:
        raise ValueError(message)

def assess(path):
    result=evaluate_trace(path/"trace.npz")
    result["research_metrics"]=evaluate_research(path/"trace.npz")
    result["motion_mode"]="transport"
    write(path/"result.json",result)
    pins={n:sha(path/n) for n in ("metadata.json","result.json","trace.npz","delivery_intervals.npz")}
    metrics=_arm(path,(read(path/"metadata.json"),result,pins))
    require(not metrics["issues"],str(metrics["issues"]))
    return metrics

def archive_reference(scene, role):
    if role=="refined":
        return ROOT/"output/v21_preview_approach_ablation_v1_20261107/runs"/f"preview__{scene}__transport__{SEED}"
    stages=read(ROOT/"review/v21_stage_attribution_v1.json")
    row=next(x for x in stages["rows"] if x["context"]==[scene,"transport",SEED])
    return (ROOT/row["metrics"]["same_prior_legacy_ratio"]["trace"]).parent

def trace_equal(a,b,until):
    state=("time","qpos","qvel","ctrl_before","normal","skin_normal","position","relative")
    actions=("ctrl","executed_reference21","substep_normal","substep_skin_normal")
    n=round(until/.005)
    with np.load(a/"trace.npz",allow_pickle=False) as x,np.load(b/"trace.npz",allow_pickle=False) as y:
        checks={name:np.array_equal(x[name][:n+1],y[name][:n+1]) for name in state}
        for name in actions:
            count=n*10 if name.startswith("substep") else n
            checks[name]=np.array_equal(x[name][:count],y[name][:count])
        control=np.any(x["ctrl"]!=y["ctrl"],axis=1)
        first=float(x["time"][np.flatnonzero(control)[0]]) if np.any(control) else None
    return dict(passed=all(checks.values()),through_s=until,fields=checks,first_control_difference_s=first)

def verify_actions(session):
    records=session.v16_supervisor.decision_evidence
    expected=np.arange(2400,3201)*.005
    require(len(records)==len(expected) and
            np.allclose([r["time_s"] for r in records],expected,atol=1e-9,rtol=0.),
            "Every release tick including terminal observation must have one decision")
    seen=[]
    fallback=0
    for r in records:
        seen.append(r["time_s"])
        u=np.asarray(r["proposed_reference21"])
        receipt=r["execution_receipt"]
        require(np.array_equal(u,receipt["actual_reference21"]),"Native writer changed selected action")
        require(np.all(u>=np.asarray(r["original_mechanical_rate_lower21"])) and
                np.all(u<=np.asarray(r["original_mechanical_rate_upper21"])),"Original action authority exceeded")
        require(not receipt["execution_changed"] and not receipt["stale_plan_replaced"],"Changed or stale execution")
        require(r["exit_path_memory_active"] is False,"Exit path memory unexpectedly active")
        fallback+=r["fallback"] is not None
    require(len(set(seen))==len(expected),"Duplicate release decisions")
    return dict(decisions=len(records),all_actions_original_bounded=True,actual_equals_selected=True,
                fine_correction_layers_inactive=True,fallback_decisions=int(fallback))

def worker(job,plan,out):
    start=time.perf_counter()
    session=None
    path=out/"runs"/job["id"]
    try:
        require(identity()==plan["active_sources_sha256"],"Core source drift")
        require(numerical_identity()==plan["numerics"],"Numerical environment drift")
        if job["arm"]=="baseline_replay":
            session=ReleaseSession(method="ratio_effort",scenario=job["scene"],seed=SEED,split="robustness",
                motion_mode="transport",task_policy="legacy",receiver_policy="blind")
        else:
            session=CoreSession(arm=job["arm"],scenario=job["scene"],seed=SEED,
                                split="robustness",motion_mode="transport")
        while session.running:
            session.step_one_sensor_tick()
        require(session.completed and session.tick==3200,"Incomplete simulation")
        actions=None if job["arm"]=="baseline_replay" else verify_actions(session)
        session.save(path)
        metrics=assess(path)
        with np.load(path/"trace.npz",allow_pickle=False) as trace:
            require(len(trace["substep_time"])==32000,"Missing 2 kHz physics samples")
            require(int(np.max(trace["warning_count"]))==0,"MuJoCo warning")
        return dict(status="evaluated",metrics=metrics,actions=actions,
                    wall_seconds=time.perf_counter()-start)
    except Exception as exc:
        error=dict(status="runtime_failure",error=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc(),time_s=None if session is None else session.time)
        if session is not None and session.trace and not path.exists():
            try:
                session.running=False
                session.completed=False
                session.save(path)
            except Exception as fail:
                error["partial_save_error"]=str(fail)
        path.mkdir(parents=True,exist_ok=True)
        write(path/"runtime_error.json",error)
        return error

def references():
    refs={}
    for scene in SCENES:
        refs[scene]={}
        for role in ("pi","refined"):
            path=archive_reference(scene,role)
            names=("metadata.json","result.json","trace.npz","delivery_intervals.npz")
            hashes={n:sha(path/n) for n in names}
            result=read(path/"result.json")
            meta=read(path/"metadata.json")
            require(meta["trace_sha256"]==hashes["trace.npz"]==result["trace_sha256"],"Archive hash disagreement")
            metrics=_arm(path,(meta,result,hashes))
            require(not metrics["issues"],str(metrics["issues"]))
            refs[scene][role]=dict(path=str(path),hashes=hashes,metrics=metrics)
    return refs

def build_plan():
    admission=check_admission()
    require(admission["passed"],"Foundation admission failed")
    return dict(schema="midterm_core_macro_ablation_v1",seed=SEED,scenes=SCENES,
        preregistered_before_physics=True,confirmation=False,
        selection="all four existing transport scenes at one fixed previously seen seed; no scenario tuning",
        numerics=numerical_identity(),environment=environment(),preflight=preflight(),
        active_sources_sha256=identity(),
        preserved_sources={str(p):sha(p) for p in preserved_sources(["ratio_effort"]).values()},
        references=references(),
        jobs=[dict(id=f"{arm}__{scene}__{SEED}",arm=arm,scene=scene)
              for scene in SCENES for arm in ARMS],
        acceptance=dict(primary="paired median impact reduction >=20% versus PI, >=3/4 lower",
            complementary="report late force RMS, safety, success, clearance and settling on every run",
            rejection="do not recommend if task successes lower, floor events higher, or guard passes lower than PI",
            causal="anticipation ablation should raise impact; handoff ablation should worsen force quality or clearance",
            statistics="descriptive engineering evidence only; no population significance claim"),
        no_parameter_search=True,physical_rollouts_planned=13)

def pair_audits(out,rows,plan):
    audits=[]
    keyed={(r["job"]["scene"],r["job"]["arm"]):r for r in rows}
    for scene in SCENES:
        core=out/"runs"/f"core__{scene}__{SEED}"
        meta=read(core/"metadata.json")
        for role in ("pi","refined"):
            reference=Path(plan["references"][scene][role]["path"])
            other=read(reference/"metadata.json")
            fields={k:meta[k]==other[k] for k in MATCHED_FIELDS}
            require(all(fields.values()),f"Physical context mismatch {scene}/{role}: {fields}")
            audits.append(dict(scene=scene,kind=f"same_context_{role}",passed=True,fields=fields))
        refined=Path(plan["references"][scene]["refined"]["path"])
        audits.append(dict(scene=scene,kind="identical_holding_before_release",**trace_equal(core,refined,12.)))
        for arm in ("without_anticipation","without_handoff"):
            other=out/"runs"/f"{arm}__{scene}__{SEED}"
            pair=trace_equal(core,other,12.)
            if arm=="without_anticipation":
                touch=min(keyed[scene,"core"]["metrics"]["first_target_top_contact_s"],
                          keyed[scene,arm]["metrics"]["first_target_top_contact_s"])
                pair["treatment_precedes_touch"]=pair["first_control_difference_s"] is not None and pair["first_control_difference_s"]<touch
                pair["passed"] &= pair["treatment_precedes_touch"]
            else:
                touch=keyed[scene,"core"]["metrics"]["first_target_top_contact_s"]
                pair["treatment_after_touch"]=pair["first_control_difference_s"] is not None and pair["first_control_difference_s"]>=touch
                pair["passed"] &= pair["treatment_after_touch"]
            audits.append(dict(scene=scene,kind=arm,**pair))
    return audits

def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--out",required=True)
    args=cli.parse_args()
    out=(ROOT/args.out).resolve()
    require(out.is_relative_to(ROOT/"output") and out!=ROOT/"output","Fresh project output required")
    require(not out.exists(),"Existing results cannot be overwritten")
    plan=build_plan()
    out.mkdir()
    write(out/"plan.json",plan)
    for name,path in source_files().items():
        target=out/"sources"/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes())
        require(sha(target)==plan["active_sources_sha256"][name],"Snapshot changed")
    write(out/"source_snapshot_manifest.json",plan["active_sources_sha256"])
    replay_job=dict(id=f"baseline_replay__nominal__{SEED}",arm="baseline_replay",scene="nominal")
    replay=worker(replay_job,plan,out)
    write(out/"baseline_replay.json",replay)
    require(replay["status"]=="evaluated",str(replay))
    reproduction=trace_equal(out/"runs"/replay_job["id"],Path(plan["references"]["nominal"]["pi"]["path"]),16.)
    write(out/"baseline_reproduction.json",reproduction)
    require(reproduction["passed"],"Archived baseline reproduction failed")
    print("BASELINE_REPRODUCTION_PASSED",flush=True)
    rows=[dict(job=j,status="queued") for j in plan["jobs"]]
    write(out/"assignment_ledger.json",rows)
    with ProcessPoolExecutor(max_workers=3,max_tasks_per_child=1) as pool:
        futures={pool.submit(worker,r["job"],plan,out):r for r in rows}
        for future in as_completed(futures):
            row=futures[future]
            try:
                row.update(future.result())
            except Exception as exc:
                row.update(status="dispatch_failure",error=str(exc))
            write(out/"assignment_ledger.json",rows)
            brief=dict(id=row["job"]["id"],status=row["status"],wall_seconds=row.get("wall_seconds"),error=row.get("error"))
            if row["status"]=="evaluated":
                m=row["metrics"]
                brief.update(success=m["original"]["task_success"],peak_N=m["first_target_top_impact20ms_peak_N"],
                             late_rms_N=m["touch_200_to_500ms_rms"]["value_N"],
                             clearance_s=m["original"]["clearance_latency_s"],settled=m["final_low_motion_suffix_found"])
            print(json.dumps(brief),flush=True)
    measured=all(r["status"]=="evaluated" for r in rows)
    audits=pair_audits(out,rows,plan) if measured else []
    write(out/"pair_audit.json",audits)
    original_ok=all(sha(Path(p))==h for p,h in plan["preserved_sources"].items())
    archives_ok=all(sha(Path(r["path"])/name)==h for refs in plan["references"].values()
                    for r in refs.values() for name,h in r["hashes"].items())
    active_ok=identity()==plan["active_sources_sha256"]
    complete=measured and len(audits)==20 and all(r["passed"] for r in audits) and original_ok and archives_ok and active_ok
    final=dict(execution_complete=complete,assessed=sum(r["status"]=="evaluated" for r in rows),
               planned=len(rows),baseline_reproduced=True,original_sources_unchanged=original_ok,
               original_results_unchanged=archives_ok,new_sources_frozen=active_ok,
               pair_audits_passed=all(r["passed"] for r in audits),wall_scope="simulation only")
    write(out/"execution_complete.json",final)
    print(json.dumps(final),flush=True)
    return 0 if complete else 1
if __name__=="__main__":
    raise SystemExit(main())
