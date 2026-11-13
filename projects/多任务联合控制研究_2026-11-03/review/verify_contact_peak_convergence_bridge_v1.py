"""Independent archived bridge reader: no simulator/execution-module import."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import argparse,ast,hashlib,json,zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PROTOTYPE=ROOT/'identification/contact_peak_convergence_v1'
FOUNDATION=ROOT.parent/'五指策略接入_2026-11-03'
METHODS=('ratio_effort','fixed_effort','fusion_effort')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def exact(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def path_for(key):return FOUNDATION/key.removeprefix('foundation/') if key.startswith('foundation/') else ROOT/key
def scalar(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,Path):return str(v)
    raise TypeError(type(v).__name__)
def source_fields():
    tree=ast.parse((PROTOTYPE/'source_record.py').read_text(encoding='utf-8'))
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Name) and t.id=='ARRAYS' for t in n.targets))
def zip_cost(path,names):
    with zipfile.ZipFile(path) as z:
        members={i.filename.removesuffix('.npy'):i for i in z.infolist()}
        rows=[members[n] for n in names if n in members]
        return dict(archive_bytes=Path(path).stat().st_size,selected_columns=len(rows),
            selected_uncompressed_bytes=sum(i.file_size for i in rows),selected_compressed_bytes=sum(i.compress_size for i in rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--bridge',type=Path,default=ROOT/'output/contact_peak_convergence_bridge_v1')
    p.add_argument('--out',type=Path,default=ROOT/'review/contact_peak_convergence_bridge_independent_v1.json');args=p.parse_args()
    run=args.bridge.resolve();out=args.out.resolve();marker=run/'execution_complete.json'
    if not marker.is_file():print(json.dumps(dict(status='pending',reason='No terminal marker',new_physics_steps=0)));return 3
    if out.exists():raise FileExistsError(out)
    finish=read(marker);manifest=read(run/'manifest.json');lock=read(run/'source_lock.json')
    checks=[];issues=[];rows=[];all_rng_before=None;all_rng_after=None
    def check(name,ok,detail=None):
        rec=dict(name=name,passed=bool(ok),detail=detail);checks.append(rec)
        if not ok:issues.append(rec)
    check('complete_three_bridge_not_only_peak_flag',finish.get('execution_complete') is True and finish.get('phase')=='bridge' and
        len(finish.get('results',[]))==3 and all(r.get('complete') is True for r in finish['results']))
    check('execution_sources_unchanged',finish.get('sources_unchanged') is True)
    check('all_three_sensor_RNG_final_gates',finish.get('identical_sensor_rng_final_per_method')=={m:True for m in METHODS})
    lock_sha=sha(run/'source_lock.json')
    check('same_live_frozen_source_lock',lock_sha==finish.get('source_lock_sha256')==manifest.get('source_lock_sha256')==sha(PROTOTYPE/'source_lock.json'))
    check('source88_exact_count',len(lock['sources'])==88)
    for key,value in lock['sources'].items():
        live=path_for(key);snapshot=run/'sources'/key
        check('source/'+key,live.is_file() and snapshot.is_file() and sha(live)==sha(snapshot)==value)
    check('source_input_set',set(lock['source_inputs'])==set(METHODS) and manifest['input_sources']==lock['source_inputs'])
    jobs=manifest['jobs'];check('exact_three_assignments',len(jobs)==3 and
        {(j['method'],j['dt']) for j in jobs}=={(m,.0005) for m in METHODS})
    fields=source_fields();mapping={k:('source_environment_json' if k=='environment_json' else k) for k in fields if k!='observation_json'}
    # observation_json contains original policy-only extra fields; the new
    # bridge samples its own common instrument packet instead of copying it.
    for method in METHODS:
        entry=lock['source_inputs'][method];oldpath=Path(entry['path']);expected=ROOT/'output/development_v12_hold_seed617902_v1/runs'/('collision__'+method+'__617902')/'trace.npz'
        check(method+'/old_V12_input_identity_not_native48',oldpath.resolve()==expected.resolve())
        oldmeta=read(oldpath.with_name('metadata.json'))
        check(method+'/input_trace_and_metadata_frozen',sha(oldpath)==entry['trace_sha256']==oldmeta['trace_sha256'] and
            sha(oldpath.with_name('metadata.json'))==entry['metadata_sha256'])
        check(method+'/same_original79_sources',all(lock['sources'].get(k)==v for k,v in oldmeta['research_source_sha256'].items()))
        job=next(j for j in jobs if j['method']==method);folder=Path(job['destination']);newpath=folder/'trace.npz'
        result=read(folder/'result.json');newmeta=read(folder/'metadata.json')
        aggregate=next(r for r in finish['results'] if r['method']==method)
        check(method+'/result_same_as_terminal',result==aggregate)
        check(method+'/new_trace_frozen',sha(newpath)==newmeta['trace_sha256']==result['trace_sha256'])
        check(method+'/same_actual_physical_config_and_model',newmeta['config']==oldmeta['config'] and
            newmeta['model_sha256']==oldmeta['model_sha256'] and newmeta['physics_timestep']==.0005)
        check(method+'/no_new_feedback_or_future_state_restore',newmeta['new_feedback_decisions']==0 and newmeta['actual_q_restored_after_reset'] is False)
        original_names=set(fields)-{'observation_json'}
        new_names=set(mapping.values())|{'rng_state_before_sample','rng_state_after_sample','substep_state_end_time',
            'substep_qpos','substep_qvel','substep_actual_ctrl','substep_warning_count'}
        costs=dict(original=zip_cost(oldpath,original_names),bridge=zip_cost(newpath,new_names))
        with np.load(oldpath,allow_pickle=False) as z:original={n:z[n] for n in original_names}
        with np.load(newpath,allow_pickle=False) as z:actual={n:z[n] for n in new_names}
        differences=[]
        for name,dest in mapping.items():
            ok=exact(original[name],actual[dest]);check(method+'/all_comparable_fields/'+name,ok)
            if not ok:differences.append(name)
        n=len(actual['substep_time']);t=actual['time'];end=actual['substep_state_end_time'];duration=actual['substep_duration']
        check(method+'/3201_boundaries_32000_integration_records',len(t)==3201 and n==32000 and
            exact(t,np.arange(3201,dtype=float)*.005) and np.all(duration==.0005))
        check(method+'/end_clock_alignment',len(end)==n and np.max(abs(end-(actual['substep_time']+duration)))<1e-9 and
            np.all(np.diff(end)>0) and abs(end[-1]-16.)<1e-8)
        check(method+'/micro_state_coverage_and_finite',actual['substep_qpos'].shape==(n,original['qpos'].shape[1]) and
            actual['substep_qvel'].shape==(n,original['qvel'].shape[1]) and actual['substep_actual_ctrl'].shape==(n,22) and
            np.isfinite(actual['substep_qpos']).all() and np.isfinite(actual['substep_qvel']).all() and np.isfinite(actual['substep_actual_ctrl']).all())
        before=actual['rng_state_before_sample'];after=actual['rng_state_after_sample']
        check(method+'/3201_actual_sensor_draws_and_chain',len(before)==len(after)==3201 and np.array_equal(before[1:],after[:-1]) and str(after[-1])==result['noise_rng_final'])
        if all_rng_before is None:all_rng_before=before.copy();all_rng_after=after.copy()
        else:check(method+'/same_sensor_draw_clock_across_methods',exact(before,all_rng_before) and exact(after,all_rng_after))
        # Additional substep states are new evidence, not an unavailable old
        # microstate equivalence assertion. Actual full22 commands are testable.
        import xml.etree.ElementTree as ET
        xml=ET.parse(run/'sources/foundation/assets/legacy_model.xml').getroot()
        names=[x.attrib['name'] for x in xml.find('actuator')]
        robot_ids=[names.index(name) for name in oldmeta['robot_actuator_telemetry']['axes21']]
        other=sorted(set(range(22))-set(robot_ids));check(method+'/one_nonrobot_obstacle_port',len(other)==1)
        expected_ctrl=original['ctrl'][:-1].copy();environments=[json.loads(str(x)) for x in original['environment_json'][:-1]]
        expected_ctrl[:,other[0]]=[float(np.clip(e['obstacle_x'],-.4,.1)) for e in environments]
        check(method+'/actual_micro_input_exact_with_current_environment',exact(actual['substep_actual_ctrl'],np.repeat(expected_ctrl,10,axis=0)))
        peak_original=original['substep_skin_normal'].max(axis=0);peak_actual=actual['substep_skin_normal'].max(axis=0)
        check(method+'/complete16s_peak_is_consequence_of_full_raw_exact',exact(peak_original,peak_actual))
        rows.append(dict(method=method,all_comparable_fields=len(mapping),differences=differences,
            original_trace_sha256=entry['trace_sha256'],bridge_trace_sha256=result['trace_sha256'],
            finger_peak_N=peak_actual.tolist(),actual_sensor_samples=3201,actual_microstates=n,input_read_cost=costs,
            old_micro_qv_available=False,old_initial_warmstart_available=False))
        print(json.dumps(dict(method=method,records_checked=len(rows),issues=len(issues))),flush=True)
        del original,actual
    check('root_bridge_flag_agrees_with_independent_full_fields',finish.get('bridge_all_exact') is True and not any(r['differences'] for r in rows))
    report=dict(schema='contact_peak_bridge_independent_readonly_acceptance_v1',passed=not issues,
        bridge_gate_passed_for_samelock_refinements=not issues,checks=checks,issues=issues,rows=rows,
        source_lock_sha256=lock_sha,plan_sha256=sha(PROTOTYPE/'plan.json'),checker_sha256=sha(Path(__file__)),
        original_context='old V12 collision/hold/617902; three methods, not V12-native48',
        new_physics_steps=0,simulator_imported=False,
        scope=['Every saved comparable field including phase/rotation/reference/environment aliases is checked, beyond the author mandatory subset.',
            'Additional micro q/v have no old counterpart; coverage/finite checked, not called old-exact.',
            '200Hz boundary forward force and cached integration force are separate records; START/END labels retained.',
            'Successful bridge admits only fixed-input timestep refinement, not a new feedback result or physical safety/convergence certificate.',
            'Raw per-patch gzip packets are retained but not exhaustively decoded by this launch gate; contact attribution is a later read-only analysis.'])
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,default=scalar)
    print(json.dumps(dict(passed=not issues,checks=len(checks),issues=len(issues),output=str(out),sha256=sha(out),new_physics_steps=0)))
    return 0 if not issues else 2


if __name__=='__main__':raise SystemExit(main())
