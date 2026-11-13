"""Predeclare an instrument bridge and three contact-task development trials."""
from pathlib import Path
import argparse,json,os,subprocess,sys
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from v17_contact_transfer.runtime import source_identity
from v17_contact_transfer.motion_domain_sensors import ADDED_FIELDS
from v13_a1.run_bridge import environment
from review.run_v16_release_small_v1 import previous_A,sha,read,write,PHYSICAL_COLUMNS


def compare_original_fields(new,old,prefix):
    """New instrument validity metadata is separate; every old value remains."""
    checks={};columns=tuple(PHYSICAL_COLUMNS)+('raw_patches_json',)
    with np.load(new/'trace.npz',allow_pickle=False) as a,np.load(old/'trace.npz',allow_pickle=False) as b:
        ma=a['time']<12.-1e-9 if prefix else np.ones(len(a['time']),bool)
        mb=b['time']<12.-1e-9 if prefix else np.ones(len(b['time']),bool)
        for name in columns:
            if name not in a or name not in b:
                checks[name]=dict(equal=False,reason='required_original_column_missing');continue
            aa=a[name][ma];bb=b[name][mb];same=aa.shape==bb.shape
            if name=='legacy_common_sensor_json' and same:
                good=True;extra_keys=set()
                for xx,yy in zip(aa.flat,bb.flat):
                    x=json.loads(str(xx));y=json.loads(str(yy));extra_keys.update(set(x)-set(y))
                    if set(x)-set(y)-set(ADDED_FIELDS) or set(y)-set(x):good=False;break
                    for key in ADDED_FIELDS:x.pop(key,None)
                    if json.dumps(x,sort_keys=True,allow_nan=True)!=json.dumps(y,sort_keys=True,allow_nan=True):good=False;break
                checks[name]=dict(equal=good,new_metadata_keys=sorted(extra_keys),rule='all_original_JSON_fields_exact_after_removing_only_declared_domain_metadata')
            elif aa.dtype.kind=='U' and bb.dtype.kind=='U':
                checks[name]=dict(equal=bool(same and all(str(x).encode('utf-8')==str(y).encode('utf-8') for x,y in zip(aa.flat,bb.flat))),rule='exact_UTF8_payload_not_full_episode_padding')
            else:
                checks[name]=dict(equal=bool(same and aa.dtype==bb.dtype and aa.tobytes()==bb.tobytes()),rule='exact_dtype_shape_bytes')
    return dict(passed=all(v['equal'] for v in checks.values()),checks=checks,columns=len(columns),
        scope='strictly_before12' if prefix else 'complete16s',
        new=str(new),old=str(old),new_trace_sha256=sha(new/'trace.npz'),old_trace_sha256=sha(old/'trace.npz'),
        additional_domain_is_not_an_extra_noise_or_identity_sensor=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-gate',required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:raise ValueError('Explicit execution switch required')
    pins=source_identity(['ratio_effort'])
    parent=read(ROOT/'review/v16b_release_small_plan_v1.json')
    if len(pins)!=133 or any(pins.get(k)!=v for k,v in parent['source_sha256'].items()):
        raise RuntimeError('Expected133-source closure or frozen parent126 changed')
    if environment()!=parent['environment']:raise RuntimeError('Original42 binary environment changed')
    gate_names=['review/v17_hierarchy_author_v1.json','review/v17_hierarchy_independent_v1.json',
        'review/v17_motion_independent_v1.json','review/v17_supervisor_independent_v2.json',args.runtime_gate]
    gates={}
    for n in gate_names:
        j=read(ROOT/n)
        if j.get('passed') is not True:raise RuntimeError('Gate not passed: '+n)
        gates[n]=dict(passed=True,sha256=sha(ROOT/n))
    numerical=read(ROOT/'review/v17_hierarchy_independent_v1.json')
    for key in ('v17_contact_transfer/hierarchy.py','v17_contact_transfer/task_solver.py'):
        if pins[key] not in json.dumps(numerical):raise RuntimeError('Numerical gate source pin stale: '+key)
    runtime=read(ROOT/args.runtime_gate)
    for key,value in pins.items():
        if value not in json.dumps(runtime):raise RuntimeError('Runtime gate does not bind current source: '+key)
    stages=[dict(id='blind_bridge',scene=['nominal'],motion='transport',policy='blind',planned=1),
        dict(id='contact_hold',scene=['fragile_conflict'],motion='hold',policy='coordinated',planned=1),
        dict(id='contact_transport',scene=['nominal','friction_drop'],motion='transport',policy='coordinated',planned=2)]
    master=ROOT/'review/v17_transfer_small_plan_v1.json'
    if master.exists():raise FileExistsError('Never overwrite a registered attempt')
    for s in stages:
        s['output']='output/v17_transfer_'+s['id']+'_v1';s['registration']='review/v17_transfer_'+s['id']+'_registration_v1.json'
        if (ROOT/s['output']).exists() or (ROOT/s['registration']).exists():raise FileExistsError('Assigned output already exists')
    plan=dict(schema='v17_contact_task_seen_failure_development_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=pins,parent126_unchanged=True,environment=environment(),software_gates=gates,
        stages=stages,scheduler_sha256=sha(Path(__file__)),contract_sha256=sha(ROOT/'review/V17接触任务空间多任务融合整体方案_v1.md'),
        known_failure_development=True,prospective_confirmation=False,
        shared_new_metadata=list(ADDED_FIELDS),original_sensor_values_and_RNG_unchanged=True,
        no_object_identity_or_truth_controller_input=True,
        original_metrics_windows_force_limits_duration_and_deadline_unchanged=True,
        numerical_repair_is_not_algorithm_contribution=True,
        stop_on_source_runtime_or_original_prefix_divergence=True,retain_task_failures=True)
    write(master,plan)
    for s in stages:
        assignment=dict(methods=['ratio_effort'],scenarios=s['scene'],seeds=[619732],motion_mode=s['motion'],
            physical_randomization=True,task_policy='integrated',receiver_policy=s['policy'])
        write(ROOT/s['registration'],dict(schema='v17_contact_transfer_registration_v1',source_sha256=pins,
            environment=environment(),assignment=assignment,software_gate_passed=True,
            master_plan_sha256=sha(master),development_only=True,performance_claim_allowed=False))
    progress=ROOT/'review/v17_transfer_small_progress_v1.json'
    state=dict(master_plan_sha256=sha(master),complete=False,physics_active=True,stages=[])
    write(progress,state);os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    for s in stages:
        if source_identity(['ratio_effort'])!=pins:raise RuntimeError('Source133 changed before stage')
        cmd=[sys.executable,'-B','-m','v17_contact_transfer.run_development','--out',s['output'],'--registration',s['registration'],
            '--methods','ratio_effort','--scenarios',*s['scene'],'--seeds','619732','--motion-mode',s['motion'],
            '--task-policy','integrated','--receiver-policy',s['policy'],'--physical-randomization','--workers',str(s['planned'])]
        print(json.dumps(dict(starting=s['id'],planned=s['planned'])),flush=True)
        done=subprocess.run(cmd,cwd=ROOT,env=os.environ.copy())
        finish=read(ROOT/s['output']/'execution_complete.json')
        row=dict(id=s['id'],returncode=done.returncode,execution=finish,comparisons=[])
        for scene in s['scene']:
            folder=ROOT/s['output']/'runs'/f'{scene}__ratio_effort__619732'
            if (folder/'trace.npz').exists():
                row['comparisons'].append(compare_original_fields(folder,previous_A(scene,s['motion'],619732),s['policy']!='blind'))
        state['stages'].append(row);write(progress,state)
        if done.returncode or not finish['execution_complete'] or len(row['comparisons'])!=s['planned'] or not all(r['passed'] for r in row['comparisons']):
            state.update(physics_active=False,stopped_after_stage=s['id']);write(progress,state);return 1
    state.update(complete=True,physics_active=False,ended_utc=datetime.now(timezone.utc).isoformat())
    write(progress,state);print(json.dumps(dict(complete=True,trials=4,sources=133)),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
