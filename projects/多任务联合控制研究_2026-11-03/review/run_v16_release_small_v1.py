"""Register once, run a passive exact bridge, then three seen release cases."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from v16_release_geometry.runtime import source_identity
from v15_support.runtime import source_identity as old_source_identity
from v13_a1.run_bridge import environment


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def read(path):return json.loads(path.read_text(encoding='utf-8'))


def previous_A(scene,motion,seed):
    rows=read(ROOT/'report/v15A_prospective_v2/paired_readout_v1.json')['records']
    matches=[r for r in rows if r['role']=='candidate_A' and r['context_key']==[scene,motion,seed]]
    if len(matches)!=1:raise RuntimeError('One preserved physical comparator required')
    return Path(matches[0]['bundle']['run_directory'])


PHYSICAL_COLUMNS=('time','normal','skin_normal','shear','palm','hand_other','all_hand_object',
    'support','penetration','world_penetration','position','relative','rotation','relative_velocity',
    'angular_velocity','lift','qpos','qvel','wrist_position','ctrl_before','ctrl','alpha_before','alpha',
    'robot_actuator_target21','executed_reference21','phase','command','observed_normal','validity',
    'observed_age','observed_auxiliary_wrench','observed_auxiliary_valid','legacy_common_sensor_json',
    'receiver_measurement_json','task_request_json','environment_json')


def compare_trace(new, old, before_release=False):
    checks={}
    with np.load(new/'trace.npz',allow_pickle=False) as a, np.load(old/'trace.npz',allow_pickle=False) as b:
        ma=a['time']<12.-1e-9 if before_release else np.ones(len(a['time']),bool)
        mb=b['time']<12.-1e-9 if before_release else np.ones(len(b['time']),bool)
        for key in PHYSICAL_COLUMNS:
            if key not in a or key not in b:
                checks[key]=dict(equal=False,reason='required_column_missing');continue
            aa=a[key][ma];bb=b[key][mb]
            checks[key]=dict(equal=bool(aa.dtype==bb.dtype and aa.shape==bb.shape and aa.tobytes()==bb.tobytes()),
                new_shape=list(aa.shape),old_shape=list(bb.shape))
    return dict(passed=all(x['equal'] for x in checks.values()),checks=checks,
        new_trace_sha256=sha(new/'trace.npz'),old_trace_sha256=sha(old/'trace.npz'),
        scope='strictly before12; release-action tick12 excluded' if before_release else 'complete16s physical and existing public instruments',
        timing_and_added_metadata_excluded=True,new=str(new),old=str(old))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-gate',required=True)
    parser.add_argument('--adapter-gate',required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:raise ValueError('Explicit execution switch required')
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    pins=source_identity(['ratio_effort'])
    old=read(ROOT/'review/v15A_prospective_candidate_A_hold_registration_v2.json')
    if old_source_identity(['ratio_effort'])!=old['source_sha256']:
        raise RuntimeError('Frozen114 changed')
    if environment()!=old['environment']:raise RuntimeError('Original binary environment changed')
    reports=['review/v16_hierarchy_independent_v2.json','review/v16_transfer_observer_independent_v2.json',
        'review/v16_supervisor_independent_v2.json','review/v16_cross_component_public_packet_v1.json',
        args.runtime_gate,args.adapter_gate]
    gates={}
    for name in reports:
        p=ROOT/name;j=read(p)
        if j.get('passed') is not True:raise RuntimeError('Software gate not passed: '+name)
        gates[name]=dict(sha256=sha(p),passed=True)
    stages=[dict(id='blind_bridge',scene=['nominal'],motion='transport',policy='blind',planned=1),
        dict(id='coordinated_hold',scene=['fragile_conflict'],motion='hold',policy='coordinated',planned=1),
        dict(id='coordinated_transport',scene=['nominal','friction_drop'],motion='transport',policy='coordinated',planned=2)]
    for stage in stages:
        stage['output']='output/v16_release_'+stage['id']+'_v1'
        stage['registration']='review/v16_release_'+stage['id']+'_registration_v1.json'
        if (ROOT/stage['output']).exists() or (ROOT/stage['registration']).exists():
            raise FileExistsError('No reruns/overwriting of an assigned attempt')
    master=ROOT/'review/v16_release_small_plan_v1.json'
    if master.exists():raise FileExistsError(master)
    plan=dict(schema='v16_release_seen_failure_development_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=pins,environment=environment(),software_gates=gates,stages=stages,
        scheduler_sha256=sha(Path(__file__)),contract_sha256=sha(ROOT/'review/V16统一释放接入合同_v1.md'),
        original114_unchanged=True,prospective_confirmation=False,known_failure_cases=True,
        original_metrics_windows_thresholds_and_labels_unchanged=True,physics_started=False)
    write(master,plan)
    for stage in stages:
        assignment=dict(methods=['ratio_effort'],scenarios=stage['scene'],seeds=[619732],
            motion_mode=stage['motion'],physical_randomization=True,task_policy='integrated',receiver_policy=stage['policy'])
        reg=dict(schema='v16_release_registration_v1',source_sha256=pins,environment=environment(),
            software_gate_passed=True,software_gates=gates,assignment=assignment,
            master_plan_sha256=sha(master),development_only=True,known_failure_diagnostic=True)
        write(ROOT/stage['registration'],reg)
    progress_path=ROOT/'review/v16_release_small_progress_v1.json'
    progress=dict(master_plan_sha256=sha(master),complete=False,physics_started=True,stages=[])
    write(progress_path,progress)
    for stage in stages:
        if source_identity(['ratio_effort'])!=pins:raise RuntimeError('Sources changed before stage')
        cmd=[sys.executable,'-B','-m','v16_release_geometry.run_development','--out',stage['output'],
            '--registration',stage['registration'],'--methods','ratio_effort','--scenarios',*stage['scene'],
            '--seeds','619732','--motion-mode',stage['motion'],'--task-policy','integrated',
            '--receiver-policy',stage['policy'],'--physical-randomization','--workers',str(stage['planned'])]
        print(json.dumps(dict(starting=stage['id'],planned=stage['planned']),ensure_ascii=False),flush=True)
        completed=subprocess.run(cmd,cwd=ROOT,env=os.environ.copy())
        finish=read(ROOT/stage['output']/'execution_complete.json')
        record=dict(id=stage['id'],returncode=completed.returncode,execution=finish,comparisons=[])
        for scene in stage['scene']:
            new=ROOT/stage['output']/'runs'/f'{scene}__ratio_effort__619732'
            if (new/'trace.npz').exists():
                record['comparisons'].append(compare_trace(new,previous_A(scene,stage['motion'],619732),stage['policy']!='blind'))
        progress['stages'].append(record);write(progress_path,progress)
        if completed.returncode or not finish['execution_complete'] or not all(x['passed'] for x in record['comparisons']):
            progress.update(physics_active=False,stopped_after_stage=stage['id'])
            write(progress_path,progress)
            print('Stage stopped; retained failed software/physical equivalence evidence.',flush=True)
            return 1
    progress.update(complete=True,physics_started=True,physics_active=False,ended_utc=datetime.now(timezone.utc).isoformat())
    write(progress_path,progress)
    print(json.dumps(dict(complete=True,total=4,source_count=len(pins)),ensure_ascii=False),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
