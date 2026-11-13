"""Register and run an implementation goal correction, retaining V16 failures."""
from pathlib import Path
import argparse,json,os,subprocess,sys,hashlib
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from v16b_release_goal.runtime import source_identity
from v13_a1.run_bridge import environment
from review.run_v16_release_small_v1 import previous_A,sha,read,write
from review.resume_v16_release_small_v2 import compare_payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:raise ValueError('Explicit execution required')
    pins=source_identity(['ratio_effort'])
    parent=read(ROOT/'review/v16_release_small_plan_v1.json')
    if len(pins)!=126 or any(pins.get(k)!=v for k,v in parent['source_sha256'].items()):
        raise RuntimeError('Frozen123 or expected3 new source closure changed')
    if environment()!=parent['environment']:raise RuntimeError('Original42 binary environment changed')
    gates={}
    for name in ['review/v16b_goal_root_pure_v1.json','review/v16b_goal_independent_v1.json','review/v16b_runtime_bridge_independent_v1.json']:
        p=ROOT/name;j=read(p)
        if j.get('passed') is not True:raise RuntimeError('Software gate not passed: '+name)
        gates[name]=dict(passed=True,sha256=sha(p))
    master=ROOT/'review/v16b_release_small_plan_v1.json'
    if master.exists():raise FileExistsError('Preserve every registration and attempt')
    stages=[dict(id='blind_bridge',scene=['nominal'],motion='transport',policy='blind',planned=1),
        dict(id='corrected_hold',scene=['fragile_conflict'],motion='hold',policy='coordinated',planned=1),
        dict(id='corrected_transport',scene=['nominal','friction_drop'],motion='transport',policy='coordinated',planned=2)]
    for s in stages:
        s['output']='output/v16b_goal_'+s['id']+'_v1'
        s['registration']='review/v16b_goal_'+s['id']+'_registration_v1.json'
        if (ROOT/s['output']).exists() or (ROOT/s['registration']).exists():raise FileExistsError('Output/registration already exists')
    helpers=['review/run_v16_release_small_v1.py','review/resume_v16_release_small_v2.py']
    plan=dict(schema='v16b_release_goal_correction_development_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=pins,environment=environment(),parent123_sha256=parent['source_sha256'],
        software_gates=gates,stages=stages,scheduler_sha256=sha(Path(__file__)),
        reader_helpers_sha256={n:sha(ROOT/n) for n in helpers},
        seen_conditions=True,prospective_confirmation=False,implementation_correction_not_algorithm_contribution=True,
        correction='preparation OPEN and release clearance endpoint explicitly separated; clearance18 from frozen lifecycle saturated endpoint',
        unchanged=['physical model','native gains','force priors','holding coordinator','sensor/noise streams','real actor permissions/rates',
            '16s duration','12s release request','3.5s original clearance deadline','original evaluators/windows/thresholds'],
        full_comparison_required=['original task/full-motion/clearance','force guard/peaks/dose','receiver/floor contact',
            'terminal object motion','actual goal/reference/sensed-contact discrepancy','task progress/fallback/transfer permissions'],
        stop_on_software_or_prefix_divergence=True,do_not_stop_or_discard_on_task_failure=True,
        trials_to_run=4,retained_previous_v16_trials=4)
    write(master,plan)
    for s in stages:
        assignment=dict(methods=['ratio_effort'],scenarios=s['scene'],seeds=[619732],motion_mode=s['motion'],
            physical_randomization=True,task_policy='integrated',receiver_policy=s['policy'])
        write(ROOT/s['registration'],dict(schema='v16b_goal_registration_v1',source_sha256=pins,
            environment=environment(),assignment=assignment,software_gate_passed=True,software_gates=gates,
            master_plan_sha256=sha(master),known_failure_development=True,performance_claim_allowed=False))
    progress=ROOT/'review/v16b_release_small_progress_v1.json'
    state=dict(schema='v16b_development_execution_progress_v1',master_plan_sha256=sha(master),complete=False,
        physics_active=True,stages=[])
    write(progress,state)
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    for s in stages:
        if source_identity(['ratio_effort'])!=pins:raise RuntimeError('Frozen source126 changed')
        cmd=[sys.executable,'-B','-m','v16b_release_goal.run_development','--out',s['output'],'--registration',s['registration'],
            '--methods','ratio_effort','--scenarios',*s['scene'],'--seeds','619732','--motion-mode',s['motion'],
            '--task-policy','integrated','--receiver-policy',s['policy'],'--physical-randomization','--workers',str(s['planned'])]
        print(json.dumps(dict(starting=s['id'],planned=s['planned'])),flush=True)
        done=subprocess.run(cmd,cwd=ROOT,env=os.environ.copy())
        finish=read(ROOT/s['output']/'execution_complete.json')
        entry=dict(id=s['id'],returncode=done.returncode,execution=finish,comparisons=[])
        for scene in s['scene']:
            folder=ROOT/s['output']/'runs'/f'{scene}__ratio_effort__619732'
            if (folder/'trace.npz').exists():
                entry['comparisons'].append(compare_payload(folder,previous_A(scene,s['motion'],619732),s['policy']!='blind'))
        state['stages'].append(entry);write(progress,state)
        if done.returncode or not finish['execution_complete'] or len(entry['comparisons'])!=s['planned'] or not all(c['passed'] for c in entry['comparisons']):
            state.update(physics_active=False,stopped_after_stage=s['id']);write(progress,state)
            return 1
    state.update(complete=True,physics_active=False,ended_utc=datetime.now(timezone.utc).isoformat())
    write(progress,state)
    print(json.dumps(dict(complete=True,trials=4,source_count=126)),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
