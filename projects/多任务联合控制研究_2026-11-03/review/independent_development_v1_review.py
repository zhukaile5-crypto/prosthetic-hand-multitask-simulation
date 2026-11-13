"""Read-only first18 development review; uses immutable evaluator snapshot."""
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import hashlib,io,json,runpy
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'output/development_v1'
SNAPSHOT=FOLDER/'sources'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def equivalent(a,b):
    if isinstance(a,dict) and isinstance(b,dict):return set(a)==set(b) and all(equivalent(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    if a is None or b is None:return a is b
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):return abs(a-b)<1e-8
    return a==b


def dwell(t,good,start,end,seconds,exclusive=False):
    since=None
    for i in np.flatnonzero((t>=start-1e-9)&(t<end-1e-9 if exclusive else t<=end+1e-9)):
        if good[i]:
            since=float(t[i]) if since is None else since
            if t[i]-since>=seconds-1e-9:return float(t[i])
        else:since=None
    return None


def physical_outcome(z,m):
    t=z['time'];n=z['normal'];support=(n[:,0]>.2)&((n>.2).sum(axis=1)>=3)
    raised=z['position'][:,2]>=m['initial_position'][2]+.025
    error=np.linalg.norm(z['relative']-m['reference_relative'],axis=1)
    drop=dwell(t,~support|~raised|(error>.08),7.5,12.,.1,True)
    tracking=dwell(t,abs(z['lift']-z['lift_ref'])>.01,7.5,12.,.3,True)
    failure=min([v for v in (drop,tracking) if v is not None],default=None)
    i=np.argmin(abs(t-12.));exposure=bool(support[i] and raised[i] and z['support'][i]<=.05 and z['all_hand_object'][i]>.2)
    clear=dwell(t,z['all_hand_object']<=.2,12.,15.5,.2)
    physical=bool(exposure and clear is not None)
    task=bool(m['handoff_passed'] and failure is None and physical and z['warning_count'].max()==0)
    skin=z['substep_skin_normal'];limits=np.asarray(m['limits']);budget=m['budget'];dur=z['substep_duration']
    safety=bool((skin<=limits+.02).all() and (skin.sum(axis=1)<=budget+.02).all() and (z['skin_normal'][-1]<=limits+.02).all() and z['skin_normal'][-1].sum()<=budget+.02)
    return dict(physical_exposure=exposure,physical_release=physical,task_success=task,
        physical_release_latency=None if clear is None or not exposure else float(clear-12.),failure_at=failure,
        force_guard=safety,local_excess=float(np.sum(np.maximum(skin-limits,0)*dur[:,None])),
        total_excess=float(np.maximum(skin.sum(axis=1)-budget,0)@dur))


def main():
    manifest=read(FOLDER/'manifest.json');finish=read(FOLDER/'execution_complete.json');trials=read(FOLDER/'trials.json')
    evaluator=runpy.run_path(str(SNAPSHOT/'evaluation.py'))['evaluate_research']
    checks=dict(development_only=manifest['phase']=='development' and not manifest['performance_claim_allowed'],
        full18_retained=finish['execution_complete'] and finish['trials']==finish['planned']==len(trials)==18 and not finish['runtime_failures'],
        trials_hash=sha(FOLDER/'trials.json')==finish['trials_sha256'],
        declared_execution_snapshot_complete=all((SNAPSHOT/n).is_file() and sha(SNAPSHOT/n)==h for n,h in manifest['research_source_sha256'].items()))
    jobs={(j['scenario'],j['method'],j['seed']):j for j in manifest['jobs']}
    checks['exact_assigned_matrix']={(r['scenario'],r['method'],r['seed']) for r in trials}==set(jobs) and len(jobs)==18
    grouped=defaultdict(dict);reviewed=[];auxiliary=set()
    for row in trials:
        identity=(row['scenario'],row['method'],row['seed']);directory=Path(jobs[identity]['destination'])
        trace=directory/'trace.npz';m=read(directory/'metadata.json');saved=read(directory/'result.json')
        blob=trace.read_bytes();h=hashlib.sha256(blob).hexdigest()
        with np.load(io.BytesIO(blob),allow_pickle=False) as a:z={k:a[k] for k in a.files}
        del blob
        local=dict(raw_trace_identity=h==row['trace_sha256']==m['trace_sha256'],saved_result_identity=saved==row,
            snapshot_execution_sources_match=all(m['research_source_sha256'].get(n)==v for n,v in manifest['research_source_sha256'].items() if n!='protocol.json'),
            protocol_snapshot_binding=row['research_metrics']['protocol_sha256']==sha(SNAPSHOT/'protocol.json'),
            complete_sensor_clock=len(z['time'])==3201 and np.allclose(z['time'],np.arange(3201)*.005,atol=1e-8,rtol=0),
            complete_raw_clock=len(z['substep_time'])==32000 and np.allclose(z['substep_time'],np.arange(32000)*.0005,atol=1e-7,rtol=0) and abs(z['substep_duration'].sum()-16.)<1e-8)
        recalculated=evaluator(trace,SNAPSHOT/'protocol.json')
        local['snapshot_research_metric_recalculation']=equivalent(recalculated,row['research_metrics'])
        own=physical_outcome(z,m);research=row['research_metrics']
        local['independent_mission_and_physical_release']=(own['task_success']==research['mission_success'] and
            own['physical_exposure']==research['release']['physical_exposure'] and own['physical_release']==research['release']['physical_success'] and
            equivalent(own['physical_release_latency'],research['release']['physical_clearance_latency_s']))
        local['independent_all_substep_safety']=(own['force_guard']==research['safety']['guard_pass'] and
            abs(own['local_excess']-research['safety']['local_excess_integral_Ns'])<1e-8 and abs(own['total_excess']-research['safety']['total_excess_integral_Ns'])<1e-8)
        local['wrist_actual_encoders_recorded']=z['wrist_position'].shape==z['requested_wrist_reference'].shape==(3201,2)
        auxiliary |= set(m['research_source_sha256'])-set(manifest['research_source_sha256'])
        checks.update({str(identity)+'/'+k:bool(v) for k,v in local.items()})
        reviewed.append(dict(identity=identity,checks=local,independent_outcome=own,trace_sha256=h))
        prefix=z['time']<=7.5+1e-9
        grouped[(row['scenario'],row['seed'])][row['method']]=dict(
            arrays={k:z[k][prefix].copy() for k in ('qpos','qvel','ctrl_before','alpha_before','normal','skin_normal','observed_normal','validity','observed_auxiliary_wrench')},
            observations=z['observation_json'][prefix].copy(),raw_patches=z['raw_patches_json'][prefix].copy(),config=m['config'],noise=m['noise'])
        print(str(identity),all(local.values()),flush=True)
    common=[]
    for identity,methods in grouped.items():
        baseline=methods['ratio'];c={}
        for method in ('unified','joint'):
            candidate=methods[method]
            c[method+'/all_qpos_force_observed_numeric_equal']=all(np.array_equal(v,candidate['arrays'][k],equal_nan=True) for k,v in baseline['arrays'].items())
            c[method+'/every_observation_field_equal']=np.array_equal(baseline['observations'],candidate['observations'])
            c[method+'/all_raw_patches_equal']=np.array_equal(baseline['raw_patches'],candidate['raw_patches'])
            c[method+'/physical_and_noise_parameters_equal']=baseline['config']==candidate['config'] and baseline['noise']==candidate['noise']
        common.append(dict(identity=identity,compared_samples=1501,checks=c));checks.update({str(identity)+'/common/'+k:bool(v) for k,v in c.items()})
    summary=[]
    for method in ('ratio','unified','joint'):
        selected=[r for r in trials if r['method']==method]
        summary.append(dict(method=method,assigned=len(selected),canonical_task_success=sum(r['task_success'] for r in selected),
            research_mission_success=sum(r['research_metrics']['mission_success'] for r in selected),
            physical_releases=sum(r['research_metrics']['release']['physical_success'] for r in selected),
            task_qualified_releases=sum(r['research_metrics']['release']['task_qualified_success'] for r in selected),
            whole_force_guard_passes=sum(r['research_metrics']['safety']['guard_pass'] for r in selected),
            mean_static_finger_TV=float(np.mean([r['research_metrics']['static_quality']['bandlimited']['finger_TV_rate_N_s'] for r in selected if r['research_metrics']['static_quality']['eligible']]))))
    result=dict(passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],
        created_utc=datetime.now(timezone.utc).isoformat(),trials_sha256=sha(FOLDER/'trials.json'),manifest_sha256=sha(FOLDER/'manifest.json'),
        snapshot_source_sha256=manifest['research_source_sha256'],reviewed=reviewed,common_preparation=common,summary=summary,
        auxiliary_metadata_source_keys_not_in_execution_snapshot=sorted(auxiliary),
        limitations=['Development only: one physical seed and six scenarios, not formal superiority.',
            'Declared7 execution/protocol snapshot files are complete; metadata additionally hashes two nonexecuted validation helpers absent from that snapshot.',
            'Known v1 pressure-uncertainty truncation/negative-pressure model issues remain unaccepted and require a separately retained source revision.',
            'No new dynamics, performance tuning, original-file modifications or full preservation rehash were performed.'])
    out=ROOT/('review/independent_development_v1_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=result['failed'],path=str(out),sha256=sha(out),summary=summary),ensure_ascii=False))


if __name__=='__main__':main()
