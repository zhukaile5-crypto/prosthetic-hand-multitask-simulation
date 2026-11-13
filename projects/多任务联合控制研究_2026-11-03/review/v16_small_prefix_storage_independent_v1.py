"""Terminal TWO-run storage/payload diagnostic; not full four-run acceptance.

No model, physics, evaluator or result-based selection. Strict numeric bytes;
Unicode payload equality is reported beside, never instead of literal failure.
"""
from pathlib import Path
import json,hashlib,sys,datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
COLS=('time','normal','skin_normal','shear','palm','hand_other','all_hand_object','support','penetration','world_penetration','position','relative','rotation','relative_velocity','angular_velocity','lift','qpos','qvel','wrist_position','ctrl_before','ctrl','alpha_before','alpha','robot_actuator_target21','executed_reference21','phase','command','observed_normal','validity','observed_age','observed_auxiliary_wrench','observed_auxiliary_valid','legacy_common_sensor_json','receiver_measurement_json','task_request_json','environment_json','raw_patches_json')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x
def compare(a,b,time_a,time_b,pre):
    ma=time_a<12.-1e-9 if pre else np.ones(len(time_a),bool);mb=time_b<12.-1e-9 if pre else np.ones(len(time_b),bool)
    out=[]
    for k in COLS:
        aa=a[k][ma];bb=b[k][mb];shape=aa.shape==bb.shape;dtype=aa.dtype==bb.dtype
        literal=shape and dtype and aa.tobytes()==bb.tobytes()
        text=aa.dtype.kind in 'US' and bb.dtype.kind in 'US'
        payload=shape and (np.array_equal(aa,bb) if text else literal)
        first=None
        if a[k].shape==b[k].shape:
            if text:changed=a[k]!=b[k];changed=changed if changed.ndim==1 else changed.any(axis=tuple(range(1,changed.ndim)))
            elif a[k].dtype==b[k].dtype:
                va=np.ascontiguousarray(a[k]).view(np.uint8).reshape(len(time_a),-1);vb=np.ascontiguousarray(b[k]).view(np.uint8).reshape(len(time_b),-1);changed=(va!=vb).any(axis=1)
            else:changed=np.ones(len(time_a),bool)
            ii=np.flatnonzero(changed)
            if len(ii):first=dict(index=int(ii[0]),time_s=float(time_a[ii[0]]))
        out.append(dict(column=k,unicode=text,new_dtype=str(aa.dtype),old_dtype=str(bb.dtype),shape=list(aa.shape),same_shape=shape,same_dtype=dtype,literal_storage_equal=bool(literal),exact_payload_equal=bool(payload),first_full_trace_payload_difference=first))
    return out
def main():
    if sys.argv[1:]!=['--execute']:raise ValueError('Explicit terminal read only')
    plan=read(ROOT/'review/v16_release_small_plan_v1.json');progress=read(ROOT/'review/v16_release_small_progress_v1.json')
    stageids=('blind_bridge','coordinated_hold');stages=[next(x for x in plan['stages'] if x['id']==key) for key in stageids]
    completions={x['id']:read(ROOT/x['output']/'execution_complete.json') for x in stages}
    if not all(x['execution_complete'] is True for x in completions.values()):raise RuntimeError('Both requested branches must be complete before any raw read')
    if progress.get('physics_active') is not False or progress.get('stopped_after_stage')!='coordinated_hold':raise RuntimeError('Bind this report to the retained stopped-v1 sequence')
    oldplan=read(ROOT/'review/v15A_prospective_multiscene_plan_v2.json');rows=[]
    for s in stages:
        scenario=s['scene'][0];folder=ROOT/s['output'];run=folder/'runs'/f'{scenario}__ratio_effort__619732'
        oldstage=next(x for x in oldplan['stages'] if x['arm']=='candidate_A' and x['motion_mode']==s['motion'])
        oldrun=ROOT/oldstage['output']/'runs'/f'{scenario}__ratio_effort__619732'
        def load(path):
            with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in COLS}
        a=load(run/'trace.npz');b=load(oldrun/'trace.npz');comparison=compare(a,b,a['time'],b['time'],s['policy']!='blind')
        meta=read(run/'metadata.json');oldmeta=read(oldrun/'metadata.json');result=read(run/'result.json')
        physical_keys=('shape','scenario','seed','split','config','initial_position','limits','budget','noise','model_sha256','motion_mode','observer_warmup_s')
        rows.append(dict(stage=s['id'],scope='entire16s' if s['policy']=='blind' else 'strictly_before12s_release_tick_excluded',new_run=str(run),old_run=str(oldrun),new_trace_sha256=sha(run/'trace.npz'),old_trace_sha256=sha(oldrun/'trace.npz'),new_metadata_sha256=sha(run/'metadata.json'),old_metadata_sha256=sha(oldrun/'metadata.json'),result_sha256=sha(run/'result.json'),completion_sha256=sha(folder/'execution_complete.json'),source_identity_matches_master=meta['research_source_sha256']==plan['source_sha256'],physical_config_same=all(meta.get(k)==oldmeta.get(k) for k in physical_keys),full16_completed=meta['completed'] and len(a['time'])==3201 and a['time'][-1]==16.,columns=comparison,literal_storage_all37=all(x['literal_storage_equal'] for x in comparison),exact_payload_all37=all(x['exact_payload_equal'] for x in comparison),original_task_success=result['task_success'],original_release=result.get('release'),original_research_mission=result.get('research_metrics',{}).get('full_motion_mission_success')))
    assigned=[]
    for s in plan['stages']:
        for scenario in s['scene']:assigned.append(dict(stage=s['id'],scenario=scenario,motion=s['motion'],seed=619732,method='ratio_effort',status='completed_audited_prefix_only' if s['id'] in stageids else 'not_read_pending_resumed_transport'))
    passed=len(COLS)==37 and all(x['exact_payload_all37'] and x['physical_config_same'] and x['source_identity_matches_master'] and x['full16_completed'] for x in rows) and rows[0]['literal_storage_all37']
    report=dict(schema='v16-small-prefix-storage-independent-v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=passed,cohort_complete=False,scope='Two completed branches only; two unexecuted planned branches retained; not physical or full four-trial acceptance',script_sha256=sha(Path(__file__)),master_plan_sha256=sha(ROOT/'review/v16_release_small_plan_v1.json'),stopped_progress_sha256=sha(ROOT/'review/v16_release_small_progress_v1.json'),stopped_progress=progress,columns37=COLS,rows=rows,assigned=assigned,source_sha256=plan['source_sha256'],limitations=['Literal storage failures remain failures, even when caused only by Unicode width/padding.','Numeric comparison remains exact dtype/shape/bytes; no tolerance relaxed.','No old score, threshold, outcome, source or retained attempt is changed.','No model, physics, FK, evaluator or full cohort performance readout.'])
    p=ROOT/'review/v16_small_prefix_storage_independent_v1.json'
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(plain(report),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(report=str(p),sha256=sha(p),passed=passed,cohort_complete=False,rows=[dict(stage=x['stage'],literal37=x['literal_storage_all37'],payload37=x['exact_payload_all37'],storage_failures=[c['column'] for c in x['columns'] if not c['literal_storage_equal']],payload_failures=[c['column'] for c in x['columns'] if not c['exact_payload_equal']]) for x in rows]),ensure_ascii=False))
    return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
