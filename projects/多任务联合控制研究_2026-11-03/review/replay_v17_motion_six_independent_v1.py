"""Six prespecified saved CONTACT checks via authenticated public robot adapter.

First/last accepted motion-available CONTACT decision per coordinated case.
No original M was logged: reconstruction is checked against saved geometry,
motion metadata and attained residuals, not claimed byte-equal to unlogged M.
Only private robot FK/Jac/tendon algebra is allowed; no plant/step/forward/solve.
"""
from pathlib import Path
from collections import Counter
from contextlib import ExitStack
from unittest.mock import patch
import sys,hashlib,json,gzip,traceback,datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[1];FOUNDATION=ROOT.parent/'五指策略接入_2026-11-03'
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(FOUNDATION));sys.dont_write_bytecode=True
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x
class PublicFrame:
    def __init__(self,values,accessed):self._values=values;self._accessed=accessed
    def __getattr__(self,name):
        if name.startswith(('object','truth','true_','target_id','scene','mass','friction')) or name in ('qpos','qvel','position','rotation','relative'):
            raise AssertionError('Forbidden truth field '+name)
        self._accessed.add(name)
        if name not in self._values:raise AttributeError(name)
        return self._values[name]

def main():
    if sys.argv[1:]!=['--execute']:raise ValueError('Explicit post-terminal replay only')
    plan=read(ROOT/'review/v17_transfer_small_plan_v1.json');corepath=ROOT/'review/v17_small_independent_v1.json';core=read(corepath)
    if not core['passed'] or len(core['rows'])!=4 or not all(read(ROOT/s['output']/'execution_complete.json')['execution_complete'] is True for s in plan['stages']):raise RuntimeError('Complete core gate required before reading any branch')
    import research_imports,mujoco,hand_plant
    from foundation_config import FoundationConfig,OPEN
    from v15_support.receiver_sensor import ReceiverMount,ReceiverSensor
    from v17_contact_transfer.motion_adapter import PublicMotionAdapter
    from v13_a1.codec import ReferenceActor21
    forbidden=[];allowed=Counter();rows=[];issues=[];selections=[];owned=[]
    def deny(name):
        def f(*a,**kw):forbidden.append(name);raise AssertionError('Forbidden pipeline '+name)
        return f
    with ExitStack() as guards:
        for name in ('mj_step','mj_step1','mj_step2','mj_Euler','mj_RungeKutta','mj_forward','mj_forwardSkip','mj_fwdPosition','mj_fwdVelocity','mj_fwdActuation','mj_fwdAcceleration','mj_fwdConstraint','mj_collision','mj_contactForce','mj_inverse'):
            if hasattr(mujoco,name):guards.enter_context(patch.object(mujoco,name,deny(name)))
        guards.enter_context(patch.object(hand_plant.HandPlant,'__init__',deny('HandPlant construction')))
        guards.enter_context(patch.object(hand_plant.HandPlant,'advance',deny('plant advance')))
        guards.enter_context(patch.object(ReferenceActor21,'apply',deny('actor apply')))
        guards.enter_context(patch.object(ReceiverSensor,'sample',deny('sensor resample')))
        for name in ('mj_kinematics','mj_comPos','mj_jac','mj_tendon','mj_transmission','mj_resetData'):
            original=getattr(mujoco,name)
            def count(*a,_name=name,_original=original,**kw):allowed[_name]+=1;return _original(*a,**kw)
            guards.enter_context(patch.object(mujoco,name,count))
        for stage in [s for s in plan['stages'] if s['policy']=='coordinated']:
          for scenario in stage['scene']:
            run=ROOT/stage['output']/'runs'/f'{scenario}__ratio_effort__619732';meta=read(run/'metadata.json')
            with gzip.open(run/'release_decisions.jsonl.gz','rt',encoding='utf-8') as f:records=[json.loads(line) for line in f]
            eligible=[q for q in records if q['phase']=='CONTACT_UNLOADING' and q['geometry_proposal_accepted'] and q['motion_task_available'] and any(l['name']=='observed_external_contact_motion_correction' for l in q['hierarchy_levels'])]
            if len(eligible)<2:issues.append(dict(scenario=scenario,motion=stage['motion'],reason='fewer_than2_eligible_CONTACT_records',eligible=len(eligible)));continue
            selected=[eligible[0],eligible[-1]];selections.append(dict(scenario=scenario,motion=stage['motion'],rule='first_and_last_accepted_motion_available_CONTACT',eligible_count=len(eligible),decision_ids=[q['decision_id'] for q in selected],times_s=[q['time_s'] for q in selected]))
            with np.load(run/'trace.npz',allow_pickle=False) as z:times=z['time'];frames=[json.loads(str(v)) for v in z['observation_json']]
            model,_=hand_plant._model(FoundationConfig(**meta['config']))
            sid=model.geom('support').id;half=float(model.geom_size[sid,2])
            mount=ReceiverMount(geom_name='support',excluded_geom_names=('floor','support_post'),origin_in_geom_m=np.array([0.,0.,half]),top_plane_tolerance_m=half,normal_alignment_min_cos=float(np.cos(np.pi/4)))
            adapter=PublicMotionAdapter(model,open18=np.concatenate(OPEN),receiver_mount=mount)
            owned.append(dict(scenario=scenario,geometry_private_data_id=id(adapter.geometry.data),servo_private_data_id=id(adapter.servo.data),distinct=adapter.geometry.data is not adapter.servo.data))
            origin=float(next(q for q in records)['public_process']['started_s']);start=max(0,round((origin-.2)/.005));last=round(selected[-1]['time_s']/.005);by_index={round(q['time_s']/.005):q for q in selected};accessed=set()
            for i in range(start,last+1):
                frame=PublicFrame(frames[i],accessed);adapter.observe(frame)
                if abs(times[i]-origin)<1e-9:adapter.begin_release(origin)
                if i not in by_index:continue
                q=by_index[i];current=np.asarray(q['current_reference21']);nominal=np.asarray(q['nominal_reference21']);lo=np.asarray(q['actual_step_lower21']);hi=np.asarray(q['actual_step_upper21']);wanted=np.clip(nominal-current,lo,hi)
                try:
                    geometry=adapter.current_geometry_tasks(frame,current,current+wanted,lo,hi);motion=adapter.motion_task
                    if motion is None:raise AssertionError('Replay motion unavailable: '+str(adapter.last_motion_diagnostics))
                    C=np.asarray(q['normal_rows']);T=np.asarray(q['tangent_rows']);Cerr=float(np.max(abs(geometry.normal_matrix_m_per_reference-C),initial=0));Terr=float(np.max(abs(geometry.tangent_matrix_m_per_reference-T),initial=0))
                    scale=np.asarray(q['scale21']);x=np.asarray(q['proposed_reference_increment21'])/scale;lower=lo/scale;upper=hi/scale;ms=motion.matrix_Mx21*scale[None,:];b=motion.target_M
                    common=max(float(np.max(np.sum(abs(ms)*np.maximum(abs(lower),abs(upper))[None,:],axis=1),initial=0)),float(np.max(abs(b),initial=0)));common=common if common else 1.
                    residual=(ms/common)@x-b/common;saved=next(l for l in q['hierarchy_levels'] if l['name']=='observed_external_contact_motion_correction');savedr=np.asarray(saved['residual']);err=float(np.max(abs(residual-savedr),initial=0))
                    tol=saved['numerical_atol']+saved['numerical_rtol']*max(1.,float(np.max(abs(x))))
                    original_meta=q['geometry']['motion_domain'];recorded_projection=np.concatenate([v['measured_contact_average_projection_m_s'] for v in original_meta['rows']]);target_from_record=-.02*recorded_projection
                    targeterr=float(np.max(abs(b-target_from_record),initial=0));digits=motion.metadata['selected_public_digits']
                    passed=(geometry.status==q['geometry_status'] and Cerr<=1e-13 and Terr<=1e-13 and motion.time_s==q['time_s'] and digits==original_meta['selected_public_digits'] and motion.metadata['task_matrix_rank']==original_meta['task_matrix_rank'] and np.isfinite(motion.matrix_Mx21).all() and targeterr<=128*np.finfo(float).eps*max(1.,float(np.max(abs(b),initial=0))) and residual.shape==savedr.shape and err<=tol+128*np.finfo(float).eps)
                    row=dict(scenario=scenario,motion=stage['motion'],decision_id=q['decision_id'],time_s=q['time_s'],passed=passed,matrix_Mx21=motion.matrix_Mx21,target_M=b,reconstructed_motion_metadata=motion.metadata,selected_digits=digits,normal_rows_max_error=Cerr,tangent_rows_max_error=Terr,normal_rows_bytes_equal=geometry.normal_matrix_m_per_reference.tobytes()==C.tobytes(),tangent_rows_bytes_equal=geometry.tangent_matrix_m_per_reference.tobytes()==T.tobytes(),recorded_target_max_error=targeterr,final_motion_residual=residual,saved_attained_motion_residual=savedr,residual_error_inf=err,declared_frozen_residual_tolerance=tol,matrix_rank=motion.metadata['task_matrix_rank'],matrix_sha256=hashlib.sha256(motion.matrix_Mx21.tobytes()).hexdigest(),input_trace_sha256=sha(run/'trace.npz'),input_decisions_sha256=sha(run/'release_decisions.jsonl.gz'),public_fields_accessed=sorted(accessed))
                    rows.append(row)
                    if not passed:issues.append(dict(scenario=scenario,decision_id=q['decision_id'],reason='reconstructed_motion_not_consistent',Cerr=Cerr,Terr=Terr,targeterr=targeterr,residualerr=err))
                except Exception as exc:issues.append(dict(scenario=scenario,decision_id=q['decision_id'],error=repr(exc),traceback=traceback.format_exc()))
    if len(rows)!=6:issues.append(dict(reason='not6_completed_records',completed=len(rows)))
    if forbidden:issues.append(dict(reason='forbidden_pipeline',calls=forbidden))
    source_changes=[k for k,v in plan['source_sha256'].items() if sha((FOUNDATION/k[11:]) if k.startswith('foundation/') else ROOT/k)!=v]
    if source_changes:issues.append(dict(reason='source_changed',files=source_changes))
    report=dict(schema='v17-motion-six-reconstruction-independent-v1',passed=not issues,role='selected_saved_public_pipeline_reconstruction_not_full_motion_matrix_audit',core_audit_sha256=sha(corepath),checker_sha256=sha(Path(__file__)),selections=selections,rows=rows,issues=issues,source_sha256=plan['source_sha256'],pipeline_calls=dict(forbidden=forbidden,allowed_robot_only=dict(allowed)),known_private_robot_data=owned,scope=dict(selected_records=6,cases=3,static_model_compilations=3,plant_constructions=0,physics_steps=0,forward_calls=0,contact_queries=0,sensor_resamples=0,control_writes=0,actual_solver_replays=0),limitations=['Original fullM/b were not logged. Reconstruction matches saved geometry/metadata/attained residuals, not byte comparison to an absent originalM.','Two transparently selected eligible records per case, not all CONTACT matrices.','Reuses authenticated production adapter and frozen robot algebra; not an independent implementation of the whole motion formula.','Motion residual evaluated at final x reconstructed from delta/scale, using original declared freeze tolerance; no evaluation threshold changed.','No future pose, object truth, target identity or actual plant data entered replay.'])
    out=ROOT/'review/v17_motion_six_reconstruction_independent_v1.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(plain(report),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],rows=len(rows),issues=issues[:6],allowed_calls=dict(allowed),path=str(out),sha256=sha(out)),ensure_ascii=False));return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
