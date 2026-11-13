"""Post80 selected public-robot reconstruction of newly logged M and target.

Three contexts are fixed before results: hold/fragile619741,
transport/nominal619741, transport/friction_drop619742. First/last accepted
motion-available CONTACT in each, without replacements if unavailable.
Private robot Data/FK/Jac/tendon is allowed; plant data, step, forward,
contact query, sensing, actor.apply and solver replays are forbidden.
"""
from pathlib import Path
from collections import Counter
from contextlib import ExitStack
from unittest.mock import patch
import gzip, hashlib, importlib.util, json, sys, traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOUNDATION=ROOT.parent/'五指策略接入_2026-11-03'
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(FOUNDATION));sys.dont_write_bytecode=True
SELECT=(('hold','fragile_conflict',619741),('transport','nominal',619741),('transport','friction_drop',619742))
PIN='b08fe0d57b2c43a8abcf58e9ff869d1fdc01ae0f2ed5cfe1546028e7268a6a06'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x
class PublicFrame:
    def __init__(self,values,accessed):self.values=values;self.accessed=accessed
    def __getattr__(self,name):
        if name.startswith(('object','truth','true_','target_id','scene','mass','friction')) or name in ('qpos','qvel','position','rotation','relative'):
            raise AssertionError('Forbidden controller truth field '+name)
        self.accessed.add(name)
        if name not in self.values:raise AttributeError(name)
        return self.values[name]


def main():
    if sys.argv[1:]!=['--execute']:raise ValueError('Explicit post-terminal appendix only')
    planpath=ROOT/'review/v17b_prospective80_plan_v1.json';plan=read(planpath)
    corepath=ROOT/'review/v17b_prospective80_execution_independent_v1.json';core=read(corepath)
    if sha(planpath)!=PIN or core['schema']!='v17b_prospective80_execution_independent_v1' or not core['passed'] or not core['all_assigned_checked'] or core['assigned_count']!=80 or core['plan_sha256']!=PIN:
        raise RuntimeError('Complete independent core80 gate required')
    if not all(read(ROOT/s['output']/'execution_complete.json')['execution_complete'] is True for s in plan['stages']):raise RuntimeError('All6 stages must be complete before reading a branch')
    out=ROOT/'review/v17b_motion_six_reconstruction_independent_v1.json'
    if out.exists():raise FileExistsError(out)
    import research_imports,mujoco,hand_plant
    from foundation_config import FoundationConfig,OPEN
    from v15_support.receiver_sensor import ReceiverMount,ReceiverSensor
    from v17_contact_transfer.motion_adapter import PublicMotionAdapter
    from v13_a1.codec import ReferenceActor21
    forbidden=[];allowed=Counter();rows=[];issues=[];selections=[];owned=[]
    def deny(name):
        def f(*a,**kw):forbidden.append(name);raise AssertionError('Forbidden '+name)
        return f
    with ExitStack() as guards:
        for name in ('mj_step','mj_step1','mj_step2','mj_Euler','mj_RungeKutta','mj_forward','mj_forwardSkip','mj_fwdPosition','mj_fwdVelocity','mj_fwdActuation','mj_fwdAcceleration','mj_fwdConstraint','mj_collision','mj_contactForce','mj_inverse'):
            if hasattr(mujoco,name):guards.enter_context(patch.object(mujoco,name,deny(name)))
        guards.enter_context(patch.object(hand_plant.HandPlant,'__init__',deny('HandPlant')))
        guards.enter_context(patch.object(hand_plant.HandPlant,'advance',deny('advance')))
        guards.enter_context(patch.object(ReferenceActor21,'apply',deny('actor.apply')))
        guards.enter_context(patch.object(ReceiverSensor,'sample',deny('receiver.sample')))
        for name in ('mj_kinematics','mj_comPos','mj_jac','mj_tendon','mj_transmission','mj_resetData'):
            original=getattr(mujoco,name)
            def count(*a,_name=name,_original=original,**kw):allowed[_name]+=1;return _original(*a,**kw)
            guards.enter_context(patch.object(mujoco,name,count))
        for movement,scene,seed in SELECT:
            stage=next(s for s in plan['stages'] if s['arm']=='full_V17' and s['motion_mode']==movement)
            run=ROOT/stage['output']/'runs'/f'{scene}__ratio_effort__{seed}';meta=read(run/'metadata.json')
            with gzip.open(run/'release_decisions.jsonl.gz','rt',encoding='utf-8') as f:records=[json.loads(s) for s in f]
            eligible=[q for q in records if q['phase']=='CONTACT_UNLOADING' and q['geometry_proposal_accepted'] and q['motion_task_available']]
            selection=dict(scenario=scene,motion=movement,seed=seed,eligible_count=len(eligible),rule='first_and_last; no alternative context selected')
            selections.append(selection)
            if len(eligible)<2:
                issues.append(dict(context=[movement,scene,seed],reason='prespecified_context_has_fewer_than2_eligible_records',eligible=len(eligible)));continue
            chosen=(eligible[0],eligible[-1]);selection.update(times_s=[q['time_s'] for q in chosen],decision_ids=[q['decision_id'] for q in chosen])
            with np.load(run/'trace.npz',allow_pickle=False) as z:
                times=z['time'];frames=[json.loads(str(v)) for v in z['observation_json']]
            model,xmlsha=hand_plant._model(FoundationConfig(**meta['config']))
            if xmlsha!=meta['model_sha256']:raise AssertionError('Model identity mismatch')
            sid=model.geom('support').id;half=float(model.geom_size[sid,2])
            mount=ReceiverMount(geom_name='support',excluded_geom_names=('floor','support_post'),origin_in_geom_m=np.array([0.,0.,half]),top_plane_tolerance_m=half,normal_alignment_min_cos=float(np.cos(np.pi/4)))
            adapter=PublicMotionAdapter(model,open18=np.concatenate(OPEN),receiver_mount=mount)
            owned.append(dict(context=[movement,scene,seed],geometry_private_data_id=id(adapter.geometry.data),servo_private_data_id=id(adapter.servo.data),distinct=adapter.geometry.data is not adapter.servo.data))
            origin=float(records[0]['public_process']['started_s']);start=max(0,round((origin-.2)/.005));last=round(chosen[-1]['time_s']/.005)
            by_index={round(q['time_s']/.005):q for q in chosen};accessed=set()
            for i in range(start,last+1):
                frame=PublicFrame(frames[i],accessed);adapter.observe(frame)
                if abs(times[i]-origin)<1e-9:adapter.begin_release(origin)
                if i not in by_index:continue
                q=by_index[i];current=np.asarray(q['current_reference21']);nominal=np.asarray(q['nominal_reference21']);lo=np.asarray(q['actual_step_lower21']);hi=np.asarray(q['actual_step_upper21']);wanted=np.clip(nominal-current,lo,hi)
                try:
                    geometry=adapter.current_geometry_tasks(frame,current,current+wanted,lo,hi);motion=adapter.motion_task
                    if motion is None:raise AssertionError('Reconstructed task unavailable '+str(adapter.last_motion_diagnostics))
                    logged=q['motion_task_public'];M=np.asarray(logged['matrix_Mx21']);b=np.asarray(logged['target_M'])
                    err=float(np.max(abs(motion.matrix_Mx21-M),initial=0.));berr=float(np.max(abs(motion.target_M-b),initial=0.))
                    c=np.asarray(q['normal_rows']);t=np.asarray(q['tangent_rows'])
                    passed=(motion.matrix_Mx21.shape==M.shape and motion.matrix_Mx21.tobytes()==M.tobytes() and motion.target_M.tobytes()==b.tobytes() and motion.time_s==q['time_s']==logged['time_s'] and geometry.normal_matrix_m_per_reference.tobytes()==c.tobytes() and geometry.tangent_matrix_m_per_reference.tobytes()==t.tobytes())
                    row=dict(context=[movement,scene,seed],decision_id=q['decision_id'],time_s=q['time_s'],passed=passed,
                        matrix_bytes_equal=motion.matrix_Mx21.tobytes()==M.tobytes(),target_bytes_equal=motion.target_M.tobytes()==b.tobytes(),matrix_error_inf=err,target_error_inf=berr,
                        matrix_sha256=hashlib.sha256(M.tobytes()).hexdigest(),shape=list(M.shape),public_fields_accessed=sorted(accessed),
                        input_trace_sha256=sha(run/'trace.npz'),input_decision_sha256=sha(run/'release_decisions.jsonl.gz'))
                    rows.append(row)
                    if not passed:issues.append(dict(context=row['context'],decision_id=q['decision_id'],reason='independent_loggedM_reconstruction_difference',matrixerr=err,targeterr=berr))
                except Exception as exc:issues.append(dict(context=[movement,scene,seed],decision_id=q['decision_id'],error=repr(exc),traceback=traceback.format_exc()))
    if len(rows)!=6:issues.append(dict(reason='not6 completed prespecified records',completed=len(rows)))
    if forbidden:issues.append(dict(reason='forbidden_pipeline',calls=forbidden))
    changed=[k for k,v in plan['source_sha256'].items() if sha((FOUNDATION/k[11:]) if k.startswith('foundation/') else ROOT/k)!=v]
    if changed:issues.append(dict(reason='source_changed',files=changed))
    report=dict(schema='v17b_motion_six_reconstruction_independent_v1',passed=not issues,plan_sha256=PIN,core80_sha256=sha(corepath),source_sha256=plan['source_sha256'],reader_sha256=sha(__file__),
        rows=rows,issues=issues,selections=selections,pipeline_calls=dict(forbidden=forbidden,allowed_robot_only=dict(allowed)),private_robot_data=owned,
        scope=dict(prespecified_records=6,actual_records=len(rows),contexts=3,plant_constructions=0,physics_steps=0,forward_calls=0,contact_queries=0,sensor_resamples=0,control_writes=0,solver_replays=0),
        limitations=['Reuses authenticated production adapter; not an independent reimplementation of every geometry formula.','Exactly six prespecified records, not all CONTACT matrices; no replacements for unavailable strata.','Static compilation includes declared scene configuration but controller replay sees robot/public packets only; object truth/actor identity/current plant Data inaccessible.'])
    out.write_text(json.dumps(plain(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    out.with_suffix('.md').write_text(f"# V17b六个预定CONTACT运动矩阵重建\n\n{'PASS' if report['passed'] else 'FAIL'}；实际{len(rows)}/6。仅三个预定context的首/末accepted motion-available CONTACT，缺少不替换。真实日志M/target与公共机器人adapter重建逐字节比较。\n\n无原plant Data/积分/forward/接触重算/传感采样/控制写入。允许私有机器人Data的FK/Jac及tendon/transmission，调用计数见JSON。不是全域或全部CONTACT独立重建，不代替全80执行门。\n",encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],rows=len(rows),issues=issues[:6],out=str(out),sha256=sha(out)),ensure_ascii=False));return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
