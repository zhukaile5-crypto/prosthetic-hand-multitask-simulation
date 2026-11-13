"""Rebuild exact archived fusion holding QPs from PUBLIC packets and evidence.

No controller replay/inverse/FK/forward/contact/dynamics is performed. A public
nominal model is compiled solely to read constant actuator force unit scales.
Past acquisition-time measurements reconstruct the original public state.mu.
Original targets, selected patch IDs, A/J/p0 and effort bounds are archived.
"""
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import gzip,json,hashlib,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
FOUNDATION=ROOT.parent/'五指策略接入_2026-11-03'
sys.path.insert(0,str(FOUNDATION))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def clean(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    return v
def main():
    out=ROOT/'review/v14_archived_fusion225_author_v1.json'
    fixture=ROOT/'review/v14_archived_fusion225_queries_v1.json'
    if out.exists() or fixture.exists():raise FileExistsError('Fresh immutable numerical replay evidence required')
    folder=ROOT/'output/development_v13_joint_pilot_v1/runs/nominal__fusion_effort__617901'
    files=[folder/n for n in ('trace.npz','metadata.json','joint_decisions.jsonl.gz','joint_decision_manifest.json')]
    before={str(p):sha(p) for p in files};meta=read(folder/'metadata.json')
    with np.load(folder/'trace.npz',allow_pickle=False) as z:
        tick=z['time'].copy();observation=z['observation_json'].copy();debug=z['debug_json'].copy()
    with gzip.open(folder/'joint_decisions.jsonl.gz','rt',encoding='utf-8') as stream:decisions=[json.loads(x) for x in stream]
    decisions={int(round(x['time_s']/.005)):x for x in decisions if x['mode']=='holding'}
    import mujoco
    forbidden_calls=[]
    for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_inverse','mj_fwdPosition','mj_fwdVelocity','mj_fwdActuation','mj_kinematics','mj_comPos','mj_collision','mj_contactForce'):
        def forbidden(*args,_name=name,**kwargs):forbidden_calls.append(_name);raise AssertionError('Pure archived QP audit forbids '+_name)
        setattr(mujoco,name,forbidden)
    from hand_plant import _model
    from foundation_config import FoundationConfig
    from vendor.frozen_controllers import make_gains
    from v12.observation import PublicTactileState
    from v10.robot_contact_space import PublicSkinPatch
    from v13_joint.intent_qp import _box_normal_cone,_normal_intent_cone
    from v14_coord.intent_qp import solve_intent_qp
    from review.verify_v13_joint_intent_qp_independent_v1 import objective,gradient
    compiled=_model(FoundationConfig(shape=meta['shape']))
    model=compiled[0] if isinstance(compiled,tuple) else compiled
    ids=np.asarray([model.actuator(n).id for n in meta['robot_actuator_telemetry']['axes21']],int)
    force_scales=np.max(abs(model.actuator_forcerange[ids]),axis=1)
    gains=make_gains(meta['actual_ratio_task_gains']);state=PublicTactileState(gains)
    checks=[];records=[];queries=[];old_status=Counter();new_status=Counter()
    def check(name,passed,detail=None):checks.append(dict(name=name,passed=bool(passed),detail=clean(detail)))
    for index,t in enumerate(tick):
        if t<7.3-1e-9 or t>=12.-1e-9:continue
        frame=SimpleNamespace(**json.loads(str(observation[index])));state.update(frame,.005)
        if index not in decisions:continue
        e=decisions[index];d=e['domain'];q=e['qp'];dbg=json.loads(str(debug[index]));P=len(e['observed_patch_normal'])
        if q is None:raise ValueError('The predeclared fusion holding cohort must carry its original QP')
        inliers=set(e['force_intent_debug']['object_group_inliers']);patches=[PublicSkinPatch.from_public(frame.skin_patches[k]) for k in e['patch_indices']]
        digits=np.asarray([p.digit for p in patches],int)
        c=np.asarray([(float(p.reference_normal[2])+state.mu[p.digit]*gains.friction_margin*np.sqrt(max(0.,1.-float(p.reference_normal[2])**2))) if k in inliers else 0. for k,p in zip(e['patch_indices'],patches)])
        A=np.asarray(d['transmission21x25']);sr=1./(abs(A.T)@force_scales);sd=1./force_scales
        query=dict(p0_21=np.asarray(d['p0_effort21']),A21x25=A,Jpx25=np.asarray(e['Jpx25']),y_p=np.asarray(e['observed_patch_normal']),
            z_intent_p=np.asarray(q['diagnostics']['original_z_intent_p']),effort_lower21=np.asarray(d['effort_lower21']),effort_upper21=np.asarray(d['effort_upper21']),
            patch_digit_p=digits,caps5=np.maximum(0.,np.asarray(meta['limits'])-gains.force_margin),total_budget=max(0.,meta['budget']-2*gains.force_margin),
            support_coeff_p=c,load_proxy_N=float(dbg['support_demand_N']),scales=dict(residual25=sr,intentP=1./np.maximum(1.,np.asarray(meta['limits'])[digits]),delta21=sd,load=1.))
        xold=np.r_[q['d21'],q['zP']];g=gradient(xold,query);oldvalue=objective(xold,query)
        groups=tuple(np.flatnonzero(digits==i) for i in range(5));box=_box_normal_cone(g[:21],xold[:21],query['effort_lower21']-query['p0_21'],query['effort_upper21']-query['p0_21']);normal=_normal_intent_cone(g[21:],xold[21:],groups,query['caps5'],query['total_budget'])
        objective_delta=oldvalue-q['objective'];d_delta=box['stationarity_inf']-q['diagnostics']['direct_KKT_inf_d21'];z_delta=normal['stationarity_inf']-q['diagnostics']['direct_KKT_inf_zP']
        check(f'{index}_old_objective_same',abs(objective_delta)<=1e-12,objective_delta)
        check(f'{index}_old_direct_d_same',abs(d_delta)<=1e-12,d_delta);check(f'{index}_old_direct_z_same',abs(z_delta)<=1e-12,z_delta)
        check(f'{index}_original_support_demand',abs(query['load_proxy_N']-(state.load+gains.load_uncertainty_weight*state.load_uncertainty+gains.support_buffer+gains.slip_load_gain*max(0.,state.slip_speed-.004)))<1e-14)
        if not all(x['passed'] for x in checks[-4:]):raise ValueError('Cannot claim exact archived query reconstruction; see per-packet differences')
        begin=time.perf_counter_ns();r=solve_intent_qp(**query);wall=(time.perf_counter_ns()-begin)/1e6
        new_status[r.status]+=1;old_status[q['status']]+=1
        if r.d21 is not None:
            xo=np.r_[r.d21,r.zP];newvalue=objective(xo,query);gg=gradient(xo,query)
            check(f'{index}_new_original_objective',abs(newvalue-r.objective)<1e-12,newvalue-r.objective)
            check(f'{index}_new_original_residual',np.array_equal(r.residual25,A.T@r.d21-query['Jpx25'].T@(r.zP-query['y_p'])))
            check(f'{index}_no_cost_increase_over_old_candidate',newvalue<=oldvalue+1e-12,newvalue-oldvalue)
        rec=dict(decision_id=e['decision_id'],time_s=t,patches=P,old_status=q['status'],new_status=r.status,new_reason=r.reason,old_objective=oldvalue,new_objective=r.objective,
            old_KKT_d=q['diagnostics']['direct_KKT_inf_d21'],old_KKT_z=q['diagnostics']['direct_KKT_inf_zP'],new_KKT_d=r.diagnostics.get('direct_KKT_inf_d21'),new_KKT_z=r.diagnostics.get('direct_KKT_inf_zP'),
            iterations=r.iterations,wall_ms=wall,expanded_objective=r.diagnostics.get('expanded_least_squares_objective'),
            original_L=r.diagnostics.get('lipschitz_upper_bound'),coordinate_scale=r.diagnostics.get('coordinate_scale'),
            d21=clean(r.d21),zP=clean(r.zP),diagnostics=clean(r.diagnostics))
        records.append(rec);queries.append(dict(decision_id=e['decision_id'],time_s=t,kwargs=clean(query),reconstruction_original_objective_delta=objective_delta,reconstruction_original_dKKT_delta=d_delta,reconstruction_original_zKKT_delta=z_delta))
    check('all225_predeclared_holding_queries',len(records)==225)
    check('original123_nonconvergence_retained',old_status['nonconvergence']==123)
    check('no_forbidden_pipeline_calls',not forbidden_calls,forbidden_calls)
    for p,h in before.items():check('unchanged_'+p,sha(p)==h)
    closure={str(p):sha(p) for p in (Path(__file__),ROOT/'v14_coord/intent_qp.py',ROOT/'v13_joint/intent_qp.py',ROOT/'v12/observation.py',ROOT/'v10/robot_contact_space.py',FOUNDATION/'hand_plant.py',FOUNDATION/'foundation_config.py',FOUNDATION/'vendor/frozen_controllers.py')}
    fixture.write_text(json.dumps(dict(schema='v14_exact_archived_public_qp_queries_v1',source_sha256=before,derivation_source_sha256=closure,queries=queries),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    timings=np.asarray([r['wall_ms'] for r in records]);iterations=np.asarray([r['iterations'] for r in records])
    result=dict(schema='v14_archived_qp_author_replay_v1',passed=all(x['passed'] for x in checks),checks=len(checks),failures=[x for x in checks if not x['passed']],
        old_status=dict(old_status),new_status=dict(new_status),count=len(records),all_new_converged=new_status['converged']==225,
        time_ms=dict(median=float(np.median(timings)),p95=float(np.quantile(timings,.95)),max=float(max(timings))),iterations=dict(median=float(np.median(iterations)),max=int(max(iterations))),
        source_sha256=before,derivation_source_sha256=closure,query_fixture=str(fixture),query_fixture_sha256=sha(fixture),records=records,details=checks,
        scope='225 holding QPs from one seen V13 nominal/fusion/617901 trajectory; exact public-state reconstruction, not new closed-loop pressure results; no physics/forward/FK/native inverse',
        public_static_model='nominal declared shape compiled solely to read known actuator effort unit scales; no runtime object pose/mass/friction read',forbidden_pipeline_calls=forbidden_calls)
    out.write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('passed','checks','old_status','new_status','time_ms','iterations','all_new_converged')},ensure_ascii=False))
    print(out)
    return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
