"""Independent V7 wiring/finite-dispatch checks, without any plant."""
from __future__ import annotations
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace,MethodType
import argparse,hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from review.verify_v7_belief_independent import document,frame,SPECS

GAINS=dict(base=2.,ratio_floor=1.8,ratio_kp=.8,ratio_ki=.2,
           slip_boost_rate=3.,baseline_tracking_gain=.25,release_rate=.8)
DEVICE={**SPECS, 'slip_sigma_m_s':.0002,'effort_sigma':.02,'tactile_delay_s':.01}
CAL=dict(reference_alpha=[.5]*5,response_error_bound=[.02]*5,selected_radius=.04,
         force_matrix=np.eye(5)*10.,force_tau=.02,config=dict(shape='capsule'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public_frame(t,alpha=None,motion=None):
    f=frame(t);f.normal[:]=2.;f.alpha=np.full(5,.5) if alpha is None else np.array(alpha)
    f.closure_min=np.full(5,.46);f.closure_max=np.full(5,.54)
    f.closure_permission=np.ones(5,bool);f.nominal_force_weights=np.ones(5)
    f.age=np.full(5,t-f.sample_time);f.slip_magnitude=np.zeros(5)
    f.load_estimate=1.2;f.load_valid=True;f.support_axis=np.array([0.,0.,1.]);f.phase='HOLDING'
    f.executed_motion_reference=np.array([.05,0.,0.]) if motion is None else np.array(motion)
    f.motion_reference_rate_limits=np.array([.035,.12,.12]);f.sensor_characteristics=DEVICE.copy()
    f.skin_patches=[dict(digit=i,robot_body=100+i,link_point=[0.,0.,0.],link_normal=[0.,0.,1.],
                         normal=2.,sample_time=f.sample_time) for i in range(5)]
    return f


def task(t,motion,mode='hold',target=None):
    from task_contract import build_task_request
    from task_trajectory import constant_trajectory
    target=np.array([.05,0.,0.]) if target is None else np.array(target)
    trajectory=constant_trajectory(float(target[0]))
    # Point targets for the two wrists remain explicitly requested below. No
    # scenario-dependent environment or future disturbance is supplied.
    preview=trajectory.preview(t) if np.all(target[1:]==0.) else None
    return build_task_request(t,mode,target,motion,12. if mode not in ('release','monitor') else t+3.5,
        [5.]*5,18.,tracking_tolerances=[.01,.015,.015],motion_preview=preview,
        motion_reference_bounds=dict(lower=[0.,-.5,-.5],upper=[.1,.5,.5]),sensor_characteristics=DEVICE)


def run():
    from v7.controller import JointTaskController,AggregateGripReference,create_joint
    from v7.feedback import PlanResult
    sources={n:sha(ROOT/'v7'/n) for n in ('controller.py','feedback.py','belief.py')}
    checks=[];metrics={}
    def ck(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    def construct(doc=None):
        ctl=JointTaskController(CAL,[5.]*5,18.,GAINS,model_document=document() if doc is None else doc)
        for tick in range(81):ctl.state.update(public_frame(tick*.005),.005)
        return ctl
    ctl=construct();calls=[]
    commanded=np.array([-.001,.001,-.001,.001,0.,.0004,.001,-.001])
    def fake_solve(self,**kwargs):
        calls.append(kwargs)
        h=len(kwargs['user_targetsH3']);u=np.zeros((h,8));u[0]=commanded
        pred=(kwargs['pressure_free5H']+kwargs['pressure_map']@u.ravel()).reshape(h,5)
        if not kwargs['model_usable']:pred[:]=np.nan
        return PlanResult(commanded.copy(),u,pred,dict(aggregate_task_available=True,
            governed_projection_gap=[0.]*h,aggregate_reference_sigma_N=[.001]*h,
            prediction_available=kwargs['model_usable'],model_usable=kwargs['model_usable']))
    ctl.planner.solve=MethodType(fake_solve,ctl.planner)
    alpha=np.full(5,.5);motion=np.array([.05,0.,0.]);deltas=[]
    for tick in range(4):
        t=.4+.005*tick;f=public_frame(t,alpha,motion)
        ctl.set_task_request(task(t,motion));r=ctl.step(f,'HOLDING','hold')
        deltas.append(np.r_[r.delta_closure,ctl.motion_reference-motion]);alpha+=r.delta_closure;motion=ctl.motion_reference.copy()
    ck('planner_called_once_per_20ms_finite_interval',len(calls)==1,len(calls))
    ck('four_actual5ms_increments_equal_planned20ms_increment',np.max(abs(np.sum(deltas,axis=0)-commanded))<1e-12)
    ck('all_four_actual_quarters_exact',np.max(abs(np.asarray(deltas)-commanded/4.))<1e-12)
    ctl.state.update(public_frame(.42,alpha,motion),.005)
    actual=ctl.belief.actual_increment(.4,.42)
    ck('same_belief_actual8_history_matches_complete_executed_interval',np.max(abs(actual-commanded))<1e-12)
    ck('allocator_uses_the_single_belief_mean',np.max(abs(calls[0]['pressure_now5']-np.full(5,2.)))<1e-12)
    ck('factory_does_not_run_legacy_force_estimator','CoupledTactileState' not in type(ctl.state).__mro__[0].__name__ and type(ctl.belief).__name__=='UnifiedBelief')
    before=len(calls);ctl.set_task_request(task(.42,motion,'release'))
    rel=ctl.step(public_frame(.42,alpha,motion),'HOLDING','release')
    ck('release_immediately_preempts_active_hold_plan',len(calls)==before and rel.debug.get('release_authority') is True)
    ck('release_has_only_bounded_opening',np.all(rel.delta_closure<=0.) and np.max(abs(rel.delta_closure))<=GAINS['release_rate']*.005+1e-12)
    ck('release_does_not_fabricate_pressure_prediction',not np.isfinite(rel.predicted_normal).any())
    alpha+=rel.delta_closure
    for j,mode in enumerate(('monitor','hold')):
        t=.425+.005*j;f=public_frame(t,alpha,motion);f.normal[:]=6.
        ctl.set_task_request(task(t,motion,mode));out=ctl.step(f,'HOLDING','hold')
        ck('release_latch_blocks_regrasp_'+mode,len(calls)==before and np.all(out.delta_closure<=0.) and out.debug.get('release_authority') is True)
        alpha+=out.delta_closure
    # Actual planner, known public motion but unknown response: the explicit
    # finite nominal movement is passed through, not tied to an old force anchor.
    moving=construct();moving.set_task_request(task(.4,[.05,0.,0.],'transport',[.05,.01,-.01]))
    out=moving.step(public_frame(.4),'HOLDING','hold')
    change=moving.motion_reference-np.array([.05,0.,0.])
    ck('actual_controller_nonzero_user_motion_not_pinned_to_anchor',abs(change[1])>1e-7 and abs(change[2])>1e-7)
    ck('actual_joint_uses_shared_public_reference_rate_envelope',np.all(abs(change)<=np.array([.035,.12,.12])*.005+2e-10),change.tolist())
    ck('unknown_motion_allocation_not_a_valid_safety_forecast',not np.isfinite(out.predicted_normal).any() and not out.debug.get('prediction_available',False))
    ck('unknown_model_retains_single_force_posterior',out.debug.get('force_posterior_source','').startswith('UnifiedBelief.mean16'))
    response_doc=document();K=np.asarray(response_doc['shapes'][0]['coordinates']['current_palm']['response_to_20ms_interval_increment'])
    K[:,1,1]=10.*np.minimum(1.,np.arange(len(K))*.005/.02)
    response_doc['shapes'][0]['coordinates']['current_palm']['response_to_20ms_interval_increment']=K.tolist()
    late=construct(response_doc);late.planner.solve=MethodType(fake_solve,late.planner)
    late.set_task_request(task(.4,[.05,0.,0.]));first=late.step(public_frame(.4),'HOLDING','hold')
    a=np.full(5,.5)+first.delta_closure;m=late.motion_reference.copy()
    changed=public_frame(.405,a,m);changed.closure_permission[1]=False
    late.set_task_request(task(.405,m));blocked=late.step(changed,'HOLDING','hold')
    ck('late_permission_change_stops_forbidden_closing',blocked.delta_closure[1]<=0. and blocked.debug.get('late_permission_projection_max',0.)>0.)
    ck('changed_actual_ramp_does_not_publish_old_conditional_prediction',not blocked.debug.get('prediction_available',True) and not np.isfinite(blocked.predicted_normal).any())
    # Full admitted gain contract, independent of plant or any performance seed.
    actual=create_joint(CAL,[5.]*5,18.,dict(GAINS))
    ck('complete_admitted_ratio_contract_preserved',all(float(getattr(actual.g,k))==v for k,v in GAINS.items()))
    for label,bad in [('missing',{k:v for k,v in GAINS.items() if k!='base'}),('base_below_floor',{**GAINS,'base':1.2})]:
        try:create_joint(CAL,[5.]*5,18.,bad);rejected=False
        except ValueError:rejected=True
        ck('invalid_gain_contract_rejected_'+label,rejected)
    ref=AggregateGripReference(actual.g);b=SimpleNamespace(mean16=np.r_[np.full(5,2.),np.full(5,.5),np.zeros(6)],valid16=np.ones(16,bool))
    values=[]
    for j in range(61):
        v,_=ref.propose(j*.02,b,np.ones(5,bool),np.ones(5,bool),np.zeros(5),DEVICE,True);values.append(v)
    ck('constant_signal_learning_has_no_one_second_reference_jump',np.max(abs(np.diff(values)))<1e-12)
    metrics['reference_scalar_min_max']=[min(values),max(values)]
    altered=SimpleNamespace(mean16=b.mean16.copy(),valid16=b.valid16.copy());altered.mean16[9]=10.
    affiliation=np.ones(5,bool);affiliation[4]=False
    scalar,refdebug=ref.propose(1.22,altered,np.ones(5,bool),affiliation,np.zeros(5),DEVICE,True)
    ck('unaffiliated_contact_does_not_create_new_grip_ratio_request',scalar==GAINS['base'] and not refdebug['ratio_observation_complete'])
    oneway_doc=document();entry=oneway_doc['shapes'][0]
    entry['input_availability'][1]['plus']=False;entry['measured_input_delta_max'][1]=0.
    oneway=construct(oneway_doc);captured=[];real_solve=oneway.planner.solve
    def observed_solve(self,**kwargs):
        captured.append(kwargs);return real_solve(**kwargs)
    oneway.planner.solve=MethodType(observed_solve,oneway.planner)
    onealpha=np.full(5,.5);onemotion=np.array([.05,0.,0.])
    for k in range(9):
        t=.4+.005*k
        if k==1:onealpha[1]-=.001
        oneway.set_task_request(task(t,onemotion,'transport' if k==8 else 'hold',[.05,.01,0.] if k==8 else None))
        oneout=oneway.step(public_frame(t,onealpha,onemotion),'HOLDING','hold')
        onealpha+=oneout.delta_closure;onemotion=oneway.motion_reference.copy()
    trusted=[x for x in captured if x['model_usable']]
    empirical=[x for x in captured if not x['model_usable']]
    ck('trusted_future_cannot_use_unmeasured_positive_return',bool(trusted) and all(x['step_upper8'][1]<=0. for x in trusted))
    ck('empirical_return_stays_below_visited_anchor_and_is_not_forecast',bool(empirical) and empirical[-1]['step_upper8'][1]>0. and empirical[-1]['alpha_upper5'][1]<=.5+1e-12 and not np.isfinite(oneout.predicted_normal).any())
    ck('tested_production_sources_unchanged',sources=={n:sha(ROOT/'v7'/n) for n in sources})
    return dict(schema='independent_v7_controller_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        passed=all(c['passed'] for c in checks),checks=checks,metrics=metrics,source_sha256=sources,checker_sha256=sha(__file__),
        scope='pure synthetic public frames and source/controller interfaces; no plant or physical execution; fake planner used only for dispatch oracle, actual planner used separately for motion and unknown-mode wiring',
        limitations=['public shape context is permitted; no shape-blind claim','native release is outside held pressure model','no physical, timing-budget, performance or global-model admission'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);args=ap.parse_args();r=run()
    if args.output:
        if args.output.exists():raise FileExistsError(args.output)
        args.output.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=r['passed'],checks=len(r['checks']),failures=[c for c in r['checks'] if not c['passed']],output=str(args.output) if args.output else None),ensure_ascii=False))
    return 0 if r['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
