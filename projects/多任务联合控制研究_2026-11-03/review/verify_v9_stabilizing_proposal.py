"""Independent V9 proposal projection and real public-frame baseline replay."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime,timezone
from types import MethodType
from dataclasses import fields
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from review.verify_v7_reachable_task import fixture,freeze_except,clean

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run():
    from v9.feedback import ReachableTaskPlanner,PlanResult
    from v9.controller import JointTaskController,REFERENCE_MEMORY
    from controller_adapter import create_controller
    from contracts import SensorFrame
    sources={n:sha(ROOT/n) for n in ('v9/controller.py','v9/feedback.py','v9/runtime.py')}
    checks=[];outputs={};planner=ReachableTaskPlanner()
    def ck(name,value,detail=None):checks.append(dict(name=name,passed=bool(value),detail=detail))
    def solve(name,f):
        f['pressure_covariance5x5']=np.zeros((5,5));f['model_usable']=False
        r=planner.solve(**deepcopy(f));a=r.first_action8;plan=r.plan
        ck(name+'/finite_actual_authority',np.isfinite(plan).all() and np.all(plan>=f['step_lower8']-2e-7) and np.all(plan<=f['step_upper8']+2e-7))
        alpha=f['alpha5']+np.cumsum(plan[:,:5],0)
        ck(name+'/cumulative_actual_domain',np.all(alpha>=f['alpha_lower5']-2e-7) and np.all(alpha<=f['alpha_upper5']+2e-7))
        ck(name+'/no_physical_forecast',not np.isfinite(r.predicted_pressure).any() and r.debug['prediction_available'] is False)
        outputs[name]=dict(action=a.tolist(),debug=r.debug);return r
    f=fixture();f['pressure_map'][:5,:5]=np.eye(5)*10.;f['force_total_target']=9.
    preferred=np.array([[.003,-.002,.001,0.,-.001,0.,0.,0.]]);f['preferred_planH8']=preferred
    accepted=solve('healthy_nonzero_proposal',f)
    ck('feasible_stabilizing_proposal_passes_bit_exact',np.array_equal(accepted.plan,preferred) and accepted.debug['nominal_proposal_accepted_exact'])
    f=freeze_except(fixture(),{1});f['pressure_now5'][0]=4.99;f['observed_pressure5'][0]=4.99;f['pressure_free5H'][0]=4.99;f['pressure_map'][0,1]=10.;f['limits5'][0]=5.;f['force_total_target']=9.;f['preferred_planH8']=np.array([[0.,.02,0.,0.,0.,0.,0.,0.]])
    unsafe=solve('unsafe_proposal_projection',f)
    ck('unsafe_proposal_cannot_be_accepted_exact',not unsafe.debug['nominal_proposal_accepted_exact'] and (f['pressure_free5H']+f['pressure_map']@unsafe.first_action8)[0]<=5.00001)
    f=freeze_except(fixture(),set());f['force_total_target']=9.;f['preferred_planH8']=np.array([[0.,.01,0.,0.,0.,0.,0.,0.]])
    fixed=solve('fixed_coordinate_illegal_proposal',f)
    ck('fixed_coordinate_is_not_falsely_passed',not fixed.debug['nominal_proposal_accepted_exact'] and np.array_equal(fixed.first_action8,np.zeros(8)))
    f['step_lower8'][1]=f['step_upper8'][1]=.005;f['preferred_planH8'][0,1]=.005
    offset=solve('actual_fixed_nonzero_coordinate',f)
    ck('legal_nonzero_fixed_coordinate_preserved_exact',offset.debug['nominal_proposal_accepted_exact'] and np.array_equal(offset.plan,f['preferred_planH8']))
    f=freeze_except(fixture(),{0,1});f['alpha5'][0]=1.;f['step_upper8'][0]=0.;f['force_total_target']=9.;f['pressure_map'][:2,:2]=[[1.,1.],[1.,2.]];f['preferred_planH8']=np.array([[.01,.01,0.,0.,0.,0.,0.,0.]])
    sat=solve('saturated_TH_active_set_projection',f)
    ck('saturation_modified_only_where_necessary',not sat.debug['nominal_proposal_accepted_exact'] and sat.first_action8[0]<=2e-7 and abs(sat.first_action8[1]-.01)<2e-6)
    f=freeze_except(fixture(),{1});f['step_upper8'][1]=0.;f['force_total_target']=9.;f['preferred_planH8']=np.array([[0.,.01,0.,0.,0.,0.,0.,0.]])
    one=solve('one_sided_proposal',f)
    ck('unavailable_positive_direction_overrides_proposal',one.first_action8[1]<=2e-7 and not one.debug['nominal_proposal_accepted_exact'])
    f=freeze_except(fixture(),{1});f['force_total_target']=10.1;f['pressure_valid5'][4]=False;f['observed_pressure5'][4]=np.nan;f['preferred_planH8']=np.zeros((1,8));f['pressure_map'][0,1]=10.
    missing=solve('missing_group_does_not_invent_reserve',f)
    ck('missing_still_preserves_original_group',np.array_equal(missing.first_action8,np.zeros(8)) and not missing.debug['aggregate_task_available'] and missing.debug['grip_support_mask']==[True]*5)
    f=freeze_except(fixture(),{1});f['force_total_target']=9.;f['pressure_map'][0,1]=10.;f['observed_pressure5'][0]=6.;f['limits5'][0]=5.;f['preferred_planH8']=np.array([[0.,.02,0.,0.,0.,0.,0.,0.]])
    risk=solve('raw_material_risk_overrides_stabilizer',f)
    ck('material_risk_priority_can_reverse_unsafe_proposal',risk.first_action8[1]<-.0399 and risk.debug['material_slack_N'][0]>=.5999)
    f=freeze_except(fixture(),{1});f['pressure_map'][0,1]=10.;f['force_total_target']=10.1;f['preferred_planH8']=np.zeros((1,8))
    reserve=solve('reserve_deficit_overrides_zero_proposal',f)
    ck('minimum_reserve_not_ignored_to_preserve_proposal',abs(reserve.first_action8[1]-.01)<1e-6 and not reserve.debug['nominal_proposal_accepted_exact'])
    # Actual recorded public SensorFrames, not a mock of baseline estimation.
    run=ROOT/'output/development_v8_counterexamples_v1/runs/nominal__ratio__610212'
    meta=json.loads((run/'metadata.json').read_text(encoding='utf-8'))
    with np.load(run/'trace.npz',allow_pickle=False) as z:t=z['time'];obs=z['observation_json'];tasks=z['task_request_json']
    names={f.name for f in fields(SensorFrame)}
    def decode(i):
        d=json.loads(str(obs[i]));core={k:v for k,v in d.items() if k in names}
        for k,v in core.items():
            if isinstance(v,list) and np.asarray(v).dtype.kind in 'biuf':core[k]=np.array(v)
        frame=SensorFrame(**core)
        for k,v in d.items():
            if k not in names:
                if isinstance(v,list):
                    a=np.asarray(v)
                    if a.dtype.kind in 'biuf':v=a
                setattr(frame,k,v)
        frame.dt=.005;return frame
    cal=meta['calibration'];gains=meta['gains'];limits=meta['limits'];budget=meta['budget']
    baseline=create_controller('ratio',cal,limits,budget,deepcopy(gains));ctl=JointTaskController(cal,limits,budget,deepcopy(gains))
    warm=np.flatnonzero((t>=7.3-1e-9)&(t<7.5-1e-9));holding=np.flatnonzero((t>=7.5-1e-9)&(t<12.-1e-9))
    for i in warm:
        f=decode(i);baseline.state.update(deepcopy(f),.005);ctl.state.update(deepcopy(f),.005)
    different=[];state_diff=[];healthy=0;modified=0
    for i in holding:
        f=decode(i);wanted=json.loads(str(tasks[i]));a=baseline.step(deepcopy(f),'HOLDING','hold');ctl.set_task_request(wanted);b=ctl.step(deepcopy(f),'HOLDING','hold')
        if not np.array_equal(a.delta_closure,b.delta_closure):different.append(dict(time=float(t[i]),max_abs_delta=float(np.max(abs(a.delta_closure-b.delta_closure))),modified=not b.debug.get('nominal_proposal_accepted_exact',False)))
        for attr in ('normal','valid','alpha','latent_alpha','force_uncertainty','contact_model_active'):
            if not np.array_equal(np.asarray(getattr(baseline.state,attr)),np.asarray(getattr(ctl.state.force_state,attr)),equal_nan=True):state_diff.append(dict(time=float(t[i]),field=attr))
        if b.debug.get('nominal_proposal_accepted_exact',False):
            healthy+=1
            ck('healthy_delta_exact_at_'+str(float(t[i])),np.array_equal(a.delta_closure,b.delta_closure))
        else:modified+=1
        if np.isfinite(b.predicted_normal).any():raise AssertionError('V9 published a physical force forecast')
    ck('one_authoritative_exact_baseline_force_state',ctl.nominal.state is ctl.state and type(ctl.state.force_state).__name__=='CoupledTactileState' and not hasattr(ctl,'belief') and not state_diff,state_diff[:10])
    ck('actual_nominal_record_has_healthy_exact_passes',healthy>0,dict(healthy_ticks=healthy,intervened_ticks=modified,delta_differences=different[:10]))
    # Intervention rollback is exercised on actual saved observation state.
    override=JointTaskController(cal,limits,budget,deepcopy(gains))
    for i in warm:override.state.update(decode(i),.005)
    original=(override.nominal.integral,override.nominal.boost)
    def proposal(self,frame,phase):self.integral=.55;self.boost=.66;return np.zeros(5)
    override.nominal._decide=MethodType(proposal,override.nominal)
    def projected(self,**kwargs):
        h=len(kwargs['user_targetsH3']);return PlanResult(np.zeros(8),np.zeros((h,8)),np.full((h,5),np.nan),dict(nominal_proposal_accepted_exact=False,prediction_available=False))
    override.planner.solve=MethodType(projected,override.planner)
    i=int(holding[0]);override.set_task_request(json.loads(str(tasks[i])));out=override.step(decode(i),'HOLDING','hold')
    ck('rejected_proposal_rolls_back_integral_and_boost',(override.nominal.integral,override.nominal.boost)==original and out.debug['reference_memory_rollback'])
    for j,mode in enumerate(('release','monitor','hold')):
        i=int(holding[1+j]);req=json.loads(str(tasks[i]));req['mode']=mode;override.set_task_request(req)
        out=override.step(decode(i),'HOLDING','release' if j==0 else 'hold')
        ck('release_authority_latched_'+mode,out.debug.get('release_authority') is True and np.all(out.delta_closure<=0.) and not np.isfinite(out.predicted_normal).any())
    ck('tested_core_source_identity_stable',sources=={n:sha(ROOT/n) for n in sources})
    return dict(schema='independent_v9_stabilizing_proposal_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=all(x['passed'] for x in checks),checks=checks,outputs=outputs,
        source_sha256=sources,checker_sha256=sha(__file__),public_replay=dict(path=str(run/'trace.npz'),sha256=sha(run/'trace.npz'),warmup_samples=len(warm),holding_samples=len(holding),healthy_exact_ticks=healthy,intervened_ticks=modified,delta_differences=different,force_state_differences=state_diff),
        physics_executed=False,plant_constructed=False,scope='9 analytic proposal projection fixtures plus full real nominal public observation replay7.3-12s, unchanged baseline force state, rollback and release. Runtime module imports its follower but no session/plant is constructed.',
        limits=['Software correctness/conditional equivalence only, not physical closed-loop superiority','Supervisor allocation matrix is uncertified empirical response, not posterior covariance','One saved developer stream; no new physics or fresh holdout'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);args=ap.parse_args();r=run()
    if args.output:
        if args.output.exists():raise FileExistsError(args.output)
        args.output.write_text(json.dumps(clean(r),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=r['passed'],checks=len(r['checks']),failed=[x for x in r['checks'] if not x['passed']],public_replay={k:v for k,v in r['public_replay'].items() if k not in ('delta_differences','force_state_differences')},output=str(args.output) if args.output else None),ensure_ascii=False))
    return 0 if r['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
