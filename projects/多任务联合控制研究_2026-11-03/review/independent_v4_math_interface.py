"""Independent pure synthetic V4 action/priority audit. No robot imported."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import hashlib,json,sys
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import coordinator as c
EXPECTED={'coordinator.py':'516ae428fa0a85134b68d8ce593f27e3ffbdf0cef73e0940c2c13eb61126da40',
 'coordinator_model.py':'9f9267e853ea1815050ba4dec9c75d76b07a1292ef7c5ee81dca8973040e60ed',
 'coordinator_contacts.py':'fa2ceb90da15756ef385bb614bb4c962e498866094245a4dfeb99bf8621eced5'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    checks={};before={k:sha(ROOT/k) for k in EXPECTED}
    checks['reviewed_expected_source_version']=before==EXPECTED
    if not checks['reviewed_expected_source_version']:raise RuntimeError('Source revision changed; do not certify a different version')
    original=c.solve_qp;calls=[]
    def logged(Q,q,A,b,*a,**kw):
        r=original(Q,q,A,b,*a,**kw)
        calls.append(dict(Q=Q.copy(),q=q.copy(),A=A.copy(),b=b.copy(),x=r.x.copy(),residual=r.residual,status=r.status,initial=np.asarray(kw.get('feasible_initial',np.full(len(q),np.nan))).copy()))
        return r
    c.solve_qp=logged
    legacy=c.self_test();checks.update({'provided_synthetic/'+k:bool(v) for k,v in legacy['checks'].items()})
    cal=dict(force_matrix=np.eye(5)*10.,force_tau=.02,reference_alpha=np.zeros(5),response_error_bound=np.full(5,.02),config={'shape':'capsule'})
    ctl=c.JointCoordinator(cal,np.full(5,5.),18.,model_document=c.synthetic_document())
    p=np.full(5,2.);tau=.6*p;y=np.r_[p,tau,[0.,0.,1.2,0.,0.,0.]]
    state=SimpleNamespace(normal=p.copy(),alpha=np.zeros(5),closure_min=np.full(5,-.04),closure_max=np.full(5,.04),valid=np.ones(5,bool),closure_permission=np.ones(5,bool),contact_model_active=np.ones(5,bool),contact=np.ones(5,bool),confidence=np.ones(5),slip_magnitude=np.zeros(5),auxiliary_wrench=np.zeros(6),auxiliary_valid=True,load=1.2,load_uncertainty=.5,up=np.array([0.,0.,1.]),force_gain=np.full(5,10.),slip_speed=0.)
    ctl.state=state;ctl.model.record_input(7.5,np.zeros(8));ctl.model.beta=y.copy()
    ctl.model.snapshot=c.OutputForecast(y,np.zeros(16),np.full(16,.001),np.full(16,.01),np.full(16,50),np.ones(16),np.ones(16,bool),y.copy(),np.full(16,7.5))
    for k in range(50):ctl.contacts.update(SimpleNamespace(time=7.25+k*.005,normal=p,shear_magnitude=tau,contact_pos=np.zeros((5,3))),state,ctl.model.snapshot)
    target=np.array([-.001,.002,0.]);ctl.set_motion_request(target,np.zeros(3));start=len(calls);ctl._solve();stages=calls[start:]
    h=ctl.g.horizon;nr=8*h;plan=ctl.previous_plan.copy();checks['four_priority_stages_called']=len(stages)==4
    checks['final_plan_feasible_with_registered_solver_tolerance']=bool(np.max(stages[2]['A']@plan-stages[2]['b'])<=2e-5)
    idx=np.r_[np.arange(nr,nr+5),nr+5]
    def chosen(d):return d['x'] if d['residual']<=2e-5 else d['initial']
    x1,x2,x3=[chosen(d) for d in stages[:3]]
    checks['componentwise_pressure_priority_tolerance_enforced']=bool(np.max(plan[idx]-x1[idx])<=ctl.settings.priority_tolerance_N+2e-5)
    delta=plan-x3;loss=float(.5*delta@stages[2]['Q']@delta+delta@(stages[2]['Q']@x3+stages[2]['q']))
    checks['excess_force_tiebreak_does_not_spend_higher_level_loss']=loss<=ctl.settings.lexicographic_loss_tolerance+1e-11
    checks['reported_loss_change_is_actual_quadratic_change']=abs(loss-ctl.debug['higher_priority_quadratic_loss_change'])<1e-10
    P,offset,error=ctl.model.trajectory(state,h,ctl.g.decision_period);prediction=P@plan[:nr]+offset
    rows=np.ravel([np.r_[np.arange(16*j,16*j+10)] for j in range(h)])
    checks['all_horizon_actual_pressure_and_shear_nonnegative_without_clipping']=bool(prediction[rows].min()>=-1e-8)
    wanted_alpha=ctl.interval_action.copy();wanted_motion=ctl.interval_motion.copy();state.update=lambda f,dt:None
    ctl.last_decision=7.5;ctl.previous_time=7.495;executed=np.zeros(3);actions=[];motions=[]
    for k in range(4):
        ctl.set_motion_request(target,executed)
        result=ctl.step(SimpleNamespace(time=7.5+k*.005,phase='HOLDING'))
        actions.append(result.delta_closure.copy());state.alpha+=result.delta_closure
        motions.append(ctl.motion_reference-executed);executed=ctl.motion_reference.copy()
    checks['four_5ms_actions_sum_to_one_planned_20ms_alpha_increment']=bool(np.allclose(np.sum(actions,0),wanted_alpha,atol=1e-12,rtol=0))
    checks['four_5ms_motion_updates_sum_to_one_planned_20ms_increment']=bool(np.allclose(np.sum(motions,0),wanted_motion,atol=1e-12,rtol=0))
    checks['no_new_qp_inside_the_same_20ms_interval']=len(calls)==start+4
    checks['source_frozen_during_readonly_math_review']=all(sha(ROOT/k)==v for k,v in before.items())
    result=dict(schema='independent_v4_synthetic_action_priority_audit_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],source_sha256=before,
        contact_model_sha256=sha(ROOT/'calibration/public_contact_model_phase2_v1.json'),reviewer_sha256=sha(__file__),
        four_tick_plan=dict(alpha_increment=wanted_alpha.tolist(),executed_alpha=np.sum(actions,0).tolist(),motion_increment=wanted_motion.tolist(),executed_motion=executed.tolist()),
        actual_higher_level_quadratic_loss_change=loss,model_admitted=False,physics_steps_executed=0,
        limitations=['Pure synthetic algebra and interface only, no new performance or physical safety evidence.',
            'Shape-specific model is preselected from public calibration context, not identified from live skin in unknown-shape operation.',
            'Current-palm local kernels do not constitute explicit delayed-wrench coordinate transport or a global pose-invariant model.',
            'Compatibility tubes enlarge with uncertainty; they are empirical overlap tests, not robust Coulomb-cone feasibility certificates.',
            'Per-step measured input extents and previously visited return directions do not validate arbitrary cumulative closed-loop excursions.',
            'Only pressure guard has precedence; full wrench and pressure/shear corridors are compatibility penalties with explicit slacks, not hard physical constraints.'])
    out=ROOT/'review'/('independent_v4_math_interface_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json');out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=result['failed'],path=str(out),sha256=sha(out)),ensure_ascii=False))
if __name__=='__main__':main()
