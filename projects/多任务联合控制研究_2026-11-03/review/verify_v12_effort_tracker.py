"""Pure/current-state algebra QA. Compile only; no dynamic or contact calls."""
from pathlib import Path
from copy import copy
from dataclasses import replace
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(1,str(ROOT.parent/'五指策略接入_2026-11-03'));sys.dont_write_bytecode=True
import mujoco
from foundation_config import FoundationConfig
from hand_plant import _model
from v12.effort_tracker import EffortTracker

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify():
    model,_=_model(FoundationConfig());tracker=EffortTracker(model);servo=tracker.servo
    q=np.mean(model.jnt_range[servo.jids],axis=1);q[-1]=.1
    q[servo.axes25.index('rh_WRJ2')]=0.;q[servo.axes25.index('rh_WRJ1')]=0.;v=np.zeros(25)
    ref=np.array([.8,1.,1.,1.,.9,.1,0.,0.]);u=tracker._port(ref)
    lower=np.r_[np.full(5,-.02),-.002,-.02,-.02];upper=-lower
    bounds={'lower':tracker.basis.held_lower8[5:].tolist(),'upper':tracker.basis.held_upper8[5:].tolist()}
    defaults=dict(q25=q,v25=v,current_reference8=ref,actual_target21=u,desired_torque_correction25=np.zeros(25),
        nominal_motion_reference3=ref[5:],step_lower8=lower,step_upper8=upper,motion_reference_bounds=bounds,tracking_tolerances3=[.01,.03,.03])
    blocked={name:getattr(mujoco,name) for name in ['mj_step','mj_step1','mj_step2','mj_Euler','mj_RungeKutta','mj_implicit','mj_forward','mj_fwdPosition','mj_fwdVelocity','mj_fwdActuation','mj_kinematics','mj_comPos','mj_collision','mj_contactForce'] if hasattr(mujoco,name)}
    calls={name:0 for name in blocked}
    def block(name):
        def forbidden(*a,**kw):calls[name]+=1;raise AssertionError('No dynamics/FK/contact:'+name)
        return forbidden
    for name in blocked:setattr(mujoco,name,block(name))
    checks=[];examples={}
    def check(name,flag):
        if not flag:raise AssertionError(name)
        checks.append(name)
    def solve(t=tracker,**changes):return t.solve(**dict(defaults,**changes))
    try:
        same=solve();examples['zero']=same.to_dict()
        check('zero correction exact current passthrough',same.available and np.array_equal(same.increment8,np.zeros(8)) and np.array_equal(same.port_target21,u))
        nominal=ref[5:]+np.array([.0003,.002,-.001]);m=solve(nominal_motion_reference3=nominal)
        check('legal nominal motion bitexact no correction',m.available and m.debug['exact_zero_correction_nominal_passthrough'] and np.array_equal(m.increment8,np.r_[np.zeros(5),nominal-ref[5:]]))
        check('nominal exact real mechanical decoder',np.array_equal(m.port_target21,tracker.basis.decode_held8(np.r_[ref[:5],nominal]).actuator_targets21))
        check('C21x8 true tendon reference not desiredq',tracker.C.shape==(21,8) and tracker.C[7,1]==2. and tracker.C[6,1]==.6)
        base=servo.evaluate(q,v,u);wanted=ref+np.array([.003,-.004,.002,.005,-.002,.0004,.003,-.002])
        target=servo.evaluate(q,v,tracker._port(wanted));correction=target.generalized_actuator_effort25-base.generalized_actuator_effort25
        attainable=solve(desired_torque_correction25=correction);examples['attainable']=attainable.to_dict()
        check('known achievable25torque correction matched',attainable.available and np.max(abs(attainable.residual25))<1e-10)
        check('unique unsaturated reference recovered',np.max(abs(attainable.reference8-wanted))<1e-10)
        check('native damping in baseline and final',attainable.debug['native_damping_preserved'])
        impossible=np.zeros(25);impossible[servo.axes25.index('rh_THJ4')]=.1
        missing=solve(desired_torque_correction25=impossible);examples['uncontrolled_residual']=missing.to_dict()
        check('unactuated holding axis residual kept not retargeted',missing.available and abs(missing.residual25[servo.axes25.index('rh_THJ4')]+.1)<1e-12 and np.array_equal(missing.desired_torque25,base.generalized_actuator_effort25+impossible))
        strong=np.zeros(25);strong[24]=1000.
        limited=solve(desired_torque_correction25=strong);examples['bounded_residual']=limited.to_dict()
        check('unreachable effort retains fixed target residual',limited.available and limited.residual25[24]<-900 and np.array_equal(limited.desired_torque25,base.generalized_actuator_effort25+strong))
        check('true step and port bounds final checked',limited.debug['true_step_reference_bounds_checked'] and limited.debug['true_actuator_port_bounds_checked'])
        # Same state stateless servo damping is essential here. At100N force
        # kink, a negative correction must enter the unsaturated branch.
        kink_v=v.copy();length=base.actuator_length21[18];kp=model.actuator_gainprm[servo.aids[18],0];kv=-model.actuator_biasprm[servo.aids[18],2]
        cap=float(servo.force_range[18,1]);kink_v[24]=(kp*(u[18]-length)-cap)/kv
        kink=servo.evaluate(q,kink_v,u);corr=np.zeros(25);corr[24]=-1.
        release=solve(v25=kink_v,desired_torque_correction25=corr);examples['force_kink']=release.to_dict()
        check('native kink Jacobian is genuinely unavailable',kink.reference_jacobian25x21 is None)
        check('known clipping kink not fake zero response',release.available and abs(release.residual25[24])<1e-10 and release.increment8[5]<0)
        saturated_v=kink_v.copy();saturated_v[24]-=10.
        flat=solve(v25=saturated_v,desired_torque_correction25=corr);examples['strict_saturated_plateau']=flat.to_dict()
        check('strict saturated plateau chooses minimum command without false match',flat.available and np.array_equal(flat.increment8,np.zeros(8)) and abs(flat.residual25[24]-1.)<1e-12)
        moving=solve(v25=np.r_[v[:24],.01]);native_moving=servo.evaluate(q,np.r_[v[:24],.01],u)
        check('nonzero actual velocity nativeD preserved',moving.available and np.array_equal(moving.desired_torque25,native_moving.generalized_actuator_effort25))
        outside=solve(nominal_motion_reference3=ref[5:]+np.array([.02,0,0]),tracking_tolerances3=[.03,None,None])
        nominal_outside=servo.evaluate(q,v,tracker._port(np.r_[ref[:5],ref[5:]+[.02,0,0]]))
        check('infeasible nominal goal not projected before matching',outside.available and np.array_equal(outside.desired_torque25,nominal_outside.generalized_actuator_effort25) and outside.reference8[5]<=ref[5]+upper[5]+1e-12)
        infeasible=solve(nominal_motion_reference3=ref[5:]+np.array([.02,0,0]),tracking_tolerances3=[0.,None,None]);examples['empty_corridor']=infeasible.to_dict()
        check('empty task-reference corridor returns explicit infeasible',not infeasible.available and infeasible.status=='INFEASIBLE_REFERENCE_SET' and infeasible.desired_torque25 is not None)
        wrong=u.copy();wrong[18]=ref[5]
        check('actual engineport missing liftbias rejected',solve(actual_target21=wrong).status=='UNAVAILABLE_REFERENCE_CONTRACT')
        check('missing velocity not zero-filled',solve(v25=None).status=='UNAVAILABLE_INPUT')
        check('full objectq33 payload rejected',solve(q25=np.zeros(33)).status=='UNAVAILABLE_INPUT')
        check('nonfinite limits unavailable',solve(step_upper8=np.full(8,np.inf)).status=='UNAVAILABLE_INPUT')
        unknown=copy(model);unknown.actuator_dyntype[servo.aids[0]]=mujoco.mjtDyn.mjDYN_FILTER
        ut=EffortTracker(unknown)
        check('unprovided activation mode unavailable not zero gain',solve(t=ut).status=='UNAVAILABLE_MODEL')
        coupled=EffortTracker(model);original=coupled.servo.evaluate
        def artificial_coupled(*a,**kw):
            s=original(*a,**kw)
            if s.transmission21x25 is None:return s
            mat=s.transmission21x25.copy();mat[6,servo.axes25.index('rh_MFJ3')]=1.
            return replace(s,transmission21x25=mat)
        coupled.servo.evaluate=artificial_coupled
        check('coupled transmission refuses false scalar decomposition',solve(t=coupled).status=='UNAVAILABLE_COUPLED_SUPPORT')
        # Independent dense-grid comparison on a synthetic narrow-clamp copy;
        # no new gain fitted and no real pressure or dynamic response claimed.
        clipped_model=copy(model);clipped_model.actuator_forcerange[servo.aids[:18]]=[-.02,.02]
        ct=EffortTracker(clipped_model);cstate=ct.servo.evaluate(q,v,u)
        rng=np.random.default_rng(121205);oracle_max_violation=0.;branches=0
        for case in range(12):
            corr=rng.normal(size=25)*.015;corr[24]*=100
            result=solve(t=ct,desired_torque_correction25=corr)
            check('synthetic clamped case'+str(case)+' available',result.available)
            A=cstate.transmission21x25;gain=clipped_model.actuator_gainprm[ct.servo.aids,0];bias=clipped_model.actuator_biasprm[ct.servo.aids]
            beta=bias[:,0]+bias[:,1]*cstate.actuator_length21+bias[:,2]*cstate.actuator_velocity21
            p0=cstate.predicted_effort21
            for axis in range(8):
                rows=ct.rows[axis];support=np.flatnonzero(np.any(abs(A[rows])>1e-13,axis=0));grid=np.linspace(result.debug['candidate_lower_reference8'][axis],result.debug['candidate_upper_reference8'][axis],2001)
                ports=ct.port_offset[rows,None]+ct.C[rows,axis,None]*grid
                raw=gain[rows,None]*ports+beta[rows,None]
                pf=np.clip(raw,ct.servo.force_range[rows,0,None],ct.servo.force_range[rows,1,None])
                residual=A[rows][:,support].T@(pf-p0[rows,None])-corr[support,None]
                grid_min=float(np.min(np.sum(residual*residual,axis=0)));actual=float(result.residual25[support]@result.residual25[support])
                oracle_max_violation=max(oracle_max_violation,actual-grid_min);branches+=1
        check('96scalar piecewise minima no worse than independent2001point grids',oracle_max_violation<1e-10)
        check('final native scalar losses all rechecked',all(x['absolute_difference']<=max(1e-24,x['roundoff_tolerance']) for x in attainable.debug['scalar_native_recheck']))
        check('no future pressure or contactforcegradient certificate',not attainable.debug['future_q_or_pressure_predicted'] and not attainable.debug['contact_force_gradient_available'] and not attainable.debug['physical_safety_certified'])
        check('known robot internal only no empirical B',not attainable.debug['empirical_pressure_B_used'])
        check('no forbidden simulation FK contact calls',not any(calls.values()))
        return dict(passed=True,checks=checks,numerical_oracle=dict(cases=12,scalar_problems=branches,grid_points_each=2001,max_solver_loss_minus_grid_min=oracle_max_violation),
            zero_physics_pipeline_calls=calls,examples=examples,robot_axes25=list(servo.axes25),actual_actuator_names21=list(tracker.servo.model.actuator(int(i)).name for i in tracker.servo.aids))
    finally:
        for name,value in blocked.items():setattr(mujoco,name,value)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError('Fresh QA result required')
    result=dict(schema='v12_current_state_effort_tracker_author_v1',scope='robot model compilation and zero-integration actuation algebra; no new physical trial',
        source_sha256={str(p):sha(p) for p in [ROOT/'v12/effort_tracker.py',ROOT/'v11/robot_servo_effort.py',ROOT/'v10/robot_contact_space.py',Path(__file__)]})
    try:result['qa']=verify()
    except Exception as exc:
        import traceback
        result.update(passed=False,error=repr(exc),traceback=traceback.format_exc())
        with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
        raise
    result['passed']=result['qa']['passed']
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(out=str(out),sha256=sha(out),passed=result['passed'],checks=len(result['qa']['checks']),oracle=result['qa']['numerical_oracle']),ensure_ascii=False))

if __name__=='__main__':main()
