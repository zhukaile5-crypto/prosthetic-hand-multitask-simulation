"""Exact-current-state robot-effort matching in real holding references.

No empirical pressure B, future robot response, contact stiffness/force gradient,
24q_des invention, object input, physical safety or global local-domain claim.
The known eight disjoint actuation supports allow exact scalar piecewise LS.
"""
from __future__ import annotations
from dataclasses import dataclass,field
from numbers import Real
from typing import Optional
import numpy as np
from v11.robot_servo_effort import RobotServoEffort,ACTUATORS21
from v10.robot_contact_space import MechanicalReferenceBasis


def _array(value,shape,name):
    raw=np.asarray(value,dtype=object)
    if raw.shape!=shape or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,Real) for x in raw.flat):raise ValueError(name+' must be real numeric '+str(shape))
    result=np.asarray(value,float)
    if not np.isfinite(result).all():raise ValueError(name+' must be finite')
    return result.copy()
def _ro(x):
    if x is None:return None
    a=np.asarray(x).copy();a.setflags(write=False);return a
def _json(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:_json(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [_json(v) for v in x]
    return x


@dataclass(frozen=True)
class EffortTrackingResult:
    increment8:Optional[np.ndarray]
    port_target21:Optional[np.ndarray]
    debug:dict=field(default_factory=dict)
    status:str='UNAVAILABLE'
    reference8:Optional[np.ndarray]=None
    desired_torque25:Optional[np.ndarray]=None
    achieved_torque25:Optional[np.ndarray]=None
    residual25:Optional[np.ndarray]=None

    @property
    def available(self):return self.increment8 is not None and self.port_target21 is not None
    def to_dict(self):return _json(vars(self))


class EffortTracker:
    def __init__(self,robot_model,open_pose=None,closed_pose=None):
        self.servo=RobotServoEffort(robot_model)
        model=self.servo.model;lift=model.joint('transport_lift').id
        # Constructor authority is real model reference authority only. Each
        # solve additionally intersects the caller's actual step/task corridor.
        motor=model.actuator('lift_motor').id;kp=float(model.actuator_gainprm[motor,0])
        if not np.isfinite(kp) or kp<=0:raise ValueError('Known positive lift reference gain required')
        bias=float(model.body_subtreemass[model.body('rh_forearm').id])*9.81/kp
        lift_range=model.actuator_ctrlrange[motor]-bias
        if model.jnt_limited[lift]:lift_range=np.array([max(lift_range[0],model.jnt_range[lift,0]),min(lift_range[1],model.jnt_range[lift,1])])
        self.basis=MechanicalReferenceBasis(model,lift_lower_m=float(lift_range[0]),lift_upper_m=float(lift_range[1]),open_pose=open_pose,closed_pose=closed_pose)
        self.C=np.asarray(self.basis.held_to_command21,float).copy()
        self.port_offset=np.asarray(self.basis.offset21,float).copy();self.port_offset[18]+=self.basis.lift_motor_offset
        self.port_order_matches=bool(np.array_equal(self.servo.ctrladr,self.basis.actuator_indices21))
        self.rows=tuple(np.flatnonzero(self.C[:,i]!=0.) for i in range(8))

    def _port(self,reference):
        # Preserve the native per-finger open+alpha*slope operation order.
        fingers=np.concatenate([self.basis.open_pose[i]+reference[i]*(self.basis.closed_pose[i]-self.basis.open_pose[i]) for i in range(5)])
        return np.r_[fingers,reference[5]+self.basis.lift_motor_offset,reference[6:]]

    def _failure(self,status,reason,debug=None,target=None):
        data=dict(architecture='known_robot_current_state_effort_tracker',status=status,reason=reason,
            empirical_pressure_B_used=False,future_q_or_pressure_predicted=False,contact_force_gradient_available=False,
            physical_safety_certified=False,global_model_admitted=False)
        if debug:data.update(debug)
        return EffortTrackingResult(None,None,_json(data),status,desired_torque25=_ro(target))

    def solve(self,*,q25,v25,current_reference8,actual_target21,desired_torque_correction25,
              nominal_motion_reference3,step_lower8,step_upper8,motion_reference_bounds,tracking_tolerances3):
        try:
            q=_array(q25,(25,),'robot q25');v=_array(v25,(25,),'robot v25')
            current=_array(current_reference8,(8,),'actual current reference8');actual=_array(actual_target21,(21,),'actual engine port target21')
            correction=_array(desired_torque_correction25,(25,),'desired torque correction25')
            motion=_array(nominal_motion_reference3,(3,),'nominal motion reference3')
            sl=_array(step_lower8,(8,),'real step lower8');su=_array(step_upper8,(8,),'real step upper8')
            if np.any(sl>su):raise ValueError('ordered real step bounds required')
            if isinstance(motion_reference_bounds,dict):
                ml=_array(motion_reference_bounds['lower'],(3,),'motion lower3');mu=_array(motion_reference_bounds['upper'],(3,),'motion upper3')
            else:
                mb=_array(motion_reference_bounds,(2,3),'motion bounds2x3');ml,mu=mb
            if np.any(ml>mu):raise ValueError('ordered motion bounds required')
            tolerances=np.asarray(tracking_tolerances3,dtype=object)
            if tolerances.shape!=(3,):raise ValueError('three explicit tracking tolerances (None=no corridor) required')
            for x in tolerances:
                if x is not None and (isinstance(x,(bool,np.bool_)) or not isinstance(x,Real) or not np.isfinite(x) or x<0):raise ValueError('nonnegative finite or None tracking tolerances required')
        except (ValueError,TypeError,KeyError) as exc:return self._failure('UNAVAILABLE_INPUT',str(exc))
        if not self.port_order_matches:return self._failure('UNAVAILABLE_MODEL','reference actuator/control port order mismatch')
        state=self.servo.evaluate(q,v,actual)
        if state.status!='KNOWN_ROBOT_ACTUATION_ALGEBRA':return self._failure('UNAVAILABLE_MODEL',';'.join(state.reasons))
        expected_current=self._port(current);mismatch=actual-expected_current
        if np.max(abs(mismatch))>1e-10:return self._failure('UNAVAILABLE_REFERENCE_CONTRACT','actual21 does not match current held8 and native lift bias',
            dict(actual_minus_expected_port21=mismatch,known_lift_bias_m=self.basis.lift_motor_offset))
        if np.any(current<self.basis.held_lower8-1e-10) or np.any(current>self.basis.held_upper8+1e-10):
            return self._failure('UNAVAILABLE_REFERENCE_CONTRACT','current holding reference outside true mechanical reference domain')
        nominal=np.r_[current[:5],motion];nominal_port=self._port(nominal)
        # The requested goal is fixed BEFORE candidate feasibility projection.
        nominal_state=self.servo.evaluate(q,v,nominal_port)
        if nominal_state.status!='KNOWN_ROBOT_ACTUATION_ALGEBRA':return self._failure('UNAVAILABLE_MODEL','nominal actuation unavailable')
        desired=nominal_state.generalized_actuator_effort25+correction
        lower=np.maximum(current+sl,self.basis.held_lower8);upper=np.minimum(current+su,self.basis.held_upper8)
        lower[5:]=np.maximum(lower[5:],ml);upper[5:]=np.minimum(upper[5:],mu)
        for i,tol in enumerate(tolerances):
            if tol is not None:lower[5+i]=max(lower[5+i],motion[i]-float(tol));upper[5+i]=min(upper[5+i],motion[i]+float(tol))
        context=dict(goal_scope='fixed current q/v; E(currentalpha,raw nominalmotion)+requested correction, never replaced by a feasible retarget',
            nominal_reference8=nominal,nominal_port_target21=nominal_port,
            requested_correction25=correction,candidate_lower_reference8=lower,candidate_upper_reference8=upper,
            nominal_native_control_clipped21=nominal_state.diagnostics['control_clipped21'],
            native_damping_preserved=True,known_lift_engine_bias_m=self.basis.lift_motor_offset)
        if np.any(lower>upper):return self._failure('INFEASIBLE_REFERENCE_SET','empty real step/port/motion-corridor intersection',context,desired)
        A=np.asarray(state.transmission21x25);supports=tuple(np.flatnonzero(np.any(abs(A[rows])>1e-13,axis=0)) if len(rows) else np.array([],int) for rows in self.rows)
        overlaps=[(i,j,sorted(set(supports[i])&set(supports[j]))) for i in range(8) for j in range(i) if set(supports[i])&set(supports[j])]
        if overlaps:return self._failure('UNAVAILABLE_COUPLED_SUPPORT','model needs a general coupled solver, no false scalar decomposition',dict(**context,overlaps=overlaps),desired)
        model=self.servo.model;gprm=model.actuator_gainprm[self.servo.aids,:3].copy()
        import mujoco
        gprm[np.asarray(model.actuator_gaintype[self.servo.aids])==mujoco.mjtGain.mjGAIN_FIXED,1:]=0.
        g=gprm[:,0]+gprm[:,1]*state.actuator_length21+gprm[:,2]*state.actuator_velocity21
        beta=state.unbounded_effort21-g*state.effective_target21
        pnom=np.asarray(nominal_state.predicted_effort21)
        force_lower=self.servo.force_range[:,0];force_upper=self.servo.force_range[:,1];limited=self.servo.force_limited
        def force_at(i,x):
            rows=self.rows[i];raw=g[rows]*(self.port_offset[rows]+self.C[rows,i]*x)+beta[rows]
            return np.where(limited[rows],np.clip(raw,force_lower[rows],force_upper[rows]),raw)
        nominal_legal=bool(np.all(nominal>=lower) and np.all(nominal<=upper))
        exact_passthrough=bool(nominal_legal and np.array_equal(correction,np.zeros(25)))
        reference=nominal.copy() if exact_passthrough else np.minimum(upper,np.maximum(lower,nominal))
        solutions=[]
        if not exact_passthrough:
            for axis in range(8):
                rows=self.rows[axis];support=supports[axis];lo=float(lower[axis]);hi=float(upper[axis]);pref=float(nominal[axis])
                breakpoints=[lo,hi]
                for row in rows:
                    slope=float(g[row]*self.C[row,axis])
                    if limited[row] and slope!=0:
                        intercept=float(g[row]*self.port_offset[row]+beta[row])
                        for boundary in self.servo.force_range[row]:
                            b=float((boundary-intercept)/slope)
                            if lo<b<hi:breakpoints.append(b)
                breaks=np.unique(breakpoints)
                trial=[(float(x),'boundary') for x in breaks]
                trial.append((float(np.clip(pref,lo,hi)),'closest_nominal'))
                M=A[rows][:,support].T
                for left,right in zip(breaks[:-1],breaks[1:]):
                    midpoint=float(left+(right-left)/2)
                    slopes=g[rows]*self.C[rows,axis];intercepts=g[rows]*self.port_offset[rows]+beta[rows]
                    raw=slopes*midpoint+intercepts;slopes=slopes.copy();intercepts=intercepts.copy()
                    for j,row in enumerate(rows):
                        if limited[row] and raw[j]<force_lower[row]:slopes[j]=0.;intercepts[j]=force_lower[row]
                        elif limited[row] and raw[j]>force_upper[row]:slopes[j]=0.;intercepts[j]=force_upper[row]
                        elif limited[row] and force_lower[row]==force_upper[row]:slopes[j]=0.;intercepts[j]=force_lower[row]
                    slope=M@slopes;intercept=M@(intercepts-pnom[rows])-correction[support]
                    curvature=float(slope@slope)
                    value=float(np.clip(-(slope@intercept)/curvature,left,right)) if curvature>0 else float(np.clip(pref,left,right))
                    trial.append((value,'analytic_segment_minimum' if curvature>0 else 'known_flat_clip_segment'))
                candidates=[]
                for value,origin in trial:
                    residual=M@(force_at(axis,value)-pnom[rows])-correction[support]
                    loss=float(residual@residual)
                    if not np.isfinite(loss):return self._failure('NUMERICAL_FAILURE','nonfinite exact piecewise objective',dict(**context,failed_axis=axis),desired)
                    candidates.append(dict(value=value,loss=loss,distance_to_nominal=abs(value-pref),origin=origin))
                best=min(candidates,key=lambda x:(x['loss'],x['distance_to_nominal'],x['value']))
                reference[axis]=best['value']
                solutions.append(dict(axis=axis,torque_support25=support.tolist(),breakpoints=breaks.tolist(),
                    candidates=candidates,selected=best,
                    objective_unit='N^2' if axis==5 else 'Nm^2',
                    lexicographic_role='first exact effort LS, among exactly equal losses closest nominal command'))
        port=nominal_port.copy() if exact_passthrough else self._port(reference)
        # Explicit actual matrix/port reconstruction and native clipping check.
        step=reference-current
        bounds_ok=bool(np.all(reference>=lower-1e-12)&np.all(reference<=upper+1e-12)&np.all(step>=sl-1e-12)&np.all(step<=su+1e-12))
        ports_ok=bool(np.all(port[self.servo.ctrl_limited]>=self.servo.ctrl_range[self.servo.ctrl_limited,0]-1e-12)&np.all(port[self.servo.ctrl_limited]<=self.servo.ctrl_range[self.servo.ctrl_limited,1]+1e-12))
        final=self.servo.evaluate(q,v,port)
        if final.status!='KNOWN_ROBOT_ACTUATION_ALGEBRA' or not bounds_ok or not ports_ok:
            return self._failure('NUMERICAL_FAILURE','final real reference/port/native evaluation failed',dict(**context,bounds_ok=bounds_ok,ports_ok=ports_ok),desired)
        residual=A.T@(final.predicted_effort21-pnom)-correction
        direct=final.generalized_actuator_effort25-desired
        law=g*final.effective_target21+beta
        law=np.where(limited,np.clip(law,force_lower,force_upper),law)
        check_error=float(np.max(abs(final.predicted_effort21-law)))
        residual_reconstruction=float(np.max(abs(direct-residual)))
        if check_error>1e-10 or residual_reconstruction>1e-10 or any(final.diagnostics['control_clipped21']):
            return self._failure('NUMERICAL_FAILURE','native final force/goal reconstruction mismatch or hidden port clipping',
                dict(**context,native_force_reconstruction_error=check_error,residual_reconstruction_error=residual_reconstruction),desired)
        scalar_recheck=[]
        for solution in solutions:
            ids=np.asarray(solution['torque_support25'],int)
            actual_loss=float(residual[ids]@residual[ids]);selected_loss=solution['selected']['loss']
            tolerance=512*np.finfo(float).eps*max(1e-24,actual_loss,selected_loss,float(correction[ids]@correction[ids]))
            scalar_recheck.append(dict(axis=solution['axis'],native_actual_loss=actual_loss,enumerated_min_loss=selected_loss,
                absolute_difference=abs(actual_loss-selected_loss),roundoff_tolerance=tolerance))
            if abs(actual_loss-selected_loss)>max(1e-24,tolerance):
                return self._failure('NUMERICAL_FAILURE','combined native effort does not realize the scalar optimum',
                    dict(**context,scalar_native_recheck=scalar_recheck),desired)
        # Orthogonal uncontrolled generalized coordinates remain real residual.
        controlled=set(i for s in supports for i in s);uncontrolled=sorted(set(range(25))-controlled)
        roundoff=256*np.finfo(float).eps*np.maximum(1.,np.maximum(abs(desired),abs(final.generalized_actuator_effort25)))
        exact_match=bool(np.all(abs(residual)<=roundoff))
        status='KNOWN_CURRENT_STATE_EFFORT_MATCH' if exact_match else 'KNOWN_CURRENT_STATE_EFFORT_LS_WITH_RESIDUAL'
        debug=dict(architecture='known_robot_current_state_effort_tracker',status=status,
            **context,exact_zero_correction_nominal_passthrough=exact_passthrough,
            matching_method='8 independent scalar piecewise-quadratic global minima; actual transmission supports checked disjoint',
            requested_target_torque25=desired,achieved_torque25=final.generalized_actuator_effort25,
            residual25=residual,residual_units25=['Nm']*24+['N'],uncontrolled_torque_coordinates25=uncontrolled,
            angular_residual_l2_Nm=float(np.linalg.norm(residual[:24])),lift_residual_N=float(residual[24]),
            residual_is_not_renamed_as_achieved_target=True,scalar_solutions=solutions,
            true_step_reference_bounds_checked=bounds_ok,true_actuator_port_bounds_checked=ports_ok,
            native_force_clipped21=final.diagnostics['force_clipped21'],native_derivative_status=final.derivative_status,
            clipping_kink_handling='known continuous clipped force law enumerated exactly; no None Jacobian replaced by zero',
            native_force_reconstruction_error=check_error,residual_reconstruction_error=residual_reconstruction,
            scalar_native_recheck=scalar_recheck,input_time_alignment='q/v/reference21 must be same actual sample; caller owns timestamp alignment',
            future_q_or_pressure_predicted=False,empirical_pressure_B_used=False,contact_force_gradient_available=False,
            mechanical_scope='position/actuator reference constraints only; actual future encoder/pressure constraints not certified',
            tracking_scope='nominal reference corridor only; no actual future tracking guarantee',
            physical_safety_certified=False,global_model_admitted=False)
        return EffortTrackingResult(_ro(step),_ro(port),_json(debug),status,_ro(reference),_ro(desired),_ro(final.generalized_actuator_effort25),_ro(residual))
