"""Fixed-snapshot native values, with a frozen-F067 full final recheck.

This is an arithmetic decomposition, not a new PD, robot or pressure model.
The first q/v snapshot and final combination use unmodified F067 evaluate.
Intermediate finite candidates retain g*u+b0+b1*length+b2*speed operation
order. Their J/K/D are deliberately NOT_COMPUTED_VALUE_ONLY, never stale.
The 6f domain, candidate family, canonical scoring and tie-break stay intact.
"""
from dataclasses import replace
from numbers import Real
import numpy as np
from v11.robot_servo_effort import EffortSnapshot,_finite,_ro
from v13_a1.codec import ReferenceActor21
from .native_effort21 import NativeInverseResult
from .prepared_native_effort21 import PreparedNativeEffort21,_FixedSnapshotServo


class _PreparedValueServo(_FixedSnapshotServo):
    def __init__(self,model):
        super().__init__(model)
        self._value_key=None;self._value_data=None
        self.full_evaluations=0;self.value_evaluations=0

    def clear_transmission_cache(self,*,reason='explicit'):
        super().clear_transmission_cache(reason=reason)
        self._value_key=None;self._value_data=None

    def evaluate_full(self,q25,v25,actual_ctrl,*,control_layout='robot21',time_s=0.):
        self.full_evaluations+=1
        return super().evaluate(q25,v25,actual_ctrl,control_layout=control_layout,time_s=time_s)

    def evaluate(self,q25,v25,actual_ctrl,*,control_layout='robot21',time_s=0.):
        if self.unavailable:
            return self.evaluate_full(q25,v25,actual_ctrl,control_layout=control_layout,time_s=time_s)
        guard=self.model_guard_signature()
        if guard!=self._initial_guard:
            self.clear_transmission_cache(reason='model')
            return self.evaluate_full(q25,v25,actual_ctrl,control_layout=control_layout,time_s=time_s)
        try:
            q=_finite(q25,(25,),'robot_q25');v=_finite(v25,(25,),'robot_velocity25')
            if isinstance(time_s,(bool,np.bool_)) or not isinstance(time_s,Real) or not np.isfinite(time_s):raise ValueError('finite_robot_sample_time_required')
            if control_layout=='robot21':u=_finite(actual_ctrl,(21,),'actual_robot_actuator_target21')
            elif control_layout=='model22':
                value=np.asarray(actual_ctrl,dtype=object)
                if value.shape!=(22,):raise ValueError('actual_model_control22_shape')
                u=_finite(value[self.ctrladr],(21,),'selected_robot_actuator_target21')
            else:raise ValueError('explicit_robot21_or_model22_control_layout_required')
        except (ValueError,TypeError) as exc:
            self.evaluation_calls+=1
            return EffortSnapshot('UNAVAILABLE',(str(exc),),robot_axes25=self.axes25)
        key=(guard,q.tobytes(),v.tobytes())
        if key!=self._value_key:
            full=self.evaluate_full(q,v,u,control_layout='robot21',time_s=time_s)
            if full.status!='KNOWN_ROBOT_ACTUATION_ALGEBRA':return full
            # Keep private independent copies, not aliases of a returned sample.
            m=self.model;gp=m.actuator_gainprm[self.aids,:3].copy();bp=m.actuator_biasprm[self.aids,:3].copy()
            import mujoco
            gp[np.asarray(m.actuator_gaintype[self.aids])==mujoco.mjtGain.mjGAIN_FIXED,1:]=0.
            bp[np.asarray(m.actuator_biastype[self.aids])==mujoco.mjtBias.mjBIAS_NONE]=0.
            A=full.transmission21x25.copy();length=full.actuator_length21.copy();speed=full.actuator_velocity21.copy()
            # g is fixed at this q/v. Bias terms are NOT summed into beta.
            gain=gp[:,0]+gp[:,1]*length+gp[:,2]*speed
            self._value_data=(A,length,speed,gain,bp,dict(full.diagnostics))
            self._value_key=key
            if control_layout=='model22':
                diag=dict(full.diagnostics);diag.update(nonrobot_control_ignored=True,control_layout='model22')
                full=replace(full,diagnostics=diag)
            return full
        self.evaluation_calls+=1;self.value_evaluations+=1
        A,length,speed,g,bias,template=self._value_data
        effective=u.copy();ctrl_kink=np.zeros(21,bool)
        import mujoco
        if not int(self.model.opt.disableflags)&int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL):
            # Original scalar clipping retains signedzero/ULP behavior.
            for i in np.flatnonzero(self.ctrl_limited):
                lo,hi=self.ctrl_range[i];effective[i]=np.clip(u[i],lo,hi)
                ctrl_kink[i]=abs(u[i]-lo)<=1e-12 or abs(u[i]-hi)<=1e-12
        # Exactly F067's native sequence, never g*u+collapsed_intercept.
        unbounded=g*effective+bias[:,0]+bias[:,1]*length+bias[:,2]*speed
        effort=unbounded.copy();force_kink=np.zeros(21,bool)
        for i in np.flatnonzero(self.force_limited):
            lo,hi=self.force_range[i];effort[i]=np.clip(unbounded[i],lo,hi)
            force_kink[i]=abs(unbounded[i]-lo)<=1e-12 or abs(unbounded[i]-hi)<=1e-12
        diag=dict(template)
        diag.update(pipeline=['fixed_q_v_admission_guarded_F067_native_value_sequence'],
            nonrobot_control_ignored=control_layout=='model22',control_layout=control_layout,
            control_clipped21=(u!=effective).tolist(),force_clipped21=(effort!=unbounded).tolist(),
            control_at_clip_kink21=ctrl_kink.tolist(),force_at_clip_kink21=force_kink.tolist(),
            prepared_value_only=True,prepared_transmission_cache_hit=True,
            derivative_computation='NOT_COMPUTED_VALUE_ONLY',native_value_arithmetic='F067_operation_order_no_bias_collapse')
        return EffortSnapshot('KNOWN_ROBOT_ACTUATION_ALGEBRA',(),float(time_s),self.axes25,
            commanded_target21=_ro(u),effective_target21=_ro(effective),transmission21x25=_ro(A),
            actuator_length21=_ro(length),actuator_velocity21=_ro(speed),servo_error21=_ro(effective-length),
            unbounded_effort21=_ro(unbounded),predicted_effort21=_ro(effort),generalized_actuator_effort25=_ro(A.T@effort),
            reference_jacobian25x21=None,actuation_stiffness25x25=None,actuation_damping25x25=None,
            derivative_status=dict(reference_jacobian='NOT_COMPUTED_VALUE_ONLY',actuation_stiffness='NOT_COMPUTED_VALUE_ONLY',actuation_damping='NOT_COMPUTED_VALUE_ONLY'),
            diagnostics=diag)

    def cache_diagnostics(self):
        result=super().cache_diagnostics()
        result.update(full_evaluations=self.full_evaluations,value_evaluations=self.value_evaluations,value_snapshot_populated=self._value_key is not None)
        return result


class PreparedValueEffort21(PreparedNativeEffort21):
    """Same keyword API/finite search; full-native verification after solve.

    No exact-zero shortcut. The original QP d21 and prepermission p0 bytes are
    untouched. No writer, new clamp/PD parameters, force or safety certificate.
    """
    def __init__(self,model):
        self.servo=_PreparedValueServo(model);self.actor=ReferenceActor21(self.servo.model)
        self.robot_axes25=tuple(self.servo.axes25);self._owner=object()
        self._initial_model_signature=self._model_signature()

    def solve(self,domain,*,p0_effort21,delta_effort21,correction25,residual_scale25,delta_scale21,requested_target_effort21=None):
        result=super().solve(domain,p0_effort21=p0_effort21,delta_effort21=delta_effort21,correction25=correction25,
            residual_scale25=residual_scale25,delta_scale21=delta_scale21,requested_target_effort21=requested_target_effort21)
        if not result.available:return result
        c=domain._context
        full=self.servo.evaluate_full(c['q'],c['v'],result.engine_port21.values,time_s=domain.time_s)
        ok=full.status=='KNOWN_ROBOT_ACTUATION_ALGEBRA'
        native_equal=delta_equal=residual_equal=objective_equal=transmission_equal=False
        if ok:
            dr=full.predicted_effort21-domain.p0_effort21
            residual=full.transmission21x25.T@dr-np.asarray(correction25,float)
            wr=np.asarray(residual_scale25,float)*residual;wd=np.asarray(delta_scale21,float)*dr
            objective=float(wr@wr+wd@wd)
            native_equal=full.predicted_effort21.tobytes()==result.native_effort21.tobytes()
            delta_equal=dr.tobytes()==result.actual_delta_effort21.tobytes()
            residual_equal=residual.tobytes()==result.residual25.tobytes()
            objective_equal=np.float64(objective).tobytes()==np.float64(result.objective).tobytes()
            transmission_equal=full.transmission21x25.tobytes()==domain.transmission21x25.tobytes()
            ok=all((native_equal,delta_equal,residual_equal,objective_equal,transmission_equal))
        diag=dict(result.diagnostics)
        diag['prepared_value_final_full_recheck']=dict(passed=bool(ok),native_bytes_equal=bool(native_equal),delta_bytes_equal=bool(delta_equal),
            residual_bytes_equal=bool(residual_equal),objective_bytes_equal=bool(objective_equal),transmission_bytes_equal=bool(transmission_equal),
            derivative_source='unmodified_F067_full_evaluate',no_zero_shortcut=True)
        if not ok:
            return NativeInverseResult('NUMERICAL_FAILURE','prepared_value_does_not_reproduce_frozen_full_native_result',
                p0_effort21=result.p0_effort21,requested_delta_effort21=result.requested_delta_effort21,diagnostics=diag)
        diag['native_derivative_status']=full.derivative_status
        diag['native_force_clipped21']=full.diagnostics['force_clipped21']
        return replace(result,diagnostics=diag)
