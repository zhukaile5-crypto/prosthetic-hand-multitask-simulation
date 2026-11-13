"""Realize one accepted effort goal; never solve a second force-allocation task.

At a fixed public robot q/v, the admitted native effort axes have disjoint
supports and independent monotone clipped-affine reference maps. Candidate
vectors share the frozen F067 value pipeline; final combined values are checked
with its full evaluation. Exact effort ties choose the closest current reference
before the nominal reference, including saturated platforms. The finite family
is not an all-float optimum certificate. No write, pressure prediction or future
joint/contact safety is provided. Original p0 and unrounded delta stay explicit.
"""
from __future__ import annotations
import numpy as np
from v13_joint.prepared_value_effort21 import PreparedValueEffort21
from v13_joint.native_effort21 import EffortDomain21,NativeInverseResult,_array,_ro,_json
from v13_a1.codec import Reference21


def _bounded(values,lower,upper):
    # Ordered comparisons preserve the scalar native clipping signed-zero rule.
    values=np.where(values<lower,lower,values)
    return np.where(values>upper,upper,values)


class EffortGoalRealizer21(PreparedValueEffort21):
    def solve(self,domain,*,p0_effort21,delta_effort21,correction25,residual_scale25,delta_scale21,requested_target_effort21=None):
        p0=delta=None
        def fail(status,reason,**diagnostics):
            return NativeInverseResult(status,reason,p0_effort21=_ro(p0),requested_delta_effort21=_ro(delta),
                diagnostics=dict(pressure_prediction_available=False,physical_safety_certified=False,writer_called=False,**diagnostics))
        try:
            if type(domain) is not EffortDomain21 or domain._owner is not self._owner:return fail('UNAVAILABLE_SOURCE','foreign_prepared_domain')
            if not domain.available:return fail('UNAVAILABLE_DOMAIN',domain.reason)
            if domain._signature!=self._signature(domain):return fail('UNAVAILABLE_SOURCE','prepared_domain_modified')
            if self._model_signature()!=self._initial_model_signature:return fail('UNAVAILABLE_MODEL','bound_robot_model_changed')
            p0=_array(p0_effort21,(21,),'original_p0_effort21')
            if p0.tobytes()!=domain.p0_effort21.tobytes():return fail('UNAVAILABLE_SOURCE','p0_not_original_preprojection_native_value')
            delta=_array(delta_effort21,(21,),'original_requested_delta_effort21')
            correction=_array(correction25,(25,),'fixed_correction25')
            sr=_array(residual_scale25,(25,),'residual_scale25',positive=True)
            sd=_array(delta_scale21,(21,),'delta_scale21',positive=True)
            supplied=None if requested_target_effort21 is None else _array(requested_target_effort21,(21,),'requested_target_effort21')
            c=domain._context;m=self.servo.model;state=c['state']
            lower=domain.reference_lower21;upper=domain.reference_upper21
            gp=m.actuator_gainprm[self.servo.aids,:3].copy();bp=m.actuator_biasprm[self.servo.aids,:3].copy()
            import mujoco
            gp[np.asarray(m.actuator_gaintype[self.servo.aids])==mujoco.mjtGain.mjGAIN_FIXED,1:]=0.
            bp[np.asarray(m.actuator_biastype[self.servo.aids])==mujoco.mjtBias.mjBIAS_NONE]=0.
            gain=gp[:,0]+gp[:,1]*state.actuator_length21+gp[:,2]*state.actuator_velocity21
            # Only analytic candidate generation folds the intercept; its values
            # have no scoring authority and are always evaluated by original F067.
            intercept=bp[:,0]+bp[:,1]*state.actuator_length21+bp[:,2]*state.actuator_velocity21
            offset=np.zeros(21);offset[18]=self.actor.lift_bias_m
            rounded_goal=p0+delta
            inverse=c['current'].copy()
            with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
                raw=(rounded_goal-intercept)/gain-offset
            usable=np.isfinite(raw)&(gain!=0.)
            inverse[usable]=raw[usable]
            inverse=_bounded(inverse,lower,upper)
            families=[('current',_bounded(c['current'].copy(),lower,upper)),('lower',lower.copy()),('upper',upper.copy()),
                ('nominal',_bounded(c['nominal'].copy(),lower,upper)),('analytic_goal_inverse',inverse)]
            for direction,label in ((-np.inf,'down'),(np.inf,'up')):
                candidate=inverse.copy()
                for k in range(2):
                    candidate=np.nextafter(candidate,direction)
                    candidate=_bounded(candidate,lower,upper)
                    families.append((f'analytic_goal_inverse_{label}{k+1}',candidate.copy()))
            values=[];references=[];engines=[]
            for label,reference in families:
                ref=Reference21(reference,domain.time_s);engine=self.actor.encode(ref)
                sample=self.servo.evaluate(c['q'],c['v'],engine.values,time_s=domain.time_s)
                if sample.status!='KNOWN_ROBOT_ACTUATION_ALGEBRA':return fail('UNAVAILABLE_MODEL','candidate_native_value_unknown',candidate=label)
                values.append(sample.predicted_effort21.copy());references.append(reference.copy());engines.append(engine.values.copy())
            values=np.asarray(values);references=np.asarray(references)
            selected=[];command=np.empty(21);expected=np.empty(21)
            for i in range(21):
                # Delta subtraction is before comparison, retaining small goals
                # that disappear when p0+delta is rounded to an absolute effort.
                errors=(values[:,i]-p0[i])-delta[i]
                j=min(range(len(families)),key=lambda k:(abs(float(errors[k])),abs(float(references[k,i]-c['current'][i])),abs(float(references[k,i]-c['nominal'][i])),k))
                command[i]=references[j,i];expected[i]=values[j,i]
                selected.append(dict(axis=i,family=families[j][0],reference=float(command[i]),expected_native_effort=float(expected[i]),
                    delta_error=float(errors[j]),candidate_values=values[:,i].tolist(),candidate_references=references[:,i].tolist()))
            reference=Reference21(command,domain.time_s);engine=self.actor.encode(reference)
            full=self.servo.evaluate_full(c['q'],c['v'],engine.values,time_s=domain.time_s)
            if full.status!='KNOWN_ROBOT_ACTUATION_ALGEBRA':return fail('UNAVAILABLE_MODEL','final_full_native_unknown')
            native=full.predicted_effort21;actual_delta=native-p0
            equal=native.tobytes()==expected.tobytes()
            bounds=bool(np.all(command>=lower) and np.all(command<=upper))
            limited=self.actor.ctrl_limited21;engine_ok=bool(np.all(engine.values[limited]>=self.actor.ctrl_range21[limited,0]) and np.all(engine.values[limited]<=self.actor.ctrl_range21[limited,1]))
            transmission=full.transmission21x25.tobytes()==domain.transmission21x25.tobytes()
            if not all((equal,bounds,engine_ok,transmission)):
                return fail('NUMERICAL_FAILURE','combined_goal_realization_full_recheck_failed',native_bytes_equal=equal,reference_bounds_ok=bounds,engine_bounds_ok=engine_ok,transmission_bytes_equal=transmission)
            residual=domain.transmission21x25.T@actual_delta-correction
            wr=sr*residual;wd=sd*actual_delta
            objective=float(wr@wr+wd@wd)
            error=actual_delta-delta
            if not np.isfinite(np.r_[actual_delta,residual,error,objective]).all():return fail('NUMERICAL_FAILURE','nonfinite_realization_result')
            diagnostics=dict(role='realize_the_single_accepted_effort_goal_not_second_force_optimization',
                tie_break='native_effort_delta_error_then_current_reference_distance_then_nominal_then_fixed_family_order',
                finite_candidate_families=[x[0] for x in families],axis_selections=selected,finite_family_only=True,all_float_global_optimum_certified=False,
                prepared_value_final_full_recheck=dict(passed=True,native_bytes_equal=equal,transmission_bytes_equal=transmission,
                    derivative_source='unmodified_F067_full_evaluate'),requested_rounded_absolute_effort21=rounded_goal,
                caller_absolute_target21=supplied,absolute_addition_lost_delta21=delta-(rounded_goal-p0),
                actual_effort_minus_requested_delta21=error,scaled_realization_linf=float(np.max(abs(sd*error))),
                diagnostic_objective='original_current_effort_residual_plus_delta_regularizer_not_inverse_selection_goal',
                native_derivative_status=full.derivative_status,native_force_clipped21=full.diagnostics['force_clipped21'],
                actual_reference_bounds_ok=bounds,actual_engine_bounds_ok=engine_ok,reference_coupled_domain_supported=False,
                pressure_prediction_available=False,physical_safety_certified=False,writer_called=False)
            return NativeInverseResult('KNOWN_NATIVE_FINITE_GOAL_REALIZATION','accepted_goal_finite_family_and_combination_full_verified',reference,engine,
                _ro(p0),_ro(delta),_ro(native),_ro(actual_delta),_ro(residual),objective,_json(diagnostics))
        except (ValueError,TypeError,KeyError,AttributeError,FloatingPointError,OverflowError) as exc:
            return fail('NUMERICAL_FAILURE',str(exc))
