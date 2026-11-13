"""Exact native interval preparation with two full21 endpoint vectors.

Preserves frozen6f domain checks, fixed raw-nominal p0 and model/clock guards.
Only the redundant42 endpoint calls become2, using native per-control algebra.
Full native final goal realization remains in the separate frozen realizer.
"""
from numbers import Real
import numpy as np
from v13_a1.codec import Reference21,EnginePort21
from v13_joint.native_effort21 import EffortDomain21,_array,_ro
from .effort_realization import EffortGoalRealizer21

class FastEffortGoal21(EffortGoalRealizer21):
    def build_domain(self,*,q25,v25,robot_axes25,time_s,current_reference,nominal_reference,actual_engine,
                     reference_lower21=None,reference_upper21=None,step_lower21=None,step_upper21=None,
                     coupled_reference_constraints=None):
        p0=None;context=None;A=None
        def unavailable(status,reason,**diag):
            return EffortDomain21(status,reason,p0_effort21=_ro(p0),transmission21x25=_ro(A),
                diagnostics=dict(pressure_prediction_available=False,physical_safety_certified=False,**diag),_owner=self._owner)
        try:
            if self._model_signature()!=self._initial_model_signature:return unavailable('UNAVAILABLE_MODEL','bound_robot_model_changed')
            if tuple(robot_axes25)!=self.robot_axes25:raise ValueError('robot_axes25_unknown_or_reordered')
            if isinstance(time_s,(bool,np.bool_)) or not isinstance(time_s,Real) or not np.isfinite(time_s) or time_s<0:raise ValueError('invalid_public_snapshot_clock')
            if type(current_reference) is not Reference21 or type(nominal_reference) is not Reference21 or type(actual_engine) is not EnginePort21:
                raise TypeError('explicit_A1_Reference21_and_EnginePort21_required')
            if any(x.time_s!=float(time_s) for x in (current_reference,nominal_reference,actual_engine)):raise ValueError('different_public_snapshot_clocks')
            if actual_engine.lift_bias_m!=self.actor.lift_bias_m:raise ValueError('foreign_engine_lift_bias')
            encoded=self.actor.encode(current_reference)
            if encoded.values.tobytes()!=actual_engine.values.tobytes():raise ValueError('actual_engine_is_not_exact_current_reference_encode')
            q=_array(q25,(25,),'q25');v=_array(v25,(25,),'v25')
            current=current_reference.values;nominal=nominal_reference.values
            state=self.servo.evaluate(q,v,actual_engine.values)
            nominal_state=self.servo.evaluate(q,v,self.actor.encode(nominal_reference).values)
            if state.status!='KNOWN_ROBOT_ACTUATION_ALGEBRA' or nominal_state.status!='KNOWN_ROBOT_ACTUATION_ALGEBRA':
                return unavailable('UNAVAILABLE_MODEL','native_effort_value_unknown',reasons=list(state.reasons)+list(nominal_state.reasons))
            A=_ro(state.transmission21x25);p0=_ro(nominal_state.predicted_effort21)
            if coupled_reference_constraints is not None:return unavailable('UNAVAILABLE_COUPLED_DOMAIN','coupled_reference_constraints_are_not_a_box')
            supports=tuple(np.flatnonzero(A[i]!=0.) for i in range(21))
            if any(set(supports[i])&set(supports[j]) for i in range(21) for j in range(i)):
                return unavailable('UNAVAILABLE_COUPLED_SUPPORT','actuator_generalized_supports_overlap')
            absolute=(reference_lower21 is not None or reference_upper21 is not None)
            steps=(step_lower21 is not None or step_upper21 is not None)
            if not absolute and not steps:raise ValueError('explicit_finite_reference_or_step_bounds_required')
            lower=np.full(21,-np.inf);upper=np.full(21,np.inf)
            if absolute:lower=np.asarray(_array(reference_lower21,(21,),'reference_lower21')).copy();upper=np.asarray(_array(reference_upper21,(21,),'reference_upper21')).copy()
            if steps:
                sl=_array(step_lower21,(21,),'step_lower21');su=_array(step_upper21,(21,),'step_upper21')
                if np.any(sl>su):return unavailable('EMPTY_REFERENCE_SET','unordered_requested_step_bounds')
                lower=np.maximum(lower,current+sl);upper=np.minimum(upper,current+su)
            mechanical_lower=np.full(21,-np.inf);mechanical_upper=np.full(21,np.inf)
            limited=self.actor.ctrl_limited21
            mechanical_lower[limited]=self.actor.ctrl_range21[limited,0];mechanical_upper[limited]=self.actor.ctrl_range21[limited,1]
            if limited[18]:
                mechanical_lower[18]-=self.actor.lift_bias_m;mechanical_upper[18]-=self.actor.lift_bias_m
                # Translation can round outward by one representable reference.
                # Tighten, never widen or clip an emitted engine command.
                for _ in range(2):
                    if mechanical_lower[18]+self.actor.lift_bias_m<self.actor.ctrl_range21[18,0]:
                        mechanical_lower[18]=np.nextafter(mechanical_lower[18],np.inf)
                    if mechanical_upper[18]+self.actor.lift_bias_m>self.actor.ctrl_range21[18,1]:
                        mechanical_upper[18]=np.nextafter(mechanical_upper[18],-np.inf)
            lower=np.maximum(lower,mechanical_lower);upper=np.minimum(upper,mechanical_upper)
            if np.any(lower>upper):return unavailable('EMPTY_REFERENCE_SET','requested_and_native_reference_intersection_empty',requested_bounds_not_retargeted=True)
            if not np.isfinite(lower).all() or not np.isfinite(upper).all():raise ValueError('finite_candidate_reference_bounds_required')
            anchor=np.minimum(upper,np.maximum(lower,nominal))
            # At fixed q/v each admitted native motor force uses only its own
            # command. Two complete endpoint vectors therefore recover all21
            # exact intervals; this is not a pressure or coupled-force model.
            endpoint_values=[]
            for endpoint in (lower,upper):
                sample=self.servo.evaluate(q,v,self.actor.encode(Reference21(endpoint,float(time_s))).values)
                if sample.status!='KNOWN_ROBOT_ACTUATION_ALGEBRA' or not np.isfinite(sample.predicted_effort21).all():return unavailable('UNAVAILABLE_MODEL','endpoint_native_effort_unknown')
                endpoint_values.append(sample.predicted_effort21.copy())
            a,b=endpoint_values
            # Match original scalar min/max: equal endpoint values retain the
            # first endpoint, including its signed-zero bits.
            pl=np.where(b<a,b,a);pu=np.where(b>a,b,a)
            context=dict(q=q,v=v,current=_ro(current),nominal=_ro(nominal),actual=_ro(actual_engine.values),
                anchor=_ro(anchor),supports=supports,state=state,nominal_state=nominal_state)
            d=EffortDomain21('KNOWN_DISJOINT_NATIVE_EFFORT_BOX','independent_reference_intervals_at_fixed_robot_qv',p0,
                _ro(pl),_ro(pu),_ro(lower),_ro(upper),A,float(time_s),
                dict(model_signature=self._initial_model_signature,p0_scope='raw nominal native effort before permission projection',
                    flat_effort_axes21=(pl==pu).tolist(),current_native_clipped21=state.diagnostics['control_clipped21'],
                    nominal_native_clipped21=nominal_state.diagnostics['control_clipped21'],pressure_prediction_available=False,
                    physical_safety_certified=False,future_joint_feasibility_certified=False),self._owner,context)
            object.__setattr__(d,'_signature',self._signature(d));return d
        except (ValueError,TypeError,KeyError,AttributeError) as exc:return unavailable('UNAVAILABLE_INPUT',str(exc))

