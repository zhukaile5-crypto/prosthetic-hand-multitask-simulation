"""Writer-free finite native-effort inverse at one public robot snapshot.

Authoritative values come from frozen F067 RobotServoEffort and A1 encode,
never an expanded PD surrogate. Scales MULTIPLY before squaring, exactly as
intent_qp. The fixed raw-nominal p0 and original requested d21 are retained.
This is a finite candidate proposal, not all-float optimality, future robot or
pressure prediction, joint-limit feasibility, or a physical safety certificate.
"""
from __future__ import annotations
from dataclasses import dataclass,field,fields
from numbers import Real
from typing import Optional
import hashlib
import numpy as np
from v11.robot_servo_effort import RobotServoEffort,ACTUATORS21
from v13_a1.codec import Reference21,EnginePort21,ReferenceActor21

NEXTAFTER_PER_SIDE=2


def _array(value,shape,name,positive=False):
    if value is None:raise ValueError(name+'_unknown')
    raw=np.asarray(value,dtype=object)
    if raw.shape!=shape or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,Real) for v in raw.flat):
        raise ValueError(name+'_requires_real_'+str(shape))
    a=np.asarray(value,float).copy()
    if not np.isfinite(a).all():raise ValueError(name+'_nonfinite_unknown')
    if positive and not np.all(a>0):raise ValueError(name+'_must_be_positive_multipliers')
    return _ro(a)
def _ro(value):
    if value is None:return None
    a=np.asarray(value).copy();a.setflags(write=False);return a
def _json(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:_json(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_json(v) for v in value]
    return value
def _public(instance):return {f.name:_json(getattr(instance,f.name)) for f in fields(instance) if not f.name.startswith('_')}


@dataclass(frozen=True)
class EffortDomain21:
    status:str
    reason:str
    p0_effort21:Optional[np.ndarray]=None
    effort_lower21:Optional[np.ndarray]=None
    effort_upper21:Optional[np.ndarray]=None
    reference_lower21:Optional[np.ndarray]=None
    reference_upper21:Optional[np.ndarray]=None
    transmission21x25:Optional[np.ndarray]=None
    time_s:Optional[float]=None
    diagnostics:dict=field(default_factory=dict)
    _owner:object=field(default=None,repr=False,compare=False)
    _context:object=field(default=None,repr=False,compare=False)
    _signature:str=field(default='',repr=False,compare=False)
    @property
    def available(self):return self.status=='KNOWN_DISJOINT_NATIVE_EFFORT_BOX'
    def to_dict(self):return _public(self)


@dataclass(frozen=True)
class NativeInverseResult:
    status:str
    reason:str
    reference21:Optional[Reference21]=None
    engine_port21:Optional[EnginePort21]=None
    p0_effort21:Optional[np.ndarray]=None
    requested_delta_effort21:Optional[np.ndarray]=None
    native_effort21:Optional[np.ndarray]=None
    actual_delta_effort21:Optional[np.ndarray]=None
    residual25:Optional[np.ndarray]=None
    objective:Optional[float]=None
    diagnostics:dict=field(default_factory=dict)
    @property
    def available(self):return self.reference21 is not None and self.engine_port21 is not None
    def to_dict(self):
        answer=_public(self)
        for name in ('reference21','engine_port21'):
            value=getattr(self,name)
            answer[name]=None if value is None else dict(values=value.values.tolist(),time_s=value.time_s,axes=list(value.axes))
        return answer


class NativeEffort21:
    def __init__(self,model):
        self.servo=RobotServoEffort(model)
        self.actor=ReferenceActor21(self.servo.model)
        self.robot_axes25=tuple(self.servo.axes25)
        self._owner=object();self._initial_model_signature=self._model_signature()

    def _model_signature(self):
        m=self.servo.model;h=hashlib.sha256();a=self.servo.aids;j=self.servo.jids
        for name,index in [('actuator_gainprm',a),('actuator_biasprm',a),('actuator_gaintype',a),('actuator_biastype',a),
            ('actuator_ctrlrange',self.servo.ctrladr),('actuator_ctrllimited',self.servo.ctrladr),
            ('actuator_forcerange',self.servo.outadr),('actuator_forcelimited',self.servo.outadr),
            ('actuator_trntype',a),('actuator_trnid',a),('actuator_gear',a),('actuator_dyntype',a),('actuator_delay',a),
            ('actuator_plugin',a),('actuator_ctrlnum',a),('actuator_ctrladr',a),('actuator_outnum',a),('actuator_outadr',a),
            ('actuator_actnum',a),('actuator_actadr',a),('jnt_type',j),('jnt_actfrclimited',j),('jnt_actgravcomp',j),('jnt_qposadr',j),('jnt_dofadr',j)]:
            value=np.asarray(getattr(m,name)[index]);h.update(name.encode());h.update(value.tobytes())
        # F067 admits only robot fixed-joint tendons; no world/object geometry.
        for name in ('tendon_adr','tendon_num','tendon_actfrclimited','wrap_type','wrap_objid','wrap_prm'):
            h.update(name.encode());h.update(np.asarray(getattr(m,name)).tobytes())
        h.update(repr((self.robot_axes25,self.actor.lift_bias_m,float(m.body_subtreemass[m.body('rh_forearm').id]),
            int(m.opt.disableflags),int(m.opt.enableflags),int(m.opt.disableactuator))).encode())
        return h.hexdigest()

    def _signature(self,d):
        h=hashlib.sha256()
        for name in ('p0_effort21','effort_lower21','effort_upper21','reference_lower21','reference_upper21','transmission21x25'):
            value=getattr(d,name);h.update(name.encode());h.update(b'None' if value is None else value.tobytes())
        c=d._context
        if c is not None:
            for name in ('q','v','current','nominal','actual'):
                h.update(name.encode());h.update(c[name].tobytes())
        h.update(repr(d.time_s).encode());return h.hexdigest()

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
            anchor=np.minimum(upper,np.maximum(lower,nominal));pl=np.empty(21);pu=np.empty(21)
            for i in range(21):
                values=[]
                for endpoint in (lower[i],upper[i]):
                    x=anchor.copy();x[i]=endpoint
                    sample=self.servo.evaluate(q,v,self.actor.encode(Reference21(x,float(time_s))).values)
                    if sample.status!='KNOWN_ROBOT_ACTUATION_ALGEBRA' or not np.isfinite(sample.predicted_effort21).all():return unavailable('UNAVAILABLE_MODEL','endpoint_native_effort_unknown')
                    values.append(sample.predicted_effort21[i])
                pl[i]=min(values);pu[i]=max(values)
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

    def solve(self,domain,*,p0_effort21,delta_effort21,correction25,residual_scale25,delta_scale21,requested_target_effort21=None):
        original_d=None;p0=None
        def failure(status,reason,**diag):
            return NativeInverseResult(status,reason,p0_effort21=_ro(p0),requested_delta_effort21=_ro(original_d),
                diagnostics=dict(pressure_prediction_available=False,physical_safety_certified=False,**diag))
        try:
            if type(domain) is not EffortDomain21 or domain._owner is not self._owner:return failure('UNAVAILABLE_SOURCE','foreign_prepared_domain')
            p0=domain.p0_effort21
            if not domain.available:return failure('UNAVAILABLE_DOMAIN',domain.reason)
            if domain._signature!=self._signature(domain):return failure('UNAVAILABLE_SOURCE','prepared_domain_modified')
            if self._model_signature()!=self._initial_model_signature:return failure('UNAVAILABLE_MODEL','bound_robot_model_changed')
            p0=_array(p0_effort21,(21,),'original_p0_effort21')
            if p0.tobytes()!=domain.p0_effort21.tobytes():return failure('UNAVAILABLE_SOURCE','p0_not_original_preprojection_native_value')
            original_d=_array(delta_effort21,(21,),'original_requested_delta_effort21')
            correction=_array(correction25,(25,),'fixed_correction25')
            sr=_array(residual_scale25,(25,),'residual_scale25',positive=True)
            sd=_array(delta_scale21,(21,),'delta_scale21',positive=True)
            supplied_target=None if requested_target_effort21 is None else _array(requested_target_effort21,(21,),'requested_target_effort21')
        except (ValueError,TypeError,KeyError,AttributeError) as exc:return failure('UNAVAILABLE_INPUT',str(exc))
        c=domain._context;A=domain.transmission21x25;lower=domain.reference_lower21;upper=domain.reference_upper21
        seed=np.asarray(c['anchor']).copy();cache={};native_calls=0
        def score(x):
            nonlocal native_calls
            reference=Reference21(x,domain.time_s);engine=self.actor.encode(reference);key=engine.values.tobytes()
            if key not in cache:
                snapshot=self.servo.evaluate(c['q'],c['v'],engine.values);native_calls+=1
                if snapshot.status!='KNOWN_ROBOT_ACTUATION_ALGEBRA':raise ValueError('candidate_native_value_unknown')
                native=np.asarray(snapshot.predicted_effort21);dreal=native-p0
                residual=A.T@dreal-correction
                weighted=sr*residual;regular=sd*dreal
                objective=float(weighted@weighted+regular@regular)
                if not np.isfinite(objective):raise ValueError('candidate_objective_nonfinite')
                cache[key]=dict(native=native.copy(),delta=dreal.copy(),residual=residual.copy(),objective=objective,snapshot=snapshot)
            return cache[key]
        try:
            state=c['state'];m=self.servo.model
            gp=m.actuator_gainprm[self.servo.aids,:3].copy();bp=m.actuator_biasprm[self.servo.aids,:3].copy()
            import mujoco
            gp[np.asarray(m.actuator_gaintype[self.servo.aids])==mujoco.mjtGain.mjGAIN_FIXED,1:]=0.
            bp[np.asarray(m.actuator_biastype[self.servo.aids])==mujoco.mjtBias.mjBIAS_NONE]=0.
            g=gp[:,0]+gp[:,1]*state.actuator_length21+gp[:,2]*state.actuator_velocity21
            # This intercept is ONLY candidate generation; never value/score authority.
            intercept=bp[:,0]+bp[:,1]*state.actuator_length21+bp[:,2]*state.actuator_velocity21
            offset=np.zeros(21);offset[18]=self.actor.lift_bias_m
            rounded_request=p0+original_d;reference=seed.copy();solutions=[]
            for axis in range(21):
                lo=float(lower[axis]);hi=float(upper[axis]);support=c['supports'][axis]
                base=[(lo,'lower'),(hi,'upper'),(float(np.clip(c['current'][axis],lo,hi)),'current'),
                      (float(np.clip(c['nominal'][axis],lo,hi)),'nominal')]
                breaks=[lo,hi]
                if g[axis]!=0.:
                    inverse=(rounded_request[axis]-intercept[axis])/g[axis]-offset[axis]
                    base.append((float(np.clip(inverse,lo,hi)),'requested_effort_inverse'))
                    if self.servo.force_limited[axis]:
                        for boundary in self.servo.force_range[axis]:
                            kink=float((boundary-intercept[axis])/g[axis]-offset[axis])
                            if lo<=kink<=hi:breaks.append(kink);base.append((kink,'force_clip_kink'))
                breaks=np.unique(breaks)
                for left,right in zip(breaks[:-1],breaks[1:]):
                    midpoint=float(left+(right-left)/2.)
                    slope=float(g[axis]);constant=float(g[axis]*offset[axis]+intercept[axis])
                    raw=slope*midpoint+constant
                    if self.servo.force_limited[axis]:
                        fl,fu=self.servo.force_range[axis]
                        if raw<fl:slope=0.;constant=float(fl)
                        elif raw>fu:slope=0.;constant=float(fu)
                    vector=np.r_[sr[support]*A[axis,support]*slope,sd[axis]*slope]
                    bias=np.r_[sr[support]*(A[axis,support]*(constant-p0[axis])-correction[support]),sd[axis]*(constant-p0[axis])]
                    curvature=float(vector@vector)
                    if curvature>0:point=float(np.clip(-(vector@bias)/curvature,left,right));base.append((point,'analytic_local_effort_quadratic'))
                # Fixed two representable neighbours on each side, chosen before
                # results. No adaptive float search or all-float optimality claim.
                candidates={}
                for value,origin in base:
                    family=[(value,origin)]
                    for direction,label in ((-np.inf,'down'),(np.inf,'up')):
                        neighbour=value
                        for k in range(NEXTAFTER_PER_SIDE):
                            neighbour=float(np.nextafter(neighbour,direction))
                            if np.isfinite(neighbour) and lo<=neighbour<=hi:family.append((neighbour,origin+f'/nextafter_{label}{k+1}'))
                    for point,label in family:
                        if lo<=point<=hi:candidates.setdefault(np.float64(point).tobytes(),(point,label))
                evaluated=[]
                for point,origin in candidates.values():
                    candidate=seed.copy();candidate[axis]=point;native=score(candidate)
                    rr=np.asarray(native['residual'][support],dtype=np.float64)
                    dd=float(native['delta'][axis]);wr=sr[support]*rr
                    local=float(wr@wr+(sd[axis]*dd)**2)
                    evaluated.append(dict(value=point,origin=origin,local_native_objective=local,
                        full_seed_native_objective=native['objective'],native_effort=float(native['native'][axis]),
                        actual_delta_effort=dd,residual_bytes_hex=rr.tobytes().hex(),delta_bytes_hex=np.float64(dd).tobytes().hex()))
                if not evaluated:raise ValueError('empty_finite_candidate_family')
                best=min(evaluated,key=lambda e:(e['local_native_objective'],abs(e['value']-c['nominal'][axis]),abs(e['value']-c['current'][axis]),e['value']))
                reference[axis]=best['value']
                solutions.append(dict(axis=axis,torque_support25=support.tolist(),known_reference_gain=float(g[axis]),
                    selected=best,candidates=evaluated))
            final=score(reference);engine=self.actor.encode(Reference21(reference,domain.time_s))
            bounds_ok=bool(np.all(reference>=lower)&np.all(reference<=upper))
            actual_port_ok=bool(np.all(engine.values[self.actor.ctrl_limited21]>=self.actor.ctrl_range21[self.actor.ctrl_limited21,0]) and
                np.all(engine.values[self.actor.ctrl_limited21]<=self.actor.ctrl_range21[self.actor.ctrl_limited21,1]))
            rechecks=[]
            for solution in solutions:
                i=solution['axis'];support=np.asarray(solution['torque_support25'],int);best=solution['selected']
                rr=np.asarray(final['residual'][support],dtype=np.float64);dd=float(final['delta'][i]);wr=sr[support]*rr
                local=float(wr@wr+(sd[i]*dd)**2)
                ok=(rr.tobytes().hex()==best['residual_bytes_hex'] and np.float64(dd).tobytes().hex()==best['delta_bytes_hex'] and
                    np.float64(local).tobytes()==np.float64(best['local_native_objective']).tobytes())
                rechecks.append(dict(axis=i,canonical_native_bytes_equal=bool(ok),final_local_native_objective=local))
            if not bounds_ok or not actual_port_ok or not all(x['canonical_native_bytes_equal'] for x in rechecks):
                return failure('NUMERICAL_FAILURE','combined_native_realization_does_not_reproduce_selected_scores',rechecks=rechecks,
                    actual_reference_bounds_ok=bounds_ok,actual_engine_bounds_ok=actual_port_ok)
            rounded_loss=rounded_request-p0
            reachable_lower=domain.effort_lower21-p0;reachable_upper=domain.effort_upper21-p0
            gap=original_d-np.clip(original_d,reachable_lower,reachable_upper)
            diagnostics=dict(scale_convention='positive multipliers before squaring, identical to intent_qp',
                objective='||Sr*(A.T@(p_native-p0)-fixed_correction)||^2+||Sd*(p_native-p0)||^2',
                requested_delta21=original_d,requested_rounded_absolute_effort21=rounded_request,
                caller_absolute_target21=supplied_target,absolute_addition_lost_delta21=original_d-rounded_loss,
                requested_delta_outside_native_interval21=gap,actual_effort_minus_requested_delta21=final['delta']-original_d,
                finite_candidate_solutions=solutions,combined_native_rechecks=rechecks,native_evaluations=native_calls,
                nextafter_neighbours_per_side=NEXTAFTER_PER_SIDE,actual_reference_bounds_ok=bounds_ok,actual_engine_bounds_ok=actual_port_ok,
                native_force_clipped21=final['snapshot'].diagnostics['force_clipped21'],
                native_derivative_status=final['snapshot'].derivative_status,finite_family_only=True,
                all_float_global_optimum_certified=False,reference_coupled_domain_supported=False,
                pressure_prediction_available=False,physical_safety_certified=False,writer_called=False)
            return NativeInverseResult('KNOWN_NATIVE_FINITE_PROPOSAL','canonical_native_values_and_combination_verified',
                Reference21(reference,domain.time_s),engine,p0,original_d,_ro(final['native']),_ro(final['delta']),
                _ro(final['residual']),final['objective'],_json(diagnostics))
        except (ValueError,TypeError,KeyError,AttributeError,FloatingPointError) as exc:
            return failure('NUMERICAL_FAILURE',str(exc),native_evaluations=native_calls)
