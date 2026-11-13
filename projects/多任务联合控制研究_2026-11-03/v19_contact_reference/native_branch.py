"""One deterministic affine native-effort branch per reference axis, at q/v.

The variable is du = reference - ORIGINAL committed reference.  Therefore
    effort_delta = offset21 + gain21 * du
retains a nonzero offset when a TASK interval excludes the original reference.
The original p0 is never changed to a feasible anchor or rounded p0+delta.

Reference membership is distinct from current-state native effort. A known
plateau has gain0 but keeps its reference interval FREE. This is a local native
law branch, not a future robot/force response, physical safety certificate,
inverse or writer. IEEE value errors are measured by final full F067, not
hidden by claiming the affine expression reproduces all floating operations.
"""
from dataclasses import dataclass,field,fields
from numbers import Real
import hashlib
import time
import numpy as np
from v13_a1.codec import Reference21,EnginePort21
from v14_coord.effort_domain import FastEffortGoal21

MAX_BOUNDARY_BISECTIONS=64
_SIGN=1<<63
_MASK=(1<<64)-1

def _array(value,shape,name):
    if value is None:raise ValueError(name+'_unknown')
    raw=np.asarray(value,dtype=object)
    if raw.shape!=shape or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,Real) for x in raw.flat):
        raise ValueError(name+'_requires_real_'+str(shape))
    a=np.asarray(value,float).copy()
    if not np.isfinite(a).all():raise ValueError(name+'_nonfinite')
    a.setflags(write=False);return a

def _ro(value):
    if value is None:return None
    a=np.asarray(value).copy();a.setflags(write=False);return a

def _plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:_plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [_plain(v) for v in x]
    return x

def _ordered(x):
    bits=int(np.float64(x).view(np.uint64))
    return (~bits)&_MASK if bits&_SIGN else bits|_SIGN

def _from_ordered(index):
    bits=(index^_SIGN) if index&_SIGN else (~index)&_MASK
    return float(np.uint64(bits).view(np.float64))


@dataclass(frozen=True)
class NativeBranchResult:
    status:str
    reason:str
    time_s:object=None
    gain21:object=None
    offset21:object=None
    delta_lower21:object=None
    delta_upper21:object=None
    reference_lower21:object=None
    reference_upper21:object=None
    current_reference21:object=None
    preferred_reference21:object=None
    anchor_reference21:object=None
    anchor_effort21:object=None
    p0_effort21:object=None
    transmission21x25:object=None
    branch_kind21:tuple=()
    diagnostics:dict=field(default_factory=dict)
    _owner:object=field(default=None,repr=False,compare=False)
    _context:object=field(default=None,repr=False,compare=False)
    _signature:str=field(default='',repr=False,compare=False)
    @property
    def available(self):return self.status=='KNOWN_CURRENT_DIRECTION_NATIVE_BRANCH'
    @property
    def absolute_reference_lower21(self):return self.reference_lower21
    @property
    def absolute_reference_upper21(self):return self.reference_upper21
    def to_dict(self):return {f.name:_plain(getattr(self,f.name)) for f in fields(self) if not f.name.startswith('_')}


class NativeBranchBuilder21:
    def __init__(self,robot_model):
        self.native=FastEffortGoal21(robot_model)
        self.servo=self.native.servo
        self.actor=self.native.actor
        self.robot_axes25=tuple(self.native.robot_axes25)
        self._owner=object()
        self._model_signature=self.native._initial_model_signature

    def _signature(self,result):
        h=hashlib.sha256()
        for name in ('gain21','offset21','delta_lower21','delta_upper21','reference_lower21','reference_upper21',
                     'current_reference21','preferred_reference21','anchor_reference21','anchor_effort21',
                     'p0_effort21','transmission21x25'):
            h.update(name.encode());value=getattr(result,name);h.update(b'None' if value is None else value.tobytes())
        if result._context is not None:
            for name in ('q','v','actual_engine'):
                h.update(name.encode());h.update(result._context[name].tobytes())
        h.update(repr((result.time_s,result.branch_kind21)).encode());return h.hexdigest()

    @staticmethod
    def _boundary(endpoint,anchor,predicate,*,lower):
        """Fixed-count scalar branch membership, not an action optimization.

        The selected branch is a connected closed interval. Each side between
        its known-valid anchor and an outside endpoint is monotone membership.
        Ordered IEEE bisection avoids folded-intercept kink cancellation. It
        selects only this axis's declared branch, never 21D combinations.
        """
        if predicate(endpoint):return float(endpoint),0
        a,b=(_ordered(endpoint),_ordered(anchor)) if lower else (_ordered(anchor),_ordered(endpoint))
        count=0
        for count in range(1,MAX_BOUNDARY_BISECTIONS+1):
            if b-a<=1:break
            mid=(a+b)//2;inside=predicate(_from_ordered(mid))
            if lower:
                if inside:b=mid
                else:a=mid
            else:
                if inside:a=mid
                else:b=mid
        result=_from_ordered(b if lower else a)
        if not predicate(result):raise ValueError('selected_branch_boundary_unverified')
        return result,count

    def build(self,*,q25,v25,robot_axes25,time_s,current_reference,actual_engine,
              preferred_reference,reference_lower21,reference_upper21):
        began=time.perf_counter_ns()
        full_before=self.servo.full_evaluations;value_before=self.servo.value_evaluations
        raw_calls=0
        base=dict(pressure_prediction_available=False,future_robot_response_available=False,
            physical_safety_certified=False,writer_called=False,inverse_called=False,
            delta_reference_origin='original committed reference21',
            native_affine_scope='known robot force law at the fixed public q/v only',
            mathematical_affine_is_not_all_float_bit_prediction=True,
            branch_selection='one current/preferred direction per axis; no combination or objective search')
        def fail(status,reason,**extra):
            return NativeBranchResult(status,reason,diagnostics=dict(base,
                construction_ms=(time.perf_counter_ns()-began)/1e6,
                native_full_evaluations_delta=self.servo.full_evaluations-full_before,
                native_value_evaluations_delta=self.servo.value_evaluations-value_before,
                scalar_raw_predicate_evaluations=raw_calls,**extra),_owner=self._owner)
        try:
            if type(preferred_reference) is not Reference21:raise TypeError('typed preferred Reference21 required')
            if preferred_reference.time_s!=float(time_s):raise ValueError('preferred_public_clock_mismatch')
            if self.native._model_signature()!=self._model_signature:return fail('UNAVAILABLE_MODEL','bound_robot_model_changed')
            # p0 is deliberately evaluated at raw current, NOT preferred or a
            # permission-projected anchor. The existing domain owns admission.
            domain=self.native.build_domain(q25=q25,v25=v25,robot_axes25=robot_axes25,time_s=time_s,
                current_reference=current_reference,nominal_reference=current_reference,actual_engine=actual_engine,
                reference_lower21=reference_lower21,reference_upper21=reference_upper21)
            if not domain.available:return fail('UNAVAILABLE_DOMAIN',domain.reason,domain_status=domain.status)
            current=current_reference.values;preferred=preferred_reference.values
            lo=domain.reference_lower21;hi=domain.reference_upper21
            if np.any(preferred<lo) or np.any(preferred>hi):
                return fail('UNAVAILABLE_PREFERRED_OUTSIDE_DOMAIN','preferred_must_already_be_executable_not_silently_clipped')
            q=_array(q25,(25,),'q25');v=_array(v25,(25,),'v25')
            state=self.servo.evaluate_full(q,v,actual_engine.values,time_s=float(time_s))
            if state.status!='KNOWN_ROBOT_ACTUATION_ALGEBRA' or not np.isfinite(state.predicted_effort21).all():
                return fail('UNAVAILABLE_NATIVE','current_full_native_value_unknown')
            if state.predicted_effort21.tobytes()!=domain.p0_effort21.tobytes():
                return fail('NUMERICAL_FAILURE','raw_current_p0_full_native_bytes_differ')
            m=self.servo.model;aids=self.servo.aids
            gp=m.actuator_gainprm[aids,:3].copy();bp=m.actuator_biasprm[aids,:3].copy()
            import mujoco
            gp[np.asarray(m.actuator_gaintype[aids])==mujoco.mjtGain.mjGAIN_FIXED,1:]=0.
            bp[np.asarray(m.actuator_biastype[aids])==mujoco.mjtBias.mjBIAS_NONE]=0.
            length=state.actuator_length21;speed=state.actuator_velocity21
            gains=gp[:,0]+gp[:,1]*length+gp[:,2]*speed
            if not np.isfinite(np.r_[gains,bp.ravel(),length,speed]).all():
                return fail('UNAVAILABLE_NATIVE','nonfinite_fixed_robot_coefficients')
            ctrl_enabled=not int(m.opt.disableflags)&int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)
            offset_reference=np.zeros(21);offset_reference[18]=self.actor.lift_bias_m
            def raw(i,u):
                nonlocal raw_calls
                raw_calls+=1
                engine=np.float64(u)+np.float64(offset_reference[i])
                effective=engine
                if ctrl_enabled and self.servo.ctrl_limited[i]:effective=np.clip(engine,*self.servo.ctrl_range[i])
                # The native order is retained. Do not fold beta for values or
                # predicates, including the historical cancellation fixture.
                value=gains[i]*effective+bp[i,0]+bp[i,1]*length[i]+bp[i,2]*speed[i]
                return engine,effective,value
            inside=(current>=lo)&(current<=hi)
            anchor=np.where(inside,current,preferred).copy()
            branchlo=np.empty(21);branchhi=np.empty(21);alpha=np.empty(21);kinds=[];directions=[];counts=[]
            for i in range(21):
                u=float(anchor[i]);direction=int(np.sign(preferred[i]-current[i]));direction_source='preferred_minus_original_current'
                if (direction>0 and not u<hi[i]) or (direction<0 and not lo[i]<u):direction=0
                if direction==0:
                    direction=1 if u<hi[i] else -1 if lo[i]<u else 0
                    direction_source='only_available_side_or_upper_first_neutral_tie'
                engine,effective,value=raw(i,u)
                if not np.isfinite([engine,effective,value]).all():raise ValueError('nonfinite_anchor_raw_native_value')
                if lo[i]==hi[i]:
                    kind='SINGLETON_REFERENCE';slope=0.;predicate=lambda x,u=u:x==u
                elif gains[i]==0.:
                    kind='KNOWN_ZERO_NATIVE_GAIN';slope=0.;predicate=lambda x:True
                elif (ctrl_enabled and self.servo.ctrl_limited[i] and
                      (engine<self.servo.ctrl_range[i,0] or (engine==self.servo.ctrl_range[i,0] and direction<0))):
                    kind='CONTROL_LOW_PLATEAU';slope=0.;predicate=lambda x,i=i:raw(i,x)[0]<=self.servo.ctrl_range[i,0]
                elif (ctrl_enabled and self.servo.ctrl_limited[i] and
                      (engine>self.servo.ctrl_range[i,1] or (engine==self.servo.ctrl_range[i,1] and direction>0))):
                    kind='CONTROL_HIGH_PLATEAU';slope=0.;predicate=lambda x,i=i:raw(i,x)[0]>=self.servo.ctrl_range[i,1]
                elif (self.servo.force_limited[i] and
                      (value<self.servo.force_range[i,0] or (value==self.servo.force_range[i,0] and gains[i]*direction<0))):
                    kind='FORCE_LOW_PLATEAU';slope=0.;predicate=lambda x,i=i:raw(i,x)[2]<=self.servo.force_range[i,0]
                elif (self.servo.force_limited[i] and
                      (value>self.servo.force_range[i,1] or (value==self.servo.force_range[i,1] and gains[i]*direction>0))):
                    kind='FORCE_HIGH_PLATEAU';slope=0.;predicate=lambda x,i=i:raw(i,x)[2]>=self.servo.force_range[i,1]
                else:
                    kind='UNCLIPPED_AFFINE_ONE_SIDED_AT_KINK';slope=float(gains[i])
                    def predicate(x,i=i):
                        en,_,ra=raw(i,x)
                        return bool((not ctrl_enabled or not self.servo.ctrl_limited[i] or self.servo.ctrl_range[i,0]<=en<=self.servo.ctrl_range[i,1]) and
                                    (not self.servo.force_limited[i] or self.servo.force_range[i,0]<=ra<=self.servo.force_range[i,1]))
                if not predicate(u):raise ValueError('anchor_does_not_belong_to_declared_native_branch')
                branchlo[i],lc=self._boundary(float(lo[i]),u,predicate,lower=True)
                branchhi[i],hc=self._boundary(float(hi[i]),u,predicate,lower=False)
                alpha[i]=slope;kinds.append(kind);directions.append(dict(direction=direction,source=direction_source));counts.append([lc,hc])
            samples=[]
            for name,reference in (('anchor',anchor),('lower',branchlo),('upper',branchhi)):
                encoded=self.actor.encode(Reference21(reference,float(time_s)))
                full=self.servo.evaluate_full(q,v,encoded.values,time_s=float(time_s))
                if full.status!='KNOWN_ROBOT_ACTUATION_ALGEBRA' or not np.isfinite(full.predicted_effort21).all():
                    return fail('UNAVAILABLE_NATIVE','full_branch_vector_native_value_unknown',vector=name)
                if full.transmission21x25.tobytes()!=domain.transmission21x25.tobytes():
                    return fail('NUMERICAL_FAILURE','branch_transmission_changed',vector=name)
                samples.append(full.predicted_effort21.copy())
            p_anchor=samples[0];p0=domain.p0_effort21
            # Current-anchor branches get exact0; others retain the fixed p0
            # and the explicit intercept in du=current-relative coordinates.
            intercept=(p_anchor-p0)+alpha*(current-anchor)
            intercept[inside]=0.
            predicted_endpoints=[intercept+alpha*(x-current) for x in (branchlo,branchhi)]
            errors=[samples[k+1]-p0-predicted_endpoints[k] for k in range(2)]
            if not np.isfinite(np.r_[alpha,intercept,*errors]).all():raise ValueError('nonfinite_affine_branch_expression')
            context=dict(q=q,v=v,actual_engine=_ro(actual_engine.values))
            result=NativeBranchResult('KNOWN_CURRENT_DIRECTION_NATIVE_BRANCH','single_admitted_native_branch_per_axis',float(time_s),
                _ro(alpha),_ro(intercept),_ro(branchlo-current),_ro(branchhi-current),_ro(branchlo),_ro(branchhi),
                _ro(current),_ro(preferred),_ro(anchor),_ro(p_anchor),_ro(p0),_ro(domain.transmission21x25),tuple(kinds),
                dict(base,source_model_signature=self._model_signature,original_actual_domain_lower21=lo.tolist(),original_actual_domain_upper21=hi.tolist(),
                    construction_ms=(time.perf_counter_ns()-began)/1e6,
                    native_full_evaluations_delta=self.servo.full_evaluations-full_before,
                    native_value_evaluations_delta=self.servo.value_evaluations-value_before,
                    scalar_raw_predicate_evaluations=raw_calls,
                    current_inside_actual_domain21=inside.tolist(),raw_current_p0_full_native_bytes_equal=True,
                    preferred_inside_selected_branch21=((preferred>=branchlo)&(preferred<=branchhi)).tolist(),
                    known_zero_slope_but_reference_not_fixed21=((alpha==0)&(branchhi>branchlo)).tolist(),
                    selection_direction21=directions,scalar_boundary_bisection_counts21=counts,
                    scalar_boundary_bisection_max=MAX_BOUNDARY_BISECTIONS,
                    endpoint_affine_error21=[x.tolist() for x in errors],
                    final_full_native_recheck_required=True,reference_geometry_is_not_actual_robot_motion=True),self._owner,context)
            object.__setattr__(result,'_signature',self._signature(result));return result
        except (ValueError,TypeError,KeyError,AttributeError,FloatingPointError,OverflowError) as exc:
            return fail('UNAVAILABLE_INPUT',str(exc))

    def evaluate_reference(self,branch,reference):
        """Full original native value and recorded affine error at final ref.

        No command is applied. No old inverse selects a different reference.
        The controller must score its SAME complete task using actual_delta.
        """
        def fail(reason):return dict(available=False,status='UNAVAILABLE_BRANCH_EVALUATION',reason=reason,writer_called=False)
        try:
            if type(branch) is not NativeBranchResult or branch._owner is not self._owner:return fail('foreign_branch_owner')
            if not branch.available:return fail('branch_unavailable')
            if self._signature(branch)!=branch._signature:return fail('branch_or_snapshot_modified')
            if self.native._model_signature()!=self._model_signature:return fail('bound_robot_model_changed')
            if type(reference) is not Reference21 or reference.time_s!=branch.time_s:return fail('typed_same_clock_reference_required')
            u=reference.values
            if np.any(u<branch.reference_lower21) or np.any(u>branch.reference_upper21):return fail('reference_outside_selected_branch')
            c=branch._context;engine=self.actor.encode(reference)
            full=self.servo.evaluate_full(c['q'],c['v'],engine.values,time_s=branch.time_s)
            if full.status!='KNOWN_ROBOT_ACTUATION_ALGEBRA' or not np.isfinite(full.predicted_effort21).all():return fail('full_native_value_unknown')
            if full.transmission21x25.tobytes()!=branch.transmission21x25.tobytes():return fail('full_native_transmission_changed')
            actual=full.predicted_effort21-branch.p0_effort21
            predicted=branch.offset21+branch.gain21*(u-branch.current_reference21)
            error=actual-predicted
            if not np.isfinite(np.r_[actual,predicted,error]).all():return fail('nonfinite_affine_error')
            return dict(available=True,status='FULL_NATIVE_REFERENCE_REEVALUATED',reference21=u.copy(),engine_port21=engine.values.copy(),
                p0_effort21=branch.p0_effort21.copy(),native_effort21=full.predicted_effort21.copy(),
                actual_delta_effort21=actual,affine_delta_effort21=predicted,affine_error21=error,
                native_derivative_status=full.derivative_status,native_diagnostics=full.diagnostics,
                original_p0_preserved=True,pressure_prediction_available=False,future_response_available=False,
                physical_safety_certified=False,writer_called=False,inverse_called=False,
                full_native_is_value_authority=True,affine_error_is_not_silently_zeroed=True)
        except (ValueError,TypeError,KeyError,AttributeError,FloatingPointError,OverflowError) as exc:return fail(str(exc))
