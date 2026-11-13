"""Pure conditional force-task ROW construction; never an allocator/solver.

Columns are [effort_delta21, normal_intentP]. Geometric rows apply only to
selected conditional-group patches. Unknown external geometry is not zero
geometry: those columns are INACTIVE for these rows. All patch normal values
and digits must still be valid. Caps/budget/load capacity remain caller-owned.
"""
from dataclasses import dataclass,field
from numbers import Integral,Real
from typing import Optional
import numpy as np

MAX_PATCHES=128
UNIT_NORMAL_TOLERANCE=1e-6

class _TaskInput(ValueError):
    def __init__(self,reason,*,unknown=False,degenerate=False):
        super().__init__(reason);self.unknown=unknown;self.degenerate=degenerate

def _real_array(value,shape,name):
    if value is None:raise _TaskInput(name+'_unknown',unknown=True)
    raw=np.asarray(value,dtype=object)
    if raw.shape!=shape:raise _TaskInput(name+'_shape_'+str(shape))
    if any(isinstance(x,(bool,np.bool_)) or not isinstance(x,Real) for x in raw.flat):
        if any(x is None for x in raw.flat):raise _TaskInput(name+'_unknown',unknown=True)
        raise _TaskInput(name+'_requires_real_values')
    array=np.asarray(value,float).copy()
    if not np.isfinite(array).all():raise _TaskInput(name+'_nonfinite_unknown',unknown=True)
    return array

def _number(value,name,*,positive=False):
    if value is None:raise _TaskInput(name+'_unknown',unknown=True)
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real):raise _TaskInput(name+'_requires_real_value')
    value=float(value)
    if not np.isfinite(value):raise _TaskInput(name+'_nonfinite_unknown',unknown=True)
    if value<0 or positive and value<=0:raise _TaskInput(name+'_outside_nonnegative_domain')
    return value

def _ro(value):
    array=np.asarray(value,float).copy();array.setflags(write=False);return array

def _plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):_plain(x) for k,x in value.items()}
    if isinstance(value,(list,tuple)):return [_plain(x) for x in value]
    return value

@dataclass(frozen=True)
class FusionTaskSpec:
    status:str
    reason:str
    base_prior_p:Optional[np.ndarray]=None
    intentP:Optional[np.ndarray]=None
    extra_matrix:Optional[np.ndarray]=None
    extra_target:Optional[np.ndarray]=None
    labels:tuple=()
    metadata:dict=field(default_factory=dict)
    @property
    def available(self):return self.status=='AVAILABLE_CONDITIONAL_FUSION_TASKS'
    @property
    def extra_task_rows(self):
        return None if not self.available else {'matrix':self.extra_matrix,'target':self.extra_target}
    def to_dict(self):
        return _plain(dict(status=self.status,reason=self.reason,base_prior_p=self.base_prior_p,intentP=self.intentP,
            extra_task_rows=self.extra_task_rows,labels=self.labels,metadata=self.metadata))

def _geometry(value,P,selected,name,*,normal=False):
    if value is None:rows=[None]*P
    else:
        try:rows=list(value)
        except TypeError:raise _TaskInput(name+'_requires_P_rows')
        if len(rows)!=P:raise _TaskInput(name+'_requires_P_rows')
    values=[None]*P;status=[]
    for i,row in enumerate(rows):
        try:
            parsed=_real_array(row,(3,),name+'_'+str(i))
            if normal:
                norm=float(np.linalg.norm(parsed))
                if norm<=1e-12:raise _TaskInput(name+'_'+str(i)+'_zero_direction',degenerate=True)
                if abs(norm-1.)>UNIT_NORMAL_TOLERANCE:raise _TaskInput(name+'_'+str(i)+'_must_be_unit_direction')
            values[i]=parsed;status.append('known_used' if selected[i] else 'known_not_used')
        except _TaskInput as exc:
            if selected[i]:raise
            # No coefficient is inferred for inactive external geometry.
            status.append('unknown_not_used' if exc.unknown else 'invalid_not_used')
    return values,status

def build_fusion_tasks(*,patch_digit_p,normal_p,reference_point_p3,reference_normal_p3,selected_p,
                       previous_conditional_force5,caps5,force_cost,reference_continuity,balance_weight,
                       support_contact_floor,characteristic_length):
    """Generate one joint-QP task specification, not optimized forces.

    previous_conditional_force5 is the last accepted CONDITIONAL INTENT, not
    actual pressure. None initializes it from the currently selected group.
    Base prior is raw soft floor apportioned by observed normal shares; it is
    never clipped to cap/budget. Caps only fix the declared unit normalization.

    Geometry is acquisition/reference-frame geometry supplied by the caller,
    which owns synchronization. No pose/mass/friction/target identity is read.
    """
    metadata=dict(force_intent_not_prediction=True,conditional_group_identity_certified=False,
        physical_support_certified=False,nested_optimizer_used=False,extra_hard_floor_implemented=False,
        unit_normal_tolerance=UNIT_NORMAL_TOLERANCE,
        geometry_frame='caller-declared common acquisition/reference frame; no current-pose propagation',
        external_prior_zero_role='desired unloading intention, never an observation of zero force',
        external_zero_row_role='task inactive, never known zero geometry',
        center_role='arithmetic mean of selected public points, not center of mass')
    try:
        if patch_digit_p is None:raise _TaskInput('patch_digit_p_unknown',unknown=True)
        raw=np.asarray(patch_digit_p,dtype=object)
        if raw.ndim!=1:raise _TaskInput('patch_digit_p_requires_vector')
        P=len(raw)
        if P>MAX_PATCHES:raise _TaskInput('patch_count_exceeds_declared_domain',unknown=True)
        if any(isinstance(x,(bool,np.bool_)) or not isinstance(x,Integral) or not 0<=int(x)<5 for x in raw.flat):raise _TaskInput('patch_digit_p_requires_integer0to4')
        digit=np.asarray(patch_digit_p,int);normal_values=_real_array(normal_p,(P,),'normal_p')
        if np.any(normal_values<0):raise _TaskInput('normal_p_must_be_nonnegative')
        if selected_p is None:raise _TaskInput('selected_p_unknown',unknown=True)
        selected=np.asarray(selected_p)
        if selected.shape!=(P,) or P and selected.dtype.kind!='b':raise _TaskInput('selected_p_requires_explicit_bool_mask')
        selected=selected.astype(bool,copy=True)
        caps=_real_array(caps5,(5,),'caps5')
        if np.any(caps<0):raise _TaskInput('caps5_must_be_nonnegative')
        economy=_number(force_cost,'force_cost',positive=True);continuity=_number(reference_continuity,'reference_continuity')
        balance=_number(balance_weight,'balance_weight');floor=_number(support_contact_floor,'support_contact_floor');length=_number(characteristic_length,'characteristic_length',positive=True)
        previous=None if previous_conditional_force5 is None else _real_array(previous_conditional_force5,(5,),'previous_conditional_force5')
        if previous is not None and np.any(previous<0):raise _TaskInput('previous_conditional_force5_must_be_nonnegative')
        observed=np.bincount(digit[selected],weights=normal_values[selected],minlength=5).astype(float)
        if not np.isfinite(observed).all():raise _TaskInput('conditional_pressure_sum_overflow',unknown=True)
        enrolled=np.bincount(digit[selected],minlength=5)>0
        active=enrolled&(observed>0)
        metadata.update(active_digits5=active.tolist(),selected_patch_count=int(selected.sum()),observed_conditional_force5=observed.tolist())
        if np.any(enrolled&~active):
            return FusionTaskSpec('UNAVAILABLE_DEGENERATE_GROUP','selected_digit_has_zero_total_pressure_for_share',metadata=metadata)
        if not active[0] or int(active.sum())<3:
            return FusionTaskSpec('UNAVAILABLE_CONDITIONAL_GROUP','thumb_and_three_positive_pressure_digits_required',metadata=metadata)
        points,point_status=_geometry(reference_point_p3,P,selected,'reference_point_p3')
        normals,normal_status=_geometry(reference_normal_p3,P,selected,'reference_normal_p3',normal=True)
        previous_source='observed_conditional_group_initialization' if previous is None else 'last_accepted_conditional_force_intent'
        if previous is None:previous=observed.copy()
        center=np.mean(np.asarray([points[i] for i in np.flatnonzero(selected)]),axis=0)
        B=np.zeros((5,P));W=np.zeros((5,P));prior=np.zeros(P)
        for i in np.flatnonzero(selected):
            d=digit[i];B[d,i]=1.;prior[i]=floor*(normal_values[i]/observed[d])
            n=normals[i];moment=np.cross(points[i]-center,n)/length
            W[:,i]=np.r_[n[:2],moment]
        sf=np.sqrt(economy)/np.maximum(1.,caps[digit])
        if not np.isfinite(sf).all() or np.any(sf<=0):raise _TaskInput('intent_multiplier_not_representably_positive',unknown=True)
        rows=[];targets=[];labels=[]
        if continuity>0:
            normalization=np.sqrt(continuity)/np.maximum(1.,caps)
            block=np.zeros((5,21+P));block[:,21:]=normalization[:,None]*B
            rows.append(block);targets.append(normalization*previous)
            labels.extend('conditional_digit_force_intent_continuity_'+str(i) for i in range(5))
        common_force=max(1.,float(caps.mean()))
        if not np.isfinite(common_force):raise _TaskInput('common_force_normalization_overflow',unknown=True)
        if balance>0:
            block=np.zeros((5,21+P));block[:,21:]=np.sqrt(balance)/common_force*W
            rows.append(block);targets.append(np.zeros(5))
            labels.extend(('conditional_horizontal_Fx','conditional_horizontal_Fy','conditional_Mx_over_length','conditional_My_over_length','conditional_Mz_over_length'))
        E=np.vstack(rows) if rows else np.empty((0,21+P));b=np.concatenate(targets) if targets else np.empty(0)
        if not all(np.isfinite(x).all() for x in (prior,sf,E,b,center,B,W)):raise _TaskInput('nonfinite_constructed_task_rows',unknown=True)
        metadata.update(previous_source=previous_source,previous_conditional_force5=previous.tolist(),
            floor_preference5=(active.astype(float)*floor).tolist(),floor_preference_above_cap5=(active&(floor>caps)).tolist(),
            floor_was_cap_or_budget_retargeted=False,center_public_point_mean=center.tolist(),
            selected_reference_point_status=point_status,selected_reference_normal_status=normal_status,
            geometry_not_used_p=(~selected).tolist(),conditional_digit_sum_map=B.tolist(),unweighted_balance_map=W.tolist(),
            common_force_normalization_N=common_force,characteristic_length_m=length,
            weights=dict(force_cost=economy,reference_continuity=continuity,balance_weight=balance),
            zero_weight_rows_skipped=True,continuity_is_digit_sum_not_patch_identity=True,
            inactive_digit_continuity_rows_are_constant_cost_only=True,hard_caps_budget_load_owned_by_caller=True)
        return FusionTaskSpec('AVAILABLE_CONDITIONAL_FUSION_TASKS','public_conditional_group_soft_rows_only',
            _ro(prior),_ro(sf),_ro(E),_ro(b),tuple(labels),metadata)
    except _TaskInput as exc:
        status='UNAVAILABLE_UNKNOWN' if exc.unknown else 'UNAVAILABLE_DEGENERATE_GEOMETRY' if exc.degenerate else 'UNAVAILABLE_INPUT'
        return FusionTaskSpec(status,str(exc),metadata=metadata)
    except (ValueError,TypeError,OverflowError,FloatingPointError) as exc:
        return FusionTaskSpec('UNAVAILABLE_INPUT',str(exc),metadata=metadata)
