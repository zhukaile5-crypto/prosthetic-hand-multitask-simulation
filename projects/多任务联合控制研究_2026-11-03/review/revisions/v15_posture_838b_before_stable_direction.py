"""Soft holding hand-manifold and sampled native-effort history task rows.

No optimizer, model, writer or force predictor. The caller admits each finger's
native reference map as affine over its complete candidate interval and at p0.
Reference units are normalized by the known control spans. Unknown branches
are inactive tasks, not invented zero inverse gains. Release must not call this
holding-only generator. Prior model effort is not an independent torque sensor.
"""
from dataclasses import dataclass
from numbers import Integral,Real
import numpy as np

GROUPS=((0,5),(5,8),(8,11),(11,14),(14,18))

def _array(value,shape,name):
    if value is None:raise ValueError(name+'_unknown')
    raw=np.asarray(value,dtype=object)
    if raw.shape!=shape or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,Real) for x in raw.flat):raise ValueError(name+'_requires_real_'+str(shape))
    a=np.asarray(value,float).copy()
    if not np.isfinite(a).all():raise ValueError(name+'_nonfinite_unknown')
    return a

def _number(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value) or value<0:raise ValueError(name+'_requires_nonnegative_clock')
    return float(value)

def _ro(value):
    a=np.asarray(value,float).copy();a.setflags(write=False);return a

@dataclass(frozen=True)
class NativeHoldingTasks:
    status:str
    reason:str
    matrix:np.ndarray
    target:np.ndarray
    labels:tuple
    metadata:dict
    @property
    def available(self):return self.status=='AVAILABLE_HOLDING_NATIVE_TASKS'
    @property
    def extra_task_rows(self):return dict(matrix=self.matrix,target=self.target)

def build_holding_native_tasks(*,patch_count,time_s,p0_21,delta_scale21,nominal_reference21,grasp_anchor21,
        native_reference_gain21,known_affine21,control_range21,open18,closed18,
        previous_model_effort21=None,previous_model_time_s=None):
    if isinstance(patch_count,(bool,np.bool_)) or not isinstance(patch_count,Integral) or not 0<=patch_count<=128:raise ValueError('patch_count_requires_integer0to128')
    n=21+int(patch_count);rows=[];targets=[];labels=[]
    meta=dict(scope='holding_only_current_native_affine_soft_reference_structure_and_past_sample_model_effort',
        physical_future_pose_certified=False,pressure_prediction_available=False,task_coefficients=dict(hand_shape=1.,native_effort_history=1.),
        history_role='previous_public_snapshot_model_computed_internal_effort_not_independent_torque_measurement',
        hand_anchor_role='actual21_reference_at_first_holding_takeover_not_legacy_alpha',
        shape_role='per_finger_soft_complement_to_declared_OPEN_CLOSED_direction_in_control_span_units',inactive_shape_groups=[])
    now=_number(time_s,'time_s')
    p0=_array(p0_21,(21,),'p0_21');sd=_array(delta_scale21,(21,),'delta_scale21')
    if np.any(sd<=0):raise ValueError('positive_delta_unit_multipliers_required')
    nominal=_array(nominal_reference21,(21,),'nominal_reference21')
    gain=_array(native_reference_gain21,(21,),'native_reference_gain21')
    flags=np.asarray(known_affine21)
    if flags.shape!=(21,) or flags.dtype.kind!='b':raise ValueError('explicit_boolean_known_affine21_required')
    ranges=_array(control_range21,(21,2),'control_range21');opened=_array(open18,(18,),'open18');closed=_array(closed18,(18,),'closed18')
    if np.any(ranges[:,0]>ranges[:,1]):raise ValueError('unordered_control_range')
    if grasp_anchor21 is None:
        meta['shape_task_status']='actual_grasp_anchor_unknown_inactive'
    else:
        anchor=_array(grasp_anchor21,(21,),'grasp_anchor21');meta['shape_task_status']='per_group_admission'
        for digit,(start,end) in enumerate(GROUPS):
            span=ranges[start:end,1]-ranges[start:end,0]
            if not flags[start:end].all() or np.any(gain[start:end]==0.) or np.any(span<=0):
                meta['inactive_shape_groups'].append(dict(digit=digit,reason='native_inverse_branch_or_control_span_not_known'));continue
            direction=(closed[start:end]-opened[start:end])/span
            norm2=float(direction@direction)
            if not np.isfinite(norm2):raise ValueError('hand_template_normalization_overflow')
            projection=np.eye(end-start) if norm2==0 else np.eye(end-start)-np.outer(direction,direction)/norm2
            reference_rows=projection/span[None,:]
            block=np.zeros((end-start,n));block[:,start:end]=reference_rows/gain[None,start:end]
            target=reference_rows@(anchor[start:end]-nominal[start:end])
            if not np.isfinite(block).all() or not np.isfinite(target).all():raise ValueError('unrepresentable_native_inverse_shape_task')
            rows.append(block);targets.append(target);labels.extend(f'grasp_shape_digit{digit}_component{k}' for k in range(end-start))
    if previous_model_effort21 is None:
        if previous_model_time_s is not None:raise ValueError('history_clock_without_model_sample')
        meta['history_task_status']='no_past_public_model_sample_inactive'
    else:
        previous=_array(previous_model_effort21,(21,),'previous_model_effort21');past=_number(previous_model_time_s,'previous_model_time_s')
        if past>now:raise ValueError('future_model_history_rejected')
        if past==now:meta['history_task_status']='same_clock_sample_not_a_past_history_task'
        else:
            block=np.zeros((21,n));block[:,:21]=np.diag(sd);target=sd*(previous-p0)
            if not np.isfinite(target).all():raise ValueError('model_history_target_overflow')
            rows.append(block);targets.append(target);labels.extend(f'native_effort_history_{i}' for i in range(21))
            meta.update(history_task_status='past_model_sample_active',history_sample_time_s=past,history_elapsed_s=now-past)
    E=np.vstack(rows) if rows else np.empty((0,n));b=np.concatenate(targets) if targets else np.empty(0)
    meta.update(rows=len(b),columns=n,physical_reference_gain_not_contact_force_gradient=True)
    return NativeHoldingTasks('AVAILABLE_HOLDING_NATIVE_TASKS','known_tasks_only_unknown_groups_explicitly_inactive',_ro(E),_ro(b),tuple(labels),meta)
