"""Contact unloading progress, observed motion, and clearance posture tasks.

Progress is normal withdrawal weighted by observed pressure, rather than a
projection onto a fixed joint OPEN path. It is a virtual task in the declared
reference-to-length map, not a prediction of contact work, force or safety.
The frozen feasible lexicographic solver supplies only linear-problem evidence.
"""
from dataclasses import replace
from numbers import Real
import numpy as np
from v16_release_geometry.hierarchy import Level,solve_hierarchy


def _array(value,shape,name):
    raw=np.asarray(value,dtype=object)
    if raw.shape!=shape or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,Real) for x in raw.flat):
        raise ValueError(name+': explicit numeric shape required')
    data=np.asarray(value,float).copy()
    if not np.isfinite(data).all():raise ValueError(name+': finite values required')
    return data


def _common_scale(matrix,lo,hi,target):
    """One positive row-block rescaling, hence no change in the block optimum."""
    magnitude=np.max(np.sum(abs(matrix)*np.maximum(abs(lo),abs(hi))[None,:],axis=1),initial=0.)
    magnitude=max(float(magnitude),float(np.max(abs(target),initial=0.)))
    if not np.isfinite(magnitude):raise FloatingPointError('Task scale not representable')
    return magnitude if magnitude>0 else 1.


def solve_contact_unloading(tasks,*,reference_step_scale21,motion_task=None):
    """Bounds/noninward -> normal unloading -> observed motion -> tangent -> pose.

    Observed patch pressures weight the normal-progress row globally. Splitting
    one identical patch with conserved pressure does not change this objective.
    A missing motion observation removes its layer; zero remains a desired
    low-tangent task, never a fabricated measured external velocity.
    """
    scale=_array(reference_step_scale21,(21,),'positive reference step units')
    if np.any(scale<=0):raise ValueError('Positive per-reference scales required')
    if tasks.normal_matrix_m_per_reference is None:return None
    raw=np.asarray(tasks.normal_matrix_m_per_reference)
    if raw.ndim!=2 or raw.shape[1]!=21 or len(raw)==0:raise ValueError('Positive observed contact rows required')
    count=len(raw)
    normal=_array(raw,(count,21),'normal geometric rows')
    offset=_array(tasks.normal_offset_m,(count,),'old-plane inequality offsets')
    lo=_array(tasks.step_lower21,(21,),'actual lower step')/scale
    hi=_array(tasks.step_upper21,(21,),'actual upper step')/scale
    if np.any(lo>0) or np.any(hi<0):raise ValueError('Current zero step must remain a genuine feasible starting point')
    patches=tasks.diagnostics.get('patches')
    if not isinstance(patches,list) or len(patches)!=count:raise ValueError('Explicit same-order observed pressure provenance required')
    pressure=_array([p['observed_normal_N'] for p in patches],(count,),'observed normal pressures')
    if np.any(pressure<=0):raise ValueError('Positive pressure rows required, not fabricated contact')
    largest=float(np.max(pressure));relative=pressure/largest;relative_sum=float(relative.sum())
    weight=relative/relative_sum
    if np.any(weight<=0):raise FloatingPointError('Positive pressure weights underflowed')
    c=normal*scale[None,:]
    outward=-weight@c
    # This is a box upper bound on progress, not an assumed feasible point.
    # Because the noninward feasible set contains0, its attainable maximum is
    # nonnegative. Minimizing distance to this bound finds that maximum.
    maximum=float(np.sum(np.where(outward>=0,outward*hi,outward*lo)))
    if maximum<0 or not np.isfinite(maximum):raise FloatingPointError('Unrepresentable normal-progress upper bound')
    progress_scale=max(float(np.max(abs(outward),initial=0.)),abs(maximum))
    if progress_scale==0:progress_scale=1.
    levels=[Level('pressure_weighted_normal_withdrawal',outward[None,:]/progress_scale,np.array([maximum/progress_scale]))]
    motion_meta=dict(available=False,reason='public_motion_task_unavailable_not_zero_motion')
    if motion_task is not None:
        m=np.asarray(motion_task.matrix_Mx21)
        if m.ndim!=2 or m.shape[1]!=21 or not 1<=len(m)<=10:
            raise ValueError('At most two measured motion rows per finger required')
        m=_array(m,m.shape,'motion rows')
        b=_array(motion_task.target_M,(len(m),),'motion reference increment target')
        now=tasks.diagnostics.get('current_time_s')
        stamp=motion_task.time_s
        if (isinstance(now,(bool,np.bool_)) or not isinstance(now,Real) or not np.isfinite(now) or
                isinstance(stamp,(bool,np.bool_)) or not isinstance(stamp,Real) or not np.isfinite(stamp) or
                now<0 or abs(float(now)-float(stamp))>1e-9):
            raise ValueError('Motion and geometry public snapshot clock unavailable/mismatched')
        ms=m*scale[None,:];s=_common_scale(ms,lo,hi,b)
        levels.append(Level('observed_external_contact_motion_correction',ms/s,b/s))
        motion_meta=dict(available=True,rows=len(m),metadata=motion_task.metadata)
    tangent=np.asarray(tasks.tangent_matrix_m_per_reference)
    if tangent.ndim!=2 or tangent.shape[1]!=21:raise ValueError('Explicit tangent task required')
    tangent=_array(tangent,tangent.shape,'geometric low tangent')*scale[None,:]
    tangent_target=_array(tasks.tangent_target_m,(len(tangent),),'desired tangent increment')
    ts=_common_scale(tangent,lo,hi,tangent_target)
    levels.append(Level('low_unnecessary_tangent_increment',tangent/ts,tangent_target/ts))
    nominal=np.clip(_array(tasks.open_target_increment21,(21,),'canonical clearance preference')/scale,lo,hi)
    levels.append(Level('canonical_clearance_posture',np.eye(21),nominal))
    result=solve_hierarchy(lower=lo,upper=hi,levels=tuple(levels),initial=np.zeros(21),
        inequality_matrix=c,inequality_upper=-offset)
    delta=result.x*scale
    progress=float(outward@result.x)
    # Multiplication by total pressure restores normal virtual work units when
    # representable. The normalized physical task remains recorded either way.
    total_pressure=largest*relative_sum
    work=progress*total_pressure
    diag=dict(result.diagnostics,task_hierarchy=[l.name for l in levels],
        declared_primary='maximize_pressure_weighted_withdrawal_in_declared_geometric_proposal',
        observed_pressure_weights=weight.tolist(),normal_progress_box_upper_m=maximum,
        normal_progress_achieved_m=progress,
        normal_virtual_unloading_work_J=float(work) if np.isfinite(work) else None,
        normal_virtual_unloading_work_role='observed-normal-load times proposed old-plane displacement; not actual contact work',
        actual_inward_proposals_m=(normal@delta+offset).tolist(),
        observed_motion_task=motion_meta,
        canonical_goal_is_not_primary_progress=True,
        force_prediction_available=False,object_motion_predicted=False,placement_certified=False)
    return replace(result,diagnostics=diag)
