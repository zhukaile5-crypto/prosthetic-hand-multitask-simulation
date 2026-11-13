"""Conditional task SDF intervals from partial public object geometry.

Only equal-acquisition-time queries. Intervals quantify the unresolved axial
center coordinate of the supplied geometry hypothesis, NOT sensor/model error,
contact force, stiffness, identity, current-time motion, or a safety certificate.
"""
from __future__ import annotations
from dataclasses import dataclass,field
from numbers import Real
from typing import Optional
import numpy as np


def _get(x,key,default=None):return x.get(key,default) if isinstance(x,dict) else getattr(x,key,default)
def _number(x,name):
    if isinstance(x,(bool,np.bool_)) or not isinstance(x,Real) or not np.isfinite(x):raise ValueError(name+' must be finite real')
    return float(x)
def _vector(x,name):
    a=np.asarray(x,float)
    if a.shape!=(3,) or not np.isfinite(a).all():raise ValueError(name+' must be a finite3-vector')
    return a.copy()
def _ro(x):
    if x is None:return None
    a=np.asarray(x).copy();a.setflags(write=False);return a
def _json(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [_json(v) for v in x]
    return x


@dataclass(frozen=True)
class SignedGapInterval:
    status:str
    reasons:tuple=()
    observation_time_s:Optional[float]=None
    query_time_s:Optional[float]=None
    point_reference:Optional[np.ndarray]=None
    gap_lower_m:Optional[float]=None
    gap_upper_m:Optional[float]=None
    all_centers_cylinder:Optional[bool]=None
    outward_normal_reference:Optional[np.ndarray]=None
    inward_contact_normal_reference:Optional[np.ndarray]=None
    center_coordinate_for_min_gap_m:Optional[float]=None
    center_coordinate_for_max_gap_m:Optional[float]=None
    diagnostics:dict=field(default_factory=dict)
    reference_frame:str='same public acquisition-reference world as supplied hypothesis'
    geometric_interval_only:bool=True
    estimation_error_not_included:bool=True
    object_identity_known:bool=False
    current_state_prediction_available:bool=False
    pressure_prediction_available:bool=False
    contact_force_gradient_available:bool=False
    mechanical_closing_permission_modified:bool=False
    physical_gap_certified:bool=False
    global_model_admitted:bool=False

    @property
    def available(self):return self.status=='AVAILABLE_CONDITIONAL_GEOMETRIC_INTERVAL'
    @property
    def normal_available(self):return self.outward_normal_reference is not None
    def to_dict(self):return _json(vars(self))


def _unavailable(reason,*,obs=None,query=None,point=None):
    return SignedGapInterval('UNAVAILABLE',(reason,),obs,query,_ro(point))


def query_hypothesis_gap(template,hypothesis,point_reference,*,observation_time_s,sample_time_s):
    """Explicit conditional hypothesis query; caller owns grouping uncertainty.

    The normal is the outward spatial SDF direction, not a force derivative.
    Every axis-line query has normalNone by conservative degeneracy policy.
    """
    try:
        now=_number(sample_time_s,'query acquisition time');observed=_number(observation_time_s,'geometry acquisition time')
        point=_vector(point_reference,'point_reference')
    except (ValueError,TypeError) as exc:return _unavailable(str(exc))
    if abs(now-observed)>1e-9:return _unavailable('query_time_differs_from_geometry_acquisition_time',obs=observed,query=now,point=point)
    try:
        kind=_get(template,'kind');r=_number(_get(template,'radius_m'),'declared radius')
        if r<=0:raise ValueError('declared_radius_must_be_positive')
        if kind=='sphere':
            center=_vector(_get(hypothesis,'conditional_center_reference'),'conditional sphere center')
            delta=point-center;distance=float(np.linalg.norm(delta));gap=distance-r
            outward=None if distance<=1e-12 else delta/distance
            return SignedGapInterval('AVAILABLE_CONDITIONAL_GEOMETRIC_INTERVAL',(),observed,now,_ro(point),gap,gap,None,
                _ro(outward),_ro(None if outward is None else -outward),None,None,
                dict(kind=kind,template_source=_get(template,'source'),normal_undefined_reason='sphere_center' if outward is None else None,
                    normal_sign='outward body-to-query; inward_contact_normal is its negative',
                    unbounded_unknown_errors=['estimated center','template','normal/acquisition geometry','contact midpoint/compliance']))
        if kind!='capsule':raise ValueError('only_declared_sphere_or_capsule_supported')
        length=_number(_get(template,'half_segment_m'),'declared halfsegment')
        if length<=0:raise ValueError('declared_halfsegment_must_be_positive')
        axis=_vector(_get(hypothesis,'axis_reference'),'conditional capsule axis')
        if abs(np.linalg.norm(axis)-1)>1e-8:raise ValueError('axis_must_be_unit_without_reinterpreting_center_units')
        origin=_vector(_get(hypothesis,'center_line_origin_reference'),'conditional center line origin')
        interval=np.asarray(_get(hypothesis,'axial_center_interval_m'),float)
        if interval.shape!=(2,) or not np.isfinite(interval).all() or interval[0]>interval[1]:raise ValueError('ordered_finite_axial_center_interval_required')
        lo,hi=map(float,interval)
        # An alternative origin on the same line is allowed only with its own
        # matching relative interval. Convert it to canonical o_perp first.
        origin_shift=float(origin@axis);canonical_origin=origin-origin_shift*axis
        lo+=origin_shift;hi+=origin_shift
        delta=point-canonical_origin;s=float(axis@delta);radial=delta-s*axis;rho=float(np.linalg.norm(radial))
        nearest=float(np.clip(s,lo,hi));farthest=lo if abs(s-lo)>=abs(s-hi) else hi
        min_axial=max(abs(s-nearest)-length,0.)
        max_axial=max(abs(s-farthest)-length,0.)
        lower=float(np.hypot(rho,min_axial)-r);upper=float(np.hypot(rho,max_axial)-r)
        cylinder=bool(hi-length<=s<=lo+length)
        outward=None;normal_reason=None
        if rho<=1e-12:normal_reason='axis_line_degeneracy; no artificial radial direction'
        elif cylinder:outward=radial/rho
        elif hi==lo:
            axial=s-(lo+np.clip(s-lo,-length,length))
            normal=radial+axis*axial;outward=normal/np.linalg.norm(normal)
        else:normal_reason='normal depends on unresolved axial center; not zero-filled'
        return SignedGapInterval('AVAILABLE_CONDITIONAL_GEOMETRIC_INTERVAL',(),observed,now,_ro(point),lower,upper,cylinder,
            _ro(outward),_ro(None if outward is None else -outward),nearest,farthest,
            dict(kind=kind,template_source=_get(template,'source'),radius_m=r,half_segment_m=length,
                axial_coordinate_s_m=s,radial_distance_rho_m=rho,canonical_line_origin_reference=canonical_origin.tolist(),
                canonical_center_interval_m=[lo,hi],origin_axial_shift_removed_m=origin_shift,
                min_axial_offset_m=min_axial,max_axial_offset_m=max_axial,
                all_centers_cylinder_band_m=[hi-length,lo+length],normal_undefined_reason=normal_reason,
                normal_sign='outward body-to-query; inward_contact_normal is its negative',
                geometric_spatial_normal_not_force_gradient=True,
                unresolved_roll='symmetry gauge; irrelevant to declared capsule SDF',
                unbounded_unknown_errors=['estimated axis','estimated perpendicular center line','template','normal/acquisition geometry','contact midpoint/compliance']))
    except (ValueError,TypeError,IndexError) as exc:return _unavailable(str(exc),obs=observed,query=now,point=point)


def query_signed_gap(snapshot,point_reference,*,sample_time_s):
    """Query a selected unambiguous partial belief, never its current-time proxy."""
    if not bool(_get(snapshot,'grouping_unambiguous',False)):
        return _unavailable('unambiguous_conditional_group_required',obs=_get(snapshot,'observation_time_s'))
    hypotheses=_get(snapshot,'hypotheses',());index=_get(snapshot,'selected_index')
    if isinstance(index,bool) or not isinstance(index,(int,np.integer)) or not 0<=index<len(hypotheses):
        return _unavailable('selected_geometry_hypothesis_unavailable',obs=_get(snapshot,'observation_time_s'))
    return query_hypothesis_gap(_get(snapshot,'template'),hypotheses[index],point_reference,
        observation_time_s=_get(snapshot,'observation_time_s'),sample_time_s=sample_time_s)


class TaskContactGeometry:
    """Stateless convenience adapter; it does not infer force or advance belief."""
    @staticmethod
    def query(snapshot,point_reference,*,sample_time_s):
        return query_signed_gap(snapshot,point_reference,sample_time_s=sample_time_s)
