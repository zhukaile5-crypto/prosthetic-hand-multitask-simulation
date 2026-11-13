"""Conditional acquisition-time geometry from public skin and declared templates.

No object state/identity, material parameters, contact solver, dynamics, force
forecast, hidden normal/shear completion or closing-permission decision. One
finger is one correlated position-noise group regardless of patch multiplicity.
"""
from __future__ import annotations
from dataclasses import dataclass,field,replace
from numbers import Real
from typing import Optional
import numpy as np


def _number(x,name,positive=False):
    if isinstance(x,(bool,np.bool_)) or not isinstance(x,Real) or not np.isfinite(x):raise ValueError(name+' must be finite real')
    if positive and x<=0:raise ValueError(name+' must be positive')
    return float(x)

def _vector(x,name):
    a=np.asarray(x,float)
    if a.shape!=(3,) or not np.isfinite(a).all():raise ValueError(name+' must be finite3-vector')
    return a.copy()

def _ro(a):
    if a is None:return None
    value=np.asarray(a).copy();value.setflags(write=False);return value

def _json(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json(v) for v in value]
    return value

def _get(frame,key,default=None):return frame.get(key,default) if isinstance(frame,dict) else getattr(frame,key,default)


@dataclass(frozen=True)
class DeclaredShapeTemplate:
    kind:str
    radius_m:float
    half_segment_m:Optional[float]=None
    source:str='explicit caller-declared geometry, not online true object parameters'

    def __post_init__(self):
        if self.kind not in ('sphere','capsule'):raise ValueError('Only declared sphere/capsule template supported')
        object.__setattr__(self,'radius_m',_number(self.radius_m,'radius_m',True))
        if self.kind=='capsule':object.__setattr__(self,'half_segment_m',_number(self.half_segment_m,'half_segment_m',True))
        elif self.half_segment_m is not None:raise ValueError('Sphere has no segment coordinate')
        if not isinstance(self.source,str) or not self.source:raise ValueError('Template source must be explicit')


@dataclass(frozen=True)
class GeometryHypothesis:
    kind:str
    status:str
    inlier_indices:tuple
    independent_fingers:tuple
    conditional_center_reference:Optional[np.ndarray]=None
    conditional_position_noise_covariance3:Optional[np.ndarray]=None
    between_finger_geometry_dispersion3:Optional[np.ndarray]=None
    axis_reference:Optional[np.ndarray]=None
    axis_projector3:Optional[np.ndarray]=None
    center_line_origin_reference:Optional[np.ndarray]=None
    axial_center_interval_m:Optional[tuple]=None
    axial_center_identified:bool=False
    rotation_about_axis_observable:bool=False
    residual_max_m:Optional[float]=None
    residual_rms_m:Optional[float]=None
    historical_link_overlap_fingers:int=0
    diagnostics:dict=field(default_factory=dict)

    def to_dict(self):return _json(vars(self))


@dataclass(frozen=True)
class RelativeGeometrySnapshot:
    status:str
    current_time_s:float
    observation_time_s:Optional[float]
    age_s:Optional[float]
    template:DeclaredShapeTemplate
    hypotheses:tuple=()
    selected_index:Optional[int]=None
    grouping_unambiguous:bool=False
    center_reference:Optional[np.ndarray]=None
    axis_reference:Optional[np.ndarray]=None
    center_line_origin_reference:Optional[np.ndarray]=None
    axial_center_interval_m:Optional[tuple]=None
    unassigned_patch_indices:tuple=()
    rejected_patches:tuple=()
    diagnostics:dict=field(default_factory=dict)
    reference_frame:str='public acquisition-reference world, NOT current robot/object frame'
    object_identity_known:bool=False
    current_state_prediction_available:bool=False
    force_prediction_available:bool=False
    mechanical_closing_permission_modified:bool=False
    global_model_admitted:bool=False

    def to_dict(self):
        value=dict(vars(self));value['template']=vars(self.template);value['hypotheses']=[h.to_dict() for h in self.hypotheses]
        return _json(value)


class RelativeObjectBelief:
    def __init__(self,template:DeclaredShapeTemplate,*,fit_tolerance_m=.003,
                 minimum_axis_span_m=.006,cap_normal_projection_threshold=.15,
                 minimum_independent_fingers=2,max_age_s=.05,link_association_m=.003):
        if not isinstance(template,DeclaredShapeTemplate):raise ValueError('Explicit DeclaredShapeTemplate required')
        self.template=template
        self.tol=_number(fit_tolerance_m,'fit_tolerance_m',True)
        self.minimum_axis_span=_number(minimum_axis_span_m,'minimum_axis_span_m',True)
        self.cap_threshold=_number(cap_normal_projection_threshold,'cap normal threshold',True)
        if self.cap_threshold>=1:raise ValueError('Normal threshold must be below1')
        if isinstance(minimum_independent_fingers,bool) or not isinstance(minimum_independent_fingers,int) or not 1<=minimum_independent_fingers<=5:raise ValueError('Independent finger count1..5')
        self.min_fingers=minimum_independent_fingers;self.max_age=_number(max_age_s,'max_age_s',True)
        self.link_tol=_number(link_association_m,'link_association_m',True)
        self.last_sample=None;self.snapshot=None;self._last_packet_snapshot=None;self.history=[];self.unique_packet_updates=0

    def _empty(self,status,now,sample=None,rejected=(),extra=None):
        result=RelativeGeometrySnapshot(status,now,sample,None if sample is None else now-sample,self.template,
            rejected_patches=tuple(rejected),diagnostics=extra or {})
        self.snapshot=result;return result

    def _parse(self,frame,now):
        raw=_get(frame,'skin_patches',None)
        if raw is None:return [],[dict(index=None,reason='skin_patches_unavailable')]
        parsed=[];rejected=[]
        allowed={'digit','robot_body','link_point','link_normal','reference_position','reference_normal','normal','sample_time','age'}
        for index,p in enumerate(raw):
            try:
                if not isinstance(p,dict) or set(p)-allowed:raise ValueError('nonpublic_or_unknown_patch_fields')
                d=p['digit'];body=p['robot_body']
                if isinstance(d,bool) or not isinstance(d,(int,np.integer)) or not 0<=d<5:raise ValueError('digit_required0..4')
                if isinstance(body,bool) or not isinstance(body,(int,np.integer)):raise ValueError('robot_body_integer_required')
                s=_number(p['sample_time'],'sample_time');pos=_vector(p['reference_position'],'acquisition position');normal=_vector(p['reference_normal'],'acquisition normal')
                norm=np.linalg.norm(normal)
                if norm<1e-8:raise ValueError('normal_direction_unavailable')
                if s>now+1e-9:raise ValueError('future_acquisition_time')
                pressure=_number(p['normal'],'patch normal')
                if pressure<=0:raise ValueError('no_positive_observed_contact')
                # Link fields are public association evidence, never object IDs.
                link=_vector(p['link_point'],'link_point')
                parsed.append(dict(index=index,digit=int(d),body=int(body),link=link,p=pos,n=normal/norm,sample=s))
            except (ValueError,KeyError,TypeError) as exc:rejected.append(dict(index=index,reason=str(exc)))
        return parsed,rejected

    def _weights(self,patches,mask):
        # Each finger contributes total1. Patch pressure never dominates votes.
        weights=np.zeros(len(patches));digits=sorted({patches[i]['digit'] for i in np.flatnonzero(mask)})
        for digit in digits:
            idx=[i for i in np.flatnonzero(mask) if patches[i]['digit']==digit]
            weights[idx]=1./len(idx)
        weights/=max(1,len(digits))
        return weights,tuple(digits)

    def _history_overlap(self,patches,mask):
        digits=set()
        for i in np.flatnonzero(mask):
            p=patches[i]
            if any(p['digit']==a['digit'] and p['body']==a['body'] and np.linalg.norm(p['link']-a['link'])<=self.link_tol for a in self.history):digits.add(p['digit'])
        return len(digits)

    def _sphere(self,patches,q,sigma):
        hypotheses=[]
        for seed in q:
            mask=np.linalg.norm(q-seed,axis=1)<=self.tol
            for _ in range(3):
                w,digits=self._weights(patches,mask);center=w@q
                new=np.linalg.norm(q-center,axis=1)<=self.tol
                if np.array_equal(new,mask):break
                mask=new
            if not mask.any():continue
            w,digits=self._weights(patches,mask);center=w@q;res=np.linalg.norm(q-center,axis=1)
            if any(h.inlier_indices==tuple(patches[i]['index'] for i in np.flatnonzero(mask)) for h in hypotheses):continue
            by_finger=np.array([q[[i for i in np.flatnonzero(mask) if patches[i]['digit']==d]].mean(0) for d in digits])
            spread=None
            if len(digits)>1:spread=np.cov(by_finger,rowvar=False,ddof=1)
            covariance=None if sigma is None else np.eye(3)*sigma**2/len(digits)
            hypotheses.append(GeometryHypothesis('sphere','CENTER_CONDITIONAL_ON_SINGLE_SPHERE_GROUP',
                tuple(patches[i]['index'] for i in np.flatnonzero(mask)),digits,_ro(center),_ro(covariance),_ro(spread),
                residual_max_m=float(res[mask].max()),residual_rms_m=float(np.sqrt(w@(res*res))),
                historical_link_overlap_fingers=self._history_overlap(patches,mask),
                diagnostics=dict(orientation_observable=False,normal_uncertainty_modeled=False,
                    covariance_scope='position-only conditional sensor covariance; shared per-finger position noise, not total geometry error',
                    uncertainty_unknown_components=['normal error','template error','penetration/compliance','grouping'],independent_position_noise_groups=len(digits))))
        return hypotheses

    def _capsule(self,patches,q,sigma):
        if len(q)<2:return []
        candidates=[];length=self.template.half_segment_m
        # Direction seeds use only simultaneous acquisition-time offset loci.
        for i in range(len(q)):
            for j in range(i):
                delta=q[i]-q[j];span=np.linalg.norm(delta)
                if span<self.minimum_axis_span or span>2*length+2*self.tol:continue
                axis=delta/span;res=np.linalg.norm(np.cross(q-q[j],axis),axis=1);mask=res<=self.tol
                # A finite segment cannot explain arbitrarily long collinear groups.
                s=q@axis;possible=np.flatnonzero(mask);best=None
                for lower in s[possible]:
                    local=mask&(s>=lower-self.tol)&(s<=lower+2*length+self.tol)
                    _,digits=self._weights(patches,local)
                    if best is None or len(digits)>best[0]:best=(len(digits),local)
                if best is not None:candidates.append(best[1])
        hypotheses=[];seen=set()
        for mask in candidates:
            for _ in range(3):
                w,digits=self._weights(patches,mask);mean=w@q;centered=q-mean
                scatter=(centered*w[:,None]).T@centered
                values,vectors=np.linalg.eigh(scatter);axis=vectors[:,-1]
                if axis[np.argmax(abs(axis))]<0:axis=-axis
                origin=mean-axis*(mean@axis);res=np.linalg.norm(np.cross(q-origin,axis),axis=1)
                new=res<=self.tol
                if np.array_equal(new,mask):break
                mask=new
            key=tuple(np.flatnonzero(mask))
            if key in seen or not mask.any():continue
            seen.add(key);w,digits=self._weights(patches,mask);mean=w@q;centered=q-mean
            values,vectors=np.linalg.eigh((centered*w[:,None]).T@centered);axis=vectors[:,-1]
            if axis[np.argmax(abs(axis))]<0:axis=-axis
            origin=mean-axis*(mean@axis);projected=q@axis
            span=float(np.ptp(projected[mask]))
            if span<self.minimum_axis_span:continue
            low=float(projected[mask].max()-length-self.tol);high=float(projected[mask].min()+length+self.tol)
            if low>high:continue
            dot=np.array([p['n']@axis for p in patches]);cap=mask&(abs(dot)>self.cap_threshold)
            cap_centers=projected[cap]+np.sign(dot[cap])*length
            if len(cap_centers):
                low=max(low,float(cap_centers.max()-self.tol));high=min(high,float(cap_centers.min()+self.tol))
            if low>high:
                # Do not fabricate a center by clipping an inconsistent interval.
                continue
            full=bool(len(cap_centers) and high-low<=2*self.tol+1e-10)
            center=origin+axis*((low+high)/2) if full else None
            res=np.linalg.norm(np.cross(q-origin,axis),axis=1)
            hypotheses.append(GeometryHypothesis('capsule','PARTIAL_CAPSULE_AXIS_AND_CENTER_INTERVAL',
                tuple(patches[i]['index'] for i in np.flatnonzero(mask)),digits,_ro(center),None,None,
                _ro(axis),_ro(np.outer(axis,axis)),_ro(origin),(low,high),full,False,
                float(res[mask].max()),float(np.sqrt(w@(res*res))),self._history_overlap(patches,mask),
                diagnostics=dict(offset_locus_axial_span_m=span,scatter_eigenvalues_m2=values.tolist(),
                    cap_normal_evidence_count=int(cap.sum()),cap_normal_evidence_fingers=sorted({patches[i]['digit'] for i in np.flatnonzero(cap)}),
                    axis_sign_observable=False,rotation_about_axis_observable=False,
                    covariance_available=False,position_sigma_m=sigma,
                    center_interval_scope='geometric compatibility interval inflated by fit_tolerance, NOT probabilistic confidence interval',
                    normal_uncertainty_modeled=False,independent_position_noise_groups=len(digits))))
        if not hypotheses:
            # Offset loci may coincide at one spherical cap, or cover too little
            # axial span. No arbitrary principal direction is returned.
            w,digits=self._weights(patches,np.ones(len(q),bool))
            hypotheses.append(GeometryHypothesis('capsule','AXIS_UNOBSERVABLE_OR_TEMPLATE_INCONSISTENT',
                tuple(p['index'] for p in patches),digits,
                diagnostics=dict(axis_reference=None,center_reference=None,offset_locus_centroid_reference=(w@q).tolist(),
                    offset_locus_diameter_m=float(np.max(np.linalg.norm(q[:,None]-q[None,:],axis=2))),
                    reason='No nondegenerate compatible axis/finite-center interval from this simultaneous packet')))
        return hypotheses

    def update(self,frame):
        now=_number(_get(frame,'time',None),'frame time')
        patches,rejected=self._parse(frame,now)
        if not patches:return self._empty('OBSERVATION_UNAVAILABLE',now,rejected=rejected)
        newest=max(p['sample'] for p in patches)
        discarded=[dict(index=p['index'],reason='different_acquisition_packet_not_merged') for p in patches if abs(p['sample']-newest)>1e-9]
        patches=[p for p in patches if abs(p['sample']-newest)<=1e-9];rejected+=discarded
        if now-newest>self.max_age+1e-9:return self._empty('STALE_ACQUISITION',now,newest,rejected)
        if self.last_sample is not None and newest<self.last_sample-1e-9:
            return self._empty('OUT_OF_ORDER_ACQUISITION',now,newest,rejected)
        if self.last_sample is not None and abs(newest-self.last_sample)<=1e-9 and self._last_packet_snapshot is not None:
            self.snapshot=replace(self._last_packet_snapshot,current_time_s=now,age_s=now-newest,
                diagnostics={**self._last_packet_snapshot.diagnostics,'duplicate_packet_no_new_information':True})
            return self.snapshot
        specs=_get(frame,'sensor_characteristics',None);sigma=None
        if specs is not None and specs.get('position_sigma_m') is not None:
            sigma=_number(specs['position_sigma_m'],'position_sigma_m')
            if sigma<0:raise ValueError('position_sigma_m must be nonnegative')
        q=np.array([p['p']+self.template.radius_m*p['n'] for p in patches])
        hypotheses=self._sphere(patches,q,sigma) if self.template.kind=='sphere' else self._capsule(patches,q,sigma)
        hypotheses.sort(key=lambda h:(-len(h.independent_fingers),-h.historical_link_overlap_fingers,
            np.inf if h.residual_rms_m is None else h.residual_rms_m))
        # Merge only equivalent geometry estimates, not geometrically distinct bodies.
        unique=[]
        for h in hypotheses:
            equivalent=False
            for old in unique:
                if h.kind=='sphere' and h.conditional_center_reference is not None and old.conditional_center_reference is not None:
                    equivalent=np.linalg.norm(h.conditional_center_reference-old.conditional_center_reference)<=self.tol
                elif h.axis_reference is not None and old.axis_reference is not None:
                    compatible_intervals=max(h.axial_center_interval_m[0],old.axial_center_interval_m[0])<=min(h.axial_center_interval_m[1],old.axial_center_interval_m[1])+self.tol
                    equivalent=compatible_intervals and abs(h.axis_reference@old.axis_reference)>np.cos(np.deg2rad(5)) and np.linalg.norm(np.cross(h.center_line_origin_reference-old.center_line_origin_reference,old.axis_reference))<=self.tol
                if equivalent:break
            if not equivalent:unique.append(h)
        hypotheses=unique[:8]
        if not hypotheses:return self._empty('GEOMETRY_UNAVAILABLE',now,newest,rejected)
        best=hypotheses[0];independent=len(best.independent_fingers)
        ambiguous=any(len(h.independent_fingers)==independent for h in hypotheses[1:])
        known=independent>=self.min_fingers and not ambiguous and (best.kind=='sphere' or best.axis_reference is not None)
        status='CONDITIONAL_GEOMETRY' if known else ('AMBIGUOUS_GROUPING' if ambiguous else 'INSUFFICIENT_GEOMETRIC_EVIDENCE')
        assigned=set(best.inlier_indices);unassigned=tuple(p['index'] for p in patches if p['index'] not in assigned)
        self.unique_packet_updates+=1;self.last_sample=newest
        if known:self.history=[dict(digit=p['digit'],body=p['body'],link=p['link'].copy()) for p in patches if p['index'] in assigned]
        result=RelativeGeometrySnapshot(status,now,newest,now-newest,self.template,tuple(hypotheses),0,known,
            best.conditional_center_reference if known else None,best.axis_reference if known else None,
            best.center_line_origin_reference if known else None,best.axial_center_interval_m if known else None,
            unassigned,tuple(rejected),dict(unique_packet_updates=self.unique_packet_updates,
                independent_position_noise_groups=independent,patch_count=len(patches),
                current_packet_only=True,past_point_cloud_accumulated=False,
                history_role='robot-link association only ranks hypotheses; never object identity proof or unknown-to-known force/closing permission',
                geometry_assumption='Each inlier group belongs to one declared rigid shape; other bodies remain possible',
                tolerance_m=self.tol,tolerance_is_not_sensor_sigma_or_safety_bound=True,
                tolerance_source='fixed declared engineering prior; not tuned on evaluation truth',
                public_position_3sigma_m=None if sigma is None else 3*sigma,
                unquantified_geometry_error_components=['contact midpoint/soft penetration','template error','surface normal uncertainty'],
                normal_sign='public reference_normal points hand into external body; p+radius*n is sphere center or capsule-axis offset locus',
                unknowns=['object identity','current-time motion propagation','external mass/friction','template mismatch','normal covariance','contact compliance','object roll for capsule'],
                capsule_roll_gauge='unobservable but geometry-irrelevant under the declared rotationally symmetric template',
                sphere_orientation_observable=False))
        self.snapshot=result;self._last_packet_snapshot=result;return result
