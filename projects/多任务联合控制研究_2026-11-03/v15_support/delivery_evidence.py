"""OFFLINE EVALUATION ONLY: cached-contact delivery ledger, never a sensor.

Target identity and free-body state are permitted ONLY in this sidecar. Do not
attach this object, its arrays or summaries to SensorFrame/controller inputs.
No dynamics, forward/FK, model/data construction, sensor sample or RNG calls.
"""
from dataclasses import dataclass
from numbers import Integral, Real
import numpy as np
from .receiver_sensor import ReceiverMount

CAPTURE_PHASE='cached_contact_solve_at_integration_end_state'
CLOCK_TOLERANCE_S=1e-7
CATEGORIES=('receiver_top','receiver_otherface','floor','other_nonhand_world','hand','other_nonworld')

def _real(x,name,minimum=None):
    if isinstance(x,(bool,np.bool_)) or not isinstance(x,Real) or not np.isfinite(x):raise ValueError(name+' must be finite real, not bool/string')
    x=float(x)
    if minimum is not None and x<minimum:raise ValueError(name+' below minimum')
    return x

def _integer(x,name,minimum=0):
    if isinstance(x,(bool,np.bool_)) or not isinstance(x,Integral) or x<minimum:raise ValueError(name+' must be integer')
    return int(x)

def _array(x,shape,name):
    raw=np.asarray(x,dtype=object)
    if raw.shape!=shape or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,Real) for v in raw.flat):raise ValueError(name+' must contain finite real values with shape '+str(shape))
    a=np.asarray(x,dtype=float)
    if not np.isfinite(a).all():raise ValueError(name+' nonfinite')
    return np.frombuffer(a.tobytes(),dtype=np.float64).reshape(shape)

def _readonly(x):
    a=np.asarray(x,float)
    return np.frombuffer(a.tobytes(),dtype=np.float64).reshape(a.shape)

def _rotation(q):
    w,x,y,z=_array(q,(4,),'receiver geom quaternion')
    if abs(w*w+x*x+y*y+z*z-1.)>1e-7:raise ValueError('Receiver quaternion not normalized')
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def _ids(values,ngeom,name):
    if values is None:raise ValueError(name+' requires explicit geometry IDs')
    try:parsed=tuple(_integer(x,name) for x in values)
    except TypeError:raise ValueError(name+' requires geometry-ID sequence')
    if len(set(parsed))!=len(parsed) or any(x>=ngeom for x in parsed):raise ValueError(name+' duplicates or out of range')
    return frozenset(parsed)

@dataclass(frozen=True,slots=True)
class DeliveryInterval:
    interval_start_s:float
    interval_end_s:float
    contact_valid:bool
    object_state_valid:bool
    invalid_reasons:tuple
    target_normal_N:np.ndarray
    target_positive_contact_count:np.ndarray
    receiver_any_actor_top_normal_N:float
    receiver_any_actor_top_wrench6:np.ndarray
    object_free_qpos7_end:np.ndarray
    object_free_qvel6_end:np.ndarray
    capture_phase:str=CAPTURE_PHASE

class DeliveryLedger:
    """One episode's privileged OFFLINE evidence; no .to_public interface.

    All target-body geometries must be explicitly listed, matching the named
    free joint. hand_geom_ids is optional: if omitted, hand contacts cannot be
    identified and nonworld contacts are retained as other_nonworld, NOT world
    support. Other world means static worldbody geoms, not every non-target.
    """
    def __init__(self,model,*,target_geom_ids,object_free_joint_name='object_free',receiver_mount:ReceiverMount,
                 floor_geom_names=('floor',),hand_geom_ids=None,diagnostic_window_s=(12.,16.),contact_force_reader=None):
        if type(receiver_mount) is not ReceiverMount:raise TypeError('Explicit ReceiverMount required')
        if not isinstance(object_free_joint_name,str) or not object_free_joint_name:raise ValueError('Named object free joint required')
        if not isinstance(floor_geom_names,tuple) or not floor_geom_names or any(not isinstance(n,str) or not n for n in floor_geom_names) or len(set(floor_geom_names))!=len(floor_geom_names):raise ValueError('Explicit unique floor geom names required')
        if not isinstance(diagnostic_window_s,(tuple,list)) or len(diagnostic_window_s)!=2:raise ValueError('Fixed diagnostic (start,end) window required')
        w0=_real(diagnostic_window_s[0],'diagnostic start',0.);w1=_real(diagnostic_window_s[1],'diagnostic end',0.)
        if w1<=w0:raise ValueError('Diagnostic window must increase')
        self._model=model;self._mount=receiver_mount;self._window=(w0,w1);self._reader=contact_force_reader
        self._ngeom=_integer(model.ngeom,'model geometry count',1)
        self._targets=_ids(target_geom_ids,self._ngeom,'target_geom_ids')
        if not self._targets:raise ValueError('At least one target geom required')
        self._hands=_ids((),self._ngeom,'hand_geom_ids') if hand_geom_ids is None else _ids(hand_geom_ids,self._ngeom,'hand_geom_ids')
        self._hand_ids_explicit=hand_geom_ids is not None
        self._receiver=_integer(model.geom(receiver_mount.geom_name).id,'receiver geom')
        self._floors=_ids(tuple(model.geom(n).id for n in floor_geom_names),self._ngeom,'floor geometry IDs')
        excluded=_ids(tuple(model.geom(n).id for n in receiver_mount.excluded_geom_names),self._ngeom,'excluded mount geometry IDs')
        if self._receiver>=self._ngeom or self._receiver in excluded or self._receiver in self._floors or self._receiver in self._targets or self._receiver in self._hands or self._targets & self._hands or self._targets & self._floors or self._hands & self._floors:raise ValueError('Receiver/target/floor/hand geometry roles overlap')
        if not self._floors<=excluded:raise ValueError('Every declared floor must be excluded by receiver mount')
        self._body=np.asarray(model.geom_bodyid,dtype=int).copy()
        if self._body.shape!=(self._ngeom,) or np.any(self._body<0):raise ValueError('Invalid model geom-body mapping')
        if int(model.geom_type[self._receiver])!=6 or self._body[self._receiver]!=0:raise ValueError('Receiver must be explicit static world box top')
        if any(self._body[i]!=0 for i in self._floors):raise ValueError('Floor names must identify static world geometries')
        self._size=_array(model.geom_size[self._receiver],(3,),'receiver half sizes')
        if np.any(self._size<=0) or not np.allclose(receiver_mount.origin_in_geom_m,[0.,0.,self._size[2]],atol=1e-10,rtol=0):raise ValueError('Receiver origin must be static box top center')
        self._R=_rotation(model.geom_quat[self._receiver]);self._center=_array(model.geom_pos[self._receiver],(3,),'receiver center')
        self._origin=self._center+self._R@receiver_mount.origin_in_geom_m
        self._jid=_integer(model.joint(object_free_joint_name).id,'object free joint ID')
        if int(model.jnt_type[self._jid])!=0:raise ValueError('Named object joint is not mjJNT_FREE')
        self._qa=_integer(model.jnt_qposadr[self._jid],'object qpos address');self._va=_integer(model.jnt_dofadr[self._jid],'object qvel address')
        self._nq=_integer(model.nq,'model nq');self._nv=_integer(model.nv,'model nv')
        if self._qa+7>self._nq or self._va+6>self._nv:raise ValueError('Free-joint coordinate addresses outside model')
        self._object_body=_integer(model.jnt_bodyid[self._jid],'object joint body',1)
        complete=frozenset(int(i) for i in np.flatnonzero(self._body==self._object_body))
        if self._targets!=complete:raise ValueError('target_geom_ids must list ALL geoms of named free-joint body')
        self._binding=self._signature();self._binding_invalid=False;self._records=[];self._data=None

    def _signature(self):
        return (tuple((name,np.asarray(getattr(self._model,name)).tobytes()) for name in ('geom_type','geom_bodyid','geom_size','geom_pos','geom_quat','jnt_type','jnt_bodyid','jnt_qposadr','jnt_dofadr')),
            int(self._model.ngeom),int(self._model.nq),int(self._model.nv))

    def _top(self,c,g0,g1):
        """Same static-top coverage as receiver90ba; no sensor or RNG use."""
        Q=_array(np.asarray(c.frame).reshape(3,3),(3,3),'cached contact frame')
        if not np.allclose(Q@Q.T,np.eye(3),atol=1e-6,rtol=0) or np.linalg.det(Q)<0:raise ValueError('Invalid cached contact frame')
        point=_array(c.pos,(3,),'cached contact position');dist=_real(c.dist,'cached contact distance')
        sign=-1. if g0==self._receiver else 1.
        surface=point+sign*.5*dist*Q[0];outward=-sign*Q[0]
        local=self._R.T@(surface-self._center);nlocal=self._R.T@outward;tol=self._mount.top_plane_tolerance_m
        is_top=abs(local[2]-self._size[2])<=tol and np.all(abs(local[:2])<=self._size[:2]+tol) and nlocal[2]>=self._mount.normal_alignment_min_cos
        return bool(is_top),Q,point,sign

    def _contacts(self,data):
        force=np.zeros(len(CATEGORIES));counts=np.zeros(len(CATEGORIES));any_normal=0.;any_wrench=np.zeros(6)
        for index in range(_integer(data.ncon,'cached contact count')):
            c=data.contact[index];pair=np.asarray(c.geom,dtype=object)
            if pair.shape!=(2,):raise ValueError('Cached contact geom pair shape')
            g0=_integer(pair[0],'contact geom0');g1=_integer(pair[1],'contact geom1')
            if g0>=self._ngeom or g1>=self._ngeom:raise ValueError('Cached contact geom ID outside model')
            is_target=(g0 in self._targets)!=(g1 in self._targets)
            is_receiver=(g0==self._receiver)!=(g1==self._receiver)
            if not is_target and not is_receiver:continue
            top=False
            if is_receiver:top,Q,point,sign=self._top(c,g0,g1)
            raw=np.full(6,np.nan)
            if self._reader is None:
                import mujoco
                mujoco.mj_contactForce(self._model,data,index,raw)
            else:self._reader(self._model,data,index,raw)
            raw=_array(raw,(6,),'cached contact wrench')
            if raw[0]<0:raise ValueError('Negative engine normal force is unknown, not zero')
            normal=float(raw[0])
            if is_receiver and top:
                any_normal+=normal
                F=sign*(Q.T@raw[:3]);M=sign*(Q.T@raw[3:])+np.cross(point-self._origin,F)
                any_wrench+=np.r_[self._R.T@F,self._R.T@M]
            if is_target:
                other=g1 if g0 in self._targets else g0
                if other==self._receiver:category='receiver_top' if top else 'receiver_otherface'
                elif other in self._floors:category='floor'
                elif other in self._hands:category='hand'
                elif self._body[other]==0:category='other_nonhand_world'
                else:category='other_nonworld'
                j=CATEGORIES.index(category);force[j]+=normal;counts[j]+=int(normal>0.)
        if not np.isfinite(force).all() or not np.isfinite(any_normal) or not np.isfinite(any_wrench).all():raise ValueError('Cached contact sum overflow')
        return force,counts,any_normal,any_wrench

    def capture_cached_interval(self,data,start_s,end_s,*,phase=CAPTURE_PHASE):
        start=_real(start_s,'integration interval start',0.);end=_real(end_s,'integration interval end',0.)
        if end<=start:raise ValueError('Positive integration duration required')
        if phase!=CAPTURE_PHASE:raise ValueError('Explicit cached-contact/integration-end-state phase required')
        if self._records:
            last=self._records[-1]
            if start==last.interval_start_s and end==last.interval_end_s:
                if data is not self._data:raise ValueError('Cached duplicate must use same episode data')
                return last
            if start<last.interval_end_s:raise ValueError('Overlapping/rewound cached interval')
        if self._data is None:self._data=data
        elif data is not self._data:raise ValueError('One ledger may bind only one episode data object')
        reasons=[];contact_valid=True;state_valid=True
        force=np.full(len(CATEGORIES),np.nan);counts=np.full(len(CATEGORIES),np.nan);any_normal=float('nan');any_wrench=np.full(6,np.nan)
        q=np.full(7,np.nan);v=np.full(6,np.nan)
        try:self._binding_invalid=self._binding_invalid or self._signature()!=self._binding
        except (ValueError,TypeError,IndexError,AttributeError):self._binding_invalid=True
        try:
            if abs(_real(data.time,'engine end-state time',0.)-end)>CLOCK_TOLERANCE_S:raise ValueError('engine_end_clock_mismatch')
        except (ValueError,TypeError,AttributeError) as exc:reasons.append(str(exc));contact_valid=False;state_valid=False
        if self._binding_invalid:reasons.append('model_geometry_or_coordinate_definition_changed');contact_valid=False;state_valid=False
        if contact_valid:
            try:force,counts,any_normal,any_wrench=self._contacts(data)
            except (ValueError,TypeError,IndexError,AttributeError,RuntimeError,OverflowError) as exc:
                contact_valid=False;reasons.append('cached_contact_unavailable:'+str(exc))
        if state_valid:
            try:
                q=_array(data.qpos[self._qa:self._qa+7],(7,),'object free qpos end');v=_array(data.qvel[self._va:self._va+6],(6,),'object free qvel end')
                qnorm=float(np.hypot.reduce(q[3:]));linear=float(np.hypot.reduce(v[:3]));angular=float(np.hypot.reduce(v[3:]))
                if not np.isfinite([qnorm,linear,angular]).all() or abs(qnorm-1.)>1e-5:raise ValueError('object quaternion or velocity norm unavailable')
            except (ValueError,TypeError,IndexError,AttributeError,OverflowError) as exc:
                state_valid=False;reasons.append('object_end_state_unavailable:'+str(exc))
        if not contact_valid:force=np.full(len(CATEGORIES),np.nan);counts=np.full(len(CATEGORIES),np.nan);any_normal=float('nan');any_wrench=np.full(6,np.nan)
        if not state_valid:q=np.full(7,np.nan);v=np.full(6,np.nan)
        record=DeliveryInterval(start,end,contact_valid,state_valid,tuple(reasons),_readonly(force),_readonly(counts),any_normal,_readonly(any_wrench),_readonly(q),_readonly(v))
        self._records.append(record);return record

    def arrays(self):
        n=len(self._records);stack=lambda key,width:np.stack([getattr(r,key) for r in self._records]) if n else np.empty((0,width))
        a=dict(interval_start_s=np.array([r.interval_start_s for r in self._records]),interval_end_s=np.array([r.interval_end_s for r in self._records]),
            interval_duration_s=np.array([r.interval_end_s-r.interval_start_s for r in self._records]),contact_valid=np.array([r.contact_valid for r in self._records],bool),
            object_state_valid=np.array([r.object_state_valid for r in self._records],bool),invalid_reason=np.array([';'.join(r.invalid_reasons) for r in self._records],str),
            receiver_any_actor_top_normal_N=np.array([r.receiver_any_actor_top_normal_N for r in self._records]),receiver_any_actor_top_wrench6=stack('receiver_any_actor_top_wrench6',6),
            object_free_qpos7_end=stack('object_free_qpos7_end',7),object_free_qvel6_end=stack('object_free_qvel6_end',6))
        force=stack('target_normal_N',len(CATEGORIES));counts=stack('target_positive_contact_count',len(CATEGORIES))
        for j,name in enumerate(CATEGORIES):a['target_'+name+'_normal_N']=force[:,j].copy();a['target_'+name+'_positive_contact_count']=counts[:,j].copy()
        return a

    def _state(self,record):
        if record is None or not record.object_state_valid:return dict(available=False,position_m=None,quaternion_wxyz=None,linear_speed_m_s=None,angular_speed_rad_s=None,qpos7=None,qvel6=None,end_time_s=None if record is None else record.interval_end_s)
        q=record.object_free_qpos7_end;v=record.object_free_qvel6_end
        return dict(available=True,end_time_s=record.interval_end_s,position_m=q[:3].tolist(),quaternion_wxyz=q[3:].tolist(),
            linear_speed_m_s=float(np.hypot.reduce(v[:3])),angular_speed_rad_s=float(np.hypot.reduce(v[3:])),qpos7=q.tolist(),qvel6=v.tolist())

    def summary(self):
        w0,w1=self._window;duration=w1-w0
        rows=[(r,max(0.,min(w1,r.interval_end_s)-max(w0,r.interval_start_s))) for r in self._records]
        rows=[(r,dt) for r,dt in rows if dt>0.];covered=sum(dt for _,dt in rows);valid_contact=sum(dt for r,dt in rows if r.contact_valid)
        coverage_complete=abs(covered-duration)<=CLOCK_TOLERANCE_S
        contact_complete=coverage_complete and all(r.contact_valid for r,_ in rows)
        def stats(value):
            valid=[(r,dt,float(value(r))) for r,dt in rows if r.contact_valid]
            peak=max((x for _,_,x in valid),default=None);impulse=sum(dt*x for _,dt,x in valid) if valid else None;positive=sum(dt for _,dt,x in valid if x>0.) if valid else None
            first=next((r for r,_,x in valid if x>0.),None)
            return dict(available=contact_complete,peak_N=peak if contact_complete else None,impulse_Ns=impulse if contact_complete else None,positive_contact_duration_s=positive if contact_complete else None,
                observed_valid_peak_N=peak,observed_valid_impulse_Ns=impulse,observed_valid_positive_contact_duration_s=positive,
                observed_positive_contact=True if first is not None else False if contact_complete else None,
                first_positive_interval_s=None if first is None else [max(w0,first.interval_start_s),min(w1,first.interval_end_s)])
        by={name:stats(lambda r,j=j:r.target_normal_N[j]) for j,name in enumerate(CATEGORIES)}
        terminal=next((r for r in reversed(self._records) if abs(r.interval_end_s-w1)<=CLOCK_TOLERANCE_S),None)
        return dict(schema='delivery-cached-contact-offline-evaluation-v1',evaluation_only=True,controller_input_permitted=False,
            capture_phase=CAPTURE_PHASE,contact_time_role='cached integration constraint-solve force/geometry; interval start/end labels, not continuous-time contact certificate',
            object_state_time_role='free-joint qpos7/qvel6 at integration END; not guaranteed same instantaneous geometry as cached contact solve',
            diagnostic_window_s=[w0,w1],records=len(self._records),window_interval_records=len(rows),window_covered_duration_s=covered,
            window_coverage_complete=coverage_complete,coverage_clock_tolerance_s=CLOCK_TOLERANCE_S,
            window_uncovered_duration_s=max(0.,duration-covered),window_contact_valid_duration_s=valid_contact,window_contact_evidence_available=contact_complete,
            target_contact_classes=by,receiver_any_actor_top=stats(lambda r:r.receiver_any_actor_top_normal_N),
            floor_event=by['floor']['observed_positive_contact'],window_terminal_object_state=self._state(terminal),last_recorded_object_state=self._state(self._records[-1] if self._records else None),
            categories_mutually_exclusive_for_each_target_contact=True,receiver_any_actor_top_is_separate_overlapping_total=True,
            normal_event_threshold_N=0.,integration_rule='cached scalar force times clipped interval duration; no threshold, resampling or zero-filled missing interval',
            hand_geometry_ids_explicit=self._hand_ids_explicit,other_world_definition='geom_bodyid==0 excluding receiver/floor/explicit hand; dynamic nonworld tracked separately',
            receiver_coverage=self._mount.coverage,receiver_geom_name=self._mount.geom_name,target_geom_ids=sorted(self._targets),
            object_free_joint_id=self._jid,object_qpos_address=self._qa,object_qvel_address=self._va,
            model_data_pair_requires_outer_runtime_binding=True,delivery_success_redefined=False,receiver_contact_proves_target_delivery=False,physical_safety_certified=False)
