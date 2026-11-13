"""Public, robot-only bridge to V16 evidence and geometry task generators.

No plant/live MjData, object state/identity/parameters, sensor acquisition, RNG,
engine writer, forward, integration or contact-force query is accepted here.
Private known-robot FK/transmission arithmetic is used only for proposals.
"""
from collections import deque
from dataclasses import replace
from numbers import Real
import json
import numpy as np
import mujoco

from v13_a1.codec import Reference21, ReferenceActor21, REFERENCE_AXES21
from v11.robot_servo_effort import RobotServoEffort
from v10.robot_contact_space import RobotContactSpace, RobotJointSample, PublicSkinPatch
from v15_support.receiver_sensor import ReceiverMount, ReceiverMeasurement, PERIOD_S
from .transfer_observer import HandSupportSample, PreReleaseLoadPrior
from .geometry_tasks import (RobotPatchGeometry, make_length_proposal_map,
                             build_geometry_tasks)


def _real(x, name, *, nonnegative=False):
    if isinstance(x, (bool,np.bool_)) or not isinstance(x,Real) or not np.isfinite(x):
        raise ValueError(name+': finite real required')
    x=float(x)
    if nonnegative and x<0:raise ValueError(name+': nonnegative required')
    return x


def _array(x,shape,name):
    raw=np.asarray(x,dtype=object)
    if raw.shape!=shape or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,Real) for v in raw.flat):
        raise ValueError(name+': strictly numeric '+str(shape)+' required')
    a=np.asarray(x,float).copy()
    if not np.isfinite(a).all():raise ValueError(name+': unknown/nonfinite')
    a.setflags(write=False)
    return a


def _mask(x,name):
    a=np.asarray(x,dtype=object)
    if a.shape!=(5,) or any(not isinstance(v,(bool,np.bool_)) for v in a):
        raise ValueError(name+': explicit five bool flags required')
    return np.asarray(x,bool)


def _tick(time_s):
    t=_real(time_s,'public time',nonnegative=True);k=int(round(t/PERIOD_S))
    if abs(t-k*PERIOD_S)>1e-9:raise ValueError('registered200Hz public clock required')
    return k,t


def affine_equilibrium_length_derivative(*,gain21x3,bias21x3,length21,speed21,engine21):
    """At fixed velocity/raw affine effort: dlength/du=-g/(g1*u+b1).

    This is an implicit RAW force-law level-set derivative, not equilibrium of
    the whole hand or a finite-time q response. Caller handles gain/bias modes,
    control/force clipping and provenance. Zero numerator is known zero; zero
    denominator is unknown, never an epsilon-filled/zero-gain substitute.
    """
    G=_array(gain21x3,(21,3),'gain');B=_array(bias21x3,(21,3),'bias')
    length=_array(length21,(21,),'length');speed=_array(speed21,(21,),'speed')
    u=_array(engine21,(21,),'engine')
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        g=G[:,0]+G[:,1]*length+G[:,2]*speed
        denominator=G[:,1]*u+B[:,1]
        derivative=-g/denominator
    known=np.isfinite(g)&np.isfinite(denominator)&(denominator!=0)&np.isfinite(derivative)
    return (derivative.copy() if known.all() else None),dict(
        known21=known.tolist(),gain_at_snapshot21=[float(x) if np.isfinite(x) else None for x in g],
        length_coefficient_at_snapshot21=[float(x) if np.isfinite(x) else None for x in denominator],
        known_zero_reference_influence21=(known&(g==0)).tolist(),
        role='raw_affine_effort_level_set_at_fixed_speed_not_whole_hand_equilibrium',
        actual_servo_response_validated=False,pressure_prediction_available=False)


class PublicReleaseAdapter:
    """One per-run public clock/cache, not another pressure observer.

    observe(frame) must precede begin_release/current_geometry_tasks on a tick.
    It returns a typed signed HandSupportSample and freezes no future truth.
    transfer_inputs(receiver) only packages an ALREADY sampled typed receiver.
    """
    def __init__(self,robot_model,*,open18,receiver_mount=None,joint_scale25=None):
        self.open18=_array(open18,(18,),'declared OPEN18')
        self.geometry=RobotContactSpace(robot_model)
        self.servo=RobotServoEffort(robot_model,raw_joint_names=self.geometry.raw_joint_names)
        self.actor=ReferenceActor21(robot_model)
        self.axes25=tuple(self.servo.axes25)
        # Both helpers select the identical ordered 24 angular + lift positions.
        if self.axes25[:24]!=self.geometry.raw_joint_names:
            raise ValueError('robot geometry/transmission axes disagree')
        if joint_scale25 is None:
            ranges=np.asarray(self.geometry.joint_position_intervals25,float)
            limited=np.asarray(self.geometry.joint_limited25,bool)
            scale=ranges[:,1]-ranges[:,0]
            if not limited.all() or not np.isfinite(scale).all() or np.any(scale<=0):
                raise ValueError('provide explicit positive robot-coordinate metric scales')
            self.joint_scale_source='known robot declared finite joint range widths; a proposal metric only'
        else:
            scale=_array(joint_scale25,(25,),'joint_scale25')
            self.joint_scale_source='caller-declared positive robot-coordinate proposal metric'
        self.joint_scale25=_array(scale,(25,),'joint scale25')
        if np.any(self.joint_scale25<=0):raise ValueError('positive joint scales required')
        self._model=robot_model
        self._aids=np.asarray(self.servo.aids,int)
        self._jids=np.asarray(self.servo.jids,int)
        self._robot_bodies=tuple(sorted(self.geometry.robot_bodies))
        self._robot_geoms=np.flatnonzero(np.isin(robot_model.geom_bodyid,self._robot_bodies)&
            ((np.asarray(robot_model.geom_contype)!=0)|(np.asarray(robot_model.geom_conaffinity)!=0)))
        self.receiver_mount=receiver_mount;self._receiver=None
        self.receiver_half_size_xy_m=None;self.receiver_up_axis3=None
        if receiver_mount is not None:
            if type(receiver_mount) is not ReceiverMount:raise TypeError('strict ReceiverMount or None required')
            gid=int(robot_model.geom(receiver_mount.geom_name).id)
            if int(robot_model.geom_type[gid])!=int(mujoco.mjtGeom.mjGEOM_BOX) or int(robot_model.geom_bodyid[gid])!=0:
                raise ValueError('only explicit static world-mounted receiver box supported')
            size=_array(robot_model.geom_size[gid],(3,),'receiver half size')
            if not np.allclose(receiver_mount.origin_in_geom_m,[0.,0.,size[2]],atol=1e-10,rtol=0):
                raise ValueError('registered receiver top-center mount required')
            quat=_array(robot_model.geom_quat[gid],(4,),'receiver rotation quaternion')
            R=np.empty(9);mujoco.mju_quat2Mat(R,quat);R=R.reshape(3,3)
            self._receiver=dict(gid=gid,size=size,center=_array(robot_model.geom_pos[gid],(3,),'receiver center'),R=R)
            self.receiver_half_size_xy_m=_array(size[:2],(2,),'public receiver half size xy')
            self.receiver_up_axis3=_array(R.T@np.array([0.,0.,1.]),(3,),'public up in receiver frame')
        self._binding=self._signature()
        self._samples={};self._order=deque();self._prior_history=deque(maxlen=256)
        self._last_tick=None;self._last_time=None;self._last_signature=None;self.hand_sample=None
        self.release_request_s=None;self.pre_release_prior=None
        self.last_geometry_diagnostics={};self.last_clearance_diagnostics={}

    def _signature(self):
        # Robot actuation/geometry and the declared static receiver only. No
        # object mass/friction/pose, environment state or plant MjData is read.
        m=self._model;parts=[]
        for name in ('actuator_gainprm','actuator_biasprm','actuator_gaintype','actuator_biastype',
                     'actuator_trntype','actuator_trnid','actuator_gear','actuator_group',
                     'actuator_dyntype','actuator_dynprm','actuator_delay','actuator_plugin'):
            parts.append((name,np.asarray(getattr(m,name))[self._aids].tobytes()))
        for name in ('actuator_ctrlrange','actuator_ctrllimited'):
            parts.append((name,np.asarray(getattr(m,name))[self.actor.indices21].tobytes()))
        for name in ('actuator_forcerange','actuator_forcelimited'):
            parts.append((name,np.asarray(getattr(m,name))[self.servo.outadr].tobytes()))
        for name in ('jnt_type','jnt_range','jnt_axis','jnt_pos'):
            parts.append((name,np.asarray(getattr(m,name))[self._jids].tobytes()))
        for name in ('body_pos','body_quat'):
            parts.append((name,np.asarray(getattr(m,name))[list(self._robot_bodies)].tobytes()))
        gids=list(map(int,self._robot_geoms))
        if self._receiver is not None:gids.append(self._receiver['gid'])
        for name in ('geom_type','geom_bodyid','geom_pos','geom_quat','geom_size','geom_rbound','geom_contype','geom_conaffinity'):
            parts.append((name,np.asarray(getattr(m,name))[gids].tobytes()))
        parts.append(('known_actuation_modes',int(m.opt.disableflags),int(m.opt.enableflags),int(m.opt.disableactuator)))
        return tuple(parts)

    def _invalid_hand(self,now,acq,reason):
        return HandSupportSample(now,acq,False,None,None,reason)

    def observe(self,frame):
        k,now=_tick(frame.time)
        up=_array(frame.support_axis,(3,),'current public up in palm frame')
        q=_array(frame.robot_position25,(25,),'current public robot q25')
        # Duplicate attachment may refresh command memory, but must not change
        # this sensor payload or add samples/prior evidence.
        keys=('sample_time','age','normal','normal_wrench_map','finger_non_normal_by_finger',
              'tangent_basis','auxiliary_wrench','valid','distributed_valid','auxiliary_valid',
              'auxiliary_age','auxiliary_uncertainty','load_estimate','load_uncertainty','load_valid')
        def plain(v):
            if isinstance(v,np.ndarray):return v.tolist()
            if isinstance(v,np.generic):return v.item()
            return v
        sig=json.dumps(dict(up=up.tolist(),q=q.tolist(),fields={n:plain(getattr(frame,n,None)) for n in keys}),sort_keys=True,allow_nan=True)
        if self._last_tick is not None:
            if k<self._last_tick:raise ValueError('public clock rewind requires a new adapter')
            if k==self._last_tick:
                if sig!=self._last_signature:raise ValueError('changed consumed public payload on same tick')
                return self.hand_sample
        self._samples[k]=dict(time=now,up=up,q=q)
        self._order.append(k)
        while len(self._order)>256:self._samples.pop(self._order.popleft(),None)
        try:
            acq=_real(frame.sample_time,'hand acquisition',nonnegative=True)
            ak=int(round(acq/PERIOD_S));saved=self._samples.get(ak)
            if abs(acq-ak*PERIOD_S)>1e-9 or saved is None or abs(saved['time']-acq)>1e-9:
                hand=self._invalid_hand(now,acq,'coacquired_public_up_history_missing')
            elif abs((now-acq)-_real(frame.age,'hand age',nonnegative=True))>1e-9:
                hand=self._invalid_hand(now,acq,'hand_acquisition_age_mismatch')
            else:
                s=frame.sensor_characteristics
                hand=HandSupportSample.from_components(read_time_s=now,acquisition_time_s=acq,
                    up_at_acquisition3=saved['up'],axis_time_s=saved['time'],normal5=frame.normal,
                    normal_wrench_map6x5=frame.normal_wrench_map,non_normal5x6=frame.finger_non_normal_by_finger,
                    tangent_basis5x2x3=frame.tangent_basis,auxiliary_wrench6=frame.auxiliary_wrench,
                    valid5=frame.valid,distributed_valid5=frame.distributed_valid,auxiliary_valid=frame.auxiliary_valid,
                    auxiliary_acquisition_s=now-_real(frame.auxiliary_age,'auxiliary age',nonnegative=True),
                    force_sigma_N=s['force_sigma_N'],tangential_sigma_N=s['tangential_sigma_N'],
                    auxiliary_uncertainty_N=frame.auxiliary_uncertainty)
        except (AttributeError,KeyError,ValueError,TypeError):
            acq=getattr(frame,'sample_time',None)
            if isinstance(acq,(bool,np.bool_)) or not isinstance(acq,Real) or not np.isfinite(acq) or acq<0 or acq>now+1e-9:acq=None
            hand=self._invalid_hand(now,acq,'public_hand_components_missing_or_invalid')
        if self.release_request_s is None:
            valid=getattr(frame,'load_valid',False)
            v=getattr(frame,'load_estimate',None);u=getattr(frame,'load_uncertainty',None)
            good=(isinstance(valid,(bool,np.bool_)) and bool(valid) and
                  isinstance(v,Real) and not isinstance(v,(bool,np.bool_)) and np.isfinite(v) and v>=0 and
                  isinstance(u,Real) and not isinstance(u,(bool,np.bool_)) and np.isfinite(u) and u>=0)
            self._prior_history.append((now,float(v) if good else 0.,float(u) if good else 0.,bool(good)))
        self._last_tick=k;self._last_time=now;self._last_signature=sig;self.hand_sample=hand
        return hand

    def begin_release(self,request_time_s):
        _,request=_tick(request_time_s)
        if self.release_request_s is not None:
            if abs(request-self.release_request_s)>1e-9:raise ValueError('release request cannot be retimed')
            return self.pre_release_prior
        if self._last_time is None or abs(request-self._last_time)>1e-9:
            raise ValueError('observe current public tick before beginning release')
        self.release_request_s=request
        expected=request-.2+np.arange(40)*PERIOD_S
        selected=[]
        for t in expected:
            hits=[v for v in self._prior_history if abs(v[0]-t)<=1e-9]
            if len(hits)!=1:break
            selected.append(hits[0])
        if len(selected)==40:
            self.pre_release_prior=PreReleaseLoadPrior.from_history(request,*[np.array([row[i] for row in selected]) for i in range(3)],np.array([row[3] for row in selected],bool))
        return self.pre_release_prior

    def _geometry_unknown(self,reason,frame,current,desired,lo,hi,details=None):
        result=build_geometry_tasks(patches=(),mapping=None,current_time_s=frame.time,maximum_geometry_age_s=.12,
            step_lower21=lo,step_upper21=hi,nominal_open_increment21=desired-current,
            mode_generation=frame.mode_generation,robot_geometry_authenticated=False)
        diag=dict(result.diagnostics,adapter_reason=reason,adapter_details=details,
                  real_domain_not_widened=True,model_or_permission_failure_not_physical_impossibility=True)
        self.last_geometry_diagnostics=diag
        return replace(result,diagnostics=diag)

    def current_geometry_tasks(self,frame,current21,desired21,lower_delta21,upper_delta21):
        _,now=_tick(frame.time)
        current=_array(current21,(21,),'current reference21');desired=_array(desired21,(21,),'desired reference21')
        lo=_array(lower_delta21,(21,),'real lower delta21');hi=_array(upper_delta21,(21,),'real upper delta21')
        if np.any(lo>hi):raise ValueError('empty real step domain')
        if self._last_time is None or abs(self._last_time-now)>1e-9:raise ValueError('observe current frame before geometry')
        if self._signature()!=self._binding:return self._geometry_unknown('known_robot_model_definition_changed',frame,current,desired,lo,hi)
        try:
            if abs(_real(frame.robot_control_time_s,'robot command clock',nonnegative=True)-now)>1e-9:
                raise ValueError('robot command clock mismatch')
            observed=_array(frame.robot_reference21,(21,),'public committed reference21')
            if observed.tobytes()!=current.tobytes():raise ValueError('current ref not the public authoritative reference')
            engine=_array(frame.robot_actuator_target21,(21,),'public engine21')
            encoded=self.actor.encode(Reference21(current,now)).values
            if encoded.tobytes()!=engine.tobytes():raise ValueError('public reference/engine snapshot mismatch')
            q=_array(frame.robot_position25,(25,),'public q25');v=_array(frame.robot_velocity25,(25,),'public v25')
            if not (_mask(frame.valid,'valid')&_mask(frame.distributed_valid,'distributed valid')).all():
                raise ValueError('partial public hand observation cannot imply absent geometry')
            sample=RobotJointSample(now,self.geometry.raw_joint_names,q[:24],q[24],v[:24],v[24])
            state=self.servo.evaluate(q,v,engine,time_s=now)
            if state.status!='KNOWN_ROBOT_ACTUATION_ALGEBRA':raise ValueError('native robot algebra unavailable: '+str(state.reasons))
            # Stay in the engine control's real unclipped box. Exact boundary
            # directions inside this box retain the raw affine control law.
            ctrl_lo=self.actor.ctrl_range21[:,0].copy();ctrl_hi=self.actor.ctrl_range21[:,1].copy()
            ctrl_lo[18]-=self.actor.lift_bias_m;ctrl_hi[18]-=self.actor.lift_bias_m
            if np.any(current+lo<ctrl_lo-1e-12) or np.any(current+hi>ctrl_hi+1e-12):
                raise ValueError('caller step domain exceeds actual known reference control box')
            if np.any(state.diagnostics['control_clipped21']) or np.any(state.diagnostics['force_clipped21']):
                raise ValueError('clipped native branch does not admit the declared equal-effort length derivative')
            m=self.servo.model;G=m.actuator_gainprm[self.servo.aids,:3].copy();B=m.actuator_biasprm[self.servo.aids,:3].copy()
            G[np.asarray(m.actuator_gaintype[self.servo.aids])==mujoco.mjtGain.mjGAIN_FIXED,1:]=0.
            B[np.asarray(m.actuator_biastype[self.servo.aids])==mujoco.mjtBias.mjBIAS_NONE]=0.
            D,dd=affine_equilibrium_length_derivative(gain21x3=G,bias21x3=B,
                length21=state.actuator_length21,speed21=state.actuator_velocity21,engine21=state.effective_target21)
            if D is None:return self._geometry_unknown('affine_length_derivative_unknown',frame,current,desired,lo,hi,dd)
            mapping=make_length_proposal_map(transmission21x25=state.transmission21x25,reference_to_length21=D,
                joint_scale25=self.joint_scale25,robot_axes25=self.axes25,reference_axes21=REFERENCE_AXES21,
                time_s=now,declared_source='raw fixed-speed affine-effort level-set derivative -g/(g1*u+b1); '+self.joint_scale_source)
            patches=[]
            for raw in frame.skin_patches:
                p=PublicSkinPatch.from_public(raw);pressure=_real(raw.get('normal'),'public patch pressure',nonnegative=True)
                if pressure<=0:raise ValueError('zero-pressure patch cannot be fabricated as known geometry')
                if p.reference_position is None or p.reference_normal is None:raise ValueError('acquisition plane missing')
                k=self.geometry.point_kinematics(sample,p)
                patches.append(RobotPatchGeometry(p.robot_body,p.digit,p.sample_time_s,now,pressure,
                    p.reference_position,p.reference_normal,k.point_world,k.jacobian3x25,self.axes25,k.point_velocity_world))
            result=build_geometry_tasks(patches=patches,mapping=mapping,current_time_s=now,maximum_geometry_age_s=.12,
                step_lower21=lo,step_upper21=hi,nominal_open_increment21=desired-current,
                mode_generation=frame.mode_generation,robot_geometry_authenticated=True)
            diag=dict(result.diagnostics,reference_length_derivative=dd,
                unit_role='reference-to-length proposal only; rad/m coordinates explicitly scaled',
                controller_hierarchy_owned_by_caller=True,not_a_20ms_servo_or_pressure_forecast=True)
            self.last_geometry_diagnostics=diag
            return replace(result,diagnostics=diag)
        except (ValueError,TypeError,AttributeError,KeyError) as exc:
            return self._geometry_unknown(str(exc),frame,current,desired,lo,hi)

    def _hand_receiver_clearance(self):
        h=self.hand_sample
        if h is None or h.acquisition_time_s is None or self._receiver is None:
            return None,None,dict(status='UNKNOWN',reason='hand_acquisition_or_explicit_receiver_definition_missing')
        if self._signature()!=self._binding:return None,None,dict(status='UNKNOWN',reason='known_robot_or_receiver_definition_changed')
        k=int(round(h.acquisition_time_s/PERIOD_S));saved=self._samples.get(k)
        if saved is None or abs(saved['time']-h.acquisition_time_s)>1e-9 or len(self._robot_geoms)==0:
            return None,None,dict(status='UNKNOWN',reason='coacquired_robot_history_or_collision_geometry_missing')
        # Read-only private robot FK. No collision/contact query or plant data.
        m=self.geometry.model;d=self.geometry.data
        mujoco.mj_resetData(m,d);d.qpos[self.geometry.qadr]=saved['q'];d.time=saved['time']
        mujoco.mj_kinematics(m,d)
        r=self._receiver;rows=[]
        for gid in self._robot_geoms:
            supported=(mujoco.mjtGeom.mjGEOM_SPHERE,mujoco.mjtGeom.mjGEOM_CAPSULE,
                mujoco.mjtGeom.mjGEOM_ELLIPSOID,mujoco.mjtGeom.mjGEOM_CYLINDER,
                mujoco.mjtGeom.mjGEOM_BOX,mujoco.mjtGeom.mjGEOM_MESH)
            if int(m.geom_type[gid]) not in tuple(map(int,supported)):
                return None,None,dict(status='UNKNOWN',reason='unsupported_unbounded_robot_collision_geometry')
            radius=float(m.geom_rbound[gid]);center=np.asarray(d.geom_xpos[gid])
            if not np.isfinite(radius) or radius<0 or not np.isfinite(center).all():
                return None,None,dict(status='UNKNOWN',reason='robot_bounding_sphere_unavailable')
            c=r['R'].T@(center-r['center'])
            clearance=float(np.linalg.norm(np.maximum(abs(c)-r['size'],0.))-radius)
            # Arithmetic allowance only; no tuned physical clearance margin.
            allowance=64*np.finfo(float).eps*max(1.,float(np.linalg.norm(c)),float(np.linalg.norm(r['size'])),radius)
            rows.append((int(gid),clearance,allowance))
        minimum=min(x[1] for x in rows)
        possible=not all(x[1]>x[2] for x in rows)
        return possible,h.acquisition_time_s,dict(status='POSSIBLE_HAND_RECEIVER_CONTACT' if possible else 'ALL_ROBOT_BOUNDING_SPHERES_STRICTLY_CLEAR',
            acquisition_time_s=h.acquisition_time_s,minimum_sphere_box_clearance_m=minimum,
            maximum_arithmetic_allowance_m=max(x[2] for x in rows),
            collision_geoms_checked=len(rows),possible_overlap_not_true_contact=possible,
            whole_static_box_conservative_not_top_contact_classifier=True,physical_safety_certified=False)

    def transfer_inputs(self,receiver):
        if type(receiver) is not ReceiverMeasurement:raise TypeError('already-sampled typed ReceiverMeasurement required')
        if self.hand_sample is None:raise ValueError('observe current public hand frame first')
        if abs(receiver.time_s-self.hand_sample.read_time_s)>1e-9:raise ValueError('receiver/hand read clock mismatch')
        confounded,stamp,diagnostics=self._hand_receiver_clearance()
        self.last_clearance_diagnostics=diagnostics
        return dict(time_s=self.hand_sample.read_time_s,hand=self.hand_sample,receiver=receiver,
                    hand_receiver_confounded=confounded,geometry_acquisition_time_s=stamp)
