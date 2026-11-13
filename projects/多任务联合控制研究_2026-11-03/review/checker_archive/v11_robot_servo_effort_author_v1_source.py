"""Known robot actuation algebra from robot q/v and actual actuator targets.

No plant MjData, external pose/parameters, contact solve, dynamics advance, force
forecast, desired24q invention or independent torque-sensor claim. Supports only
robot scalar joint/fixed-tendon transmissions with stateless affine force laws.
Reference order is18finger, lift, WRJ2, WRJ1. Lift target includes the actual
engine-port gravity bias. Sparse MuJoCo moments are decoded using their CSR.
"""
from __future__ import annotations
from copy import copy
from dataclasses import dataclass,field
from numbers import Real
from typing import Optional
import numpy as np
import mujoco

ACTUATORS21=tuple('rh_A_'+name for name in ('THJ5','THJ4','THJ3','THJ2','THJ1','FFJ4','FFJ3','FFJ0','MFJ4','MFJ3','MFJ0','RFJ4','RFJ3','RFJ0','LFJ5','LFJ4','LFJ3','LFJ0'))+('lift_motor','rh_A_WRJ2','rh_A_WRJ1')


def _finite(value,shape,name):
    if value is None:raise ValueError(name+'_missing')
    raw=np.asarray(value,dtype=object)
    if raw.shape!=shape or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,Real) for v in raw.flat):raise ValueError(name+'_requires_real_numeric_'+str(shape))
    a=np.asarray(value,float)
    if not np.isfinite(a).all():raise ValueError(name+'_nonfinite')
    return a.copy()


def _ro(value):
    a=np.asarray(value).copy();a.setflags(write=False);return a


@dataclass(frozen=True)
class EffortSnapshot:
    status:str
    reasons:tuple=()
    time_s:Optional[float]=None
    robot_axes25:tuple=()
    actuator_names21:tuple=ACTUATORS21
    commanded_target21:Optional[np.ndarray]=None
    effective_target21:Optional[np.ndarray]=None
    transmission21x25:Optional[np.ndarray]=None
    actuator_length21:Optional[np.ndarray]=None
    actuator_velocity21:Optional[np.ndarray]=None
    servo_error21:Optional[np.ndarray]=None
    unbounded_effort21:Optional[np.ndarray]=None
    predicted_effort21:Optional[np.ndarray]=None
    generalized_actuator_effort25:Optional[np.ndarray]=None
    reference_jacobian25x21:Optional[np.ndarray]=None
    actuation_stiffness25x25:Optional[np.ndarray]=None
    actuation_damping25x25:Optional[np.ndarray]=None
    derivative_status:dict=field(default_factory=dict)
    diagnostics:dict=field(default_factory=dict)
    contact_force_prediction_available:bool=False
    measured_effort:bool=False
    global_model_admitted:bool=False


class RobotServoEffort:
    def __init__(self,model,*,raw_joint_names=None):
        # MjModel copy owns private arrays; live plant MjData is never accepted.
        self.model=copy(model);self.data=mujoco.MjData(self.model);m=self.model;self.unavailable=[]
        names=tuple(raw_joint_names) if raw_joint_names is not None else tuple(m.joint(j).name for j in range(m.njnt) if (m.joint(j).name or '').startswith('rh_'))
        if len(names)!=24 or len(set(names))!=24 or any(not isinstance(n,str) or not n.startswith('rh_') for n in names):raise ValueError('Explicit24named_robot_angular_axes_required')
        self.axes25=names+('transport_lift',);self.jids=np.array([m.joint(n).id for n in self.axes25],int)
        if any(m.jnt_type[j]!=mujoco.mjtJoint.mjJNT_HINGE for j in self.jids[:24]) or m.jnt_type[self.jids[24]]!=mujoco.mjtJoint.mjJNT_SLIDE:raise ValueError('robot25_must_be24hinges_and_lift')
        self.qadr=np.asarray(m.jnt_qposadr[self.jids],int);self.dadr=np.asarray(m.jnt_dofadr[self.jids],int)
        self.aids=np.array([m.actuator(n).id for n in ACTUATORS21],int)
        self.ctrladr=np.asarray(m.actuator_ctrladr[self.aids],int);self.outadr=np.asarray(m.actuator_outadr[self.aids],int)
        if len(set(self.aids))!=21 or m.nu!=22 or m.nout!=22 or m.nactuator!=22:self.unavailable.append('expected_known21robot_plus_one_disconnected_scalar_actuator')
        for aid in self.aids:
            name=m.actuator(aid).name
            if m.actuator_ctrlnum[aid]!=1 or m.actuator_outnum[aid]!=1:self.unavailable.append(name+':non_SISO_actuator')
            if m.actuator_dyntype[aid]!=mujoco.mjtDyn.mjDYN_NONE or m.actuator_actnum[aid]!=0 or m.actuator_actadr[aid]>=0:self.unavailable.append(name+':unknown_activation_state')
            if m.actuator_delay[aid]!=0:self.unavailable.append(name+':unprovided_control_delay_history')
            if m.actuator_plugin[aid]>=0:self.unavailable.append(name+':unsupported_plugin')
            if m.actuator_gaintype[aid] not in (mujoco.mjtGain.mjGAIN_FIXED,mujoco.mjtGain.mjGAIN_AFFINE):self.unavailable.append(name+':unsupported_force_gain')
            if m.actuator_biastype[aid] not in (mujoco.mjtBias.mjBIAS_NONE,mujoco.mjtBias.mjBIAS_AFFINE):self.unavailable.append(name+':unsupported_force_bias')
            trn=m.actuator_trntype[aid];target=int(m.actuator_trnid[aid,0])
            if trn==mujoco.mjtTrn.mjTRN_JOINT:
                if target not in set(self.jids):self.unavailable.append(name+':nonrobot_joint_transmission')
            elif trn==mujoco.mjtTrn.mjTRN_TENDON:
                if m.tendon_actfrclimited[target]:self.unavailable.append(name+':unimplemented_total_tendon_force_clamp')
                for wi in range(int(m.tendon_adr[target]),int(m.tendon_adr[target]+m.tendon_num[target])):
                    if m.wrap_type[wi]!=mujoco.mjtWrap.mjWRAP_JOINT or int(m.wrap_objid[wi]) not in set(self.jids):self.unavailable.append(name+':only_robot_fixed_joint_tendon_supported')
            else:self.unavailable.append(name+':unsupported_transmission_geometry')
        if np.any(m.jnt_actfrclimited[self.jids]):self.unavailable.append('unimplemented_joint_total_actuator_force_clamp')
        if np.any(m.jnt_actgravcomp[self.jids]):self.unavailable.append('unprovided_actuation_gravity_compensation_state')
        if int(m.opt.disableflags)&int(mujoco.mjtDisableBit.mjDSBL_ACTUATION):self.unavailable.append('actuation_disabled_mode')
        if int(m.opt.enableflags)&int(mujoco.mjtEnableBit.mjENBL_SLEEP):self.unavailable.append('unprovided_sleep_activation_mode')
        if any(int(m.opt.disableactuator)&(1<<int(m.actuator_group[j])) for j in self.aids):self.unavailable.append('robot_actuator_group_disabled')
        self.ctrl_limited=np.asarray(m.actuator_ctrllimited[self.ctrladr],bool)
        self.ctrl_range=np.asarray(m.actuator_ctrlrange[self.ctrladr],float).copy()
        self.force_limited=np.asarray(m.actuator_forcelimited[self.outadr],bool)
        self.force_range=np.asarray(m.actuator_forcerange[self.outadr],float).copy()

    def _transmission(self,q,v):
        m=self.model;d=self.data;mujoco.mj_resetData(m,d)
        d.qpos[self.qadr]=q;d.qvel[self.dadr]=v
        # No forward position/dynamics, collisions, constraints or integrator.
        mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_tendon(m,d);mujoco.mj_transmission(m,d)
        full=np.zeros((m.nout,m.nv))
        if d.actuator_moment.ndim!=1 or len(d.moment_rownnz)!=m.nout:raise ValueError('unsupported_sparse_moment_API_layout')
        for row in range(m.nout):
            start=int(d.moment_rowadr[row]);count=int(d.moment_rownnz[row]);columns=np.asarray(d.moment_colind[start:start+count],int)
            if len(set(columns))!=len(columns):raise ValueError('duplicate_sparse_transmission_columns')
            full[row,columns]=d.actuator_moment[start:start+count]
        nonrobot=np.array([i for i in range(m.nv) if i not in set(self.dadr)],int)
        if np.max(abs(full[self.outadr][:,nonrobot]),initial=0)>1e-12:raise ValueError('robot_transmission_depends_on_unprovided_nonrobot_coordinates')
        other=np.array([i for i in range(m.nout) if i not in set(self.outadr)],int)
        if np.max(abs(full[other][:,self.dadr]),initial=0)>1e-12:raise ValueError('unprovided_other_actuator_affects_robot')
        # Same CSR-times-qvel rule as native3.15 mj_fwdVelocity, without its
        # passive/full-body velocity pipeline or any extra measured signal.
        d.actuator_velocity[:]=full@d.qvel
        return full[self.outadr][:,self.dadr].copy(),d.actuator_length[self.outadr].copy(),d.actuator_velocity[self.outadr].copy()

    def evaluate(self,q25,v25,actual_ctrl,*,control_layout='robot21',time_s=0.):
        if self.unavailable:return EffortSnapshot('UNAVAILABLE',tuple(self.unavailable),robot_axes25=self.axes25)
        try:
            q=_finite(q25,(25,),'robot_q25');v=_finite(v25,(25,),'robot_velocity25')
            if isinstance(time_s,(bool,np.bool_)) or not isinstance(time_s,Real) or not np.isfinite(time_s):raise ValueError('finite_robot_sample_time_required')
            if control_layout=='robot21':u=_finite(actual_ctrl,(21,),'actual_robot_actuator_target21')
            elif control_layout=='model22':
                # Select the known robot controls BEFORE checking values. The
                # obstacle command is neither a feature nor an authority input.
                value=np.asarray(actual_ctrl,dtype=object)
                if value.shape!=(22,):raise ValueError('actual_model_control22_shape')
                u=_finite(value[self.ctrladr],(21,),'selected_robot_actuator_target21')
            else:raise ValueError('explicit_robot21_or_model22_control_layout_required')
            A,length,speed=self._transmission(q,v)
        except (ValueError,TypeError) as exc:return EffortSnapshot('UNAVAILABLE',(str(exc),),robot_axes25=self.axes25)
        m=self.model;effective=u.copy();ctrl_free=np.ones(21,bool);ctrl_kink=np.zeros(21,bool)
        if not int(m.opt.disableflags)&int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL):
            for i in np.flatnonzero(self.ctrl_limited):
                lo,hi=self.ctrl_range[i];effective[i]=np.clip(u[i],lo,hi)
                ctrl_free[i]=lo<u[i]<hi;ctrl_kink[i]=abs(u[i]-lo)<=1e-12 or abs(u[i]-hi)<=1e-12
        gain=m.actuator_gainprm[self.aids,:3].copy();bias=m.actuator_biasprm[self.aids,:3].copy()
        gain[np.asarray(m.actuator_gaintype[self.aids])==mujoco.mjtGain.mjGAIN_FIXED,1:]=0.
        bias[np.asarray(m.actuator_biastype[self.aids])==mujoco.mjtBias.mjBIAS_NONE]=0.
        g=gain[:,0]+gain[:,1]*length+gain[:,2]*speed
        unbounded=g*effective+bias[:,0]+bias[:,1]*length+bias[:,2]*speed
        effort=unbounded.copy();force_free=np.ones(21,bool);force_kink=np.zeros(21,bool)
        for i in np.flatnonzero(self.force_limited):
            lo,hi=self.force_range[i];effort[i]=np.clip(unbounded[i],lo,hi)
            force_free[i]=lo<unbounded[i]<hi;force_kink[i]=abs(unbounded[i]-lo)<=1e-12 or abs(unbounded[i]-hi)<=1e-12
        torque=A.T@effort
        # Fixed/joint transmissions make A constant. All partials below hold
        # other actual q/v/command arguments fixed, and exclude passive/contact
        # stiffness. Clip kinks have no single Jacobian: returnNone, not zero.
        Ju=None if np.any(ctrl_kink|force_kink) else A.T@np.diag(g*ctrl_free*force_free)
        K=None if np.any(force_kink) else -A.T@np.diag((gain[:,1]*effective+bias[:,1])*force_free)@A
        D=None if np.any(force_kink) else -A.T@np.diag((gain[:,2]*effective+bias[:,2])*force_free)@A
        limited=np.asarray(m.jnt_limited[self.jids],bool);ranges=m.jnt_range[self.jids]
        qoutside=limited&((q<ranges[:,0])|(q>ranges[:,1]))
        diagnostics=dict(pipeline=['mj_resetData','mj_kinematics','mj_comPos','mj_tendon','mj_transmission','CSR_A_times_robot_velocity','known_affine_actuator_law'],
            live_plant_data_read=False,external_pose_or_parameters_read=False,physics_steps=0,contact_solve_called=False,
            nonrobot_control_ignored=control_layout=='model22',control_layout=control_layout,
            robot_position_outside_declared_joint_ranges25=qoutside.tolist(),robot_positions_not_clipped=True,robot_velocity_limits_known=False,
            control_clipped21=(u!=effective).tolist(),force_clipped21=(effort!=unbounded).tolist(),control_at_clip_kink21=ctrl_kink.tolist(),force_at_clip_kink21=force_kink.tolist(),
            robot_effort_units25=['Nm']*24+['N'],actuator_effort_units='transmission-coordinate effort; joint/fixed-angular-tendon vs linear-lift units differ',
            servo_error_scope='actual actuator target minus actual transmission length, NOT a desired24joint posture',
            stiffness_scope='known actuation partial derivative only; not total joint/contact/environment stiffness or force-gradient',
            evidence_role='model-computed internal effort, not an independent current/torque measurement',activation_assumed_zero=False,global_model_admitted=False)
        return EffortSnapshot('KNOWN_ROBOT_ACTUATION_ALGEBRA',(),float(time_s),self.axes25,ACTUATORS21,
            _ro(u),_ro(effective),_ro(A),_ro(length),_ro(speed),_ro(effective-length),_ro(unbounded),_ro(effort),_ro(torque),
            None if Ju is None else _ro(Ju),None if K is None else _ro(K),None if D is None else _ro(D),
            dict(reference_jacobian='UNAVAILABLE_CLIP_KINK' if Ju is None else 'AVAILABLE_KNOWN_BRANCH',
                 actuation_stiffness='UNAVAILABLE_FORCE_CLIP_KINK' if K is None else 'AVAILABLE_KNOWN_BRANCH',
                 actuation_damping='UNAVAILABLE_FORCE_CLIP_KINK' if D is None else 'AVAILABLE_KNOWN_BRANCH'),diagnostics)
