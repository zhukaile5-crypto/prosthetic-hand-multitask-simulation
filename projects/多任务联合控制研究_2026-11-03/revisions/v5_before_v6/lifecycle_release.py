"""Causal contact-separation release in the real, finite robot action space.

This experimental mode uses robot encoder kinematics and virtual skin only.
It unloads on request, projects opening velocity away from sensed contacts,
and keeps the hand at its release height while observing post-clear contact.
It does not infer object pose or certify collision-free space after touch ends.
"""
from dataclasses import dataclass
import numpy as np
import research_imports
from vendor.frozen_controllers import solve_qp


@dataclass
class ReleaseCommand:
    target: np.ndarray
    lift_reference: float
    wrist_reference: np.ndarray
    stage: str
    debug: dict


class RobotSkinKinematics:
    """FK of robot encoders; object generalized coordinates are never read."""
    def __init__(self, plant):
        import mujoco
        self.mj=mujoco;self.model=plant.model;self.data=mujoco.MjData(self.model)
        self.qadr=plant.raw_qadr.copy();self.lift_qadr=plant.lift_qadr
        self.palm=plant.palm
        self.groups=[g.copy() for g in plant.actuator_groups]
        self.aids=np.concatenate(self.groups)
        self.geoms=[[int(g) for g,finger in plant.geom_finger.items() if finger==i] for i in range(5)]
        self.control_map=np.zeros((self.model.nv,len(self.aids)))
        self.contact_memory={}
        for column,aid in enumerate(self.aids):
            name=self.model.actuator(int(aid)).name.removeprefix('rh_A_')
            names=[name[:-1]+'1',name[:-1]+'2'] if name.endswith('J0') else [name]
            for joint_name in names:
                joint=self.model.joint('rh_'+joint_name)
                self.control_map[int(joint.dofadr[0]),column]=1./len(names)

    def contact_rows(self, frame):
        self.data.qpos[self.qadr]=np.asarray(frame.raw_joint)
        self.data.qpos[self.lift_qadr]=float(frame.lift)
        self.mj.mj_kinematics(self.model,self.data)
        self.mj.mj_comPos(self.model,self.data)
        R=self.data.xmat[self.palm].reshape(3,3);origin=self.data.xpos[self.palm]
        rows=[];digits=[];limits=[];seen={}
        if hasattr(frame,'skin_patches'):
            for patch in frame.skin_patches:
                if patch['normal']<.04:continue
                digit=int(patch['digit']);body=int(patch['robot_body'])
                Rb=self.data.xmat[body].reshape(3,3);pb=self.data.xpos[body]
                local=np.asarray(patch['link_point'])
                n=np.asarray(patch.get('reference_normal',Rb@np.asarray(patch['link_normal'])))
                n=n/max(1e-12,np.linalg.norm(n))
                p=np.asarray(patch.get('reference_position',pb+Rb@local))
                seen.setdefault((digit,body),[]).append((local.copy(),n.copy(),p.copy()))
            self.contact_memory.update(seen)
            for (digit,body),points in self.contact_memory.items():
                Rb=self.data.xmat[body].reshape(3,3);pb=self.data.xpos[body]
                for local,n,plane in points:
                    p=pb+Rb@local;J=np.zeros((3,self.model.nv))
                    # A rotating capsule can present a NEW surface point.
                    # Constrain its full geometric support towards the sensed
                    # surface plane, rather than only the old touched taxel.
                    candidates=[g for g in self.geoms[digit]
                        if int(self.model.geom_bodyid[g])==body and
                        (self.model.geom_contype[g] or self.model.geom_conaffinity[g])]
                    extremal=[]
                    for geom in candidates:
                        kind=int(self.model.geom_type[geom]);c=self.data.geom_xpos[geom]
                        Rg=self.data.geom_xmat[geom].reshape(3,3);size=self.model.geom_size[geom]
                        if kind==int(self.mj.mjtGeom.mjGEOM_CAPSULE):
                            axis=Rg[:,2];point=c+np.sign(n@axis)*size[1]*axis+size[0]*n
                        elif kind==int(self.mj.mjtGeom.mjGEOM_SPHERE):point=c+size[0]*n
                        elif kind==int(self.mj.mjtGeom.mjGEOM_BOX):point=c+Rg@(np.sign(Rg.T@n)*size)
                        else:continue
                        extremal.append(point)
                    if extremal:p=max(extremal,key=lambda x:float(n@x))
                    self.mj.mj_jac(self.model,self.data,J,None,p,body)
                    gap=-float(n@(p-plane))
                    rows.append(n@J@self.control_map);digits.append(digit)
                    limits.append(2.*(gap-.003))
            return np.asarray(rows).reshape(-1,len(self.aids)),digits,np.asarray(limits)
        for i in range(5):
            if not frame.valid[i] or not np.isfinite(frame.normal[i]) or frame.normal[i]<.08:
                continue
            n=np.asarray(frame.contact_normal[i],float)
            if not np.isfinite(n).all() or np.linalg.norm(n)<.2:continue
            n=R@(n/np.linalg.norm(n));p=origin+R@np.asarray(frame.contact_pos[i])
            candidates=self.geoms[i]
            if not candidates:continue
            geom=min(candidates,key=lambda g:float(np.linalg.norm(self.data.geom_xpos[g]-p)))
            body=int(self.model.geom_bodyid[geom]);J=np.zeros((3,self.model.nv))
            self.mj.mj_jac(self.model,self.data,J,None,p,body)
            rows.append(n@J@self.control_map);digits.append(i)
        return np.asarray(rows).reshape(-1,len(self.aids)),digits,np.full(len(digits),-.002)


class LifecycleRelease:
    def __init__(self,plant):
        self.kinematics=RobotSkinKinematics(plant)
        self.aids=self.kinematics.aids
        self.lower=plant.model.actuator_ctrlrange[self.aids,0].copy()
        self.upper=plant.model.actuator_ctrlrange[self.aids,1].copy()
        self.full_open=np.concatenate([np.array([-1.,.1,0.,0.,0.]),*plant.open_pose[1:]])
        self.clearance_pose=np.concatenate([np.array([-1.03,1.21,.13,-.02,.985]),*plant.open_pose[1:]])
        self.target=self.clearance_pose.copy()
        # Existing finger synergy rate (0.8/s) and original thumb-path envelope.
        self.rate=np.concatenate([np.full(5,2.5),*[.8*np.abs(c-o) for o,c in
            zip(plant.open_pose[1:],plant.closed_pose[1:])]])
        self.started=None;self.last_time=None;self.command=None
        self.clear_since=None;self.first_clear=None;self.recontacts=0
        self.was_clear=False;self.stage='not_requested';self.holding_called=False
        self.opening_committed=False

    def request(self,frame,actuator_target,lift_reference,wrist_reference):
        if self.started is not None:return
        self.started=float(frame.time);self.last_time=float(frame.time)-float(frame.dt)
        self.command=np.asarray(actuator_target,float).copy()
        self.lift_reference=float(lift_reference);self.wrist_reference=np.asarray(wrist_reference,float).copy()
        # A release can take ownership after preparation has already opened the
        # robot. Preserve that executed phase; command state is not evidence of
        # physical clearance, which still requires the public skin dwell below.
        self.opening_committed=bool(np.allclose(self.command,self.full_open,rtol=0.,atol=1e-8))
        self.stage='continue_full_open' if self.opening_committed else 'unloading'

    def step(self,frame):
        if self.started is None:raise RuntimeError('Release must first be requested')
        dt=float(np.clip(float(frame.time)-self.last_time,.0001,.02));self.last_time=float(frame.time)
        before=self.command.copy();maximum=np.minimum(self.rate,(self.upper-before)/dt)
        minimum=np.maximum(-self.rate,(self.lower-before)/dt)
        self.target=self.full_open.copy() if self.opening_committed else self.clearance_pose.copy()
        desired=np.clip((self.target-before)/.20,minimum,maximum)
        contact, digits,separation_limit=self.kinematics.contact_rows(frame)
        n=len(before);m=len(digits);I=np.eye(n+m)
        rows=[I[:n],-I[:n]];bounds=[maximum,-minimum]
        if m:
            # Outward hand velocity is positive into the sensed surface. A
            # retracting hand moves opposite the hand-on-external normal.
            rows += [np.column_stack((contact,-np.eye(m))),-I[n:]]
            bounds += [separation_limit,np.zeros(m)]
        A=np.vstack(rows);b=np.concatenate(bounds)
        seed=np.r_[desired,np.maximum(0.,contact@desired-separation_limit)+1e-8]
        Q=np.eye(n+m)*1e-6;q=np.zeros(n+m)
        Q[:n,:n]+=np.eye(n)*2.;q[:n]=-2.*desired
        if m:Q[n:,n:]+=np.eye(m)*2.e5
        result=solve_qp(Q,q,A,b,180,2e-5,feasible_initial=seed)
        choice=result.x if result.residual<=2e-5 else seed
        velocity=choice[:n];self.command=before+velocity*dt
        if np.any(self.command<self.lower-1e-8) or np.any(self.command>self.upper+1e-8):
            raise RuntimeError('Release planned an out-of-range actuator target')
        measured=np.asarray(frame.normal,float)
        valid=bool(np.all(frame.valid) and np.isfinite(measured).all() and frame.auxiliary_valid)
        clear=bool(valid and measured.sum()+float(frame.auxiliary_pressure)<=.2)
        if clear:
            if self.clear_since is None:self.clear_since=float(frame.time)
            if frame.time-self.clear_since>=.2-1e-9:
                if self.first_clear is None:self.first_clear=float(frame.time)
                self.was_clear=True;self.opening_committed=True;self.stage='post_clear_monitoring'
        elif valid:
            if self.was_clear:self.recontacts+=1
            self.was_clear=False;self.clear_since=None;self.stage='contact_clearance'
        else:
            self.clear_since=None;self.stage='release_observation_uncertain'
        return ReleaseCommand(self.command.copy(),self.lift_reference,self.wrist_reference.copy(),self.stage,
            dict(release_authority=True,prediction_available=False,
                prediction_unavailable_reason='release uses robot kinematics, outside held-force prediction',
                lifecycle_stage=self.stage,monitoring_active=True,first_clear_confirmation=self.first_clear,
                release_posture_mode='committed_full_open' if self.opening_committed else 'public_thumb_clearance',
                opening_phase_committed=self.opening_committed,
                observed_recontact_count=self.recontacts,contact_digits=digits,qp_status=result.status,
                separation_slack_m_s=choice[n:].tolist(),actuator_before=before.tolist(),
                actual_actuator_target=self.command.tolist(),actuator_lower=self.lower.tolist(),
                actuator_upper=self.upper.tolist(),actuator_rate_limit=self.rate.tolist(),dt=dt,
                release_geometry_scope='delayed distributed skin, retained planes and full robot capsule support; joint/tendon approximation; no collision certificate',
                return_scope='hold release height; no safe downward return is inferred from lost touch'))

