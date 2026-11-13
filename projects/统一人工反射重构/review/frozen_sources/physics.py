"""Public finite-mass five-pad MuJoCo benchmark, not full Shadow dynamics.

The rigid object moves freely. All methods execute the same five compliant
closure coordinates. Actual solver forces are never clipped or replaced.
SensorFrame contains local tactile skin and fixture proprioception only.
"""
from collections import deque
from dataclasses import dataclass
import copy
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

DT=.005
TASK_LIFT=.160
MAX_DELTA=.003
CLOSURE_SCALE=.030
PAD_RADIUS=.009
CONTACT_GAP=.010
# Three-shape development steps give diagonal 14.24--14.31 N/closure;
# the fitted force response is at or faster than the 5 ms recording resolution.
FORCE_GAIN=np.full(5,14.25)
FORCE_TAU=.005
BARE_HAND_EFFORT_OFFSET=.0
ANGLES=2*np.pi*np.arange(5)/5.
DIRECTIONS=np.column_stack([np.cos(ANGLES),np.sin(ANGLES),np.zeros(5)])
SUBSTEP_COLUMNS=('solver_time',*(f'normal_{i}_N' for i in range(5)),
    *(f'shear_{i}_N' for i in range(5)), 'palm_N','support_N','collision_N',
    'penetration_m','world_penetration_m','duration_s',
    *(f'skin_normal_{i}_N' for i in range(5)), 'hand_other_N','all_hand_object_N')
_MODEL_CACHE={}


@dataclass(frozen=True)
class SensorFrame:
    time:float
    normal:np.ndarray
    tangential:np.ndarray
    contact_pos:np.ndarray
    contact_normal:np.ndarray
    tangent_basis:np.ndarray
    slip_local:np.ndarray
    joint:np.ndarray
    joint_velocity:np.ndarray
    valid:np.ndarray
    age:float
    alpha:np.ndarray
    phase:str
    lift:float
    lift_ref:float
    support_axis:np.ndarray
    raw_joint:np.ndarray
    raw_joint_velocity:np.ndarray
    force_gain:np.ndarray
    closure_permission:np.ndarray
    closure_max:np.ndarray
    closure_min:np.ndarray
    nominal_force_weights:np.ndarray
    motor_effort:float
    handload_prior:float
    load_estimate:float
    load_uncertainty:float
    load_valid:bool
    load_confidence:float
    force_tau:float=FORCE_TAU
    dt:float=DT


def tangent_basis(n):
    n=np.asarray(n,float)
    if np.linalg.norm(n)<1e-8:return np.array([[0.,0.,1.],[0.,1.,0.]])
    n=n/np.linalg.norm(n);a=np.array([0.,0.,1.])
    if abs(n@a)>.9:a=np.array([0.,1.,0.])
    t=a-(a@n)*n;t/=np.linalg.norm(t)
    return np.stack([t,np.cross(n,t)])


def support_point(shape,size,n):
    if shape=='sphere':return .050*size*n
    if shape=='box':
        # Supporting face/corner rather than ray intersection: a sphere pad
        # placed corner+radius*n has exactly the prescribed normal on a box.
        return np.array([.028*size*np.sign(n[0]) if abs(n[0])>1e-8 else 0.,
                         .065*size*np.sign(n[1]) if abs(n[1])>1e-8 else 0.,0.])
    return .028*size*n


def load_model(shape,size,solref=.020,timestep=.0005):
    key=(shape,size,solref,timestep)
    if key not in _MODEL_CACHE:
        root=ET.Element('mujoco',model='five compliant tactile contacts')
        ET.SubElement(root,'compiler',angle='radian',autolimits='true')
        ET.SubElement(root,'option',timestep=str(timestep),integrator='implicitfast',
            cone='elliptic',impratio='100',iterations='60',noslip_iterations='5',
            tolerance='1e-10',gravity='0 0 -9.81')
        default=ET.SubElement(root,'default')
        ET.SubElement(default,'geom',solref=f'{solref} 1',solimp='.98 .995 .001 .5 2',
            condim='3',friction='.6 .001 .0001')
        world=ET.SubElement(root,'worldbody')
        ET.SubElement(world,'light',pos='.4 -.5 1',dir='-.4 .5 -1')
        ET.SubElement(world,'geom',name='floor',type='plane',size='1 1 .1',
            rgba='.85 .88 .90 1',contype='4',conaffinity='2')
        height=(.050 if shape=='sphere' else .028 if shape=='box' else .047)*size
        tabletop=.30-height
        ET.SubElement(world,'geom',name='support',type='box',pos=f'0 0 {tabletop-.005}',
            size='.10 .12 .005',rgba='.40 .53 .61 1',contype='4',conaffinity='2')
        hand=ET.SubElement(world,'body',name='hand',pos='0 0 .30')
        ET.SubElement(hand,'inertial',mass='.60',pos='0 0 0',diaginertia='.004 .004 .006')
        ET.SubElement(hand,'joint',name='transport_lift',type='slide',axis='0 0 1',
            range='0 .20',damping='8',armature='.05')
        wrist=ET.SubElement(hand,'body',name='contact_frame')
        ET.SubElement(wrist,'inertial',mass='.05',pos='0 0 0',diaginertia='.0004 .0004 .0005')
        ET.SubElement(wrist,'joint',name='wrist',axis='0 1 0',range='-.3 .3',damping='.05')
        ET.SubElement(wrist,'geom',name='fixture_outline',type='cylinder',size='.095 .003',
            pos='0 0 -.070',rgba='.23 .37 .42 .6',contype='0',conaffinity='0')
        for digit,n in enumerate(DIRECTIONS):
            position=support_point(shape,size,n)+(PAD_RADIUS+CONTACT_GAP)*n
            body=ET.SubElement(wrist,'body',name=f'finger_{digit}',pos=' '.join(str(x) for x in position))
            ET.SubElement(body,'joint',name=f'closure_{digit}',type='slide',
                axis=' '.join(str(x) for x in -n),range='0 .0465',damping='1',armature='.003')
            ET.SubElement(body,'geom',name=f'pad_{digit}',type='sphere',size=str(PAD_RADIUS),
                mass='.04',rgba=('.85 .43 .19 1' if digit==0 else '.12 .57 .63 1'),
                contype='1',conaffinity='10')
        obj=ET.SubElement(world,'body',name='object',pos='0 0 .30')
        ET.SubElement(obj,'freejoint',name='object_free')
        dimensions=(f'{.028*size} {.065*size} {.028*size}' if shape=='box' else
                    str(.050*size) if shape=='sphere' else f'{.028*size} {.047*size}')
        ET.SubElement(obj,'geom',name='object_geom',type=shape,size=dimensions,
            mass='.12',rgba='.95 .65 .21 1',contype='2',conaffinity='5')
        # Finite-mass/finite-force obstacle avoids an infinite-work kinematic crusher.
        obstacle=ET.SubElement(world,'body',name='collision_fixture',pos='.28 0 .46')
        ET.SubElement(obstacle,'joint',name='obstacle_travel',type='slide',axis='1 0 0',range='-.26 .04',damping='2')
        ET.SubElement(obstacle,'geom',name='obstacle_geom',type='box',size='.010 .12 .012',
            mass='.12',rgba='.8 .2 .2 1',contype='8',conaffinity='3')
        act=ET.SubElement(root,'actuator')
        for digit in range(5):
            ET.SubElement(act,'position',name=f'close_motor_{digit}',joint=f'closure_{digit}',
                kp='500',kv='8',ctrlrange='0 .0465',forcerange='-6 6')
        ET.SubElement(act,'position',name='lift_motor',joint='transport_lift',kp='1000',kv='60',
            ctrlrange='0 .23',forcerange='-25 25')
        ET.SubElement(act,'position',name='wrist_motor',joint='wrist',kp='5',kv='.2',
            ctrlrange='-.3 .3',forcerange='-1 1')
        ET.SubElement(act,'position',name='obstacle_motor',joint='obstacle_travel',kp='200',kv='5',
            ctrlrange='-.26 .04',forcerange='-8 8')
        _MODEL_CACHE[key]=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        if len(_MODEL_CACHE)>4:del _MODEL_CACHE[next(iter(_MODEL_CACHE))]
    return copy.copy(_MODEL_CACHE[key])


class Plant:
    def __init__(self,scenario,p,timestep=.0005):
        self.scenario,self.p=scenario,dict(p)
        self.model=load_model(scenario.shape,p['size'],p.get('solref',.020),timestep)
        self.data=mujoco.MjData(self.model)
        self.bid=lambda n:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,n)
        self.gid=lambda n:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,n)
        self.aid=lambda n:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,n)
        self.object=self.bid('object');self.hand=self.bid('hand');self.palm=self.bid('contact_frame')
        self.object_geom=self.gid('object_geom');self.obstacle_geom=self.gid('obstacle_geom')
        self.finger_bodies=[self.bid(f'finger_{i}') for i in range(5)]
        self.pad_geoms=[self.gid(f'pad_{i}') for i in range(5)]
        self.geom_finger={g:i for i,g in enumerate(self.pad_geoms)};self.hand_geoms=set(self.pad_geoms)
        self.actuators=[self.aid(f'close_motor_{i}') for i in range(5)]
        self.lift_motor=self.aid('lift_motor');self.wrist_motor=self.aid('wrist_motor');self.obstacle_motor=self.aid('obstacle_motor')
        self.closure_joints=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,f'closure_{i}') for i in range(5)]
        self.qadr=self.model.jnt_qposadr[self.closure_joints];self.dadr=self.model.jnt_dofadr[self.closure_joints]
        self.lift_joint=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'transport_lift')
        self.lift_qadr=int(self.model.jnt_qposadr[self.lift_joint]);self.lift_dadr=int(self.model.jnt_dofadr[self.lift_joint])
        ratio=p['mass']/self.model.body_mass[self.object]
        self.model.body_mass[self.object]*=ratio;self.model.body_inertia[self.object]*=ratio
        mujoco.mj_setConst(self.model,self.data);mujoco.mj_resetData(self.model,self.data)
        self.alpha=np.zeros(5);self.closure_min=self.model.actuator_ctrlrange[self.actuators,0]/CLOSURE_SCALE
        self.closure_max=self.model.actuator_ctrlrange[self.actuators,1]/CLOSURE_SCALE
        self._control_tick=0;self.phase='CLOSING';self.filtered_force=np.zeros(5)
        self.contact_since=None;self.lift_started=None;self.hold_started=None;self.release_started=None
        self.release_lift=0.;self.lift_ref=0.;self.task_lift_height=TASK_LIFT;self.failed_setup=False
        self.limits=np.array(scenario.limit,float);self.budget=float(scenario.budget);self.substep_records=[]
        self.set_friction(p['friction']);self._set_actuators()
        self.data.ctrl[self.lift_motor]=self.model.body_subtreemass[self.hand]*9.81/1000.
        mujoco.mj_forward(self.model,self.data)
        self.start_position=self.data.xpos[self.object].copy();self.reference_relative=self._relative_pose()[0]
        self.reference_rotation=self._relative_pose()[1];self.initial_penetration=self.truth()['penetration']
        self.model_metadata=dict(model='finite-mass five compliant tactile pads',full_shadow_dynamics=False,
            closure_scale_m=CLOSURE_SCALE,pad_radius_m=PAD_RADIUS,initial_gap_m=CONTACT_GAP,pad_mass_kg=.04,
            actuator_kp_N_per_m=500.,actuator_damping_Ns_per_m=8.,actuator_force_cap_N=6.,
            task_lift_m=TASK_LIFT,squeeze_lever_m=.04,angles_rad=ANGLES.tolist(),
            obstacle_mass_kg=.12,obstacle_force_cap_N=8.,obstacle_x_reference_shift_m=-.012)

    @property
    def time(self):return self._control_tick*DT

    def set_friction(self,mu):
        for g in [*self.pad_geoms,self.object_geom]:self.model.geom_friction[g,0]=mu

    def _relative_pose(self):
        R=self.data.xmat[self.palm].reshape(3,3)
        return R.T@(self.data.xpos[self.object]-self.data.xpos[self.palm]),R.T@self.data.xmat[self.object].reshape(3,3)

    def _measure(self,kinematics=False):
        normal=np.zeros(5);shear=np.zeros(5);skin=np.zeros(5)
        positions=np.zeros((5,3));directions=np.zeros((5,3));loads=np.zeros((5,3));vels=np.zeros((5,3))
        support=collision=penetration=world_penetration=0.
        R=self.data.xmat[self.palm].reshape(3,3);force=np.zeros(6)
        for i in range(self.data.ncon):
            c=self.data.contact[i];mujoco.mj_contactForce(self.model,self.data,i,force);amount=max(0.,float(force[0]))
            if self.object_geom in (c.geom1,c.geom2):
                other=c.geom2 if c.geom1==self.object_geom else c.geom1
                if other in self.geom_finger:
                    d=self.geom_finger[other];normal[d]+=amount;shear[d]+=np.linalg.norm(force[1:3])
                    penetration=max(penetration,max(0.,-float(c.dist)))
                elif self.model.geom_bodyid[other]==0:support+=amount;world_penetration=max(world_penetration,max(0.,-float(c.dist)))
            if self.obstacle_geom in (c.geom1,c.geom2):
                other=c.geom2 if c.geom1==self.obstacle_geom else c.geom1
                if other in self.geom_finger:collision+=amount
            for finger,other,sign in ((c.geom1,c.geom2,1.),(c.geom2,c.geom1,-1.)):
                if finger not in self.geom_finger or other in self.hand_geoms:continue
                d=self.geom_finger[finger];skin[d]+=amount
                if kinematics and amount>0.:
                    positions[d]+=amount*(R.T@(c.pos-self.data.xpos[self.palm]));directions[d]+=amount*(R.T@(sign*c.frame[:3]))
                    loads[d]+=R.T@(sign*c.frame.reshape(3,3).T@force[:3])
                    other_body=int(self.model.geom_bodyid[other]);finger_body=int(self.model.geom_bodyid[finger]);vo,vf=np.zeros(6),np.zeros(6)
                    mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,other_body,vo,0)
                    mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,finger_body,vf,0)
                    rel=vo[3:]+np.cross(vo[:3],c.pos-self.data.xipos[other_body])-vf[3:]-np.cross(vf[:3],c.pos-self.data.xipos[finger_body])
                    vels[d]+=amount*(R.T@rel)
        basis=np.zeros((5,2,3));tangent=np.zeros((5,2));slip=np.zeros((5,2))
        if kinematics:
            for d in range(5):
                if skin[d]>1e-8:
                    positions[d]/=skin[d];length=np.linalg.norm(directions[d])
                    if length>1e-8:directions[d]/=length
                    basis[d]=tangent_basis(directions[d]);tangent[d]=basis[d]@loads[d];slip[d]=basis[d]@(vels[d]/skin[d])
                else:basis[d]=tangent_basis(np.zeros(3))
        return dict(normal=normal,shear=shear,skin_normal=skin,palm=0.,hand_other=0.,all_hand_object=float(normal.sum()),
            support=support,collision=collision,penetration=penetration,world_penetration=world_penetration,
            tactile=dict(normal=skin,tangential=tangent,contact_pos=positions,contact_normal=directions,tangent_basis=basis,slip_local=slip))

    def truth(self):
        result=self._measure(True);R=self.data.xmat[self.palm].reshape(3,3);vo,vh=np.zeros(6),np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.object,vo,0)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.palm,vh,0)
        relative,rotation=self._relative_pose();lever=self.data.xipos[self.object]-self.data.xipos[self.palm]
        v=R.T@(vo[3:]-vh[3:]-np.cross(vh[:3],lever));w=R.T@(vo[:3]-vh[:3])
        result.update(position=self.data.xpos[self.object].copy(),relative=relative,rotation=rotation,
            relvel=v,relomega=w,relative_velocity=v.copy(),angular_velocity=w.copy(),
            tangential=result['tactile']['tangential'].copy(),slip_local=result['tactile']['slip_local'].copy(),
            contact_pos=result['tactile']['contact_pos'].copy(),contact_normal=result['tactile']['contact_normal'].copy(),tangent_basis=result['tactile']['tangent_basis'].copy(),
            joint=self.data.qpos[self.qadr].copy()/CLOSURE_SCALE,joint_velocity=self.data.qvel[self.dadr].copy()/CLOSURE_SCALE,
            raw_joint=self.data.qpos[self.qadr].copy(),raw_joint_velocity=self.data.qvel[self.dadr].copy(),
            alpha=self.alpha.copy(),phase=self.phase,lift=float(self.data.qpos[self.lift_qadr]),lift_ref=self.lift_ref,
            closure_min=self.closure_min.copy(),closure_max=self.closure_max.copy(),
            lift_velocity=float(self.data.qvel[self.lift_dadr]),motor_effort=float(self.data.actuator_force[self.lift_motor]),
            bare_hand_weight=float(self.model.body_subtreemass[self.hand]*9.81),known_hand_inertia=float(self.model.body_subtreemass[self.hand]+self.model.dof_armature[self.lift_dadr]),
            lift_damping=float(self.model.dof_damping[self.lift_dadr]),support_axis=R.T@np.array([0.,0.,1.]))
        return result

    def task_update(self,obs,command='hold',pause_transport=False):
        t=self.time
        if command=='release' and self.release_started is None:self.release_started=t;self.release_lift=self.lift_ref;self.phase='RELEASING'
        valid=obs.valid & np.isfinite(obs.normal);self.filtered_force[valid]=.75*self.filtered_force[valid]+.25*obs.normal[valid]
        if self.phase=='CLOSING':
            contact=np.count_nonzero(valid&(self.filtered_force>.22))>=3 and valid[0] and self.filtered_force[0]>.22
            if contact:
                if self.contact_since is None:self.contact_since=t
                if t-self.contact_since>=.35:self.lift_started=t;self.phase='LIFTING'
            else:self.contact_since=None
            if t>=7.:self.failed_setup=True;self.phase='FAILED'
        if self.phase=='LIFTING' and not pause_transport:
            self.lift_ref=min(TASK_LIFT,self.lift_ref+TASK_LIFT/1.5*DT)
            if self.lift_ref>=TASK_LIFT-1e-9 and abs(obs.lift-TASK_LIFT)<.006:self.hold_started=t;self.phase='HOLDING'
        if self.phase=='RELEASING':self.lift_ref=self.release_lift

    def execute(self,action_increment,obs,env=None,command='hold'):
        env={} if env is None else env;delta=np.asarray(getattr(action_increment,'delta_closure',action_increment),float)
        if delta.shape!=(5,) or not np.isfinite(delta).all():raise ValueError('five finite closure increments required')
        pause=bool(getattr(action_increment,'pause_transport',False));self.task_update(obs,command,pause)
        if self.phase=='FAILED':delta=np.full(5,-MAX_DELTA)
        if self.phase=='RELEASING':delta=np.minimum(delta,0.)
        self.alpha=np.clip(self.alpha+np.clip(delta,-MAX_DELTA,MAX_DELTA),self.closure_min,self.closure_max)
        if self.phase=='HOLDING' and not pause:self.lift_ref=TASK_LIFT+float(env.get('lift_delta',0.))
        self._set_actuators(float(env.get('wrist',0.)))

    def _set_actuators(self,wrist=0.):
        self.data.ctrl[self.actuators]=CLOSURE_SCALE*self.alpha;gravity=self.model.body_subtreemass[self.hand]*9.81/1000.
        self.data.ctrl[self.lift_motor]=np.clip(self.lift_ref+gravity,self.data.ctrl[self.lift_motor]-.08*DT,self.data.ctrl[self.lift_motor]+.08*DT)
        self.data.ctrl[self.wrist_motor]=wrist

    def advance(self,env):
        self.set_friction(env['friction']);obstacle=np.asarray(env.get('obstacle',[.28,0,.49]),float)
        self.data.ctrl[self.obstacle_motor]=np.clip(float(obstacle[0]-.28-.012),-.26,.04)
        self.data.xfrc_applied[:]=0.;self.data.qfrc_applied[:]=0.
        if abs(self.data.xpos[self.object,1])<.20:self.data.xfrc_applied[self.object,:3]=env.get('force',[0.,0.,0.])
        self.data.qfrc_applied[self.dadr]=float(env.get('squeeze',0.))/.04
        for _ in range(round(DT/self.model.opt.timestep)):
            t=float(self.data.time);mujoco.mj_step(self.model,self.data);c=self._measure()
            self.substep_records.append((t,*c['normal'],*c['shear'],c['palm'],c['support'],c['collision'],c['penetration'],c['world_penetration'],
                float(self.model.opt.timestep),*c['skin_normal'],c['hand_other'],c['all_hand_object']))
        mujoco.mj_forward(self.model,self.data);self._control_tick+=1


class SensorPipeline:
    def __init__(self,scenario,p,seed):
        self.scenario,self.p=scenario,dict(p);self.rng=np.random.default_rng(seed+910_000);self.queue=deque(maxlen=100)
        self.previous_lift_velocity=None;self.encoder_acceleration=0.;self.load_estimate=0.

    def sample(self,truth,time):
        s=truth['tactile'];sigma=float(self.p.get('force_noise',self.scenario.noise))
        # Fixed draw shapes every tick preserve identical noise paths across methods.
        n_noise=self.rng.normal(0,sigma,5);t_noise=self.rng.normal(0,sigma*.5,(5,2))
        p_noise=self.rng.normal(0,.00025,(5,3));v_noise=self.rng.normal(0,.00025+sigma*.002,(5,2));effort_noise=float(self.rng.normal(0,.02))
        normal=np.maximum(0.,s['normal']+n_noise);tangential=s['tangential']+t_noise;position=s['contact_pos'].copy()
        contact=s['normal']>.02;position[contact]+=p_noise[contact];slip=s['slip_local']+v_noise;slip[~contact]=0.
        direction=s['contact_normal'].copy();basis=s['tangent_basis'].copy();valid=np.ones(5,bool)
        if self.scenario.event=='noise':
            if 9.5<=time<9.62:slip[1,0]+=.045
            if 11.<=time<11.03:normal[1]+=2.
        if self.scenario.event=='dropout':
            if 9.3<=time<9.7:valid[1:3]=False
            if 12.<=time<12.15:valid[:]=False
        for a in (normal,tangential,position,slip,direction,basis):a[~valid]=np.nan
        self.queue.append(dict(time=time,normal=normal,tangential=tangential,contact_pos=position,contact_normal=direction,tangent_basis=basis,slip_local=slip,valid=valid))
        delay=round(float(self.p.get('delay',.01))/DT);e=self.queue[max(0,len(self.queue)-1-delay)]
        v=float(truth['lift_velocity']);raw_a=0. if self.previous_lift_velocity is None else (v-self.previous_lift_velocity)/DT
        self.previous_lift_velocity=v;self.encoder_acceleration+=(1-np.exp(-DT/.025))*(raw_a-self.encoder_acceleration)
        effort=float(truth['motor_effort']+effort_noise)
        prior=truth['bare_hand_weight']+BARE_HAND_EFFORT_OFFSET+truth['known_hand_inertia']*self.encoder_acceleration+truth['lift_damping']*v
        self.load_estimate+=(1-np.exp(-DT/.045))*(effort-prior-self.load_estimate)
        speed=float(np.linalg.norm(truth['joint_velocity']));quiet=abs(self.encoder_acceleration)<.05 and abs(v)<.005 and speed<.02
        uncertainty=(.10 if quiet else .55)+.10*abs(self.encoder_acceleration)+.25*speed
        load_valid=bool(truth['lift']>.01 and truth['phase'] in ('LIFTING','HOLDING') and abs(self.encoder_acceleration)<2.)
        confidence=.85 if load_valid and quiet else .5 if load_valid else 0.
        return SensorFrame(time=time,**{k:e[k].copy() for k in ('normal','tangential','contact_pos','contact_normal','tangent_basis','slip_local','valid')},
            joint=truth['joint'].copy(),joint_velocity=truth['joint_velocity'].copy(),age=float(time-e['time']),alpha=truth['alpha'].copy(),phase=truth['phase'],lift=truth['lift'],lift_ref=truth['lift_ref'],
            support_axis=truth['support_axis'].copy(),raw_joint=truth['raw_joint'].copy(),raw_joint_velocity=truth['raw_joint_velocity'].copy(),
            force_gain=FORCE_GAIN.copy(),closure_permission=np.ones(5,bool),closure_min=truth['closure_min'].copy(),closure_max=truth['closure_max'].copy(),nominal_force_weights=np.ones(5),
            motor_effort=effort,handload_prior=float(prior),load_estimate=max(0.,float(self.load_estimate)),load_uncertainty=float(uncertainty),load_valid=load_valid,load_confidence=confidence)
