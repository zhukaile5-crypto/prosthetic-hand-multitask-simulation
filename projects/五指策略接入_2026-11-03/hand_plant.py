"""Articulated Shadow plant and common actuation; contains no grasp algorithm.

All force values originate from MuJoCo.  Skin observations and object-specific
evaluation are separate.  Per-contact force is hand-on-external-object; slip is
external-point velocity relative to the touching hand point, in palm coordinates.
"""
from pathlib import Path
from dataclasses import asdict
import hashlib
import copy
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from foundation_config import FoundationConfig, GROUPS, OPEN, CLOSED, PREFIX, BENCHMARK_ID

ROOT = Path(__file__).resolve().parent
_MODELS = {}

def target_profile(t,lift_started,config=None):
    config=FoundationConfig() if config is None else config
    target=np.asarray(config.acquisition_force,float)
    if lift_started is not None:
        blend=np.clip((t-lift_started-config.force_transition_delay)/config.force_transition_duration,0.,1.)
        target=target+blend*(np.asarray(config.holding_force)-target)
    return target

class CommonPreparation:
    """Public force-reference preparation, shared and excluded from policy credit."""
    def __init__(self,plant):
        self.plant=plant;self.filtered=np.zeros(5);self.contact_since=None;self.contact_tick=None;self.lift_started=None;self.lift_tick=None
    def step(self,observed_normal=None):
        p=self.plant;c=p.config;t=p.time
        if observed_normal is None:
            observed_normal=np.zeros(5)
            for patch in p.raw_sensors()['patches']:
                if patch['digit']>=0:observed_normal[patch['digit']]+=patch['normal']
        self.filtered=.7*self.filtered+.3*np.asarray(observed_normal)
        target=target_profile(t,self.lift_started,c)
        p.execute_delta(np.clip((target-self.filtered)*.001,-.006,.004))
        if self.lift_started is None:
            if np.all(self.filtered>c.prelift_contact_force):
                if self.contact_since is None:self.contact_since=t;self.contact_tick=p.control_steps
                if p.control_steps-self.contact_tick>=round(c.prelift_dwell/c.control_dt):
                    self.lift_started=t;self.lift_tick=p.control_steps;p.phase='LIFTING'
            else:self.contact_since=None;self.contact_tick=None
        else:
            p.set_motion(c.lift_target*min(1.,(t-self.lift_started)/c.lift_duration))
            if p.control_steps-self.lift_tick>=round(c.lift_duration/c.control_dt):p.phase='HOLDING'
        return target.copy()

def _model(config):
    key=tuple(asdict(config).items())
    if key not in _MODELS:
        path=ROOT/'assets/legacy_model.xml'
        tree=ET.parse(path); root=tree.getroot()
        root.set('model',BENCHMARK_ID)
        root.find('compiler').set('meshdir','')
        root.find('option').set('timestep',str(config.timestep))
        root.find('option').set('iterations','100')
        forearm=root.find(".//body[@name='rh_forearm']")
        hand_pos=np.fromstring(forearm.get('pos','0 0 0'),sep=' ')
        hand_pos[2]+=config.hand_offset_z;forearm.set('pos',' '.join(map(str,hand_pos)))
        obj=root.find(".//body[@name='object']")
        geom=obj.find("geom[@name='object_geom']")
        scale=config.size
        height=.028*scale
        if config.shape=='capsule':
            geom.set('type','capsule');geom.set('size',f'{.028*scale} {.047*scale}')
        elif config.shape=='cylinder':
            geom.set('type','cylinder');geom.set('size',f'{.028*scale} {.060*scale}')
        elif config.shape=='sphere':
            geom.set('type','sphere');geom.set('size',str(.028*scale));geom.attrib.pop('quat',None)
        elif config.shape=='box':
            height=.026*scale
            geom.set('type','box');geom.set('size',f'{.018*scale} {.060*scale} {.026*scale}')
            geom.attrib.pop('quat',None)
        else: raise ValueError(f'Unsupported shape: {config.shape}')
        obj.set('pos',f'{.047+config.object_offset_x} 0 {.3375+height+config.object_offset_z}')
        geom.set('mass',str(config.mass))
        for g in root.iter('geom'):
            if g.get('class')=='plastic_collision' or g.get('name')=='object_geom':
                g.set('friction',f'{config.friction} .015 .003')
                g.set('solref',f'{config.solref} 1')
                g.set('condim',str(config.contact_dimension))
        if config.collision_model=='smooth_distal_capsules':
            for body in root.iter('body'):
                if not body.get('name','').startswith('rh_') or not body.get('name','').endswith('distal'):continue
                for g in body.findall('geom'):
                    if g.get('class')=='plastic_collision' and g.get('type')=='mesh':
                        g.attrib.pop('mesh',None);g.set('type','capsule')
                        if body.get('name')=='rh_thdistal':
                            g.set('size','.0095 .0063');g.set('pos','0 0 .0192')
                        else:
                            g.set('size','.0075 .0062');g.set('pos','0 0 .01835')
        elif config.collision_model!='original_mesh':raise ValueError('Unknown collision representation')
        if config.shape=='box' and config.box_rounding_radius>0.:
            # Analytic rounded cuboid: same outside envelope and body inertia as
            # the sharp box, with a 2 mm Minkowski rounding of its contact skin.
            outer=np.array([.018,.060,.026])*scale
            radius=float(config.box_rounding_radius)*scale
            inner=outer-radius
            if np.min(inner)<=0.:raise ValueError('rounding radius exceeds box extent')
            inertia=config.mass/3.*np.array([outer[1]**2+outer[2]**2,outer[0]**2+outer[2]**2,outer[0]**2+outer[1]**2])
            ET.SubElement(obj,'inertial',mass=str(config.mass),pos='0 0 0',diaginertia=' '.join(map(str,inertia)))
            geom.set('size',' '.join(map(str,inner)));geom.attrib.pop('mass',None);geom.set('density','0')
            common={k:v for k,v in geom.attrib.items() if k not in ('name','type','size','pos','quat','mass')}
            for axis in range(3):
                for sign in (-1,1):
                    pos=np.zeros(3);pos[axis]=sign*inner[axis]
                    size=inner.copy();size[axis]=radius
                    ET.SubElement(obj,'geom',**common,name=f'rounded_face_{axis}_{sign}',type='box',
                        pos=' '.join(map(str,pos)),size=' '.join(map(str,size)))
                other=[j for j in range(3) if j!=axis]
                for a in (-1,1):
                    for b in (-1,1):
                        p0=np.zeros(3);p0[other[0]]=a*inner[other[0]];p0[other[1]]=b*inner[other[1]]
                        p1=p0.copy();p0[axis]=-inner[axis];p1[axis]=inner[axis]
                        ET.SubElement(obj,'geom',**common,name=f'rounded_edge_{axis}_{a}_{b}',type='capsule',
                            size=str(radius),fromto=' '.join(map(str,np.r_[p0,p1])))
            for x in (-1,1):
                for y in (-1,1):
                    for z in (-1,1):
                        pos=inner*np.array([x,y,z])
                        ET.SubElement(obj,'geom',**common,name=f'rounded_corner_{x}_{y}_{z}',type='sphere',
                            size=str(radius),pos=' '.join(map(str,pos)))
        if not config.object_enabled:
            obj.set('pos','.047 0 -2')
            for g in obj.findall('geom'):g.set('contype','0');g.set('conaffinity','0')
        # Optional collision fixture has finite mass and finite drive force.
        world=root.find('worldbody')
        obstacle=ET.SubElement(world,'body',name='integration_obstacle',pos='.40 0 .44')
        ET.SubElement(obstacle,'joint',name='integration_obstacle_x',type='slide',axis='1 0 0',range='-.4 .1',damping='2')
        ET.SubElement(obstacle,'geom',name='integration_obstacle_geom',type='sphere',size='.022',mass='.12',
                      rgba='.7 .22 .22 .6',contype='2',conaffinity='1',friction='.5 .005 .0001')
        ET.SubElement(root.find('actuator'),'position',name='integration_obstacle_motor',joint='integration_obstacle_x',
                      kp='500',kv='8',ctrlrange='-.4 .1',forcerange='-8 8')
        xml=ET.tostring(root,encoding='unicode')
        assets={p.name:p.read_bytes() for p in (ROOT/'assets/shadow/assets').iterdir() if p.is_file()}
        model=mujoco.MjModel.from_xml_string(xml,assets=assets)
        _MODELS[key]=(model,hashlib.sha256(xml.encode()).hexdigest())
    return _MODELS[key]

class HandPlant:
    def __init__(self,config=None,**kwargs):
        if config is None: config=FoundationConfig(**kwargs)
        elif isinstance(config,dict): config=FoundationConfig(**config)
        self.config=config
        base,self.model_sha256=_model(config)
        self.model=copy.copy(base)
        self.data=mujoco.MjData(self.model)
        self.bid=lambda n:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,n)
        self.aid=lambda n:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,n)
        self.object=self.bid('object');self.hand=self.bid('rh_forearm');self.palm=self.bid('rh_palm')
        self.object_geom=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,'object_geom')
        self.object_geoms=set(int(i) for i in np.flatnonzero(self.model.geom_bodyid==self.object))
        self.obstacle_motor=self.aid('integration_obstacle_motor')
        self.lift_actuator=self.aid('lift_motor')
        self.lift_jid=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'transport_lift')
        self.lift_qadr=int(self.model.jnt_qposadr[self.lift_jid]);self.lift_dadr=int(self.model.jnt_dofadr[self.lift_jid])
        self.actuator_groups=[np.array([self.aid('rh_A_'+k) for k in group]) for group in GROUPS]
        self.open_pose=[np.array(x,float) for x in OPEN];self.closed_pose=[np.array(x,float) for x in CLOSED]
        self.geom_finger={};self.hand_geoms=set()
        for gid in range(self.model.ngeom):
            b=int(self.model.geom_bodyid[gid]);name=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,b) or ''
            if name.startswith('rh_'):self.hand_geoms.add(gid)
            for digit,pre in enumerate(PREFIX):
                if name.startswith(pre):self.geom_finger[gid]=digit
        self.raw_joint_ids=[j for j in range(self.model.njnt) if (mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_JOINT,j) or '').startswith('rh_')]
        self.raw_qadr=np.array([self.model.jnt_qposadr[j] for j in self.raw_joint_ids],int)
        self.raw_dadr=np.array([self.model.jnt_dofadr[j] for j in self.raw_joint_ids],int)
        lower=[];upper=[]
        for i,indices in enumerate(self.actuator_groups):
            slope=self.closed_pose[i]-self.open_pose[i];lo=-np.inf;hi=np.inf
            for k,aid in enumerate(indices):
                if abs(slope[k])<1e-10: continue
                bounds=(self.model.actuator_ctrlrange[aid]-self.open_pose[i][k])/slope[k]
                lo=max(lo,float(min(bounds)));hi=min(hi,float(max(bounds)))
            lower.append(lo);upper.append(hi)
        self.closure_min=np.array(lower);self.closure_max=np.minimum(np.array(upper),1.55)
        self.reset()

    @property
    def time(self): return self.control_steps*self.config.control_dt

    def reset(self):
        mujoco.mj_resetData(self.model,self.data)
        for keys,values in zip(GROUPS,self.open_pose):
            for key,value in zip(keys,values):
                if key.endswith('J0'):continue
                jid=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'rh_'+key)
                self.data.qpos[self.model.jnt_qposadr[jid]]=value
        self.control_steps=0;self.alpha=np.zeros(5);self.phase='CLOSING';self.lift_ref=0.;self.wrist_ref=np.zeros(2)
        self.release_started=None;self.release_thumb=None;self.release_lift=0.;self.release_alpha=None;self.release_ctrl=None
        self.env={'force':np.zeros(3),'torque':np.zeros(3),'squeeze':0.,'obstacle_x':0.}
        self.substep_records=[];self.control_steps=0;self.last_requested=np.zeros(5);self.last_executed=np.zeros(5)
        self.max_hidden_clip=0.;self._cache=None
        self.force_peak=np.zeros(5);self.force_peak_contacts=[None]*5
        self._apply_ctrl();mujoco.mj_forward(self.model,self.data)
        self.initial_position=self.data.xpos[self.object].copy()
        self.initial_penetration=self.truth()['penetration']
        self.substep_records.clear()

    def set_closure(self,alpha):
        requested=np.asarray(alpha,float)
        if requested.shape!=(5,) or not np.isfinite(requested).all():raise ValueError('five finite closure coordinates required')
        before=self.alpha.copy();self.alpha=np.clip(requested,self.closure_min,self.closure_max)
        self.last_requested=requested-before;self.last_executed=self.alpha-before
        self._apply_ctrl();self.last_executed=self.alpha-before
        return self.last_executed.copy()

    def execute_delta(self,delta,dt=None):
        dt=self.config.control_dt if dt is None else float(dt)
        delta=np.asarray(getattr(delta,'delta_closure',delta),float)
        if delta.shape!=(5,) or not np.isfinite(delta).all():raise ValueError('five finite closure increments required')
        # Rate limit belongs to common execution; always return actual increment.
        bounded=np.clip(delta,-.8*dt,.8*dt)
        self.raw_requested_delta=delta.copy()
        return self.set_closure(self.alpha+bounded)

    def set_motion(self,lift=None,wrist=None):
        if lift is not None:self.lift_ref=float(np.clip(lift,0.,self.config.lift_target))
        if wrist is not None:self.wrist_ref=np.broadcast_to(np.asarray(wrist,float),(2,)).copy()
        self._apply_ctrl()

    def set_environment(self,env=None,**kwargs):
        values={} if env is None else dict(env);values.update(kwargs)
        for key in ('force','torque','squeeze','obstacle_x'):
            if key in values:self.env[key]=values[key]
        if 'friction' in values:
            mu=float(values['friction'])
            for gid in [*self.geom_finger,*self.object_geoms]:self.model.geom_friction[gid,0]=mu

    def begin_release(self):
        if self.release_started is None:
            self.release_started=self.time;self.release_thumb=self.data.ctrl[self.actuator_groups[0]].copy()
            self.release_lift=self.lift_ref;self.release_alpha=self.alpha.copy();self.release_ctrl=self.data.ctrl.copy();self.phase='RELEASING'

    def _apply_ctrl(self):
        for i,indices in enumerate(self.actuator_groups):
            requested=self.open_pose[i]+self.alpha[i]*(self.closed_pose[i]-self.open_pose[i])
            actual=np.clip(requested,self.model.actuator_ctrlrange[indices,0],self.model.actuator_ctrlrange[indices,1])
            self.max_hidden_clip=max(self.max_hidden_clip,float(np.max(np.abs(requested-actual))))
            self.data.ctrl[indices]=actual
        if self.release_started is not None:
            elapsed=self.time-self.release_started
            self.alpha[1:]=np.maximum(0.,self.release_alpha[1:]-self.config.release_open_rate*elapsed)
            for i in range(1,5):
                indices=self.actuator_groups[i]
                self.data.ctrl[indices]=self.open_pose[i]+self.alpha[i]*(self.closed_pose[i]-self.open_pose[i])
            tucked=np.array([-1.03,1.21,.13,-.02,.985]);full_open=np.array([-1.,.1,0.,0.,0.])
            if elapsed<.8:thumb=self.release_thumb+np.clip(elapsed/.8,0,1)*(tucked-self.release_thumb)
            elif elapsed<1.4:thumb=tucked
            else:thumb=tucked+np.clip((elapsed-1.4)/.8,0,1)*(full_open-tucked)
            self.data.ctrl[self.actuator_groups[0]]=np.clip(thumb,self.model.actuator_ctrlrange[self.actuator_groups[0],0],self.model.actuator_ctrlrange[self.actuator_groups[0],1])
            self.alpha[0]=(thumb[0]-self.open_pose[0][0])/(self.closed_pose[0][0]-self.open_pose[0][0])
            self.lift_ref=self.release_lift*(1.-np.clip(elapsed/.6,0.,1.))
        gravity=self.model.body_subtreemass[self.hand]*9.81/1500.
        self.data.ctrl[self.lift_actuator]=self.lift_ref+gravity
        for j,key in enumerate(('rh_A_WRJ2','rh_A_WRJ1')):self.data.ctrl[self.aid(key)]=self.wrist_ref[j]
        self.data.ctrl[self.obstacle_motor]=float(np.clip(self.env.get('obstacle_x',0.),-.4,.1))

    def _contacts(self,patches=True):
        normal=np.zeros(5);skin=np.zeros(5);shear=np.zeros(5);palm=other_hand=table=0.
        penetration=world_penetration=0.;all_hand_object=0.;result=[];R=self.data.xmat[self.palm].reshape(3,3)
        for i in range(self.data.ncon):
            c=self.data.contact[i];f=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,i,f)
            amount=max(0.,float(f[0]));g1,g2=int(c.geom1),int(c.geom2)
            object_contact=g1 in self.object_geoms or g2 in self.object_geoms
            if object_contact:
                g=g2 if g1 in self.object_geoms else g1
                if g in self.geom_finger:
                    d=self.geom_finger[g];normal[d]+=amount;shear[d]+=float(np.linalg.norm(f[1:3]))
                elif g in self.hand_geoms:
                    if self.model.geom_bodyid[g]==self.palm:palm+=amount
                    else:other_hand+=amount
                else:table+=amount;world_penetration=max(world_penetration,max(0.,-float(c.dist)))
                if g in self.hand_geoms:all_hand_object+=amount;penetration=max(penetration,max(0.,-float(c.dist)))
            for hand_geom,external,sign in ((g1,g2,1.),(g2,g1,-1.)):
                if hand_geom not in self.hand_geoms or external in self.hand_geoms:continue
                digit=self.geom_finger.get(hand_geom,-1)
                if digit>=0:skin[digit]+=amount
                if not patches or amount<=1e-12:continue
                hand_body=int(self.model.geom_bodyid[hand_geom]);external_body=int(self.model.geom_bodyid[external])
                vhand=np.zeros(6);vext=np.zeros(6)
                mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,hand_body,vhand,0)
                mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,external_body,vext,0)
                rel=vext[3:]+np.cross(vext[:3],c.pos-self.data.xipos[external_body])-vhand[3:]-np.cross(vhand[:3],c.pos-self.data.xipos[hand_body])
                frame=c.frame.reshape(3,3)
                part='finger' if digit>=0 else 'palm' if hand_body==self.palm else 'other'
                result.append(dict(digit=digit,part=part,normal=amount,force=R.T@(sign*frame.T@f[:3]),
                    torque=R.T@(sign*frame.T@f[3:]),position=R.T@(c.pos-self.data.xpos[self.palm]),
                    normal_dir=R.T@(sign*frame[0]),slip_velocity=R.T@rel,
                    world_position=c.pos.copy(),is_target=external in self.object_geoms,
                    hand_geom=hand_geom,external_geom=external,penetration=max(0.,-float(c.dist))))
        return dict(normal=normal,skin_normal=skin,shear=shear,palm=palm,hand_other=other_hand,support=table,
                    all_hand_object=all_hand_object,penetration=penetration,world_penetration=world_penetration,patches=result)

    def raw_sensors(self):
        c=self._contacts(True);joint=[];velocity=[]
        for i,indices in enumerate(self.actuator_groups):
            slope=self.closed_pose[i]-self.open_pose[i];denom=float(slope@slope)
            joint.append(float((self.data.actuator_length[indices]-self.open_pose[i])@slope/denom))
            velocity.append(float(self.data.actuator_velocity[indices]@slope/denom))
        R=self.data.xmat[self.palm].reshape(3,3)
        return dict(time=self.time,patches=c['patches'],contacts=c['patches'],joint=np.asarray(joint),joint_velocity=np.asarray(velocity),
            raw_joint=self.data.qpos[self.raw_qadr].copy(),raw_joint_velocity=self.data.qvel[self.raw_dadr].copy(),alpha=self.alpha.copy(),
            closure_min=self.closure_min.copy(),closure_max=self.closure_max.copy(),closure_permission=np.ones(5,bool),
            lift=float(self.data.qpos[self.lift_qadr]),lift_ref=self.lift_ref,lift_velocity=float(self.data.qvel[self.lift_dadr]),
            motor_effort=float(self.data.actuator_force[self.lift_actuator]),bare_hand_weight=float(self.model.body_subtreemass[self.hand]*9.81),
            known_hand_inertia=float(self.model.body_subtreemass[self.hand]+self.model.dof_armature[self.lift_dadr]),
            lift_damping=float(self.model.dof_damping[self.lift_dadr]),support_axis=R.T@np.array([0.,0.,1.]),phase=self.phase,
            release_mode=self.release_started is not None)

    def truth(self):
        c=self._contacts(False);R=self.data.xmat[self.palm].reshape(3,3);vo=np.zeros(6);vh=np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.object,vo,0)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.palm,vh,0)
        lever=self.data.xpos[self.object]-self.data.xpos[self.palm]
        c.update(time=self.time,position=self.data.xpos[self.object].copy(),relative=R.T@lever,
            rotation=R.T@self.data.xmat[self.object].reshape(3,3),relative_velocity=R.T@(vo[3:]-vh[3:]-np.cross(vh[:3],lever)),
            angular_velocity=R.T@(vo[:3]-vh[:3]),lift=float(self.data.qpos[self.lift_qadr]),lift_ref=self.lift_ref,
            qpos=self.data.qpos.copy(),qvel=self.data.qvel.copy(),ctrl=self.data.ctrl.copy(),alpha=self.alpha.copy(),phase=self.phase,
            warning_count=int(sum(int(x.number) for x in self.data.warning)))
        return c

    def advance(self,step=None):
        step=self.config.control_dt if step is None else float(step)
        if abs(step-self.config.control_dt)>1e-10:raise ValueError('advance executes exactly one public control tick')
        ticks=round(step/self.model.opt.timestep)
        if ticks<1 or abs(ticks*self.model.opt.timestep-step)>1e-10:raise ValueError('step must contain integer physical ticks')
        for _ in range(ticks):
            self._apply_ctrl();self.data.xfrc_applied[:]=0.;self.data.qfrc_applied[:]=0.
            self.data.xfrc_applied[self.object,:3]=self.env.get('force',np.zeros(3))
            self.data.xfrc_applied[self.object,3:]=self.env.get('torque',np.zeros(3))
            squeeze=float(self.env.get('squeeze',0.))
            for key in ('THJ5','FFJ3','MFJ3','RFJ3','LFJ3'):
                jid=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'rh_'+key)
                self.data.qfrc_applied[self.model.jnt_dofadr[jid]]=squeeze
            start=float(self.data.time);mujoco.mj_step(self.model,self.data);self._cache=None
            c=self._contacts(False)
            for digit in np.where(c['normal']>self.force_peak+1e-10)[0]:
                self.force_peak[digit]=c['normal'][digit]
                self.force_peak_contacts[digit]=dict(time=start,normal=float(c['normal'][digit]),
                    patches=[p for p in self._contacts(True)['patches'] if p['is_target'] and p['digit']==digit])
            self.substep_records.append((start,float(self.model.opt.timestep),*c['normal'],*c['skin_normal'],c['palm'],c['hand_other'],c['all_hand_object'],c['support'],c['penetration'],c['world_penetration']))
        self.control_steps+=1
        mujoco.mj_forward(self.model,self.data)
        return self.truth()

    def clone_state(self):
        return dict(qpos=self.data.qpos.copy(),qvel=self.data.qvel.copy(),act=self.data.act.copy(),ctrl=self.data.ctrl.copy(),
                    qacc_warmstart=self.data.qacc_warmstart.copy(),
                    time=self.time,physical_time=float(self.data.time),control_steps=self.control_steps,
                    alpha=self.alpha.copy(),lift_ref=self.lift_ref,phase=self.phase)

    def restore_state(self,state):
        self.data.qpos[:]=state['qpos'];self.data.qvel[:]=state['qvel'];self.data.act[:]=state['act'];self.data.ctrl[:]=state['ctrl']
        self.data.time=state.get('physical_time',state['time']);self.control_steps=int(state.get('control_steps',round(state['time']/self.config.control_dt)))
        self.alpha=state['alpha'].copy();self.lift_ref=state['lift_ref'];self.phase=state['phase']
        self.release_started=None;self._cache=None;mujoco.mj_forward(self.model,self.data)
        if 'qacc_warmstart' in state:self.data.qacc_warmstart[:]=state['qacc_warmstart']
