"""Rigid-body grasp dynamics and explicit noisy/delayed tactile observation boundary."""
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import copy
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / '五指假肢抓握仿真'
DT = .005
GROUPS = [['THJ5','THJ4','THJ3','THJ2','THJ1'], ['FFJ4','FFJ3','FFJ0'],
          ['MFJ4','MFJ3','MFJ0'], ['RFJ4','RFJ3','RFJ0'], ['LFJ5','LFJ4','LFJ3','LFJ0']]
PREFIX = ['rh_th','rh_ff','rh_mf','rh_rf','rh_lf']
OPEN = [np.array(a, dtype=float) for a in
        [[.2,1.15,.1,.2,1.15], [0,0,0], [0,0,0], [0,0,0], [.1,0,0,0]]]
CLOSED = [np.array(a, dtype=float) for a in
          [[1.047,1.15,.1,.2,1.15], [0,.6,2], [0,.6,2], [0,.6,2], [.1,0,.7,2]]]
WEIGHTS = np.array([1.6667, 1., 1., 1., 1.])
_MODEL_CACHE = {}


@dataclass(frozen=True)
class SensorFrame:
    time: float
    normal: np.ndarray
    shear: np.ndarray
    slip: float
    collision: float
    valid: np.ndarray
    age: float


def load_model(shape, size, solref, timestep=.001):
    key = shape, size, solref, timestep
    if key not in _MODEL_CACHE:
        root = ET.fromstring((SOURCE/'model.xml').read_bytes())
        root.find('compiler').set('meshdir','assets')
        opt = root.find('option')
        opt.set('timestep',str(timestep))
        # Keep the original model's solver and torsional/rolling friction for all methods.
        world = root.find('worldbody')
        obj = world.find("body[@name='object']/geom")
        obj.set('type',shape)
        if shape == 'box':
            obj.set('size',f'{.028*size} {.065*size} {.028*size}')
            obj.set('quat','1 0 0 0')
        elif shape == 'sphere':
            obj.set('size',str(.050*size))
            obj.set('quat','1 0 0 0')
            world.find("body[@name='object']").set('pos','.065 0 .3655')
            world.find("geom[@name='support']").set('pos',f'.095 0 {.3655-.050*size-.005}')
        else:
            obj.set('size',f'{.028*size} {.047*size}')
        for geom in world.iter('geom'):
            if geom.get('class') == 'plastic_collision' or geom is obj:
                geom.set('solref',f'{solref} 1')
        obstacle = ET.SubElement(world,'body',name='collision_fixture',mocap='true',pos='.28 0 .49')
        ET.SubElement(obstacle,'geom',name='obstacle_geom',type='box',size='.010 .12 .012',
                      rgba='.8 .2 .2 1',contype='8',conaffinity='3',friction='.35 .005 .0001')
        for geom in world.iter('geom'):
            if geom.get('class') == 'plastic_collision':
                geom.set('conaffinity','11')
        assets = {'assets/'+p.name:p.read_bytes() for p in (SOURCE/'upstream_shadow/assets').glob('*.obj')}
        _MODEL_CACHE[key] = mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'),assets=assets)
        # Bound cache size during parameter sweeps.
        if len(_MODEL_CACHE) > 4:
            oldest = next(iter(_MODEL_CACHE))
            if oldest != key:
                del _MODEL_CACHE[oldest]
    return copy.copy(_MODEL_CACHE[key])


class Plant:
    def __init__(self, scenario, p, timestep=.001):
        self.model = load_model(scenario.shape,p['size'],p['solref'],timestep)
        self.data = mujoco.MjData(self.model)
        self.bid = lambda name:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,name)
        self.aid = lambda name:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,name)
        self.gid = lambda name:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,name)
        self.object = self.bid('object')
        self.forearm = self.bid('rh_forearm')
        self.palm = self.bid('rh_palm')
        self.object_geom = self.gid('object_geom')
        self.palm_geoms = [gid for gid in range(self.model.ngeom)
                           if self.model.geom_bodyid[gid] == self.palm
                           and self.model.geom_contype[gid] != 0]
        self.obstacle_geom = self.gid('obstacle_geom')
        self.obstacle_mocap = self.model.body_mocapid[self.bid('collision_fixture')]
        self.actuators = [[self.aid('rh_A_'+k) for k in group] for group in GROUPS]
        self.lift_motor = self.aid('lift_motor')
        self.geom_finger = {}
        for gid in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,self.model.geom_bodyid[gid]) or ''
            for digit,prefix in enumerate(PREFIX):
                if name.startswith(prefix):
                    self.geom_finger[gid] = digit
        ratio = p['mass']/self.model.body_mass[self.object]
        self.model.body_mass[self.object] *= ratio
        self.model.body_inertia[self.object] *= ratio
        mujoco.mj_setConst(self.model,self.data)
        mujoco.mj_resetData(self.model,self.data)
        for keys,values in zip(GROUPS,OPEN):
            for key,val in zip(keys,values):
                if key.endswith('J0'):
                    continue
                joint = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'rh_'+key)
                self.data.qpos[self.model.jnt_qposadr[joint]] = val
        self.set_friction(p['friction'])
        self.alpha = np.array([.3,.5,.5,.5,.5])
        for indices,values in zip(self.actuators,OPEN):
            self.data.ctrl[indices] = values
        self.filtered_force = np.zeros(5)
        self.phase = 'CLOSING'
        self.contact_since = None
        self.lift_started = None
        self.hold_started = None
        self.release_started = None
        self.release_ctrl = None
        self.release_lift = 0.
        self.lift_ref = 0.
        self.start_position = None
        self.relative_anchor = None
        self.relative_rotation = None
        self.first_grasp_time = None
        self.drop_since = None
        self.dropped = False
        self.failed_setup = False
        self.collision_until = 0.
        self.last_normal = np.zeros(5)
        self.set_actuators()
        mujoco.mj_forward(self.model,self.data)
        self.start_position = self.data.xpos[self.object].copy()

    @property
    def time(self):
        return float(self.data.time)

    def set_friction(self, mu):
        # MuJoCo combines both contacting materials. Include palm contacts so a
        # whole-object friction drop cannot leave a hidden high-friction support.
        for gid in [*self.geom_finger,*self.palm_geoms,self.object_geom]:
            self.model.geom_friction[gid,0] = mu

    def truth(self):
        normal = np.zeros(5)
        shear = np.zeros(5)
        palm = 0.
        collision = 0.
        support = 0.
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            force = np.zeros(6)
            if self.object_geom in (c.geom1,c.geom2):
                other = c.geom2 if c.geom1 == self.object_geom else c.geom1
                mujoco.mj_contactForce(self.model,self.data,i,force)
                if other in self.geom_finger:
                    digit = self.geom_finger[other]
                    normal[digit] += max(0.,force[0])
                    shear[digit] += np.linalg.norm(force[1:3])
                elif self.model.geom_bodyid[other] == self.palm:
                    palm += max(0.,force[0])
                elif self.model.geom_bodyid[other] == 0:
                    support += max(0.,force[0])
            if self.obstacle_geom in (c.geom1,c.geom2):
                other = c.geom2 if c.geom1 == self.obstacle_geom else c.geom1
                if other in self.geom_finger or self.model.geom_bodyid[other] == self.palm:
                    mujoco.mj_contactForce(self.model,self.data,i,force)
                    collision += max(0.,force[0])
        vo,vh = np.zeros(6),np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.object,vo,0)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.palm,vh,0)
        # Remove transport translation and angular velocity: wrist motion is not slip.
        lever = self.data.xipos[self.object]-self.data.xipos[self.palm]
        rel = vo[3:]-vh[3:]-np.cross(vh[:3],lever)
        local_velocity = self.data.xmat[self.palm].reshape(3,3).T @ rel
        slip = float(np.hypot(local_velocity[1],max(0.,-local_velocity[2])))
        rot = self.data.xmat[self.palm].reshape(3,3)
        relative = rot.T @ (self.data.xpos[self.object]-self.data.xpos[self.palm])
        relative_rot = rot.T @ self.data.xmat[self.object].reshape(3,3)
        return dict(normal=normal,shear=shear,slip=slip,palm=palm,collision=collision,
                    support=support,position=self.data.xpos[self.object].copy(),
                    relative=relative,rotation=relative_rot)

    def request_release(self):
        if self.release_started is None:
            self.release_started = self.time
            self.release_ctrl = self.data.ctrl[self.actuators[0]].copy()
            self.release_lift = self.lift_ref
            self.phase = 'RELEASING'

    def task_update(self, obs, command, env):
        t = self.time
        if command == 'release':
            self.request_release()
        if obs.collision > 1.:
            self.collision_until = t+.12
        if self.phase == 'CLOSING':
            required = 3 if self.model.geom_type[self.object_geom] == mujoco.mjtGeom.mjGEOM_SPHERE else 5
            if np.count_nonzero(self.filtered_force > .25) >= required and self.filtered_force[0] > .25:
                if self.contact_since is None:
                    self.contact_since = t
                if t-self.contact_since > .65:
                    self.lift_started = t
                    self.phase = 'LIFTING'
            else:
                self.contact_since = None
            if t > 7.:
                self.failed_setup = True
                self.phase = 'FAILED'
        if self.phase == 'LIFTING':
            self.lift_ref = .10*min(1.,(t-self.lift_started)/1.5)
            if t-self.lift_started > 3.:
                self.phase = 'HOLDING'
                self.hold_started = t
                self.first_grasp_time = t
        if self.phase == 'HOLDING':
            self.lift_ref = .10+env['lift_delta']
            if t < self.collision_until:
                self.lift_ref -= .02
        if self.phase == 'RELEASING':
            self.lift_ref = self.release_lift*(1-np.clip((t-self.release_started-.8)/.6,0,1))

    def execute(self, desired, obs, limits, env):
        # Identical force servo and saturation for every strategy. No truth feedback here.
        finite = np.isfinite(obs.normal) & obs.valid
        self.filtered_force[finite] = .7*self.filtered_force[finite]+.3*obs.normal[finite]
        self.task_update(obs,'release' if self.release_started is not None else 'hold',env)
        if self.phase in ('FAILED','DROPPED'):
            self.alpha = np.maximum(0.,self.alpha-.006)
        elif self.phase == 'RELEASING':
            self.alpha = np.maximum(0.,self.alpha-.006)
        else:
            desired = np.clip(desired,0.,limits)
            update = np.clip((desired-self.filtered_force)*.001,-.006,.004)
            update[~finite] = 0.
            self.alpha += update
            self.alpha -= np.clip((self.filtered_force-limits)*.006,0.,.02)
            # Continuous force-servo state: never jump to a prescribed closed pose
            # when a requested force crosses a threshold. Such a jump bypassed the
            # force loop and favoured outer loops tuned exactly to 1.45 N.
            self.alpha = np.clip(self.alpha,0.,1.2)
        self.set_actuators(env['wrist'] if self.time >= self.collision_until else 0.)

    def set_actuators(self, wrist=0.):
        for digit,indices in enumerate(self.actuators):
            target = np.clip(OPEN[digit]+self.alpha[digit]*(CLOSED[digit]-OPEN[digit]),
                             self.model.actuator_ctrlrange[indices,0],self.model.actuator_ctrlrange[indices,1])
            if self.phase in ('CLOSING','LIFTING'):
                target = np.clip(target,self.data.ctrl[indices]-1.*DT,self.data.ctrl[indices]+1.*DT)
            self.data.ctrl[indices] = target
        if self.release_started is not None:
            elapsed = self.time-self.release_started
            tucked = np.array([-1.03,1.21,.13,-.02,.985])
            full_open = np.array([-1.,.1,0.,0.,0.])
            if elapsed < .8:
                ctrl = self.release_ctrl+np.clip(elapsed/.8,0.,1.)*(tucked-self.release_ctrl)
            elif elapsed < 1.4:
                ctrl = tucked
            else:
                ctrl = tucked+np.clip((elapsed-1.4)/.8,0.,1.)*(full_open-tucked)
            self.data.ctrl[self.actuators[0]] = ctrl
        self.data.ctrl[self.lift_motor] = self.lift_ref+self.model.body_subtreemass[self.forearm]*9.81/1500
        self.data.ctrl[self.aid('rh_A_WRJ2')] = wrist
        self.data.ctrl[self.aid('rh_A_WRJ1')] = 0.

    def advance(self, env):
        self.set_friction(env['friction'])
        self.data.mocap_pos[self.obstacle_mocap] = env['obstacle']
        self.data.xfrc_applied[:] = 0.
        self.data.qfrc_applied[:] = 0.
        # Finite travel of the pulling fixture; avoid indefinite acceleration after loss.
        if abs(self.data.xpos[self.object,1]) < .20:
            self.data.xfrc_applied[self.object,:3] = env['force']
        for key in ['THJ5','FFJ3','MFJ3','RFJ3','LFJ3']:
            joint = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'rh_'+key)
            self.data.qfrc_applied[self.model.jnt_dofadr[joint]] = env['squeeze']
        for _ in range(round(DT/self.model.opt.timestep)):
            mujoco.mj_step(self.model,self.data)


class SensorPipeline:
    def __init__(self, scenario, p, seed):
        self.scenario = scenario
        self.p = p
        self.rng = np.random.default_rng(seed+910_000)
        self.queue = deque(maxlen=100)
        self.ticks = 0

    def sample(self, truth, time):
        sigma = self.p['force_noise']
        normal = np.maximum(0.,truth['normal']+self.rng.normal(0.,sigma,5))
        shear = np.maximum(0.,truth['shear']+self.rng.normal(0.,sigma*.5,5))
        slip = max(0.,truth['slip']+self.rng.normal(0.,.0005+sigma*.004))
        collision = max(0.,truth['collision']+self.rng.normal(0.,sigma))
        valid = np.ones(5,dtype=bool)
        if self.scenario.event == 'noise':
            if 9.5 <= time < 9.62:
                slip += .045
            if 11. <= time < 11.03:
                normal[1] += 2.
        if self.scenario.event == 'dropout':
            if 9.3 <= time < 9.7:
                valid[1:3] = False
            if 12. <= time < 12.15:
                valid[:] = False
        normal[~valid] = np.nan
        shear[~valid] = np.nan
        self.queue.append((time,normal,shear,slip,collision,valid))
        delay_ticks = round(self.p['delay']/DT)
        entry = self.queue[max(0,len(self.queue)-1-delay_ticks)]
        self.ticks += 1
        return SensorFrame(time,entry[1].copy(),entry[2].copy(),float(entry[3]),float(entry[4]),
                           entry[5].copy(),float(time-entry[0]))
