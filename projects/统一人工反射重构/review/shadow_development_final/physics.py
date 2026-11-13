"""Common compliant Shadow execution and a local tactile/evaluation boundary.

Controllers receive SensorFrame, never the full object-state truth dictionary.
All methods act through five calibrated closing-coordinate increments. Contact
forces are unmodified MuJoCo solver outputs, including every physical substep.
"""
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
PREGRASP = [np.array(a, dtype=float) for a in
        [[0.,1.15,.1,.2,1.15], [0,0,0], [0,0,0], [0,0,0], [.1,0,0,0]]]
CLOSED = [np.array(a, dtype=float) for a in
          [[1.047,1.15,.1,.2,1.15], [0,.6,2], [0,.6,2], [0,.6,2], [.1,0,.7,2]]]
PAD_PROXIES = {
    'f_distal_pst': dict(size=(.007772171512916157,.008598503538833105,.014372029705740653),
        pos=(-.00002398440054210861,.00042073253715676145,.0176464753928483),
        quat=(.7030358436460032,-.0009495615836725945,.0009525013178396008,.711153143579491)),
    'th_distal_pst': dict(size=(.009606185921134386,.010583471741189761,.01618607736207571),
        pos=(-.0000008254521460715179,.0008025187771040246,.019553373070678535),
        quat=(.7071507168146272,-.0011598706415721226,.001084121970924425,.7070610603676373)),
}
# Local development-cylinder central differences; not a universal force map.
FORCE_GAIN = np.array([5.41, 7.33, 34.24, 34.03, 19.28])
FORCE_TAU = .04
BARE_HAND_EFFORT_OFFSET = .010
MAX_DELTA = .003
PREGRASP_CENTER_HEIGHT = .3655
LIFT_HEIGHT = .160
TASK_LIFT = LIFT_HEIGHT
DEFAULT_PREGRASP_OFFSETS = {'cylinder':(0.,0.,0.), 'box':(0.,.012,0.),
                           'sphere':(0.,.012,0.)}
SUBSTEP_COLUMNS = ('solver_time', *(f'normal_{i}_N' for i in range(5)),
    *(f'shear_{i}_N' for i in range(5)),
    'palm_N','support_N','collision_N','penetration_m','world_penetration_m','duration_s')
SUBSTEP_COLUMNS = (*SUBSTEP_COLUMNS, *(f'skin_normal_{i}_N' for i in range(5)))
SUBSTEP_COLUMNS = (*SUBSTEP_COLUMNS, 'hand_other_N', 'all_hand_object_N')
_MODEL_CACHE = {}


@dataclass(frozen=True)
class SensorFrame:
    time: float
    normal: np.ndarray
    tangential: np.ndarray
    contact_pos: np.ndarray
    contact_normal: np.ndarray
    tangent_basis: np.ndarray
    slip_local: np.ndarray
    joint: np.ndarray
    joint_velocity: np.ndarray
    valid: np.ndarray
    age: float
    alpha: np.ndarray
    phase: str
    lift: float
    lift_ref: float
    support_axis: np.ndarray
    raw_joint: np.ndarray
    raw_joint_velocity: np.ndarray
    force_gain: np.ndarray
    closure_permission: np.ndarray
    closure_max: np.ndarray
    closure_min: np.ndarray
    motor_effort: float
    handload_prior: float
    load_estimate: float
    load_uncertainty: float
    load_valid: bool
    load_confidence: float
    force_tau: float = FORCE_TAU
    dt: float = DT


def _tangent_basis(normal):
    """Stable local coordinates: project hand y, then right-handed second axis."""
    n = np.asarray(normal, dtype=float)
    if np.linalg.norm(n) < .1:
        return np.array([[0., 1., 0.], [0., 0., 1.]])
    n = n / np.linalg.norm(n)
    axis = np.array([0., 1., 0.])
    if abs(n @ axis) > .85:
        axis = np.array([0., 0., 1.])
    t1 = axis - (axis @ n)*n
    t1 /= np.linalg.norm(t1)
    return np.stack([t1, np.cross(n, t1)])


def load_model(shape, size, solref=.025, timestep=.0005,pregrasp_offset=None,vertical=False,
               box_half=(.028,.065,.028),box_angle=0.):
    """Preserve Shadow morphology/inertias; rebuild common compliant actuation."""
    pregrasp_offset=DEFAULT_PREGRASP_OFFSETS[shape] if pregrasp_offset is None else pregrasp_offset
    key = shape, size, solref, timestep,tuple(pregrasp_offset),bool(vertical),tuple(box_half),float(box_angle)
    if key not in _MODEL_CACHE:
        root = ET.fromstring((SOURCE/'model.xml').read_bytes())
        root.find('compiler').set('meshdir', 'assets')
        opt = root.find('option')
        opt.set('timestep', str(timestep))
        opt.set('iterations', '50')
        opt.set('noslip_iterations', '5')
        opt.set('impratio', '100')
        opt.set('tolerance', '1e-10')
        world = root.find('worldbody')
        world.find("body[@name='rh_forearm']/joint[@name='transport_lift']").set('range','0 .18')
        body = world.find("body[@name='object']")
        obj = body.find('geom')
        obj.set('type', shape)
        if shape == 'box':
            obj.set('size',' '.join(str(v*size) for v in box_half))
            obj.set('quat', f'{np.cos(box_angle/2)} {np.sin(box_angle/2)} 0 0')
        elif shape == 'sphere':
            obj.set('size', str(.050*size))
            obj.set('quat', '1 0 0 0')
            body.set('pos', f'.065 0 {PREGRASP_CENTER_HEIGHT}')
            world.find("geom[@name='support']").set('pos', f'.095 0 {PREGRASP_CENTER_HEIGHT-.050*size-.005}')
        else:
            obj.set('size', f'{.028*size} {.047*size}')
            if vertical:obj.set('quat','1 0 0 0')
        if shape != 'sphere':
            body.set('pos', f'{.027+.028*size} 0 {PREGRASP_CENTER_HEIGHT}')
            halfheight = ((abs(np.sin(box_angle))*box_half[1]+abs(np.cos(box_angle))*box_half[2])
                          if shape=='box' else .047 if vertical else .028)*size
            world.find("geom[@name='support']").set('pos', f'.095 0 {PREGRASP_CENTER_HEIGHT-halfheight-.005}')
        pose=np.fromstring(body.get('pos'),sep=' ')+np.asarray(pregrasp_offset)
        body.set('pos',' '.join(str(x) for x in pose))
        support=world.find("geom[@name='support']")
        tablepose=np.fromstring(support.get('pos'),sep=' ')
        tablepose[2]+=float(pregrasp_offset[2])
        support.set('pos',' '.join(str(x) for x in tablepose))
        support_geom = world.find("geom[@name='support']")
        table_center = float(support_geom.get('pos').split()[2])
        post_half = (table_center-.005)/2.
        world.find("geom[@name='support_post']").set('pos', f'.10 0 {post_half}')
        world.find("geom[@name='support_post']").set('size', f'.025 .045 {post_half}')
        obstacle = ET.SubElement(world, 'body', name='collision_fixture', mocap='true', pos='.28 0 .49')
        ET.SubElement(obstacle, 'geom', name='obstacle_geom', type='box', size='.010 .12 .012',
            rgba='.8 .2 .2 1', contype='8', conaffinity='3', friction='.35 .001 .0001')
        for geom in world.iter('geom'):
            if geom.get('class') == 'plastic_collision':
                geom.set('conaffinity', '11')
                proxy = PAD_PROXIES.get(geom.get('mesh'))
                if proxy is not None and geom.get('type') == 'mesh':
                    geom.set('conaffinity', '11')
                    geom.set('type', 'box')
                    geom.attrib.pop('mesh')
                    for attribute, values in proxy.items():
                        geom.set(attribute, ' '.join(str(v) for v in values))
            if geom.get('class') == 'plastic_collision' or geom.get('contype', '0') != '0':
                geom.set('solref', f'{max(.012, solref)} 1')
                geom.set('solimp', '.90 .95 .002 .5 2')
                geom.set('condim', '3')
                geom.set('friction', '.6 .001 .0001')
        for actuator in root.find('actuator'):
            name = actuator.get('name', '')
            if name == 'lift_motor':
                # Shared transport motor has gravity feedforward and rate-limited reference.
                actuator.set('kp', '1000')
                actuator.set('kv', '100')
                actuator.set('forcerange', '-75 75')
                actuator.set('ctrlrange', '0 .25')
            elif name in ('rh_A_WRJ1', 'rh_A_WRJ2'):
                actuator.set('kp', '4')
                actuator.set('kv', '.3')
                actuator.set('forcerange', '-1 1')
            else:
                actuator.set('kp', '.8')
                actuator.set('kv', '.025')
                actuator.set('forcerange', '-.16 .16')
                if name == 'rh_A_THJ5':
                    actuator.set('kp', '.7')
                    actuator.set('forcerange', '-.30 .30')
                elif name.startswith('rh_A_TH'):
                    actuator.set('forcerange', '-.25 .25')
        assets = {'assets/'+p.name:p.read_bytes() for p in (SOURCE/'upstream_shadow/assets').glob('*.obj')}
        _MODEL_CACHE[key] = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'), assets=assets)
        if len(_MODEL_CACHE) > 4:
            del _MODEL_CACHE[next(iter(_MODEL_CACHE))]
    return copy.copy(_MODEL_CACHE[key])


class Plant:
    def __init__(self, scenario, p, timestep=.0005):
        self.scenario, self.p = scenario, dict(p)
        self.model = load_model(scenario.shape, p['size'], p.get('solref', .025), timestep,
            p.get('pregrasp_offset'),p.get('vertical',False),p.get('box_half',(.028,.065,.028)),p.get('box_angle',0.))
        self.data = mujoco.MjData(self.model)
        self.bid = lambda name:mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        self.aid = lambda name:mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        self.gid = lambda name:mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        self.object, self.forearm, self.palm = map(self.bid, ['object','rh_forearm','rh_palm'])
        self.object_geom, self.obstacle_geom = map(self.gid, ['object_geom','obstacle_geom'])
        self.obstacle_mocap = self.model.body_mocapid[self.bid('collision_fixture')]
        self.actuators = [[self.aid('rh_A_'+k) for k in group] for group in GROUPS]
        self.open_pose = [x.copy() for x in PREGRASP]
        self.closed_pose = [x.copy() for x in CLOSED]
        if 'thumb_j4' in p:
            self.open_pose[0][1]=self.closed_pose[0][1]=float(p['thumb_j4'])
        if 'thumb_j5_open' in p:
            self.open_pose[0][0]=float(p['thumb_j5_open'])
        self.lift_motor = self.aid('lift_motor')
        self.lift_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'transport_lift')
        self.lift_qadr = self.model.jnt_qposadr[self.lift_joint]
        self.lift_dadr = self.model.jnt_dofadr[self.lift_joint]
        self.geom_finger, self.hand_geoms = {}, set()
        for gid in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[gid]) or ''
            if name.startswith('rh_'):
                self.hand_geoms.add(gid)
            for digit, prefix in enumerate(PREFIX):
                if name.startswith(prefix):
                    self.geom_finger[gid] = digit
        self.hand_joints = [j for j in range(self.model.njnt) if
            (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or '').startswith('rh_')]
        self.hand_qadr = self.model.jnt_qposadr[self.hand_joints]
        self.hand_dadr = self.model.jnt_dofadr[self.hand_joints]
        ratio = p['mass']/self.model.body_mass[self.object]
        self.model.body_mass[self.object] *= ratio
        self.model.body_inertia[self.object] *= ratio
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_resetData(self.model, self.data)
        for keys, values in zip(GROUPS, self.open_pose):
            for key, val in zip(keys, values):
                if key.endswith('J0'):
                    for suffix in ('J1','J2'):
                        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'rh_'+key[:-2]+suffix)
                        self.data.qpos[self.model.jnt_qposadr[joint]] = val/2.
                    continue
                joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'rh_'+key)
                self.data.qpos[self.model.jnt_qposadr[joint]] = val
        self._control_tick = 0
        self.alpha = np.zeros(5)
        self.closure_max=np.full(5,np.inf)
        self.closure_min=np.full(5,-np.inf)
        for digit,indices in enumerate(self.actuators):
            direction=self.closed_pose[digit]-self.open_pose[digit]
            for idx,origin,change in zip(indices,self.open_pose[digit],direction):
                if change>1e-9:
                    self.closure_max[digit]=min(self.closure_max[digit],(self.model.actuator_ctrlrange[idx,1]-origin)/change)
                    self.closure_min[digit]=max(self.closure_min[digit],(self.model.actuator_ctrlrange[idx,0]-origin)/change)
                elif change<-1e-9:
                    self.closure_max[digit]=min(self.closure_max[digit],(self.model.actuator_ctrlrange[idx,0]-origin)/change)
                    self.closure_min[digit]=max(self.closure_min[digit],(self.model.actuator_ctrlrange[idx,1]-origin)/change)
        self.closure_max=np.minimum(self.closure_max,float(p.get('alpha_max',1.55)))
        self.phase = 'CLOSING'
        self.filtered_force = np.zeros(5)
        self.contact_since = None
        self.lift_started = self.hold_started = self.release_started = None
        self.lift_ref = 0.
        self.task_lift_height = TASK_LIFT
        self.release_lift = 0.
        self.failed_setup = False
        # Truth-based failure remains a recording concern, never controls the plant.
        self.limits = np.asarray(scenario.limit, dtype=float)
        self.budget = float(scenario.budget)
        self.substep_records = []
        self.set_friction(p['friction'])
        self._set_actuators(0.)
        mujoco.mj_forward(self.model, self.data)
        self.start_position = self.data.xpos[self.object].copy()
        self.reference_relative = self._relative_pose()[0]
        self.reference_rotation = self._relative_pose()[1]
        self.initial_penetration = self.truth()['penetration']

    @property
    def time(self):
        return self._control_tick * DT

    def set_friction(self, mu):
        for gid in [*self.geom_finger, self.object_geom]:
            self.model.geom_friction[gid, 0] = mu
        for gid in self.hand_geoms:
            self.model.geom_friction[gid, 0] = mu

    def _relative_pose(self):
        R = self.data.xmat[self.palm].reshape(3,3)
        return R.T @ (self.data.xpos[self.object]-self.data.xpos[self.palm]), R.T @ self.data.xmat[self.object].reshape(3,3)

    def _contacts(self, velocities=False):
        normal, force_world, pos_sum, normal_sum = np.zeros(5), np.zeros((5,3)), np.zeros((5,3)), np.zeros((5,3))
        velocity_sum = np.zeros((5,3))
        palm = support = collision = penetration = world_penetration = other_hand = 0.
        R = self.data.xmat[self.palm].reshape(3,3)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            force = np.zeros(6)
            if self.object_geom in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self.object_geom else c.geom1
                mujoco.mj_contactForce(self.model, self.data, i, force)
                amount = max(0., float(force[0]))
                if other in self.hand_geoms:
                    penetration = max(penetration, max(0., -float(c.dist)))
                elif self.model.geom_bodyid[other] == 0:
                    world_penetration = max(world_penetration, max(0., -float(c.dist)))
                if other in self.geom_finger:
                    digit = self.geom_finger[other]
                    # mj_contactForce is the force on geom2, in contact coordinates.
                    sign = 1. if c.geom2 == self.object_geom else -1.
                    inward = sign * c.frame[:3]
                    normal[digit] += amount
                    force_world[digit] += sign * c.frame.reshape(3,3).T @ force[:3]
                    pos_sum[digit] += amount * (R.T @ (c.pos-self.data.xpos[self.palm]))
                    normal_sum[digit] += amount * (R.T @ inward)
                    if velocities and amount > 0.:
                        vo, vf = np.zeros(6), np.zeros(6)
                        finger_body = int(self.model.geom_bodyid[other])
                        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.object, vo, 0)
                        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, finger_body, vf, 0)
                        # Spatial body velocity is centred at inertial COM xipos.
                        vop = vo[3:] + np.cross(vo[:3], c.pos-self.data.xipos[self.object])
                        vfp = vf[3:] + np.cross(vf[:3], c.pos-self.data.xipos[finger_body])
                        velocity_sum[digit] += amount * (R.T @ (vop-vfp))
                elif self.model.geom_bodyid[other] == self.palm:
                    palm += amount
                elif other in self.hand_geoms:
                    other_hand += amount
                elif self.model.geom_bodyid[other] == 0:
                    support += amount
            if self.obstacle_geom in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self.obstacle_geom else c.geom1
                if other in self.hand_geoms:
                    mujoco.mj_contactForce(self.model, self.data, i, force)
                    collision += max(0., float(force[0]))
        contact_pos, contact_normal, tangent, slip = np.zeros((5,3)), np.zeros((5,3)), np.zeros((5,2)), np.zeros((5,2))
        basis = np.zeros((5,2,3))
        for i in range(5):
            if normal[i] > 1e-8:
                contact_pos[i] = pos_sum[i]/normal[i]
                length = np.linalg.norm(normal_sum[i])
                if length > 1e-8:
                    contact_normal[i] = normal_sum[i]/length
                basis[i] = _tangent_basis(contact_normal[i])
                tangent[i] = basis[i] @ (R.T @ force_world[i])
                slip[i] = basis[i] @ (velocity_sum[i]/normal[i])
            else:
                basis[i] = _tangent_basis(np.zeros(3))
        return dict(normal=normal, tangential=tangent, contact_pos=contact_pos,
            contact_normal=contact_normal, tangent_basis=basis, slip_local=slip,
            palm=palm, support=support, collision=collision, penetration=penetration,
            world_penetration=world_penetration, hand_other=other_hand,
            all_hand_object=float(normal.sum()+palm+other_hand))

    def _substep_forces(self):
        """Raw force vectors at every solver tick without redundant kinematics."""
        normal, shear, skin = np.zeros(5), np.zeros(5), np.zeros(5)
        palm = support = collision = penetration = world_penetration = other_hand = 0.
        force = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for finger_geom, other in ((c.geom1,c.geom2),(c.geom2,c.geom1)):
                if finger_geom in self.geom_finger and other not in self.hand_geoms:
                    mujoco.mj_contactForce(self.model,self.data,i,force)
                    skin[self.geom_finger[finger_geom]] += max(0.,float(force[0]))
            if self.object_geom in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self.object_geom else c.geom1
                mujoco.mj_contactForce(self.model, self.data, i, force)
                amount = max(0., float(force[0]))
                if other in self.hand_geoms:
                    penetration = max(penetration, max(0., -float(c.dist)))
                elif self.model.geom_bodyid[other] == 0:
                    world_penetration = max(world_penetration, max(0., -float(c.dist)))
                if other in self.geom_finger:
                    digit = self.geom_finger[other]
                    normal[digit] += amount
                    shear[digit] += np.linalg.norm(force[1:3])
                elif self.model.geom_bodyid[other] == self.palm:
                    palm += amount
                elif other in self.hand_geoms:
                    other_hand += amount
                elif self.model.geom_bodyid[other] == 0:
                    support += amount
            if self.obstacle_geom in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self.obstacle_geom else c.geom1
                if other in self.hand_geoms:
                    mujoco.mj_contactForce(self.model, self.data, i, force)
                    collision += max(0., float(force[0]))
        return (normal, shear, palm, support, collision, penetration, world_penetration, skin,other_hand)

    def _skin_contacts(self):
        """External contact on instrumented fingers, independent of object identity.

        Internal hand collision constraints are excluded using the registered own
        robot geometry. Object, fixture and world/support contacts are all sensed.
        This models an external-contact virtual tactile skin, not a real sensor.
        """
        pressure=np.zeros(5);pos=np.zeros((5,3));direction=np.zeros((5,3))
        loads=np.zeros((5,3));velocity=np.zeros((5,3))
        R=self.data.xmat[self.palm].reshape(3,3)
        for i in range(self.data.ncon):
            c=self.data.contact[i]
            for finger_geom,other,sign in ((c.geom1,c.geom2,1.),(c.geom2,c.geom1,-1.)):
                if finger_geom not in self.geom_finger or other in self.hand_geoms:continue
                force=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,i,force)
                amount=max(0.,float(force[0]));d=self.geom_finger[finger_geom]
                pressure[d]+=amount
                pos[d]+=amount*(R.T@(c.pos-self.data.xpos[self.palm]))
                direction[d]+=amount*(R.T@(sign*c.frame[:3]))
                loads[d]+=R.T@(sign*c.frame.reshape(3,3).T@force[:3])
                other_body=int(self.model.geom_bodyid[other]);finger_body=int(self.model.geom_bodyid[finger_geom])
                vo,vf=np.zeros(6),np.zeros(6)
                mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,other_body,vo,0)
                mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,finger_body,vf,0)
                vop=vo[3:]+np.cross(vo[:3],c.pos-self.data.xipos[other_body])
                vfp=vf[3:]+np.cross(vf[:3],c.pos-self.data.xipos[finger_body])
                velocity[d]+=amount*(R.T@(vop-vfp))
        basis=np.zeros((5,2,3));tangent=np.zeros((5,2));slip=np.zeros((5,2))
        for d in range(5):
            if pressure[d]>1e-8:
                pos[d]/=pressure[d]
                length=np.linalg.norm(direction[d])
                if length>1e-8:direction[d]/=length
                basis[d]=_tangent_basis(direction[d]);tangent[d]=basis[d]@loads[d]
                slip[d]=basis[d]@(velocity[d]/pressure[d])
            else:basis[d]=_tangent_basis(np.zeros(3))
        return dict(normal=pressure,tangential=tangent,contact_pos=pos,
            contact_normal=direction,tangent_basis=basis,slip_local=slip)

    def truth(self):
        tactile = self._contacts(velocities=True)
        vo, vh = np.zeros(6), np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.object, vo, 0)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.palm, vh, 0)
        R = self.data.xmat[self.palm].reshape(3,3)
        lever = self.data.xipos[self.object]-self.data.xipos[self.palm]
        relative, rotation = self._relative_pose()
        joint, joint_velocity = np.zeros(5), np.zeros(5)
        for i, indices in enumerate(self.actuators):
            direction = self.closed_pose[i]-self.open_pose[i]
            denom = float(direction @ direction)
            joint[i] = direction @ (self.data.actuator_length[indices]-self.open_pose[i])/denom
            joint_velocity[i] = direction @ self.data.actuator_velocity[indices]/denom
        tactile.update(position=self.data.xpos[self.object].copy(), relative=relative,
            rotation=rotation, relvel=R.T @ (vo[3:]-vh[3:]-np.cross(vh[:3], lever)),
            relomega=R.T @ (vo[:3]-vh[:3]), joint=joint, joint_velocity=joint_velocity,
            raw_joint=self.data.qpos[self.hand_qadr].copy(), raw_joint_velocity=self.data.qvel[self.hand_dadr].copy(),
            alpha=self.alpha.copy(), phase=self.phase, lift=float(self.data.qpos[self.lift_qadr]),
            closure_max=self.closure_max.copy(),
            closure_min=self.closure_min.copy(),
            lift_velocity=float(self.data.qvel[self.lift_dadr]),
            motor_effort=float(self.data.actuator_force[self.lift_motor]),
            bare_hand_weight=float(self.model.body_subtreemass[self.forearm]*9.81),
            known_hand_inertia=float(self.model.body_subtreemass[self.forearm]+self.model.dof_armature[self.lift_dadr]),
            lift_damping=float(self.model.dof_damping[self.lift_dadr]),
            lift_ref=self.lift_ref, support_axis=R.T @ np.array([0.,0.,1.]))
        tactile['relative_velocity'] = tactile['relvel'].copy()
        tactile['angular_velocity'] = tactile['relomega'].copy()
        tactile['tactile'] = self._skin_contacts()
        return tactile

    def task_update(self, obs, command='hold', pause_transport=False):
        t = self.time
        if command == 'release' and self.release_started is None:
            self.release_started, self.release_lift = t, self.lift_ref
            self.phase = 'RELEASING'
        finite = obs.valid & np.isfinite(obs.normal)
        self.filtered_force[finite] = .75*self.filtered_force[finite] + .25*obs.normal[finite]
        # Shared stage logic uses tactile observations only, and has no success oracle.
        required = 3
        if self.phase == 'CLOSING':
            contact = np.count_nonzero(finite & (self.filtered_force > .22)) >= required and finite[0] and self.filtered_force[0] > .22
            if contact:
                if self.contact_since is None:
                    self.contact_since = t
                if t-self.contact_since >= .35:
                    self.lift_started, self.phase = t, 'LIFTING'
            else:
                self.contact_since = None
            if t >= 7.:
                self.failed_setup, self.phase = True, 'FAILED'
        if self.phase == 'LIFTING' and not pause_transport:
            self.lift_ref = min(LIFT_HEIGHT, self.lift_ref + LIFT_HEIGHT/1.5*DT)
            if self.lift_ref >= LIFT_HEIGHT-1e-9 and abs(obs.lift-LIFT_HEIGHT) < .006:
                self.hold_started, self.phase = t, 'HOLDING'
        if self.phase == 'RELEASING':
            # User release decouples transport: hold pose during contact removal.
            self.lift_ref = self.release_lift

    def execute(self, action_increment, obs, env=None, command='hold'):
        env = {} if env is None else env
        pause = bool(getattr(action_increment, 'pause_transport', False))
        delta = np.asarray(getattr(action_increment, 'delta_closure', action_increment), dtype=float)
        if delta.shape != (5,) or not np.all(np.isfinite(delta)):
            raise ValueError('Action must contain five finite closing-coordinate increments')
        self.task_update(obs, command, pause)
        permission = np.asarray(getattr(obs, 'closure_permission', np.ones(5,bool)))
        delta = delta.copy()
        delta[~permission] = np.minimum(delta[~permission], 0.)
        if self.phase == 'FAILED':
            delta = np.full(5, -MAX_DELTA)
        # Release is a commanded task; it uses the same five motion coordinates.
        if self.phase == 'RELEASING':
            delta = np.minimum(delta, 0.)
        self.alpha = np.clip(self.alpha + np.clip(delta, -MAX_DELTA, MAX_DELTA), self.closure_min, self.closure_max)
        if self.phase == 'HOLDING' and not pause:
            self.lift_ref = np.clip(LIFT_HEIGHT + float(env.get('lift_delta', 0.)), 0., .177)
        self._set_actuators(float(env.get('wrist', 0.)))

    def _set_actuators(self, wrist=0.):
        for i, indices in enumerate(self.actuators):
            target = np.clip(self.open_pose[i]+self.alpha[i]*(self.closed_pose[i]-self.open_pose[i]),
                self.model.actuator_ctrlrange[indices,0], self.model.actuator_ctrlrange[indices,1])
            self.data.ctrl[indices] = target
        gravity = self.model.body_subtreemass[self.forearm]*9.81/1000.
        target = self.lift_ref + gravity
        self.data.ctrl[self.lift_motor] = np.clip(target,
            self.data.ctrl[self.lift_motor]-.08*DT, self.data.ctrl[self.lift_motor]+.08*DT)
        self.data.ctrl[self.aid('rh_A_WRJ2')] = wrist
        self.data.ctrl[self.aid('rh_A_WRJ1')] = 0.

    def advance(self, env):
        self.set_friction(env['friction'])
        self.data.mocap_pos[self.obstacle_mocap] = env.get('obstacle', [.28,0,.49])
        self.data.xfrc_applied[:] = 0.
        self.data.qfrc_applied[:] = 0.
        # Environment fixture travel bounds belong to the apparatus, not controller.
        if abs(self.data.xpos[self.object,1]) < .20:
            self.data.xfrc_applied[self.object,:3] = env.get('force', [0.,0.,0.])
        for key in ['THJ5','FFJ3','MFJ3','RFJ3','LFJ3']:
            joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'rh_'+key)
            self.data.qfrc_applied[self.model.jnt_dofadr[joint]] = float(env.get('squeeze',0.))
        for _ in range(round(DT/self.model.opt.timestep)):
            solver_time = float(self.data.time)
            mujoco.mj_step(self.model, self.data)
            normal, shear, palm, support, collision, penetration, world_penetration, skin,other_hand = self._substep_forces()
            self.substep_records.append((solver_time, *normal, *shear, palm, support,
                collision, penetration, world_penetration, float(self.model.opt.timestep), *skin,
                other_hand,float(normal.sum()+palm+other_hand)))
        mujoco.mj_forward(self.model, self.data)
        self._control_tick += 1


class SensorPipeline:
    """Generate only local tactile and hand encoder observations, with impairment."""
    def __init__(self, scenario, p, seed):
        self.scenario, self.p = scenario, dict(p)
        self.rng = np.random.default_rng(seed+910_000)
        self.queue = deque(maxlen=100)
        self.previous_lift_velocity = None
        self.encoder_acceleration = 0.
        self.load_estimate = 0.

    def sample(self, truth, time):
        sigma = float(self.p.get('force_noise', self.scenario.noise))
        skin=truth['tactile']
        normal = np.maximum(0., skin['normal'] + self.rng.normal(0., sigma, 5))
        tangential = skin['tangential'] + self.rng.normal(0., sigma*.5, (5,2))
        position = skin['contact_pos'].copy()
        contact = skin['normal'] > .02
        position[contact] += self.rng.normal(0., .00025, (np.count_nonzero(contact),3))
        slip = skin['slip_local'] + self.rng.normal(0., .00025+sigma*.002, (5,2))
        slip[~contact] = 0.
        valid = np.ones(5, dtype=bool)
        if self.scenario.event == 'noise':
            if 9.5 <= time < 9.62:
                slip[1,0] += .045
            if 11. <= time < 11.03:
                normal[1] += 2.
        if self.scenario.event == 'dropout':
            if 9.3 <= time < 9.7:
                valid[1:3] = False
            if 12. <= time < 12.15:
                valid[:] = False
        contact_normal=skin['contact_normal'].copy();basis=skin['tangent_basis'].copy()
        for values in (normal, tangential, position, slip,contact_normal,basis):
            values[~valid] = np.nan
        self.queue.append(dict(time=time, normal=normal, tangential=tangential,
            contact_pos=position, contact_normal=contact_normal,
            tangent_basis=basis, slip_local=slip,
            valid=valid))
        delay = round(float(self.p.get('delay', .01))/DT)
        entry = self.queue[max(0,len(self.queue)-1-delay)]
        # Tactile delay is explicit; contemporaneous encoders/task references are separate.
        permission = np.ones(5, dtype=bool)
        if truth['phase'] == 'CLOSING' and self.scenario.shape != 'box':
            fingers = entry['valid'][1:] & np.isfinite(entry['normal'][1:]) & (entry['normal'][1:] > .20)
            permission[0] = np.count_nonzero(fingers) >= 2 or np.median(truth['joint'][1:]) >= .40
        encoder_velocity=float(truth['lift_velocity'])
        raw_accel=0. if self.previous_lift_velocity is None else (encoder_velocity-self.previous_lift_velocity)/DT
        self.previous_lift_velocity=encoder_velocity
        self.encoder_acceleration += (1.-np.exp(-DT/.025))*(raw_accel-self.encoder_acceleration)
        # Known own-hand mass/inertia and encoder dynamics only. No unknown object
        # mass, friction, full-object movement or evaluation state is consumed.
        prior=(truth['bare_hand_weight']+BARE_HAND_EFFORT_OFFSET+truth['known_hand_inertia']*self.encoder_acceleration
               +truth['lift_damping']*encoder_velocity)
        residual=float(truth['motor_effort']-prior)
        self.load_estimate += (1.-np.exp(-DT/.045))*(residual-self.load_estimate)
        finger_speed=float(np.linalg.norm(truth['joint_velocity']))
        quiet=abs(self.encoder_acceleration)<.05 and abs(encoder_velocity)<.005 and finger_speed<.02
        uncertainty=(.10 if quiet else .55)+.10*abs(self.encoder_acceleration)+.25*finger_speed
        load_valid=bool(truth['lift']>.01 and truth['phase'] in ('LIFTING','HOLDING')
                        and abs(self.encoder_acceleration)<2.)
        confidence=.85 if load_valid and quiet else .5 if load_valid else 0.
        return SensorFrame(time=time, normal=entry['normal'].copy(),
            tangential=entry['tangential'].copy(), contact_pos=entry['contact_pos'].copy(),
            contact_normal=entry['contact_normal'].copy(), tangent_basis=entry['tangent_basis'].copy(),
            slip_local=entry['slip_local'].copy(), joint=truth['joint'].copy(),
            joint_velocity=truth['joint_velocity'].copy(), valid=entry['valid'].copy(),
            age=float(time-entry['time']), alpha=truth['alpha'].copy(), phase=truth['phase'],
            lift=truth['lift'], lift_ref=truth['lift_ref'], support_axis=truth['support_axis'].copy(),
            raw_joint=truth['raw_joint'].copy(), raw_joint_velocity=truth['raw_joint_velocity'].copy(),
            force_gain=FORCE_GAIN.copy(),closure_permission=permission,closure_max=truth['closure_max'].copy(),
            closure_min=truth['closure_min'].copy(),
            motor_effort=float(truth['motor_effort']),handload_prior=float(prior),
            load_estimate=max(0.,float(self.load_estimate)),load_uncertainty=float(uncertainty),
            load_valid=load_valid,load_confidence=confidence)
