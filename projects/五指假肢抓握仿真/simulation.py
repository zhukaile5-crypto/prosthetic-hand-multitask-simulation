"""Five articulated digits, free-object contact dynamics and three reflex modules."""
from pathlib import Path
from collections import deque
import os,struct,zlib
import mujoco,numpy as np
ROOT=Path(__file__).resolve().parent
FINGER_NAMES=['拇指','食指','中指','无名指','小指']
PREFIX=['rh_th','rh_ff','rh_mf','rh_rf','rh_lf']
GROUPS=[['THJ5','THJ4','THJ3','THJ2','THJ1'],['FFJ4','FFJ3','FFJ0'],['MFJ4','MFJ3','MFJ0'],['RFJ4','RFJ3','RFJ0'],['LFJ5','LFJ4','LFJ3','LFJ0']]
OPEN=[np.array(a) for a in [[.2,1.15,.1,.2,1.15],[0,0,0],[0,0,0],[0,0,0],[.1,0,0,0]]]
CLOSED=[np.array(a) for a in [[1.047,1.15,.1,.2,1.15],[0,.6,2],[0,.6,2],[0,.6,2],[.1,0,.7,2]]]

def png_bytes(pixels):
    h,w,_=pixels.shape
    def chunk(tag,payload):return struct.pack('!I',len(payload))+tag+payload+struct.pack('!I',zlib.crc32(tag+payload)&0xffffffff)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(b''.join(b'\0'+row.tobytes() for row in pixels),2))+chunk(b'IEND',b'')

class GraspSimulation:
    def __init__(self):
        old=os.getcwd()
        try:os.chdir(ROOT);self.model=mujoco.MjModel.from_xml_path('model.xml')
        finally:os.chdir(old)
        self.data=mujoco.MjData(self.model)
        self.bid=lambda name:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,name)
        self.aid=lambda name:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,name)
        self.object=self.bid('object');self.forearm=self.bid('rh_forearm')
        self.object_geom=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,'object_geom')
        self.actuator_groups=[[self.aid('rh_A_'+k) for k in keys] for keys in GROUPS]
        self.lift_actuator=self.aid('lift_motor')
        self.geom_finger={}
        for gid in range(self.model.ngeom):
            name=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,self.model.geom_bodyid[gid]) or ''
            for j,pre in enumerate(PREFIX):
                if name.startswith(pre):self.geom_finger[gid]=j
        self.enabled={'slip':True,'force':True,'gate':True};self.mu=.6;self.limit=5.
        self.base_force=1.8;self.disturb_force=6.;self.squeeze_torque=.15
        self.camera={'azimuth':150.,'elevation':-15.,'distance':.60}
        self.history=deque(maxlen=4000);self.events=deque(maxlen=60)
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model,self.data)
        for keys,values in zip(GROUPS,OPEN):
            for key,val in zip(keys,values):
                if key.endswith('J0'):continue
                j=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'rh_'+key)
                self.data.qpos[self.model.jnt_qposadr[j]]=val
        for gid in [*self.geom_finger,self.object_geom]:self.model.geom_friction[gid,0]=self.mu
        self.state='READY';self.alpha=np.zeros(5);self.normal=np.zeros(5);self.filtered=np.zeros(5)
        self.palm_force=0.;self.tip_forces=np.zeros(5);self.lift_ref=0.;self.force_ref=self.base_force
        self.slip_speed=0.;self.contact_since=None;self.lift_start=None;self.release_time=None;self.release_thumb_start=None;self.release_lift_start=0.
        self.disturb_until=0.;self.squeeze_until=0.;self.slip_until=0.;self.force_until=0.;self.last_slip=-1.
        self.slip_events=0;self.force_events=0;self.release_slip_events=0;self.peak_force=0.;self.steps=0
        self.history.clear();self.events.clear();mujoco.mj_forward(self.model,self.data)
        self.initial_z=self.object_z;self.initial_y=float(self.data.xpos[self.object,1]);self.set_actuators()
        self.event('reset','五指手已复位，物体为自由刚体')

    @property
    def object_z(self):return float(self.data.xpos[self.object,2])

    def event(self,kind,message):self.events.append(dict(time=round(float(self.data.time),3),kind=kind,message=message))

    def command(self,action,value=None):
        if action=='reset':self.reset()
        elif action=='grasp':
            self.reset();self.state='CLOSING';self.alpha[:]=[.3,.5,.5,.5,.5]
            self.force_ref=min(self.base_force,(self.limit-.2)/1.6667)
            self.event('grasp','五指协同闭合，建立接触后抬升')
        elif action=='release':
            if self.state.startswith('RELEAS'):return
            self.state='RELEASING';self.release_time=float(self.data.time)
            self.release_thumb_start=self.data.ctrl[self.actuator_groups[0]].copy()
            self.release_lift_start=self.lift_ref
            self.event('release','主动释放：阻断防滑补力，五指张开' if self.enabled['gate'] else '释放门控关闭：降低持握目标，防滑仍可补力')
        elif action=='disturb':
            self.disturb_until=float(self.data.time)+.65
            self.event('disturb',f'沿物体长轴施加{self.disturb_force:.1f} N拉动，持续0.65 s')
        elif action=='squeeze':
            self.squeeze_until=float(self.data.time)+.7
            self.event('squeeze',f'向五指闭合关节施加{self.squeeze_torque:.2f} N·m外加力矩')
        elif action in self.enabled:
            self.enabled[action]=bool(value);self.event('toggle',f'{action}={bool(value)}')
        elif action=='limit':
            self.limit=float(np.clip(value,2.,9.));self.force_ref=min(self.force_ref,(self.limit-.2)/1.6667)
        elif action=='friction':
            self.mu=float(np.clip(value,.15,1.))
            for gid in [*self.geom_finger,self.object_geom]:self.model.geom_friction[gid,0]=self.mu
            self.event('friction',f'接触摩擦系数={self.mu:.2f}')
        elif action=='camera':
            self.camera.update({k:float(v) for k,v in value.items() if k in self.camera})
            self.camera['elevation']=float(np.clip(self.camera['elevation'],-85,20));self.camera['distance']=float(np.clip(self.camera['distance'],.35,1.3))
        else:raise ValueError('Unknown action')

    def observe(self):
        self.normal[:]=0.;self.tip_forces[:]=0.;self.palm_force=0.
        for i in range(self.data.ncon):
            c=self.data.contact[i]
            if self.object_geom not in (c.geom1,c.geom2):continue
            other=c.geom2 if c.geom1==self.object_geom else c.geom1
            f=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,i,f);normal=abs(float(f[0]))
            if other in self.geom_finger:
                digit=self.geom_finger[other];self.normal[digit]+=normal
                name=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,self.model.geom_bodyid[other]) or ''
                if name.endswith('distal'):self.tip_forces[digit]+=normal
            elif self.model.geom_bodyid[other]==self.bid('rh_palm'):self.palm_force+=normal
        self.filtered=.7*self.filtered+.3*self.normal
        vo=np.zeros(6);vh=np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.object,vo,0)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.forearm,vh,0)
        # Idealized slip sensor: axial translation and downward translation relative to the hand.
        rel=vo[3:]-vh[3:];self.slip_speed=float(np.hypot(rel[1],max(0.,-rel[2])))
        self.peak_force=max(self.peak_force,float(max(self.normal)))

    def set_actuators(self):
        for j,indices in enumerate(self.actuator_groups):
            self.data.ctrl[indices]=np.clip(OPEN[j]+self.alpha[j]*(CLOSED[j]-OPEN[j]),self.model.actuator_ctrlrange[indices,0],self.model.actuator_ctrlrange[indices,1])
        if self.state.startswith('RELEAS'):
            full_open=np.array([-.6,.1,0.,0.,0.])
            if self.enabled['gate']:
                full_open[0]=-1.0
                elapsed=float(self.data.time)-self.release_time
                tucked=np.array([-1.03,1.21,.13,-.02,.985])
                # Clear the entire thumb, lower the empty hand, then extend it.
                if elapsed<.8:
                    blend=max(0.,elapsed/.8)
                    self.data.ctrl[self.actuator_groups[0]]=self.release_thumb_start+blend*(tucked-self.release_thumb_start)
                elif elapsed<1.4:self.data.ctrl[self.actuator_groups[0]]=tucked
                else:
                    blend=min(1.,(elapsed-1.4)/.8)
                    self.data.ctrl[self.actuator_groups[0]]=tucked+blend*(full_open-tucked)
            else:
                strength=min(1.,self.force_ref/self.base_force)
                grasp_thumb=self.data.ctrl[self.actuator_groups[0]].copy()
                self.data.ctrl[self.actuator_groups[0]]=full_open+strength*(grasp_thumb-full_open)
        self.data.ctrl[self.lift_actuator]=self.lift_ref+self.model.body_subtreemass[self.forearm]*9.81/1500
        self.data.ctrl[self.aid('rh_A_WRJ2')]=0.;self.data.ctrl[self.aid('rh_A_WRJ1')]=0.

    def control(self):
        t=float(self.data.time);contact=bool(min(self.filtered)>.35)
        if self.state=='CLOSING':
            if contact:
                if self.contact_since is None:self.contact_since=t
                if t-self.contact_since>.65:
                    self.state='LIFTING';self.lift_start=t;self.event('lift','五指均已接触，开始抬升')
            else:self.contact_since=None
            if t>7:self.state='FAILED';self.event('failed','接触未在7秒内建立，请复位调整参数')
        if self.state=='LIFTING':
            self.lift_ref=.1*min(1.,(t-self.lift_start)/1.5)
            if t-self.lift_start>3.0:
                self.state='HOLDING';self.event('hold','五指持握中，可扰动、过力测试或释放')
        releasing=self.state in ('RELEASING','RELEASED')
        active=self.state in ('CLOSING','LIFTING','HOLDING') or (releasing and not self.enabled['gate'])
        overload=active and self.enabled['force'] and max(self.filtered)>self.limit+.1
        if overload:
            if t>self.force_until:self.force_events+=1;self.event('force','过力保护触发：对应手指快速卸载，抑制防滑补力')
            self.force_until=t+.12
        if releasing and not self.enabled['gate']:self.force_ref=max(0.,self.force_ref-.02)
        slip=active and self.state not in ('CLOSING','LIFTING') and (sum(self.filtered>.25)>=2 or t-self.last_slip<.15) and self.enabled['slip'] and self.slip_speed>.004
        if slip and (not self.enabled['force'] or t>=self.force_until) and t-self.last_slip>.02:
            previous=self.force_ref;self.force_ref=min((self.limit-.2)/1.6667,self.force_ref+.25)
            self.last_slip=t;self.slip_until=t+.2
            if self.force_ref>previous+1e-6:
                self.slip_events+=1
                if releasing:self.release_slip_events+=1
                self.event('slip','检测到滑移：五指协同提高持握力目标')
        if self.state in ('READY','DROPPED','FAILED') or (releasing and self.enabled['gate']):
            self.alpha=np.maximum(0.,self.alpha-.006)
            if releasing:
                self.force_ref=0.
                self.lift_ref=self.release_lift_start*(1-np.clip((t-self.release_time-.8)/.6,0,1))
                if t-self.release_time>2.6 and max(self.normal)<.05:self.state='RELEASED'
        else:
            if releasing:self.alpha=np.maximum(0.,self.alpha-.0015)
            ref=np.minimum(self.limit-.2,self.force_ref*np.array([1.6667,1,1,1,1]))
            self.alpha+=np.clip((ref-self.filtered)*.001,-.006,.004)
            if active and self.enabled['force']:
                self.alpha-=np.clip((self.filtered-self.limit)*.006,0.,.02)
            lower=np.zeros(5) if releasing or (overload and self.enabled['force']) else np.array([0,.45,.45,.45,.45])
            self.alpha=np.clip(self.alpha,lower,1.2)
        self.set_actuators()
        if releasing and not self.enabled['gate'] and t-self.release_time>1.2 and max(self.alpha)<.05 and max(self.normal)<.05:
            self.state='RELEASED'
        if self.state=='HOLDING' and (self.object_z<self.initial_z+.025 or abs(self.data.xpos[self.object,1])>.14):
            self.state='DROPPED';self.event('drop','物体离开持握范围，本次抓握失败')

    def step(self,n=1):
        for _ in range(n):
            self.data.xfrc_applied[:]=0.;self.data.qfrc_applied[:]=0.
            if self.data.time<self.disturb_until:self.data.xfrc_applied[self.object,1]=self.disturb_force
            if self.data.time<self.squeeze_until:
                for key in ['THJ5','FFJ3','MFJ3','RFJ3','LFJ3']:
                    j=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'rh_'+key)
                    self.data.qfrc_applied[self.model.jnt_dofadr[j]]=self.squeeze_torque
            if self.steps%5==0:self.observe();self.control()
            mujoco.mj_step(self.model,self.data);self.steps+=1
            if self.steps%5==0:
                self.observe();self.history.append([round(float(self.data.time),4),*[round(float(f),4) for f in self.normal],round(self.force_ref,4),round(self.object_z,5),round(self.slip_speed,5)])

    def advance(self,seconds):self.step(round(seconds/self.model.opt.timestep))

    def snapshot(self):
        t=float(self.data.time)
        return dict(time=round(t,3),state=self.state,normal=[round(float(x),3) for x in self.normal],tip_forces=[round(float(x),3) for x in self.tip_forces],palm_force=round(self.palm_force,3),force_ref=round(self.force_ref,3),limit=self.limit,friction=self.mu,object_z=round(self.object_z,4),object_y=round(float(self.data.xpos[self.object,1]),4),lift_mm=round((self.object_z-self.initial_z)*1000,1),slip_mm_s=round(self.slip_speed*1000,2),slip_active=t<self.slip_until and not self.state.startswith('RELEAS'),force_active=t<self.force_until,gate_active=self.enabled['gate'] and self.state.startswith('RELEAS'),enabled=self.enabled.copy(),slip_events=self.slip_events,force_events=self.force_events,release_slip_events=self.release_slip_events,peak_force=round(self.peak_force,3),history=list(self.history)[-3000:],events=list(self.events)[-12:],camera=self.camera.copy(),finger_names=FINGER_NAMES,alpha=[round(float(x),3) for x in self.alpha],hand_joints=24,hand_actuators=20)

    def render(self,renderer):
        cam=mujoco.MjvCamera();cam.lookat[:]=[.045,0,.36]
        for k,v in self.camera.items():setattr(cam,k,v)
        renderer.update_scene(self.data,camera=cam);return renderer.render()
