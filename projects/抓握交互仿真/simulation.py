"""MuJoCo contact dynamics and three observable reflex modules, SI units."""
from pathlib import Path
from collections import deque
import json,os,struct,zlib,time
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parent

def png_bytes(pixels):
    h,w,_=pixels.shape
    def chunk(tag,payload):
        return struct.pack('!I',len(payload))+tag+payload+struct.pack('!I',zlib.crc32(tag+payload)&0xffffffff)
    raw=b''.join(b'\0'+row.tobytes() for row in pixels)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw,2))+chunk(b'IEND',b'')

class GraspSimulation:
    def __init__(self):
        # Native Windows XML paths containing CJK can fail; resolve includes relatively.
        old=os.getcwd()
        try:
            os.chdir(ROOT);self.model=mujoco.MjModel.from_xml_path('model.xml')
        finally:os.chdir(old)
        self.data=mujoco.MjData(self.model)
        self.ids={name:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,name) for name in ['object','left','right','carriage']}
        self.geoms={name:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,name) for name in ['object_geom','left_pad','right_pad']}
        self.enabled={'slip':True,'force':True,'gate':True}
        self.limit=3.;self.mu=.6;self.base_force=1.5
        self.history=deque(maxlen=2400);self.events=deque(maxlen=40)
        self.camera={'azimuth':110.,'elevation':-22.,'distance':.62}
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model,self.data)
        for gid in self.geoms.values():self.model.geom_friction[gid,0]=self.mu
        self.state='READY';self.force_ref=self.base_force;self.normal=np.zeros(2);self.filtered=np.zeros(2)
        self.force_trim=np.zeros(2)
        self.slip_speed=0.;self.lift_ref=0.;self.lift_start=None;self.contact_since=None
        self.down_until=0.;self.squeeze_until=0.;self.slip_until=0.;self.force_until=0.
        self.last_slip=-1.;self.slip_events=0;self.force_events=0;self.release_slip_events=0
        self.release_time=None;self.release_z=None;self.peak_force=0.;self.last_log=-1.;self.steps=0
        self.history.clear();self.events.clear();mujoco.mj_forward(self.model,self.data)
        self.event('reset','已复位：物体位于支撑台上')

    def event(self,kind,message):
        self.events.append({'time':round(float(self.data.time),3),'kind':kind,'message':message})

    def command(self,action,value=None):
        if action=='reset':self.reset()
        elif action=='grasp':
            if self.state!='READY':self.reset()
            self.state='CLOSING';self.force_ref=min(self.base_force,self.limit-.15)
            self.event('grasp','开始闭合，建立接触后自动抬升')
        elif action=='release':
            self.state='RELEASING';self.release_time=float(self.data.time);self.release_z=self.object_z
            self.event('release','用户释放：关闭自动补力并张开双指' if self.enabled['gate'] else '释放门控关闭：用于观察防滑与释放冲突')
        elif action=='disturb':
            self.down_until=float(self.data.time)+.65;self.event('disturb','向物体施加1.5 N向下扰动，持续0.65 s')
        elif action=='squeeze':
            self.squeeze_until=float(self.data.time)+.7;self.event('squeeze','向两指施加各3.5 N内向外力，持续0.7 s')
        elif action in self.enabled:
            self.enabled[action]=bool(value);self.event('toggle',f'{action}={bool(value)}')
        elif action=='limit':
            self.limit=float(np.clip(float(value),.8,5.));self.force_ref=min(self.force_ref,self.limit-.1)
        elif action=='friction':
            self.mu=float(np.clip(float(value),.15,1.))
            for gid in self.geoms.values():self.model.geom_friction[gid,0]=self.mu
            self.event('friction',f'接触摩擦系数设为{self.mu:.2f}')
        elif action=='camera':
            self.camera.update({k:float(v) for k,v in value.items() if k in self.camera})
            self.camera['elevation']=float(np.clip(self.camera['elevation'],-85,10))
            self.camera['distance']=float(np.clip(self.camera['distance'],.3,1.1))
        else:raise ValueError('Unknown action')

    @property
    def object_z(self):return float(self.data.xpos[self.ids['object'],2])

    def observe(self):
        normals=np.zeros(2);obj=self.geoms['object_geom']
        for i in range(self.data.ncon):
            c=self.data.contact[i]
            if obj not in (c.geom1,c.geom2):continue
            other=c.geom2 if c.geom1==obj else c.geom1
            for side,pad in enumerate(['left_pad','right_pad']):
                if other==self.geoms[pad]:
                    f=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,i,f);normals[side]+=abs(float(f[0]))
        self.normal=normals;self.filtered=.6*self.filtered+.4*normals
        v_obj=np.zeros(6);v_hand=np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.ids['object'],v_obj,0)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.ids['carriage'],v_hand,0)
        # Idealized optical slip sensor: downward velocity relative to the jaws.
        self.slip_speed=max(0.,float(v_hand[5]-v_obj[5]))
        self.peak_force=max(self.peak_force,float(max(normals)))

    def control(self):
        t=float(self.data.time);contact=bool(min(self.filtered)>.25)
        if self.state=='CLOSING':
            if contact:
                if self.contact_since is None:self.contact_since=t
                if t-self.contact_since>.18:
                    self.state='LIFTING';self.lift_start=t;self.event('lift','接触已建立，开始抬升')
            else:self.contact_since=None
        if self.state=='LIFTING':
            self.lift_ref=.12*min(1.,(t-self.lift_start)/.9)
            if t-self.lift_start>1.2:
                self.state='HOLDING';self.event('hold','持握中，可施加扰动或释放')
        releasing=self.state in ('RELEASING','RELEASED')
        if releasing and not self.enabled['gate']:self.force_ref=max(0.,self.force_ref-.04)
        active=self.state in ('CLOSING','LIFTING','HOLDING') or (releasing and not self.enabled['gate'])
        overload=active and self.enabled['force'] and max(self.filtered)>self.limit+.05
        if overload:
            if t>self.force_until:self.force_events+=1;self.event('force','过力保护触发：卸载并抑制补力')
            self.force_until=t+.08;self.force_ref=max(.5,min(self.force_ref,self.limit-.2))
        force_guard=active and self.enabled['force'] and t<self.force_until
        if active and self.enabled['force']:
            self.force_trim=np.clip(self.force_trim+.2*(self.filtered-(self.limit-.15)),0.,6.)
        else:self.force_trim[:]=0.
        slip=active and self.state not in ('CLOSING','LIFTING') and (contact or t-self.last_slip<.1) and self.enabled['slip'] and self.slip_speed>.004
        if slip and not force_guard and t-self.last_slip>.01:
            previous=self.force_ref;self.force_ref=min(self.limit-.1,self.force_ref+.3)
            self.last_slip=t;self.slip_until=t+.15
            if self.force_ref>previous+1e-6:
                self.slip_events+=1
                if releasing:self.release_slip_events+=1
                self.event('slip','检测到向下滑移：增加0.3 N夹持力目标')
        self.data.ctrl[0]=self.lift_ref
        if self.state=='READY':self.data.ctrl[1:]=-1.2
        elif self.state=='DROPPED':self.data.ctrl[1:]=0
        elif releasing and self.enabled['gate']:
            self.force_ref=0.;self.data.ctrl[1:]=-2.5
            if t-self.release_time>.7:self.state='RELEASED'
        else:
            u=self.force_ref+.5*(self.force_ref-self.filtered)-4*self.data.qvel[1:3]
            for side in range(2):
                if self.filtered[side]<.15:u[side]=min(u[side],1.2)
            u-=self.force_trim
            self.data.ctrl[1:]=np.clip(u,-6,6)
        if self.state=='HOLDING' and self.object_z<.145:
            self.state='DROPPED';self.event('drop','物体掉回支撑台：本次持握失败')

    def step(self,n=1):
        for _ in range(n):
            self.data.xfrc_applied[:]=0.
            if self.data.time<self.down_until:self.data.xfrc_applied[self.ids['object'],2]=-1.5
            if self.data.time<self.squeeze_until:
                self.data.xfrc_applied[self.ids['left'],0]=3.5
                self.data.xfrc_applied[self.ids['right'],0]=-3.5
            if self.steps%5==0:self.observe();self.control()
            mujoco.mj_step(self.model,self.data);self.steps+=1
            if self.steps%5==0:
                self.observe()
                self.history.append([round(float(self.data.time),4),round(float(self.normal[0]),4),round(float(self.normal[1]),4),round(self.force_ref,4),round(self.object_z,5),round(self.slip_speed,5)])

    def advance(self,seconds):self.step(round(seconds/self.model.opt.timestep))

    def snapshot(self):
        t=float(self.data.time)
        return dict(time=round(t,3),state=self.state,normal=[round(float(x),3) for x in self.normal],force_ref=round(self.force_ref,3),limit=self.limit,friction=self.mu,object_z=round(self.object_z,4),slip_mm_s=round(self.slip_speed*1000,2),slip_active=t<self.slip_until and self.state not in ('RELEASED','RELEASING'),force_active=t<self.force_until,gate_active=self.enabled['gate'] and self.state in ('RELEASING','RELEASED'),enabled=self.enabled.copy(),slip_events=self.slip_events,force_events=self.force_events,release_slip_events=self.release_slip_events,peak_force=round(self.peak_force,3),history=list(self.history)[-1600:],events=list(self.events)[-12:],camera=self.camera.copy())

    def render(self,renderer):
        cam=mujoco.MjvCamera();cam.lookat[:]=[0,0,.15]
        for k,v in self.camera.items():setattr(cam,k,v)
        renderer.update_scene(self.data,camera=cam)
        return renderer.render()
